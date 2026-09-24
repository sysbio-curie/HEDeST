"""End-to-end HEDeST pipeline: slide -> mask -> segmentation -> embeddings -> model.

The stages are:

1. ``check``    the slide opens, is pyramidal and has a known resolution. A slide that is
                not usable is converted to a pyramidal TIFF; if it cannot be converted the
                pipeline stops with an explicit error.
2. ``mask``     a tissue mask is computed. HoVer-Net's own automatic mask is never used:
                the segmentation stage refuses to run without a mask.
3. ``segment``  HoVer-Net nuclei segmentation, writing one JSON whose nuclei are numbered
                0..N-1, which is the cell id convention of the whole package.
4. ``features`` H-Optimus-0 tile embeddings pooled per cell (see hedest.features.hoptimus).
5. ``train``    HEDeST itself, through hedest/main.py.

HoVer-Net needs its own environment, so the mask and segmentation stages are run as
subprocesses whose interpreter is configurable (``python`` key of each section).

Everything is driven by a YAML file; ``hedest/pipeline.py init-config my_run.yaml``
writes a documented template.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import typer
import yaml
from loguru import logger

from hedest.slide import ensure_pyramidal_slide
from hedest.slide import inspect_slide
from hedest.slide import SlideError

app = typer.Typer(help="Run the HEDeST pipeline, from a whole-slide image to cell type predictions.")

STAGES = ["check", "mask", "segment", "features", "train"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "slide": None,
    "out_dir": None,
    "mpp": None,
    "convert_slide": True,
    "mask": {
        "python": None,
        "level": 3,
    },
    "segmentation": {
        "python": None,
        "model_path": None,
        "type_info_path": "external/hovernet/type_info.json",
        "nr_types": 6,
        "model_mode": "fast",
        "batch_size": 16,
        "gpu": "0",
        "proc_mag": 40,
        "cache_path": "cache",
        "save_geojson": False,
        "image_dict_path": None,
        "image_dict_size_px": 64,
        "image_dict_size_um": 20,
    },
    "features": {
        "batch_size": 32,
        "num_workers": 6,
    },
    "train": {
        "spot_prop_file": None,
        "path_st_adata": None,
        "adata_name": None,
        "spot_dict_file": None,
        "hidden_dims": "512,256",
        "norm": False,
        "dropout": 0.0,
        "batch_size": 64,
        "lr": 0.0001,
        "divergence": "l2",
        "alpha": 0.0,
        "beta": 0.0,
        "adjustment": "interpolated",
        "gated": False,
        "epochs": 60,
        "train_size": 0.7,
        "val_size": 0.15,
        "save_geojson": False,
        "color_dict_file": None,
        "rs": 42,
    },
}

CONFIG_TEMPLATE = """# HEDeST pipeline configuration.
# Run with:  python -m hedest.pipeline run this_file.yaml

slide: /path/to/slide.tif        # whole-slide image (pyramidal TIFF preferred)
out_dir: /path/to/results        # everything the pipeline produces goes here
mpp: null                        # microns per pixel; null = read it from the slide
convert_slide: true              # convert to pyramidal TIFF if the slide is not usable

mask:
  python: null                   # interpreter for the mask step; null = the current one
  level: 3                       # higher = faster and coarser

segmentation:
  python: /path/to/envs/hovernet/bin/python   # HoVer-Net needs its own environment
  model_path: /path/to/hovernet_fast_pannuke_type_tf2pytorch.tar
  type_info_path: external/hovernet/type_info.json
  nr_types: 6
  model_mode: fast
  batch_size: 16
  gpu: "0"
  proc_mag: 40
  cache_path: /path/to/cache
  save_geojson: false
  image_dict_path: null          # null = no cell crops. A .pt path also saves the crops,
  image_dict_size_px: 64         # which are only needed to plot cells later (they are big).
  image_dict_size_um: 20         # null = crop size_px pixels instead of size_um microns

features:
  batch_size: 32
  num_workers: 6

train:
  spot_prop_file: /path/to/proportions.csv   # required
  path_st_adata: /path/to/adata.h5ad         # needed to map cells to spots and for PPSA
  adata_name: null                           # null = first sample of adata.uns['spatial']
  spot_dict_file: null                       # null = computed from the adata and the segmentation
  hidden_dims: "512,256"
  norm: false
  dropout: 0.0
  batch_size: 64
  lr: 0.0001
  divergence: l2
  alpha: 0.0
  beta: 0.0
  adjustment: interpolated       # PPSA for the cells outside spots: interpolated | nearest
  gated: false                   # true = adjust only the cells inside spots
  epochs: 60
  train_size: 0.7
  val_size: 0.15
  save_geojson: false
  color_dict_file: null
  rs: 42
"""


def repo_root() -> Path:
    """
    Locates the repository root, where the external tools live.

    Returns:
        The path of the directory containing the 'hedest' and 'external' folders.
    """

    return Path(__file__).resolve().parents[1]


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a configuration into the defaults, one level of nesting deep.

    Args:
        base: The default configuration.
        new: The user configuration.

    Returns:
        The merged configuration.
    """

    merged = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value

    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a pipeline configuration and checks the keys every stage needs.

    Args:
        config_path: Path to the YAML configuration.

    Returns:
        The configuration, merged with the defaults.

    Raises:
        ValueError: If a mandatory key is missing or unknown.
    """

    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}

    unknown = set(user_config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration key(s): {sorted(unknown)}. Expected {sorted(DEFAULT_CONFIG)}.")

    config = _deep_update(DEFAULT_CONFIG, user_config)

    for key in ("slide", "out_dir"):
        if not config.get(key):
            raise ValueError(f"'{key}' is required in {config_path}.")

    return config


def _paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the output layout of a run.

    Args:
        config: The pipeline configuration.

    Returns:
        The paths of every stage output.
    """

    out_dir = Path(config["out_dir"]).resolve()
    name = Path(config["slide"]).stem

    return {
        "out_dir": out_dir,
        "name": name,
        "slide_dir": out_dir / "slide",
        "mask_dir": out_dir / "mask",
        "seg_dir": out_dir / "segmentation",
        "seg_input": out_dir / "segmentation" / "input",
        "features": out_dir / "features" / f"{name}_hoptimus.pt",
        "model_dir": out_dir / "model",
        "state": out_dir / "pipeline.json",
    }


def _mask_path(paths: Dict[str, Any], state: Dict[str, Any]) -> Path:
    """
    Gives the path of the tissue mask.

    HoVer-Net looks for a mask named after the slide file it processes, so the mask is
    named after the slide actually used, which is not the one given in the configuration
    when a conversion happened.

    Args:
        paths: The output layout.
        state: The state of the run, holding the slide actually used.

    Returns:
        The path of the mask.
    """

    slide = state.get("slide") or paths["name"]

    return paths["mask_dir"] / f"{Path(slide).stem}.png"


def _read_state(paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads what previous runs of the pipeline recorded.

    Args:
        paths: The output layout.

    Returns:
        The recorded state, empty if the pipeline never ran here.
    """

    if paths["state"].exists():
        with open(paths["state"]) as f:
            return json.load(f)

    return {}


def _write_state(paths: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Records what the pipeline produced, so later stages and later runs can find it.

    Args:
        paths: The output layout.
        state: The state to record.
    """

    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["state"], "w") as f:
        json.dump(state, f, indent=2, default=str)


def _run_command(command: List[Any], stage: str) -> None:
    """
    Runs an external stage and fails loudly.

    The repository and the external tools are put on the PYTHONPATH of the subprocess,
    so the stage works whatever environment it runs in.

    Args:
        command: The command to run.
        stage: Name of the stage, for the error message.

    Raises:
        RuntimeError: If the command returns a non-zero status.
    """

    root = repo_root()
    env = os.environ.copy()
    extra = [str(root), str(root / "external"), str(root / "external" / "hovernet")]
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    logger.info(f"[{stage}] {' '.join(str(c) for c in command)}")
    result = subprocess.run([str(c) for c in command], cwd=str(root), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"The '{stage}' stage failed with exit code {result.returncode}.")


def _segmentation_json(paths: Dict[str, Any]) -> Optional[Path]:
    """
    Finds the segmentation file produced by HoVer-Net.

    Args:
        paths: The output layout.

    Returns:
        The path of the segmentation JSON, or None if there is none.
    """

    # HoVer-Net writes the JSON either directly in the output directory or in a 'json'
    # subfolder, depending on whether it also saves a mask or a thumbnail.
    candidates = sorted(paths["seg_dir"].glob("*.json")) + sorted((paths["seg_dir"] / "json").glob("*.json"))

    return candidates[0] if candidates else None


def stage_check(config: Dict[str, Any], paths: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes sure the slide is usable, converting it if it is not.

    Args:
        config: The pipeline configuration.
        paths: The output layout.
        state: The state of the run.

    Returns:
        The updated state, holding the slide to use and its resolution.
    """

    converted = paths["slide_dir"] / f"{paths['name']}_pyramidal.tif"
    slide_path, mpp = ensure_pyramidal_slide(
        config["slide"],
        mpp=config.get("mpp"),
        convert=config.get("convert_slide", True),
        converted_path=str(converted),
    )

    state["slide"] = str(slide_path)
    state["mpp"] = mpp

    return state


def stage_mask(config: Dict[str, Any], paths: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the tissue mask.

    Args:
        config: The pipeline configuration.
        paths: The output layout.
        state: The state of the run.

    Returns:
        The updated state, holding the mask path.
    """

    paths["mask_dir"].mkdir(parents=True, exist_ok=True)
    interpreter = config["mask"].get("python") or sys.executable
    mask = _mask_path(paths, state)

    _run_command(
        [
            interpreter,
            "-u",
            str(repo_root() / "external" / "hovernet" / "run_mask.py"),
            state["slide"],
            str(mask),
            "--mask-level",
            config["mask"]["level"],
        ],
        stage="mask",
    )

    if not mask.exists():
        raise RuntimeError(f"The mask stage did not produce {mask}.")

    state["mask"] = str(mask)

    return state


def _check_image_dict(image_dict_path: Path, seg_json: Path) -> None:
    """
    Checks that the cell crops match the segmentation they were extracted from.

    The cells of a slide are numbered by their position in the segmentation file, and
    that numbering is what ties the crops, the embeddings and the spot dictionary
    together, so it is verified rather than assumed.

    Args:
        image_dict_path: Path of the saved cell crops.
        seg_json: Path of the segmentation file.

    Raises:
        RuntimeError: If the crops are missing or do not match the segmentation.
    """

    import torch

    if not image_dict_path.exists():
        raise RuntimeError(f"The segmentation stage did not produce the cell crops at {image_dict_path}.")

    with open(seg_json) as f:
        n_cells = len(json.load(f)["nuc"])

    image_dict = torch.load(image_dict_path, map_location="cpu")
    keys = list(image_dict.keys())
    del image_dict

    if keys != [str(i) for i in range(n_cells)]:
        raise RuntimeError(
            f"The cell crops at {image_dict_path} do not match {seg_json.name}: "
            f"{len(keys)} crops against {n_cells} nuclei, expected the ids '0'..'{n_cells - 1}'."
        )

    logger.info(f"-> Cell crops saved to {image_dict_path} ({n_cells} cells).")


def stage_segment(config: Dict[str, Any], paths: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs HoVer-Net, using the mask computed by the pipeline.

    HoVer-Net's automatic mask is deliberately not an option: without a mask the stage
    refuses to run, because the automatic one is unreliable.

    Args:
        config: The pipeline configuration.
        paths: The output layout.
        state: The state of the run.

    Returns:
        The updated state, holding the segmentation path.

    Raises:
        RuntimeError: If the mask is missing, or if no segmentation is produced.
    """

    mask = Path(state.get("mask") or _mask_path(paths, state))
    if not mask.exists():
        raise RuntimeError(f"No tissue mask at {mask}. Run the 'mask' stage before 'segment'.")

    expected = Path(state["slide"]).stem + ".png"
    if mask.name != expected:
        raise RuntimeError(
            f"The mask {mask.name} does not match the slide {Path(state['slide']).name}. "
            f"HoVer-Net looks for {expected} in {paths['mask_dir']} and would silently fall back "
            "to its own automatic mask, so the stage stops here."
        )

    segmentation = config["segmentation"]
    if not segmentation.get("model_path"):
        raise ValueError("'segmentation.model_path' is required to run HoVer-Net.")

    # Cell crops are only useful to plot cells afterwards, and they are large, so they
    # are written only when a path is given. Checked now rather than after the inference.
    image_dict_path = segmentation.get("image_dict_path")
    if image_dict_path is not None:
        image_dict_path = Path(image_dict_path)
        if image_dict_path.suffix != ".pt":
            raise ValueError(f"'segmentation.image_dict_path' must end with .pt, got {image_dict_path}.")
        image_dict_path.parent.mkdir(parents=True, exist_ok=True)
        if segmentation.get("image_dict_size_um") is not None and not state.get("mpp"):
            raise ValueError("'image_dict_size_um' needs the resolution of the slide, which is unknown.")

    # HoVer-Net reads a directory, and names its outputs after the slide, so the slide is
    # linked alone in a dedicated folder and the mask carries the same name.
    paths["seg_input"].mkdir(parents=True, exist_ok=True)
    linked = paths["seg_input"] / Path(state["slide"]).name
    if not os.path.lexists(linked):
        os.symlink(os.path.abspath(state["slide"]), linked)

    interpreter = segmentation.get("python") or sys.executable
    command = [
        interpreter,
        "-u",
        str(repo_root() / "external" / "hovernet" / "run_infer.py"),
        f"--gpu={segmentation['gpu']}",
        f"--nr_types={segmentation['nr_types']}",
        f"--type_info_path={segmentation['type_info_path']}",
        f"--batch_size={segmentation['batch_size']}",
        f"--model_mode={segmentation['model_mode']}",
        f"--model_path={segmentation['model_path']}",
        f"--mpp={state['mpp']}",
        f"--size_px={segmentation['image_dict_size_px']}",
    ]
    if segmentation.get("image_dict_size_um") is not None:
        command.append(f"--size_um={segmentation['image_dict_size_um']}")
    command += [
        "wsi",
        f"--input_dir={paths['seg_input']}",
        f"--output_dir={paths['seg_dir']}",
        f"--input_mask_dir={paths['mask_dir']}",
        f"--cache_path={segmentation['cache_path']}",
        f"--proc_mag={segmentation['proc_mag']}",
    ]
    if image_dict_path is None:
        command.append("--skip_image_dict")
    else:
        command.append(f"--image_dict_path={image_dict_path}")
    if segmentation.get("save_geojson"):
        command.append("--save_geojson")

    _run_command(command, stage="segment")

    seg_json = _segmentation_json(paths)
    if seg_json is None:
        raise RuntimeError(f"The segmentation stage did not produce a JSON file in {paths['seg_dir']}.")

    state["segmentation"] = str(seg_json)

    if image_dict_path is not None:
        _check_image_dict(image_dict_path, seg_json)
        state["image_dict"] = str(image_dict_path)

    return state


def stage_features(config: Dict[str, Any], paths: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the H-Optimus-0 cell embeddings.

    Args:
        config: The pipeline configuration.
        paths: The output layout.
        state: The state of the run.

    Returns:
        The updated state, holding the embeddings path.

    Raises:
        RuntimeError: If the segmentation is missing.
    """

    from hedest.features.hoptimus import extract_hoptimus_embeddings

    seg_json = state.get("segmentation") or _segmentation_json(paths)
    if seg_json is None or not Path(seg_json).exists():
        raise RuntimeError("No segmentation file. Run the 'segment' stage before 'features'.")

    extract_hoptimus_embeddings(
        slide_path=state["slide"],
        json_path=str(seg_json),
        out_path=str(paths["features"]),
        mpp=state.get("mpp"),
        batch_size=config["features"]["batch_size"],
        num_workers=config["features"]["num_workers"],
    )

    state["features"] = str(paths["features"])

    return state


def stage_train(config: Dict[str, Any], paths: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains HEDeST on the embeddings, through hedest/main.py.

    Args:
        config: The pipeline configuration.
        paths: The output layout.
        state: The state of the run.

    Returns:
        The updated state, holding the model directory.

    Raises:
        ValueError: If the proportions are missing.
        RuntimeError: If the embeddings are missing.
    """

    train = config["train"]
    if not train.get("spot_prop_file"):
        raise ValueError("'train.spot_prop_file' is required to train HEDeST.")

    features = state.get("features", str(paths["features"]))
    if not Path(features).exists():
        raise RuntimeError(f"No embeddings at {features}. Run the 'features' stage before 'train'.")

    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "hedest.main",
        features,
        train["spot_prop_file"],
        "--json-path",
        state.get("segmentation", "none"),
        "--mpp",
        state["mpp"],
        "--hidden-dims",
        train["hidden_dims"],
        "--dropout",
        train["dropout"],
        "--batch-size",
        train["batch_size"],
        "--lr",
        train["lr"],
        "--divergence",
        train["divergence"],
        "--alpha",
        train["alpha"],
        "--beta",
        train["beta"],
        "--adjustment",
        train["adjustment"],
        "--epochs",
        train["epochs"],
        "--train-size",
        train["train_size"],
        "--val-size",
        train["val_size"],
        "--out-dir",
        str(paths["model_dir"]),
        "--rs",
        train["rs"],
    ]

    for option, key in (
        ("--path-st-adata", "path_st_adata"),
        ("--adata-name", "adata_name"),
        ("--spot-dict-file", "spot_dict_file"),
        ("--color-dict-file", "color_dict_file"),
    ):
        if train.get(key):
            command += [option, train[key]]

    command.append("--norm" if train.get("norm") else "--no-norm")
    command.append("--gated" if train.get("gated") else "--no-gated")
    if train.get("save_geojson"):
        command.append("--save-geojson")

    _run_command(command, stage="train")
    state["model"] = str(paths["model_dir"])

    return state


STAGE_FUNCTIONS = {
    "check": stage_check,
    "mask": stage_mask,
    "segment": stage_segment,
    "features": stage_features,
    "train": stage_train,
}


def _parse_stages(stages: str) -> List[str]:
    """
    Turns the --stages option into the list of stages to run.

    Args:
        stages: "all", or a comma-separated list of stage names.

    Returns:
        The stages to run, in pipeline order.

    Raises:
        ValueError: If a stage name is unknown.
    """

    if stages.strip().lower() == "all":
        return list(STAGES)

    wanted = [s.strip() for s in stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Available stages: {STAGES}.")

    return [s for s in STAGES if s in wanted]


@app.command("init-config")
def init_config(path: str = typer.Argument(..., help="Where to write the configuration template.")) -> None:
    """Writes a documented configuration template."""

    if os.path.exists(path):
        raise typer.BadParameter(f"{path} already exists.")

    with open(path, "w") as f:
        f.write(CONFIG_TEMPLATE)
    logger.info(f"Configuration template written to {path}. Fill it in, then run: hedest/pipeline.py run {path}")


@app.command("check")
def check_command(
    slide: str = typer.Argument(..., help="Path to the slide to inspect."),
    mpp: Optional[float] = typer.Option(None, help="Microns per pixel, if the slide does not carry it."),
    convert: bool = typer.Option(False, help="Convert the slide to a pyramidal TIFF if it is not usable."),
    converted_path: Optional[str] = typer.Option(None, help="Where to write the converted slide."),
) -> None:
    """Checks that a slide is readable, pyramidal and has a known resolution."""

    info = inspect_slide(slide)
    logger.info(
        f"{slide}: readable={info['readable']} levels={info['level_count']} "
        f"dimensions={info['dimensions']} mpp={info['mpp']} vendor={info['vendor']}"
        + (f" error={info['error']}" if info["error"] else "")
    )

    try:
        path, resolved_mpp = ensure_pyramidal_slide(slide, mpp=mpp, convert=convert, converted_path=converted_path)
    except SlideError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1)

    logger.info(f"-> Slide to use: {path} ({resolved_mpp:.4f} um/px)")


@app.command("run")
def run_command(
    config_path: str = typer.Argument(..., help="Path to the pipeline configuration."),
    stages: str = typer.Option("all", help=f"Stages to run: 'all' or a comma-separated subset of {STAGES}."),
) -> None:
    """Runs the pipeline, or a subset of its stages."""

    config = load_config(config_path)
    paths = _paths(config)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)

    state = _read_state(paths)
    state["config"] = config_path

    try:
        to_run = _parse_stages(stages)
        # The other stages all need the slide and its resolution, which 'check' resolves.
        if "check" not in to_run and "slide" not in state:
            state = stage_check(config, paths, state)
    except (SlideError, ValueError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1)

    logger.info(f"Running stages: {to_run}")
    for stage in to_run:
        logger.info("=" * 60)
        logger.info(f"STAGE: {stage}")
        logger.info("=" * 60)
        try:
            state = STAGE_FUNCTIONS[stage](config, paths, state)
        except (SlideError, RuntimeError, ValueError) as exc:
            logger.error(f"Stage '{stage}' stopped: {exc}")
            raise typer.Exit(code=1)
        state.setdefault("done", [])
        if stage not in state["done"]:
            state["done"].append(stage)
        _write_state(paths, state)

    logger.info(f"Pipeline finished. State recorded in {paths['state']}.")


if __name__ == "__main__":
    app()
