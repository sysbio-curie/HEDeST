"""
Counts the parameters of HEDeST, HistoCell and PanoSpace.

Two numbers per method, because they diverge by three orders of magnitude here and only
reporting one would misrepresent every method:

* **trainable** -- the weights gradient descent actually updates on each run. This is the
  model complexity in the sense of "what is being fitted to this slide".
* **frozen** -- the pretrained image backbone that is run but never updated. HEDeST and
  PanoSpace both have one; HistoCell does not, because it fine-tunes its ImageNet ResNet-18
  end to end, so its backbone is counted as trainable.

Where each number comes from:

* HEDeST     ``hedest.model.CellClassifier`` is instantiated, so the count is whatever the
             study actually trains. Its frozen side is H-Optimus-0, loaded through the same
             ``_load_hoptimus`` the feature stage uses.
* HistoCell  summed from a real ``epoch_49.ckpt``. Nothing is assumed about the
             architecture and nothing is frozen -- ``external/HistoCell/model/arch.py``
             builds ``resnet18(pretrained=True)`` and never sets ``requires_grad = False``.
* PanoSpace  the trainable head is ``DINOv2NeighborClassifier.classifier``, two Linear
             layers (``superres_utils.py:313-317``), rebuilt here with torch alone so the
             count does not need the panospace environment. Its frozen side is
             DINOv2-base, counted from the cached ``model.safetensors`` shapes.
             PanoSpace's third stage, the optimal-transport assignment, fits no parameters.

The count depends on the number of cell types, which is 2 to 12 across the 39 STHELAR
sample-levels, so every method is counted at every level and the table carries the spread.

Usage:
    python benchmark/comparison_code/count_params.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict
from typing import Optional

import pandas as pd
import torch
from loguru import logger
from torch import nn

REPO = Path("/cluster/CBIO/home/lgortana/HEDeST")
RESULTS = REPO / "benchmark/results"
OUT_DIR = RESULTS / "comparison/timing"
DINOV2 = Path.home() / ".panospace_cache/dinov2-base/model.safetensors"

EMBED_DIM = 1536  # H-Optimus-0 patch-token width, hedest.features.hoptimus.EMBED_DIM
HEDEST_HIDDEN = [1024, 512]  # config/params.yaml hedest.hidden_dims


def trainable(module: nn.Module) -> int:
    """Returns the number of parameters a module would update."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def safetensors_numel(path: Path) -> int:
    """
    Returns the total number of parameters in a safetensors file.

    The shapes live in a JSON header -- an 8-byte little-endian length followed by that
    many bytes of JSON -- so the count needs neither the weights nor transformers.
    """

    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))

    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta.get("shape", [])
        n = 1
        for dim in shape:
            n *= dim
        total += n

    return total


def hedest_head(num_classes: int) -> int:
    """Returns the trainable size of the HEDeST classifier for a given number of cell types."""

    from hedest.model.cell_classifier import CellClassifier

    model = CellClassifier(
        num_classes=num_classes,
        embed_size=EMBED_DIM,
        hidden_dims=HEDEST_HIDDEN,
        norm=True,
        dropout=0.0,
    )

    return trainable(model)


def panospace_head(num_classes: int) -> int:
    """
    Returns the trainable size of PanoSpace's super-resolution head.

    Rebuilt from ``DINOv2NeighborClassifier.classifier`` (``superres_utils.py:313-317``):
    ``Linear(1536, 512) -> ReLU -> Linear(512, num_classes)``. The 1536 is two concatenated
    768-d DINOv2 CLS tokens, the spot's own crop and its neighbourhood, and the dropout it
    sits behind holds no parameters.
    """

    head = nn.Sequential(
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )

    return trainable(head)


def hoptimus_total() -> Optional[int]:
    """Returns H-Optimus-0's parameter count, or None if the model cannot be loaded."""

    try:
        from hedest.features.hoptimus import _load_hoptimus

        model = _load_hoptimus()
        total = sum(p.numel() for p in model.parameters())
        del model

        return total
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"could not load H-Optimus-0 ({exc}); its count is left blank.")

        return None


def histocell_checkpoints() -> pd.DataFrame:
    """Sums every HistoCell checkpoint on disk, one row per sample-level and seed."""

    rows = []
    for ckpt in sorted(RESULTS.glob("HistoCell/*/level*/gt/seed*/model/epoch_*.ckpt")):
        seed = int(ckpt.parent.parent.name.replace("seed", ""))
        level = ckpt.parent.parent.parent.parent.name
        sample = ckpt.parent.parent.parent.parent.parent.name
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        total = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
        rows.append({"sample": sample, "level": level, "seed": seed, "trainable": total})
        del state

    return pd.DataFrame(rows)


def main() -> None:
    """Writes the per-sample-level parameter table and a per-method summary."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # how many cell types each sample-level has, taken from the runs themselves
    classes: Dict[tuple, int] = {}
    for path in sorted(RESULTS.glob("HistoCell/*/level*/gt/seed0/metrics.json")):
        info = json.loads(path.read_text())
        classes[(info["sample"], f"level{info['level']}")] = int(info["n_cell_types"])
    logger.info(f"{len(classes)} gt sample-levels, " f"{min(classes.values())}-{max(classes.values())} cell types")

    dinov2 = safetensors_numel(DINOV2) if DINOV2.exists() else None
    if dinov2 is None:
        logger.warning(f"{DINOV2} is missing; PanoSpace's frozen count is left blank.")
    hoptimus = hoptimus_total()
    logger.info(f"frozen backbones: H-Optimus-0 {hoptimus:,}" if hoptimus else "H-Optimus-0 unknown")
    logger.info(f"frozen backbones: DINOv2-base {dinov2:,}" if dinov2 else "DINOv2-base unknown")

    hc = histocell_checkpoints()
    logger.info(f"{len(hc)} HistoCell checkpoints summed")
    hc_by_unit = hc.groupby(["sample", "level"])["trainable"].mean().round().astype("int64")
    if hc.groupby(["sample", "level"])["trainable"].nunique().gt(1).any():
        logger.warning("a HistoCell sample-level has seeds of differing size; the mean is used")

    rows = []
    for (sample, level), n_classes in sorted(classes.items()):
        rows.append(
            {
                "method": "HEDeST",
                "sample": sample,
                "level": level,
                "n_classes": n_classes,
                "trainable": hedest_head(n_classes),
                "frozen": hoptimus,
                "frozen_name": "H-Optimus-0 (ViT-g/14)",
            }
        )
        rows.append(
            {
                "method": "PanoSpace",
                "sample": sample,
                "level": level,
                "n_classes": n_classes,
                "trainable": panospace_head(n_classes),
                "frozen": dinov2,
                "frozen_name": "DINOv2-base (ViT-B/14)",
            }
        )
        hc_value = hc_by_unit.get((sample, level))
        rows.append(
            {
                "method": "HistoCell",
                "sample": sample,
                "level": level,
                "n_classes": n_classes,
                "trainable": int(hc_value) if hc_value is not None else None,
                "frozen": 0,
                "frozen_name": "none (ResNet-18 fine-tuned end to end)",
            }
        )

    df = pd.DataFrame(rows)
    per_unit = OUT_DIR / "params_per_unit.csv"
    df.to_csv(per_unit, index=False)

    summary = (
        df.groupby(["method", "frozen_name"])
        .agg(
            trainable_mean=("trainable", "mean"),
            trainable_min=("trainable", "min"),
            trainable_max=("trainable", "max"),
            frozen=("frozen", "first"),
            n_units=("trainable", "count"),
        )
        .reset_index()
    )
    summary["trainable_mean"] = summary.trainable_mean.round().astype("Int64")
    summary["total_mean"] = (summary.trainable_mean.fillna(0) + summary.frozen.fillna(0)).astype("Int64")
    out = OUT_DIR / "params_summary.csv"
    summary.to_csv(out, index=False)

    logger.info(f"-> {per_unit}")
    logger.info(f"-> {out}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
