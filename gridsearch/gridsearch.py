"""HEDeST hyper-parameter gridsearch.

A study is described by one YAML file (see ``gridsearch/semi_sim.yaml``):

* ``grid``      the parameter lists; every combination is trained,
* ``fixed``     the parameters that do not vary,
* ``seeds``     the seeds each combination is trained with,
* ``datasets``  per dataset: the cell embeddings of each feature type, the spot
                proportions (per annotation level), the spots, the segmentation and
                the ground truth cell types.

A *unit* is one (feature type, dataset, annotation level). Each unit is loaded once
and all its runs (combinations x seeds) are trained in the same process. A run is
exactly what ``hedest/main.py`` trains with the same parameters and seed: the
training uses the HEDeST blocks unchanged (``split_data``, ``custom_collate``,
``CellClassifier``, ``ModelTrainer``, the PPSA classes). Only what is identical
across the runs of a unit is computed once: the loading, and the neighbour spots
PPSA uses for every cell.

Each run is scored against the ground truth with the raw predictions and with the
PPSA-adjusted ones (balanced accuracy over the cells that have a ground truth type).

    python gridsearch/gridsearch.py check   CONFIG           # inputs are consistent
    python gridsearch/gridsearch.py plan    CONFIG           # units, runs, progress
    python gridsearch/gridsearch.py run     CONFIG --unit 3  # train one unit
    python gridsearch/gridsearch.py collect CONFIG           # one table of all runs

Outputs, per run: ``{out_dir}/{feature}/{dataset}/{level}/{combination}/seed_{seed}/``
with ``best_model.pth``, ``history.png``, ``metrics.json`` (scores, loss history,
confusion matrices) and ``predictions.npz`` (raw and PPSA-adjusted probabilities, rows
in the order of the unit's ``cell_ids.txt``). A run whose ``metrics.json`` exists is
skipped, so ``run`` can be restarted at will.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hedest.dataset import SpotEmbedDataset  # noqa: E402
from hedest.dataset_utils import custom_collate  # noqa: E402
from hedest.dataset_utils import pp_prop  # noqa: E402
from hedest.dataset_utils import split_data  # noqa: E402
from hedest.model.cell_classifier import CellClassifier  # noqa: E402
from hedest.ppsa import ADJUSTMENT_CLASSES  # noqa: E402
from hedest.trainer import ModelTrainer  # noqa: E402
from hedest.utils import load_spatial_adata  # noqa: E402
from hedest.utils import set_seed  # noqa: E402

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False, help=__doc__.split("\n\n")[0])

# Defaults of hedest/main.py; `fixed` and `grid` override them.
DEFAULTS: Dict[str, Any] = {
    "hidden_dims": [512, 256],
    "norm": False,
    "dropout": 0.0,
    "batch_size": 64,
    "lr": 1e-4,
    "divergence": "l2",
    "alpha": 0.0,
    "beta": 0.0,
    "adjustment": "interpolated",
    "gated": False,
    "epochs": 60,
    "train_size": 0.7,
    "val_size": 0.15,
}

# Prefix of each parameter in the combination folder name, e.g. norm1_do0.1_a0.005_lr0.0003_l2.
PREFIX = {
    "norm": "norm",
    "dropout": "do",
    "alpha": "a",
    "lr": "lr",
    "divergence": "",
    "hidden_dims": "hd",
    "batch_size": "bs",
    "beta": "b",
    "adjustment": "",
    "gated": "gated",
    "epochs": "ep",
    "train_size": "tr",
    "val_size": "va",
}

PPSA_KEYS = ("adjustment", "gated", "beta")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """
    Reads a study configuration and fills in the defaults.

    Args:
        path: Path to the YAML file.

    Returns:
        The configuration.
    """

    with open(path) as f:
        cfg = yaml.safe_load(f)

    unknown = set(cfg.get("grid", {})) | set(cfg.get("fixed", {}))
    unknown -= set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown parameter(s) {sorted(unknown)}; valid ones: {sorted(DEFAULTS)}")
    both = set(cfg.get("grid", {})) & set(cfg.get("fixed", {}))
    if both:
        raise ValueError(f"{sorted(both)} are both in 'grid' and in 'fixed'")

    cfg.setdefault("grid", {})
    cfg.setdefault("fixed", {})
    cfg.setdefault("seeds", [0])
    cfg.setdefault("predictions", "float16")
    if cfg["predictions"] not in ("float16", "float32", "none"):
        raise ValueError("'predictions' must be float16, float32 or none")

    root = os.path.dirname(os.path.abspath(path))
    for key in ("out_dir", "results_dir", "plots_dir"):
        if cfg.get(key) and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(root, cfg[key]))
    for ds in cfg["datasets"]:
        ds.setdefault("levels", ["single"])
        ds.setdefault("group", ds["name"])
    cfg["_path"] = os.path.abspath(path)
    return cfg


def fill(template: Optional[str], ds: Dict[str, Any], level: str) -> Optional[str]:
    """
    Fills a path (or column) template of a dataset: ``{level}`` is the annotation level,
    and any other ``{key}`` is the dataset's own ``key`` entry (``{name}``, ``{root}``...).
    """

    if template is None:
        return None
    fields = {k: v for k, v in ds.items() if isinstance(v, (str, int, float))}
    out = str(template)
    for _ in range(5):  # entries may themselves hold placeholders ({root} -> .../{name})
        filled = out.format(**{**fields, "level": level})
        if filled == out:
            break
        out = filled
    return out


def combinations(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lists the parameter combinations of the grid.

    Args:
        cfg: The study configuration.

    Returns:
        One dict per combination, holding every run parameter (grid, fixed and defaults).
    """

    keys = list(cfg["grid"])
    out = []
    for values in itertools.product(*(cfg["grid"][k] for k in keys)):
        params = {**DEFAULTS, **cfg["fixed"], **dict(zip(keys, values))}
        params["hidden_dims"] = [int(h) for h in params["hidden_dims"]]
        params["combination"] = combination_name(params, keys)
        out.append(params)
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return "-".join(str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def combination_name(params: Dict[str, Any], keys: List[str]) -> str:
    """Folder name of a combination, built from the grid parameters only."""

    return "_".join(f"{PREFIX[k]}{_fmt(params[k])}" for k in keys) or "default"


def units(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Lists the (feature type, dataset, level) units of the study, in a fixed order.

    Args:
        cfg: The study configuration.

    Returns:
        One dict per unit.
    """

    features = cfg.get("features") or sorted({f for ds in cfg["datasets"] for f in ds["features"]})
    out = []
    for feature in features:
        for ds in cfg["datasets"]:
            if feature not in ds["features"]:
                continue
            for level in ds["levels"]:
                out.append({"feature": feature, "dataset": ds["name"], "group": ds["group"], "level": str(level)})
    for i, u in enumerate(out):
        u["id"] = i
        u["dir"] = os.path.join(cfg["out_dir"], u["feature"], u["dataset"], u["level"])
    return out


def get_unit(cfg: Dict[str, Any], unit: str) -> Dict[str, Any]:
    """Finds a unit by index or by 'feature/dataset/level'."""

    all_units = units(cfg)
    if unit.isdigit():
        return all_units[int(unit)]
    for u in all_units:
        if f"{u['feature']}/{u['dataset']}/{u['level']}" == unit:
            return u
    raise ValueError(f"Unknown unit {unit!r}")


def dataset_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(ds for ds in cfg["datasets"] if ds["name"] == name)


# --------------------------------------------------------------------------------------
# Inputs of a unit
# --------------------------------------------------------------------------------------


def load_ground_truth(spec: Dict[str, Any], ds: Dict[str, Any], level: str) -> pd.Series:
    """
    Reads the ground truth cell types, one label per segmented cell.

    Three layouts are understood:

    * a JSON dict ``{cell_id: label}``, possibly nested under ``key``;
    * a CSV with one row per cell and a label ``column``;
    * a CSV with one row per cell and one column per cell type (one-hot or soft
      labels): the label is the column holding the largest value.

    Cells whose ground truth lives on another segmentation (e.g. DAPI cells) must be
    matched to the segmented cells beforehand; the file then maps segmented cell ids
    to the label of their matched cell.

    Args:
        spec: The ``ground_truth`` entry of the dataset.
        ds: The dataset configuration.
        level: The annotation level.

    Returns:
        Labels indexed by cell id (str).
    """

    path = fill(spec["path"], ds, level)
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if spec.get("key"):
            data = data[spec["key"]]
        labels = pd.Series(data, dtype=object)
    else:
        df = pd.read_csv(path, index_col=spec.get("index_col", 0))
        if spec.get("column"):
            labels = df[fill(spec["column"], ds, level)]
        else:
            values = df.to_numpy(dtype=float)
            labels = pd.Series(df.columns.to_numpy()[values.argmax(axis=1)], index=df.index)
            labels[values.max(axis=1) <= 0] = np.nan
    labels.index = labels.index.astype(str)
    return labels.dropna().astype(str)


def load_unit(cfg: Dict[str, Any], u: Dict[str, Any], device: torch.device, embeddings: bool = True) -> Dict[str, Any]:
    """
    Loads everything a unit needs, once for all its runs.

    Args:
        cfg: The study configuration.
        u: The unit.
        device: Where the embeddings are kept during training.
        embeddings: If False, only the embedding keys are read (for `check`).

    Returns:
        A dict with the embeddings, proportions, spots, AnnData, segmentation and labels.
    """

    ds = dataset_cfg(cfg, u["dataset"])
    level = u["level"]

    # PPSA only needs the cell centroids: the rest of the segmentation is dropped at once
    seg = None
    if ds.get("segmentation"):
        with open(fill(ds["segmentation"], ds, level)) as f:
            nuc = json.load(f)
        nuc = nuc.get("nuc", nuc)
        seg = {"nuc": {str(k): {"centroid": v["centroid"]} for k, v in nuc.items() if "centroid" in v}}
        del nuc

    embed_dict = torch.load(fill(ds["features"][u["feature"]], ds, level), map_location="cpu")
    cell_ids = [str(c) for c in embed_dict]
    dim = next(iter(embed_dict.values())).numel()
    if embeddings:
        emb = torch.empty((len(cell_ids), dim), dtype=torch.float32, device=device)
        values = list(embed_dict.values())
        for i in range(0, len(values), 50000):
            emb[i : i + 50000] = torch.stack([v.reshape(-1).float() for v in values[i : i + 50000]]).to(device)
        del values
    else:
        emb = torch.empty((len(cell_ids), dim))
    del embed_dict

    prop = pp_prop(fill(ds["proportions"], ds, level))
    with open(fill(ds["spot_dict"], ds, level)) as f:
        spot_dict = {str(k): [str(c) for c in v] for k, v in json.load(f).items()}
    spot_dict = {k: v for k, v in spot_dict.items() if v}

    adata = load_spatial_adata(fill(ds["adata"], ds, level)) if ds.get("adata") else None
    labels = load_ground_truth(ds["ground_truth"], ds, level)

    return {
        "ds": ds,
        "cell_ids": cell_ids,
        "emb": emb,
        "classes": list(prop.columns),
        "prop": prop,
        "spot_dict": spot_dict,
        "adata": adata,
        "adata_name": fill(ds.get("adata_name"), ds, level),
        "seg": seg,
        "labels": labels,
    }


def check_unit(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the inputs of a unit are consistent, and summarises them.

    Args:
        data: The output of `load_unit`.

    Returns:
        A summary, with a list of problems (empty when everything is fine).
    """

    problems = []
    cells = set(data["cell_ids"])
    in_spot = {c for cs in data["spot_dict"].values() for c in cs}
    missing_cells = in_spot - cells
    if missing_cells:
        problems.append(f"{len(missing_cells)} cells of spot_dict have no embedding")
    missing_spots = set(data["spot_dict"]) - set(data["prop"].index)
    if missing_spots:
        problems.append(f"{len(missing_spots)} spots of spot_dict have no proportions")

    labels = data["labels"]
    known = labels[labels.isin(data["classes"])]
    unknown = sorted(set(labels) - set(data["classes"]))
    if unknown:
        problems.append(
            f"ground truth labels absent from the proportions: {unknown} ({(~labels.isin(data['classes'])).sum()} cells, ignored)"
        )
    scored = known.index.intersection(pd.Index(data["cell_ids"]))
    if len(scored) == 0:
        problems.append("no embedded cell has a ground truth label")

    if data["adata"] is not None:
        name = data["adata_name"]
        if name not in data["adata"].uns.get("spatial", {}):
            problems.append(
                f"adata_name {name!r} not in adata.uns['spatial'] ({list(data['adata'].uns.get('spatial', {}))})"
            )
        n_obs_prop = int(data["adata"].obs_names.isin(data["prop"].index).sum())
        if n_obs_prop == 0:
            problems.append("no adata spot has proportions (obs_names must match the proportion index)")
    outside = cells - in_spot
    if outside and (data["adata"] is None or data["seg"] is None):
        problems.append(
            f"{len(outside)} cells lie outside spots but adata or segmentation is missing: PPSA will be gated"
        )
    if outside and data["seg"] is not None:
        nuc = data["seg"].get("nuc", data["seg"])
        no_centroid = sum(1 for c in outside if c not in nuc or "centroid" not in nuc[c])
        if no_centroid:
            problems.append(f"{no_centroid} cells outside spots have no centroid in the segmentation")

    return {
        "cells": len(cells),
        "embed_dim": int(data["emb"].shape[1]),
        "spots": len(data["spot_dict"]),
        "cells_in_spots": len(in_spot),
        "classes": len(data["classes"]),
        "scored_cells": len(scored),
        "scored_in_spots": len(scored.intersection(pd.Index(sorted(in_spot)))),
        "problems": problems,
    }


class UnitSpotDataset(SpotEmbedDataset):
    """
    ``hedest.dataset.SpotEmbedDataset`` reading a unit's embedding matrix, already on the
    device, instead of the embedding dict. The items are the same (the cells of a spot,
    in ``spot_dict`` order, and the spot's proportions); only the lookups are done once.
    """

    def __init__(
        self, spot_dict: Dict[str, List[str]], spot_prop_df: pd.DataFrame, emb: torch.Tensor, row_of: Dict[str, int]
    ):
        self.spot_dict = spot_dict
        self.spot_prop_df = spot_prop_df
        self.spot_ids = list(spot_dict.keys())
        self.embed_size = emb.shape[1]
        self.emb = emb
        self.rows = [torch.tensor([row_of[c] for c in spot_dict[s]], device=emb.device) for s in self.spot_ids]
        self.props = torch.tensor(spot_prop_df.loc[self.spot_ids].values, dtype=torch.float32)

    def __getitem__(self, idx):
        return {"embeddings": self.emb[self.rows[idx]], "proportions": self.props[idx]}


class UnitPPSA:
    """
    PPSA of one unit, set up once for all its runs.

    The expensive part of ``hedest.ppsa`` is finding, for every cell, the local
    proportions it is adjusted with (its own spot, or the neighbour spots of a cell
    outside spots). That depends on the spots and on the cell positions, not on the
    predictions, so the HEDeST PPSA object is built once, with placeholder
    probabilities, and each run only swaps in its own probabilities and global prior
    before calling ``adjust``.
    """

    def __init__(self, data: Dict[str, Any], adjustment: str, gated: bool, beta: float, device: torch.device):
        placeholder = pd.DataFrame(
            np.full((len(data["cell_ids"]), len(data["classes"])), 1.0 / len(data["classes"]), dtype=np.float32),
            index=data["cell_ids"],
            columns=data["classes"],
        )
        self.ppsa = ADJUSTMENT_CLASSES[adjustment](
            placeholder,
            data["spot_dict"],
            data["prop"],
            data["prop"].mean(axis=0),
            adata=data["adata"],
            adata_name=data["adata_name"],
            seg_dict=data["seg"],
            gated=gated,
            beta=beta,
            batch_size=65536,
            device=device,
        )
        self.ppsa.nuc = None  # the segmentation is only needed for the set-up
        row_of = {c: i for i, c in enumerate(data["cell_ids"])}
        self.rows = torch.tensor([row_of[str(c)] for c in self.ppsa.adjustable_cells], device=device)
        self.device = device

    @property
    def method(self) -> str:
        return self.ppsa.method

    @property
    def gated(self) -> bool:
        return self.ppsa.gated

    def adjust(self, probs: torch.Tensor, global_prop: pd.Series) -> torch.Tensor:
        """
        Adjusts the probabilities of a run.

        Args:
            probs: Raw probabilities of every cell of the unit (cells x types, unit order).
            global_prop: Global prior (mean proportions over the training spots).

        Returns:
            Adjusted probabilities, same shape and order. Cells PPSA cannot reach keep
            their raw probabilities, as in ``BasePPSA.adjust``.
        """

        ppsa = self.ppsa
        ppsa.global_prop = global_prop
        ppsa.p_c = torch.tensor(global_prop.values, dtype=torch.float32, device=self.device).clamp(min=ppsa.eps)
        ppsa.p_cell = probs[self.rows].to(self.device)
        out = probs.clone()
        if len(self.rows):
            out[self.rows] = self._adjust_rows()
        return out

    def _adjust_rows(self) -> torch.Tensor:
        # BasePPSA.adjust, returning the adjustable rows as a tensor instead of a DataFrame.
        ppsa = self.ppsa
        return _adjust_tensor(ppsa.p_cell, ppsa.p_local, ppsa.beta_cell, ppsa.p_c, ppsa.eps)


def _adjust_tensor(p_cell, p_local, beta, p_c, eps):
    sim = torch.sum(p_cell * (p_local / p_c), dim=1).clamp(min=eps)
    alpha = (1.0 / sim).clamp(max=1e6)
    p_adj = p_cell * alpha.unsqueeze(1) * (p_local / p_c)
    return (1.0 - beta.unsqueeze(1)) * p_adj + beta.unsqueeze(1) * p_cell


@torch.no_grad()
def predict(model: CellClassifier, emb: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    """``hedest.predict.predict_slide`` on the embedding matrix: same forward, as a tensor."""

    model.eval()
    return torch.cat([model(emb[i : i + batch_size]) for i in range(0, emb.shape[0], batch_size)])


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def score(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Balanced accuracy over the masked cells (None when no cell is left)."""

    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return None
    return float(balanced_accuracy_score(y_true, y_pred))


def evaluate(
    raw: torch.Tensor,
    adj: torch.Tensor,
    ev: Dict[str, Any],
    spot_split: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Scores a run on the cells that have a ground truth.

    Balanced accuracy with the raw and the PPSA-adjusted predictions, overall and on
    subsets of cells: inside / outside spots, in the train / val / test spots of the
    run's split. ``gated`` adjusts only the cells inside spots, ``antigated`` only the
    cells outside spots, both from the same adjusted probabilities.

    Args:
        raw: Raw probabilities (all cells).
        adj: PPSA-adjusted probabilities (all cells).
        ev: Scored cells: ``rows`` into the unit, integer ``y`` labels, ``in_spot`` mask.
        spot_split: Masks over the scored cells for the train/val/test spots.

    Returns:
        The metrics.
    """

    rows, y, in_spot = ev["rows"], ev["y"], ev["in_spot"]
    p_raw = raw[rows].argmax(dim=1).cpu().numpy()
    p_adj = adj[rows].argmax(dim=1).cpu().numpy()
    p_gated = np.where(in_spot, p_adj, p_raw)
    p_anti = np.where(in_spot, p_raw, p_adj)
    n_classes = len(ev["classes"])

    m = {
        "bal_acc_raw": score(y, p_raw),
        "bal_acc_ppsa": score(y, p_adj),
        "bal_acc_gated": score(y, p_gated),
        "bal_acc_antigated": score(y, p_anti),
    }
    for name, mask in [("in_spot", in_spot), ("out_spot", ~in_spot), *spot_split.items()]:
        m[f"bal_acc_raw_{name}"] = score(y, p_raw, mask)
        m[f"bal_acc_ppsa_{name}"] = score(y, p_adj, mask)
        m[f"n_{name}"] = int(mask.sum())
    m["confusion"] = {
        "classes": ev["classes"],
        "raw": confusion_matrix(y, p_raw, labels=range(n_classes)).tolist(),
        "ppsa": confusion_matrix(y, p_adj, labels=range(n_classes)).tolist(),
    }
    return m


# --------------------------------------------------------------------------------------
# Training a unit
# --------------------------------------------------------------------------------------


def run_dir(u: Dict[str, Any], params: Dict[str, Any], seed: int) -> str:
    return os.path.join(u["dir"], params["combination"], f"seed_{seed}")


def _quiet_hedest_logs() -> None:
    """Keeps the gridsearch messages, drops the per-epoch messages of the HEDeST modules."""

    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: not r["name"].startswith("hedest"))
    logger.add(sys.stderr, level="WARNING", filter=lambda r: r["name"].startswith("hedest"))


def run_unit(cfg: Dict[str, Any], u: Dict[str, Any], shard: int = 0, nshards: int = 1) -> None:
    """
    Trains and scores every run of a unit (or one shard of them), skipping finished runs.

    Args:
        cfg: The study configuration.
        u: The unit.
        shard: Index of this shard.
        nshards: Number of shards the runs are split into (runs i with i % nshards == shard).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{u['feature']}/{u['dataset']}/{u['level']}" + (f" [{shard}/{nshards}]" if nshards > 1 else "")
    combos = combinations(cfg)
    runs = [(seed, p) for seed in cfg["seeds"] for p in combos]
    runs = [r for i, r in enumerate(runs) if i % nshards == shard]
    todo = [(s, p) for s, p in runs if not os.path.exists(os.path.join(run_dir(u, p, s), "metrics.json"))]
    logger.info(f"[{tag}] {len(runs)} runs, {len(runs) - len(todo)} already done, device {device}")
    if not todo:
        return

    t0 = time.time()
    data = load_unit(cfg, u, device)
    cell_ids, classes, prop, spot_dict = data["cell_ids"], data["classes"], data["prop"], data["spot_dict"]
    row_of = {c: i for i, c in enumerate(cell_ids)}

    # scored cells: embedded, with a ground truth label among the proportion columns
    labels = data["labels"]
    labels = labels[labels.isin(classes) & labels.index.isin(cell_ids)]
    cell_to_spot = {c: s for s, cs in spot_dict.items() for c in cs}
    ev = {
        "classes": classes,
        "rows": torch.tensor([row_of[c] for c in labels.index], device=device),
        "y": np.array([classes.index(v) for v in labels.values]),
        "in_spot": np.array([c in cell_to_spot for c in labels.index]),
    }
    ev_spot = pd.Series([cell_to_spot.get(c, "") for c in labels.index])
    os.makedirs(u["dir"], exist_ok=True)
    ids_path = os.path.join(u["dir"], "cell_ids.txt")
    if not os.path.exists(ids_path):
        tmp = f"{ids_path}.{os.getpid()}.tmp"  # shards of a unit may get here together
        with open(tmp, "w") as f:
            f.write("\n".join(cell_ids) + "\n")
        os.replace(tmp, ids_path)
    logger.info(
        f"[{tag}] loaded in {time.time() - t0:.0f}s: {len(cell_ids)} cells, {len(spot_dict)} spots, "
        f"{len(classes)} classes, {len(labels)} scored cells ({ev['in_spot'].sum()} in spots), "
        f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB"
    )

    ppsa_cache: Dict[tuple, UnitPPSA] = {}
    data["seg"] = None if all(p["gated"] for _, p in todo) else data["seg"]
    for _, p in todo:
        key = tuple(p[k] for k in PPSA_KEYS)
        if key not in ppsa_cache:
            t1 = time.time()
            ppsa_cache[key] = UnitPPSA(data, p["adjustment"], p["gated"], p["beta"], device)
            logger.info(
                f"[{tag}] PPSA {key} set up in {time.time() - t1:.0f}s: "
                f"{len(ppsa_cache[key].rows)}/{len(cell_ids)} cells adjustable"
            )
    data["seg"] = data["adata"] = None

    scratch = tempfile.mkdtemp(prefix="hedest_gs_")
    split_cache: Dict[tuple, Any] = {}
    t_start, n_done = time.time(), 0
    try:
        for seed, p in todo:
            out = run_dir(u, p, seed)
            if os.path.exists(os.path.join(out, "metrics.json")):
                continue
            t_run = time.time()

            # ---- split and data loaders (as in hedest.run_model.run_hedest) ----
            skey = (seed, p["train_size"], p["val_size"], p["batch_size"])
            if skey not in split_cache:
                split_cache.clear()
                tr_sd, tr_p, va_sd, va_p, te_sd, te_p = split_data(
                    spot_dict, prop, train_size=p["train_size"], val_size=p["val_size"], rs=seed
                )
                loaders = [
                    DataLoader(
                        UnitSpotDataset(sd, pr, data["emb"], row_of),
                        batch_size=p["batch_size"],
                        shuffle=(i == 0),
                        collate_fn=custom_collate,
                    )
                    for i, (sd, pr) in enumerate([(tr_sd, tr_p), (va_sd, va_p), (te_sd, te_p)])
                ]
                split_masks = {
                    name: ev_spot.isin(set(sd)).to_numpy()
                    for name, sd in (("train_spot", tr_sd), ("val_spot", va_sd), ("test_spot", te_sd))
                }
                global_prop = prop.loc[list(tr_sd.keys())].mean(axis=0)
                split_cache[skey] = (loaders, split_masks, global_prop)
            loaders, split_masks, global_prop = split_cache[skey]

            # ---- training (hedest.run_model.run_hedest) ----
            set_seed(seed)
            model = CellClassifier(
                num_classes=len(classes),
                embed_size=data["emb"].shape[1],
                hidden_dims=p["hidden_dims"],
                norm=bool(p["norm"]),
                dropout=float(p["dropout"]),
                device=device,
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=float(p["lr"]))
            work = tempfile.mkdtemp(dir=scratch)
            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                train_loader=loaders[0],
                val_loader=loaders[1],
                test_loader=loaders[2],
                divergence=p["divergence"],
                alpha=float(p["alpha"]),
                num_epochs=int(p["epochs"]),
                out_dir=work,
                rs=seed,
            )
            trainer.train()
            t_train = time.time() - t_run
            trainer.save_history()

            # ---- prediction and PPSA ----
            # after train(), trainer.model is the best model, reloaded from best_model.pth
            test_loss = trainer.evaluate(loaders[2])[0]
            raw = predict(trainer.model, data["emb"])
            ppsa = ppsa_cache[tuple(p[k] for k in PPSA_KEYS)]
            adj = ppsa.adjust(raw, global_prop)

            # ---- scores and outputs ----
            hist_val = trainer.history_val
            metrics = {
                "feature": u["feature"],
                "dataset": u["dataset"],
                "group": u["group"],
                "level": u["level"],
                "n_classes": len(classes),
                **{k: p[k] for k in cfg["grid"]},
                "combination": p["combination"],
                "seed": seed,
                "params": {k: p[k] for k in DEFAULTS},
                "ppsa_method": ppsa.method,
                "ppsa_gated": ppsa.gated,
                "n_cells": len(cell_ids),
                "n_scored": int(len(ev["y"])),
                "best_epoch": int(np.argmin(hist_val)) + 1,
                "best_val_loss": float(np.min(hist_val)),
                "test_loss": float(test_loss),
                **evaluate(raw, adj, ev, split_masks),
                "history": {"train": trainer.history_train, "val": hist_val},
                "time_train_s": round(t_train, 2),
            }

            os.makedirs(out, exist_ok=True)
            shutil.move(trainer.best_model_path, os.path.join(out, "best_model.pth"))
            shutil.move(os.path.join(work, "history.png"), os.path.join(out, "history.png"))
            if cfg["predictions"] != "none":
                dtype = np.float16 if cfg["predictions"] == "float16" else np.float32
                np.savez_compressed(
                    os.path.join(out, "predictions.npz"),
                    raw=raw.cpu().numpy().astype(dtype),
                    ppsa=adj.cpu().numpy().astype(dtype),
                    classes=np.array(classes),
                )
            metrics["time_run_s"] = round(time.time() - t_run, 2)
            with open(os.path.join(out, "metrics.json.tmp"), "w") as f:
                json.dump(metrics, f)
            os.replace(os.path.join(out, "metrics.json.tmp"), os.path.join(out, "metrics.json"))
            shutil.rmtree(work, ignore_errors=True)

            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                el = time.time() - t_start
                logger.info(
                    f"[{tag}] {n_done}/{len(todo)} runs, {el / n_done:.1f}s/run, "
                    f"ETA {el / n_done * (len(todo) - n_done) / 3600:.1f}h | last: {p['combination']} seed {seed} "
                    f"raw {metrics['bal_acc_raw']:.3f} ppsa {metrics['bal_acc_ppsa']:.3f}"
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    logger.info(f"[{tag}] done: {n_done} runs in {(time.time() - t_start) / 3600:.2f}h")


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


@app.command()
def check(
    config: str, unit: Optional[str] = typer.Option(None, help="Only this unit (index or feature/dataset/level).")
):
    """Loads every unit (no training) and reports inconsistencies in the inputs."""

    cfg = load_config(config)
    todo = [get_unit(cfg, unit)] if unit else units(cfg)
    ok = True
    for u in todo:
        t0 = time.time()
        summary = check_unit(load_unit(cfg, u, torch.device("cpu"), embeddings=False))
        problems = summary.pop("problems")
        ok &= not problems
        print(f"[{u['id']:3d}] {u['feature']}/{u['dataset']}/{u['level']} ({time.time() - t0:.0f}s) {summary}")
        for pb in problems:
            print(f"      ! {pb}")
    print("all inputs consistent" if ok else "some inputs need attention (see '!' lines)")


@app.command()
def plan(config: str):
    """Lists the units with their number of runs and how many are finished."""

    cfg = load_config(config)
    combos = combinations(cfg)
    n_per_unit = len(combos) * len(cfg["seeds"])
    all_units = units(cfg)
    total_done = 0
    for u in all_units:
        done = len(glob.glob(os.path.join(u["dir"], "*", "seed_*", "metrics.json")))
        total_done += done
        print(f"[{u['id']:3d}] {u['feature']:>10s} {u['dataset']:>28s} {u['level']:>8s}  {done:6d}/{n_per_unit}")
    print(
        f"{len(all_units)} units x {len(combos)} combinations x {len(cfg['seeds'])} seeds = "
        f"{len(all_units) * n_per_unit} runs, {total_done} done"
    )


@app.command()
def run(
    config: str,
    unit: str = typer.Option(..., help="Unit index (see `plan`) or feature/dataset/level."),
    shard: int = typer.Option(0, help="Shard of the unit's runs handled by this process."),
    nshards: int = typer.Option(1, help="Number of shards the unit's runs are split into."),
):
    """Trains and scores every run of one unit."""

    _quiet_hedest_logs()
    # ModelTrainer empties the CUDA cache after every batch. That only returns memory to
    # the driver (the results are unchanged) but costs 1.4x (A40) to 2.5x (P100) and makes
    # the processes sharing a GPU wait for each other, so it is a no-op here.
    torch.cuda.empty_cache = lambda: None
    cfg = load_config(config)
    run_unit(cfg, get_unit(cfg, unit), shard, nshards)


@app.command()
def collect(config: str):
    """Gathers the metrics of every finished run into {results_dir}/runs.csv."""

    cfg = load_config(config)
    rows = []
    for u in units(cfg):
        for path in sorted(glob.glob(os.path.join(u["dir"], "*", "seed_*", "metrics.json"))):
            with open(path) as f:
                m = json.load(f)
            for key in ("history", "confusion", "params"):
                m.pop(key, None)
            m["hidden_dims"] = _fmt(m["hidden_dims"]) if "hidden_dims" in m else None
            m["path"] = os.path.dirname(path)
            rows.append(m)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], "runs.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    app()
