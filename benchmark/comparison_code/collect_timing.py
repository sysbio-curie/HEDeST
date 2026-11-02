"""
Collects what each method costs to turn an image plus spot proportions into cell types.

Scope: the 39 STHELAR ``gt`` sample-levels, three seeds each. Segmentation and
deconvolution are excluded throughout -- every method is handed the same HoVer-Net
segmentation and the same proportions, so neither is part of what is being compared. What
is left is the whole of each method's own work, from reading the image to a label per cell.

What is counted, per method
---------------------------
**HEDeST** runs in two steps and is charged for both:

* ``t_feature``     H-Optimus-0: loading the ViT and running tile inference with per-cell
                    pooling. Measured by ``time_hoptimus.py``; see that file for what is
                    inside it. Charged to EVERY seed, although in practice a slide is
                    extracted once and reused by every seed and every level -- see the
                    ``amortised`` columns for what that reuse is worth.
* ``t_load``        reading the embeddings, proportions, spot dictionary and AnnData. Taken
                    from the gridsearch's own log line, per slide. Mostly the 1.1 GB
                    embedding cache being read back, which a single-program implementation
                    would not pay; it is charged anyway rather than argued away.
* ``t_model``       splits, 100 epochs, prediction and PPSA: the gridsearch's ``time_run_s``.

**HistoCell** does it in one step, fine-tuning a ResNet-18 on tiles:

* ``t_model``       its own ``runtime_seconds``, minus ``figure_seconds``. The figures are
                    this repository's additions, not part of the method.

**PanoSpace** runs three stages, two of which are charged:

* ``t_feature``     DINOv2-base feature extraction. PanoSpace caches these features across
                    seeds, so 79 of its 120 gt runs never paid for them. For the same
                    reason H-Optimus-0 is charged to every HEDeST seed, the extraction is
                    added back to every cache hit, at the cost measured on the cache misses
                    of that same slide (recovered from the run logs).
* ``t_model``       the rest of super-resolution (training the head) plus ``annotation``,
                    the optimal-transport assignment, plus ``load_st``.
* excluded          ``segmentation`` (parsing the supplied HoVer-Net JSON, ~31 s) and
                    ``deconvolution`` (a no-op, 0.1 s, since proportions are supplied).

Hardware
--------
Every number is reported per GPU type, and nothing is averaged across them. This is not
bookkeeping: HistoCell's own records run 5.5x slower on this partition's P100 nodes than on
its A40 node, so a mean over a mixed pool would say more about the scheduler than about the
method. The competitors' GPU is recovered by joining each run's log tag to ``sacct``.

The comparison therefore reports **P100**, the one GPU type all three methods have been
measured on. HistoCell and PanoSpace also carry A40 records, and those are kept in
``timing_per_run.csv`` so the hardware sensitivity can be read, but no A40 row exists for
HEDeST: node006 was booked out for days when this was measured, so rather than estimate it,
the A40 column is simply absent. HistoCell's A40 mean is the number to keep in mind when
reading the P100 ranking -- see section 6 of the README.

HEDeST's classifier step is GPU-insensitive and is taken from the published runs: the three
seeds of a unit were scheduled on arbitrary nodes and still agree to 3.2%, so the mix that
invalidates HistoCell's pooled mean is harmless here.

Usage:
    python benchmark/comparison_code/collect_timing.py
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path("/cluster/CBIO/home/lgortana/HEDeST")
RESULTS = REPO / "benchmark/results"
LOGS = REPO / "log"
GRIDSEARCH_LOGS = REPO / "gridsearch/logs"
OUT_DIR = RESULTS / "comparison/timing"
HOPTIMUS_DIR = OUT_DIR / "hoptimus"

# The HEDeST default combination, the one the comparison reports.
COMBINATION = "norm1_do0_a0.01_lr0.0001_l2_hd1024-512"
HEDEST_RUNS = REPO / "models/gridsearch_semi_sim/hoptimus0"

# The annot arrays that produced every HistoCell and PanoSpace run.
ANNOT_JOBS = ["5172915", "5173062", "5173873", "5184912"]
GPU_OF_NODE = {"node003": "p100", "node004": "p100", "node006": "a40"}

TAG_RE = re.compile(r"^===\s*(\S+?)_(\S+)_level(\d+)_(gt|endecon)_seed(\d+)\s*===")
TS = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
DINO_RE = re.compile(TS + r".*Successfully loaded DINOv2")
SAVED_RE = re.compile(TS + r".*Saved features to .*features\.pt")
CACHED_RE = re.compile(TS + r".*Loaded cached features from")
LOAD_RE = re.compile(r"\[hoptimus0/(?P<sample>[^/]+)/(?P<level>level\d+).*?\] loaded in (?P<secs>\d+)s")


# --------------------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------------------
def node_of_task() -> Dict[tuple, str]:
    """Maps every annot array task to the node it ran on, from the accounting database."""

    mapping: Dict[tuple, str] = {}
    for job in ANNOT_JOBS:
        done = subprocess.run(
            ["sacct", "-j", job, "-X", "-o", "JobID%22,NodeList%12", "--noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in done.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and "_" in parts[0]:
                jid, task = parts[0].split("_", 1)
                mapping[(jid, task)] = parts[1]

    return mapping


def competitor_logs() -> pd.DataFrame:
    """
    Reads every annot log and returns one row per attempt.

    Each log opens with the run it is about, so the identity comes from the file and the
    node from ``sacct``. PanoSpace logs also say whether its DINOv2 features were cached,
    and when they were not, the timestamps bracket the extraction.
    """

    node_map = node_of_task()
    rows = []
    for path in sorted(LOGS.glob("annot_*_*.log")):
        _, jid, task = path.stem.split("_", 2)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if not lines or (m := TAG_RE.match(lines[0].strip())) is None:
            continue
        method, sample, level, kind, seed = m.groups()

        t_dino = t_saved = t_cached = None
        if method == "PanoSpace":
            for line in lines:
                if t_dino is None and (mm := DINO_RE.match(line)):
                    t_dino = datetime.strptime(mm[1], "%Y-%m-%d %H:%M:%S")
                if t_saved is None and (mm := SAVED_RE.match(line)):
                    t_saved = datetime.strptime(mm[1], "%Y-%m-%d %H:%M:%S")
                if t_cached is None and (mm := CACHED_RE.match(line)):
                    t_cached = datetime.strptime(mm[1], "%Y-%m-%d %H:%M:%S")

        node = node_map.get((jid, task), "?")
        rows.append(
            {
                "method": method,
                "sample": sample,
                "level": f"level{level}",
                "kind": kind,
                "seed": int(seed),
                "node": node,
                "gpu": GPU_OF_NODE.get(node, "?"),
                "cache": "miss" if t_saved else ("hit" if t_cached else None),
                "dino_extract_s": (t_saved - t_dino).total_seconds() if (t_saved and t_dino) else None,
                "jobid": jid,
                "task": task,
            }
        )

    df = pd.DataFrame(rows)
    # A retried identity has several logs; the last attempt is the one that wrote the result.
    return df.sort_values(["jobid", "task"]).drop_duplicates(["method", "sample", "level", "kind", "seed"], keep="last")


# --------------------------------------------------------------------------------------
# the three methods
# --------------------------------------------------------------------------------------
def hedest_feature_times() -> pd.DataFrame:
    """Reads the per-slide H-Optimus-0 measurements, one row per slide and GPU type."""

    rows = []
    for path in sorted(HOPTIMUS_DIR.glob("*/*.json")):
        gpu = path.parent.name
        if gpu == "cal":  # the calibration run, a bounded number of tiles
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or "t_charged_s" not in rec:
            logger.warning(f"skipping {path}: status={rec.get('status')}")
            continue
        rows.append(
            {
                "sample": rec["sample"],
                "gpu": gpu,
                "t_feature": rec["t_charged_s"],
                "t_feature_model": rec["t_model_s"],
                "t_feature_infer": rec["t_infer_s"],
                "t_feature_setup_excluded": rec.get("t_setup_s"),
                "n_tiles": rec.get("n_tiles"),
                "tiles_per_s": rec.get("tiles_per_s"),
                "n_cells": rec.get("n_cells"),
            }
        )

    return pd.DataFrame(rows)


def hedest_load_times() -> pd.Series:
    """
    Returns the median input-reading time per slide, in seconds.

    From the gridsearch's own logs. The median is taken because the first read of a slide
    in a job pays for a cold page cache and the rest do not, and the median is the cost of
    a run rather than of a job's first run.
    """

    per_sample: Dict[str, List[int]] = {}
    for path in sorted(GRIDSEARCH_LOGS.glob("*.log")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in LOAD_RE.finditer(text):
            per_sample.setdefault(m["sample"], []).append(int(m["secs"]))

    return pd.Series({s: float(np.median(v)) for s, v in per_sample.items()}, name="t_load")


def hedest_runs() -> pd.DataFrame:
    """Reads the classifier half of every gt run at the default combination."""

    rows = []
    for path in sorted(HEDEST_RUNS.glob(f"*/level*/{COMBINATION}/seed_*/metrics.json")):
        info = json.loads(path.read_text())
        if info.get("group") != "STHELAR":
            continue
        rows.append(
            {
                "sample": info["dataset"],
                "level": info["level"],
                "seed": int(info["seed"]),
                "t_model": info["time_run_s"],
                "t_train": info.get("time_train_s"),
                "n_cells": info["n_cells"],
                "n_classes": info["n_classes"],
            }
        )

    return pd.DataFrame(rows)


def histocell_runs(hardware: pd.DataFrame) -> pd.DataFrame:
    """Reads HistoCell's own runtime for every gt run, less the time it spent plotting."""

    gpu_of = {
        (r.sample, r.level, r.seed): r.gpu
        for r in hardware[(hardware.method == "HistoCell") & (hardware.kind == "gt")].itertuples()
    }
    rows = []
    for path in sorted(RESULTS.glob("HistoCell/*/level*/gt/seed*/metrics.json")):
        info = json.loads(path.read_text())
        sample, level = info["sample"], f"level{info['level']}"
        seed = int(info["seed"])
        figures = float(info.get("figure_seconds") or 0.0)
        rows.append(
            {
                "method": "HistoCell",
                "sample": sample,
                "level": level,
                "seed": seed,
                "gpu": gpu_of.get((sample, level, seed), "?"),
                "t_feature": 0.0,  # no separate step: it is in t_model
                "t_load": 0.0,  # likewise
                "t_model": float(info["runtime_seconds"]) - figures,
                "t_excluded_figures": figures,
                "n_cells": info.get("n_cells_total"),
                "n_classes": info.get("n_cell_types"),
            }
        )

    return pd.DataFrame(rows)


def panospace_runs(hardware: pd.DataFrame) -> pd.DataFrame:
    """
    Reads PanoSpace's stage timings for every gt run, charging DINOv2 to every seed.

    A cache hit never ran the extraction, so the cost measured on the cache misses of the
    same slide is added back. Without that, PanoSpace would be credited with a reuse that
    HEDeST is explicitly denied.
    """

    pano = hardware[(hardware.method == "PanoSpace") & (hardware.kind == "gt")]
    gpu_of = {(r.sample, r.level, r.seed): r.gpu for r in pano.itertuples()}
    cache_of = {(r.sample, r.level, r.seed): r.cache for r in pano.itertuples()}
    own_of = {(r.sample, r.level, r.seed): r.dino_extract_s for r in pano.itertuples()}

    measured = pano[pano.dino_extract_s.notna()]
    # Prefer the miss of the same sample-level: the cache is keyed to the sub-spot grid,
    # which changes with the level, and level 0 costs about a third more than the rest.
    # There is normally exactly one miss per sample-level, so this is a measurement rather
    # than an average. The slide and the overall medians are fallbacks.
    per_unit = measured.groupby(["sample", "level"])["dino_extract_s"].median()
    per_slide = measured.groupby("sample")["dino_extract_s"].median()
    overall = float(measured.dino_extract_s.median()) if len(measured) else np.nan
    logger.info(
        f"DINOv2 extraction measured on {len(measured)} cache misses, "
        f"covering {len(per_unit)} of the 39 sample-levels"
    )

    rows = []
    for path in sorted(RESULTS.glob("PanoSpace/*/level*/gt/seed*/run_info.json")):
        seed = int(path.parent.name.replace("seed", ""))
        level = path.parent.parent.parent.name
        sample = path.parent.parent.parent.parent.name
        info = json.loads(path.read_text())
        stages = info.get("timings_sec") or {}
        key = (sample, level, seed)
        cache = cache_of.get(key)

        superres = float(stages.get("superres", 0.0))
        own = own_of.get(key)
        if cache == "miss" and own is not None and not pd.isna(own):
            # This run extracted the features itself, so its own log bracketed the cost and
            # the recorded superres already contains it; no estimate is needed.
            feature = float(own)
            head = max(superres - feature, 0.0)
        else:
            # A cache hit never ran the extraction. Charge it what a miss of the same
            # sample-level cost, or of the same slide, or the overall median.
            feature = float(per_unit.get((sample, level), per_slide.get(sample, overall)))
            head = superres

        rows.append(
            {
                "method": "PanoSpace",
                "sample": sample,
                "level": level,
                "seed": seed,
                "gpu": gpu_of.get(key, "?"),
                "t_feature": feature,
                "t_load": float(stages.get("load_st", 0.0)),
                "t_model": head + float(stages.get("annotation", 0.0)),
                "t_excluded_segmentation": float(stages.get("segmentation", 0.0)),
                "t_excluded_deconvolution": float(stages.get("deconvolution", 0.0)),
                "cache": cache,
                "n_cells": info.get("n_cells_in"),
                "n_classes": info.get("n_cell_types"),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
def main() -> None:
    """Writes the per-run timing table and the per-method summary."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hardware = competitor_logs()
    logger.info(f"{len(hardware)} competitor run attempts identified from their logs")

    features = hedest_feature_times()
    if features.empty:
        raise SystemExit(f"no H-Optimus-0 measurements in {HOPTIMUS_DIR}; run run_time_hoptimus.sh first")
    logger.info(f"H-Optimus-0 measured on {features.gpu.nunique()} GPU type(s): " f"{sorted(features.gpu.unique())}")

    loads = hedest_load_times()
    logger.info(f"input-reading medians (s): {loads.round(0).to_dict()}")

    runs = hedest_runs()
    logger.info(f"{len(runs)} HEDeST gt runs at {COMBINATION}")

    # HEDeST: one row per run and per GPU type the feature step was measured on
    hedest = runs.merge(features[["sample", "gpu", "t_feature"]], on="sample", how="inner")
    hedest = hedest.merge(loads, left_on="sample", right_index=True, how="left")
    hedest["method"] = "HEDeST"

    frames = [hedest, histocell_runs(hardware), panospace_runs(hardware)]
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["t_total"] = df.t_feature.fillna(0) + df.t_load.fillna(0) + df.t_model.fillna(0)

    # The reuse that charging every seed denies, for the record. The two methods cache at
    # different granularities and the difference is most of the story:
    #   HEDeST     one embedding file per SLIDE, read by every level and every seed of it
    #              (15 runs, 12 for the four-level sample).
    #   PanoSpace  one feature cache per SAMPLE-LEVEL -- its key covers the sub-spot grid,
    #              which changes with the level -- so only the 3 seeds share it. That is
    #              visible in the logs: five cache misses per slide, one per level.
    # HistoCell caches nothing; it fine-tunes from ImageNet on every run.
    df["t_total_amortised"] = np.nan
    hedest = df.method == "HEDeST"
    df.loc[hedest, "t_total_amortised"] = (
        df.loc[hedest, "t_feature"] / df.loc[hedest, "sample"].map(runs.groupby("sample").size())
        + df.loc[hedest, "t_load"].fillna(0)
        + df.loc[hedest, "t_model"]
    )

    pano = df.method == "PanoSpace"
    shared = df[pano].groupby(["sample", "level"]).size()
    denominator = pd.MultiIndex.from_frame(df.loc[pano, ["sample", "level"]]).map(shared)
    df.loc[pano, "t_total_amortised"] = (
        df.loc[pano, "t_feature"].to_numpy() / np.asarray(denominator, dtype=float)
        + df.loc[pano, "t_load"].fillna(0).to_numpy()
        + df.loc[pano, "t_model"].to_numpy()
    )

    order = [
        "method",
        "gpu",
        "sample",
        "level",
        "seed",
        "n_cells",
        "n_classes",
        "t_feature",
        "t_load",
        "t_model",
        "t_total",
        "t_total_amortised",
        "t_train",
        "t_excluded_figures",
        "t_excluded_segmentation",
        "t_excluded_deconvolution",
        "cache",
    ]
    df = df[[c for c in order if c in df.columns]].sort_values(["method", "gpu", "sample", "level", "seed"])
    per_run = OUT_DIR / "timing_per_run.csv"
    df.to_csv(per_run, index=False)

    known = df[df.gpu.isin(["p100", "a40"])].copy()
    known["unit"] = known["sample"] + "/" + known["level"]
    summary = (
        known.groupby(["method", "gpu"])
        .agg(
            n_runs=("t_total", "size"),
            n_units=("unit", "nunique"),
            t_feature_mean=("t_feature", "mean"),
            t_load_mean=("t_load", "mean"),
            t_model_mean=("t_model", "mean"),
            t_total_mean=("t_total", "mean"),
            t_total_median=("t_total", "median"),
            t_total_std=("t_total", "std"),
            t_total_min=("t_total", "min"),
            t_total_max=("t_total", "max"),
            t_total_amortised_mean=("t_total_amortised", "mean"),
        )
        .round(1)
        .reset_index()
    )
    out = OUT_DIR / "timing_summary.csv"
    summary.to_csv(out, index=False)

    logger.info(f"-> {per_run}")
    logger.info(f"-> {out}")
    print()
    print(summary.to_string(index=False))
    print()
    unknown = df[~df.gpu.isin(["p100", "a40"])]
    if len(unknown):
        print(f"runs whose GPU could not be recovered, left out of the summary: {len(unknown)}")
        print(unknown.groupby("method").size().to_string())
    print("\ncoverage (runs per method and GPU):")
    print(Counter(zip(known.method, known.gpu)))


if __name__ == "__main__":
    main()
