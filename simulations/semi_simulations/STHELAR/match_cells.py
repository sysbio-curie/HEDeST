"""HoVerNet <-> DAPI cell matching on the STHELAR benchmark samples, and the ground truth it gives.

The STHELAR samples carry two independent segmentations of the same slide: the HoVerNet
nuclei HEDeST is run on (``hovernet.json``) and the DAPI cells the cell types are annotated
on (``sim/{level}/dapi_seg.json``). A HoVerNet nucleus therefore has no type of its own; it
gets the type of the DAPI cell it is matched to. This script builds that matching and writes
the per-level ground truth HEDeST and the gridsearch are scored against.

Matching, in three steps:

1. **Candidates** — a ``cKDTree`` on the DAPI centroids, every DAPI cell within
   ``CANDIDATE_RADIUS`` pixels of the HoVerNet centroid.
2. **Rasterised IoU** — both contours are filled with ``skimage.draw.polygon`` on a shared
   bounding box and the intersection over union of the two masks is computed. Contour
   overlap, not centroid distance.
3. **Mutual best, ``IOU_THRESHOLD``** — a pair is kept when each cell is the other's best
   partner *and* their IoU reaches the threshold. The result is one-to-one.

The DAPI geometry does not change across the annotation levels (each level only relabels the
same cells, strictly refining the previous one), so the matching is computed **once per
sample** from one level and reused for all of them; only ``type_name`` is read per level.

Outputs, per sample, under ``--out-dir``:

    matches_{sample}.csv        one row per matched pair: hovernet_id, dapi_id, iou
    gt_{sample}_{level}.json    {"gt": {hovernet_id: cell type}}, the file the scorers read
    matching_summary.csv        one row per sample: counts, match rate, median IoU

    python simulations/semi_simulations/STHELAR/match_cells.py --sample ovary_s1
    sbatch simulations/semi_simulations/STHELAR/run_match.sh      # all 8 samples
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.draw import polygon

BENCH_ROOT = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "matches")

SAMPLES = [
    "breast_s6",
    "cervix_s0_0",
    "cervix_s0_1",
    "lung_s3",
    "lymph_node_s0",
    "ovary_s1",
    "prostate_s0",
    "skin_s4",
]

CANDIDATE_RADIUS = 30.0  # px, the cKDTree search radius around a HoVerNet centroid
IOU_THRESHOLD = 0.3  # minimum contour IoU of an accepted pair


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------


def load_nuclei(path: str) -> Tuple[List[str], np.ndarray, List[np.ndarray]]:
    """
    Reads a segmentation JSON: the cell ids, their centroids and their contours.

    Centroids and contours are both ``(x, y)`` and in the same global frame in the two
    segmentations, which is what makes them comparable (``bbox`` is not: HoVerNet stores it
    in tile coordinates, so it is ignored here).

    Args:
        path: Path to the ``nuc``-style segmentation JSON.

    Returns:
        ``(ids, centroids, contours)``, aligned; centroids is ``(n, 2)`` float.
    """

    with open(path) as handle:
        nuc = json.load(handle)["nuc"]

    ids = list(nuc)
    centroids = np.array([nuc[i]["centroid"] for i in ids], dtype=np.float64)
    contours = [np.asarray(nuc[i]["contour"], dtype=np.int32) for i in ids]

    del nuc
    gc.collect()

    return ids, centroids, contours


def load_types(path: str) -> Dict[str, str]:
    """
    Reads only the cell type of every cell of a segmentation JSON.

    Args:
        path: Path to the ``nuc``-style segmentation JSON.

    Returns:
        ``{cell id: type_name}``.
    """

    with open(path) as handle:
        nuc = json.load(handle)["nuc"]

    types = {i: entry["type_name"] for i, entry in nuc.items()}

    del nuc
    gc.collect()

    return types


def levels_of(sample: str) -> List[str]:
    """
    The annotation levels of a sample, in order.

    Args:
        sample: The sample name.

    Returns:
        The level folder names, e.g. ``["level0", ..., "level4"]``.
    """

    sim_dir = os.path.join(BENCH_ROOT, sample, "sim")

    return sorted(
        (d for d in os.listdir(sim_dir) if d.startswith("level") and os.path.isdir(os.path.join(sim_dir, d))),
        key=lambda d: int(d.removeprefix("level")),
    )


# ----------------------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------------------


def pair_iou(first: np.ndarray, second: np.ndarray) -> float:
    """
    Intersection over union of two polygons, by rasterising both on a shared bounding box.

    Args:
        first: ``(n, 2)`` contour, ``(x, y)``.
        second: ``(m, 2)`` contour, ``(x, y)``.

    Returns:
        The IoU, 0.0 when the bounding boxes do not overlap.
    """

    x0, y0 = first.min(axis=0)
    x1, y1 = first.max(axis=0)
    u0, v0 = second.min(axis=0)
    u1, v1 = second.max(axis=0)

    if x1 < u0 or u1 < x0 or y1 < v0 or v1 < y0:
        return 0.0

    ox, oy = min(x0, u0), min(y0, v0)
    shape = (int(max(y1, v1) - oy) + 1, int(max(x1, u1) - ox) + 1)

    mask_a = np.zeros(shape, dtype=bool)
    rows, cols = polygon(first[:, 1] - oy, first[:, 0] - ox, shape)
    mask_a[rows, cols] = True

    mask_b = np.zeros(shape, dtype=bool)
    rows, cols = polygon(second[:, 1] - oy, second[:, 0] - ox, shape)
    mask_b[rows, cols] = True

    intersection = int(np.count_nonzero(mask_a & mask_b))
    if intersection == 0:
        return 0.0

    return intersection / int(np.count_nonzero(mask_a | mask_b))


def match_cells(
    hn_centroids: np.ndarray,
    hn_contours: List[np.ndarray],
    dapi_centroids: np.ndarray,
    dapi_contours: List[np.ndarray],
    radius: float = CANDIDATE_RADIUS,
    threshold: float = IOU_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mutual-best contour-IoU matching of two segmentations.

    Args:
        hn_centroids: ``(n, 2)`` HoVerNet centroids.
        hn_contours: The HoVerNet contours, aligned with the centroids.
        dapi_centroids: ``(m, 2)`` DAPI centroids.
        dapi_contours: The DAPI contours, aligned with the centroids.
        radius: Candidate search radius, in pixels.
        threshold: Minimum IoU of an accepted pair.

    Returns:
        ``(hn_index, dapi_index, iou)``, one entry per accepted pair.
    """

    tree = cKDTree(dapi_centroids)
    candidates = tree.query_ball_point(hn_centroids, r=radius)

    n_hn, n_dapi = len(hn_contours), len(dapi_contours)
    best_for_hn = np.full(n_hn, -1, dtype=np.int64)
    best_iou_hn = np.zeros(n_hn)
    best_for_dapi = np.full(n_dapi, -1, dtype=np.int64)
    best_iou_dapi = np.zeros(n_dapi)

    n_pairs = 0
    for i, neighbours in enumerate(candidates):
        if not neighbours:
            continue
        contour = hn_contours[i]
        for j in neighbours:
            value = pair_iou(contour, dapi_contours[j])
            n_pairs += 1
            if value < threshold:
                continue
            if value > best_iou_hn[i]:
                best_iou_hn[i], best_for_hn[i] = value, j
            if value > best_iou_dapi[j]:
                best_iou_dapi[j], best_for_dapi[j] = value, i
        if (i + 1) % 100_000 == 0:
            log(f"   {i + 1}/{n_hn} HoVerNet cells, {n_pairs} pairs scored")

    hn_index = np.flatnonzero(best_for_hn >= 0)
    dapi_index = best_for_hn[hn_index]
    mutual = best_for_dapi[dapi_index] == hn_index  # each is the other's best

    log(f"   {n_pairs} candidate pairs scored, {int(mutual.sum())} mutual-best pairs kept")

    return hn_index[mutual], dapi_index[mutual], best_iou_hn[hn_index][mutual]


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------


def run_sample(sample: str, out_dir: str, force: bool = False) -> dict:
    """
    Matches one sample and writes its pairs and its per-level ground truth.

    Args:
        sample: The sample name.
        out_dir: Where the outputs go.
        force: Whether to recompute even when the pair file is already there.

    Returns:
        A summary row for the sample.
    """

    os.makedirs(out_dir, exist_ok=True)
    pairs_path = os.path.join(out_dir, f"matches_{sample}.csv")
    meta_path = os.path.join(out_dir, f"matches_{sample}.meta.json")
    levels = levels_of(sample)
    t0 = time.time()

    if os.path.exists(pairs_path) and os.path.exists(meta_path) and not force:
        log(f"{sample}: pairs already there, reusing {pairs_path}")
        pairs = pd.read_csv(pairs_path, dtype={"hovernet_id": str, "dapi_id": str})
        with open(meta_path) as handle:
            meta = json.load(handle)
        n_hn, n_dapi = meta["n_hovernet"], meta["n_dapi"]
    else:
        log(f"{sample}: loading HoVerNet")
        hn_ids, hn_centroids, hn_contours = load_nuclei(os.path.join(BENCH_ROOT, sample, "hovernet.json"))
        log(f"{sample}: {len(hn_ids)} HoVerNet nuclei")

        geometry_level = levels[0]  # the DAPI geometry is the same at every level
        log(f"{sample}: loading DAPI geometry from {geometry_level}")
        dapi_path = os.path.join(BENCH_ROOT, sample, "sim", geometry_level, "dapi_seg.json")
        dapi_ids, dapi_centroids, dapi_contours = load_nuclei(dapi_path)
        log(f"{sample}: {len(dapi_ids)} DAPI cells, matching")

        hn_index, dapi_index, iou = match_cells(hn_centroids, hn_contours, dapi_centroids, dapi_contours)
        pairs = pd.DataFrame(
            {
                "hovernet_id": [hn_ids[i] for i in hn_index],
                "dapi_id": [dapi_ids[j] for j in dapi_index],
                "iou": iou,
            }
        )
        pairs.to_csv(pairs_path, index=False)
        n_hn, n_dapi = len(hn_ids), len(dapi_ids)
        with open(meta_path, "w") as handle:
            json.dump(
                {
                    "sample": sample,
                    "n_hovernet": n_hn,
                    "n_dapi": n_dapi,
                    "geometry_level": geometry_level,
                    "candidate_radius": CANDIDATE_RADIUS,
                    "iou_threshold": IOU_THRESHOLD,
                },
                handle,
            )
        del hn_contours, dapi_contours, hn_centroids, dapi_centroids
        gc.collect()
        log(f"{sample}: {len(pairs)} pairs -> {pairs_path}")

    summary = {
        "sample": sample,
        "n_hovernet": n_hn,
        "n_dapi": n_dapi,
        "n_matched": len(pairs),
        "match_rate": len(pairs) / max(n_hn, 1),
        "median_iou": float(pairs["iou"].median()),
        "n_levels": len(levels),
    }

    for level in levels:
        gt_path = os.path.join(out_dir, f"gt_{sample}_{level}.json")
        types = load_types(os.path.join(BENCH_ROOT, sample, "sim", level, "dapi_seg.json"))
        gt = {hn: types[da] for hn, da in zip(pairs["hovernet_id"], pairs["dapi_id"]) if da in types}
        with open(gt_path, "w") as handle:
            json.dump({"gt": gt}, handle)
        summary[f"n_gt_{level}"] = len(gt)
        log(f"{sample}/{level}: {len(gt)} labelled cells -> {os.path.basename(gt_path)}")
        del types, gt
        gc.collect()

    summary["time_s"] = round(time.time() - t0, 1)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default=None, help="One sample; every sample when omitted.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Where the pairs and the ground truth go.")
    parser.add_argument("--force", action="store_true", help="Recompute pairs that are already there.")
    parser.add_argument(
        "--merge", action="store_true", help="Only gather the per-sample summaries into matching_summary.csv."
    )
    args = parser.parse_args()

    if args.merge:
        parts = [
            pd.read_csv(os.path.join(args.out_dir, name))
            for name in sorted(os.listdir(args.out_dir))
            if name.startswith("matching_summary_") and name.endswith(".csv")
        ]
        if not parts:
            raise SystemExit(f"No matching_summary_*.csv under {args.out_dir}")
        merged = os.path.join(args.out_dir, "matching_summary.csv")
        pd.concat(parts, ignore_index=True).sort_values("sample").to_csv(merged, index=False)
        log(f"{len(parts)} summaries -> {merged}")
        return

    samples = [args.sample] if args.sample else SAMPLES
    unknown = set(samples) - set(SAMPLES)
    if unknown:
        raise SystemExit(f"Unknown sample(s): {sorted(unknown)}")

    rows = [run_sample(sample, args.out_dir, force=args.force) for sample in samples]

    path = os.path.join(args.out_dir, f"matching_summary_{samples[0]}.csv" if args.sample else "matching_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    log(f"summary -> {path}")


if __name__ == "__main__":
    main()
