"""CellViT <-> HoVerNet nucleus matching, so the two segmentations can be compared cell by cell.

Every STHELAR benchmark sample carries two independent segmentations of the same slide,
``hovernet.json`` and ``cellvit.json``. Both were re-indexed ``0..n-1`` after the 300 um
spot-proximity filter, independently of one another, so **a CellViT id has nothing to do with
the HoVerNet id of the same nucleus**. The other three methods predict on the HoVerNet nuclei
and the ground truth is keyed by HoVerNet id, so CellViT's types have to be carried onto the
HoVerNet ids before anything can be compared.

The matching is the one that built the ground truth: ``match_cells.match_cells`` is imported
from ``simulations/semi_simulations/STHELAR/`` rather than copied, so the CellViT <-> HoVerNet
pairs are produced by exactly the same rule as the HoVerNet <-> DAPI ones (cKDTree candidates
within 30 px -> rasterised contour IoU -> mutual best, IoU >= 0.3, one-to-one).

Outputs, per sample, under ``--out-dir``:

    matches_{sample}.csv        hovernet_id, cellvit_id, iou, hovernet_type, cellvit_type
    matches_{sample}.meta.json  nucleus counts and the PanNuke type histogram of each backend
    matching_summary.csv        one row per sample

The PanNuke type of both partners is stored in the pairs file, so the scoring never has to
re-read the segmentation JSONs (117-358 MB each) and the PanNuke -> level 0 mapping stays a
decision of ``config.PANNUKE_TO_BROAD`` rather than of this script.

    python benchmark/comparison_code/segmentation_match.py --sample prostate_s0
    sbatch benchmark/comparison_code/run_segmentation_match.sh      # the 7 samples
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import sys
import time
from typing import Dict
from typing import List
from typing import Tuple

import config as C
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(C.REPO, "simulations", "semi_simulations", "STHELAR"))
import match_cells as M  # noqa: E402  (needs C.REPO on the path first)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_typed(path: str) -> Tuple[List[str], np.ndarray, List[np.ndarray], List[str]]:
    """
    Reads a segmentation JSON once: ids, centroids, contours and PanNuke type names.

    ``match_cells`` splits this into ``load_nuclei`` and ``load_types``, which parses the file
    twice; here both segmentations are needed whole, so they are read in a single pass.
    Centroids and contours are ``(x, y)`` in the same global frame in both files, which is
    what makes them comparable (``bbox`` is not: it is in tile coordinates).

    Args:
        path: Path to the ``nuc``-style segmentation JSON.

    Returns:
        ``(ids, centroids, contours, types)``, all aligned; centroids is ``(n, 2)`` float.
    """

    with open(path) as handle:
        nuc = json.load(handle)["nuc"]

    ids = list(nuc)
    centroids = np.array([nuc[i]["centroid"] for i in ids], dtype=np.float64)
    contours = [np.asarray(nuc[i]["contour"], dtype=np.int32) for i in ids]
    types = [str(nuc[i]["type_name"]) for i in ids]

    del nuc
    gc.collect()

    return ids, centroids, contours, types


def run_sample(sample: str, out_dir: str, force: bool = False) -> dict:
    """
    Matches the two segmentations of one sample and writes its pairs.

    Args:
        sample: The sample name.
        out_dir: Where the pairs go.
        force: Whether to recompute pairs that are already there.

    Returns:
        A summary row for the sample.
    """

    os.makedirs(out_dir, exist_ok=True)
    pairs_path = C.seg_pairs_path(sample)
    meta_path = C.seg_meta_path(sample)
    t0 = time.time()

    if os.path.exists(pairs_path) and os.path.exists(meta_path) and not force:
        log(f"{sample}: pairs already there, reusing {pairs_path}")
        pairs = pd.read_csv(pairs_path, dtype={"hovernet_id": str, "cellvit_id": str})
        with open(meta_path) as handle:
            meta = json.load(handle)
    else:
        log(f"{sample}: loading HoVerNet")
        hn_ids, hn_centroids, hn_contours, hn_types = load_typed(C.segmentation_json(sample, "hovernet"))
        log(f"{sample}: {len(hn_ids)} HoVerNet nuclei")

        log(f"{sample}: loading CellViT")
        cv_ids, cv_centroids, cv_contours, cv_types = load_typed(C.segmentation_json(sample, "cellvit"))
        log(f"{sample}: {len(cv_ids)} CellViT nuclei, matching")

        hn_index, cv_index, iou = M.match_cells(hn_centroids, hn_contours, cv_centroids, cv_contours)
        pairs = pd.DataFrame(
            {
                "hovernet_id": [hn_ids[i] for i in hn_index],
                "cellvit_id": [cv_ids[j] for j in cv_index],
                "iou": iou,
                "hovernet_type": [hn_types[i] for i in hn_index],
                "cellvit_type": [cv_types[j] for j in cv_index],
            }
        )
        pairs.to_csv(pairs_path, index=False)

        meta = {
            "sample": sample,
            "n_hovernet": len(hn_ids),
            "n_cellvit": len(cv_ids),
            "hovernet_types": dict(collections.Counter(hn_types)),
            "cellvit_types": dict(collections.Counter(cv_types)),
            "candidate_radius": M.CANDIDATE_RADIUS,
            "iou_threshold": M.IOU_THRESHOLD,
        }
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=1)

        del hn_contours, cv_contours, hn_centroids, cv_centroids, hn_types, cv_types
        gc.collect()
        log(f"{sample}: {len(pairs)} pairs -> {pairs_path}")

    # How often the two backends agree once their PanNuke classes are reduced to the three
    # level 0 categories: the headline number of the matching, and a check that the pairs are
    # the same nuclei rather than neighbours.
    broad = {key: pairs[f"{key}_type"].map(C.PANNUKE_TO_BROAD) for key in ("hovernet", "cellvit")}
    both = broad["hovernet"].notna() & broad["cellvit"].notna()

    summary = {
        "sample": sample,
        "n_hovernet": meta["n_hovernet"],
        "n_cellvit": meta["n_cellvit"],
        "n_matched": len(pairs),
        "match_rate_hovernet": len(pairs) / max(meta["n_hovernet"], 1),
        "match_rate_cellvit": len(pairs) / max(meta["n_cellvit"], 1),
        "median_iou": float(pairs["iou"].median()),
        "n_both_broad": int(both.sum()),
        "broad_agreement": float((broad["hovernet"][both] == broad["cellvit"][both]).mean()),
        "time_s": round(time.time() - t0, 1),
    }
    log(
        f"{sample}: {summary['n_matched']} pairs "
        f"({summary['match_rate_hovernet']:.1%} of HoVerNet, {summary['match_rate_cellvit']:.1%} of CellViT), "
        f"median IoU {summary['median_iou']:.3f}, broad agreement {summary['broad_agreement']:.1%}"
    )

    return summary


def merge_summaries(out_dir: str) -> str:
    """
    Gathers the per-sample summaries written by the array tasks into one table.

    Args:
        out_dir: Where the pairs are.

    Returns:
        The path of the merged table.
    """

    parts = [
        pd.read_csv(os.path.join(out_dir, name))
        for name in sorted(os.listdir(out_dir))
        if name.startswith("matching_summary_") and name.endswith(".csv")
    ]
    if not parts:
        raise SystemExit(f"No matching_summary_*.csv under {out_dir}")

    merged = os.path.join(out_dir, "matching_summary.csv")
    pd.concat(parts, ignore_index=True).sort_values("sample").to_csv(merged, index=False)
    log(f"{len(parts)} summaries -> {merged}")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", default=None, help="One sample; every sample when omitted.")
    parser.add_argument("--out-dir", default=C.SEG_MATCH_DIR, help="Where the pairs go.")
    parser.add_argument("--force", action="store_true", help="Recompute pairs that are already there.")
    parser.add_argument("--merge", action="store_true", help="Only gather the per-sample summaries.")
    args = parser.parse_args()

    if args.merge:
        merge_summaries(args.out_dir)
        return

    samples = [args.sample] if args.sample else C.SEG_SAMPLES
    unknown = set(samples) - set(C.SEG_SAMPLES)
    if unknown:
        raise SystemExit(f"Unknown sample(s): {sorted(unknown)}; this study covers {C.SEG_SAMPLES}")

    rows: List[Dict] = [run_sample(sample, args.out_dir, force=args.force) for sample in samples]

    name = f"matching_summary_{samples[0]}.csv" if args.sample else "matching_summary.csv"
    path = os.path.join(args.out_dir, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    log(f"summary -> {path}")


if __name__ == "__main__":
    main()
