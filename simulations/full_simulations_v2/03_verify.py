"""Step 3 - check every file in sim_v2/ and write the dataset summary.

Checks (the ones that held for the legacy sim/, plus the cross-dataset ones):
  * all expected files exist, nothing unexpected in sim_v2/
  * spot_dict: number of spots, no empty spot, no cell in two spots
  * gt: exactly the cells present in spots, sorted by id, one-hot, columns Cluster 0..n-1
  * prop: same columns as gt, rows in spot order, equal to the proportions recomputed from spot_dict + gt
  * image / emb dicts: keys = gt cells in the same order, values identical to the full dicts ("<id>-1" -> <id>)
  * every gt label is the cell's label in the shared K-means for that K (copies: Cluster K, copied from
    cluster 0), every cell is a retained cell of its cluster, and cells requested <= retained cells
  * dup_not_mixed: no spot contains cluster 0 together with its copy
  * perturbed props: rows sum to 1, consistent with (1-s)*p + s*U(0,1) renormalised from the base prop,
    mean change increasing with strength
  * the hardcoded drawn weights are reproduced by draw_imbalanced_weights
Writes figures/datasets_summary.csv and figures/verification_report.txt; exits 1 on any failure.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
from utils.data_simulation import draw_imbalanced_weights  # noqa: E402
from utils.data_simulation import sort_key  # noqa: E402


class Checker:
    def __init__(self):
        self.lines, self.failures = [], []

    def header(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        line = f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
        self.lines.append(line)
        print(line, flush=True)
        if not ok:
            self.failures.append(line)


def perturbation_consistent(base: np.ndarray, perturbed: np.ndarray, strength: float) -> float:
    """Fraction of rows q for which some Z and noise n in [0,1] give q = ((1-s) p + s n) / Z."""

    s, K = strength, base.shape[1]
    ok = []
    for p, q in zip(base, perturbed):
        pos = q > 0
        if np.any(~pos & (p > 0)):
            ok.append(False)
            continue
        lo = max(1 - s, np.max((1 - s) * p[pos] / q[pos]))
        hi = min(1 - s + s * K, np.min((s + (1 - s) * p[pos]) / q[pos]))
        ok.append(lo <= hi * (1 + 1e-9))
    return float(np.mean(ok))


def main() -> None:
    t0 = time.time()
    check = Checker()

    cell_ids = np.load(os.path.join(C.CONSTRUCTION_DIR, "cell_ids.npy"))
    row_of = {c: i for i, c in enumerate(cell_ids.tolist())}
    kmeans = {K: dict(np.load(os.path.join(C.CONSTRUCTION_DIR, f"kmeans_K{K}.npz"))) for K in C.KS}
    emb_full = torch.load(C.EMB_PATH, map_location="cpu", weights_only=False)
    img_full = torch.load(C.IMAGE_DICT_PATH, map_location="cpu", weights_only=False)
    print(f"full dicts loaded in {time.time() - t0:.0f}s", flush=True)

    check.header("== global")
    for K in C.DRAWN_WEIGHT_KS:
        drawn, _ = draw_imbalanced_weights(K, seed=C.SEED)
        check(
            f"K={K} imbalanced weights reproduced by draw_imbalanced_weights",
            np.allclose(drawn, C.IMBALANCED_WEIGHTS[K], rtol=0, atol=1e-12),
        )

    expected = set()
    for d in C.DATASETS:
        tag = C.dataset_tag(d)
        expected |= {
            f"{tag}_{s}" for s in ("prop.csv", "gt.csv", "spot_dict.json", "emb_dict.pt", f"{C.IMAGE_SUFFIX}.pt")
        }
        if d["perturb"]:
            expected |= {f"{tag}_perturb_{s}_prop.csv" for s in C.PERTURBATION_STRENGTHS}
    present = {f for f in os.listdir(C.SIM_DIR) if os.path.isfile(os.path.join(C.SIM_DIR, f))}
    check(
        "all expected files present",
        expected <= present,
        f"missing {sorted(expected - present)}" if expected - present else f"{len(expected)} files",
    )
    check("no unexpected files", present <= expected, f"{sorted(present - expected)}" if present - expected else "")

    labels_seen = defaultdict(lambda: defaultdict(set))  # K -> plain cell id -> labels across datasets
    summary = []

    for d in C.DATASETS:
        tag, K = C.dataset_tag(d), d["K"]
        n_columns = K + int(d["dup"])
        path = lambda suffix: os.path.join(C.SIM_DIR, f"{tag}_{suffix}")  # noqa: E731
        check.header(f"== {tag}")
        km = kmeans[K]
        retained = {k: set(km["pool_idx"][km["pool_labels"] == k].tolist()) for k in range(K)}

        spot_dict = json.load(open(path("spot_dict.json")))
        gt = pd.read_csv(path("gt.csv"), index_col=0)
        gt.index = gt.index.astype(str)
        prop = pd.read_csv(path("prop.csv"), index_col=0)
        prop.index = prop.index.astype(str)
        emb = torch.load(path("emb_dict.pt"), weights_only=False)
        img = torch.load(path(f"{C.IMAGE_SUFFIX}.pt"), weights_only=False)

        cells = [c for bag in spot_dict.values() for c in bag]
        counts = Counter(cells)
        sizes = np.array([len(bag) for bag in spot_dict.values()])
        columns = [f"Cluster {k}" for k in range(n_columns)]
        gt_labels = gt.values.argmax(axis=1)

        # spot_dict
        check("spots", list(spot_dict) == [str(i) for i in range(d["n_spots"])], f"{len(spot_dict)}")
        check(
            "no empty spot",
            bool((sizes > 0).all()),
            f"sizes {sizes.min()}-{sizes.max()}, mean {sizes.mean():.2f}, var {sizes.var(ddof=1):.2f}",
        )
        check("no cell in two spots", max(counts.values()) == 1)

        # gt
        check("gt cells == cells in spots", gt.index.is_unique and set(gt.index) == set(counts), f"{len(gt)} cells")
        check("gt sorted by id", list(gt.index) == sorted(gt.index, key=sort_key))
        check("gt one-hot", bool(np.isin(gt.values, [0, 1]).all() and (gt.values.sum(axis=1) == 1).all()))
        check("gt columns", list(gt.columns) == columns, f"{n_columns} clusters")

        # prop
        check(
            "prop columns == gt columns, rows in spot order",
            list(prop.columns) == columns and list(prop.index) == list(spot_dict),
        )
        label_of = dict(zip(gt.index, gt_labels))
        recomputed = np.zeros(prop.shape)
        for i, bag in enumerate(spot_dict.values()):
            for c in bag:
                recomputed[i, label_of[c]] += 1
            recomputed[i] /= len(bag)
        diff = np.abs(recomputed - prop.values).max()
        check("prop == recomputed from spot_dict + gt", diff <= 1e-12, f"max |diff| {diff:.1e}")

        # cell-level dicts
        check("image_dict keys == gt cells (same order)", list(img) == list(gt.index))
        check(
            "image_dict values == full image dict",
            all(torch.equal(v, img_full[c.split("-")[0]]) for c, v in img.items()),
        )
        check("emb_dict keys == gt cells (same order)", list(emb) == list(gt.index))
        check(
            "emb_dict values == full embeddings", all(torch.equal(v, emb_full[c.split("-")[0]]) for c, v in emb.items())
        )

        # labels vs the shared K-means
        label_ok, retained_ok = True, True
        for c, lab in zip(gt.index, gt_labels):
            row = row_of[c.split("-")[0]]
            km_label = int(km["labels"][row])
            if "-" in c:
                label_ok &= lab == K and km_label == C.DUPLICATED_CLUSTER
            else:
                label_ok &= lab == km_label
                labels_seen[K][c].add(int(lab))
            retained_ok &= row in retained[km_label]
        check(
            f"gt labels == shared K-means K={K} labels" + (" (copies: Cluster K from cluster 0)" if d["dup"] else ""),
            bool(label_ok),
        )
        check("every cell is a retained cell of its cluster", bool(retained_ok))

        weights = np.array(C.dataset_weights(d), dtype=float)
        weights /= weights.sum()
        drawn = (weights * int(d["mean"] * d["n_spots"])).astype(int)
        pool = np.array(
            [len(retained[k]) for k in range(K)] + ([len(retained[C.DUPLICATED_CLUSTER])] if d["dup"] else [])
        )
        check(
            "cells requested per cluster <= retained cells",
            bool((drawn <= pool).all()),
            f"max request/retained {np.max(drawn / pool):.2f}",
        )

        twin_spots = None
        if d["dup"]:
            twin_spots = sum(
                any(label_of[c] == C.DUPLICATED_CLUSTER for c in bag) and any(label_of[c] == K for c in bag)
                for bag in spot_dict.values()
            )
            if d["not_mixed"]:
                check(f"no spot with cluster {C.DUPLICATED_CLUSTER} and its copy (cluster {K})", twin_spots == 0)
            else:
                check.header(f"  [info] spots with cluster {C.DUPLICATED_CLUSTER} and its copy: {twin_spots}")

        # perturbations
        perturb_files = sorted(glob.glob(path("perturb_*_prop.csv")))
        if d["perturb"]:
            changes = []
            for strength in C.PERTURBATION_STRENGTHS:
                q = pd.read_csv(path(f"perturb_{strength}_prop.csv"), index_col=0)
                q.index = q.index.astype(str)
                same_frame = list(q.index) == list(prop.index) and list(q.columns) == list(prop.columns)
                rows_ok = np.allclose(q.values.sum(axis=1), 1) and (q.values >= 0).all()
                consistent = perturbation_consistent(prop.values, q.values, strength)
                changes.append(np.abs(q.values - prop.values).mean())
                check(
                    f"perturb {strength}: same frame, rows sum to 1, consistent with base prop",
                    same_frame and rows_ok and consistent == 1.0,
                    f"consistent rows {consistent:.0%}, mean |diff| {changes[-1]:.4f}",
                )
            check("perturbation grows with strength", bool(np.all(np.diff(changes) > 0)))
        else:
            check("no perturbed props", not perturb_files)

        in_spots = gt.values.sum(axis=0).astype(int)
        summary.append(
            dict(
                tag=tag,
                K=K,
                columns=n_columns,
                spots=d["n_spots"],
                balance=d["balance"],
                duplicate=d["dup"],
                not_mixed=d["not_mixed"],
                perturbations=";".join(map(str, C.PERTURBATION_STRENGTHS)) if d["perturb"] else "",
                target_mean=d["mean"],
                target_var=d["var"],
                spot_size_mean=round(sizes.mean(), 3),
                spot_size_var=round(sizes.var(ddof=1), 3),
                spot_size_min=sizes.min(),
                spot_size_max=sizes.max(),
                cells=len(gt),
                target_share=";".join(f"{w:.4f}" for w in weights),
                cells_drawn=";".join(map(str, drawn)),
                cells_in_spots=";".join(map(str, in_spots)),
                share_in_spots=";".join(f"{x:.4f}" for x in in_spots / in_spots.sum()),
                spots_with_cluster0_and_copy="" if twin_spots is None else twin_spots,
            )
        )

    check.header("== cluster numbering across datasets")
    for K, seen in sorted(labels_seen.items()):
        n_conflicts = sum(len(v) > 1 for v in seen.values())
        check(
            f"K={K}: a cell has the same label in every dataset",
            n_conflicts == 0,
            f"{len(seen)} cells, {n_conflicts} conflicts",
        )

    os.makedirs(C.FIG_DIR, exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(C.FIG_DIR, "datasets_summary.csv"), index=False)
    verdict = f"{len(check.failures)} failed checks" if check.failures else "all checks passed"
    check.header(f"\n{verdict} ({time.time() - t0:.0f}s)")
    with open(os.path.join(C.FIG_DIR, "verification_report.txt"), "w") as f:
        f.write("\n".join(check.lines) + "\n")

    sys.exit(1 if check.failures else 0)


if __name__ == "__main__":
    main()
