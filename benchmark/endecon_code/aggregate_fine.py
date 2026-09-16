"""Sum a fine-grained deconvolution back into a level's reported vocabulary.

The fine run is deconvolved once per sample on the split vocabulary
(Endothelial / Perivascular / Fibroblast instead of Blood_vessel and
Fibroblast_Myofibroblast). Those classes are the same at levels 0, 1 and 2 --
only the aggregation differs -- so one deconvolution serves all three.

The map is derived, not hard-coded: fine -> level 2 comes from `FINE_SPLIT`, and
level 2 -> level 1 -> level 0 is read off the proportions files, since annotation
levels are nested and a coarse class equals the sum of its children spot by spot.

    python aggregate_fine.py --sample lung_s3 --level 1 \
        --fine-proportions <fine run>/proportions.csv --out <dir>/proportions.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from celltype_rules import fine_to_level2
from hierarchy import BENCH, children as _children


def aggregation_map(sample, level):
    """``{fine class: class at `level`}``."""
    rd = lambda k: pd.read_csv(f"{BENCH}/{sample}/sim/level{k}/proportions.csv", index_col=0).rename(index=str)
    l2 = list(rd(2).columns)
    fine2l2 = fine_to_level2(l2)
    if level == 2:
        return fine2l2
    # {class at `level`: [level-2 classes it contains]}, straight from the data
    parent = {}
    for coarse_cls, members in _children(rd(level), rd(2)).items():
        for m in members:
            parent[m] = coarse_cls
    return {f: parent[c] for f, c in fine2l2.items()}


def aggregate(sample, level, fine_csv):
    fine = pd.read_csv(fine_csv, index_col=0).rename(index=str)
    amap = aggregation_map(sample, level)
    missing = [c for c in fine.columns if c not in amap]
    if missing:
        raise SystemExit(f"fine classes with no target at level {level}: {missing}")
    target = list(pd.read_csv(f"{BENCH}/{sample}/sim/level{level}/proportions.csv", nrows=0, index_col=0).columns)
    out = pd.DataFrame(0.0, index=fine.index, columns=target)
    for c in fine.columns:
        out[amap[c]] += fine[c]
    out.index.name = "spot_id"
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--fine-proportions", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = aggregate(a.sample, a.level, a.fine_proportions)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out)
    print(f"{df.shape[0]} spots x {df.shape[1]} classes -> {a.out}")
