"""The annotation-level tree, read off the proportions files.

Levels are nested: a coarse class equals the sum of its children spot by spot.
So a fine class can only sit under a coarse class that is >= it everywhere, and
the true parent is the one whose children reproduce it exactly. That makes the
tree recoverable from two `proportions.csv` files, with no hierarchy table to
keep in sync.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

BENCH = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"


def read_level(sample, level):
    return pd.read_csv(f"{BENCH}/{sample}/sim/level{level}/proportions.csv", index_col=0).rename(index=str)


def levels_of(sample):
    sim = f"{BENCH}/{sample}/sim"
    return sorted(
        int(os.path.basename(d)[5:]) for d in glob.glob(sim + "/level*") if os.path.exists(d + "/proportions.csv")
    )


def children(coarse, fine, tol=1e-6):
    """``{coarse class: [fine classes it contains]}`` from the proportions."""
    spots = coarse.index.intersection(fine.index)
    C, F = coarse.loc[spots], fine.loc[spots]
    out = {c: [] for c in coarse.columns}
    for leaf in fine.columns:
        f = F[leaf].to_numpy()
        cand = {
            c: float((C[c].to_numpy() - f).sum()) for c in coarse.columns if bool(((C[c].to_numpy() - f) >= -tol).all())
        }
        out[min(cand, key=cand.get)].append(leaf)
    return out


def parent_map(sample, coarse_level, fine_level):
    """``{class at fine_level: its class at coarse_level}``."""
    if coarse_level == fine_level:
        return {c: c for c in read_level(sample, fine_level).columns}
    out = {}
    for coarse_cls, members in children(read_level(sample, coarse_level), read_level(sample, fine_level)).items():
        for m in members:
            out[m] = coarse_cls
    return out
