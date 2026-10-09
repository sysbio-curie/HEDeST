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
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "simulations",
        "semi_simulations",
        "STHELAR",
    ),
)
import palette as _palette  # noqa: E402  (needs the path above first)

BENCH = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"


def read_level(sample, level):
    return pd.read_csv(f"{BENCH}/{sample}/sim/level{level}/proportions.csv", index_col=0).rename(index=str)


def levels_of(sample):
    sim = f"{BENCH}/{sample}/sim"
    return sorted(
        int(os.path.basename(d)[5:]) for d in glob.glob(sim + "/level*") if os.path.exists(d + "/proportions.csv")
    )


def children(coarse, fine, tol=1e-6):
    """``{coarse class: [fine classes it contains]}`` from the proportions.

    The nesting is implemented once, beside the data it describes, because the cell-type
    colour code is built on it; this re-exports it so the callers here are unchanged.
    """
    return _palette.children(coarse, fine, tol=tol)


def parent_map(sample, coarse_level, fine_level):
    """``{class at fine_level: its class at coarse_level}``."""
    if coarse_level == fine_level:
        return {c: c for c in read_level(sample, fine_level).columns}
    out = {}
    for coarse_cls, members in children(read_level(sample, coarse_level), read_level(sample, fine_level)).items():
        for m in members:
            out[m] = coarse_cls
    return out
