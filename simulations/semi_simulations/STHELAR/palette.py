"""The STHELAR cell-type colours: one definition, used everywhere in this repository.

The colour code is **hierarchically consistent**. Each level-0 category is a colour *family*
with a base hue -- Epithelial blue, Immune green, Structural orange, Melanocyte purple -- and
the finest (leaf) types are spread inside their family over a hue band with an alternating
lightness, so that even crowded families stay recognisably in-family yet remain easy to tell
apart. Any coarser category is drawn in the **mean colour of its member leaves**, so it reads
as "the parent of" those leaves. A cell type therefore keeps its colour across annotation
levels, across methods, and across every figure in the repository.

The family -> leaf grouping is **derived from the proportions files themselves** rather than
from a hierarchy table. Annotation levels are nested: a coarse class equals the sum of its
children spot by spot, so a leaf can only sit under a coarse class that is >= it everywhere,
and the true parent is the one whose children reproduce it exactly. The tree is therefore
recoverable from two ``proportions.csv`` files, with nothing to keep in sync.

This module lives beside the semi-simulated data it describes, and is the canonical copy. The
scheme originated in HEDeST-bench (``hedest_bench.hier.colors``); it is reproduced here, once,
so that this repository is self-contained. The two agree exactly: on all 39 STHELAR
sample-levels and 218 categories the colours derived here are identical, to the float, to the
ones HEDeST-bench's hand-written hierarchy produces.

    from palette import level_palette
    level_palette("lung_s3", 2)        # {category: (r, g, b)} in 0-1

One copy is deliberately *not* wired to this module: ``external/HistoCell/bench.py`` keeps its
own, so that the vendored package does not depend on this repository.
"""
from __future__ import annotations

import colorsys
import glob
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

BENCH_ROOT = "/cluster/CBIO/data1/lgortana/STHELAR/bench_data"

# Base hue (HLS, 0..1) of the recurring level-0 families; anything else gets an evenly spaced
# hue that keeps its distance from the ones already taken.
_FAMILY_HUE = {
    "Epithelial": 0.60,  # blue
    "Immune": 0.33,  # green
    "Structural": 0.075,  # orange
    "Melanocyte": 0.80,  # purple
}
_HUE_BAND = 0.18  # max total hue spread across a family's leaves (+/- 0.09)
_HUE_BAND_NREF = 6  # family size at which the full hue band is used
_L_LO, _L_HI = 0.34, 0.72  # lightness ramp endpoints, before the zig-zag
_L_ZIG = 0.13  # zig-zag amplitude (alternating leaves)
_L_CLAMP = (0.26, 0.80)  # hard lightness bounds after the zig-zag
_SAT = 0.66


def _family_hue(top_cats: List[str]) -> Dict[str, float]:
    """
    The base hue of every level-0 category.

    Args:
        top_cats: The level-0 category names.

    Returns:
        ``{category: hue}`` in 0-1.
    """

    hues: Dict[str, float] = {}
    free = [c for c in top_cats if c not in _FAMILY_HUE]
    taken = [_FAMILY_HUE[c] for c in top_cats if c in _FAMILY_HUE]
    for i, c in enumerate(free):
        h = (i / max(1, len(free))) % 1.0
        while any(abs(h - t) < 0.06 for t in taken):  # nudge away from the known families
            h = (h + 0.07) % 1.0
        taken.append(h)
        hues[c] = h
    for c in top_cats:
        if c in _FAMILY_HUE:
            hues[c] = _FAMILY_HUE[c]

    return hues


def _leaf_hls(h0: float, i: int, n: int) -> Tuple[float, float, float]:
    """
    Hue, lightness and saturation of leaf ``i`` of ``n`` in the family of base hue ``h0``.

    The hue is spread over a band whose width grows with the family size, so a small family
    stays tight around its base hue (Structural reads orange, not red) while a crowded one
    fans out. The lightness follows a gentle ramp plus an alternating zig-zag, which keeps
    adjacent leaves apart even where the hue step is tiny. A lone leaf gets the pure
    mid-family colour.

    Args:
        h0: The family's base hue.
        i: The leaf's index inside the family.
        n: The family size.

    Returns:
        ``(hue, lightness, saturation)``.
    """

    if n == 1:
        return h0 % 1.0, (_L_LO + _L_HI) / 2.0, _SAT

    t = i / (n - 1)
    band = _HUE_BAND * min(1.0, (n - 1) / (_HUE_BAND_NREF - 1))
    light = _L_LO + (_L_HI - _L_LO) * t + (_L_ZIG if i % 2 == 0 else -_L_ZIG)
    light = min(max(light, _L_CLAMP[0]), _L_CLAMP[1])

    return (h0 + band * (t - 0.5)) % 1.0, light, _SAT


def children(coarse: pd.DataFrame, fine: pd.DataFrame, tol: float = 1e-6) -> Dict[str, List[str]]:
    """
    Groups the fine cell types under their parent, from the proportions alone.

    Args:
        coarse: The coarse level's ``spots x categories`` proportion matrix.
        fine: The fine level's proportion matrix.
        tol: Tolerance on the "is everywhere >=" test.

    Returns:
        ``{coarse category: [fine categories it contains]}``.
    """

    spots = coarse.index.intersection(fine.index)
    C, F = coarse.loc[spots], fine.loc[spots]
    out: Dict[str, List[str]] = {c: [] for c in coarse.columns}
    for leaf in fine.columns:
        f = F[leaf].to_numpy()
        cand = {
            c: float((C[c].to_numpy() - f).sum()) for c in coarse.columns if bool(((C[c].to_numpy() - f) >= -tol).all())
        }
        out[min(cand, key=cand.get)].append(leaf)

    return out


def levels_of(sample: str, bench_root: Optional[str] = None) -> List[int]:
    """
    The annotation levels of a sample that have a proportion matrix, coarse to fine.

    Args:
        sample: The sample name.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        The level numbers, ascending.
    """

    sim = os.path.join(bench_root or BENCH_ROOT, sample, "sim")

    return sorted(
        int(os.path.basename(d)[5:])
        for d in glob.glob(os.path.join(sim, "level*"))
        if os.path.exists(os.path.join(d, "proportions.csv"))
    )


def _read_level(sample: str, level: int, bench_root: Optional[str] = None) -> pd.DataFrame:
    """Reads one level's proportion matrix, spot ids as strings."""

    path = os.path.join(bench_root or BENCH_ROOT, sample, "sim", f"level{level}", "proportions.csv")

    return pd.read_csv(path, index_col=0).rename(index=str)


def leaf_colors(sample: str, bench_root: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    The colour of every finest-level cell type of a sample, grouped by level-0 family.

    Args:
        sample: The sample name.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        ``{leaf cell type: (r, g, b)}`` in 0-1.
    """

    levels = levels_of(sample, bench_root)
    fine = _read_level(sample, levels[-1], bench_root)
    families = children(_read_level(sample, levels[0], bench_root), fine)
    hue = _family_hue(list(families))

    out: Dict[str, Tuple[float, float, float]] = {}
    for family, leaves in families.items():
        for i, leaf in enumerate(leaves):
            out[leaf] = colorsys.hls_to_rgb(*_leaf_hls(hue[family], i, len(leaves)))

    return out


def level_palette(sample: str, level: int, bench_root: Optional[str] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    The colour of every cell type of one sample-level.

    Args:
        sample: The sample name.
        level: The annotation level, as an integer.
        bench_root: The ``bench_data`` root; ``BENCH_ROOT`` when omitted.

    Returns:
        ``{category: (r, g, b)}`` in 0-1, the mean colour of each category's leaves.
    """

    levels = levels_of(sample, bench_root)
    leaves = leaf_colors(sample, bench_root)
    fine = _read_level(sample, levels[-1], bench_root)

    if level == levels[-1]:
        members = {c: [c] for c in fine.columns}
    else:
        members = children(_read_level(sample, level, bench_root), fine)

    return {
        cat: tuple(sum(leaves[m][k] for m in members_of) / len(members_of) for k in range(3))
        for cat, members_of in members.items()
    }
