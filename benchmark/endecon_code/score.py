"""Score one deconvolution run and draw its two figures.

Metrics, per (sample, level):

  mean_pearson   for each cell type, Pearson r between the predicted and the
                 true proportion **across spots**, then averaged over cell
                 types (unweighted -- a rare class counts as much as a dominant
                 one, which is why rare types pull the mean down)
  mae, rmse      elementwise over the spot x cell-type matrix
  mean_jsd       Jensen-Shannon divergence per spot, averaged over spots

Figures, identical in style to every run in benchmark/results/PanoSpace/deconv:

  spot_pies_true_vs_endecon.png   the slide twice, one pie per spot at its own
                                  position: ground truth left, EnDecon right
  deconv_vs_truth.png             per-cell-type scatter against the truth

Colours are the benchmark's hierarchical code -- level-0 categories are colour
families (Epithelial blue, Immune green, Structural orange, Melanocyte purple),
finer types are shades within their family, and an intermediate category is the
mean colour of its leaves. The family/leaf grouping is *derived from the
proportions files themselves*: annotation levels are nested, so a coarse class
equals the sum of its children spot by spot, which makes the tree recoverable
without any hierarchy table to keep in sync. Verified against the recorded
palette.json: 56/56 colours identical.

    python score.py --sample lung_s3 --level 3 --run-dir <dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import colorsys
import glob
import json
import os
import sys

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hierarchy import BENCH, children as _children

_FAM = {"Epithelial": 0.60, "Immune": 0.33, "Structural": 0.075, "Melanocyte": 0.80}
_BAND, _NREF, _LLO, _LHI, _ZIG, _CLAMP, _SAT = 0.18, 6, 0.34, 0.72, 0.13, (0.26, 0.80), 0.66


def _family_hue(cats):
    hues, free = {}, [c for c in cats if c not in _FAM]
    taken = [_FAM[c] for c in cats if c in _FAM]
    for i, c in enumerate(free):
        h = (i / max(1, len(free))) % 1.0
        while any(abs(h - t) < 0.06 for t in taken):
            h = (h + 0.07) % 1.0
        taken.append(h)
        hues[c] = h
    for c in cats:
        if c in _FAM:
            hues[c] = _FAM[c]
    return hues


def _leaf_hls(h0, i, n):
    if n == 1:
        return h0 % 1.0, (_LLO + _LHI) / 2.0, _SAT
    t = i / (n - 1)
    band = _BAND * min(1.0, (n - 1) / (_NREF - 1))
    light = min(max(_LLO + (_LHI - _LLO) * t + (_ZIG if i % 2 == 0 else -_ZIG), _CLAMP[0]), _CLAMP[1])
    return (h0 + band * (t - 0.5)) % 1.0, light, _SAT


def palette(sample, level):
    sim = f"{BENCH}/{sample}/sim"
    lv = sorted(
        int(os.path.basename(d)[5:]) for d in glob.glob(sim + "/level*") if os.path.exists(d + "/proportions.csv")
    )
    rd = lambda k: pd.read_csv(f"{sim}/level{k}/proportions.csv", index_col=0).rename(index=str)
    fine = rd(lv[-1])
    fams = _children(rd(lv[0]), fine)
    hue = _family_hue(list(fams))
    leaf = {}
    for fam, leaves in fams.items():
        for i, lf in enumerate(leaves):
            leaf[lf] = colorsys.hls_to_rgb(*_leaf_hls(hue[fam], i, len(leaves)))
    members = {c: [c] for c in fine.columns} if level == lv[-1] else _children(rd(level), fine)
    return {c: tuple(sum(leaf[m][k] for m in v) / len(v) for k in range(3)) for c, v in members.items()}


def thumbnail(sample):
    with tifffile.TiffFile(f"{BENCH}/{sample}/he.tiff") as tf:
        s = tf.series[0]
        lv = getattr(s, "levels", None)
        return (lv[-1] if lv and len(lv) > 1 else s).asarray(), s.shape[1], s.shape[0]


def draw_pies(ax, xy, V, cts, cmap, r):
    pat, col = [], []
    for (x, y), row in zip(xy, V):
        tot = row.sum()
        if tot <= 0:
            continue
        a0 = 90.0
        for k, c in enumerate(cts):
            f = row[k] / tot
            if f <= 0:
                continue
            a1 = a0 - 360.0 * f
            pat.append(Wedge((x, y), r, a1, a0))
            col.append(cmap[c])
            a0 = a1
    ax.add_collection(PatchCollection(pat, facecolors=col, edgecolors="none", linewidths=0, rasterized=True))


def score(sample, level, run_dir, out_dir, method="direct"):
    truth = pd.read_csv(f"{BENCH}/{sample}/sim/level{level}/proportions.csv", index_col=0).rename(index=str)
    est = pd.read_csv(f"{run_dir}/proportions.csv", index_col=0).rename(index=str)
    st = ad.read_h5ad(f"{BENCH}/{sample}/pseudovisium.h5ad", backed="r")
    sp = pd.DataFrame(np.asarray(st.obsm["spatial"])[:, :2], index=st.obs_names.astype(str), columns=["x", "y"])
    cts = list(truth.columns)
    spots = [s for s in est.index if s in truth.index and s in sp.index]
    G = truth.loc[spots, cts].to_numpy(float)
    E = est.reindex(index=spots, columns=cts).fillna(0.0).to_numpy(float)
    xy = sp.loc[spots, ["x", "y"]].to_numpy(float)

    eps = 1e-12
    P = np.clip(E, 0, None)
    P /= np.clip(P.sum(1, keepdims=True), eps, None)
    Q = np.clip(G, 0, None)
    Q /= np.clip(Q.sum(1, keepdims=True), eps, None)
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(np.where(A > 0, A * np.log((A + eps) / (B + eps)), 0.0), axis=1)
    per = {
        c: dict(
            pearson=float(np.corrcoef(E[:, k], G[:, k])[0, 1]),
            mae=float(np.abs(E[:, k] - G[:, k]).mean()),
            rmse=float(np.sqrt(((E[:, k] - G[:, k]) ** 2).mean())),
            mean_true=float(G[:, k].mean()),
            mean_est=float(E[:, k].mean()),
        )
        for k, c in enumerate(cts)
    }
    ov = dict(
        mae=float(np.abs(E - G).mean()),
        rmse=float(np.sqrt(((E - G) ** 2).mean())),
        mean_jsd=float(np.mean(0.5 * kl(P, M) + 0.5 * kl(Q, M))),
        mean_pearson=float(np.nanmean([v["pearson"] for v in per.values()])),
    )

    cmap = palette(sample, level)
    os.makedirs(out_dir, exist_ok=True)
    img, W, H = thumbnail(sample)
    d, _ = cKDTree(xy).query(xy, k=2)
    pitch = float(np.median(d[:, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(21, 11 * H / W + 1.8))
    for ax, V, t in ((axes[0], G, "ground truth"), (axes[1], E, "EnDecon (PanoSpace deconvolution)")):
        ax.imshow(img, extent=[0, W, H, 0], alpha=0.35, interpolation="bilinear")
        draw_pies(ax, xy, V, cts, cmap, 0.48 * pitch)
        ax.set_xlim(xy[:, 0].min() - 3 * pitch, xy[:, 0].max() + 3 * pitch)
        ax.set_ylim(xy[:, 1].max() + 3 * pitch, xy[:, 1].min() - 3 * pitch)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(t, fontsize=15)
    fig.legend(
        handles=[Patch(facecolor=cmap[c], label=c) for c in cts],
        loc="lower center",
        ncol=min(len(cts), 7),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"{sample} — level {level}   ·   {len(spots):,} spots, {len(cts)} cell types"
        f"   ·   mean r = {ov['mean_pearson']:.3f}, MAE = {ov['mae']:.3f}"
        f"   ·   {method}-granularity fit   ·   pies at 0.48x the spot pitch",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.965])
    fig.savefig(f"{out_dir}/spot_pies_true_vs_endecon.png", dpi=130)
    plt.close(fig)

    n = len(cts)
    ncol = min(n, 6)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.9 * nrow), squeeze=False)
    for k, c in enumerate(cts):
        ax = axes[k // ncol][k % ncol]
        ax.scatter(G[:, k], E[:, k], s=6, alpha=0.3, linewidths=0, color=cmap[c], rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{c}\nr = {per[c]['pearson']:.3f}   MAE = {per[c]['mae']:.3f}", fontsize=10)
        ax.set_xlabel("true proportion")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if k % ncol == 0:
            ax.set_ylabel("EnDecon proportion")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"{sample} — level{level} ({len(spots):,} spots) · "
        f"mean r = {ov['mean_pearson']:.3f} · MAE = {ov['mae']:.3f} · "
        f"JSD = {ov['mean_jsd']:.3f} · {method} fit",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/deconv_vs_truth.png", dpi=140)
    plt.close(fig)

    return dict(
        sample=sample,
        level=level,
        n_spots=len(spots),
        cell_types=cts,
        method=method,
        overall=ov,
        per_cell_type=per,
        palette={c: "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in cmap[c]) for c in cts},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method", default="direct", choices=["direct", "fine"])
    a = p.parse_args()
    r = score(a.sample, a.level, a.run_dir, a.out_dir, a.method)
    print(json.dumps({k: r[k] for k in ("sample", "level", "n_spots", "method", "overall")}, indent=1))
