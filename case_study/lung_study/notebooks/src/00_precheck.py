# %% [markdown]
# # Pre-check of HEDeST on {{SAMPLE}}
#
# Before comparing HEDeST with the deconvolution, this notebook checks that the HEDeST cell
# types of this slide are sensible: composition, fit to the deconvolution, agreement between
# seeds, cell types over the whole slide and in zooms, galleries of cells per type, nucleus
# shapes and neighbourhoods. It follows LungScape's `notebooks/sample_analysis.ipynb`, with
# the settings of the case study:
#
# | setting | value |
# |---|---|
# | annotation level | **lv3** (cell2location) |
# | proportions | **10pct_filter** (types below 10% of a spot set to 0, the rest renormalised) |
# | HEDeST predictions | **without PPSA**, mean of the **5 seeds** |
#
# Figures are also written to `figures/00_precheck/{{SAMPLE}}/`.
# %%
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

STUDY = Path.cwd().resolve().parent
sys.path.insert(0, str(STUDY))

import warnings

warnings.filterwarnings("ignore")
%matplotlib inline
%config InlineBackend.figure_format = "png"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display

from lungstudy import cache, paths, views
from lungstudy import validation as V
from lungstudy.data import SampleData, pick_sites

from loguru import logger  # after hedest, which sets up its own handlers on import

logger.remove()
logger.add(sys.stderr, level="WARNING")
plt.rcParams["figure.dpi"] = 80
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

SAMPLE = "{{SAMPLE}}"
PART = f"00_precheck/{SAMPLE}"


def save(fig, name):
    return paths.save(fig, PART, name)


display(Markdown(f"**{SAMPLE}**: {paths.ROLES[SAMPLE]}"))

# %% [markdown]
# ## Loading
#
# `SampleData` reads the tables cached by `scripts/build_cache.py` (cells, predictions without
# PPSA, proportions, spots). HEDeST's own `PredAnalyzer` is built on the same run, with the
# predictions **without PPSA** and the seed agreement recomputed on them (the agreement saved
# by the aggregation was computed on the PPSA predictions).

# %%
import scanpy as sc
import torch
from hedest.analysis import PredAnalyzer, load_run
from hedest.analysis.seeds import load_ensemble

data = SampleData(SAMPLE)
run_dir = paths.sample(SAMPLE).run_dir()

run = load_run(run_dir, adjusted=False)
ensemble = load_ensemble(run_dir, adjusted=False)
run.agreement = ensemble.agreement()

adata = sc.read_h5ad(data.meta["adata"])
image_dict = torch.load(data.meta["image_dict"])
analyzer = PredAnalyzer(
    run,
    seg=data.seg_dict(),
    slide_path=data.meta["slide"],
    adata=adata,
    adata_name=data.meta["adata_name"],
    image_dict=image_dict,
    mpp=data.mpp,
    palette=data.palette,
    adjusted=False,
)
print(data)
analyzer

# %% [markdown]
# ## Composition and fit to the deconvolution

# %%
analyzer.describe()

# %%
summary = analyzer.celltype_summary()
summary.to_csv(paths.result_path(f"precheck_{SAMPLE}_celltypes.csv"))
summary.round(3)

# %%
save(analyzer.plot_abundance(), "abundance")

# %% [markdown]
# **Composition inside spots vs between spots.** The deconvolution only describes the cells inside
# the spots (26% of the nuclei); HEDeST also types the 74% between them.

# %%
cells = data.cells
composition = pd.DataFrame(
    {
        "deconvolution\n(mean over spots)": data.props.mean(),
        "HEDeST\ncells in spots": cells.loc[cells["in_spot"], "label"].value_counts(normalize=True),
        "HEDeST\ncells between spots": cells.loc[~cells["in_spot"], "label"].value_counts(normalize=True),
    }
).reindex(data.ct).fillna(0)
present = composition.index[composition.max(axis=1) > 0.002]
fig, ax = plt.subplots(figsize=(7, 4.5))
bottom = np.zeros(composition.shape[1])
for cell_type in present:
    ax.bar(composition.columns, composition.loc[cell_type], bottom=bottom, color=data.palette.rgb(cell_type), label=cell_type, width=0.6)
    bottom += composition.loc[cell_type].to_numpy()
ax.set_ylabel("share")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
ax.set_title(f"{SAMPLE}: composition ({cells['in_spot'].sum():,} cells in spots, {(~cells['in_spot']).sum():,} between)", fontsize=11)
save(fig, "composition_in_vs_between")

# %% [markdown]
# **Fit to the deconvolution on spots never seen in training.** Each seed was trained on 80% of the
# spots; its 10% test spots (rebuilt exactly as HEDeST split them) are predicted from their cells'
# morphology alone. The aggregate of the 5 seeds cannot be scored this way, since a spot held out by
# one seed was seen by the others.

# %%
heldout = V.heldout_metrics(data)["per_type"]
main_types = [c for c in data.ct if data.props[c].mean() > 0.005]
table = (
    heldout[(heldout["method"] == "HEDeST") & heldout["cell_type"].isin(main_types)]
    .groupby("cell_type")[["pearson", "mae", "mean_truth"]]
    .agg(["mean", "std"])
    .round(3)
)
table

# %%
seed0 = PredAnalyzer(load_run(run_dir / "seed_0", adjusted=False), palette=data.palette, adjusted=False)
display(seed0.spot_metrics(subset="held_out").round(3))
save(seed0.plot_proportion_scatter(subset="held_out"), "proportion_scatter_seed0_heldout")

# %%
train_test = []
for seed, group in cache.seed_spots(SAMPLE).groupby("seed"):
    for split, subset in group.groupby("split"):
        subset = subset.set_index("spot")
        truth = data.props.reindex(subset.index).dropna()
        for cell_type in main_types:
            x, y = truth[cell_type], subset.loc[truth.index, cell_type]
            if x.std() > 0 and y.std() > 0:
                train_test.append({"seed": seed, "split": split, "cell_type": cell_type, "pearson": np.corrcoef(x, y)[0, 1]})
train_test = pd.DataFrame(train_test)
fig, ax = plt.subplots(figsize=(9, 4))
sns.barplot(data=train_test, x="cell_type", y="pearson", hue="split", hue_order=["train", "val", "test"], palette=["#bbbbbb", "#8da0cb", "#1b9e77"], errorbar="sd", ax=ax)
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.set_ylabel("Pearson r with the deconvolution\n(per spot, mean over 5 seeds)")
ax.tick_params(axis="x", rotation=30)
ax.set_title("HEDeST spot composition vs deconvolution, by split", fontsize=11)
save(fig, "fit_by_split")

# %%
save(analyzer.plot_confidence(), "confidence")

# %% [markdown]
# ## Agreement between seeds (without PPSA)
#
# The seeds fit the spots equally well but can call a cell differently: their agreement is a
# confidence measure that needs no ground truth.

# %%
seed_summary = ensemble.summary()
seed_summary.round(3)

# %%
save(ensemble.plot_agreement(palette=data.palette), "seed_agreement")

# %%
save(ensemble.plot_abundance_across_seeds(palette=data.palette), "abundance_across_seeds")

# %%
viz = analyzer.visualizer()
extent = views.slide_extent(data)
save(ensemble.plot_agreement_map(viz, window=(extent[0], extent[1]), figsize=(11, 11)), "seed_agreement_map")

# %% [markdown]
# ## The whole slide
#
# H&E, the deconvolution (one pie per spot) and HEDeST (one dot per cell), side by side.

# %%
fig, window = views.whole_slide_compare(data)
save(fig, "whole_slide_compare")

# %%
save(viz.plot_celltype_map(window=(extent[0], extent[1]), figsize=(13, 13)), "celltype_map")

# %% [markdown]
# **The share of each type, per spot (deconvolution) and per 20 µm pixel (HEDeST).** The top row
# is all the deconvolution can draw; the bottom row is the same quantity computed from the HEDeST
# cells, between the spots included.

# %%
shown = [c for c in ["epithelial", "B cell", "T cell", "CAF/fibro", "macrophage"] if c in data.ct and data.props[c].mean() > 0.005]
save(views.type_maps(data, shown, panel_size=4.6), "type_maps")

# %% [markdown]
# ## Zooms
#
# Places picked from the deconvolution alone (see `pick_sites`), each shown at 350 µm as
# H&E with the spots | the deconvolution pies | the HEDeST cells, then as a series of zooms.

# %%
sites = pick_sites(data)
pd.DataFrame(sites).T

# %%
for name, site in sites.items():
    display(Markdown(f"### {name}: {site['why']} (spot `{site['spot']}`)"))
    window = data.window(site["x"], site["y"], 350)
    display(save(views.compare_panels(data, window, title=f"{SAMPLE} — {name}"), f"site_{name}_compare"))
    display(save(views.zoom_series(data, site["x"], site["y"]), f"site_{name}_zooms"))

# %% [markdown]
# **Every cell type on its own** (600 µm around the lymphoid site; the type in colour, the rest grey).

# %%
site = sites.get("lymphoid", sites["busiest"])
save(views.per_type_panels(data, data.window(site["x"], site["y"], 600)), "per_type_panels")

# %% [markdown]
# **HEDeST's own views** (as in `sample_analysis.ipynb`): the nuclei coloured by type with the
# spots, and the deconvolution pies, on the same ~345 µm window around the busiest spot.

# %%
busiest = sites["busiest"]["spot"]
x, y = data.spot_xy(busiest)
WINDOW = ((x - 750, y - 750), (1500, 1500))
save(viz.plot_seg(WINDOW, show_spots=True), "hedest_plot_seg")

# %%
save(viz.plot_slide(WINDOW, spot_prop_df=analyzer.proportions), "hedest_plot_slide_pies")

# %% [markdown]
# ## Single spots
#
# The tissue of a spot, the mean predicted probabilities, the HEDeST composition and the
# deconvolution it is meant to match, then the crops of its cells with their type.

# %%
for name in ("busiest", "lymphoid", "interface"):
    if name not in sites:
        continue
    spot = sites[name]["spot"]
    display(Markdown(f"### {name} spot `{spot}`"))
    display(save(analyzer.plot_spot_report(spot), f"spot_report_{name}"))
    display(save(analyzer.plot_cell_mosaic(spot_id=spot, num_cols=10), f"spot_mosaic_{name}"))

# %% [markdown]
# ## Cell galleries
#
# The most confident cells of each predicted type, then a random sample of each type (the
# honest view).

# %%
save(analyzer.plot_celltype_grid(num_rows=4, num_cols=6), "gallery_max")

# %%
save(analyzer.plot_celltype_grid(num_rows=4, num_cols=6, selection="random"), "gallery_random")

# %% [markdown]
# **Cells in spots vs cells between spots.** Random cells of each main type, taken inside the
# spots and between them: the between-spot cells must look the same as the in-spot ones.

# %%
rng = np.random.default_rng(0)
rows, colors = {}, {}
for cell_type in main_types:
    for where, selection in (("in spots", cells["in_spot"]), ("between", ~cells["in_spot"])):
        ids = cells.index[(cells["label"] == cell_type) & selection]
        if len(ids):
            rows[f"{cell_type} — {where}"] = list(rng.choice(ids, size=min(14, len(ids)), replace=False))
            colors[f"{cell_type} — {where}"] = data.palette.rgb(cell_type)
save(views.gallery_rows(image_dict, rows, colors, ncols=14), "gallery_in_vs_between")

# %% [markdown]
# ## Nucleus shape per predicted type
#
# Measured on the HoVer-Net contours. Lymphocytes should be small and round, fibroblasts and
# smooth muscle elongated, epithelial nuclei large.

# %%
save(analyzer.compare_area(cell_types=main_types), "nucleus_area")

# %%
shape = cells[cells["label"].isin(main_types)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (column, label) in zip(axes, [("area_um2", "area (µm²)"), ("eccentricity", "eccentricity"), ("circularity", "circularity")]):
    sns.violinplot(data=shape, x="label", y=column, order=main_types, palette={c: data.palette.rgb(c) for c in main_types}, cut=0, inner="quartile", ax=ax, density_norm="width")
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=35)
    if column == "area_um2":
        ax.set_yscale("log")
fig.suptitle("nucleus shape per HEDeST type", fontsize=12)
fig.tight_layout()
save(fig, "nucleus_shape")

# %% [markdown]
# **HoVer-Net's own nucleus types per HEDeST type.** HoVer-Net (PanNuke) classifies nuclei from
# their appearance only, without transcriptomics: an independent check.

# %%
crosstab = pd.crosstab(cells["label"], cells["hv_name"], normalize="index").reindex(main_types)
order = [n for n in ["neoplastic", "non-neoplastic epithelial", "inflammatory", "connective", "necrosis", "no label"] if n in crosstab]
hv_colors = {"neoplastic": "#d62728", "non-neoplastic epithelial": "#ff9896", "inflammatory": "#2ca02c", "connective": "#1f77b4", "necrosis": "#bcbd22", "no label": "#7f7f7f"}
fig, ax = plt.subplots(figsize=(8, 4.5))
crosstab[order].plot.barh(stacked=True, color=[hv_colors[n] for n in order], ax=ax, width=0.75)
ax.invert_yaxis()
ax.set_xlabel("share of the cells of the HEDeST type")
ax.set_ylabel("HEDeST type")
ax.legend(title="HoVer-Net type", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
save(fig, "hovernet_per_hedest_type")

# %% [markdown]
# ## Neighbourhoods
#
# Delaunay neighbours closer than 15 µm.

# %%
max_px = 15 / data.mpp
save(analyzer.plot_neighborhood_composition(max_distance=max_px), "neighbourhood_composition")

# %%
save(analyzer.plot_colocalization_graph(max_distance=max_px), "neighbourhood_graph")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, source in zip(axes, ["deconvolution", "predicted"]):
    frame = analyzer.proportions if source == "deconvolution" else analyzer.predicted_proportions
    matrix = frame.reindex(columns=main_types).corr()
    sns.heatmap(matrix, vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f", square=True, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson across spots — " + ("deconvolution" if source == "deconvolution" else "HEDeST, mean per spot"), fontsize=11)
fig.tight_layout()
save(fig, "spot_correlation_deconv_vs_hedest")

# %% [markdown]
# ## Summary

# %%
agreement = run.agreement
fit = heldout[heldout["method"] == "HEDeST"]
summary_row = {
    "sample": SAMPLE,
    "role": paths.ROLES[SAMPLE],
    "cells": len(cells),
    "cells_in_spots": int(cells["in_spot"].sum()),
    "share_between_spots": float((~cells["in_spot"]).mean()),
    "spots": len(data.props),
    "mean_seed_agreement": float(agreement.mean()),
    "unanimous": float((agreement == 1).mean()),
    "test_pearson_mean": float(fit[fit["cell_type"].isin(main_types)].groupby("seed")["pearson"].mean().mean()),
}
pd.Series(summary_row).to_frame(SAMPLE).to_csv(paths.result_path(f"precheck_{SAMPLE}_summary.csv"))
pd.Series(summary_row)
