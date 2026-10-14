# %% [markdown]
# # 5. Tissue niches and immune infiltration
#
# **Niches.** A deconvolution study finds tissue regions by clustering the spots on their
# proportions: one label per 55 µm spot. With HEDeST, every cell gets the composition of its 30
# nearest neighbours (~50 µm around it) and the cells are clustered on it (cellular
# neighbourhoods): one label per cell, wherever it lies. Both are fitted on the three slides
# together with the same number of clusters (8), so their compositions can be compared.
#
# **Infiltration.** Do the immune cells enter the epithelial nests, or stay in the stroma around
# them? The epithelial compartment is drawn from the HEDeST cells (20 µm smoothing, epithelial
# cells ≥ 50% of the cells, patches ≥ 5,000 µm²) and each cell gets its signed distance to its
# border (negative inside). This is the distinction pathologists make between intra-epithelial
# and stromal lymphocytes; a spot only says how many T cells share it with epithelial cells.
# %%
from __future__ import annotations

{{SETUP}}
from sklearn.metrics import normalized_mutual_info_score

from lungstudy import niches as N

PART = "05_niches_infiltration"


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}
types = [
    c
    for c in data[SAMPLES[0]].ct
    if any(d.cells["label"].value_counts(normalize=True).get(c, 0) >= 0.005 for d in data.values())
]
types

# %% [markdown]
# ## 5.1 Niches from cells and from spots

# %%
compositions = {s: N.neighbourhood_composition(d, types, k=30) for s, d in data.items()}
cell_niches, cell_centroids = N.fit_niches(compositions, n_clusters=8)
spot_props = {s: d.props[types].div(d.props[types].sum(axis=1), axis=0).fillna(0) for s, d in data.items()}
spot_clusters, spot_centroids = N.fit_niches(spot_props, n_clusters=8)
# Both sets are ordered by their dominant type, so the k-th spot cluster and the k-th cellular
# niche get the same colour: matching colours are matching compositions.
shared_colors = sns.color_palette("tab10", len(cell_centroids))
niche_palette = dict(zip(cell_centroids.index, shared_colors))
cluster_palette = dict(zip(spot_centroids.index, shared_colors))
for s in SAMPLES:
    pd.DataFrame({"niche": cell_niches[s]}).to_csv(paths.result_path(f"niches_cells_{s}.csv"))

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
sns.heatmap(spot_centroids, annot=True, fmt=".2f", cmap="magma_r", vmin=0, vmax=1, ax=axes[0], cbar=False)
axes[0].set_title("8 spot clusters (deconvolution): mean proportions", fontsize=11)
sns.heatmap(
    cell_centroids, annot=True, fmt=".2f", cmap="magma_r", vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "share"}
)
axes[1].set_title("8 cellular niches (HEDeST): mean neighbourhood of a cell", fontsize=11)
fig.tight_layout()
save(fig, "centroids")

# %% [markdown]
# **Maps.** Spot clusters (one disc per spot) and cellular niches (one dot per cell).

# %%
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def niche_maps(d, window, title, contours=False):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
    for ax in axes:
        views.show_tissue(ax, d, window, alpha=0.35, max_pixels=3000)
    labels = spot_clusters[d.name]
    xy = d.spots.loc[labels.index, ["x", "y"]].to_numpy()
    keep = views.in_window(xy, window, d.radius_px)
    patches = [Circle((x, y), d.radius_px) for x, y in xy[keep]]
    axes[0].add_collection(
        PatchCollection(patches, facecolor=[cluster_palette[l] for l in labels[keep]], edgecolor="none")
    )
    axes[0].set_title("spot clusters (deconvolution)", fontsize=12)
    names = cell_niches[d.name]
    colors = {n: niche_palette[n] for n in niche_palette}
    views.add_cells(
        axes[1], d, window, labels=names, colors=colors, contours=contours, alpha=0.9 if not contours else 0.65
    )
    views.add_spot_circles(axes[1], d, window, linewidth=0.3 if not contours else 0.8)
    axes[1].set_title("cellular niches (HEDeST)", fontsize=12)
    for ax, palette in zip(axes, [cluster_palette, niche_palette]):
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=n)
            for n, c in palette.items()
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8)
        views.add_scalebar(ax, d, 1000 if window[1][0] * d.mpp > 3000 else 100)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


for sample, d in data.items():
    display(save(niche_maps(d, views.slide_extent(d), sample), f"maps_{sample}"))

# %%
for sample in SAMPLES:
    d = data[sample]
    site = pick_sites(d).get("interface")
    display(
        save(
            niche_maps(
                d,
                d.window(site["x"], site["y"], 500),
                f"{sample}: 500 µm around an epithelial / stroma interface",
                contours=True,
            ),
            f"maps_zoom_{sample}",
        )
    )

# %% [markdown]
# ## 5.2 One spot, several niches
#
# For every spot with at least 8 nuclei, how many cellular niches share it, each holding at least
# 25% of its nuclei. A spot cluster gives it a single label.

# %%
rows, diversity = [], []
for sample, d in data.items():
    table = N.within_spot_diversity(d, cell_niches[sample])
    diversity.append(table.assign(sample=sample))
    inside = d.cells[d.cells["in_spot"]]
    spot_label = spot_clusters[sample].reindex(inside["spot"]).to_numpy()
    keep = pd.notna(spot_label)
    rows.append(
        {
            "sample": sample,
            "spots with ≥ 2 niches": float((table["n_niches"] >= 2).mean()),
            "spots with ≥ 3 niches": float((table["n_niches"] >= 3).mean()),
            "NMI(cell niche, spot cluster)": normalized_mutual_info_score(
                cell_niches[sample].loc[inside.index][keep], spot_label[keep]
            ),
        }
    )
diversity = pd.concat(diversity)
summary = pd.DataFrame(rows).set_index("sample")
summary.to_csv(paths.result_path("niches_summary.csv"))
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
summary[["spots with ≥ 2 niches", "spots with ≥ 3 niches"]].plot.bar(
    ax=axes[0], color=["#6baed6", "#08519c"], width=0.7
)
axes[0].set_ylabel("share of spots")
axes[0].tick_params(axis="x", rotation=15)
sns.histplot(
    data=diversity,
    x="dominant_share",
    hue="sample",
    palette=SAMPLE_COLORS,
    element="step",
    stat="density",
    common_norm=False,
    bins=20,
    ax=axes[1],
)
axes[1].set_xlabel("share of a spot's nuclei in its main cellular niche")
fig.tight_layout()
save(fig, "within_spot_diversity")
summary.round(3)

# %%
inside_rows = []
for sample, d in data.items():
    inside = d.cells[d.cells["in_spot"]]
    inside_rows.append(
        pd.DataFrame(
            {
                "niche": cell_niches[sample].loc[inside.index].to_numpy(),
                "spot cluster": spot_clusters[sample].reindex(inside["spot"]).to_numpy(),
            }
        )
    )
pooled = pd.concat(inside_rows).dropna()
table = pd.crosstab(pooled["spot cluster"], pooled["niche"], normalize="index").reindex(
    index=spot_centroids.index, columns=cell_centroids.index
)
fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(
    table,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax,
    cbar_kws={"label": "share of the nuclei of the spot cluster"},
)
ax.set_ylabel("spot cluster (deconvolution)")
ax.set_xlabel("cellular niche of the nuclei (HEDeST)")
ax.set_title("What the nuclei of each spot cluster are, at the cell level", fontsize=11)
save(fig, "cluster_vs_niche")

# %% [markdown]
# **Niches of the nuclei in spots vs between spots**: the 74% of nuclei that no spot measures.

# %%
rows = []
for sample, d in data.items():
    for where, selection in (("in spots", d.cells["in_spot"]), ("between spots", ~d.cells["in_spot"])):
        shares = cell_niches[sample].loc[d.cells.index[selection]].value_counts(normalize=True)
        rows.append(shares.rename(f"{sample}\n{where}"))
frame = pd.DataFrame(rows).reindex(columns=cell_centroids.index).fillna(0)
fig, ax = plt.subplots(figsize=(13, 4.5))
frame.plot.bar(stacked=True, color=[niche_palette[n] for n in frame.columns], ax=ax, width=0.75)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
ax.set_ylabel("share of nuclei")
ax.tick_params(axis="x", rotation=0, labelsize=8)
save(fig, "niches_in_vs_between")

# %% [markdown]
# ## 5.3 Immune infiltration of the epithelium
#
# Density of each immune type (cells per mm² of tissue) by signed distance to the border of the
# epithelial compartment: negative inside the epithelium, positive in the stroma.

# %%
compartments, profiles = {}, []
immune = [t for t in ["T cell", "B cell", "macrophage"] if t in types]
for sample, d in data.items():
    compartments[sample] = N.epithelial_compartment(d)
    profiles.append(
        N.infiltration_profile(d, compartments[sample], immune + ["epithelial", "CAF/fibro"]).assign(sample=sample)
    )
profiles = pd.concat(profiles)
profiles.to_csv(paths.result_path("infiltration_profiles.csv"), index=False)

fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5.5 * len(SAMPLES), 4.5))
for ax, sample in zip(axes, SAMPLES):
    subset = profiles[profiles["sample"] == sample]
    for cell_type in immune + ["CAF/fibro"]:
        curve = subset[subset["cell_type"] == cell_type]
        ax.plot(
            curve["distance_um"],
            curve["density_per_mm2"],
            marker="o",
            markersize=3,
            color=data[sample].palette.rgb(cell_type),
            label=cell_type,
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.axvspan(-100, 0, color="#fde0dd", zorder=0)
    ax.text(-95, ax.get_ylim()[1] * 0.92, "inside the\nepithelium", fontsize=8)
    ax.set_xlabel("signed distance to the epithelial border (µm)")
    ax.set_title(sample, fontsize=11)
axes[0].set_ylabel("cells per mm² of tissue")
axes[-1].legend(frameon=False, fontsize=9)
fig.tight_layout()
save(fig, "infiltration_profiles")

# %% [markdown]
# **The compartments on the slides**: the epithelial compartment (outlined), the T cells (green)
# and the B lineage (blue).

# %%
from matplotlib.colors import ListedColormap

for sample in SAMPLES:
    d = data[sample]
    comp = compartments[sample]
    site = pick_sites(d)["interface"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
    for ax, window in zip(axes, [views.slide_extent(d), d.window(site["x"], site["y"], 600)]):
        views.show_tissue(ax, d, window, alpha=0.5, max_pixels=3000)
        mask = comp["mask"]
        origin, bin_um = comp["origin"], comp["bin_um"]
        extent = (
            origin[0] / d.mpp,
            (origin[0] + mask.shape[1] * bin_um) / d.mpp,
            (origin[1] + mask.shape[0] * bin_um) / d.mpp,
            origin[1] / d.mpp,
        )
        ax.contour(
            np.linspace(extent[0], extent[1], mask.shape[1]),
            np.linspace(extent[3], extent[2], mask.shape[0]),
            mask.astype(float),
            levels=[0.5],
            colors="#d6604d",
            linewidths=1.2,
        )
        contours = window[1][0] * d.mpp <= 700
        views.add_cells(ax, d, window, contours=contours, cell_types=["T cell", "B cell"], alpha=0.85)
        views.add_scalebar(ax, d, 1000 if not contours else 100)
    axes[0].set_title(f"{sample}: epithelial compartment (red outline), T and B cells", fontsize=11)
    axes[1].set_title("600 µm zoom", fontsize=11)
    fig.tight_layout()
    display(save(fig, f"compartment_{sample}"))

# %% [markdown]
# ## 5.4 What a spot says, and where its T cells actually are
#
# **The deconvolution view.** T-cell proportion of the spots by their epithelial proportion: T cells
# co-occur with epithelial cells in many spots.
#
# **The cell view.** For the *mixed spots* (epithelial ≥ 50% and T cells ≥ 10% of the
# deconvolution), where are their T cells: inside the epithelium (< −5 µm from the border), at
# the border (−5 to +15 µm) or in the stroma (> 15 µm)?

# %%
fig, axes = plt.subplots(1, 2, figsize=(17, 4.5))
rows, where_rows = [], []
for sample, d in data.items():
    if "T cell" not in d.props or d.props["T cell"].mean() < 0.005:
        continue
    bins = pd.cut(d.props["epithelial"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    rows.append(
        pd.DataFrame(
            {"sample": sample, "epithelial proportion": bins.astype(str), "T-cell proportion": d.props["T cell"]}
        )
    )
    mixed = d.props.index[(d.props["epithelial"] >= 0.5) & (d.props["T cell"] >= 0.1)]
    cells = d.cells[d.cells["in_spot"] & d.cells["spot"].isin(mixed) & (d.cells["label"] == "T cell")]
    distance = compartments[sample]["cells"].loc[cells.index]
    where = pd.cut(distance, [-1e9, -5, 15, 1e9], labels=["inside the epithelium", "at the border", "in the stroma"])
    where_rows.append(
        where.value_counts(normalize=True).rename(f"{sample}\n({len(mixed)} mixed spots,\n{len(cells)} T cells)")
    )
rows = pd.concat(rows)
sns.boxplot(
    data=rows,
    x="epithelial proportion",
    y="T-cell proportion",
    hue="sample",
    palette=SAMPLE_COLORS,
    showfliers=False,
    ax=axes[0],
    order=sorted(rows["epithelial proportion"].unique()),
)
axes[0].set_title("deconvolution: T cells in spots of every epithelial content", fontsize=11)
axes[0].legend(frameon=False, fontsize=8)
where = pd.DataFrame(where_rows)[["inside the epithelium", "at the border", "in the stroma"]]
where.plot.bar(stacked=True, color=["#d6604d", "#f4a582", "#92c5de"], ax=axes[1], width=0.7)
axes[1].set_ylabel("share of the T cells of the mixed spots")
axes[1].tick_params(axis="x", rotation=0, labelsize=8)
axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
axes[1].set_title("HEDeST: where the T cells of the mixed spots are", fontsize=11)
fig.tight_layout()
save(fig, "mixed_spots_t_cells")
where.round(3)

# %% [markdown]
# **Examples of mixed spots** (the three with the most T cells by deconvolution among those with
# ≥ 50% epithelium), on A_AAAA1879: the pie says "epithelial + T", the cells say on which side of
# the border the T cells are.

# %%
for sample in ["A_AAAA1879", "A_AAAA1665"]:
    d = data[sample]
    mixed = d.props[(d.props["epithelial"] >= 0.5) & (d.spots["n_cells"].reindex(d.props.index) >= 10)]
    for rank, spot in enumerate(mixed["T cell"].sort_values(ascending=False).index[:3]):
        x, y = d.spot_xy(spot)
        window = d.window(x, y, 250)
        fig = views.compare_panels(
            d,
            window,
            panels=("pies", "cells"),
            panel_size=5.5,
            title=f"{sample} — spot {spot}: epithelial {d.props.loc[spot, 'epithelial']:.0%}, T cell {d.props.loc[spot, 'T cell']:.0%}",
        )
        comp = compartments[sample]
        mask, origin, bin_um = comp["mask"], comp["origin"], comp["bin_um"]
        for ax in fig.axes[:2]:
            ax.contour(
                np.linspace(origin[0] / d.mpp, (origin[0] + mask.shape[1] * bin_um) / d.mpp, mask.shape[1]),
                np.linspace(origin[1] / d.mpp, (origin[1] + mask.shape[0] * bin_um) / d.mpp, mask.shape[0]),
                mask.astype(float),
                levels=[0.5],
                colors="#d6604d",
                linewidths=2,
            )
            ax.set_xlim(window[0][0], window[0][0] + window[1][0])
            ax.set_ylim(window[0][1] + window[1][1], window[0][1])
        display(save(fig, f"mixed_spot_{sample}_{rank + 1}"))

# %% [markdown]
# ## 5.5 Intra-epithelial vs stromal lymphocytes
#
# Densities a pathologist scores on H&E (TILs), measured on every cell: T and B cells per mm² inside
# the epithelial compartment and in the stroma within 100 µm of it.

# %%
rows = []
for sample, d in data.items():
    comp = compartments[sample]
    grid, tissue = comp["signed"], comp["tissue"]
    pixel_mm2 = (comp["bin_um"] / 1000) ** 2
    area_in = (tissue & (grid < 0)).sum() * pixel_mm2
    area_out = (tissue & (grid > 0) & (grid <= 100)).sum() * pixel_mm2
    distance = comp["cells"]
    for cell_type in ["T cell", "B cell"]:
        selection = d.cells["label"] == cell_type
        rows.append(
            {
                "sample": sample,
                "cell_type": cell_type,
                "intra-epithelial (per mm²)": float(((distance < 0) & selection).sum() / area_in)
                if area_in
                else np.nan,
                "stromal ≤ 100 µm (per mm²)": float(((distance > 0) & (distance <= 100) & selection).sum() / area_out)
                if area_out
                else np.nan,
            }
        )
tils = pd.DataFrame(rows)
tils["stromal / intra-epithelial"] = tils["stromal ≤ 100 µm (per mm²)"] / tils["intra-epithelial (per mm²)"]
tils.to_csv(paths.result_path("infiltration_tils.csv"), index=False)
melted = tils.melt(
    id_vars=["sample", "cell_type"],
    value_vars=["intra-epithelial (per mm²)", "stromal ≤ 100 µm (per mm²)"],
    var_name="where",
    value_name="density",
)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
for ax, cell_type in zip(axes, ["T cell", "B cell"]):
    sns.barplot(
        data=melted[melted["cell_type"] == cell_type],
        x="sample",
        y="density",
        hue="where",
        palette=["#d6604d", "#92c5de"],
        ax=ax,
    )
    ax.set_ylabel(f"{cell_type}s per mm²")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
save(fig, "tils")
tils.round(1)
