# %% [markdown]
# # 2. Coverage and resolution: what the spots see, what HEDeST sees
#
# A Visium spot is a 55 µm disc, and spots are 100 µm apart: the deconvolution describes the
# ~14 nuclei of each disc as one vector of proportions, and nothing about the nuclei in
# between. HEDeST gives a type to every nucleus. This notebook measures what that changes:
#
# 1. how much of the tissue the spots actually sample;
# 2. what the slide looks like at the two resolutions;
# 3. how sharp a tissue boundary is in each (transects);
# 4. whether the cells between spots look like the spot next to them;
# 5. how many cells of a type lie where the deconvolution says the type is absent.
# %%
from __future__ import annotations

{{SETUP}}
from scipy.ndimage import binary_closing

PART = "02_coverage"


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}

# %% [markdown]
# ## 2.1 How much of the tissue do the spots sample?

# %%
rows = []
for sample, d in data.items():
    cells = d.cells
    bin_um = 20
    gx = (cells["x_um"] // bin_um).astype(int)
    gy = (cells["y_um"] // bin_um).astype(int)
    grid = np.zeros((gy.max() + 1, gx.max() + 1), dtype=bool)
    grid[gy, gx] = True
    grid = binary_closing(grid, iterations=2)
    tissue_mm2 = grid.sum() * (bin_um / 1000) ** 2
    spot_area_mm2 = np.pi * (paths.SPOT_DIAMETER_UM / 2 / 1000) ** 2
    spots_with_cells = int((d.spots["n_cells"] > 0).sum())
    rows.append(
        {
            "sample": sample,
            "nuclei": len(cells),
            "nuclei in spots": int(cells["in_spot"].sum()),
            "nuclei between spots": int((~cells["in_spot"]).sum()),
            "share between spots": float((~cells["in_spot"]).mean()),
            "spots with nuclei": spots_with_cells,
            "nuclei per spot (median)": float(d.spots.loc[d.spots["n_cells"] > 0, "n_cells"].median()),
            "tissue area (mm²)": tissue_mm2,
            "share of tissue under a spot": spots_with_cells * spot_area_mm2 / tissue_mm2,
        }
    )
coverage = pd.DataFrame(rows).set_index("sample")
coverage.to_csv(paths.result_path("coverage_summary.csv"))
coverage.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
coverage[["nuclei in spots", "nuclei between spots"]].plot.bar(
    stacked=True, color=["#333333", "#bbbbbb"], ax=axes[0], width=0.7
)
axes[0].set_ylabel("nuclei")
axes[0].set_title("typed by the deconvolution (in spots) vs typed by HEDeST only", fontsize=10)
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(frameon=False)
for sample, d in data.items():
    axes[1].hist(
        d.cells["nearest_spot_dist_um"],
        bins=np.arange(0, 121, 2),
        histtype="step",
        linewidth=1.8,
        color=SAMPLE_COLORS[sample],
        label=sample,
        density=True,
    )
axes[1].axvline(paths.SPOT_DIAMETER_UM / 2, color="black", linestyle="--")
axes[1].text(paths.SPOT_DIAMETER_UM / 2 + 1, axes[1].get_ylim()[1] * 0.9, "spot edge\n(27.5 µm)", fontsize=9)
axes[1].set_xlabel("distance of a nucleus to the nearest spot centre (µm)")
axes[1].set_ylabel("density")
axes[1].legend(frameon=False, fontsize=8)
axes[1].set_title("most nuclei lie outside every spot", fontsize=10)
for sample, d in data.items():
    counts = d.spots.loc[d.spots["n_cells"] > 0, "n_cells"]
    axes[2].hist(
        counts,
        bins=np.arange(0, 60, 2),
        histtype="step",
        linewidth=1.8,
        color=SAMPLE_COLORS[sample],
        label=sample,
        density=True,
    )
axes[2].set_xlabel("nuclei per spot")
axes[2].set_title("each pie averages ~10-20 nuclei", fontsize=10)
fig.tight_layout()
save(fig, "coverage_counts")

# %% [markdown]
# ## 2.2 The measured nuclei vs all the nuclei
#
# Left: the spots. Middle: the nuclei the deconvolution actually describes (inside a spot),
# coloured by their HEDeST type, the others hidden. Right: every nucleus typed by HEDeST.

# %%
def measured_vs_all(d, window, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    views.show_tissue(axes[0], d, window)
    views.add_spot_circles(axes[0], d, window, linewidth=1.0)
    axes[0].set_title("H&E + spots", fontsize=11)
    views.show_tissue(axes[1], d, window, alpha=0.5)
    inside = d.cells["label"].where(d.cells["in_spot"])
    views.add_cells(
        axes[1],
        d,
        window,
        labels=inside.fillna("—"),
        colors={"—": None},
        alpha=0.8,
        contours=window[1][0] * d.mpp <= 700,
    )
    views.add_spot_circles(axes[1], d, window, linewidth=0.8)
    axes[1].set_title(f"nuclei inside spots ({d.cells['in_spot'].mean():.0%} of all)", fontsize=11)
    views.show_tissue(axes[2], d, window, alpha=0.5)
    views.add_cells(axes[2], d, window, alpha=0.8, contours=window[1][0] * d.mpp <= 700)
    views.add_spot_circles(axes[2], d, window, linewidth=0.8)
    axes[2].set_title("every nucleus, typed by HEDeST", fontsize=11)
    for ax in axes:
        views.add_scalebar(ax, d, 100)
    xy = d.cells[["x", "y"]].to_numpy()
    shown = set(d.cells.loc[views.in_window(xy, window), "label"])
    views.add_legend(axes[2], d, [c for c in d.ct if c in shown])
    fig.suptitle(title, fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


for sample, d in data.items():
    sites = pick_sites(d)
    site = sites.get("interface", sites["busiest"])
    display(
        save(
            measured_vs_all(
                d,
                d.window(site["x"], site["y"], 500),
                f"{sample}: the deconvolution describes the nuclei inside the circles only",
            ),
            f"measured_vs_all_{sample}",
        )
    )

# %% [markdown]
# ## 2.3 Four views of the same places
#
# H&E with the spots | the deconvolution as pies | the HEDeST cells counted per spot (what HEDeST
# says at the scale of the spots: it matches the deconvolution) | the HEDeST cells themselves.

# %%
for sample, d in data.items():
    sites = pick_sites(d)
    for name in ("lymphoid", "interface", "myeloid"):
        if name not in sites:
            continue
        site = sites[name]
        window = d.window(site["x"], site["y"], 400)
        fig = views.compare_panels(
            d,
            window,
            panels=("tissue", "pies", "hedest_pies", "cells"),
            panel_size=5.0,
            title=f"{sample} — {name}: {site['why']}",
        )
        display(save(fig, f"four_views_{sample}_{name}"))

# %% [markdown]
# ## 2.4 How sharp is a boundary? Transects
#
# A line is drawn from an epithelial-rich spot to the most lymphoid place of the slide. Along it,
# the share of each type: HEDeST (cells within 15 µm of each point, every 5 µm), the deconvolution
# of the spot the point crosses (thick segments: the line is inside a spot) and the deconvolution
# of the nearest spot (dotted: what any spot-based map shows).

# %%
def pick_transect(d, min_um=350, max_um=700):
    sites = pick_sites(d)
    end = np.array([sites["lymphoid"]["x"], sites["lymphoid"]["y"]])
    xy = d.spots.loc[d.props.index, ["x", "y"]].to_numpy()
    distance = np.linalg.norm(xy - end, axis=1) * d.mpp
    candidates = np.flatnonzero(
        (d.props["epithelial"].to_numpy() >= 0.85) & (distance >= min_um) & (distance <= max_um)
    )
    if not len(candidates):
        candidates = np.flatnonzero((distance >= min_um) & (distance <= max_um))
        candidates = candidates[np.argsort(-d.props["epithelial"].to_numpy()[candidates])][:1]
    start = xy[candidates[np.argmin(distance[candidates])]]
    return start, end


def transect_figure(d, start, end, types):
    profile = views.transect_profile(d, start, end, types)
    centre = (start + end) / 2
    length_um = np.linalg.norm(end - start) * d.mpp
    window = d.window(centre[0], centre[1], length_um * 1.25)
    fig = plt.figure(figsize=(18, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax = fig.add_subplot(grid[0])
    views.show_tissue(ax, d, window)
    views.add_cells(ax, d, window, contours=False, alpha=0.9)
    views.add_spot_circles(ax, d, window, linewidth=0.7)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="black", linewidth=2.5))
    views.add_scalebar(ax, d, 100)
    ax.set_title(f"{d.name}: transect of {length_um:.0f} µm", fontsize=11)
    ax = fig.add_subplot(grid[1])
    for cell_type in types:
        color = d.palette.rgb(cell_type)
        ax.plot(
            profile["position_um"],
            profile[f"hedest_{cell_type}"],
            color=color,
            linewidth=2.2,
            label=f"{cell_type} — HEDeST",
        )
        ax.plot(profile["position_um"], profile[f"nearest_{cell_type}"], color=color, linewidth=1.2, linestyle=":")
        ax.plot(
            profile["position_um"],
            profile[f"deconv_{cell_type}"],
            color=color,
            linewidth=6,
            alpha=0.5,
            solid_capstyle="butt",
        )
    ax.set_xlabel("position along the transect (µm)")
    ax.set_ylabel("share of the type")
    ax.set_ylim(-0.02, 1.02)
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=d.palette.rgb(c), linewidth=2.2, label=c) for c in types]
    handles += [
        Line2D([0], [0], color="grey", linewidth=2.2, label="HEDeST (cells within 15 µm)"),
        Line2D([0], [0], color="grey", linewidth=6, alpha=0.5, label="deconvolution (inside a spot)"),
        Line2D([0], [0], color="grey", linewidth=1.2, linestyle=":", label="deconvolution of the nearest spot"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("share of each type along the line", fontsize=11)
    fig.tight_layout()
    return fig, profile


transects = {}
for sample in SAMPLES:
    d = data[sample]
    start, end = pick_transect(d)
    fig, profile = transect_figure(d, start, end, ["epithelial", "B cell", "T cell"])
    transects[sample] = profile
    display(save(fig, f"transect_{sample}"))

# %% [markdown]
# **Where is the border?** Along each transect, the first position where the epithelial share falls
# below 50%: from the HEDeST cells, and from the nearest-spot map, which can only place it on the
# edge of a spot's Voronoi cell.

# %%
def first_crossing(position, values, level=0.5):
    values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    below = np.flatnonzero(values < level)
    return float(position[below[0]]) if len(below) else np.nan


borders = pd.DataFrame(
    {
        sample: {
            "HEDeST (µm)": first_crossing(
                p["position_um"].to_numpy(),
                p["hedest_epithelial"].rolling(3, center=True, min_periods=1).mean().to_numpy(),
            ),
            "nearest spot (µm)": first_crossing(p["position_um"].to_numpy(), p["nearest_epithelial"].to_numpy()),
        }
        for sample, p in transects.items()
    }
).T
borders["shift (µm)"] = borders["nearest spot (µm)"] - borders["HEDeST (µm)"]
borders

# %% [markdown]
# ## 2.5 Do the cells between spots look like the spot next to them?
#
# If they did, the deconvolution of a spot would describe its surroundings too. For every spot
# with at least 8 nuclei and 8 nuclei around it (the between-spot nuclei for which it is the
# nearest spot), the distance between its deconvolution and the HEDeST composition of (a) its own
# nuclei and (b) the nuclei around it.

# %%
rows = []
for sample, d in data.items():
    cells = d.cells
    inside = pd.crosstab(cells.loc[cells["in_spot"], "spot"], cells.loc[cells["in_spot"], "label"]).reindex(
        columns=d.ct, fill_value=0
    )
    around = pd.crosstab(cells.loc[~cells["in_spot"], "nearest_spot"], cells.loc[~cells["in_spot"], "label"]).reindex(
        columns=d.ct, fill_value=0
    )
    spots = (
        inside.index[(inside.sum(axis=1) >= 8)]
        .intersection(around.index[around.sum(axis=1) >= 8])
        .intersection(d.props.index)
    )
    p = d.props.loc[spots]
    a = inside.loc[spots].div(inside.loc[spots].sum(axis=1), axis=0)
    b = around.loc[spots].div(around.loc[spots].sum(axis=1), axis=0)
    rows.append(
        pd.DataFrame(
            {
                "sample": sample,
                "spot": spots,
                "own nuclei": 0.5 * (a - p).abs().sum(axis=1).to_numpy(),
                "nuclei around": 0.5 * (b - p).abs().sum(axis=1).to_numpy(),
                "own vs around (HEDeST)": 0.5 * (a - b).abs().sum(axis=1).to_numpy(),
            }
        )
    )
ring = pd.concat(rows)
ring.to_csv(paths.result_path("coverage_spot_vs_surroundings.csv"), index=False)
melted = ring.melt(
    id_vars=["sample", "spot"],
    value_vars=["own nuclei", "nuclei around"],
    var_name="HEDeST composition of",
    value_name="distance to the spot's deconvolution",
)
fig, axes = plt.subplots(1, 2, figsize=(17, 4.5))
sns.violinplot(
    data=melted,
    x="sample",
    y="distance to the spot's deconvolution",
    hue="HEDeST composition of",
    split=True,
    inner="quartile",
    cut=0,
    palette=["#1f78b4", "#a6cee3"],
    ax=axes[0],
)
axes[0].set_xlabel("")
axes[0].set_title("the surroundings of a spot differ more from its pie than its own nuclei", fontsize=11)
share = (
    ring.groupby("sample")
    .apply(
        lambda g: pd.Series(
            {
                "around differs by > 0.3": (g["nuclei around"] > 0.3).mean(),
                "around differs by > 0.5": (g["nuclei around"] > 0.5).mean(),
            }
        )
    )
    .loc[SAMPLES]
)
share.plot.bar(ax=axes[1], color=["#a6cee3", "#1f78b4"], width=0.7)
axes[1].set_ylabel("share of spots")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=20)
axes[1].set_title("spots whose surroundings are badly described by their pie\n(total-variation distance)", fontsize=11)
fig.tight_layout()
save(fig, "spot_vs_surroundings")
ring.groupby("sample")[["own nuclei", "nuclei around", "own vs around (HEDeST)"]].median().round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, sample in zip(axes, SAMPLES):
    d = data[sample]
    subset = ring[ring["sample"] == sample].set_index("spot")
    window = views.slide_extent(d)
    views.show_tissue(ax, d, window, alpha=0.4, max_pixels=2500)
    xy = d.spots.loc[subset.index, ["x", "y"]].to_numpy()
    scatter = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=subset["nuclei around"],
        cmap="magma_r",
        vmin=0,
        vmax=0.8,
        s=views.nucleus_marker_size(ax, d, window, diameter_um=70),
        linewidths=0,
    )
    ax.set_title(f"{sample}", fontsize=11)
    views.add_scalebar(ax, d, 1000)
fig.colorbar(scatter, ax=axes, fraction=0.015, label="distance between a spot's pie and the HEDeST cells around it")
save(fig, "spot_vs_surroundings_maps")

# %% [markdown]
# ## 2.6 Cells where the deconvolution says the type is absent
#
# With the 10% filter, a type below 10% of a spot is set to zero. For each HEDeST type: the share
# of its cells whose spot (or nearest spot) has a deconvolution proportion of zero for that type.
# These cells cannot appear in any spot-level analysis.

# %%
rows = []
for sample, d in data.items():
    cells = d.cells
    spot_of = cells["spot"].where(cells["in_spot"], cells["nearest_spot"])
    for cell_type in d.ct:
        members = cells["label"] == cell_type
        if members.sum() < 200:
            continue
        zero = d.props[cell_type].reindex(spot_of[members]).fillna(0).to_numpy() == 0
        rows.append(
            {
                "sample": sample,
                "cell_type": cell_type,
                "cells": int(members.sum()),
                "share where the deconvolution says 0": float(zero.mean()),
            }
        )
absent = pd.DataFrame(rows)
absent.to_csv(paths.result_path("coverage_cells_where_deconvolution_is_zero.csv"), index=False)
fig, ax = plt.subplots(figsize=(11, 4))
sns.barplot(
    data=absent, x="cell_type", y="share where the deconvolution says 0", hue="sample", palette=SAMPLE_COLORS, ax=ax
)
ax.set_xlabel("")
ax.set_ylabel("share of the HEDeST cells of the type\nwhose spot has proportion 0 for it")
ax.legend(frameon=False, fontsize=8)
save(fig, "cells_where_deconvolution_is_zero")
absent.pivot(index="cell_type", columns="sample", values="share where the deconvolution says 0").round(3)

# %% [markdown]
# ## 2.7 Maps of each type at both resolutions

# %%
for sample, d in data.items():
    shown = [c for c in ["epithelial", "B cell", "T cell", "macrophage", "CAF/fibro"] if d.props[c].mean() > 0.01]
    display(save(views.type_maps(d, shown, panel_size=4.2), f"type_maps_{sample}"))
