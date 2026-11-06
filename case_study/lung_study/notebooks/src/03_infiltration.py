# %% [markdown]
# # 3. Immune infiltration of the tumour epithelium
#
# **The question.** Do immune cells, T cells first, enter the tumour nests, or stay in the stroma
# around them? Pathologists call a tumour *inflamed* when T cells are inside the nests, *immune
# excluded* when they stop at the border, *immune desert* when there are none; the distinction
# predicts the response to immunotherapy. It is a question about *which side of a border* a cell
# lies on, often within tens of µm.
#
# **With the deconvolution**, a spot on the border holds tumour and T cells: its pie says
# "epithelial + T" whether the T cells are inside the nest or along it. A spot whose filtered
# proportions say 0% T can still hold a few T cells (the 10% filter erased them, or the
# deconvolution missed them).
#
# **With HEDeST**, every cell is placed. The **epithelial compartment** is drawn from the HEDeST
# cells (10 µm grid, 20 µm smoothing, epithelial cells ≥ 50% of the cells, patches ≥ 5,000 µm²;
# `lungstudy/infiltration.py`), and every cell gets its signed distance to its border (negative
# inside). There is no tumour annotation: the compartment is mostly tumour on these slides, but
# can include normal epithelium.
#
# **Slides**: A_AAAA1879, A_AAAA1638, A_AAAA1665 (`paths.ANALYSIS_SAMPLES`).
#
# Every claim below comes with its deconvolution counterpart and with H&E crops chosen by stated
# rules.
# %%
from __future__ import annotations

{{SETUP}}
from scipy.spatial import cKDTree

from lungstudy import figures as F
from lungstudy import infiltration as I
from lungstudy import style

PART = "03_infiltration"
SAMPLES = paths.ANALYSIS_SAMPLES


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}
raw_props = {s: cache.proportions(s, "lv3", "raw") for s in SAMPLES}
compartments = {s: I.epithelial_compartment(d) for s, d in data.items()}
IMMUNE = ["T cell", "B cell", "macrophage"]
pd.DataFrame(
    {
        SHORT[s]: {
            "epithelial cells (HEDeST)": float((d.cells["label"] == "epithelial").mean()),
            "T cells (HEDeST)": float((d.cells["label"] == "T cell").mean()),
            "epithelial compartment (mm²)": float(
                compartments[s]["mask"].sum() * (compartments[s]["bin_um"] / 1000) ** 2
            ),
            "tissue (mm²)": float(compartments[s]["tissue"].sum() * (compartments[s]["bin_um"] / 1000) ** 2),
        }
        for s, d in data.items()
    }
).round(3)

# %% [markdown]
# ## 3.1 The compartments on the slides
#
# The epithelial compartment (red outline) and the HEDeST T cells (green), on the whole slide and
# on a 600 µm window around the spot where the deconvolution puts the most epithelial and
# CAF/fibro cells together (an interface, chosen from the deconvolution).

# %%
for sample in SAMPLES:
    d = data[sample]
    comp = compartments[sample]
    site = pick_sites(d)["interface"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5))
    for ax, window in zip(axes, [views.slide_extent(d), d.window(site["x"], site["y"], 600)]):
        views.show_tissue(ax, d, window, alpha=0.5, max_pixels=3000)
        contours = window[1][0] * d.mpp <= 700
        views.add_cells(ax, d, window, contours=contours, cell_types=["T cell"], alpha=0.9)
        I.draw_border(ax, d, comp, linewidth=1.2)
        views.add_scalebar(ax, d, 100 if contours else 1000)
    axes[0].set_title(f"{sample}: epithelial compartment (red), T cells (green)", fontsize=11)
    axes[1].set_title("600 µm window on an epithelium / stroma interface", fontsize=11)
    fig.tight_layout()
    display(save(fig, f"compartment_{sample}"))

# %% [markdown]
# ## 3.2 Immune cells by distance to the border: cells vs spots
#
# Top: the share of each immune type among all the HEDeST cells, by signed distance to the HEDeST
# epithelial border, in 20 µm bins (shaded: inside the epithelium). Bottom: the same question asked
# to the deconvolution, at the spots' resolution and about the same border:
#
# - a spot is **inside the epithelium** when its centre lies in the HEDeST compartment (HEDeST only
#   says which spots are inside);
# - every spot is placed in **rings of spots** from the border, spot to spot (−1: inside, touching
#   a stroma spot; 1: in the stroma, touching an epithelium spot; one ring ≈ 100 µm;
#   `infiltration.spot_ring_profile`);
# - each ring gets the **mean proportion** of each type over its spots (10% filter), as for the
#   plasma cells around the aggregates (`02_structures`).

# %%
RING_TICKS = list(range(-4, 0)) + list(range(1, 5))
RING_LABELS = ["≤ −4", "−3", "−2", "−1", "1", "2", "3", "≥ 4"]
profiles, spot_profiles = [], []
for sample, d in data.items():
    types = [t for t in IMMUNE + ["CAF/fibro"] if t in d.ct]
    profiles.append(I.infiltration_profile(d, compartments[sample], types).assign(sample=sample))
    spot_profiles.append(
        I.spot_ring_profile(d, compartments[sample], types, max_ring=RING_TICKS[-1]).assign(sample=sample)
    )
profiles = pd.concat(profiles)
spot_profiles = pd.concat(spot_profiles)
profiles.to_csv(paths.result_path("infiltration_profiles.csv"), index=False)
spot_profiles.to_csv(paths.result_path("infiltration_profiles_spots.csv"), index=False)

fig, axes = plt.subplots(2, len(SAMPLES), figsize=(4.6 * len(SAMPLES), 8.5))
for column, sample in enumerate(SAMPLES):
    for row, (frame, x, y, label) in enumerate(
        [
            (profiles, "distance_um", "share", "HEDeST cells"),
            (spot_profiles, "ring", "proportion", "deconvolution, spots"),
        ]
    ):
        ax = axes[row, column]
        subset = frame[frame["sample"] == sample]
        for cell_type in IMMUNE + ["CAF/fibro"]:
            curve = subset[subset["cell_type"] == cell_type]
            if len(curve):
                ax.plot(
                    curve[x],
                    curve[y],
                    marker="o",
                    markersize=3,
                    color=style.COLORS[cell_type],
                    label=cell_type,
                    linewidth=2 if cell_type == "T cell" else 1,
                )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_ylim(0, 0.6)
        ax.set_title(f"{SHORT[sample]}: {label}", fontsize=11)
    axes[0, column].axvspan(-100, 0, color="#fde0dd", zorder=0)
    axes[0, column].set_xlim(-100, 300)
    axes[0, column].set_xlabel("signed distance to the epithelial border (µm)")
    axes[0, column].text(-95, 0.58, "inside the\nepithelium", fontsize=9, va="top")
    axes[1, column].axvspan(-4.5, 0, color="#fde0dd", zorder=0)
    axes[1, column].set_xlim(-4.5, 4.5)
    axes[1, column].set_xticks(RING_TICKS)
    axes[1, column].set_xticklabels(RING_LABELS)
    axes[1, column].set_xlabel("rings of spots from the epithelial border\n(1 ring ≈ 100 µm; negative: inside)")
axes[0, 0].set_ylabel("share of all the HEDeST cells\nat that distance")
axes[1, 0].set_ylabel("mean proportion over the spots of the ring\n(deconvolution, 10% filter)")
axes[0, -1].legend(frameon=False, fontsize=9)
fig.tight_layout()
save(fig, "infiltration_profiles")
t_cells = profiles[(profiles["cell_type"] == "T cell")].pivot(index="distance_um", columns="sample", values="share")[
    SAMPLES
]
t_spots = spot_profiles[(spot_profiles["cell_type"] == "T cell")].pivot(
    index="ring", columns="sample", values=["proportion", "spots"]
)
display(t_cells.round(3))
t_spots.round(3)

# %% [markdown]
# ## 3.3 Seeing them on the H&E
#
# On each slide, the two 150 µm windows holding the most T cells deep
# inside the epithelium (< −20 µm), at least 1 mm apart: H&E, the deconvolution, and the HEDeST cells
# (one palette for pies and cells). The compartment border is not drawn at this zoom: the window is
# 150 µm wide and the line covers the cells it is there to place. It is on the overviews of 3.1.

# %%
for sample, d in data.items():
    comp = compartments[sample]
    distance = comp["cells"]
    deep = d.cells.index[(d.cells["label"] == "T cell") & (distance < -20)]
    if len(deep) < 5:
        continue
    xy = d.cells.loc[deep, ["x", "y"]].to_numpy()
    counts = cKDTree(xy).query_ball_point(xy, 75 / d.mpp, return_length=True)
    chosen = []
    for index in np.argsort(-counts):
        if all(np.hypot(*(xy[index] - c)) * d.mpp >= 1000 for c in chosen):
            chosen.append(xy[index])
        if len(chosen) == 2:
            break
    fig, axes = plt.subplots(len(chosen), 3, figsize=(19, 6.2 * len(chosen)), squeeze=False)
    for i, (x, y) in enumerate(chosen):
        window = d.window(x, y, 150)
        in_view = views.in_window(d.cells.loc[deep, ["x", "y"]].to_numpy(), window).sum()
        F.crop_panels(
            d,
            window,
            [
                {"kind": "he"},
                {"kind": "pies"},
                {"kind": "cells", "title": f"HEDeST: {in_view} T cells deeper than 20 µm"},
            ],
            axes=axes[i],
        )
    fig.suptitle(f"{sample}: the places with the most T cells inside the epithelium", fontsize=13, y=1.0)
    fig.tight_layout()
    display(save(fig, f"intraepithelial_t_{sample}"))

# %% [markdown]
# ## 3.4 T cells the deconvolution does not report
#
# Spots deep in the epithelium: their hexagon's cells (closer to that spot than to any other) at a
# median depth below −30 µm, the filtered deconvolution ≥ 80% epithelial, ≥ 15 nuclei. In each, the
# T cells HEDeST finds (and how many lie deeper than 20 µm), against the T proportion of the
# deconvolution, **filtered** (HEDeST's training input: types below 10% set to 0) and **raw**.
#
# - *hidden*: ≥ 5 HEDeST T cells while the filtered deconvolution says 0% T;
# - *hidden even raw*: the same, and the raw deconvolution says ≤ 5% T.

# %%
hexagon_tables, rows = {}, []
for sample, d in data.items():
    table = I.deep_hexagons(d, compartments[sample], raw_props[sample])
    hexagon_tables[sample] = table
    hidden = table[(table["T"] >= 5) & (table["deconv_T"] == 0)]
    rows.append(
        {
            "sample": sample,
            "deep spots": len(table),
            "filtered T = 0": int((table["deconv_T"] == 0).sum()),
            "hidden (≥ 5 HEDeST T, filtered 0%)": len(hidden),
            "... HEDeST T per hexagon (median)": float(hidden["T"].median()) if len(hidden) else np.nan,
            "... HEDeST T share of the nuclei (mean)": float(hidden["hedest_T_share"].mean())
            if len(hidden)
            else np.nan,
            "... raw deconvolution T (mean)": float(hidden["deconv_T_raw"].mean()) if len(hidden) else np.nan,
            "hidden even raw (raw ≤ 5%)": int((hidden["deconv_T_raw"] <= 0.05).sum()),
        }
    )
    table.to_csv(paths.result_path(f"infiltration_deep_spots_{sample}.csv"))
pd.DataFrame(rows).set_index("sample").round(3)

# %% [markdown]
# **On the H&E** (by rule): on each slide, the two hidden spots with the most HEDeST T cells deeper
# than 20 µm in their hexagon, those where the raw deconvolution also says ≤ 5% T first, at least
# 1 mm apart. 200 µm windows: H&E, the filtered deconvolution (HEDeST's input) and the HEDeST
# cells, with the spot drawn as a thick circle. As in 3.3 the compartment border is left out at
# this zoom; it is on the overviews of 3.1.

# %%
for sample, d in data.items():
    table = hexagon_tables[sample]
    hidden = table[(table["T"] >= 5) & (table["deconv_T"] == 0)].copy()
    if not len(hidden):
        continue
    hidden["rank_group"] = (hidden["deconv_T_raw"] > 0.05).astype(int)
    hidden = hidden.sort_values(["rank_group", "T_deep", "T"], ascending=[True, False, False])
    chosen = []
    for spot, row in hidden.iterrows():
        xy = np.array(d.spot_xy(spot))
        if all(np.hypot(*(xy - np.array(d.spot_xy(c)))) * d.mpp >= 1000 for c in chosen):
            chosen.append(spot)
        if len(chosen) == 2:
            break
    fig, axes = plt.subplots(len(chosen), 3, figsize=(19, 6.4 * len(chosen)), squeeze=False)
    for i, spot in enumerate(chosen):
        row = hidden.loc[spot]
        x, y = d.spot_xy(spot)
        window = d.window(x, y, 200)
        F.crop_panels(
            d,
            window,
            [
                {"kind": "he", "title": f"H&E — spot {spot}"},
                {"kind": "pies", "title": f"deconvolution, 10% filter (HEDeST input): T {row['deconv_T']:.0%}"},
                {
                    "kind": "cells",
                    "title": f"HEDeST: {int(row['T'])} T of {int(row['n'])} nuclei in the hexagon ({int(row['T_deep'])} > 20 µm inside)",
                },
            ],
            axes=axes[i],
        )
        for ax in axes[i]:
            views.add_spot_circles(ax, d, window, spots=[spot], color="black", linewidth=2.0)
    fig.suptitle(
        f"{sample}: T cells inside the epithelium that the filtered deconvolution does not report", fontsize=13, y=1.0
    )
    fig.tight_layout()
    display(save(fig, f"hidden_t_{sample}"))
