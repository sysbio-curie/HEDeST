# %% [markdown]
# # 2. Structures: lymphoid aggregates, their B and T zones, TLS markers, plasma cells
#
# **Why it matters.** Tertiary lymphoid structures (TLS) are lymph-node-like aggregates that form
# in chronically inflamed tissues, tumours included. They form in steps: B and T cells
# **aggregate**, then separate into a **B-cell zone and a T-cell zone**, then the B zone becomes a
# **follicle** with a network of follicular dendritic cells (FDCs; CD21 / CR2, CD23 / FCER2,
# CXCL13), and may grow a **germinal centre** (BCL6, AICDA, Ki-67), fed through **high endothelial
# venules** (HEVs; MECA-79 / PNAd), with mature dendritic cells (DC-LAMP / LAMP3) in the T zone.
# B cells activated there become **plasma cells**, which leave and settle around the structure.
# Mature TLS are associated with longer survival and response to immunotherapy in lung cancer.
#
# **Slides**: A_AAAA1879, A_AAAA1638, A_AAAA1665 (`paths.ANALYSIS_SAMPLES`).
#
# **What a deconvolution can say**: which spots hold many B + T cells (a region made of 100 µm
# hexagons, with no inside), and the B : T balance of each spot. It cannot tell whether B and T
# cells form separate zones inside a spot, nor see a follicle smaller than a spot. cell2location
# could not detect FDCs, HEVs or mature DCs at all: they are not in its single-cell reference.
#
# **What HEDeST adds**: every lymphocyte, so the aggregates' outlines, and their B and T zones.
#
# **Lymphoid aggregates from the cells** (`lungstudy/tls.py`):
#
# 1. a nucleus is lymphoid when its HEDeST label (lv3, mean of 5 seeds, no PPSA) is **B cell or T
#    cell** (lv3's B cell holds the plasma cells: the B lineage);
# 2. **DBSCAN** on the positions of the lymphoid nuclei only: points with at least 10 others within
#    15 µm (dense packing), plus the points within 15 µm of those; isolated points are left out;
# 3. kept with ≥ 50 lymphoid nuclei and ≥ 60% lymphoid nuclei inside the outline;
# 4. outline: concave hull of its nuclei, grown by 5 µm.
#
# **Lymphoid aggregates from the spots** (the deconvolution alone): neighbouring spots (≤ 1.2
# spot pitches apart) whose lv3 proportions of B cell + T cell reach **0.9**, joined into
# regions; outline: the union of their 100 µm hexagons. The same lv3 B and T as the cells (B
# lineage, plasma cells included), and 0.9 because it is the threshold whose regions coincide
# best with the HEDeST aggregates and with the reference below (2.2).
#
# **Reference owing nothing to the transcriptomics**: the same DBSCAN on the *small round nuclei*
# of the segmentation (area < 25 µm², circularity > 0.8, the shape of lymphocytes).
# %%
from __future__ import annotations

{{SETUP}}
import json

import shapely
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from lungstudy import figures as F
from lungstudy import geometry as G
from lungstudy import markers as M
from lungstudy import style
from lungstudy import tls

PART = "02_structures"
SAMPLES = paths.ANALYSIS_SAMPLES


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}
lv4 = {s: SampleData(s, "lv4") for s in SAMPLES}
labels4 = {s: style.merge_t(lv4[s].cells["label"].reindex(data[s].cells.index)) for s in SAMPLES}
props4 = {s: style.merge_t_props(lv4[s].props) for s in SAMPLES}
B_COLOR, T_COLOR, PLASMA_COLOR = (style.COLORS[c] for c in ("B cell", "T cell", "plasma cell"))
OUTLINE = {"HEDeST": "#111111", "spots": "#f28e2b", "reference": "#1f78b4", "follicle": "#08306b"}
QUPATH = {"HEDeST": (0, 200, 0), "spots": (255, 140, 0), "reference": (30, 100, 255), "follicle": (0, 0, 120)}
THRESHOLD = tls.SPOT_THRESHOLD  # lv3 B + T of a spot, motivated in 2.2

# %% [markdown]
# ## 2.1 Detection

# %%
aggregates, references, regions = {}, {}, {}
for sample, d in data.items():
    aggregates[sample] = tls.detect_cell_aggregates(d, tls.lymphoid_cells(d), **tls.DEFAULTS)
    references[sample] = tls.detect_cell_aggregates(d, tls.small_round_score(d), min_purity=0.45)
    regions[sample] = tls.detect_spot_regions(d, threshold=THRESHOLD)
pd.DataFrame(
    {
        s: {
            "HEDeST aggregates": len(aggregates[s]),
            "median area (µm²)": aggregates[s]["area_um2"].median(),
            "total area (mm²)": aggregates[s]["area_um2"].sum() / 1e6,
            "≥ 20,000 µm²": int((aggregates[s]["area_um2"] >= 20000).sum()),
            "deconvolution aggregates": len(regions[s]),
            "... of ≥ 2 spots": int((regions[s]["n_spots"] >= 2).sum()),
            "... total area (mm²)": regions[s]["area_um2"].sum() / 1e6,
            "reference aggregates": len(references[s]),
        }
        for s in SAMPLES
    }
).round(2)

# %% [markdown]
# ## 2.2 Why B + T ≥ 0.9 for the deconvolution aggregates
#
# Drawing aggregates from the spots means picking a threshold on their B + T proportion, and each
# threshold gives other regions. For every threshold from 0.1 to 0.9 (`tls.threshold_sweep`, lv3):
#
# - the **area** of the spot regions over the area of the HEDeST aggregates;
# - the **share of the HEDeST aggregates they touch** (overlap ≥ 20% of the smaller outline);
# - the **IoU** of the areas they cover (intersection over union): how well the two outlines
#   coincide, against the HEDeST aggregates (solid) and against the small-round-nuclei reference,
#   which owes nothing to HEDeST nor to the transcriptomics (dashed).
#
# The threshold kept is the one with the highest IoU: the most favourable setting a
# deconvolution analysis could pick, so that HEDeST is compared with the spots at their best. Left:
# the B + T proportions of the spots themselves.

# %%
sweep = pd.concat([tls.threshold_sweep(data[s], aggregates[s], references[s]) for s in SAMPLES], ignore_index=True)
sweep["area_over_hedest"] = sweep["area_mm2"] / sweep["sample"].map(
    {s: aggregates[s]["area_um2"].sum() / 1e6 for s in SAMPLES}
)
sweep.to_csv(paths.result_path("structures_spot_threshold_sweep.csv"), index=False)

fig, axes = plt.subplots(1, 4, figsize=(24, 4.9))
for sample in SAMPLES:
    color = SAMPLE_COLORS[sample]
    subset = sweep[sweep["sample"] == sample]
    axes[0].hist(
        data[sample].props[tls.LYMPHOID].sum(axis=1),
        bins=np.linspace(0, 1, 51),
        histtype="step",
        color=color,
        linewidth=1.8,
        label=sample,
    )
    axes[1].plot(subset["threshold"], subset["area_over_hedest"], marker="o", color=color)
    axes[2].plot(subset["threshold"], subset["recall"], marker="o", color=color)
    axes[3].plot(subset["threshold"], subset["iou"], marker="o", color=color)
    axes[3].plot(subset["threshold"], subset["iou_reference"], marker="o", ms=3, linestyle="--", color=color)
    best = subset.loc[subset["iou"].idxmax()]
    axes[3].scatter([best["threshold"]], [best["iou"]], s=260, marker="*", color=color, edgecolor="black", zorder=5)
axes[0].axvspan(0.9, 0.98, color="#fdd0a2", zorder=0)
axes[0].set_yscale("log")
axes[0].set_ylim(0.7, 1e5)
axes[0].annotate(
    "no spot between 0.9 and 1:\nthe 10% filter sets every\nother type below 10% to 0",
    xy=(0.94, 3e3),
    xytext=(0.86, 2.5e4),
    ha="right",
    va="center",
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black"),
)
axes[0].set_xlabel("lv3 B + T proportion of the spot")
axes[0].set_ylabel("spots")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
axes[0].set_title("the spots", fontsize=12)
axes[1].set_yscale("log")
axes[1].axhline(1, color="black", linewidth=0.8)
axes[1].set_ylabel("area of the spot regions /\narea of the HEDeST aggregates")
axes[1].set_title("how much area they cover", fontsize=12)
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel("share of the HEDeST aggregates\ntouched by a spot region")
axes[2].set_title("which HEDeST aggregates they touch", fontsize=12)
axes[3].set_ylim(0, None)
axes[3].set_ylabel("IoU of the covered areas")
axes[3].set_title("how well the outlines coincide (star: best)", fontsize=12)
axes[3].plot([], [], color="black", label="vs HEDeST aggregates")
axes[3].plot([], [], color="black", linestyle="--", label="vs small-round reference")
axes[3].legend(frameon=False, fontsize=9, loc="upper left")
for ax in axes[1:]:
    ax.set_xlabel("threshold on the lv3 B + T proportion")
for ax in axes:
    ax.axvline(THRESHOLD, color=OUTLINE["spots"], linewidth=2, zorder=0)
axes[1].text(
    THRESHOLD - 0.01,
    0.97,
    f"kept: {THRESHOLD:g}",
    transform=axes[1].get_xaxis_transform(),
    ha="right",
    va="top",
    fontsize=10,
    color=OUTLINE["spots"],
)
fig.tight_layout()
save(fig, "threshold_sweep")
best = pd.concat(
    {
        "best threshold (IoU vs HEDeST)": sweep.loc[sweep.groupby("sample")["iou"].idxmax()].set_index("sample")[
            "threshold"
        ],
        "best threshold (IoU vs reference)": sweep.loc[sweep.groupby("sample")["iou_reference"].idxmax()].set_index(
            "sample"
        )["threshold"],
    },
    axis=1,
)
at = sweep[sweep["threshold"].isin([0.5, THRESHOLD])].pivot(
    index="sample", columns="threshold", values=["area_over_hedest", "recall", "iou", "iou_reference"]
)
at.columns = [f"{value} at {threshold:g}" for value, threshold in at.columns]
best.join(at).loc[SAMPLES].round(2)

# %% [markdown]
# **Why 0.9.**
#
# - **Closest outlines.** On every slide the IoU rises with the threshold and is highest at 0.9,
#   against HEDeST and against the reference alike. Low thresholds touch every HEDeST aggregate,
#   but only by covering several times their area: spots where lymphocytes are a share of a mixed
#   tissue.
# - **A natural cut.** With the 10%-filtered proportions (HEDeST's training labels), no spot lies
#   between 0.9 and 1 (left panel). B + T ≥ 0.9 therefore keeps the spots made of B and T cells
#   only, and any threshold up to 1 gives the same regions.
# - **The price.** Fewer of the small HEDeST aggregates are touched: an aggregate smaller than a
#   spot rarely fills one.

# %% [markdown]
# ## 2.3 On the slides
#
# Orange: deconvolution aggregates (lv3 B + T ≥ 0.9); black: HEDeST aggregates; blue dashed: the
# small-round-nuclei reference.

# %%
for sample, d in data.items():
    window = views.slide_extent(d)
    fig, axes = plt.subplots(1, 3, figsize=(27, 9.5))
    for ax in axes:
        views.show_tissue(ax, d, window, max_pixels=3000)
        views.add_scalebar(ax, d, 1000)
    for _, row in regions[sample].iterrows():
        tls.draw_polygon(axes[0], row["polygon_um"], d.mpp, OUTLINE["spots"], 1.2, 0.35)
    axes[0].set_title(f"deconvolution: {len(regions[sample])} aggregates (lv3 B + T ≥ {THRESHOLD:g})", fontsize=12)
    views.add_cells(
        axes[1], d, window, labels=d.cells["label"], contours=False, cell_types=["B cell", "T cell"], alpha=0.9
    )
    for _, row in aggregates[sample].iterrows():
        tls.draw_polygon(axes[1], row["polygon_um"], d.mpp, OUTLINE["HEDeST"], 0.8)
    axes[1].set_title(
        f"HEDeST: B (blue) and T (green) cells, {len(aggregates[sample])} aggregates (black)", fontsize=12
    )
    for _, row in regions[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, OUTLINE["spots"], 1.0)
    for _, row in aggregates[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, OUTLINE["HEDeST"], 1.0)
    for _, row in references[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, OUTLINE["reference"], 1.0, linestyle="--")
    axes[2].set_title("all three (blue dashed: small-round-nuclei reference)", fontsize=12)
    fig.suptitle(f"{sample} — {paths.ROLES[sample]}", fontsize=13, y=1.03)
    fig.tight_layout()
    display(save(fig, f"overview_{sample}"))

# %% [markdown]
# ## 2.4 Inside the aggregates: B and T zones
#
# For each aggregate (`tls.compartments`), with the lv3 cells:
#
# - **B–T contact ratio**: among the touching pairs of lymphocytes inside it, the share of B–T
#   pairs, over the share expected if B and T cells were mixed at random (B / T labels shuffled
#   200 times among its lymphocytes; p-value from the same shuffles). 1: mixed; below 1: B cells
#   touch B cells and T cells touch T cells, i.e. separate zones. A spot gives two proportions,
#   never this;
# - **B share at the rim and in the core**: lymphocytes in the outer third of the depth (distance
#   to the outline over the largest depth) and in the inner third;
# - **B follicles** (`tls.follicles`): DBSCAN on the B cells inside the aggregate (12 µm, 8
#   neighbours), kept with ≥ 30 B cells and ≥ 70% B among the lymphocytes inside its outline:
#   a dense B core;
# - **stage**, from the cells alone: *with B follicle*; else *B / T zones* (contact ratio ≤ 0.8,
#   p < 0.01: B and T touch at least 20% less than if mixed); else *mixed*;
# - **strength of the zones**: the spread (SD) of the *local B share* (B lineage among the
#   lymphocytes within 25 µm of each lymphocyte) over the aggregate's lymphocytes;
# - **the same with the lv4 model**, trained independently on another cell-type list (B lineage =
#   B + plasma cells, T = any T subtype): its B–T contact ratio, and the Spearman ρ between the
#   lv3 and lv4 local B share maps (B vs T is HEDeST's least reliable call on single cells,
#   `00_settings` 0.5; the zones can agree while single calls differ).

# %%
described, follicle_tables = [], {}
for sample, d in data.items():
    table = aggregates[sample]
    comp = tls.compartments(table, d)
    follicle_tables[sample] = tls.follicles(table, d)
    counts = (
        follicle_tables[sample]["aggregate_id"].value_counts() if len(follicle_tables[sample]) else pd.Series(dtype=int)
    )
    comp["n_follicles"] = counts.reindex(comp.index).fillna(0).astype(int)
    comp["stage"] = comp.apply(tls.stage, axis=1)
    comp["BT_contact_ratio_lv4"] = tls.compartments(table, d, labels=lv4[sample].cells["label"], n_permutations=100)[
        "BT_contact_ratio"
    ]
    agree = tls.zone_agreement(table, d, {"lv4": lv4[sample].cells["label"]})
    comp = comp.join(agree[["zone_sd", "rho_lv4"]])
    centres = d.spots.loc[d.props.index, ["x_um", "y_um"]].to_numpy()
    comp["spots_inside"] = [
        int(shapely.contains_xy(p, centres[:, 0], centres[:, 1]).sum()) for p in table["polygon_um"].reindex(comp.index)
    ]
    comp = comp.join(table[["area_um2", "n_cells", "purity"]])
    described.append(comp.assign(sample=sample))
described = pd.concat(described)
described.to_csv(paths.result_path("structures_aggregates.csv"))
follicles_all = pd.concat([t.assign(sample=s) for s, t in follicle_tables.items() if len(t)], ignore_index=True)
follicles_all.drop(columns=["polygon_um"]).to_csv(paths.result_path("structures_follicles.csv"), index=False)
stage_counts = (
    described.groupby(["sample", "stage"]).size().unstack(fill_value=0).reindex(columns=tls.STAGES, fill_value=0)
)
stage_counts["follicles"] = follicles_all.groupby("sample").size().reindex(stage_counts.index).fillna(0).astype(int)
stage_counts

# %% [markdown]
# ## 2.5 A B follicle zone beside a T zone, on the H&E
#
# **Which aggregate** (by rule), one per slide: among the aggregates holding a HEDeST B follicle and
# at least 15 spot centres (enough spots to read their genes), the one with the strongest zones
# (largest spread of the local B share). H&E, the deconvolution, the HEDeST lv3 B and T cells
# (follicles dashed), and the local T share of each lymphocyte (red: the T zone, blue: the B
# follicle; the T share is one minus the B share, the lymphocytes being B or T).

# %%
def local_t_share(d, aggregate_row, radius_um=25.0, labels=None):
    """Each lymphocyte's share of T cells among the lymphocytes within ``radius_um``.

    ``lineage_codes`` labels a lymphocyte 1 (B lineage) or 0 (T), and only lymphocytes are kept,
    so this is exactly one minus the local B share: the T zone is where it is high.
    """

    labels = d.cells["label"] if labels is None else labels
    codes = tls.lineage_codes(labels.reindex(d.cells.index))
    position = pd.Series(np.arange(len(d.cells)), index=d.cells.index)
    members = position.loc[aggregate_row["cell_ids"]].to_numpy()
    lymph = members[codes[members] >= 0]
    xy = d.cells[["x_um", "y_um"]].to_numpy()[lymph]
    neighbours = cKDTree(xy).query_ball_point(xy, radius_um)
    is_t = (codes[lymph] == 0).astype(float)
    return lymph, np.array([is_t[n].mean() for n in neighbours])


def zoned_figure(sample, aggregate_id):
    d = data[sample]
    row = aggregates[sample].loc[aggregate_id]
    side = float(np.clip(np.sqrt(row["area_um2"]) * 1.45, 300, 1200))
    window = d.window(row["x"], row["y"], side)
    outlines = [(row["polygon_um"], OUTLINE["HEDeST"], "-", 1.6)]
    table = follicle_tables[sample]
    if len(table):
        outlines += [
            (p, OUTLINE["follicle"], "--", 1.6) for p in table.loc[table["aggregate_id"] == aggregate_id, "polygon_um"]
        ]
    panels = [
        {"kind": "he"},
        {"kind": "pies", "props": d.props, "title": "deconvolution lv3"},
        {"kind": "cells", "highlight": ["B cell", "T cell"], "title": "HEDeST lv3: B and T cells (others grey)"},
    ]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6.2 * (len(panels) + 1), 6.4))
    F.crop_panels(d, window, panels, outlines=outlines, axes=axes[:-1])
    ax = axes[-1]
    views.show_tissue(ax, d, window, alpha=0.45)
    lymph, share = local_t_share(d, row)
    points = d.cells[["x", "y"]].to_numpy()[lymph]
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=share,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=views.nucleus_marker_size(ax, d, window),
        linewidths=0,
    )
    for polygon, color, linestyle, linewidth in outlines:
        tls.draw_polygon(ax, polygon, d.mpp, color, linewidth, linestyle=linestyle)
    fig.colorbar(scatter, ax=ax, fraction=0.04, label="T share within 25 µm (red: T zone, blue: B follicle)")
    ax.set_title("local T share of the lymphocytes", fontsize=11)
    views.add_scalebar(ax, d, 100)
    info = described.loc[aggregate_id]
    fig.suptitle(
        f"{sample} {aggregate_id.split('_')[-1]}: {row['area_um2']:,.0f} µm², {info['n_lymphocytes']:,} lymphocytes, B {info['B_share']:.0%}, "
        f"B–T contacts ×{info['BT_contact_ratio']:.2f} of random (lv4 ×{info['BT_contact_ratio_lv4']:.2f}), "
        f"B rim {info['B_share_rim']:.0%} → core {info['B_share_core']:.0%}, {info['n_follicles']} follicle(s)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return save(fig, f"zoned_{aggregate_id}")


ZONED = []
for sample in SAMPLES:
    part = described[
        (described["sample"] == sample) & (described["n_follicles"] >= 1) & (described["spots_inside"] >= 15)
    ]
    ZONED.append((sample, part["zone_sd"].idxmax()))
for sample, aggregate_id in ZONED:
    display(zoned_figure(sample, aggregate_id))
described.loc[
    [a for _, a in ZONED],
    [
        "sample",
        "area_um2",
        "n_lymphocytes",
        "spots_inside",
        "B_share",
        "BT_contact_ratio",
        "BT_contact_ratio_lv4",
        "zone_sd",
        "rho_lv4",
        "B_share_rim",
        "B_share_core",
        "n_follicles",
        "stage",
    ],
].round(2)

# %% [markdown]
# ## 2.6 The genes of the zones (spots)
#
# The genes of the TLS compartments (`lungstudy/markers.py`), in counts per 10,000 (CP10k) of each
# spot: on the three aggregates above, each spot coloured by its CP10k of MS4A1 (B cells), CR2
# (FDCs), CXCL13 and CCL19 (T zone / stromal scaffold), with the HEDeST outline and follicles.
# cell2location could not see FDCs, HEVs or mature DCs (not in its reference): their genes are an
# outside check of what HEDeST draws.

# %%
expression = {s: M.cp10k(cache.markers(s)).reindex(data[s].props.index) for s in SAMPLES}


def marker_spots(sample, aggregate_id, genes=("MS4A1", "CR2", "CXCL13", "CCL19")):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    d = data[sample]
    row = aggregates[sample].loc[aggregate_id]
    side = float(np.clip(np.sqrt(row["area_um2"]) * 1.6, 350, 1300))
    window = d.window(row["x"], row["y"], side)
    table = follicle_tables[sample]
    follicle_polygons = list(table.loc[table["aggregate_id"] == aggregate_id, "polygon_um"]) if len(table) else []
    fig, axes = plt.subplots(1, len(genes), figsize=(5.6 * len(genes), 5.6))
    spots = d.props.index
    xy = d.spots.loc[spots, ["x", "y"]].to_numpy()
    keep = views.in_window(xy, window, d.radius_px)
    for ax, gene in zip(axes, genes):
        views.show_tissue(ax, d, window, alpha=0.6)
        values = expression[sample].loc[spots[keep], gene].to_numpy()
        vmax = max(np.nanquantile(expression[sample][gene], 0.99), 1e-3)
        collection = PatchCollection(
            [Circle((x, y), d.radius_px) for x, y in xy[keep]], cmap="viridis", edgecolor="white", linewidth=0.4
        )
        collection.set_array(np.nan_to_num(values))
        collection.set_clim(0, vmax)
        ax.add_collection(collection)
        tls.draw_polygon(ax, row["polygon_um"], d.mpp, "white", 1.8)
        for polygon in follicle_polygons:
            tls.draw_polygon(ax, polygon, d.mpp, "#ff4fa3", 1.8, linestyle="--")
        fig.colorbar(collection, ax=ax, fraction=0.04, label=f"{gene} (CP10k)")
        ax.set_title(f"{gene} — {M.COMPARTMENT_OF[gene]}", fontsize=11)
        views.add_scalebar(ax, d, 100)
    fig.suptitle(
        f"{sample} {aggregate_id.split('_')[-1]}: marker genes per spot (white: HEDeST aggregate, pink dashed: HEDeST B follicles)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig


for sample, aggregate_id in ZONED:
    display(save(marker_spots(sample, aggregate_id), f"markers_{aggregate_id}"))

# %% [markdown]
# **Do the genes follow the zones, and would the spots alone have found them?** On the same three
# aggregates, the spots whose centre lies inside, split three ways:
#
# - **B / T halves from the cells**: the B share of B + T among the HEDeST lymphocytes inside the
#   disc, above or below its median over these spots;
# - **B / T halves from the deconvolution**: the same with the spot's B and T proportions;
# - **on a HEDeST follicle**: the disc touches a follicle, against the other spots of the aggregate.
#
# For each gene, log2 of the mean CP10k on the B side over the T side, with 0.05 CP10k added to
# both (filled: Mann–Whitney p < 0.05). A split with fewer than 3 spots on one side is not shown.

# %%
ZONE_GENES = ["MS4A1", "CR2", "FCER2", "CXCL13", "AICDA", "CD3E", "CCL19", "CCL21", "LAMP3"]
SPLITS = {
    "B / T halves, HEDeST cells in the spot": ("o", METHOD_COLORS["HEDeST"]),
    "B / T halves, deconvolution": ("s", METHOD_COLORS["deconvolution"]),
    "on a HEDeST follicle vs the rest": ("D", "#ff4fa3"),
}


def spots_by_zone(sample, aggregate_id):
    d = data[sample]
    row = aggregates[sample].loc[aggregate_id]
    centres = d.spots.loc[d.props.index, ["x_um", "y_um"]].to_numpy()
    spots = d.props.index[shapely.contains_xy(row["polygon_um"], centres[:, 0], centres[:, 1])]
    cells = d.cells[d.cells["in_spot"] & d.cells["spot"].isin(spots)]
    codes = pd.Series(tls.lineage_codes(cells["label"]), index=cells.index)
    lymph = codes >= 0
    discs = shapely.buffer(shapely.points(d.spots.loc[spots, "x_um"], d.spots.loc[spots, "y_um"]), d.radius_px * d.mpp)
    table = follicle_tables[sample]
    on_follicle = np.zeros(len(spots), dtype=bool)
    for polygon in table.loc[table["aggregate_id"] == aggregate_id, "polygon_um"]:
        on_follicle |= shapely.intersects(discs, polygon)
    hedest_b = codes[lymph].groupby(cells.loc[lymph, "spot"]).mean().reindex(spots)
    deconv_b = (d.props["B cell"] / (d.props["B cell"] + d.props["T cell"])).reindex(spots)
    sides = pd.DataFrame(
        {
            "B / T halves, HEDeST cells in the spot": hedest_b >= hedest_b.median(),
            "B / T halves, deconvolution": deconv_b >= deconv_b.median(),
            "on a HEDeST follicle vs the rest": on_follicle,
        },
        index=spots,
    )
    rows = []
    for split in SPLITS:
        b_side, t_side = spots[sides[split].to_numpy()], spots[~sides[split].to_numpy()]
        for gene in ZONE_GENES:
            a, b = expression[sample].loc[b_side, gene], expression[sample].loc[t_side, gene]
            rows.append(
                {
                    "aggregate": aggregate_id,
                    "split": split,
                    "gene": gene,
                    "spots B side": len(a),
                    "spots T side": len(b),
                    "B side": a.mean(),
                    "T side": b.mean(),
                    "log2 ratio": np.log2((a.mean() + 0.05) / (b.mean() + 0.05)),
                    "p (Mann-Whitney)": mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan,
                }
            )
    rho = np.corrcoef(hedest_b.rank(), deconv_b.rank())[0, 1]
    return pd.DataFrame(rows), rho


zone_genes, zone_rho = [], {}
for sample, aggregate_id in ZONED:
    table, zone_rho[aggregate_id] = spots_by_zone(sample, aggregate_id)
    zone_genes.append(table)
zone_genes = pd.concat(zone_genes, ignore_index=True)
zone_genes.to_csv(paths.result_path("structures_markers_by_zone.csv"), index=False)

fig, axes = plt.subplots(1, len(ZONED), figsize=(8.5 * len(ZONED), 6.4), sharey=True)
for ax, (sample, aggregate_id) in zip(np.atleast_1d(axes), ZONED):
    part = zone_genes[zone_genes["aggregate"] == aggregate_id]
    ax.axhspan(4.5, len(ZONE_GENES) - 0.5, color="#f0f0f0", zorder=0)
    for offset, (split, (marker, color)) in zip((-0.22, 0, 0.22), SPLITS.items()):
        rows = part[part["split"] == split].set_index("gene").loc[ZONE_GENES]
        y = np.arange(len(ZONE_GENES)) + offset
        significant = rows["p (Mann-Whitney)"].to_numpy() < 0.05
        counts = f" ({rows['spots B side'].iloc[0]} vs {rows['spots T side'].iloc[0]} spots)"
        if min(rows["spots B side"].iloc[0], rows["spots T side"].iloc[0]) < 3:  # too few spots on one side to compare
            ax.scatter(
                [],
                [],
                marker=marker,
                s=80,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                label=split + counts + ": not shown",
            )
            continue
        ax.scatter(
            rows["log2 ratio"][significant],
            y[significant],
            marker=marker,
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            label=split + counts,
        )
        ax.scatter(
            rows["log2 ratio"][~significant],
            y[~significant],
            marker=marker,
            s=80,
            facecolors="white",
            edgecolors=color,
            linewidths=1.5,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(ZONE_GENES)))
    ax.set_yticklabels([f"{g} ({M.COMPARTMENT_OF[g]})" for g in ZONE_GENES])
    ax.set_xlabel("log2 (mean CP10k on the B side / on the T side)")
    ax.set_title(
        f"{sample} {aggregate_id.split('_')[-1]}: {part['spots B side'].iloc[0] + part['spots T side'].iloc[0]} spots inside\ncells vs deconvolution B share ρ = {zone_rho[aggregate_id]:.2f}",
        fontsize=11,
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        title="filled: p < 0.05; grey band: T-zone genes",
        title_fontsize=9,
    )
np.atleast_1d(axes)[0].invert_yaxis()  # shared y: once
fig.tight_layout()
save(fig, "markers_by_zone")
zone_genes.pivot_table(index=["aggregate", "gene"], columns="split", values="log2 ratio").round(2)

# %% [markdown]
# ## 2.7 Plasma cells around the aggregates
#
# **HEDeST (lv4 cells).** The aggregates are those of 2.1, drawn from the lv3 cells (B or T label;
# lv3's B cell is the B lineage, plasma cells included). The plasma cells come from the lv4 model,
# which splits the B lineage into B cells and plasma cells (T CD4 and CD8 merged into T cells:
# HEDeST does not separate them, `00_settings`). Every cell gets its signed distance to the border
# of the nearest HEDeST aggregate (negative inside), on a 5 µm raster. **Density** = cells of a type per mm² of tissue at that distance; **share** = its share
# of all the cells at that distance (every type counts in the denominator). Plasma cells are the
# output of a working TLS: activated B cells become antibody-secreting plasma cells, which leave
# the structure.

# %%
EDGES = [-60, -40, -20, 0, 10, 20, 30, 40, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500]
profiles = []
for sample in SAMPLES:
    d = data[sample]
    grid = G.tissue_grid(d, bin_um=5.0)
    distance = G.signed_distance(G.rasterize(aggregates[sample]["polygon_um"], grid), grid.bin_um)
    labels = labels4[sample]
    profile = G.density_profile(grid, distance, labels, ["plasma cell", "B cell", "T cell"], EDGES)
    profile["slide_share"] = profile["cell_type"].map(labels.value_counts(normalize=True))
    profiles.append(profile.assign(sample=sample))
profiles = pd.concat(profiles, ignore_index=True)
profiles.to_csv(paths.result_path("structures_plasma_profiles.csv"), index=False)

n = len(SAMPLES)
fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.5), sharex=True, squeeze=False)
for column, sample in enumerate(SAMPLES):
    subset = profiles[profiles["sample"] == sample]
    for cell_type, color in (("plasma cell", PLASMA_COLOR), ("B cell", B_COLOR), ("T cell", T_COLOR)):
        curve = subset[subset["cell_type"] == cell_type]
        axes[0, column].plot(
            curve["distance_um"], curve["density_per_mm2"], marker="o", ms=3, color=color, label=cell_type
        )
    plasma = subset[subset["cell_type"] == "plasma cell"]
    axes[1, column].plot(
        plasma["distance_um"], plasma["share"], marker="o", ms=4, color=PLASMA_COLOR, label="plasma cells"
    )
    axes[1, column].axhline(
        plasma["slide_share"].iloc[0], color=PLASMA_COLOR, linestyle="--", linewidth=1, label="slide average"
    )
    axes[0, column].set_title(sample, fontsize=12)
    for ax in axes[:, column]:
        ax.axvspan(-60, 0, color="#e5f5e0", zorder=0)
        ax.axvline(0, color="black", linewidth=1)
    axes[1, column].set_xlabel("signed distance to the nearest\nHEDeST aggregate (µm)")
axes[0, 0].set_ylabel("cells per mm² of tissue")
axes[1, 0].set_ylabel("share of all cells")
axes[0, -1].legend(frameon=False, fontsize=9)
axes[1, -1].legend(frameon=False, fontsize=9)
fig.tight_layout()
save(fig, "plasma_profiles")
profiles[profiles["cell_type"] == "plasma cell"].pivot(index="distance_um", columns="sample", values="share").round(3)

# %% [markdown]
# **The deconvolution alone, around the aggregates HEDeST found.** The same question asked to the
# spots, at their own resolution, restricted to the HEDeST aggregates:
#
# - **aggregate spots**: the spots whose centre lies inside a HEDeST aggregate (2.1). HEDeST only
#   says which spots are in an aggregate; an aggregate holding no spot centre (smaller than the
#   grid) is left out;
# - **distance** of every other spot to the nearest aggregate spot, spot to spot, in **rings of
#   spots** (1: touching one, 2: the next ring, ...; one ring ≈ 100 µm, centre to centre;
#   `geometry.spot_rings`);
# - **plasma cells**: each spot's lv4 plasma-cell proportion, from the deconvolution.
#
# Every spot is a dot (jittered within its ring); the solid line is the mean of the ring and the
# dashed one the mean over all spots of the slide, both in the plasma-cell colour of the study, as
# in the profiles above. The dots on 0 are spots whose plasma proportion the 10% filter set to 0.
# Aggregates drawn from the spots themselves (lv3 B + T ≥ 0.9) would not do here: lv3 counts plasma
# cells as B cells, so they also take in plasma-cell-rich spots.

# %%
RINGS = list(range(7))
RING_NAMES = ["in an\naggregate", "1", "2", "3", "4", "5", "≥ 6"]
ring_rows, ring_spots, held = [], {}, {}
for sample in SAMPLES:
    spots = props4[sample].index
    xy = lv4[sample].spots.loc[spots, ["x_um", "y_um"]].to_numpy()
    polygons = list(aggregates[sample]["polygon_um"])
    union = shapely.unary_union(polygons)
    inside = pd.Series(shapely.contains_xy(union, xy[:, 0], xy[:, 1]), index=spots)
    held[sample] = sum(bool(shapely.contains_xy(polygon, xy[:, 0], xy[:, 1]).any()) for polygon in polygons)
    table = pd.DataFrame(
        {
            "ring": G.spot_rings(lv4[sample], inside, max_ring=RINGS[-1]),
            "plasma": props4[sample]["plasma cell"],
            "outline_um": shapely.distance(shapely.points(xy[:, 0], xy[:, 1]), union),
        }
    )
    ring_spots[sample] = table
    for ring in RINGS:
        values = table.loc[table["ring"] == ring, "plasma"]
        ring_rows.append(
            {
                "sample": sample,
                "ring": ring,
                "spots": len(values),
                "plasma_mean": values.mean() if len(values) else np.nan,
                "slide_mean": table["plasma"].mean(),
                "outline_um_median": table.loc[values.index, "outline_um"].median(),
            }
        )
rings = pd.DataFrame(ring_rows)
rings["over_slide"] = rings["plasma_mean"] / rings["slide_mean"]
rings.to_csv(paths.result_path("structures_plasma_ring_deconvolution.csv"), index=False)

rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), squeeze=False)
for ax, sample in zip(axes[0], SAMPLES):
    table = ring_spots[sample]
    ax.scatter(
        table["ring"] + rng.uniform(-0.3, 0.3, len(table)),
        table["plasma"],
        s=6,
        color=PLASMA_COLOR,
        alpha=0.25,
        linewidths=0,
    )
    part = rings[rings["sample"] == sample]
    ax.plot(
        part["ring"],
        part["plasma_mean"],
        color=PLASMA_COLOR,
        marker="o",
        ms=6,
        linewidth=2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="mean of the ring",
    )
    ax.axhline(part["slide_mean"].iloc[0], color=PLASMA_COLOR, linestyle="--", linewidth=1, label="mean of the slide")
    ax.set_xticks(RINGS)
    ax.set_xticklabels(RING_NAMES)
    ax.set_xlabel("rings of spots from the nearest spot inside a HEDeST aggregate (1 ring ≈ 100 µm)")
    ax.set_ylim(-0.02, min(1, table["plasma"].quantile(0.995) + 0.05))
    ax.set_title(
        f"{sample}: {part.loc[part['ring'] == 0, 'spots'].iloc[0]} spots inside {held[sample]} of {len(aggregates[sample])} HEDeST aggregates",
        fontsize=11,
    )
axes[0, 0].set_ylabel("plasma-cell proportion of the spot (lv4 deconvolution;\n0: below the 10% filter)")
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
save(fig, "plasma_ring_deconvolution")
rings.pivot(index="ring", columns="sample", values=["plasma_mean", "over_slide", "spots", "outline_um_median"])[
    ["plasma_mean", "over_slide", "spots", "outline_um_median"]
].round(3)

# %% [markdown]
# `outline_um_median`: the median distance from the spot centres of a ring to the nearest HEDeST
# outline, to read the rings against the cell profile above (ring 1 ≈ 40-50 µm, ring 2 ≈ 110 µm).

# %% [markdown]
# ## 2.8 QuPath export
#
# One GeoJSON per slide with the outlines as annotations (classes *HEDeST aggregate*, *HEDeST B
# follicle*, *deconvolution aggregate*, *small-round reference*), in the coordinates of
# `images/hr_tif/<slide>_hr.tiff`, to load with the cells and the spot pies.

# %%
for sample, d in data.items():
    follicle_table = follicle_tables[sample]
    if len(follicle_table):
        follicle_table = follicle_table.copy()
        follicle_table.index = [f"{a}_F{i + 1}" for i, a in enumerate(follicle_table["aggregate_id"])]
    payload = tls.to_geojson(
        {
            "HEDeST aggregate": aggregates[sample],
            "HEDeST B follicle": follicle_table,
            "deconvolution aggregate": regions[sample],
            "small-round reference": references[sample],
        },
        d,
        {
            "HEDeST aggregate": QUPATH["HEDeST"],
            "HEDeST B follicle": QUPATH["follicle"],
            "deconvolution aggregate": QUPATH["spots"],
            "small-round reference": QUPATH["reference"],
        },
    )
    path = paths.QUPATH_DIR / f"{sample}_lymphoid_aggregates.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    print(f"{path.name}: {len(payload['features'])} annotations")
