# %% [markdown]
# # 4. Lymphoid aggregates and tertiary lymphoid structures (TLS)
#
# **With the deconvolution**, a TLS is a group of neighbouring spots rich in B + T cells: its
# outline is made of 100 µm hexagons, its size is a number of spots, and it has no inside.
#
# **With HEDeST**, a lymphoid aggregate is a dense cluster of lymphoid nuclei:
#
# 1. a nucleus is lymphoid when P(B cell) + P(T cell) > 0.5 (lv3, mean of 5 seeds, no PPSA).
#    Summing the two keeps the detection robust to HEDeST hesitating between B and T; NK cells
#    are left out (their lv3 proportions follow the epithelium in these slides);
# 2. DBSCAN on the lymphoid nuclei, positions in µm: eps = 15 µm, min_samples = 10 (at least
#    10 lymphoid nuclei within 15 µm, i.e. dense packing). The plan said 20 µm; on A_AAAA1879,
#    20 µm chained the peritumoral infiltrates into rings around the tumour nests, so it was
#    tightened before any comparison, and the other values are shown as a sensitivity analysis;
# 3. kept with ≥ 50 lymphoid nuclei and ≥ 60% lymphoid nuclei inside the outline;
# 4. outline: concave hull of its nuclei, grown by 5 µm.
#
# Each aggregate is then described from inside: size, B (lineage) vs T balance, whether B and T
# cells are mixed or in separate zones, B-cell follicles, and plasma cells around it (lv4).
#
# **Reference owing nothing to the transcriptomics**: the same DBSCAN on the *small round
# nuclei* of the segmentation (area < 25 µm², circularity > 0.8, the shape of lymphocytes; purity
# ≥ 45%, since about half of the lymphocytes are small and round). HoVer-Net's own inflammatory
# class cannot be used: it calls most nuclei of these slides neoplastic (see notebook 1).
#
# QuPath annotations of every outline are written to `qupath/<sample>_lymphoid_aggregates.geojson`.
# %%
from __future__ import annotations

{{SETUP}}
import json

from lungstudy import tls

PART = "04_tls"


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}
lv4 = {s: SampleData(s, "lv4") for s in SAMPLES}
B_COLOR, T_COLOR, OTHER = (
    data[SAMPLES[0]].palette.rgb("B cell"),
    data[SAMPLES[0]].palette.rgb("T cell"),
    (0.8, 0.8, 0.8),
)
COLORS = {"HEDeST": (0, 200, 0), "spots": (255, 140, 0), "reference": (30, 100, 255), "follicle": (0, 0, 120)}

# %% [markdown]
# ## 4.1 Detection

# %%
aggregates, regions, references = {}, {}, {}
rows = []
for sample, d in data.items():
    aggregates[sample] = tls.detect_cell_aggregates(d, tls.lymphoid_score(d), **tls.DEFAULTS)
    regions[sample] = tls.detect_spot_regions(d, threshold=0.5)
    references[sample] = tls.detect_cell_aggregates(d, tls.small_round_score(d), min_purity=0.45)
    for method, table in (
        ("HEDeST", aggregates[sample]),
        ("spots", regions[sample]),
        ("reference", references[sample]),
    ):
        rows.append(
            {
                "sample": sample,
                "method": method,
                "n": len(table),
                "median area (µm²)": float(table["area_um2"].median()) if len(table) else np.nan,
                "smallest (µm²)": float(table["area_um2"].min()) if len(table) else np.nan,
                "total area (mm²)": float(table["area_um2"].sum() / 1e6) if len(table) else 0.0,
            }
        )
detection = pd.DataFrame(rows)
detection.to_csv(paths.result_path("tls_detection_summary.csv"), index=False)
detection.pivot(index="sample", columns="method").round(1).loc[SAMPLES]

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
sns.barplot(
    data=detection,
    x="sample",
    y="n",
    hue="method",
    hue_order=["spots", "HEDeST", "reference"],
    palette={
        "spots": METHOD_COLORS["spots"],
        "HEDeST": METHOD_COLORS["HEDeST"],
        "reference": METHOD_COLORS["reference"],
    },
    ax=axes[0],
)
axes[0].set_ylabel("number of lymphoid aggregates / regions")
sns.barplot(
    data=detection,
    x="sample",
    y="total area (mm²)",
    hue="method",
    hue_order=["spots", "HEDeST", "reference"],
    palette={
        "spots": METHOD_COLORS["spots"],
        "HEDeST": METHOD_COLORS["HEDeST"],
        "reference": METHOD_COLORS["reference"],
    },
    ax=axes[1],
)
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
axes[1].get_legend().remove()
axes[0].legend(frameon=False, labels=["spot regions (B+T ≥ 50%)", "HEDeST aggregates", "small-round-nuclei reference"])
fig.tight_layout()
save(fig, "detection_counts")

# %% [markdown]
# ## 4.2 On the slides
#
# Green: HEDeST aggregates; orange: spot regions; blue dashed: the small-round-nuclei reference.

# %%
for sample, d in data.items():
    window = views.slide_extent(d)
    fig, axes = plt.subplots(1, 3, figsize=(27, 9.5))
    for ax in axes:
        views.show_tissue(ax, d, window, max_pixels=3000)
        views.add_scalebar(ax, d, 1000)
    for _, row in regions[sample].iterrows():
        tls.draw_polygon(axes[0], row["polygon_um"], d.mpp, np.array(COLORS["spots"]) / 255, 1.2, 0.35)
    axes[0].set_title(f"deconvolution: {len(regions[sample])} spot regions (B + T ≥ 50%)", fontsize=12)
    for _, row in aggregates[sample].iterrows():
        tls.draw_polygon(axes[1], row["polygon_um"], d.mpp, np.array(COLORS["HEDeST"]) / 255, 1.2, 0.35)
    axes[1].set_title(f"HEDeST: {len(aggregates[sample])} lymphoid aggregates", fontsize=12)
    for _, row in regions[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, np.array(COLORS["spots"]) / 255, 1.0)
    for _, row in aggregates[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, np.array(COLORS["HEDeST"]) / 255, 1.0, 0.3)
    for _, row in references[sample].iterrows():
        tls.draw_polygon(axes[2], row["polygon_um"], d.mpp, np.array(COLORS["reference"]) / 255, 1.2, linestyle="--")
    axes[2].set_title("all three (blue dashed: small-round-nuclei reference)", fontsize=12)
    fig.suptitle(f"{sample} — {paths.ROLES[sample]}", fontsize=13, y=1.03)
    fig.tight_layout()
    display(save(fig, f"overview_{sample}"))

# %% [markdown]
# ## 4.3 Size: the spot grid cannot draw anything smaller than a spot
#
# Areas of the aggregates; the dashed lines are the area of one spot (2,376 µm²) and of one cell
# of the spot grid (~8,000 µm², the smallest spot region).

# %%
pooled = []
for sample in SAMPLES:
    for method, table in (
        ("HEDeST", aggregates[sample]),
        ("spots", regions[sample]),
        ("reference", references[sample]),
    ):
        if len(table):
            pooled.append(pd.DataFrame({"sample": sample, "method": method, "area_um2": table["area_um2"].to_numpy()}))
pooled = pd.concat(pooled)
fig, axes = plt.subplots(1, 3, figsize=(20, 4.2), sharey=True)
bins = np.logspace(3, 7, 40)
for ax, sample in zip(axes, SAMPLES):
    for method in ("spots", "HEDeST", "reference"):
        values = pooled[(pooled["sample"] == sample) & (pooled["method"] == method)]["area_um2"]
        ax.hist(values, bins=bins, histtype="step", linewidth=2, color=METHOD_COLORS[method], label=method)
    ax.set_xscale("log")
    for value, label in ((np.pi * 27.5**2, "1 spot"), (np.sqrt(3) / 2 * 100**2, "1 grid cell")):
        ax.axvline(value, color="black", linestyle="--", linewidth=0.8)
        ax.text(value * 1.05, ax.get_ylim()[1] * 0.92, label, fontsize=8)
    ax.set_xlabel("area (µm²)")
    ax.set_title(sample, fontsize=11)
axes[0].set_ylabel("number of aggregates")
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "size_distributions")

# %% [markdown]
# ## 4.4 Aggregates the spots do not see
#
# A HEDeST aggregate is *seen* by the deconvolution when a spot region overlaps at least 20% of
# the smaller of the two. For each aggregate, the share of its nuclei that lie inside a spot (the
# only ones the deconvolution measures).

# %%
rows = []
for sample, d in data.items():
    table = aggregates[sample]
    if not len(table):
        continue
    pairs = tls.overlaps(table, regions[sample])
    seen = set(pairs.loc[pairs["share_of_smaller"] >= 0.2, "left"]) if len(pairs) else set()
    for aggregate_id, row in table.iterrows():
        rows.append(
            {
                "sample": sample,
                "aggregate": aggregate_id,
                "area_um2": row["area_um2"],
                "n_cells": row["n_cells"],
                "share_in_spots": row["share_in_spots"],
                "seen by the spots": aggregate_id in seen,
            }
        )
seen = pd.DataFrame(rows)
seen.to_csv(paths.result_path("tls_seen_by_spots.csv"), index=False)
fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
summary = (
    seen.groupby("sample")["seen by the spots"]
    .agg(["mean", "size"])
    .loc[[s for s in SAMPLES if s in seen["sample"].unique()]]
)
axes[0].bar(summary.index, 1 - summary["mean"], color=METHOD_COLORS["HEDeST"])
for i, (sample, row) in enumerate(summary.iterrows()):
    axes[0].text(
        i,
        1 - row["mean"] + 0.01,
        f"{int(round((1 - row['mean']) * row['size']))}/{int(row['size'])}",
        ha="center",
        fontsize=9,
    )
axes[0].set_ylabel("share of HEDeST aggregates\nmissed by the spot regions")
axes[0].tick_params(axis="x", rotation=15)
sns.boxplot(
    data=seen,
    x="sample",
    y="area_um2",
    hue="seen by the spots",
    palette=["#cccccc", METHOD_COLORS["HEDeST"]],
    showfliers=False,
    ax=axes[1],
)
axes[1].set_yscale("log")
axes[1].set_ylabel("area of the HEDeST aggregate (µm²)")
axes[1].tick_params(axis="x", rotation=15)
sns.histplot(
    data=seen,
    x="share_in_spots",
    hue="sample",
    palette=SAMPLE_COLORS,
    element="step",
    stat="density",
    common_norm=False,
    bins=20,
    ax=axes[2],
)
axes[2].set_xlabel("share of the aggregate's nuclei inside a spot")
axes[2].set_title("the deconvolution measures about a quarter of each aggregate", fontsize=10)
for ax in axes[:2]:
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "missed_by_spots")

# %% [markdown]
# ## 4.5 Agreement with the reference (small round nuclei)
#
# A detection matches a reference aggregate when they overlap by at least 20% of the smaller one.
# Recall: share of the reference aggregates found; precision: share of the detections lying on a
# reference aggregate. And for matched pairs, how well the outlines agree (intersection over union).

# %%
rows, ious = [], []
for sample in SAMPLES:
    for method, table in (("HEDeST", aggregates[sample]), ("spots", regions[sample])):
        scores = tls.detection_scores(table, references[sample])
        rows.append({"sample": sample, "method": method, **scores})
        pairs = tls.overlaps(references[sample], table)
        if len(pairs):
            best = pairs.sort_values("iou").groupby("left").tail(1)
            ious.append(pd.DataFrame({"sample": sample, "method": method, "iou": best["iou"].to_numpy()}))
agreement = pd.DataFrame(rows)
ious = pd.concat(ious)
agreement.to_csv(paths.result_path("tls_agreement_with_reference.csv"), index=False)
fig, axes = plt.subplots(1, 4, figsize=(22, 4.2))
for ax, metric in zip(axes[:3], ["recall", "precision", "f1"]):
    sns.barplot(data=agreement, x="sample", y=metric, hue="method", palette=METHOD_COLORS, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.get_legend().remove()
sns.boxplot(data=ious, x="sample", y="iou", hue="method", palette=METHOD_COLORS, ax=axes[3])
axes[3].set_ylabel("IoU with the matched reference aggregate")
axes[3].set_xlabel("")
axes[3].tick_params(axis="x", rotation=20)
fig.tight_layout()
save(fig, "agreement_with_reference")
agreement.round(3)

# %% [markdown]
# ## 4.6 Sensitivity to the settings
#
# Left: the spot regions depend strongly on the B + T threshold. Right: number of HEDeST
# aggregates and their agreement with the reference over a grid of DBSCAN settings.

# %%
rows = []
for sample, d in data.items():
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        table = tls.detect_spot_regions(d, threshold=threshold)
        rows.append(
            {
                "sample": sample,
                "threshold": threshold,
                "n": len(table),
                "total area (mm²)": table["area_um2"].sum() / 1e6 if len(table) else 0,
                "f1": tls.detection_scores(table, references[sample])["f1"],
            }
        )
spot_grid = pd.DataFrame(rows)
rows = []
for sample, d in data.items():
    score = tls.lymphoid_score(d)
    for eps in [12, 15, 20]:
        for min_samples in [5, 10, 15]:
            table = tls.detect_cell_aggregates(d, score, eps_um=eps, min_samples=min_samples)
            rows.append(
                {
                    "sample": sample,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n": len(table),
                    "total area (mm²)": table["area_um2"].sum() / 1e6 if len(table) else 0,
                    "f1": tls.detection_scores(table, references[sample])["f1"],
                }
            )
cell_grid = pd.DataFrame(rows)
spot_grid.to_csv(paths.result_path("tls_sensitivity_spots.csv"), index=False)
cell_grid.to_csv(paths.result_path("tls_sensitivity_hedest.csv"), index=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 4.3))
for sample in SAMPLES:
    subset = spot_grid[spot_grid["sample"] == sample]
    axes[0].plot(subset["threshold"], subset["total area (mm²)"], marker="o", color=SAMPLE_COLORS[sample], label=sample)
    axes[1].plot(subset["threshold"], subset["n"], marker="o", color=SAMPLE_COLORS[sample], label=sample)
axes[0].set_ylabel("total area of the spot regions (mm²)")
axes[1].set_ylabel("number of spot regions")
for ax in axes[:2]:
    ax.set_xlabel("threshold on the B + T proportion")
axes[0].legend(frameon=False, fontsize=8)
heat = cell_grid[cell_grid["sample"] == "A_AAAA1879"].pivot(index="eps", columns="min_samples", values="n")
sns.heatmap(heat, annot=True, fmt="d", cmap="Greens", ax=axes[2])
axes[2].set_title("A_AAAA1879: number of HEDeST aggregates", fontsize=11)
axes[2].set_ylabel("eps (µm)")
fig.tight_layout()
save(fig, "sensitivity")
cell_grid.pivot_table(index=["eps", "min_samples"], columns="sample", values=["n", "f1"]).round(2)

# %% [markdown]
# ## 4.7 Atlas of the largest aggregates
#
# For each aggregate: the deconvolution (pies) and the HEDeST cells (B lineage blue, T green,
# other nuclei grey), with the HEDeST outline (green) and spot regions (orange).

# %%
def aggregate_panels(d, rows, title, size_um=None):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    axes = np.atleast_2d(axes)
    for i, (aggregate_id, row) in enumerate(rows.iterrows()):
        side = size_um or float(np.clip(np.sqrt(row["area_um2"]) * 1.8, 250, 1500))
        window = d.window(row["x"], row["y"], side)
        contours = side <= 700
        views.show_tissue(axes[i, 0], d, window)
        views.add_pies(axes[i, 0], d, window)
        views.show_tissue(axes[i, 1], d, window, alpha=0.6)
        views.add_cells(
            axes[i, 1], d, window, contours=contours, cell_types=["B cell", "T cell"], other_color=OTHER, alpha=0.8
        )
        views.add_spot_circles(axes[i, 1], d, window, linewidth=0.6)
        for ax in axes[i]:
            tls.draw_polygon(ax, row["polygon_um"], d.mpp, np.array(COLORS["HEDeST"]) / 255, 2.0)
            for _, region in regions[d.name].iterrows():
                if region["polygon_um"].intersects(row["polygon_um"]):
                    tls.draw_polygon(
                        ax, region["polygon_um"], d.mpp, np.array(COLORS["spots"]) / 255, 1.5, linestyle="--"
                    )
            views.add_scalebar(ax, d, 100 if side <= 700 else 250)
        axes[i, 0].set_title(f"{aggregate_id}: deconvolution", fontsize=10)
        axes[i, 1].set_title(
            f"HEDeST: {row['n_cells']} nuclei, {row['area_um2']:,.0f} µm², {row['share_in_spots']:.0%} in spots",
            fontsize=10,
        )
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


for sample in SAMPLES:
    table = aggregates[sample]
    if len(table):
        display(
            save(
                aggregate_panels(data[sample], table.head(4), f"{sample}: the four largest HEDeST aggregates"),
                f"atlas_largest_{sample}",
            )
        )

# %% [markdown]
# **Small aggregates the spot regions miss** (the largest of those not overlapping any spot region).

# %%
for sample in SAMPLES:
    missed = seen[(seen["sample"] == sample) & ~seen["seen by the spots"]].sort_values("area_um2", ascending=False)
    if len(missed):
        rows = aggregates[sample].loc[missed["aggregate"].head(4)]
        display(
            save(
                aggregate_panels(data[sample], rows, f"{sample}: aggregates missed by the spot regions", size_um=300),
                f"atlas_missed_{sample}",
            )
        )

# %% [markdown]
# ## 4.8 Inside the aggregates: mixed or zoned?
#
# **Zoning.** Among the 6 nearest lymphocytes of each lymphocyte of an aggregate, the share of
# B-T pairs, divided by what it would be if B and T were mixed like salt and pepper (2·pB·pT): 1
# for mixed, towards 0 for separate B and T zones.
#
# **Follicles.** Dense clusters of B-lineage cells (DBSCAN, 12 µm, 8 neighbours, ≥ 30 cells,
# ≥ 70% of the lymphocytes inside being B).
#
# A spot only gives two numbers (pB, pT) per 55 µm: neither can be measured from it.

# %%
descriptions, follicle_tables = [], {}
for sample, d in data.items():
    table = aggregates[sample]
    if not len(table):
        continue
    described = tls.describe_aggregates(table, d, lv4=lv4[sample])
    described["BT_mixing"] = tls.zoning(table, d)
    follicle_tables[sample] = tls.follicles(table, d)
    counts = (
        follicle_tables[sample]["aggregate_id"].value_counts() if len(follicle_tables[sample]) else pd.Series(dtype=int)
    )
    described["n_follicles"] = counts.reindex(described.index).fillna(0).astype(int)
    described = described.join(table[["area_um2", "n_cells", "purity", "share_in_spots"]])
    described["agreement"] = [d.cells.loc[ids, "agree"].mean() for ids in table["cell_ids"]]
    descriptions.append(described.assign(sample=sample))
described = pd.concat(descriptions)
described.to_csv(paths.result_path("tls_aggregates_described.csv"))
described.groupby("sample")[["n_B", "n_T", "B_share", "BT_mixing", "n_follicles", "agreement"]].agg(
    ["mean", "median"]
).round(2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
for sample in SAMPLES:
    subset = described[described["sample"] == sample]
    axes[0].scatter(
        subset["area_um2"],
        subset["BT_mixing"],
        s=20 + 10 * subset["n_follicles"],
        color=SAMPLE_COLORS[sample],
        alpha=0.7,
        label=sample,
        edgecolors="black",
        linewidths=0.3,
    )
axes[0].set_xscale("log")
axes[0].axhline(1, color="black", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("area of the aggregate (µm²)")
axes[0].set_ylabel("B-T mixing (1 = salt and pepper)")
axes[0].set_title("larger aggregates are more zoned (marker size: follicles)", fontsize=10)
axes[0].legend(frameon=False, fontsize=8)
sns.boxplot(data=described, x="sample", y="B_share", color=B_COLOR, ax=axes[1], showfliers=False)
sns.stripplot(data=described, x="sample", y="B_share", color="black", size=3, ax=axes[1])
axes[1].set_ylabel("B lineage among B + T")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=15)
counts = described.groupby("sample")["n_follicles"].agg(lambda x: (x > 0).mean()).reindex(SAMPLES)
axes[2].bar(counts.index, counts.values, color=B_COLOR)
axes[2].set_ylabel("share of aggregates holding\nat least one B follicle")
axes[2].tick_params(axis="x", rotation=15)
fig.tight_layout()
save(fig, "zoning_and_follicles")

# %% [markdown]
# **Examples**: the aggregates with the most follicles (follicle outlines in dark blue).

# %%
for sample in ["A_AAAA1638", "A_AAAA1879", "A_AAAA1665"]:
    subset = (
        described[(described["sample"] == sample) & (described["n_follicles"] > 0)]
        .sort_values("n_follicles", ascending=False)
        .head(2)
    )
    d = data[sample]
    for aggregate_id, info in subset.iterrows():
        row = aggregates[sample].loc[aggregate_id]
        side = float(np.clip(np.sqrt(row["area_um2"]) * 1.4, 300, 1600))
        window = d.window(row["x"], row["y"], side)
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        views.show_tissue(axes[0], d, window)
        axes[0].set_title("H&E", fontsize=11)
        views.show_tissue(axes[1], d, window)
        views.add_pies(axes[1], d, window)
        axes[1].set_title("deconvolution", fontsize=11)
        views.show_tissue(axes[2], d, window, alpha=0.6)
        views.add_cells(
            axes[2], d, window, contours=side <= 700, cell_types=["B cell", "T cell"], other_color=OTHER, alpha=0.85
        )
        axes[2].set_title(f"HEDeST: {info['n_follicles']} B follicles, B-T mixing {info['BT_mixing']:.2f}", fontsize=11)
        follicle_rows = follicle_tables[sample][follicle_tables[sample]["aggregate_id"] == aggregate_id]
        for ax in axes:
            tls.draw_polygon(ax, row["polygon_um"], d.mpp, np.array(COLORS["HEDeST"]) / 255, 2.0)
            views.add_scalebar(ax, d, 100 if side <= 700 else 250)
        for _, follicle in follicle_rows.iterrows():
            tls.draw_polygon(
                axes[2], follicle["polygon_um"], d.mpp, np.array(COLORS["follicle"]) / 255, 2.0, linestyle="--"
            )
        fig.suptitle(f"{sample} — {aggregate_id}", fontsize=12, y=1.03)
        fig.tight_layout()
        display(save(fig, f"follicles_{sample}_{aggregate_id}"))

# %% [markdown]
# **The same question from the spots.** For the spot regions large enough to have inner spots
# (all 6 neighbours in the region), the B share of B + T in their edge and inner spots: the only
# "architecture" the deconvolution can offer, at 100 µm steps.

# %%
profiles = pd.concat(
    [tls.spot_region_profiles(regions[s], data[s]).assign(sample=s) for s in SAMPLES if len(regions[s])]
)
if len(profiles):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=profiles, x="sample", y="B_share", hue="where", palette=["#fdd0a2", "#e6550d"], ax=ax)
    ax.set_ylabel("B share of B + T (deconvolution)")
    ax.set_xlabel("")
    ax.set_title(f"{profiles['region'].nunique()} spot regions have inner spots", fontsize=11)
    save(fig, "spot_region_edge_vs_inner")
profiles.groupby(["sample", "where"])["B_share"].describe().round(2) if len(profiles) else None

# %% [markdown]
# ## 4.9 Plasma cells around the aggregates (lv4)
#
# The lv4 HEDeST model splits the B lineage into B cells and plasma cells. Share of plasma cells
# and B cells inside the aggregates, in a 30 µm ring around them, and on the whole slide.

# %%
rows = []
for sample in SAMPLES:
    subset = described[described["sample"] == sample]
    whole = lv4[sample].cells["label"].value_counts(normalize=True)
    for cell_type in ["plasma cell", "B cell"]:
        rows.append(
            {
                "sample": sample,
                "cell_type": cell_type,
                "where": "inside",
                "share": subset[f"lv4_{cell_type}_inside"].mean(),
            }
        )
        rows.append(
            {
                "sample": sample,
                "cell_type": cell_type,
                "where": "30 µm ring",
                "share": subset[f"lv4_{cell_type}_ring"].mean(),
            }
        )
        rows.append(
            {"sample": sample, "cell_type": cell_type, "where": "whole slide", "share": whole.get(cell_type, 0.0)}
        )
plasma = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, cell_type in zip(axes, ["plasma cell", "B cell"]):
    sns.barplot(
        data=plasma[plasma["cell_type"] == cell_type],
        x="sample",
        y="share",
        hue="where",
        palette=["#08519c", "#6baed6", "#cccccc"],
        ax=ax,
    )
    ax.set_title(f"lv4 {cell_type}", fontsize=11)
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "plasma_cells_lv4")
plasma.pivot_table(index=["sample", "where"], columns="cell_type", values="share").round(3)

# %% [markdown]
# ## 4.10 A classification only cells allow
#
# - *follicular*: holds at least one B follicle;
# - *B-rich*: B lineage ≥ 60% of B + T, no follicle;
# - *T-rich*: B lineage < 40%;
# - *mixed*: the rest.

# %%
def category(row):
    if row["n_follicles"] > 0:
        return "follicular"
    if row["B_share"] >= 0.6:
        return "B-rich"
    if row["B_share"] < 0.4:
        return "T-rich"
    return "mixed"


described["category"] = described.apply(category, axis=1)
order = ["follicular", "B-rich", "mixed", "T-rich"]
palette = {"follicular": "#08306b", "B-rich": "#4292c6", "mixed": "#9ecae1", "T-rich": "#74c476"}
counts = described.groupby(["sample", "category"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
areas = (
    described.groupby(["sample", "category"])["area_um2"]
    .sum()
    .unstack(fill_value=0)
    .reindex(columns=order, fill_value=0)
    / 1e6
)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
counts.plot.bar(stacked=True, color=[palette[c] for c in order], ax=axes[0], width=0.7)
areas.plot.bar(stacked=True, color=[palette[c] for c in order], ax=axes[1], width=0.7, legend=False)
axes[0].set_ylabel("aggregates")
axes[1].set_ylabel("area (mm²)")
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "categories")
counts

# %% [markdown]
# ## 4.11 Robustness: another level, the seeds
#
# The same detection with the lv2 HEDeST model (P(immune_lymphoid) > 0.5), trained independently:
# do the aggregates coincide? And the seed agreement of the nuclei inside the aggregates.

# %%
rows = []
for sample in SAMPLES:
    d2 = SampleData(sample, "lv2")
    lv2_table = tls.detect_cell_aggregates(d2, d2.pred["immune_lymphoid"].reindex(d2.cells.index), **tls.DEFAULTS)
    scores = tls.detection_scores(lv2_table, aggregates[sample])
    rows.append(
        {
            "sample": sample,
            "lv3 aggregates": len(aggregates[sample]),
            "lv2 aggregates": len(lv2_table),
            "lv3 found by lv2 (recall)": scores["recall"],
            "lv2 matching lv3 (precision)": scores["precision"],
        }
    )
robust = pd.DataFrame(rows)
robust.to_csv(paths.result_path("tls_robustness_lv2.csv"), index=False)
display(robust.round(3))
fig, ax = plt.subplots(figsize=(7, 4))
inside = described.groupby("sample")["agreement"].mean()
overall = pd.Series({s: data[s].cells["agree"].mean() for s in SAMPLES})
pd.DataFrame({"inside aggregates": inside, "whole slide": overall}).loc[SAMPLES].plot.bar(
    ax=ax, color=["#1f78b4", "#cccccc"], width=0.7
)
ax.set_ylim(0.5, 1)
ax.set_ylabel("mean seed agreement")
ax.tick_params(axis="x", rotation=15)
save(fig, "robustness_seed_agreement")

# %% [markdown]
# ## 4.12 QuPath export
#
# One GeoJSON per slide with the outlines as annotations (classes *HEDeST aggregate*, *HEDeST B
# follicle*, *spot region*, *small-round reference*), in the coordinates of
# `images/hr_tif/<slide>_hr.tiff`, to load with the cells and the spot pies.

# %%
for sample, d in data.items():
    follicle_table = follicle_tables.get(sample, pd.DataFrame())
    if len(follicle_table):
        follicle_table = follicle_table.copy()
        follicle_table.index = [f"{a}_F{i + 1}" for i, a in enumerate(follicle_table["aggregate_id"])]
    payload = tls.to_geojson(
        {
            "HEDeST aggregate": aggregates[sample],
            "HEDeST B follicle": follicle_table,
            "spot region": regions[sample],
            "small-round reference": references[sample],
        },
        d,
        {
            "HEDeST aggregate": COLORS["HEDeST"],
            "HEDeST B follicle": COLORS["follicle"],
            "spot region": COLORS["spots"],
            "small-round reference": COLORS["reference"],
        },
    )
    path = paths.QUPATH_DIR / f"{sample}_lymphoid_aggregates.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    print(f"{path.name}: {len(payload['features'])} annotations")
