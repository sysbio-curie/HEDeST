# %% [markdown]
# # 3. Colocalization: proportions vs cells
#
# With a deconvolution, "A and B colocalize" means their proportions go up and down together
# across spots (Pearson correlation), or that B is more abundant in the spots holding A
# (co-occurrence). Both are measured at the scale of a spot (55 µm, ~14 nuclei). HEDeST measures
# colocalization from the cells themselves, at any distance: from contact (10-15 µm) to several
# spots (200 µm).
#
# **Scores.** For each pair (A, B):
#
# - deconvolution: Pearson r of the proportions across spots (also on centred log-ratios);
#   co-occurrence log2(mean(pA·pB) / (mean pA · mean pB));
# - HEDeST: neighbourhood enrichment at radius r, log2(share of B among the cells within r of an A
#   cell / share of B among all cells), averaged over the two directions. 0: B is as common around
#   A as anywhere; > 0: B gathers around A; < 0: B avoids A. Self-pairs (A, A) measure how much a
#   type clusters.
#
# The deconvolution co-occurrence and the HEDeST enrichment are on the same scale, so they can be
# compared directly.
# %%
from __future__ import annotations

{{SETUP}}
from scipy.stats import spearmanr

from lungstudy import coloc as C

PART = "03_colocalization"


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}

# %%
pairs, enrichments, types_of = [], {}, {}
for sample, d in data.items():
    types = C.main_types(d, min_share=0.005)
    types_of[sample] = types
    enrichment = C.cell_enrichment(d, types)
    enrichments[sample] = enrichment
    pairs.append(
        C.pair_table(
            sample,
            types,
            C.spot_pearson(d.props, types),
            C.spot_pearson(d.props, types, transform="clr"),
            C.spot_cooccurrence(d.props, types),
            C.spot_cooccurrence(d.hedest_spot_composition(min_cells=5), types),
            enrichment,
        )
    )
pairs = pd.concat(pairs, ignore_index=True)
pairs.to_csv(paths.result_path("colocalization_pairs.csv"), index=False)
display(pd.Series(types_of, name="types analysed (≥ 0.5% of the deconvolution and of the cells)"))
pairs.head()

# %% [markdown]
# ## 3.1 At the scale of a spot, HEDeST agrees with the deconvolution
#
# Counting the HEDeST cells of each spot and computing the same co-occurrence gives back the
# deconvolution's: HEDeST contains the spot-level analysis.

# %%
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5 * len(SAMPLES), 4.6))
for ax, sample in zip(axes, SAMPLES):
    subset = pairs[pairs["sample"] == sample]
    ax.scatter(
        subset["deconv_cooc"],
        subset["hedest_spot_cooc"],
        c=["black" if s else SAMPLE_COLORS[sample] for s in subset["self"]],
        s=40,
    )
    low, high = (
        min(subset["deconv_cooc"].min(), subset["hedest_spot_cooc"].min()) - 0.3,
        max(subset["deconv_cooc"].max(), subset["hedest_spot_cooc"].max()) + 0.3,
    )
    ax.plot([low, high], [low, high], color="grey", linestyle="--")
    r = np.corrcoef(subset["deconv_cooc"], subset["hedest_spot_cooc"])[0, 1]
    ax.set_title(f"{sample}: r = {r:.2f}", fontsize=11)
    ax.set_xlabel("co-occurrence, deconvolution (log2)")
    ax.set_ylabel("co-occurrence, HEDeST cells per spot (log2)" if ax is axes[0] else "")
fig.suptitle("Spot-level co-occurrence of every pair of types (black: a type with itself)", fontsize=12)
fig.tight_layout()
save(fig, "spot_level_agreement")

# %% [markdown]
# ## 3.2 The matrices
#
# Per slide: the usual Pearson matrix of the deconvolution, the deconvolution co-occurrence, and
# the HEDeST enrichment at contact (15 µm) and at 100 µm.

# %%
for sample in SAMPLES:
    d = data[sample]
    types = types_of[sample]
    matrices = {
        "deconvolution: Pearson across spots": C.spot_pearson(d.props, types),
        "deconvolution: co-occurrence (log2)": C.spot_cooccurrence(d.props, types),
        "HEDeST: enrichment at 15 µm (log2)": C.symmetric(enrichments[sample][15]),
        "HEDeST: enrichment at 100 µm (log2)": C.symmetric(enrichments[sample][100]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(6 * 4, 5.2))
    for ax, (title, matrix) in zip(axes, matrices.items()):
        limit = 1 if "Pearson" in title else 3
        sns.heatmap(
            matrix,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            annot=True,
            fmt=".2f",
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(title, fontsize=11)
    fig.suptitle(sample, fontsize=13)
    fig.tight_layout()
    display(save(fig, f"matrices_{sample}"))

# %% [markdown]
# ## 3.3 From contact to several spots: where the two views part
#
# Left: for each radius, the mean absolute difference between the HEDeST enrichment and the
# deconvolution co-occurrence over all pairs. It is smallest around the spot scale (50-100 µm),
# where HEDeST says what the deconvolution says, and grows towards contact (10-15 µm). Right:
# the Spearman correlation across pairs stays high at every radius: the broad ordering of the
# pairs (epithelium apart from immune cells, B with T, ...) is the same at all scales, and the
# differences are concentrated in specific pairs (section 3.4).

# %%
rows = []
for sample in SAMPLES:
    subset = pairs[pairs["sample"] == sample]
    for radius in C.RADII_UM:
        rows.append(
            {
                "sample": sample,
                "radius_um": radius,
                "mean_abs_gap": float((subset[f"cell_{radius:g}um"] - subset["deconv_cooc"]).abs().mean()),
                "spearman": spearmanr(subset["deconv_cooc"], subset[f"cell_{radius:g}um"])[0],
            }
        )
concordance = pd.DataFrame(rows)
concordance.to_csv(paths.result_path("colocalization_by_radius.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, (column, label) in zip(
    axes,
    [
        ("mean_abs_gap", "mean |HEDeST enrichment − deconvolution co-occurrence|\n(log2, over all pairs)"),
        ("spearman", "Spearman ρ across pairs"),
    ],
):
    for sample in SAMPLES:
        subset = concordance[concordance["sample"] == sample]
        ax.plot(subset["radius_um"], subset[column], marker="o", color=SAMPLE_COLORS[sample], label=sample)
    ax.axvspan(9, paths.SPOT_DIAMETER_UM, color="#eeeeee", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(C.RADII_UM)
    ax.set_xticklabels([f"{r:g}" for r in C.RADII_UM])
    ax.set_xlabel("radius of the HEDeST neighbourhood (µm; grey: below a spot diameter)")
    ax.set_ylabel(label)
axes[0].legend(frameon=False)
axes[1].set_ylim(0.8, 1.0)
fig.tight_layout()
save(fig, "concordance_by_radius")

# %% [markdown]
# ## 3.4 The key comparison: spot-level Pearson vs contact between cells
#
# Every pair of different types on the three slides: x, what a deconvolution study reports
# (Pearson across spots); y, what HEDeST measures at contact (15 µm). Pairs off the diagonal
# trend are the discordances: types that share spots but avoid each other at contact
# (lower-right), or that look unrelated or anti-correlated across spots but touch (upper-left).

# %%
hetero = pairs[~pairs["self"]].copy()
hetero["contact"] = hetero["cell_15um"]
fit = np.polyfit(hetero["deconv_pearson"], hetero["contact"], 1)
hetero["residual"] = hetero["contact"] - np.polyval(fit, hetero["deconv_pearson"])
fig, ax = plt.subplots(figsize=(10, 8))
for sample in SAMPLES:
    subset = hetero[hetero["sample"] == sample]
    ax.scatter(
        subset["deconv_pearson"],
        subset["contact"],
        s=70,
        color=SAMPLE_COLORS[sample],
        label=sample,
        edgecolors="black",
        linewidths=0.4,
    )
x = np.linspace(-1, 1, 10)
ax.plot(x, np.polyval(fit, x), color="grey", linestyle="--", label="trend")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
for _, row in hetero.reindex(hetero["residual"].abs().sort_values(ascending=False).index).head(12).iterrows():
    ax.annotate(
        f"{row['pair']}\n({row['sample'][-4:]})",
        (row["deconv_pearson"], row["contact"]),
        fontsize=8,
        xytext=(5, 5),
        textcoords="offset points",
    )
ax.set_xlabel("deconvolution: Pearson r of the proportions across spots")
ax.set_ylabel("HEDeST: enrichment of contact at 15 µm (log2)")
ax.set_title("Same pairs, two resolutions", fontsize=12)
ax.legend(frameon=False, loc="lower right")
save(fig, "pearson_vs_contact")

# %%
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for ax, (column, label) in zip(
    axes,
    [
        ("deconv_pearson_clr", "deconvolution: Pearson r on centred log-ratios"),
        ("deconv_cooc", "deconvolution: co-occurrence (log2)"),
    ],
):
    for sample in SAMPLES:
        subset = hetero[hetero["sample"] == sample]
        ax.scatter(
            subset[column],
            subset["contact"],
            s=60,
            color=SAMPLE_COLORS[sample],
            label=sample,
            edgecolors="black",
            linewidths=0.4,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    if column == "deconv_cooc":
        low, high = hetero[[column, "contact"]].min().min(), hetero[[column, "contact"]].max().max()
        ax.plot([low, high], [low, high], color="grey", linestyle="--", label="same score")
    ax.set_xlabel(label)
    ax.set_ylabel("HEDeST: enrichment at 15 µm (log2)")
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "clr_and_cooccurrence_vs_contact")

# %% [markdown]
# **Discordant pairs**: in the same slide, co-occurrence across spots and enrichment at contact
# with opposite signs.

# %%
hetero["discordant"] = np.sign(hetero["deconv_cooc"]) != np.sign(hetero["contact"])
hetero["gap"] = hetero["contact"] - hetero["deconv_cooc"]
discordant = hetero[
    hetero["discordant"] & ((hetero["deconv_cooc"].abs() > 0.15) | (hetero["contact"].abs() > 0.15))
].sort_values("gap")
discordant.to_csv(paths.result_path("colocalization_discordant_pairs.csv"), index=False)
discordant[["sample", "pair", "deconv_pearson", "deconv_cooc", "cell_15um", "cell_50um", "cell_100um"]].round(2)

# %% [markdown]
# ## 3.5 Enrichment curves
#
# For the main pairs, the HEDeST enrichment from 10 to 200 µm (solid) and the deconvolution
# co-occurrence (dashed, placed over the spot scale).

# %%
focus = [
    ("B cell", "T cell"),
    ("T cell", "epithelial"),
    ("B cell", "epithelial"),
    ("CAF/fibro", "epithelial"),
    ("CAF/fibro", "T cell"),
    ("macrophage", "epithelial"),
    ("CAF/fibro", "B cell"),
    ("SMC", "CAF/fibro"),
]
fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True)
for ax, (a, b) in zip(axes.ravel(), focus):
    for sample in SAMPLES:
        if a not in types_of[sample] or b not in types_of[sample]:
            continue
        curve = [C.symmetric(enrichments[sample][r]).loc[a, b] for r in C.RADII_UM]
        ax.plot(C.RADII_UM, curve, marker="o", color=SAMPLE_COLORS[sample], label=sample)
        cooc = C.spot_cooccurrence(data[sample].props, types_of[sample]).loc[a, b]
        ax.hlines(cooc, 40, 75, color=SAMPLE_COLORS[sample], linestyle="--", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(0, paths.SPOT_DIAMETER_UM, color="#eeeeee", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200])
    ax.set_xticklabels(["10", "25", "50", "100", "200"])
    ax.set_title(f"{a} – {b}", fontsize=11)
    ax.set_xlabel("radius (µm)")
    ax.set_ylabel("log2 enrichment")
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle(
    "HEDeST enrichment by distance (solid) vs deconvolution co-occurrence (dashed bar at the spot scale)", fontsize=12
)
fig.tight_layout()
save(fig, "enrichment_curves")

# %% [markdown]
# ## 3.6 How clustered is each type, and at what scale?
#
# A type's enrichment with itself is highest at contact and fades with distance. The *decay
# radius* is where it is halfway between its value at 10 µm and at 200 µm: small for types that
# form small clumps or thin layers, large for types filling wide regions. The deconvolution
# gives one number per type (its self co-occurrence across spots), at the spot scale only.

# %%
rows = []
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5.5 * len(SAMPLES), 4.5), sharey=True)
for ax, sample in zip(axes, SAMPLES):
    for cell_type in types_of[sample]:
        curve = pd.Series([enrichments[sample][r].loc[cell_type, cell_type] for r in C.RADII_UM], index=C.RADII_UM)
        ax.plot(curve.index, curve.values, marker="o", color=data[sample].palette.rgb(cell_type), label=cell_type)
        rows.append(
            {
                "sample": sample,
                "cell_type": cell_type,
                "contact_self_enrichment": curve.loc[15],
                "self_enrichment_100um": curve.loc[100],
                "decay_radius_um": C.patch_radius(curve - curve.iloc[-1]),
                "deconv_self_cooc": C.spot_cooccurrence(data[sample].props, [cell_type]).iloc[0, 0],
            }
        )
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200])
    ax.set_xticklabels(["10", "25", "50", "100", "200"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(sample, fontsize=11)
    ax.set_xlabel("radius (µm)")
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("log2 self-enrichment")
fig.tight_layout()
save(fig, "self_enrichment_curves")
patches = pd.DataFrame(rows)
patches.to_csv(paths.result_path("colocalization_self_enrichment.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
melted = patches.melt(
    id_vars=["sample", "cell_type"],
    value_vars=["deconv_self_cooc", "contact_self_enrichment"],
    var_name="score",
    value_name="log2",
)
melted["score"] = melted["score"].map(
    {"deconv_self_cooc": "deconvolution (spot scale)", "contact_self_enrichment": "HEDeST at 15 µm"}
)
sns.barplot(
    data=melted,
    x="cell_type",
    y="log2",
    hue="score",
    palette=[METHOD_COLORS["deconvolution"], METHOD_COLORS["HEDeST"]],
    errorbar=None,
    ax=axes[0],
)
axes[0].set_title("how much a type gathers with itself (mean over slides)", fontsize=11)
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=30)
sns.barplot(data=patches, x="cell_type", y="decay_radius_um", hue="sample", palette=SAMPLE_COLORS, ax=axes[1])
axes[1].axhline(paths.SPOT_DIAMETER_UM, color="black", linestyle="--", linewidth=0.8)
axes[1].text(-0.4, paths.SPOT_DIAMETER_UM + 2, "spot diameter", fontsize=8)
axes[1].set_ylabel("decay radius (µm)")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "self_enrichment_summary")
patches.pivot(index="cell_type", columns="sample", values="decay_radius_um").round(0)

# %% [markdown]
# ## 3.7 What the discordances look like
#
# For each discordant pair, the three spots where the deconvolution puts the most of both types
# (largest min(pA, pB)), chosen from the deconvolution alone: the pies say "mixed", the cells say
# how.

# %%
shown = discordant.drop_duplicates(subset=["pair"]).head(4)
for _, row in shown.iterrows():
    d = data[row["sample"]]
    for rank, spot in enumerate(C.cooccurring_spots(d, row["A"], row["B"], n=3)):
        x, y = d.spot_xy(spot)
        fig = views.compare_panels(
            d,
            d.window(x, y, 220),
            panels=("pies", "cells"),
            panel_size=5.2,
            title=f"{row['sample']}: {row['pair']} — Pearson {row['deconv_pearson']:.2f}, co-occurrence {row['deconv_cooc']:+.2f}, contact {row['contact']:+.2f}",
        )
        display(
            save(
                fig, f"discordant_{row['sample']}_{row['A']}_{row['B']}_{rank + 1}".replace("/", "-").replace(" ", "_")
            )
        )

# %% [markdown]
# ## 3.8 Sensitivity: raw proportions
#
# The same analysis with HEDeST trained on (and the deconvolution taken as) cell2location's raw
# proportions, which keep the rare types.

# %%
rows = []
for sample in SAMPLES:
    d = SampleData(sample, kind="raw")
    types = C.main_types(d, min_share=0.005)
    enrichment = C.cell_enrichment(d, types, radii_um=[15, 100])
    table = C.pair_table(
        sample,
        types,
        C.spot_pearson(d.props, types),
        C.spot_pearson(d.props, types, "clr"),
        C.spot_cooccurrence(d.props, types),
        C.spot_cooccurrence(d.hedest_spot_composition(5), types),
        enrichment,
    )
    rows.append(table)
raw_pairs = pd.concat(rows)
raw_pairs.to_csv(paths.result_path("colocalization_pairs_raw.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, (frame, title) in zip(axes, [(pairs, "10pct_filter (main)"), (raw_pairs, "raw proportions")]):
    subset = frame[~frame["self"]]
    for sample in SAMPLES:
        s = subset[subset["sample"] == sample]
        ax.scatter(
            s["deconv_pearson"],
            s["cell_15um"],
            s=50,
            color=SAMPLE_COLORS[sample],
            label=sample,
            edgecolors="black",
            linewidths=0.4,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("deconvolution: Pearson across spots")
    ax.set_ylabel("HEDeST: enrichment at 15 µm (log2)")
    ax.set_title(title, fontsize=11)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, "sensitivity_raw")
