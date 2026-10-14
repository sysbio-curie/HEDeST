# %% [markdown]
# # 1. Can HEDeST be trusted without a ground truth?
#
# Four checks, none of which needs a per-cell truth, on the three slides
# (lv3, 10pct_filter, HEDeST without PPSA, mean of 5 seeds):
#
# 1. **Spots never seen in training.** Each seed predicts its 10% test spots from their cells'
#    morphology alone. The deconvolution-only way to guess an unmeasured place is to interpolate
#    the neighbouring spots. Which is closer to the measured deconvolution, and do the two carry
#    the same information?
# 2. **HoVer-Net's nucleus types**, an independent image classifier, against HEDeST and against
#    a deconvolution-only labelling of the cells.
# 3. **Consistency across levels**: the lv1 to lv4 HEDeST models are trained independently.
# 4. **Nucleus shape** per type: measured on every cell by HEDeST, against what the proportions
#    alone can infer.
# %%
from __future__ import annotations

{{SETUP}}
from lungstudy import validation as V

PART = "01_validation"


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}

# %% [markdown]
# ## 1.1 Held-out spots: HEDeST vs interpolation of the neighbours
#
# For every seed, the 10% test spots are rebuilt exactly as HEDeST drew them. HEDeST's guess is
# the mean prediction of the cells inside the spot; the interpolation is the inverse-distance
# mean of the 6 nearest *training* spots' deconvolution. Both are compared with the
# deconvolution of the held-out spot (total-variation distance: 0 identical, 1 disjoint).

# %%
per_type, per_spot, blends, dispersion = [], [], [], []
for sample, d in data.items():
    metrics = V.heldout_metrics(d)
    per_type.append(metrics["per_type"].assign(sample=sample))
    per_spot.append(metrics["per_spot"].assign(sample=sample))
    blends.append(V.blend_curve(d).assign(sample=sample))
    dispersion.append(V.neighbour_dispersion(d).assign(sample=sample))
per_type = pd.concat(per_type)
per_spot = pd.concat(per_spot).merge(pd.concat(dispersion), on=["sample", "seed", "spot"])
blends = pd.concat(blends)
per_type.to_csv(paths.result_path("validation_heldout_per_type.csv"), index=False)
per_spot.to_csv(paths.result_path("validation_heldout_per_spot.csv"), index=False)

wide = per_spot.pivot_table(index=["sample", "seed", "spot"], columns="method", values="tv")
summary = pd.DataFrame(
    {
        "test spots (x5 seeds)": wide.groupby("sample").size(),
        "TV HEDeST": wide.groupby("sample")["HEDeST"].mean(),
        "TV interpolation": wide.groupby("sample")["interpolation"].mean(),
        "TV 50/50 blend": blends[blends["w_hedest"] == 0.5].groupby("sample")["tv"].mean(),
        "HEDeST closer (share of spots)": (wide["HEDeST"] < wide["interpolation"]).groupby("sample").mean(),
    }
).loc[SAMPLES]
summary.to_csv(paths.result_path("validation_heldout_summary.csv"))
summary.round(3)

# %%
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5 * len(SAMPLES), 4.2), sharey=True)
for ax, sample in zip(axes, SAMPLES):
    d = data[sample]
    shown = [c for c in d.ct if d.props[c].mean() > 0.005]
    subset = per_type[(per_type["sample"] == sample) & per_type["cell_type"].isin(shown)]
    sns.barplot(
        data=subset,
        x="cell_type",
        y="pearson",
        hue="method",
        order=shown,
        hue_order=["HEDeST", "interpolation"],
        palette=METHOD_COLORS,
        errorbar="sd",
        ax=ax,
    )
    ax.set_title(f"{sample}", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    ax.set_ylim(0, 1)
    if ax is not axes[-1]:
        ax.get_legend().remove()
axes[0].set_ylabel("Pearson r with the deconvolution\non held-out spots (5 seeds)")
fig.suptitle(
    "Composition of held-out spots: HEDeST (morphology of the spot's cells) vs interpolation of the neighbouring spots",
    fontsize=12,
)
fig.tight_layout()
save(fig, "heldout_pearson_per_type")

# %%
long = per_spot.copy()
blend_rows = []
for sample, d in data.items():
    hold = V.heldout_predictions(d)
    for seed, group in hold.groupby("seed"):
        truth = group[group["method"] == "deconvolution"].set_index("spot")[d.ct]
        mix = (
            0.5 * group[group["method"] == "HEDeST"].set_index("spot").loc[truth.index, d.ct]
            + 0.5 * group[group["method"] == "interpolation"].set_index("spot").loc[truth.index, d.ct]
        )
        blend_rows.append(
            pd.DataFrame(
                {
                    "sample": sample,
                    "seed": seed,
                    "spot": truth.index,
                    "method": "blend",
                    "tv": 0.5 * (mix - truth).abs().sum(axis=1).to_numpy(),
                }
            )
        )
long = pd.concat([long[["sample", "seed", "spot", "method", "tv"]], *blend_rows])
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(
    data=long,
    x="sample",
    y="tv",
    hue="method",
    hue_order=["interpolation", "HEDeST", "blend"],
    palette=METHOD_COLORS,
    showfliers=False,
    ax=ax,
)
ax.set_ylabel("distance to the deconvolution\n(total variation, held-out spots)")
ax.set_xlabel("")
ax.legend(
    title="", labels=["interpolation of the neighbours", "HEDeST (cells of the spot)", "50/50 blend"], frameon=False
)
ax.set_title("Error on spots never seen in training", fontsize=12)
save(fig, "heldout_error_boxplot")

# %% [markdown]
# **Do HEDeST and the interpolation carry the same information?** If they did, mixing them would
# not help. The error of `w × HEDeST + (1 − w) × interpolation` is lowest for a mix, on every
# slide: the morphology of the cells adds what the neighbours' proportions do not hold.

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for sample in SAMPLES:
    curve = blends[blends["sample"] == sample].groupby("w_hedest")["tv"].agg(["mean", "std"])
    ax.errorbar(
        curve.index, curve["mean"], yerr=curve["std"], marker="o", capsize=3, color=SAMPLE_COLORS[sample], label=sample
    )
ax.set_xlabel("weight of HEDeST in the guess (0 = interpolation only, 1 = HEDeST only)")
ax.set_ylabel("mean distance to the deconvolution\n(held-out spots)")
ax.legend(frameon=False)
ax.set_title("HEDeST and interpolation are complementary", fontsize=12)
save(fig, "heldout_blend_curve")

# %% [markdown]
# **Where does each win?** The held-out spots are split by how heterogeneous their neighbourhood
# is, measured on the neighbours alone (so that the split favours neither method).

# %%
per_spot["neighbourhood"] = per_spot.groupby("sample")["dispersion"].transform(
    lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1\nhomogeneous", "Q2", "Q3", "Q4\nheterogeneous"])
)
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5 * len(SAMPLES), 4), sharey=False)
for ax, sample in zip(axes, SAMPLES):
    subset = per_spot[per_spot["sample"] == sample]
    sns.pointplot(
        data=subset,
        x="neighbourhood",
        y="tv",
        hue="method",
        hue_order=["interpolation", "HEDeST"],
        palette=METHOD_COLORS,
        errorbar=("ci", 95),
        dodge=0.2,
        ax=ax,
    )
    ax.set_title(sample, fontsize=11)
    ax.set_xlabel("heterogeneity of the neighbouring spots")
    ax.set_ylabel("distance to the deconvolution" if ax is axes[0] else "")
    if ax is not axes[-1]:
        ax.get_legend().remove()
fig.tight_layout()
save(fig, "heldout_by_heterogeneity")

# %% [markdown]
# **Examples.** On seed 0, the held-out spots where interpolation misses most and HEDeST does not,
# and (for honesty) the reverse. Left: the HEDeST cells; middle: the neighbours' deconvolution
# with the held-out spot left empty; right: the three compositions of the held-out spot.

# %%
def heldout_example(d, spot, seed=0, size_um=300):
    hold = V.heldout_predictions(d)
    hold = hold[(hold["seed"] == seed) & (hold["spot"] == spot)].set_index("method")[d.ct]
    x, y = d.spot_xy(spot)
    window = d.window(x, y, size_um)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    views.show_tissue(axes[0], d, window)
    views.add_cells(axes[0], d, window, alpha=0.6)
    views.add_spot_circles(axes[0], d, window, color="black", linewidth=0.8)
    views.add_spot_circles(axes[0], d, window, color="red", linewidth=2.5, spots=[spot])
    axes[0].set_title("HEDeST cells (held-out spot in red)", fontsize=10)
    views.show_tissue(axes[1], d, window)
    views.add_pies(axes[1], d, window, props=d.props.drop(index=spot))
    views.add_spot_circles(axes[1], d, window, color="red", linewidth=2.5, spots=[spot])
    axes[1].set_title("deconvolution of the neighbours", fontsize=10)
    for ax in axes[:2]:
        views.add_scalebar(ax, d, 50)
    shown = [c for c in d.ct if hold[c].max() > 0.02]
    frame = hold.loc[["deconvolution", "interpolation", "HEDeST"], shown]
    left = np.zeros(len(frame))
    for cell_type in shown:
        axes[2].barh(range(len(frame)), frame[cell_type], left=left, color=d.palette.rgb(cell_type), label=cell_type)
        left += frame[cell_type].to_numpy()
    tv = {m: 0.5 * np.abs(frame.loc[m] - frame.loc["deconvolution"]).sum() for m in ("interpolation", "HEDeST")}
    axes[2].set_yticks(range(len(frame)))
    axes[2].set_yticklabels(
        [
            "measured\n(deconvolution)",
            f"interpolation\n(error {tv['interpolation']:.2f})",
            f"HEDeST\n(error {tv['HEDeST']:.2f})",
        ]
    )
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
    axes[2].set_title(f"{d.name} — spot {spot}", fontsize=10)
    fig.tight_layout()
    return fig


for sample in ["A_AAAA1879", "A_AAAA1638"]:
    d = data[sample]
    subset = wide.loc[sample].loc[0]
    gain = (subset["interpolation"] - subset["HEDeST"]).sort_values()
    counts = d.spots["n_cells"]
    gain = gain[counts.reindex(gain.index) >= 8]
    for rank, spot in enumerate(gain.index[::-1][:3]):
        display(save(heldout_example(d, spot), f"heldout_example_{sample}_hedest_better_{rank + 1}"))
    for rank, spot in enumerate(gain.index[:2]):
        display(save(heldout_example(d, spot), f"heldout_example_{sample}_interpolation_better_{rank + 1}"))

# %% [markdown]
# ## 1.2 HoVer-Net's nucleus types
#
# HoVer-Net (PanNuke, fast mode) classified every nucleus at segmentation time, from its
# appearance only. **On these slides its classes are badly calibrated**: it calls most nuclei
# *neoplastic*, lymphocytes included, so it cannot serve as a ground truth. Its calls still carry
# a relative signal: which HoVer-Net class is over-represented in each HEDeST type.

# %%
hv_order = ["neoplastic", "non-neoplastic epithelial", "inflammatory", "connective", "necrosis", "no label"]
hv_colors = {
    "neoplastic": "#d62728",
    "non-neoplastic epithelial": "#ff9896",
    "inflammatory": "#2ca02c",
    "connective": "#1f77b4",
    "necrosis": "#bcbd22",
    "no label": "#7f7f7f",
}
overall = (
    pd.DataFrame({s: data[s].cells["hv_name"].value_counts(normalize=True) for s in SAMPLES})
    .reindex(hv_order)
    .fillna(0)
)
fig, ax = plt.subplots(figsize=(8, 3.8))
overall.T.plot.barh(stacked=True, color=[hv_colors[n] for n in hv_order], ax=ax, width=0.7)
ax.set_xlabel("share of the nuclei")
ax.legend(title="HoVer-Net class", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.set_title("HoVer-Net calls most nuclei 'neoplastic' on these slides", fontsize=11)
save(fig, "hovernet_overall")

# %% [markdown]
# **Enrichment of each HoVer-Net class in each cell type**, log2(share of the class among the
# cells of the type / share among all cells), for HEDeST and for a deconvolution-only labelling
# (every cell takes the most abundant type of its spot, or of the nearest spot when it lies
# between spots). Expected: lymphocytes enriched in *inflammatory*, CAF/fibro and SMC in
# *connective*, epithelial in *neoplastic*.

# %%
def deconvolution_labels(d):
    dominant = d.props.idxmax(axis=1)
    spot_of = d.cells["spot"].where(d.cells["in_spot"], d.cells["nearest_spot"])
    return spot_of.map(dominant)


def enrichment(labels, hv, types):
    table = pd.crosstab(labels, hv, normalize="index").reindex(index=types, columns=hv_order[:4]).fillna(0)
    base = hv.value_counts(normalize=True).reindex(hv_order[:4])
    return np.log2((table + 1e-3) / base)


rows = []
fig, axes = plt.subplots(2, len(SAMPLES), figsize=(5.5 * len(SAMPLES), 9))
for column, sample in enumerate(SAMPLES):
    d = data[sample]
    types = [c for c in d.ct if (d.cells["label"] == c).mean() > 0.005]
    for row, (method, labels) in enumerate(
        (("HEDeST", d.cells["label"]), ("deconvolution-only", deconvolution_labels(d)))
    ):
        matrix = enrichment(labels, d.cells["hv_name"], types)
        sns.heatmap(
            matrix,
            cmap="RdBu_r",
            center=0,
            vmin=-2,
            vmax=2,
            annot=True,
            fmt=".1f",
            ax=axes[row, column],
            cbar=column == len(SAMPLES) - 1,
            cbar_kws={"label": "log2 enrichment"},
        )
        axes[row, column].set_title(f"{sample} — {method}", fontsize=10)
        axes[row, column].set_ylabel("")
        axes[row, column].set_xlabel("")
    from scipy.stats import chi2_contingency

    for method, labels in (("HEDeST", d.cells["label"]), ("deconvolution-only", deconvolution_labels(d))):
        table = pd.crosstab(labels, d.cells["hv_name"])
        chi2 = chi2_contingency(table)[0]
        n = table.to_numpy().sum()
        rows.append(
            {"sample": sample, "method": method, "cramers_v": float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))}
        )
fig.tight_layout()
save(fig, "hovernet_enrichment")

# %%
association = pd.DataFrame(rows)
association.to_csv(paths.result_path("validation_hovernet_association.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 3.8))
sns.barplot(
    data=association,
    x="sample",
    y="cramers_v",
    hue="method",
    palette={"HEDeST": METHOD_COLORS["HEDeST"], "deconvolution-only": METHOD_COLORS["deconvolution"]},
    ax=ax,
)
ax.set_ylabel("Cramér's V with HoVer-Net's class")
ax.set_xlabel("")
ax.legend(frameon=False, title="cell labels")
ax.set_title("Association of the cell labels with an independent image classifier", fontsize=11)
save(fig, "hovernet_cramers_v")

# %% [markdown]
# Grouping into three compartments (HoVer-Net: neoplastic + non-neoplastic epithelial →
# epithelial, inflammatory → immune, connective → stromal; HEDeST lv1: epithelial, immune,
# stromal + endothelial → stromal), Cohen's kappa against HoVer-Net:

# %%
scores = []
for sample in SAMPLES:
    table = V.compartment_labels(sample)
    scores.append(V.agreement_scores(table).assign(sample=sample))
scores = pd.concat(scores)
scores.to_csv(paths.result_path("validation_hovernet_compartments.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
for ax, where in zip(axes, ["in spots", "between spots"]):
    subset = scores[scores["where"] == where]
    sns.barplot(data=subset, x="sample", y="kappa", hue="method", palette=METHOD_COLORS, ax=ax)
    ax.set_title(f"cells {where}", fontsize=11)
    ax.set_ylabel("Cohen's kappa with HoVer-Net\n(3 compartments)")
    ax.set_xlabel("")
fig.tight_layout()
save(fig, "hovernet_kappa")
scores.pivot_table(
    index=["sample", "where"], columns="method", values=["kappa", "recall_immune", "recall_stromal"]
).round(3)

# %% [markdown]
# ## 1.3 Consistency across annotation levels
#
# The parent of each fine type follows the names of the annotation levels (lv4 T CD4 / T CD8 →
# lv3 T cell, lv4 plasma cell → lv3 B cell, lv3 B / T / NK → lv2 immune_lymphoid, ...). The HEDeST
# models of two levels never saw each other: a cell called B cell at lv3 should still be lymphoid
# at lv2.
#
# **First, the input.** How well does the deconvolution itself nest? Correlation across spots
# between each lv4 type and its lv3 parent, and the best-correlated lv3 type:

# %%
ct3 = list(cache.proportions(SAMPLES[0], "lv3").columns)
ct4 = list(cache.proportions(SAMPLES[0], "lv4").columns)
ct2 = list(cache.proportions(SAMPLES[0], "lv2").columns)
mapping_43 = V.hierarchy("lv4", "lv3", ct4)
mapping_32 = V.hierarchy("lv3", "lv2", ct3)
mapping_21 = V.hierarchy("lv2", "lv1", ct2)
rows = []
for sample in SAMPLES:
    fine, coarse = cache.proportions(sample, "lv4"), cache.proportions(sample, "lv3")
    shared = fine.index.intersection(coarse.index)
    for cell_type in ["B cell", "plasma cell", "T CD4", "T CD8", "CAF/fibro", "epithelial", "macrophage", "SMC"]:
        x = fine.loc[shared, cell_type]
        if x.std() == 0:
            continue
        corr = coarse.loc[shared].apply(lambda column: np.corrcoef(x, column)[0, 1] if column.std() > 0 else np.nan)
        rows.append(
            {
                "sample": sample,
                "lv4 type": cell_type,
                "lv3 parent": mapping_43[cell_type],
                "r with parent": corr[mapping_43[cell_type]],
                "best lv3 type": corr.idxmax(),
                "r with best": corr.max(),
            }
        )
nesting = pd.DataFrame(rows)
nesting.to_csv(paths.result_path("validation_deconvolution_nesting.csv"), index=False)
nesting.pivot(index="lv4 type", columns="sample", values="r with parent").round(2)

# %% [markdown]
# The deconvolution's lv4 *B cell* follows lv3 *T cell* as much as lv3 *B cell*: in these slides,
# lv3 *B cell* is mostly the B lineage seen through plasma cells, and the B / T split is the least
# stable part of the input. HEDeST inherits this.

# %%
consistency = []
for sample in SAMPLES:
    for fine, coarse, mapping in (("lv3", "lv2", mapping_32), ("lv2", "lv1", mapping_21), ("lv4", "lv3", mapping_43)):
        table = V.level_consistency(sample, mapping, fine, coarse).reset_index()
        consistency.append(table.assign(sample=sample, step=f"{fine} → {coarse}"))
consistency = pd.concat(consistency)
consistency.to_csv(paths.result_path("validation_level_consistency.csv"), index=False)
overall = consistency.groupby(["sample", "step"]).apply(lambda g: np.average(g["consistent"], weights=g["n"])).unstack()
display(overall.round(3))

fig, axes = plt.subplots(1, 3, figsize=(21, 4.5))
for ax, step in zip(axes, ["lv4 → lv3", "lv3 → lv2", "lv2 → lv1"]):
    subset = consistency[(consistency["step"] == step) & (consistency["n"] >= 200)]
    sns.barplot(data=subset, x="fine", y="consistent", hue="sample", palette=SAMPLE_COLORS, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("share of the cells in the parent type\nat the coarser level")
    ax.tick_params(axis="x", rotation=40)
    ax.set_title(f"HEDeST {step} (independent models)", fontsize=11)
    if ax is not axes[-1]:
        ax.get_legend().remove()
fig.tight_layout()
save(fig, "level_consistency")

# %%
labels = cache.labels("A_AAAA1638")
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, (fine, coarse) in zip(axes, [("lv3", "lv2"), ("lv4", "lv3")]):
    table = pd.crosstab(labels[f"label_{fine}"], labels[f"label_{coarse}"], normalize="index")
    table = table.loc[labels[f"label_{fine}"].value_counts().index[labels[f"label_{fine}"].value_counts() > 200]]
    sns.heatmap(
        table, cmap="magma_r", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "share of the row"}
    )
    ax.set_xlabel(f"HEDeST {coarse}")
    ax.set_ylabel(f"HEDeST {fine}")
    ax.set_title(f"A_AAAA1638: {fine} vs {coarse}", fontsize=11)
fig.tight_layout()
save(fig, "level_crosstab_A_AAAA1638")

# %% [markdown]
# ## 1.4 Nucleus size per type: measured (HEDeST) vs inferred from the proportions
#
# HEDeST gives the full distribution of nucleus areas of each type. From the deconvolution alone,
# the only way is a non-negative regression of each spot's mean nucleus area on its proportions:
# one number per type, with wide uncertainty (bootstrap 95% interval), and none for the cells
# between spots.

# %%
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(5.5 * len(SAMPLES), 4.5), sharey=True)
regressions = []
for ax, sample in zip(axes, SAMPLES):
    d = data[sample]
    estimate = V.area_from_proportions(d)
    regressions.append(estimate.assign(sample=sample))
    types = [c for c in estimate.index if (d.cells["label"] == c).sum() >= 100]
    subset = d.cells[d.cells["label"].isin(types)]
    sns.boxplot(
        data=subset,
        x="label",
        y="area_um2",
        order=types,
        palette={c: d.palette.rgb(c) for c in types},
        showfliers=False,
        ax=ax,
        width=0.6,
    )
    positions = np.arange(len(types))
    est = estimate.loc[types]
    ax.errorbar(
        positions + 0.38,
        est["deconvolution_estimate"],
        yerr=[est["deconvolution_estimate"] - est["low"], est["high"] - est["deconvolution_estimate"]],
        fmt="D",
        color="black",
        capsize=3,
        label="inferred from the proportions",
    )
    ax.set_title(sample, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("nucleus area (µm²)" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=40)
    ax.set_ylim(0, 70)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle(
    "Nucleus area per type: boxes = HEDeST cells (measured), diamonds = regression of spot means on proportions",
    fontsize=12,
)
fig.tight_layout()
save(fig, "nucleus_area_measured_vs_inferred")
pd.concat(regressions).round(1)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
pooled = pd.concat([data[s].cells.assign(sample=s) for s in SAMPLES])
types = ["B cell", "T cell", "macrophage", "epithelial", "CAF/fibro", "SMC"]
palette = {c: data["A_AAAA1879"].palette.rgb(c) for c in types}
subset = pooled[pooled["label"].isin(types)]
sns.boxplot(data=subset, x="label", y="eccentricity", order=types, palette=palette, showfliers=False, ax=axes[0])
sns.boxplot(data=subset, x="label", y="circularity", order=types, palette=palette, showfliers=False, ax=axes[1])
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
axes[0].set_title("elongation (eccentricity): highest for CAF/fibro and SMC", fontsize=11)
axes[1].set_title("roundness (circularity): highest for T cells", fontsize=11)
fig.tight_layout()
save(fig, "nucleus_shape_all_samples")

# %% [markdown]
# ## 1.5 Confidence: cells in spots vs cells between spots
#
# The cells between spots are never seen by the training loss. They are predicted with the same
# confidence and the same seed agreement as the cells inside spots.

# %%
rows = []
for sample in SAMPLES:
    cells = data[sample].cells
    for where, selection in (("in spots", cells["in_spot"]), ("between spots", ~cells["in_spot"])):
        rows.append(
            {
                "sample": sample,
                "where": where,
                "mean probability": cells.loc[selection, "conf"].mean(),
                "seed agreement": cells.loc[selection, "agree"].mean(),
                "unanimous seeds": (cells.loc[selection, "agree"] == 1).mean(),
            }
        )
confidence = pd.DataFrame(rows)
confidence.to_csv(paths.result_path("validation_confidence_in_vs_between.csv"), index=False)
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, metric in zip(axes, ["mean probability", "seed agreement", "unanimous seeds"]):
    sns.barplot(data=confidence, x="sample", y=metric, hue="where", palette=["#555555", "#bbbbbb"], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    if ax is not axes[-1]:
        ax.get_legend().remove()
fig.tight_layout()
save(fig, "confidence_in_vs_between")
confidence.round(3)

# %% [markdown]
# ## 1.6 Sensitivity: HEDeST trained on the raw proportions
#
# The same slides, HEDeST trained on cell2location's raw proportions instead of the 10% filter:
# how many cells keep their type?

# %%
rows = []
for sample in SAMPLES:
    labels = cache.labels(sample)
    same = labels["label_lv3"] == labels["label_lv3_raw"]
    rows.append({"sample": sample, "same type (all cells)": same.mean()})
    for cell_type, share in same.groupby(labels["label_lv3"]).mean().items():
        if (labels["label_lv3"] == cell_type).mean() > 0.005:
            rows.append({"sample": sample, "cell_type": cell_type, "same type": share})
raw = pd.DataFrame(rows)
display(raw.dropna(subset=["same type (all cells)"])[["sample", "same type (all cells)"]].round(3))
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(
    data=raw.dropna(subset=["cell_type"]), x="cell_type", y="same type", hue="sample", palette=SAMPLE_COLORS, ax=ax
)
ax.set_ylim(0, 1)
ax.set_xlabel("HEDeST type (10pct_filter)")
ax.set_ylabel("share keeping the same type\nwhen trained on raw proportions")
ax.legend(frameon=False, fontsize=8)
save(fig, "sensitivity_raw_proportions")
