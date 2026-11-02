# %% [markdown]
# # 0. Settings: which level, which proportions, PPSA or not
#
# Every LungScape slide has 20 HEDeST runs: 5 annotation levels (lv1 4 types, lv2 7, lv3 12,
# lv4 14, lv5 21) × 2 proportion files (raw, or 10pct_filter: types below 10% of a spot set to
# 0) × with or without PPSA (the predictions rescaled by each cell's local spot composition).
# lv1 and lv2 are too coarse for immune cells (lv2 has one "immune_lymphoid" class). This
# notebook compares the rest on the six slides and fixes **one rule per level, the same on every
# slide**.
#
# | question | rule | why (sections below) |
# |---|---|---|
# | level | **lv3** for every analysis; **lv4** for plasma cells; **lv4** and **lv5** to check that B / T zones do not depend on one model | lv3 is the finest level whose types HEDeST does not collapse; lv4 adds plasma cells, which HEDeST does separate; the T subtypes of lv4 / lv5 mostly collapse to one per slide (0.4) |
# | proportions | **10pct_filter** at lv2-lv4, **raw** at lv5 | the filter keeps the main types of lv3 / lv4; at lv5 it erases most of the 21 types (0.2) |
# | PPSA | **off everywhere** | it changes 3-11% of the labels, does not restore the collapsed subtypes, and pulls the cells towards what the spots say, never towards what they miss (0.3) |
#
# The raw proportions are also used (i) as a sensitivity check (HEDeST trained on them), and
# (ii) as the deconvolution's unfiltered answer whenever we say the deconvolution misses cells.
# %%
from __future__ import annotations

{{SETUP}}
import itertools
import pickle

from lungstudy import style
from lungstudy import tls

PART = "00_settings"


def save(fig, name):
    return paths.save(fig, PART, name)


UNITS = [(lv, kind) for lv in ("lv3", "lv4", "lv5") for kind in ("10pct_filter", "raw")]


def info(sample, level, kind):
    """The 5 seeds combined (mean predictions with and without PPSA) of one LungScape run."""

    with open(paths.sample(sample).run_dir(level, kind) / "info_aggregated.pickle", "rb") as handle:
        return pickle.load(handle)


def raw_proportions(sample, level):
    table = pd.read_csv(paths.sample(sample).proportions(level, "raw"), index_col=0)
    return table.div(table.sum(axis=1), axis=0)


# %% [markdown]
# ## 0.1 Fit to the deconvolution and seed agreement, per level and proportions
#
# From LungScape's summary (`hedest_summary.csv`): the Pearson correlation between the mean HEDeST
# prediction of each held-out spot and its proportions (spots never seen in training), and the seed
# agreement. All units fit well (r ≥ 0.8 on held-out spots for lv3 / lv4).

# %%
summary = pd.read_csv(paths.DATA_DIR / "hedest_summary.csv")
summary = summary[
    summary["sample"].isin(SAMPLES)
    & summary["level"].isin(["lv2", "lv3", "lv4", "lv5"])
    & (summary["run_name"] == "default")
]
table = summary.pivot_table(index=["level", "proportions"], columns="sample", values="pearson_held_out")[SAMPLES]
table.columns = [SHORT[s] for s in table.columns]
display(table.round(2))
agreement = summary.pivot_table(index=["level", "proportions"], columns="sample", values="mean_seed_agreement")[SAMPLES]
agreement.columns = [SHORT[s] for s in agreement.columns]
agreement.round(2)

# %% [markdown]
# ## 0.2 What the 10% filter removes, per level
#
# The number of types whose mean proportion on the slide is at least 0.5%, in the raw and in the
# filtered proportions. At lv3 / lv4 the filter keeps the main types (and drops the rare ones: DC,
# NK, mast cells, a few % at most). At lv5 it removes most types: subtypes are split into small
# proportions that rarely reach 10% of a spot.

# %%
rows = []
for sample in SAMPLES:
    for level in ("lv3", "lv4", "lv5"):
        raw = raw_proportions(sample, level)
        filtered = pd.read_csv(paths.sample(sample).proportions(level, "10pct_filter"), index_col=0)
        filtered = filtered.div(filtered.sum(axis=1), axis=0)
        rows.append(
            {
                "sample": SHORT[sample],
                "level": level,
                "types": raw.shape[1],
                "≥ 0.5% (raw)": int((raw.mean() >= 0.005).sum()),
                "≥ 0.5% (10pct_filter)": int((filtered.mean() >= 0.005).sum()),
            }
        )
kept = pd.DataFrame(rows)
kept.to_csv(paths.result_path("settings_filter_types.csv"), index=False)
kept.pivot(index="sample", columns="level", values=["≥ 0.5% (raw)", "≥ 0.5% (10pct_filter)"])

# %% [markdown]
# ## 0.3 PPSA
#
# PPSA multiplies each cell's probabilities by its local spot composition over the slide's
# (local: its spot, or the ≤ 3 nearest spots within 110 µm for cells between spots), then
# renormalises. Two things to check: how many labels it changes, and whether it brings back the
# subtypes HEDeST collapses (section 0.4). And one consequence of its design: where the filtered
# proportions of a spot say 0% of a type, PPSA sets that type's probability to 0 for every cell
# inside the spot (between spots, the prior is interpolated from up to 3 spots, so it is only
# lowered): it pulls the cells towards what the spots say, never towards what they miss.

# %%
rows, composition = [], []
for sample in SAMPLES:
    for level, kind in UNITS:
        payload = info(sample, level, kind)
        before = payload["predictions"].idxmax(axis=1)
        after = payload["predictions_adjusted"].idxmax(axis=1)
        rows.append(
            {
                "sample": SHORT[sample],
                "unit": f"{level}/{kind}",
                "labels changed by PPSA": float((before != after).mean()),
            }
        )
        raw = raw_proportions(sample, level)
        for adjusted, labels in (("without PPSA", before), ("with PPSA", after)):
            share = labels.value_counts(normalize=True)
            for cell_type in payload["predictions"].columns:
                composition.append(
                    {
                        "sample": SHORT[sample],
                        "level": level,
                        "kind": kind,
                        "ppsa": adjusted,
                        "cell_type": cell_type,
                        "HEDeST share of cells": float(share.get(cell_type, 0.0)),
                        "deconvolution (raw, mean)": float(raw[cell_type].mean()),
                    }
                )
changed = pd.DataFrame(rows)
composition = pd.DataFrame(composition)
changed.to_csv(paths.result_path("settings_ppsa_changes.csv"), index=False)
composition.to_csv(paths.result_path("settings_composition.csv"), index=False)
changed.pivot(index="sample", columns="unit", values="labels changed by PPSA").round(3)

# %% [markdown]
# **What PPSA does to the T cells the filtered spots do not see.** lv3, 10pct_filter: the cells
# HEDeST calls T cells (without PPSA) inside a spot whose filtered T proportion is 0, and between
# spots in the hexagon of such a spot (closer to it than to any other spot); how many remain T
# cells with PPSA.

# %%
rows = []
for sample in SAMPLES:
    d = SampleData(sample)
    payload = info(sample, "lv3", "10pct_filter")
    after = payload["predictions_adjusted"].idxmax(axis=1).reindex(d.cells.index)
    hexagon = d.pitch_px * d.mpp / np.sqrt(3)
    zero_t = d.props.index[d.props["T cell"] == 0]
    is_t = d.cells["label"] == "T cell"
    inside = d.cells["in_spot"] & d.cells["spot"].isin(zero_t) & is_t
    between = (
        ~d.cells["in_spot"] & d.cells["nearest_spot"].isin(zero_t) & (d.cells["nearest_spot_dist_um"] <= hexagon) & is_t
    )
    rows.append(
        {
            "sample": SHORT[sample],
            "spots with T = 0 (filtered)": len(zero_t),
            "T cells inside these spots": int(inside.sum()),
            "... still T with PPSA": int((inside & (after == "T cell")).sum()),
            "T cells between spots (their hexagons)": int(between.sum()),
            "... still T with PPSA ": int((between & (after == "T cell")).sum()),
        }
    )
erased = pd.DataFrame(rows)
erased.to_csv(paths.result_path("settings_ppsa_zero_spots.csv"), index=False)
erased

# %% [markdown]
# ## 0.4 Subtypes HEDeST cannot separate
#
# lv4 splits T cells into CD4 and CD8; lv5 into T helper, T reg and three CD8 states, and
# macrophages into three states. CD4 and CD8 T cells, or T helper and T reg cells, look the same on
# H&E. On most slides HEDeST gives (almost) every T cell the same subtype, the one that dominates
# its deconvolution, with or without PPSA: the share of the cells it calls each T subtype, against
# the mean proportion in the deconvolution (lv4: CD8 or CD4 under 0.5% of the cells on 4 of the 6
# slides; lv5: no T reg on any slide, two or three of the five T subtypes at most). So T subtypes
# are never analysed per cell; lv4 and lv5 T cells are merged into "T cell" in every figure.
# Plasma cells, on the contrary, are separated from B cells at lv4: their share follows the
# deconvolution on every slide.

# %%
focus = [
    "T CD4",
    "T CD8",
    "plasma cell",
    "T helper",
    "T reg",
    "T CD8 EM",
    "T CD8 cytotoxic",
    "T CD8 exhaust",
    "LAM macroph",
    "intermediate macroph",
    "proinflammatory macroph",
]
rule = composition["kind"] == composition["level"].map(paths.KIND_OF)
subset = composition[
    rule
    & composition["cell_type"].isin(focus)
    & ~((composition["level"] == "lv5") & (composition["cell_type"] == "plasma cell"))
]
pivot = subset.pivot_table(
    index=["level", "cell_type"], columns=["sample", "ppsa"], values="HEDeST share of cells"
).round(3)
deconv = subset.groupby(["level", "cell_type", "sample"])["deconvolution (raw, mean)"].first().unstack().round(3)
display(pivot)
deconv

# %%
fig, axes = plt.subplots(1, 2, figsize=(17, 5))
for ax, (level, types) in zip(
    axes, [("lv4", ["T CD4", "T CD8"]), ("lv5", ["T helper", "T reg", "T CD8 EM", "T CD8 cytotoxic", "T CD8 exhaust"])]
):
    kind = paths.KIND_OF[level]
    part = composition[
        (composition["level"] == level) & (composition["kind"] == kind) & composition["cell_type"].isin(types)
    ]
    total_hedest = part[part["ppsa"] == "without PPSA"].groupby("sample")["HEDeST share of cells"].sum()
    total_ppsa = part[part["ppsa"] == "with PPSA"].groupby("sample")["HEDeST share of cells"].sum()
    total_deconv = part[part["ppsa"] == "without PPSA"].groupby("sample")["deconvolution (raw, mean)"].sum()
    x = np.arange(len(SAMPLES))
    width = 0.27
    for offset, source, totals, label in (
        (-width, part[part["ppsa"] == "without PPSA"], total_hedest, "HEDeST"),
        (0, part[part["ppsa"] == "with PPSA"], total_ppsa, "HEDeST + PPSA"),
        (width, part[part["ppsa"] == "without PPSA"], total_deconv, "deconvolution"),
    ):
        bottom = np.zeros(len(SAMPLES))
        column = "deconvolution (raw, mean)" if label == "deconvolution" else "HEDeST share of cells"
        for i, cell_type in enumerate(types):
            values = (
                source[source["cell_type"] == cell_type]
                .set_index("sample")[column]
                .reindex([SHORT[s] for s in SAMPLES])
                .fillna(0)
                .to_numpy()
            )
            share = values / np.maximum(totals.reindex([SHORT[s] for s in SAMPLES]).to_numpy(), 1e-9)
            ax.bar(
                x + offset,
                share,
                width * 0.95,
                bottom=bottom,
                color=plt.get_cmap("tab10")(i),
                edgecolor="white",
                label=cell_type if offset == -width else None,
            )
            bottom += share
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT[s]}\nH  P  D" for s in SAMPLES])
    ax.set_ylabel("share of the T cells")
    ax.set_title(f"{level} ({kind}): T subtypes. H = HEDeST, P = HEDeST + PPSA, D = deconvolution", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
save(fig, "t_subtypes")

# %% [markdown]
# ## 0.5 B or T: does a lymphocyte keep its type from one model to another?
#
# Each unit is trained independently. For the cells two units both call B or T, Cohen's kappa
# of the two calls (1: same calls, 0: no better than chance; B lineage = B and plasma cells, T =
# any T subtype). The same level on the two proportion files agrees well; different levels
# agree poorly on single lymphocytes. B and T cells look alike on H&E: a single-cell B/T call is
# the least reliable output of HEDeST. Section 2 therefore checks every B/T *zone* against the other
# models (zones can agree where single cells do not).

# %%
rows = []
for sample in SAMPLES:
    lineages = {}
    for level, kind in UNITS:
        labels = info(sample, level, kind)["predictions"].idxmax(axis=1)
        lineages[f"{level}/{kind}"] = pd.Series(tls.lineage_codes(labels), index=labels.index)
    for first, second in itertools.combinations(lineages, 2):
        x, y = lineages[first], lineages[second].reindex(lineages[first].index)
        both = (x >= 0) & (y >= 0)
        agree = float((x[both] == y[both]).mean())
        px, py = float((x[both] == 1).mean()), float((y[both] == 1).mean())
        expected = px * py + (1 - px) * (1 - py)
        rows.append(
            {
                "sample": SHORT[sample],
                "pair": f"{first} vs {second}",
                "agreement": agree,
                "kappa": (agree - expected) / (1 - expected) if expected < 1 else np.nan,
            }
        )
kappa = pd.DataFrame(rows)
kappa.to_csv(paths.result_path("settings_bt_kappa.csv"), index=False)
table = kappa.pivot(index="pair", columns="sample", values="kappa")[[SHORT[s] for s in SAMPLES]]
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(table, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("B vs T calls of single lymphocytes: Cohen's kappa between units", fontsize=11)
save(fig, "bt_kappa")
table.round(2)
