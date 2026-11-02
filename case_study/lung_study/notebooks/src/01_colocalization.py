# %% [markdown]
# # 1. Colocalization: proportions vs cells
#
# **The question.** Which cell types sit together, and which avoid each other? With a
# deconvolution, "A and B colocalize" means that B is more abundant in the spots holding A (or
# that their proportions go up and down together across spots): measured at the scale of a spot
# (55 µm, ~14 nuclei). HEDeST gives every nucleus a type, so colocalization can be measured
# between the cells themselves: do A and B cells **touch**?
#
# **Slides**: A_AAAA1879, A_AAAA1638, A_AAAA1665 (`paths.ANALYSIS_SAMPLES`).
#
# **Touching cells.** H&E shows nuclei, not cell borders. A cell is drawn the way QuPath's cell
# expansion draws it: its nucleus grown by up to 5 µm, never into another cell (each point within
# 5 µm of a nucleus goes to the closest nucleus outline). Two cells touch when these shapes share a
# border (`lungstudy/contacts.py`). Touching nuclei lie 5-16 µm apart (centre to centre), and a
# cell touches 4 others (median).
#
# **The scores**, for each pair of types (A, B), on the same scale:
#
# - **co-occurrence** (deconvolution): log2( mean(pA·pB) / (mean pA · mean pB) ) across spots;
# - **enrichment at contact** (HEDeST): among the cells touching an A cell, is B more common than
#   on the whole slide? 0 = as common as with random mixing, +1 = twice, −1 = half. Averaged both
#   ways (B around A and A around B); a pseudo-count of 0.5 contact keeps rare pairs finite.
#
# Both are Squidpy's co-occurrence score (Palla et al., *Nature Methods* 2022),
# P(B | near an A) / P(B), with two definitions of "near". For the spots, "near" is "in the same
# spot": pick an A cell (it lies in spot s with probability pA(s) / Σ pA), then another cell of
# its spot (B with probability pB(s)): P(B | same spot as an A) = mean(pA·pB) / mean(pA), and
# P(B) = mean(pB). This assumes every spot holds as many cells, as Pearson on proportions does. For
# the cells, "near" is "touching". The diagonal of the plots below therefore means "same answer".
#
# **Types analysed**: every lv3 type with at least 100 HEDeST cells and a non-zero deconvolution
# on the slide (a contact score on fewer cells means nothing).
# %%
from __future__ import annotations

{{SETUP}}
from lungstudy import coloc as C
from lungstudy import figures as F

PART = "01_colocalization"
SAMPLES = paths.ANALYSIS_SAMPLES


def save(fig, name):
    return paths.save(fig, PART, name)


data = {s: SampleData(s) for s in SAMPLES}
types_of = {s: C.analysed_types(d) for s, d in data.items()}
contacts = pd.DataFrame(
    {
        SHORT[s]: {
            "touching pairs": len(d.contacts),
            "contacts per cell (median)": float(np.median(np.bincount(d.contacts.ravel(), minlength=len(d.cells)))),
            "centre distance, median (µm)": float(
                np.median(
                    np.linalg.norm(
                        d.cells[["x_um", "y_um"]].to_numpy()[d.contacts[:, 0]]
                        - d.cells[["x_um", "y_um"]].to_numpy()[d.contacts[:, 1]],
                        axis=1,
                    )
                )
            ),
            "types analysed": len(types_of[s]),
        }
        for s, d in data.items()
    }
)
display(contacts.round(1))
pd.Series({s: ", ".join(t) for s, t in types_of.items()}, name="types analysed (≥ 100 HEDeST cells)")

# %% [markdown]
# ## 1.1 Where do the spots and the cells disagree?
#
# For every pair: x = deconvolution co-occurrence, y = HEDeST enrichment at contact, both in log2
# of "how much more than random". On the diagonal, the two say the same thing. Zones:
#
# - **same answer** (grey band): |y − x| < 0.5, i.e. within a factor 1.41;
# - **HEDeST: more colocalization** (light blue): the cells touch more than the spots suggest;
# - **HEDeST: less colocalization** (light orange): the cells touch less than the spots suggest;
# - **opposite answers** (dark blue / dark orange): the spots say "together" and the cells "apart",
#   or the reverse; called only when both signs hold over the 95% intervals.
#
# **Error bars**: 95% intervals from a spatial block bootstrap. The slide is cut in 500 µm
# squares; a replicate draws as many squares with replacement (a square drawn twice counts twice:
# its cells, its contacts and its spots), and recomputes both scores and their gap; 300 replicates,
# the bar runs from the 2.5th to the 97.5th percentile. They answer "would another piece of the
# same tissue give the same answer?". Squares, not cells, because neighbouring cells are not
# independent. The cell labels are fixed (top type of the 5-seed mean): seed variation is not in
# the bars.
#
# **Filled or hollow.** A pair off the band is filled when the 95% interval of its *gap*
# (y − x, recomputed on each replicate) excludes 0; hollow when it does not (the two methods may
# agree: counted as "same answer"). The gap's interval is not drawn: both scores move together
# over the replicates, so it is narrower than the two bars suggest.

# %%
scores = []
for sample, d in data.items():
    scores.append(C.pair_scores(d, types_of[sample]))
scores = pd.concat(scores, ignore_index=True)
scores["zone_name"] = scores["zone"].map(C.ZONES)
scores.to_csv(paths.result_path("colocalization_zones.csv"), index=False)

for sample in SAMPLES:
    subset = scores[scores["sample"] == sample]
    fig, ax = plt.subplots(figsize=(10, 10))
    F.zone_panel(ax, subset, f"{sample}: all {len(subset)} pairs of cell types")
    display(save(fig, f"zones_{sample}"))

# %%
display(F.zone_counts(scores, SAMPLES).rename(columns=C.ZONES))
scores.loc[
    (scores["zone"] != "same") & scores["robust"],
    ["sample", "pair", "deconv_cooc", "contact", "gap", "gap_low", "gap_high", "zone_name"],
].round(2)

# %% [markdown]
# ## 1.2 What the disagreements look like on the H&E
#
# **Which pairs** (by rule): the robust disagreements a deconvolution study would get wrong
# and that involve an immune type: the spots say "together" (co-occurrence ≥ 0) and the cells
# say less, or the cells touch more than the spots say. "Opposite" pairs first, then by the size
# of the gap; at most two pairs per slide.
#
# **Which places** (never by eye): windows of 220 µm with at least 20 cells of each type (10 on a
# slide where no pair has such a window), at least 1 mm apart. For "apart" pairs (HEDeST less
# than the spots): windows centred on the spots where the deconvolution gives both types ≥ 15%
# (the spots say "together"), the two with the most contacts *missing* compared with random
# mixing of the window's cells. For "together" pairs: windows centred on touching A–B pairs, the
# two with the most contacts *in excess* of random mixing. Each row: H&E alone, the deconvolution
# pies, and the HEDeST cells of the two types with every touching pair drawn in black.

# %%
IMMUNE = {"B cell", "T cell", "macrophage"}
candidates = scores[
    (scores["zone"] != "same") & scores["robust"] & (scores["A"].isin(IMMUNE) | scores["B"].isin(IMMUNE))
].copy()
candidates = candidates[(candidates["zone"].isin(["more", "flip_together"])) | (candidates["deconv_cooc"] >= 0)]
candidates["priority"] = 10 * candidates["zone"].str.startswith("flip") + candidates["gap"].abs()
candidates = candidates.sort_values("priority", ascending=False)
display(candidates[["sample", "pair", "deconv_cooc", "contact", "gap", "zone"]].round(2))

examples = []
for min_cells in (20, 10):
    for _, row in candidates.iterrows():
        sample = row["sample"]
        done = sum(e["sample"] == sample for e in examples)
        if done >= 2 or (min_cells < 20 and done):
            continue
        places = C.example_windows(data[sample], row["A"], row["B"], together=row["gap"] > 0, min_cells=min_cells)
        if len(places):
            examples.append({"sample": sample, "row": row, "places": places, "min_cells": min_cells})
examples.sort(key=lambda e: SAMPLES.index(e["sample"]))

example_rows = []
for example in examples:
    sample, row, places = example["sample"], example["row"], example["places"]
    a, b = row["A"], row["B"]
    title = f"{sample}: {a} – {b}  |  spots: co-occurrence {row['deconv_cooc']:+.2f}  |  cells: contact {row['contact']:+.2f}  ({C.ZONES[row['zone']]})"
    name = f"examples_{sample}_{a}_{b}".replace("/", "-").replace(" ", "_")
    display(save(F.pair_crops(data[sample], a, b, places, title=title), name))
    example_rows.append(
        places.assign(sample=sample, pair=f"{a} – {b}", zone=row["zone"], min_cells=example["min_cells"])
    )
example_table = pd.concat(example_rows)
example_table.to_csv(paths.result_path("colocalization_examples.csv"), index=False)
example_table.round(2)
