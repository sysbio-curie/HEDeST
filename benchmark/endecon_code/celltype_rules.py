"""Atlas cell type -> canonical tag -> the class names of a sample's level.

Two stages, so the mapping stays auditable rather than being a lookup table:

  stage A   a regex table gives each atlas cell type one canonical tag
            (B, T, NK, Mono, Mac, DC, Mast, Fib, SMC, Peri, Endo, AT1, AT2, ...)
  stage B   each class of a level's vocabulary declares which tags it accepts;
            a tag goes to the *most specific* class that accepts it (smallest
            accepted-tag set), so `NK` lands on `NK` when the level has such a
            class and on `T` / `T_NK` when it does not.

Validation
----------
Checked against the first round's ``benchmark/results/PanoSpace/deconv/label_map.json``:
403 of 405 assignments identical. The two differences are prostate `B_cell`
(178 cells, 0.8% of that atlas), which the first round dropped at every level
because prostate has no B class at level 2; it has no home at level 3 either,
so no run is affected.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- stage A
# first match wins; matched case-insensitively, with "_" read as a space so
# 3CA labels such as "Smooth_muscle" and "T_cell" hit the same word rules.
TAG_RULES = [
    # --- immune, lymphoid
    (r"\bplasma cell\b|\bplasmablast\b", "Plasma"),
    (r"\b(pro|pre)b cell\b|\bb cell\b|\bmemory b\b|\bnaive b\b", "B"),
    (r"\bnk cell\b|\bnk\b", "NK"),
    (r"\bilc\b", "ILC"),
    (r"\bt/nk\b", "T"),
    (
        r"\bt cell\b|\bt_cell\b|\btreg\b|\btfh\b|\bmait\b|\bcd4\b|\bcd8\b|"
        r"gamma delta|double negative t|inf-activated t",
        "T",
    ),
    # --- immune, myeloid
    (r"\bmast\b", "Mast"),
    (r"\bcdc[12]?\b|\bpdc\b|\bmregdc\b|dendritic|pre-pdc", "DC"),
    (r"monocyte", "Mono"),
    (r"macrophage|osteoclast|microglia", "Mac"),
    (r"neutrophil|myelocyte|promyelocyte", "Neut"),
    (r"\bmyeloid\b", "Mac"),
    (r"erythroblast|erythrocyte|red blood cell", "Ery"),
    (
        r"megakaryocyte|hematopoietic stem|multipotent progenitor|"
        r"(myeloid|lymphoid|erythroid|dendritic|monocyte) progenitor",
        "Prog",
    ),
    # --- structural
    (r"myofibroblast", "Myofib"),
    (r"fibroblast|mesenchymal stromal", "Fib"),
    (r"smooth muscle", "SMC"),
    (r"pericyte|perivascular", "Peri"),
    (r"\bec\b|endothelial|aerocyte", "Endo"),
    (r"\bglial\b|\bneuron\b|schwann", "Neural"),
    # --- epithelial (tissue-specific subsets before the generic rule)
    (r"^at1$|alveolar type i\b|typei ", "AT1"),
    (r"^at2$|cycling at2|alveolar type ii\b|typeii ", "AT2"),
    (r"multiciliated|ciliated", "Cil"),
    (r"club cell", "Club"),
    (r"airway basal", "BasalAir"),
    (r"goblet", "Goblet"),
    (r"neuroendocrine", "NE"),
    (r"keratinocyte", "Kerat"),
    (r"merkel", "Merkel"),
    (r"melanocyte", "Mel"),
    (r"mammary|lactocyte", "Mamm"),
    (r"mesothelial", "Meso"),
    (r"granulosa", "Granulosa"),
    (r"oocyte|germ cell", "Germ"),
    (r"podocyte", "Podo"),
    (r"endocrine cell", "Endocrine"),
    (r"malignant", "Malig"),
    (r"secretory cell", "Secr"),
    (r"epithelial", "Epi"),
]

EPI_TAGS = {
    "Epi",
    "AT1",
    "AT2",
    "Cil",
    "Club",
    "BasalAir",
    "Goblet",
    "NE",
    "Kerat",
    "Merkel",
    "Secr",
    "Mamm",
    "Meso",
    "Granulosa",
    "Malig",
}
IMMUNE_TAGS = {"B", "Plasma", "T", "NK", "ILC", "Mono", "Mac", "DC", "Mast", "Neut", "Ery", "Prog"}
STROMAL_TAGS = {"Fib", "Myofib"}
VESSEL_TAGS = {"Endo", "SMC", "Peri"}


def tag_of(cell_type: str):
    """Canonical tag for one atlas cell type, or None when no rule matches."""
    name = str(cell_type).strip().lower().replace("_", " ")
    for pat, tag in TAG_RULES:
        if re.search(pat, name):
            return tag
    return None


# ---------------------------------------------------------------- stage B
# class name -> tags it accepts. Overlaps are resolved by specificity, so a set
# is deliberately wide when the class is the only one of its lineage in a
# vocabulary ("T" also takes NK where the level has no NK class).
CLASS_TAGS = {
    "Immune": IMMUNE_TAGS,
    "Structural": STROMAL_TAGS | VESSEL_TAGS,
    "Stromal": STROMAL_TAGS,
    "Lymphoid": {"B", "Plasma", "T", "NK", "ILC"},
    "Myeloid": {"Mono", "Mac", "DC", "Mast", "Neut", "Ery", "Prog"},
    "B_Plasma": {"B", "Plasma"},
    "T_NK": {"T", "NK", "ILC"},
    "B": {"B"},
    "Plasma": {"Plasma"},
    "T": {"T", "NK", "ILC"},
    "T_CD8": {"T", "NK", "ILC"},
    "NK": {"NK"},
    "Monocyte/Macrophage": {"Mono", "Mac", "DC", "Neut"},
    "Monocyte/Macrophage/DC": {"Mono", "Mac", "DC", "Neut"},
    "DC": {"DC"},
    "Mast": {"Mast"},
    "Fibroblast_Myofibroblast": STROMAL_TAGS,
    "Fibroblast/Myofibroblast": STROMAL_TAGS,
    "Fibroblast": STROMAL_TAGS,
    "CAF": STROMAL_TAGS,
    "Myofibroblast": {"Myofib"},
    "Blood_vessel": VESSEL_TAGS,
    "Perivascular": {"SMC", "Peri"},  # fine-grained split of Blood_vessel
    "Smooth_muscle": {"SMC", "Peri"},
    "Pericyte": {"Peri"},
    "Endothelial": {"Endo"},
    "Epithelial": EPI_TAGS,
    "Melanocyte": {"Mel"},
    "Keratinocyte": {"Kerat", "Merkel", "Secr"},
    "Lung_epithelial_cell": {"Epi", "Club", "BasalAir", "Goblet", "NE"},
    "Lung_alveolar_typeI_epithelial_cell": {"AT1"},
    "Lung_alveolar_typeII_epithelial_cell": {"AT2"},
    "Lung_ciliated_epithelial_cell": {"Cil"},
    "Breast_epithelial_cell": EPI_TAGS,
}


def map_types(cell_types, vocabulary, granulosa_as_epithelial=True):
    """``({cell type: class}, {cell type: why it was dropped})``.

    ``granulosa_as_epithelial=False`` removes the DISCO ovary atlas's granulosa
    cells (31% of it) from the reference; see the README, the choice is decided
    by measurement per sample, not assumed.
    """
    assigned, dropped = {}, {}
    for ct in cell_types:
        tag = tag_of(ct)
        if tag is None or (tag == "Granulosa" and not granulosa_as_epithelial):
            dropped[str(ct)] = "no rule" if tag is None else "granulosa excluded"
            continue
        cands = [c for c in vocabulary if tag in CLASS_TAGS.get(c, ())]
        if not cands:
            dropped[str(ct)] = f"tag '{tag}' not in this level's vocabulary"
            continue
        assigned[str(ct)] = min(cands, key=lambda c: (len(CLASS_TAGS[c]), c))
    return assigned, dropped


# --------------------------------------------------------- fine-grained fit
# The reported `Blood_vessel` lumps endothelium with perivascular cells, and the
# atlas's blend of the two is not the tissue's (lung: 27.6% perivascular in the
# reference against 48.9% in the truth; prostate 31.9% against 76.0%). A single
# averaged signature is therefore wrong for the tissue, the class gets
# under-called, and the missing mass lands on `Fibroblast` -- smooth muscle is
# transcriptionally far closer to a fibroblast than to an endothelial cell.
#
# The fix is to deconvolve at a finer granularity and sum back: each signature is
# then homogeneous and the model decides how much smooth muscle the tissue holds.
# No ground truth enters the reference, only the fit's granularity changes.
# (Reweighting the reference to match the truth's composition *would* leak: that
# composition is the answer.)
FINE_SPLIT = {
    "Blood_vessel": ["Endothelial", "Perivascular"],
    "Fibroblast_Myofibroblast": ["Fibroblast"],
    "Fibroblast/Myofibroblast": ["Fibroblast"],
}


def fine_vocabulary(level2_vocabulary):
    """The level-2 vocabulary with its structural classes split apart."""
    out = []
    for c in level2_vocabulary:
        out.extend(FINE_SPLIT.get(c, [c]))
    return out


def fine_to_level2(level2_vocabulary):
    """``{fine class: the level-2 class it is summed back into}``."""
    m = {}
    for c in level2_vocabulary:
        for f in FINE_SPLIT.get(c, [c]):
            m[f] = c
    return m
