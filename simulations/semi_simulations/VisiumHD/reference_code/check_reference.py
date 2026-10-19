"""
Checks the written references against the samples they were built for.

Four questions, none of which ``build_reference.py`` can answer about its own output:

1. does each reference's vocabulary match the header of the proportions it will be compared to?
2. is the hierarchy still the sample's own (every lv2 type under the right lv1, and so on)?
3. does the reference cover the genes the sample actually measured?
4. does each cell type score highest on its own marker panel?

The fourth is the C6 check of the samples' pseudo-Visium builds, asked of the reference instead.
It is reported, never enforced: a panel can lose to a neighbouring type for honest reasons, and
the point is to see which and decide, not to fail the build.

Usage: python check_reference.py [--samples P1 P2 ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import pandas as pd
import yaml

sys.dont_write_bytecode = True
from build_reference import DATA_ROOT, LEVELS, SAMPLE_DIR, marker_score, sample_levels  # noqa: E402

HERE = Path(__file__).resolve().parent


def check_vocabulary(reference: ad.AnnData, sample: str, level: str) -> list[str]:
    """The reference's types against the level contract and the proportions header."""

    problems = []
    contract = sample_levels(sample)
    entry = next(e for e in contract["levels"] if f"lv{e['level']}" == level)

    found = list(reference.obs.cell_type.cat.categories)
    if found != entry["cell_types"]:
        problems.append(f"{level}: types {found} != levels.json {entry['cell_types']}")

    proportions = DATA_ROOT / SAMPLE_DIR.format(sample=sample) / "pseudovisium" / "sim" / f"proportions_{level}.csv"
    columns = pd.read_csv(proportions, nrows=0).columns.tolist()[1:]
    if sorted(found) != sorted(columns):
        problems.append(f"{level}: types do not match {proportions.name} header {columns}")

    empty = [k for k, v in reference.obs.cell_type.value_counts().items() if v == 0]
    if empty:
        problems.append(f"{level}: no cells for {empty}")

    return problems


def check_hierarchy(references: dict[str, ad.AnnData], sample: str) -> list[str]:
    """Every fine type must sit under the parent the sample's contract gives it."""

    problems = []
    contract = sample_levels(sample)
    by_level = {f"lv{e['level']}": e for e in contract["levels"]}

    for child, parent in (("lv2", "lv1"), ("lv1", "lv0")):
        frame = references[child].obs
        for kind, group in frame.groupby("cell_type", observed=True):
            seen = set(group[parent].astype(str).unique())
            expected = by_level[child]["parent"][str(kind)]
            if seen != {expected}:
                problems.append(f"{child} {kind!r}: {parent} is {sorted(seen)}, expected {expected!r}")

    return problems


def check_genes(reference: ad.AnnData, sample: str) -> list[str]:
    """The reference has to carry the genes the sample measured, or it cannot explain them."""

    path = DATA_ROOT / SAMPLE_DIR.format(sample=sample) / f"{SAMPLE_DIR.format(sample=sample)}_annotated.h5ad"
    # Read var straight out of HDF5 rather than through anndata: these objects were written by a
    # newer anndata, and this one cannot decode their `uns/log1p/base` null. Only the gene names
    # are wanted here, so the rest of the file need not be parsed at all.
    with h5py.File(path, "r") as handle:
        index = handle["var"].attrs["_index"]
        index = index.decode() if isinstance(index, bytes) else index
        measured = {name.decode() for name in handle[f"var/{index}"][:]}
    missing = measured - set(reference.var_names)
    share = 1 - len(missing) / len(measured)
    print(
        f"    genes: {len(measured) - len(missing):,}/{len(measured):,} of the sample's "
        f"genes are in the reference ({share:.2%})"
    )

    return [f"reference misses {len(missing)} of the sample's genes"] if share < 0.99 else []


def check_markers(reference: ad.AnnData, markers: dict) -> None:
    """For each type, whether its own panel scores highest on it."""

    kinds = [k for k in reference.obs.cell_type.cat.categories if k in markers]
    scores = pd.DataFrame(
        {kind: marker_score(reference, markers[kind]) for kind in kinds},
        index=reference.obs_names,
    )
    means = scores.groupby(reference.obs.cell_type.to_numpy(), observed=True).mean()

    wins = 0
    for kind in kinds:
        best = means[kind].idxmax()
        flag = "ok  " if best == kind else "LOST"
        wins += best == kind
        detail = "" if best == kind else f"  (highest on {best!r})"
        print(f"      {flag} {kind}{detail}")
    print(f"    markers: {wins}/{len(kinds)} types score highest on their own panel")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=["P1", "P2", "P5"])
    arguments = parser.parse_args()

    markers = yaml.safe_load((HERE / "mapping.yaml").read_text())["markers"]
    problems = []

    for sample in arguments.samples:
        folder = DATA_ROOT / SAMPLE_DIR.format(sample=sample) / "references"
        print(f"\n{sample}  {folder}")
        if not folder.is_dir():
            problems.append(f"{sample}: no references folder")
            continue

        references = {level: ad.read_h5ad(folder / f"reference_{level}.h5ad") for level in LEVELS}
        full = ad.read_h5ad(folder / "reference_full.h5ad")
        print(f"    full: {full.n_obs:,} cells x {full.n_vars:,} genes, " f"{full.obs.cell_type.nunique()} fine types")

        for level in LEVELS:
            reference = references[level]
            found = problems_here = check_vocabulary(reference, sample, level)
            problems += problems_here
            counts = reference.obs.cell_type.value_counts()
            print(
                f"    {level}: {len(counts)} types, {reference.n_obs:,} cells" f"{'  OK' if not found else '  PROBLEM'}"
            )

        problems += check_hierarchy(references, sample)
        problems += check_genes(references["lv2"], sample)

        # The fine vocabulary is the one that has to be separable; the coarse levels are sums of it.
        print("    marker self-check on lv2:")
        check_markers(references["lv2"], markers)

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("all checks passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
