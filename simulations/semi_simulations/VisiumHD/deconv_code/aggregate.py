"""
Sum a sample's deconvolution into an annotation level's vocabulary.

The run uses ``reference_full.h5ad``, whose cell types refine all three levels, so one
deconvolution per sample serves lv0, lv1 and lv2. The map from a reference type to a level's label
is read from the ``mapping.csv`` written beside the references — the same table the references were
built from, so the proportions and the reference cannot disagree about what a label means.

Reference types the level has no label for carry ``drop`` in that table. Their columns are removed
and each spot is renormalised, which is the comparison the ground truth asks for: a level's
proportions are a composition over that level's types only. P5's lv2 has no macrophage label, for
instance, so the macrophage abundance the run estimated is excluded rather than smeared over the
types that remain.

    python aggregate.py --sample P5 --level lv2
    python aggregate.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

DROP = "drop"


def aggregation_map(sample: paths.Sample, level: str) -> dict[str, str]:
    """``{reference cell type: label at this level}``, excluding the ones the level drops."""

    table = pd.read_csv(sample.mapping)

    return {row.cell_type: row[level] for _, row in table.iterrows() if row[level] != DROP}


def aggregate(sample: paths.Sample, level: str) -> pd.DataFrame:
    """The run's proportions, summed into ``level`` and renormalised per spot."""

    fine = pd.read_csv(paths.run_dir(sample.name) / "proportions.csv", index_col=0)
    mapping = aggregation_map(sample, level)
    wanted = sample.vocabulary(level)

    kept = [c for c in fine.columns if c in mapping]
    dropped = [c for c in fine.columns if c not in mapping]
    # Transpose to group rows rather than columns: groupby over an axis is deprecated, and this
    # reads the same.
    summed = fine[kept].T.groupby([mapping[c] for c in kept]).sum().T.reindex(columns=wanted)
    if summed.isna().any().any():
        absent = [c for c in wanted if summed[c].isna().all()]
        raise SystemExit(f"{sample.name} {level}: nothing summed into {absent}")

    totals = summed.sum(axis=1).replace(0.0, float("nan"))
    summed = summed.div(totals, axis=0).fillna(0.0)
    summed.index.name = "spot_id"

    print(
        f"  {sample.name} {level}: {len(kept)} reference types -> {len(wanted)} labels"
        + (f", dropped {len(dropped)}: {', '.join(sorted(dropped))}" if dropped else "")
    )

    return summed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=paths.SAMPLES)
    parser.add_argument("--level", choices=paths.LEVELS)
    parser.add_argument("--all", action="store_true", help="every sample and level")
    arguments = parser.parse_args()

    if arguments.all:
        wanted = [(s, lv) for s in paths.SAMPLES for lv in paths.LEVELS]
    elif arguments.sample and arguments.level:
        wanted = [(arguments.sample, arguments.level)]
    else:
        raise SystemExit("give --sample and --level, or --all")

    for name, level in wanted:
        sample = paths.sample(name)
        summed = aggregate(sample, level)
        out = paths.proportions_path(name, level)
        summed.to_csv(out)
        print(f"    wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
