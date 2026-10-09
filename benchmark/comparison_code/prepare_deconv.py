"""Builds the inputs HEDeST needs to run on the deconvolution-derived proportions.

The deconvolution did not estimate every spot: its proportion matrices are a strict subset
of the true ones (1 to 289 spots short, depending on the sample). HEDeST trains on the spots
of ``spot_dict``, so a spot with no proportions would stop the run. This writes, per
sample-level, a spot dictionary restricted to the spots the deconvolution produced — the same
spots HistoCell and PanoSpace were given — leaving the shared engine untouched.

The cells of the dropped spots keep their predictions: they simply fall outside the training
spots, and PPSA reaches them by interpolation like any other cell outside a spot.

    python benchmark/comparison_code/prepare_deconv.py
"""
from __future__ import annotations

import argparse
import json
import os

import config as C
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=C.DECONV_INPUT_DIR, help="Where the filtered spot dicts go.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    for sample, level in C.units("deconv"):
        with open(C.spot_dict_path(sample, level, "gt")) as handle:
            spot_dict = {str(k): [str(c) for c in v] for k, v in json.load(handle).items()}
        spot_dict = {k: v for k, v in spot_dict.items() if v}

        prop = pd.read_csv(C.proportions_path(sample, level, "deconv"), index_col=0)
        prop.index = prop.index.astype(str)
        kept = {k: v for k, v in spot_dict.items() if k in prop.index}

        path = os.path.join(args.out_dir, f"spot_dict_{sample}_{level}.json")
        with open(path, "w") as handle:
            json.dump(kept, handle)

        rows.append(
            {
                "sample": sample,
                "level": level,
                "spots_total": len(spot_dict),
                "spots_kept": len(kept),
                "spots_dropped": len(spot_dict) - len(kept),
                "cells_total": sum(len(v) for v in spot_dict.values()),
                "cells_dropped": sum(len(v) for k, v in spot_dict.items() if k not in prop.index),
                "n_types": prop.shape[1],
            }
        )
        print(f"{sample}/{level}: kept {len(kept)}/{len(spot_dict)} spots -> {os.path.basename(path)}", flush=True)

    table = pd.DataFrame(rows)
    summary = os.path.join(args.out_dir, "deconv_spot_coverage.csv")
    table.to_csv(summary, index=False)
    print(
        f"\n{len(table)} sample-levels, {table.spots_dropped.sum()} spots dropped in total "
        f"({100 * table.spots_dropped.sum() / table.spots_total.sum():.2f}% of spots, "
        f"{100 * table.cells_dropped.sum() / table.cells_total.sum():.2f}% of cells in spots)\n-> {summary}"
    )


if __name__ == "__main__":
    main()
