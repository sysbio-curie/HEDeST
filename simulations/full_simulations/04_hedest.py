"""HEDeST on every fully simulated dataset, with the package defaults, one run per seed.

Nothing is reimplemented here: every run is ``hedest.run_model.run_hedest`` with its default
parameters (the gridsearch optimum, see the repository README), so a run of this script is
exactly what a user of the package would get. The only non-default argument is ``gated=True``:
the simulated spots have no coordinates and every cell lies inside a spot, so PPSA is applied
with each cell's own spot proportions and nothing is extrapolated.

A *sample* is a dataset of ``config.DATASETS`` together with one proportion matrix: the exact
proportions, or one of the four perturbed versions when the dataset has them. The 16 datasets
therefore give 32 samples, each run with ``--seeds`` seeds under
``models/full_sim/{sample}/seed_{rs}/``, which is the layout ``hedest.analysis.load_run``
expects (it reads a folder of ``seed_*`` runs as an ensemble).

Runs are skipped when their ``info.pickle`` is already there, so the script is resumable and
can be sharded over several jobs with ``--shard i --nshards n``.

    python simulations/full_simulations/04_hedest.py --help
    python simulations/full_simulations/04_hedest.py --only 30spots --seeds 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import torch
from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import config as C  # noqa: E402

from hedest.analysis.loaders import INFO_NAME  # noqa: E402
from hedest.dataset_utils import pp_prop  # noqa: E402
from hedest.run_model import run_hedest  # noqa: E402
from hedest.utils import format_time  # noqa: E402

OUT_ROOT = os.path.join(REPO, "models", "full_sim")
N_SEEDS = 10


def samples(datasets: Optional[List[dict]] = None) -> List[dict]:
    """
    The (dataset, proportion matrix) pairs to run, in the order of ``config.DATASETS``.

    Args:
        datasets: The dataset dictionaries to expand. ``config.DATASETS`` when None.

    Returns:
        One dictionary per sample: its ``name`` (the output folder), the dataset ``tag``, the
        proportion file ``prop`` and the perturbation ``strength`` (0 for the exact proportions).
    """

    out = []
    for d in datasets or C.DATASETS:
        tag = C.dataset_tag(d)
        out.append({"name": tag, "tag": tag, "prop": f"{tag}_prop.csv", "strength": 0.0})
        if d["perturb"]:
            for s in C.PERTURBATION_STRENGTHS:
                out.append(
                    {
                        "name": f"{tag}_perturb_{s}",
                        "tag": tag,
                        "prop": f"{tag}_perturb_{s}_prop.csv",
                        "strength": float(s),
                    }
                )
    return out


def load_sample(sample: dict) -> tuple:
    """
    Loads the three inputs of a sample.

    The cell ids of ``spot_dict`` are strings, so the embedding keys are read as strings too;
    the proportions go through ``pp_prop``, which is what ``hedest/main.py`` does (it casts the
    spot ids to strings and renormalises every row).

    Args:
        sample: A sample of :func:`samples`.

    Returns:
        ``(embed_dict, spot_prop_df, spot_dict)``.
    """

    embed_dict: Dict[str, torch.Tensor] = {
        str(cell): emb for cell, emb in torch.load(os.path.join(C.SIM_DIR, f"{sample['tag']}_emb_dict.pt")).items()
    }
    spot_prop_df = pp_prop(os.path.join(C.SIM_DIR, sample["prop"]))
    with open(os.path.join(C.SIM_DIR, f"{sample['tag']}_spot_dict.json")) as handle:
        spot_dict = {str(spot): [str(cell) for cell in cells] for spot, cells in json.load(handle).items()}

    missing = {cell for cells in spot_dict.values() for cell in cells} - set(embed_dict)
    if missing:
        raise ValueError(f"{sample['name']}: {len(missing)} cells of spot_dict have no embedding")

    return embed_dict, spot_prop_df, spot_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of seeds per sample (rs = 0 .. seeds-1).")
    parser.add_argument("--only", type=str, default=None, help="Run only the samples whose name contains this.")
    parser.add_argument("--shard", type=int, default=0, help="Index of this shard.")
    parser.add_argument("--nshards", type=int, default=1, help="Number of shards the samples are split into.")
    parser.add_argument("--out-root", type=str, default=OUT_ROOT, help="Where the runs are written.")
    parser.add_argument("--force", action="store_true", help="Rerun the runs that already have an info.pickle.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be run and exit.")
    args = parser.parse_args()

    todo = samples()
    if args.only:
        todo = [s for s in todo if args.only in s["name"]]
    todo = [s for i, s in enumerate(todo) if i % args.nshards == args.shard]
    if not todo:
        raise SystemExit("No sample selected.")

    logger.info(f"{len(todo)} samples x {args.seeds} seeds -> {args.out_root}")
    if args.dry_run:
        for s in todo:
            print(f"{s['name']:70s} prop={s['prop']}")
        return

    start, done, skipped = time.time(), 0, 0
    for n, sample in enumerate(todo, 1):
        dirs = {rs: os.path.join(args.out_root, sample["name"], f"seed_{rs}") for rs in range(args.seeds)}
        pending = [rs for rs, path in dirs.items() if args.force or not os.path.exists(os.path.join(path, INFO_NAME))]
        if not pending:
            logger.info(f"[{n}/{len(todo)}] {sample['name']}: all {args.seeds} seeds already done, skipping")
            skipped += args.seeds
            continue

        logger.info(f"[{n}/{len(todo)}] {sample['name']}: {len(pending)} seed(s) to run")
        embed_dict, spot_prop_df, spot_dict = load_sample(sample)
        logger.info(
            f"-> {len(embed_dict)} cells, {len(spot_dict)} spots, {spot_prop_df.shape[1]} clusters, "
            f"perturbation {sample['strength']}"
        )
        for rs in pending:
            t0 = time.time()
            run_hedest(
                embed_dict=embed_dict,
                spot_prop_df=spot_prop_df,
                spot_dict=spot_dict,
                gated=True,  # no cell outside a spot, and no coordinates: adjust in-spot only
                out_dir=dirs[rs],
                rs=rs,
            )
            pd.Series(
                {"sample": sample["name"], "tag": sample["tag"], "strength": sample["strength"], "seed": rs}
            ).to_json(os.path.join(dirs[rs], "sample.json"))
            done += 1
            logger.info(f"-> seed {rs} done in {format_time(time.time() - t0)}")

        elapsed = time.time() - start
        left = sum(args.seeds for s in todo[n:])
        logger.info(
            f"== {done} runs in {format_time(elapsed)} ({elapsed / max(done, 1):.0f}s/run), "
            f"{skipped} skipped, ~{format_time(left * elapsed / max(done, 1))} left"
        )

    logger.info(f"Finished: {done} runs, {skipped} skipped, {format_time(time.time() - start)}")


if __name__ == "__main__":
    main()
