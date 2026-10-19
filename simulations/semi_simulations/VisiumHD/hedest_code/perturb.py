"""
Writes the perturbed spot proportions the ``*_perturb<s>`` arms of ``visiumhd.yaml`` train on.

The perturbation is the one the fully simulated datasets use, imported rather than
reimplemented: ``p_s = normalise((1 - s) * p + s * U(0, 1))``. Importing it is the point — the
whole reason these arms exist is to place the deconvolution on *that* axis, which only holds if
it is the same function.

One matrix per (sample, level, strength), fixed across the ten training seeds, exactly as the
fully simulated datasets do it: the strength is a property of the dataset, the seed is a
property of the run. The noise is drawn from a seed built out of the sample, level and strength,
so re-running this writes the same files.

The gt proportions each one is built from, and the path each one is written to, are both read
out of ``visiumhd.yaml`` through the gridsearch's own path filling, so the two cannot drift.

Usage: python perturb.py [--config visiumhd.yaml] [--check-only]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "gridsearch"))
sys.path.insert(0, str(REPO / "simulations" / "full_simulations"))

from gridsearch import dataset_cfg  # type: ignore[attr-defined] # noqa: E402
from gridsearch import fill  # type: ignore[attr-defined] # noqa: E402
from gridsearch import load_config  # type: ignore[attr-defined] # noqa: E402
from utils.data_simulation import add_perturbation  # noqa: E402

SAMPLES = ["P1", "P2", "P5"]
SEED = 42


def noise_seed(code: str, level: str, strength: float) -> list[int]:
    """A stable seed per (sample, level, strength), so the files are reproducible."""

    return [SEED, SAMPLES.index(code), int(level[-1]), int(round(strength * 1000))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "visiumhd.yaml"))
    parser.add_argument("--check-only", action="store_true", help="verify the files, write nothing")
    arguments = parser.parse_args()

    cfg = load_config(arguments.config)
    wanted = [ds for ds in cfg["datasets"] if "strength" in ds]
    if not wanted:
        raise SystemExit("no dataset carries a 'strength'")

    changes: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for ds in wanted:
        code, strength = ds["code"], float(ds["strength"])
        source = dataset_cfg(cfg, f"{code}_gt")
        for level in ds["levels"]:
            base = pd.read_csv(fill(source["proportions"], source, level), index_col=0)
            out = Path(fill(ds["proportions"], ds, level))

            if arguments.check_only:
                if not out.exists():
                    raise SystemExit(f"missing {out}")
                perturbed = pd.read_csv(out, index_col=0)
            else:
                perturbed = add_perturbation(base, strength=strength, random_state=noise_seed(code, level, strength))
                out.parent.mkdir(parents=True, exist_ok=True)
                perturbed.to_csv(out)

            same_frame = list(perturbed.index) == list(base.index) and list(perturbed.columns) == list(base.columns)
            rows_ok = bool(np.allclose(perturbed.to_numpy().sum(axis=1), 1.0)) and bool((perturbed >= 0).all().all())
            if not (same_frame and rows_ok):
                raise SystemExit(f"{out.name}: frame or row sums are wrong")

            # The same magnitude 03_verify.py checks the simulated perturbations with, and the
            # quantity equivalence.py inverts.
            change = float(np.abs(perturbed.to_numpy() - base.to_numpy()).mean())
            changes.setdefault((code, level), []).append((strength, change))
            print(f"  {out.name}: mean |diff| {change:.4f}")

    print()
    for (code, level), series in sorted(changes.items()):
        series.sort()
        grows = all(b[1] > a[1] for a, b in zip(series, series[1:]))
        detail = ", ".join(f"s={s} {c:.4f}" for s, c in series)
        print(f"  {code} {level}: {'grows with strength' if grows else 'NOT MONOTONE'} — {detail}")
        if not grows:
            raise SystemExit(f"{code} {level}: perturbation does not grow with strength")

    print(f"\n{len(wanted) * 3} files {'checked' if arguments.check_only else 'written'}")
    print(f"  in {os.path.dirname(fill(wanted[0]['proportions'], wanted[0], wanted[0]['levels'][0]))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
