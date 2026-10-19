"""
Where the deconvolved proportions sit on the perturbation axis of the fully simulated datasets.

The perturbation there is ``p_s = normalise((1 - s) * p + s * U(0, 1))``. This asks what ``s``
the cell2location proportions correspond to, two independent ways:

* **by distance** — the ``s`` whose perturbed matrices are, on average, as far from the truth as
  the deconvolved one. Distance is ``mean |p' - p|`` over the whole matrix, the magnitude
  ``03_verify.py`` uses, and repeated with the mean per-spot Jensen-Shannon divergence, which is
  scale free. A curve ``d(s)`` is built from many noise draws and inverted at the deconvolution's
  own distance.
* **by accuracy** — the ``s`` at which HEDeST trained on perturbed proportions reaches the
  balanced accuracy it reaches when trained on the deconvolved ones. This needs the trained arms
  (``runs.csv``); it is the reading that actually answers "what does deconvolution cost".

The two do not have to agree, and where they disagree that is the result: uniform noise is
unbiased per cell type and spatially white, while deconvolution error is biased per cell type and
spatially correlated. A distance-equivalent ``s`` says the proportions are as *wrong*; only the
accuracy-equivalent one says they are as *harmful*.

Everything is computed per (sample, level): ``mean |p' - p|`` averages over cell types, so a
level with 11 of them is not on the same scale as one with 3.

Usage: python equivalence.py [--config visiumhd.yaml] [--draws 50] [--no-figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "gridsearch"))
sys.path.insert(0, str(REPO / "simulations" / "full_simulations"))
sys.path.insert(0, str(STUDY / "deconv_code"))

from gridsearch import dataset_cfg  # type: ignore[attr-defined] # noqa: E402
from gridsearch import fill  # type: ignore[attr-defined] # noqa: E402
from gridsearch import load_config  # type: ignore[attr-defined] # noqa: E402
from score import jensen_shannon  # noqa: E402
from utils.data_simulation import add_perturbation  # noqa: E402

SAMPLES = ["P1", "P2", "P5"]
LEVELS = ["lv0", "lv1", "lv2"]
GRID = np.round(np.arange(0.0, 1.0001, 0.005), 4)
METRICS = ["l1", "jsd"]


def distances(base: pd.DataFrame, other: pd.DataFrame) -> dict[str, float]:
    """Both magnitudes, between a proportion matrix and the truth."""

    p, q = base.to_numpy(float), other.to_numpy(float)

    return {"l1": float(np.abs(q - p).mean()), "jsd": float(jensen_shannon(p, q).mean())}


def curve(base: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    """``d(s)`` for both magnitudes: the mean over ``draws`` noise draws, and its spread."""

    rows = []
    for i, strength in enumerate(GRID):
        values = {name: [] for name in METRICS}
        for draw in range(draws):
            # A seed per (strength, draw) so the curve is reproducible; these draws are
            # independent of the four matrices perturb.py wrote.
            perturbed = add_perturbation(base, strength=float(strength), random_state=[seed, i, draw])
            for name, value in distances(base, perturbed).items():
                values[name].append(value)
        row = {"strength": float(strength)}
        for name in METRICS:
            row[f"{name}_mean"] = float(np.mean(values[name]))
            row[f"{name}_lo"] = float(np.percentile(values[name], 2.5))
            row[f"{name}_hi"] = float(np.percentile(values[name], 97.5))
        rows.append(row)

    return pd.DataFrame(rows)


def invert(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """The ``x`` at which the increasing curve ``y(x)`` reaches ``target`` (nan outside it)."""

    if not np.all(np.diff(y) > -1e-12):
        order = np.argsort(y)
        x, y = x[order], y[order]
    if target < y[0] or target > y[-1]:
        return float("nan")

    return float(np.interp(target, y, x))


# ------------------------------------------------------------------- the accuracy-based reading


def accuracy_table(results: Path) -> pd.DataFrame | None:
    """Mean balanced accuracy per (sample, level, supervision, strength), over the seeds."""

    path = results / "runs.csv"
    if not path.exists():
        return None
    runs = pd.read_csv(path)
    runs["sample"] = runs.dataset.str.split("_").str[0]
    runs["supervision"] = runs.dataset.str.split("_").str[1].str.replace(r"\d.*", "", regex=True)
    runs["strength"] = runs.dataset.str.extract(r"perturb([\d.]+)$")[0].astype(float).fillna(0.0)
    keys = ["sample", "level", "supervision", "strength"]

    return runs.groupby(keys)[["bal_acc_raw", "bal_acc_ppsa"]].agg(["mean", "std", "count"]).reset_index()


def accuracy_equivalent(table: pd.DataFrame, name: str, level: str, metric: str) -> dict:
    """The strength at which the perturbed arms reach the deconvolved arm's accuracy."""

    here = table[(table["sample"] == name) & (table["level"] == level)]
    column = (metric, "mean")
    gt = here[here.supervision == "gt"]
    deconv = here[here.supervision == "deconv"]
    perturbed = here[here.supervision == "perturb"].sort_values("strength")
    if gt.empty or deconv.empty or perturbed.empty:
        return {}

    # Accuracy falls as the strength grows, so the curve is inverted on -accuracy.
    strengths = np.concatenate([[0.0], perturbed["strength"].to_numpy(float)])
    scores = np.concatenate([gt[column].to_numpy(float), perturbed[column].to_numpy(float)])
    target = float(deconv[column].iloc[0])

    # Falling off the end of the curve is a result, not a missing value: the deconvolved arm can
    # be *worse* than the strongest perturbation trained, which is the interesting case.
    beyond = ""
    if target < scores.min():
        beyond = f"> {strengths[-1]}"
    elif target > scores.max():
        beyond = "< 0"

    return {
        f"{metric}_gt": float(scores[0]),
        f"{metric}_deconv": target,
        f"{metric}_s_accuracy": invert(strengths, -scores, -target),
        f"{metric}_s_accuracy_beyond": beyond,
        f"{metric}_worst_strength": float(strengths[-1]),
        f"{metric}_at_worst": float(scores[-1]),
    }


# ------------------------------------------------------------------------------------- the driver


def figure(curves: dict, summary: pd.DataFrame, table: pd.DataFrame | None, out: Path) -> None:
    """One panel per (sample, level): the distance curve, and the accuracy curve beside it."""

    fig, axes = plt.subplots(len(SAMPLES), len(LEVELS), figsize=(5.2 * len(LEVELS), 3.9 * len(SAMPLES)), squeeze=False)
    for row, name in enumerate(SAMPLES):
        for col, level in enumerate(LEVELS):
            ax = axes[row][col]
            frame = curves.get((name, level))
            entry = summary[(summary["sample"] == name) & (summary.level == level)]
            if frame is None or entry.empty:
                ax.axis("off")
                continue
            entry = entry.iloc[0]

            ax.plot(frame.strength, frame.l1_mean, color="0.3", lw=1.4, label="mean |p' − p| of perturbation s")
            ax.fill_between(frame.strength, frame.l1_lo, frame.l1_hi, color="0.3", alpha=0.18, lw=0)
            ax.axhline(entry.l1_deconv, color="#d08b2f", ls="--", lw=1.2, label="deconvolution")
            if np.isfinite(entry.l1_s_distance):
                ax.axvline(entry.l1_s_distance, color="#d08b2f", ls=":", lw=1.2)
                ax.plot([entry.l1_s_distance], [entry.l1_deconv], "o", color="#d08b2f", ms=5)
            ax.set_xlim(0, 0.6)
            ax.set_xlabel("perturbation strength s", fontsize=8)
            ax.set_ylabel("mean |p' − p|", fontsize=8)
            ax.tick_params(labelsize=7)

            title = f"{name} {level} — distance s* = {entry.l1_s_distance:.3f}"
            if "bal_acc_raw_s_accuracy" in entry:
                beyond = str(entry.get("bal_acc_raw_s_accuracy_beyond", "") or "")
                value = entry.get("bal_acc_raw_s_accuracy", np.nan)
                shown = beyond if beyond else (f"{value:.3f}" if np.isfinite(value) else "n/a")
                title += f" | accuracy s* = {shown}"
            ax.set_title(title, fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle(
        "Where the deconvolved proportions sit on the perturbation axis "
        "(dotted: the strength as far from the truth as the deconvolution)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def shown(row: dict) -> str:
    """The accuracy-based strength, or the bound when it falls off the end of the curve."""

    beyond = row.get("bal_acc_raw_s_accuracy_beyond") or ""

    return beyond if beyond else format(row.get("bal_acc_raw_s_accuracy", float("nan")), ".3f")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "visiumhd.yaml"))
    parser.add_argument("--draws", type=int, default=50, help="noise draws per strength")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()

    cfg = load_config(arguments.config)
    results = Path(cfg["results_dir"])
    table = accuracy_table(results)
    if table is None:
        print("no runs.csv yet — the accuracy-based reading is skipped")

    rows, curves = [], {}
    for name in SAMPLES:
        source = dataset_cfg(cfg, f"{name}_gt")
        target = dataset_cfg(cfg, f"{name}_deconv")
        for level in LEVELS:
            base = pd.read_csv(fill(source["proportions"], source, level), index_col=0)
            other = pd.read_csv(fill(target["proportions"], target, level), index_col=0)[base.columns]
            observed = distances(base, other)

            frame = curve(base, arguments.draws, arguments.seed)
            curves[(name, level)] = frame
            row = {"sample": name, "level": level, "n_types": base.shape[1]}
            for metric in METRICS:
                row[f"{metric}_deconv"] = observed[metric]
                row[f"{metric}_s_distance"] = invert(
                    frame.strength.to_numpy(), frame[f"{metric}_mean"].to_numpy(), observed[metric]
                )
            if table is not None:
                for metric in ["bal_acc_raw", "bal_acc_ppsa"]:
                    row.update(accuracy_equivalent(table, name, level, metric))
            rows.append(row)
            print(
                f"  {name} {level}: |diff| {observed['l1']:.4f} -> s {row['l1_s_distance']:.3f}"
                f" | JSD {observed['jsd']:.4f} -> s {row['jsd_s_distance']:.3f}"
                + (f" | accuracy -> s {shown(row)}" if "bal_acc_raw_s_accuracy" in row else "")
            )

    summary = pd.DataFrame(rows)
    results.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results / "equivalence.csv", index=False)
    print(f"\nwrote {results / 'equivalence.csv'}")

    if not arguments.no_figures:
        out = Path(cfg["plots_dir"]) / "perturbation_equivalence.png"
        figure(curves, summary, table, out)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
