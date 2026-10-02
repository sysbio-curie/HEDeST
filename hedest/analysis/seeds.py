"""
What several seeds of the same run agree on.

HEDeST is trained on spot-level proportions, so a single seed can settle on a different
assignment of individual cells while fitting the spots just as well. Running a handful of
seeds and looking at where they agree is the cheapest available measure of confidence, and
it needs no ground truth.

:class:`SeedEnsemble` takes the runs of one study and exposes the mean prediction, the
spread across seeds, and the per-cell agreement: the share of seeds that vote for the
majority cell type. A cell with an agreement of 1.0 was called the same way by every seed;
one at 1/n_seeds was called differently by each.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.figure import Figure

from hedest.analysis.loaders import HedestRun
from hedest.analysis.palette import Palette
from hedest.analysis.plots import close


def prediction_entropy(predictions: pd.DataFrame, normalise: bool = True) -> pd.Series:
    """
    Shannon entropy of each cell's probability vector.

    Args:
        predictions: Cells x cell types probabilities.
        normalise: Whether to divide by ``log(n_cell_types)``, which puts the result in
            [0, 1]: 0 for a one-hot prediction, 1 for a uniform one.

    Returns:
        The entropy per cell.
    """

    probabilities = predictions.to_numpy(dtype="float64")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    if normalise and predictions.shape[1] > 1:
        entropy = entropy / math.log(predictions.shape[1])

    return pd.Series(entropy, index=predictions.index, name="entropy")


class SeedEnsemble:
    """
    A set of runs that differ only by their random seed.

    Attributes:
        runs: The runs, in the order they were given.
        seeds: Their seeds.
        cells: The cells common to all of them.
        ct_list: The cell types.
    """

    def __init__(self, runs: Sequence[HedestRun], adjusted: Optional[bool] = None) -> None:
        """
        Aligns a set of runs on their common cells and cell types.

        Args:
            runs: The runs to combine. At least one.
            adjusted: Whether to use the adjusted predictions. Defaults to what each run
                was loaded with.

        Raises:
            ValueError: If no run is given, or if they share no cell.
        """

        if not runs:
            raise ValueError("SeedEnsemble needs at least one run.")

        self.runs = list(runs)
        if adjusted is not None:
            self.runs = [run.use(adjusted) for run in self.runs]

        self.seeds = [run.params.get("rs", seed) for seed, run in enumerate(self.runs)]
        reference = self.runs[0]
        self.ct_list = reference.ct_list

        cells = reference.predictions.index
        for run in self.runs[1:]:
            if list(run.ct_list) != list(self.ct_list):
                raise ValueError("The runs do not share the same cell types, so they cannot be combined.")
            cells = cells.intersection(run.predictions.index)

        if not len(cells):
            raise ValueError("The runs share no cell.")
        if len(cells) < len(reference.predictions):
            logger.warning(
                f"Keeping the {len(cells)} cells common to all {len(self.runs)} seeds "
                f"(the first run has {len(reference.predictions)})."
            )

        self.cells = cells
        self._votes: Optional[pd.DataFrame] = None
        # (n_seeds, n_cells, n_types), aligned once so every statistic below is a plain
        # numpy reduction rather than a loop over frames.
        self._stack = np.stack(
            [run.predictions.reindex(index=cells, columns=self.ct_list).to_numpy(dtype="float64") for run in self.runs]
        )

    def __len__(self) -> int:
        return len(self.runs)

    def __repr__(self) -> str:
        return f"SeedEnsemble({len(self.runs)} seeds {self.seeds}, {len(self.cells)} cells)"

    # ------------------------------------------------------------------ statistics

    def mean(self) -> pd.DataFrame:
        """The mean predicted probability per cell and cell type."""

        return pd.DataFrame(self._stack.mean(axis=0), index=self.cells, columns=self.ct_list)

    def std(self) -> pd.DataFrame:
        """The standard deviation across seeds, per cell and cell type."""

        return pd.DataFrame(self._stack.std(axis=0), index=self.cells, columns=self.ct_list)

    def labels_per_seed(self) -> pd.DataFrame:
        """
        The cell type each seed assigns to each cell.

        Returns:
            Cells x seeds table of cell type names.
        """

        indices = self._stack.argmax(axis=2)
        names = np.asarray(self.ct_list)

        return pd.DataFrame(names[indices].T, index=self.cells, columns=[f"seed_{s}" for s in self.seeds])

    def vote_counts(self) -> pd.DataFrame:
        """
        How many seeds voted for each cell type, per cell.

        Computed with a single ``bincount`` over a flattened index rather than a pass per
        cell, which matters at 10^5 cells.

        Returns:
            Cells x cell types table of vote counts.
        """

        if self._votes is None:
            codes = self._stack.argmax(axis=2).T  # (n_cells, n_seeds)
            n_cells, n_types = len(self.cells), len(self.ct_list)
            flat = (codes + np.arange(n_cells)[:, None] * n_types).ravel()
            counts = np.bincount(flat, minlength=n_cells * n_types).reshape(n_cells, n_types)
            self._votes = pd.DataFrame(counts, index=self.cells, columns=self.ct_list)

        return self._votes

    def majority_labels(self) -> pd.Series:
        """
        The cell type most seeds agree on, per cell.

        Ties are broken by the mean probability, so the answer always matches the label the
        aggregated predictions would give.
        """

        # Adding a fraction of the mean probability breaks ties towards the stronger class
        # without ever overriding a real majority, since probabilities are below one.
        scores = self.vote_counts().to_numpy() + 0.5 * self.mean().to_numpy()
        names = np.asarray(self.ct_list)

        return pd.Series(names[scores.argmax(axis=1)], index=self.cells, name="cell_type")

    def agreement(self) -> pd.Series:
        """
        Share of seeds voting for the majority cell type, per cell, in ``(0, 1]``.

        Returns:
            The agreement per cell.
        """

        top = self.vote_counts().to_numpy().max(axis=1).astype("float64")

        return pd.Series(top / len(self.runs), index=self.cells, name="agreement")

    def entropy(self) -> pd.Series:
        """Normalised entropy of the mean prediction, per cell."""

        return prediction_entropy(self.mean())

    def abundance_per_seed(self) -> pd.DataFrame:
        """
        The composition of the slide as each seed sees it.

        Returns:
            Seeds x cell types table of the fraction of cells assigned to each type.
        """

        labels = self.labels_per_seed()
        table = pd.DataFrame(
            {
                column: labels[column].value_counts(normalize=True).reindex(self.ct_list).fillna(0.0)
                for column in labels.columns
            }
        ).T

        return table[self.ct_list]

    def summary(self) -> pd.DataFrame:
        """
        Per cell type: mean abundance across seeds, its spread, and how often the seeds
        agree on the cells assigned to it.

        Returns:
            A table indexed by cell type.
        """

        abundance = self.abundance_per_seed()
        agreement = self.agreement()
        majority = self.majority_labels()

        rows = []
        for cell_type in self.ct_list:
            selection = majority == cell_type
            rows.append(
                {
                    "cell_type": cell_type,
                    "mean_fraction": float(abundance[cell_type].mean()),
                    "std_fraction": float(abundance[cell_type].std(ddof=0)),
                    "n_cells_majority": int(selection.sum()),
                    "mean_agreement": float(agreement[selection].mean()) if selection.any() else np.nan,
                    "unanimous_fraction": (float((agreement[selection] == 1.0).mean()) if selection.any() else np.nan),
                }
            )

        return pd.DataFrame(rows).set_index("cell_type")

    def to_run(self, run_dir: Optional[Path] = None) -> HedestRun:
        """
        Collapses the ensemble into a single run holding the mean predictions.

        The per-seed spread and agreement travel with it, so an aggregate can still be told
        apart from a single seed downstream.

        Args:
            run_dir: The directory the aggregate belongs to.

        Returns:
            The aggregated run.
        """

        reference = self.runs[0]
        raw_mean = pd.DataFrame(
            np.stack(
                [
                    run.predictions_raw.reindex(index=self.cells, columns=self.ct_list).to_numpy(dtype="float64")
                    for run in self.runs
                ]
            ).mean(axis=0),
            index=self.cells,
            columns=self.ct_list,
        )

        adjusted_mean = None
        if all(run.predictions_adjusted is not None for run in self.runs):
            adjusted_mean = pd.DataFrame(
                np.stack(
                    [
                        run.predictions_adjusted.reindex(  # type: ignore[union-attr]
                            index=self.cells, columns=self.ct_list
                        ).to_numpy(dtype="float64")
                        for run in self.runs
                    ]
                ).mean(axis=0),
                index=self.cells,
                columns=self.ct_list,
            )

        params = dict(reference.params)
        params.pop("rs", None)

        return HedestRun(
            predictions_raw=raw_mean,
            predictions_adjusted=adjusted_mean,
            spot_dict=reference.spot_dict,
            proportions=reference.proportions,
            params=params,
            history=None,
            train_spot_dict=reference.train_spot_dict,
            run_dir=run_dir,
            n_seeds=len(self.runs),
            seeds=[int(seed) for seed in self.seeds],
            predictions_std=self.std(),
            agreement=self.agreement(),
            adjusted=reference.adjusted,
        )

    # ------------------------------------------------------------------ plots

    def plot_agreement(
        self,
        palette: Optional[Palette] = None,
        figsize: Tuple[float, float] = (11.0, 4.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        How much the seeds agree: the distribution over cells, and the mean per cell type.

        Args:
            palette: Colours to use.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        agreement = self.agreement()
        summary = self.summary()
        palette = palette or Palette(self.ct_list)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        levels = sorted(agreement.unique())
        counts = [float((agreement == level).mean()) for level in levels]
        axes[0].bar([f"{level:.2f}" for level in levels], counts, color="#4c72b0")
        axes[0].set_xlabel(f"share of the {len(self.runs)} seeds agreeing")
        axes[0].set_ylabel("fraction of cells")
        axes[0].set_title(
            f"unanimous on {float((agreement == 1.0).mean()) * 100:.1f}% of cells " f"(mean {agreement.mean():.3f})",
            fontsize=10,
        )

        order = summary.index
        axes[1].barh(
            np.arange(len(order)),
            summary["mean_agreement"].to_numpy(),
            color=palette.color_list(list(order)),
        )
        axes[1].set_yticks(np.arange(len(order)))
        axes[1].set_yticklabels(order, fontsize=9)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("mean agreement")
        axes[1].set_title("per predicted cell type", fontsize=10)

        return close(fig, savefig)

    def plot_abundance_across_seeds(
        self,
        palette: Optional[Palette] = None,
        figsize: Tuple[float, float] = (9.0, 4.5),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        The composition each seed produces, as one point per seed and cell type, so a cell
        type whose abundance depends on the seed stands out.

        Args:
            palette: Colours to use.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        abundance = self.abundance_per_seed()
        palette = palette or Palette(self.ct_list)
        positions = np.arange(len(self.ct_list))

        fig, ax = plt.subplots(figsize=figsize)
        for offset, (seed, row) in enumerate(abundance.iterrows()):
            jitter = (offset - (len(abundance) - 1) / 2) * 0.12
            ax.scatter(
                positions + jitter,
                row.reindex(self.ct_list).to_numpy(),
                s=28,
                color=palette.color_list(self.ct_list),
                edgecolors="black",
                linewidths=0.4,
                label=str(seed),
            )

        means = abundance[self.ct_list].mean(axis=0).to_numpy()
        ax.plot(positions, means, "_", color="black", markersize=22, markeredgewidth=1.4)

        ax.set_xticks(positions)
        ax.set_xticklabels(self.ct_list, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("fraction of cells")
        ax.set_title(f"composition across {len(abundance)} seeds (dash = mean)", fontsize=11)

        return close(fig, savefig)

    def plot_agreement_map(
        self,
        visualizer: "object",
        window: Union[str, Tuple[Tuple[float, float], Tuple[float, float]]] = "full",
        cmap: str = "viridis",
        point_size: Optional[float] = None,
        max_pixels: int = 4000,
        figsize: Tuple[float, float] = (12.0, 11.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Where on the slide the seeds disagree.

        Args:
            visualizer: A :class:`~hedest.analysis.postseg.SlideVisualizer` for the slide.
            window: ``"full"`` or ``((x, y), (w, h))`` in level-0 coordinates.
            cmap: Colormap for the agreement values.
            point_size: Marker size. Chosen from the cell density when not given.
            max_pixels: Cap on the longest side of the background image.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        region = visualizer.read(window, max_pixels=max_pixels)  # type: ignore[attr-defined]
        agreement = self.agreement()

        cell_ids = np.asarray(visualizer.cell_ids)  # type: ignore[attr-defined]
        centroids = visualizer.centroids  # type: ignore[attr-defined]
        known = pd.Index(cell_ids).isin(agreement.index)
        inside = region.contains(centroids)
        selection = known & inside

        values = agreement.reindex(cell_ids[selection]).to_numpy()
        local = region.to_local(centroids[selection])

        if point_size is None:
            per_cell = (region.image.shape[0] * region.image.shape[1]) / max(int(selection.sum()), 1)
            point_size = float(np.clip(per_cell / 12.0, 0.4, 24.0))

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(region.image, extent=(0, region.image.shape[1], region.image.shape[0], 0))
        scatter = ax.scatter(
            local[:, 0], local[:, 1], c=values, s=point_size, cmap=cmap, vmin=0.0, vmax=1.0, edgecolors="none"
        )
        ax.axis("off")
        ax.set_title(f"seed agreement over {int(selection.sum())} cells", fontsize=11)
        fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02, label=f"share of {len(self.runs)} seeds")

        return close(fig, savefig)


def load_ensemble(run_dir: Union[str, Path], adjusted: bool = True) -> SeedEnsemble:
    """
    Builds an ensemble from a directory of ``seed_*`` runs.

    Args:
        run_dir: The directory holding the seed runs.
        adjusted: Whether to use the adjusted predictions.

    Returns:
        The ensemble.
    """

    from hedest.analysis.loaders import load_seed_runs

    runs = load_seed_runs(run_dir, adjusted=adjusted)
    if not runs:
        raise FileNotFoundError(f"No seed_*/info.pickle found under {run_dir}.")

    return SeedEnsemble(runs)


__all__ = ["SeedEnsemble", "load_ensemble", "prediction_entropy"]
