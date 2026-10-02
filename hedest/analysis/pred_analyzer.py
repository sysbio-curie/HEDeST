"""
Analysing the predictions of a HEDeST run.

:class:`PredAnalyzer` wraps one run (or one aggregate of seeds) and everything optional that
makes it richer: the slide, the segmentation, the spots, the cell crops and, when it exists,
a ground truth.

**Ground truth is optional everywhere.** In practice nobody has per-cell labels, so the
default answer to "is this run any good" is built from the spots: HEDeST is trained to
reproduce the deconvolution proportions, so the per-spot means of its predictions should
match them, and that comparison needs nothing extra. The methods that do need labels say so
by name (:meth:`cell_metrics`, :meth:`plot_confusion_matrix`) and raise a clear error
without them.

Typical use::

    from hedest.analysis import PredAnalyzer

    analyzer = PredAnalyzer(
        "results/my_run",
        seg="seg/slide.json",
        slide_path="slide.tif",
        adata=adata, adata_name="sample",
        mpp=0.2738,
    )
    analyzer.describe()
    analyzer.plot_proportion_scatter()
    analyzer.visualizer().plot_celltype_map()
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from matplotlib.figure import Figure
from scipy.spatial import Delaunay
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from hedest.analysis import plots
from hedest.analysis.loaders import HedestRun
from hedest.analysis.loaders import load_run
from hedest.analysis.palette import Palette
from hedest.analysis.seeds import prediction_entropy
from hedest.spots import cells_in_spots
from hedest.spots import load_seg_dict


class PredAnalyzer:
    """
    Everything that can be said about the predictions of a run.

    Attributes:
        run: The run being analysed.
        predictions: Cells x cell types probability table (adjusted unless asked otherwise).
        ct_list: The cell types, in model-class order.
        palette: The colour of each cell type.
        labels: Predicted cell type per cell.
        confidence: Probability of the predicted cell type, per cell.
        spot_dict: Spot id to the cells it holds.
        proportions: The deconvolution proportions the run was trained on.
    """

    def __init__(
        self,
        run: Union[HedestRun, str, Path, Dict[str, Any]],
        seg: Optional[Union[str, Dict[str, Any]]] = None,
        slide_path: Optional[str] = None,
        adata: Optional[AnnData] = None,
        adata_name: Optional[str] = None,
        image_dict: Optional[Dict[str, Any]] = None,
        mpp: Optional[float] = None,
        ground_truth: Optional[Union[pd.DataFrame, pd.Series, Dict[str, str]]] = None,
        palette: Optional[Palette] = None,
        adjusted: bool = True,
    ) -> None:
        """
        Builds an analyzer around a run.

        Args:
            run: A :class:`~hedest.analysis.loaders.HedestRun`, a path to a run directory or
                to an ``info.pickle``, or the unpickled dictionary itself.
            seg: Segmentation dictionary or path to the HoVer-Net JSON. Needed for anything
                that involves positions, contours or neighbours.
            slide_path: Path to the slide. Needed for the views that show tissue.
            adata: AnnData object of the sample, for the spots.
            adata_name: Key under ``adata.uns['spatial']``.
            image_dict: Cell crops, keyed by cell id, as saved by the segmentation stage.
                Only needed for the crop mosaics.
            mpp: Microns per pixel. Used for real spot diameters and for areas and distances
                in microns.
            ground_truth: Optional per-cell truth, either a cells x cell types table or a
                mapping from cell id to cell type.
            palette: Colours to use. Built from the cell types when not given.
            adjusted: Whether to analyse the adjusted predictions.
        """

        if isinstance(run, (str, Path)):
            run = load_run(run, adjusted=adjusted)
        elif isinstance(run, dict):
            from hedest.analysis.loaders import _from_dict

            run = _from_dict(run, None, adjusted)
        elif adjusted != run.adjusted:
            run = run.use(adjusted)

        self.run = run
        self.predictions = run.predictions
        self.ct_list = run.ct_list
        self.palette = palette or Palette(self.ct_list)
        missing = [name for name in self.ct_list if name not in self.palette]
        if missing:
            logger.warning(f"The palette has no colour for {missing}; those cells will be drawn grey.")
        self.spot_dict = run.spot_dict
        self.proportions = run.proportions
        self.mpp = mpp

        self.slide_path = slide_path
        self.adata = adata
        self.adata_name = adata_name
        self.image_dict = image_dict

        self.seg_dict: Optional[Dict[str, Any]] = load_seg_dict(seg) if seg is not None else None

        self.labels = self.predictions.idxmax(axis=1).rename("cell_type")
        self.confidence = self.predictions.max(axis=1).rename("confidence")

        self.true_labels: Optional[pd.Series] = None
        self.ground_truth: Optional[pd.DataFrame] = None
        if ground_truth is not None:
            self.set_ground_truth(ground_truth)

        # Filled on demand, because each is expensive on a slide with 10^5 cells.
        self._predicted_proportions: Optional[pd.DataFrame] = None
        self._centroids: Optional[pd.DataFrame] = None
        self._areas: Optional[pd.Series] = None
        self._edge_cache: Optional[np.ndarray] = None
        self._edges_key: Optional[Tuple[Any, Any]] = None

        train_spots = set(run.train_spot_dict or {})
        self.train_spots = sorted(train_spots)
        self.held_out_spots = sorted(set(self.spot_dict) - train_spots)
        self.train_cells = sorted(cells_in_spots(run.train_spot_dict or {}))
        self.held_out_cells = sorted(cells_in_spots(self.spot_dict) - set(self.train_cells))

    def __repr__(self) -> str:
        extras = [
            name
            for name, value in (
                ("seg", self.seg_dict),
                ("slide", self.slide_path),
                ("adata", self.adata),
                ("crops", self.image_dict),
                ("ground_truth", self.true_labels),
            )
            if value is not None
        ]

        return (
            f"PredAnalyzer({len(self.predictions)} cells, {len(self.ct_list)} cell types, "
            f"{len(self.spot_dict)} spots, adjusted={self.run.adjusted}, "
            f"attached=[{', '.join(extras) or 'nothing'}])"
        )

    # ------------------------------------------------------------------ inputs

    def set_ground_truth(self, ground_truth: Union[pd.DataFrame, pd.Series, Dict[str, str]]) -> "PredAnalyzer":
        """
        Attaches a per-cell ground truth.

        Args:
            ground_truth: Either a cells x cell types table, whose row-wise maximum gives
                the true cell type, or a mapping from cell id to cell type.

        Returns:
            The analyzer, so calls can be chained.
        """

        if isinstance(ground_truth, pd.DataFrame):
            self.ground_truth = ground_truth
            self.true_labels = ground_truth.idxmax(axis=1).rename("cell_type")
        else:
            series = pd.Series(ground_truth) if isinstance(ground_truth, dict) else ground_truth
            self.ground_truth = None
            self.true_labels = series.astype(str).rename("cell_type")

        shared = self.predictions.index.intersection(self.true_labels.index)
        logger.info(f"Ground truth attached for {len(shared)}/{len(self.predictions)} predicted cells.")
        unknown = set(self.true_labels.loc[shared].unique()) - set(self.ct_list)
        if unknown:
            logger.warning(f"Ground-truth cell types absent from the model: {sorted(unknown)}")

        return self

    @property
    def has_ground_truth(self) -> bool:
        """Whether a per-cell ground truth is attached."""

        return self.true_labels is not None

    def _require(self, **attributes: Any) -> None:
        """
        Raises a readable error when an optional input is missing.

        Args:
            **attributes: The attributes to check, by the name to show the user.
        """

        missing = [name for name, value in attributes.items() if value is None]
        if missing:
            raise ValueError(
                f"This needs {', '.join(missing)}, which was not given to PredAnalyzer. "
                "Pass it to the constructor, or set the attribute on the analyzer."
            )

    # ------------------------------------------------------------------ derived tables

    @property
    def predicted_proportions(self) -> pd.DataFrame:
        """
        The mean prediction over the cells of each spot: what HEDeST is trained to match.

        Returns:
            Spots x cell types table, restricted to the spots holding at least one
            predicted cell.
        """

        if self._predicted_proportions is None:
            spot_ids = list(self.spot_dict)
            counts = [len(self.spot_dict[spot_id]) for spot_id in spot_ids]
            cells = np.concatenate([np.asarray(self.spot_dict[s], dtype=object) for s in spot_ids]) if spot_ids else []
            owners = np.repeat(np.asarray(spot_ids, dtype=object), counts)

            rows = self.predictions.index.get_indexer(cells)
            keep = rows >= 0
            if not keep.all():
                logger.info(f"{int((~keep).sum())} cells of spot_dict have no prediction and are ignored.")

            values = self.predictions.to_numpy()[rows[keep]]
            frame = pd.DataFrame(values, columns=self.ct_list)
            frame["__spot__"] = owners[keep]
            grouped = frame.groupby("__spot__", sort=False).mean()
            grouped.index.name = "spot"
            self._predicted_proportions = grouped

        return self._predicted_proportions

    @property
    def centroids(self) -> pd.DataFrame:
        """
        Nucleus centroids of the predicted cells.

        Returns:
            A frame indexed by cell id with ``x`` and ``y`` columns.
        """

        if self._centroids is None:
            self._require(seg=self.seg_dict)
            nuc = self.seg_dict["nuc"]  # type: ignore[index]
            cells = [cell_id for cell_id in self.predictions.index if cell_id in nuc]
            coordinates = np.asarray([nuc[cell_id]["centroid"] for cell_id in cells], dtype="float64")
            self._centroids = pd.DataFrame(coordinates, index=pd.Index(cells, name="cell_id"), columns=["x", "y"])

        return self._centroids

    def cell_areas(self, in_microns: bool = True) -> pd.Series:
        """
        Nucleus area of every predicted cell.

        Args:
            in_microns: Whether to convert to µm² using ``mpp``. Falls back to pixels, with
                a warning, when no mpp is known.

        Returns:
            The areas, indexed by cell id.
        """

        if self._areas is None:
            self._require(seg=self.seg_dict)
            nuc = self.seg_dict["nuc"]  # type: ignore[index]
            cells = [cell_id for cell_id in self.predictions.index if cell_id in nuc]
            areas = plots.polygon_areas([nuc[cell_id]["contour"] for cell_id in cells])
            self._areas = pd.Series(areas, index=pd.Index(cells, name="cell_id"), name="area_px2")

        if not in_microns:
            return self._areas

        if self.mpp is None:
            logger.warning("No mpp given, so areas stay in squared pixels.")
            return self._areas

        return (self._areas * self.mpp**2).rename("area_um2")

    def label_table(self) -> pd.DataFrame:
        """
        One row per cell: predicted cell type, its probability, and whatever else is known.

        Returns:
            A frame indexed by cell id, with the centroid when a segmentation is attached,
            the entropy of the prediction, the seed agreement for an aggregate, and the true
            cell type when there is one.
        """

        table = pd.DataFrame(
            {
                "cell_type": self.labels,
                "confidence": self.confidence,
                "entropy": prediction_entropy(self.predictions),
            }
        )

        if self.run.agreement is not None:
            table["seed_agreement"] = self.run.agreement.reindex(table.index)
        if self.seg_dict is not None:
            table = table.join(self.centroids)
        if self.true_labels is not None:
            table["true_cell_type"] = self.true_labels.reindex(table.index)
            table["correct"] = table["true_cell_type"] == table["cell_type"]

        return table

    def spot_membership(self) -> pd.Series:
        """
        Whether each predicted cell falls inside a spot.

        Returns:
            A boolean series indexed by cell id.
        """

        inside = cells_in_spots(self.spot_dict)

        return pd.Series(self.predictions.index.isin(inside), index=self.predictions.index, name="in_spot")

    # ------------------------------------------------------------------ summaries

    def describe(self) -> pd.DataFrame:
        """
        A compact recap of the run, for the top of a notebook.

        Returns:
            A one-column frame of run properties.
        """

        inside = int(self.spot_membership().sum())
        rows = {
            "run": self.run.name,
            "cells": f"{len(self.predictions):,}",
            "cells in spots": f"{inside:,} ({inside / max(len(self.predictions), 1):.1%})",
            "cell types": len(self.ct_list),
            "spots with cells": f"{len(self.spot_dict):,}",
            "spots in proportions": f"{len(self.proportions):,}",
            "predictions": "adjusted (PPSA)" if self.run.adjusted else "raw",
            "seeds": self.run.n_seeds if self.run.is_aggregate else self.run.params.get("rs", "?"),
        }
        for key in (
            "hidden_dims",
            "norm",
            "dropout",
            "lr",
            "alpha",
            "beta",
            "divergence",
            "epochs",
            "adjustment",
            "gated",
        ):
            if key in self.run.params:
                rows[key] = self.run.params[key]
        if self.mpp is not None:
            rows["mpp"] = f"{self.mpp:.4f}"
        if self.run.agreement is not None:
            rows["mean seed agreement"] = f"{self.run.agreement.mean():.3f}"
        if self.has_ground_truth:
            rows["ground truth"] = f"{len(self.true_labels):,} cells"  # type: ignore[arg-type]

        # Everything is rendered as text: the values are of mixed types, and a mixed object
        # column comes back from a spreadsheet with False turned into 0.
        return pd.DataFrame.from_dict(
            {key: str(value) for key, value in rows.items()}, orient="index", columns=["value"]
        )

    def celltype_summary(self) -> pd.DataFrame:
        """
        Per cell type: how many cells were assigned to it, how confident those calls are,
        and how its predicted abundance compares to the deconvolution.

        Returns:
            A frame indexed by cell type, in model-class order.
        """

        counts = self.labels.value_counts()
        mean_spot_proportion = self.proportions.mean(axis=0)
        rows = []

        for cell_type in self.ct_list:
            selection = self.labels == cell_type
            probabilities = self.confidence[selection]
            all_probabilities = self.predictions[cell_type]
            rows.append(
                {
                    "cell_type": cell_type,
                    "n_cells": int(counts.get(cell_type, 0)),
                    "fraction": float(counts.get(cell_type, 0) / max(len(self.labels), 1)),
                    "deconv_fraction": float(mean_spot_proportion.get(cell_type, np.nan)),
                    "min_prob": float(probabilities.min()) if selection.any() else np.nan,
                    "median_prob": float(probabilities.median()) if selection.any() else np.nan,
                    "mean_prob": float(probabilities.mean()) if selection.any() else np.nan,
                    "max_prob": float(probabilities.max()) if selection.any() else np.nan,
                    "mean_prob_all_cells": float(all_probabilities.mean()),
                }
            )

        table = pd.DataFrame(rows).set_index("cell_type")
        if self.run.agreement is not None:
            table["mean_agreement"] = [
                float(self.run.agreement[self.labels == ct].mean()) if (self.labels == ct).any() else np.nan
                for ct in self.ct_list
            ]

        return table

    def spot_metrics(self, subset: str = "all") -> pd.DataFrame:
        """
        How well the per-spot means of the predictions reproduce the deconvolution.

        This is the headline evaluation when there is no ground truth. Note that the spots
        used for training are fitted by construction, so ``subset="held_out"`` is the honest
        number.

        Args:
            subset: ``"all"``, ``"train"`` or ``"held_out"``.

        Returns:
            A frame indexed by cell type with Pearson r, Spearman rho, MSE and MAE, plus a
            ``__global__`` row holding the means.
        """

        predicted = self.predicted_proportions
        if subset == "train":
            predicted = predicted.reindex(self.train_spots).dropna(how="all")
        elif subset == "held_out":
            predicted = predicted.reindex(self.held_out_spots).dropna(how="all")
        elif subset != "all":
            raise ValueError(f"subset must be 'all', 'train' or 'held_out', got '{subset}'.")

        truth, predicted = self.proportions.align(predicted, join="inner", axis=0)
        truth, predicted = truth.align(predicted, join="inner", axis=1)

        if truth.empty:
            raise ValueError(f"No spot left for subset='{subset}'.")

        rows = []
        for cell_type in truth.columns:
            x = truth[cell_type].to_numpy()
            y = predicted[cell_type].to_numpy()
            constant = x.std() == 0 or y.std() == 0
            rows.append(
                {
                    "cell_type": cell_type,
                    "pearson": np.nan if constant else float(pearsonr(x, y)[0]),
                    "spearman": np.nan if constant else float(spearmanr(x, y)[0]),
                    "mse": float(np.mean((x - y) ** 2)),
                    "mae": float(np.mean(np.abs(x - y))),
                }
            )

        table = pd.DataFrame(rows).set_index("cell_type")
        table.loc["__global__"] = table.mean(numeric_only=True)
        table.attrs["n_spots"] = len(truth)
        table.attrs["subset"] = subset

        return table

    def cell_metrics(self, subset: str = "all", per_class: bool = True) -> Dict[str, Any]:
        """
        Cell-level accuracy against a ground truth.

        Args:
            subset: ``"all"``, ``"train"`` or ``"held_out"``.
            per_class: Whether to add the per-cell-type scores.

        Returns:
            A dictionary of metrics; the per-class entries are pandas objects.

        Raises:
            ValueError: If no ground truth is attached.
        """

        if self.true_labels is None:
            raise ValueError(
                "Cell-level metrics need a per-cell ground truth, which most datasets do not have. "
                "Attach one with set_ground_truth(), or use spot_metrics() instead."
            )

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        cells = self._subset_cells(subset)
        shared = [cell for cell in cells if cell in self.true_labels.index]
        if not shared:
            raise ValueError(f"No cell of subset='{subset}' has a ground-truth label.")

        truth = self.true_labels.loc[shared]
        predicted = self.labels.loc[shared]

        metrics: Dict[str, Any] = {
            "n_cells": len(shared),
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
            "weighted_precision": float(precision_score(truth, predicted, average="weighted", zero_division=0)),
            "weighted_recall": float(recall_score(truth, predicted, average="weighted", zero_division=0)),
        }

        if per_class:
            classes = sorted(set(truth.unique()) | set(predicted.unique()))
            metrics["per_class"] = pd.DataFrame(
                {
                    "f1": f1_score(truth, predicted, average=None, labels=classes, zero_division=0),
                    "precision": precision_score(truth, predicted, average=None, labels=classes, zero_division=0),
                    "recall": recall_score(truth, predicted, average=None, labels=classes, zero_division=0),
                    "support": [int((truth == cls).sum()) for cls in classes],
                },
                index=pd.Index(classes, name="cell_type"),
            )
            metrics["confusion_matrix"] = pd.DataFrame(
                confusion_matrix(truth, predicted, labels=classes), index=classes, columns=classes
            )

        return metrics

    def _subset_cells(self, subset: str) -> List[str]:
        """
        Resolves a cell subset name.

        Args:
            subset: ``"all"``, ``"train"`` or ``"held_out"``.

        Returns:
            The cell ids.
        """

        if subset == "all":
            return list(self.predictions.index)
        if subset == "train":
            return [cell for cell in self.train_cells if cell in self.predictions.index]
        if subset == "held_out":
            return [cell for cell in self.held_out_cells if cell in self.predictions.index]

        raise ValueError(f"subset must be 'all', 'train' or 'held_out', got '{subset}'.")

    # ------------------------------------------------------------------ plots

    def plot_history(self, savefig: Optional[str] = None) -> Figure:
        """
        The training and validation loss of the run.

        Args:
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if self.run.history is None:
            raise ValueError("This run carries no loss history (an aggregate of seeds never does).")

        return plots.plot_history(self.run.history["train"], self.run.history["val"], savefig=savefig)

    def plot_abundance(self, savefig: Optional[str] = None, **kwargs: Any) -> Figure:
        """
        Predicted composition of the slide against the mean deconvolution proportions.

        Args:
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_abundance`.

        Returns:
            The figure.
        """

        counts = self.labels.value_counts(normalize=True).reindex(self.ct_list).fillna(0.0)
        reference = self.proportions.mean(axis=0).reindex(self.ct_list)

        return plots.plot_abundance(counts, reference=reference, palette=self.palette, savefig=savefig, **kwargs)

    def plot_proportion_scatter(self, subset: str = "all", savefig: Optional[str] = None, **kwargs: Any) -> Figure:
        """
        Predicted against deconvoluted proportion, one panel per cell type.

        Args:
            subset: ``"all"``, ``"train"`` or ``"held_out"``.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_proportion_scatter`.

        Returns:
            The figure.
        """

        predicted = self.predicted_proportions
        if subset == "train":
            predicted = predicted.reindex(self.train_spots).dropna(how="all")
        elif subset == "held_out":
            predicted = predicted.reindex(self.held_out_spots).dropna(how="all")

        return plots.plot_proportion_scatter(
            self.proportions, predicted, palette=self.palette, savefig=savefig, **kwargs
        )

    def plot_probability_histograms(
        self, compare_to_truth: bool = False, savefig: Optional[str] = None, **kwargs: Any
    ) -> Figure:
        """
        Distribution of the predicted probabilities, per cell type.

        Args:
            compare_to_truth: Whether to split each histogram by the ground truth.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_probability_histograms`.

        Returns:
            The figure.
        """

        truth = None
        if compare_to_truth:
            if self.true_labels is None:
                raise ValueError("compare_to_truth needs a ground truth; call set_ground_truth() first.")
            truth = self.true_labels

        return plots.plot_probability_histograms(
            self.predictions, truth=truth, palette=self.palette, savefig=savefig, **kwargs
        )

    def plot_confidence(
        self, bins: int = 60, figsize: Tuple[float, float] = (11.0, 4.0), savefig: Optional[str] = None
    ) -> Figure:
        """
        How decided the predictions are: the probability of the winning class, the entropy
        of the whole vector, and the seed agreement when the run is an aggregate.

        Args:
            bins: Number of histogram bins.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        panels = 3 if self.run.agreement is not None else 2
        fig, axes = plt.subplots(1, panels, figsize=(figsize[0] * panels / 3, figsize[1]))

        axes[0].hist(self.confidence.to_numpy(), bins=bins, color="#4c72b0")
        axes[0].set_xlabel("probability of the predicted type")
        axes[0].set_ylabel("cells")
        axes[0].set_title(f"median {self.confidence.median():.2f}", fontsize=10)

        entropy = prediction_entropy(self.predictions)
        axes[1].hist(entropy.to_numpy(), bins=bins, color="#55a868")
        axes[1].set_xlabel("normalised entropy")
        axes[1].set_title(f"median {entropy.median():.2f}", fontsize=10)

        if self.run.agreement is not None:
            agreement = self.run.agreement
            levels = sorted(agreement.unique())
            axes[2].bar(
                [f"{level:.2f}" for level in levels], [float((agreement == x).mean()) for x in levels], color="#c44e52"
            )
            axes[2].set_xlabel(f"share of the {self.run.n_seeds} seeds agreeing")
            axes[2].set_title(f"unanimous on {float((agreement == 1.0).mean()):.1%} of cells", fontsize=10)

        return plots.close(fig, savefig)

    def plot_adjustment_effect(
        self, figsize: Tuple[float, float] = (11.0, 4.5), savefig: Optional[str] = None
    ) -> Figure:
        """
        What Prior Probability Shift Adjustment changed: the composition before and after,
        and how many cells switched cell type.

        Args:
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        if self.run.predictions_adjusted is None:
            raise ValueError("This run holds no adjusted predictions, so there is nothing to compare.")

        raw_labels = self.run.predictions_raw.idxmax(axis=1)
        adjusted_labels = self.run.predictions_adjusted.idxmax(axis=1)
        shared = raw_labels.index.intersection(adjusted_labels.index)
        raw_labels, adjusted_labels = raw_labels.loc[shared], adjusted_labels.loc[shared]

        raw_fraction = raw_labels.value_counts(normalize=True).reindex(self.ct_list).fillna(0.0)
        adjusted_fraction = adjusted_labels.value_counts(normalize=True).reindex(self.ct_list).fillna(0.0)
        changed = float((raw_labels != adjusted_labels).mean())

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        positions = np.arange(len(self.ct_list))
        width = 0.4
        axes[0].bar(positions - width / 2, raw_fraction.to_numpy(), width, label="raw", color="#8c8c8c")
        axes[0].bar(
            positions + width / 2,
            adjusted_fraction.to_numpy(),
            width,
            label="adjusted",
            color=self.palette.color_list(self.ct_list),
        )
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(self.ct_list, rotation=40, ha="right", fontsize=8)
        axes[0].set_ylabel("fraction of cells")
        axes[0].legend(frameon=False, fontsize=9)
        axes[0].set_title(f"{changed:.1%} of cells change type", fontsize=10)

        flow = (
            pd.crosstab(raw_labels, adjusted_labels, normalize="index")
            .reindex(index=self.ct_list, columns=self.ct_list)
            .fillna(0.0)
        )
        image = axes[1].imshow(flow.to_numpy(), cmap="magma_r", vmin=0, vmax=1)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(self.ct_list, rotation=40, ha="right", fontsize=7)
        axes[1].set_yticks(positions)
        axes[1].set_yticklabels(self.ct_list, fontsize=7)
        axes[1].set_xlabel("adjusted")
        axes[1].set_ylabel("raw")
        fig.colorbar(image, ax=axes[1], fraction=0.046, label="share of the raw class")

        return plots.close(fig, savefig)

    def plot_cell_mosaic(
        self,
        spot_id: Optional[str] = None,
        cell_ids: Optional[Sequence[str]] = None,
        num_cols: int = 8,
        show_probs: bool = True,
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        The crops of the cells of one spot, titled with their predicted cell type.

        Args:
            spot_id: The spot to show. A random one is picked when neither this nor
                ``cell_ids`` is given.
            cell_ids: Explicit list of cells to show, instead of a spot.
            num_cols: Number of columns.
            show_probs: Whether to print the probability under each label.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        self._require(image_dict=self.image_dict)

        if cell_ids is None:
            if spot_id is None:
                spot_id = str(np.random.choice(list(self.spot_dict)))
                logger.info(f"Randomly selected spot {spot_id}.")
            if spot_id not in self.spot_dict:
                raise ValueError(f"Spot {spot_id} holds no cell.")
            cell_ids = self.spot_dict[spot_id]

        known = [cell for cell in cell_ids if cell in self.labels.index]

        return plots.plot_cell_mosaic(
            self.image_dict,  # type: ignore[arg-type]
            known,
            labels=self.labels.loc[known].to_dict(),
            true_labels=None if self.true_labels is None else self.true_labels.reindex(known).dropna().to_dict(),
            probs=self.confidence.loc[known].to_dict() if show_probs else None,
            num_cols=num_cols,
            suptitle=None if spot_id is None else f"spot {spot_id} — {len(known)} cells",
            savefig=savefig,
        )

    def plot_celltype_grid(
        self,
        cell_type: Optional[str] = None,
        num_rows: int = 8,
        num_cols: int = 8,
        selection: str = "max",
        cell_size: float = 0.32,
        mosaic: Optional[Tuple[int, int]] = None,
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        A mosaic of galleries, one per predicted cell type.

        Each cell type gets a ``num_rows x num_cols`` block of crops that touch, so the
        morphologies can be compared at a glance, and the blocks are laid out on a grid
        whose shape is chosen from their number to keep the figure roughly landscape.

        Args:
            cell_type: The cell type to show. All of them, as a mosaic, when None.
            num_rows: Number of rows of crops per cell type.
            num_cols: Number of columns of crops per cell type.
            selection: ``"max"`` for the most confident cells, ``"random"`` for a sample.
            cell_size: Side of one crop, in inches.
            mosaic: ``(rows, columns)`` of cell types. Chosen internally when None.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        self._require(image_dict=self.image_dict)
        cell_types = self.ct_list if cell_type is None else [cell_type]
        n = max(1, num_rows) * max(1, num_cols)

        blocks: Dict[str, List[str]] = {}
        for candidate in cell_types:
            cells = self.labels.index[self.labels == candidate]
            cells = [cell for cell in cells if cell in self.image_dict]  # type: ignore[operator]
            if not cells:
                continue
            probabilities = self.confidence.loc[cells]
            if selection == "max":
                chosen = list(probabilities.nlargest(n).index)
            elif selection == "random":
                size = min(n, len(cells))
                chosen = list(pd.Series(cells).sample(n=size, random_state=0))
            else:
                raise ValueError(f"selection must be 'max' or 'random', got '{selection}'.")
            blocks[candidate] = chosen

        if not blocks:
            raise ValueError("None of the requested cell types has a cell with a crop.")

        counts = self.labels.value_counts()

        return plots.plot_gallery(
            self.image_dict,  # type: ignore[arg-type]
            blocks,
            num_rows=num_rows,
            num_cols=num_cols,
            cell_size=cell_size,
            mosaic=mosaic,
            palette=self.palette,
            counts={name: int(counts.get(name, 0)) for name in blocks},
            savefig=savefig,
        )

    def plot_spot_report(
        self,
        spot_id: Optional[str] = None,
        draw_seg: bool = True,
        figsize: Tuple[float, float] = (15.0, 8.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        One spot, four ways: the tissue, the mean predicted probabilities, the predicted
        composition and the deconvolution it is meant to match.

        Args:
            spot_id: The spot to show. A random one is picked when None.
            draw_seg: Whether to draw the nuclei on the tissue panel.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        self._require(slide_path=self.slide_path, adata=self.adata)

        if spot_id is None:
            spot_id = str(np.random.choice(list(self.spot_dict)))
            logger.info(f"Randomly selected spot {spot_id}.")
        if spot_id not in self.spot_dict:
            raise ValueError(f"Spot {spot_id} holds no cell.")

        cells = [cell for cell in self.spot_dict[spot_id] if cell in self.predictions.index]
        predictions = self.predictions.loc[cells]

        viewer = self.visualizer(with_labels=draw_seg)
        tissue = viewer.plot_spot(spot_id=spot_id, draw_seg=draw_seg, title="", legend=False, figsize=(7, 7))

        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(2, 3, width_ratios=[1.6, 1, 1], figure=fig)

        ax_image = fig.add_subplot(grid[:, 0])
        ax_image.imshow(_figure_to_array(tissue))
        ax_image.axis("off")
        ax_image.set_title(f"spot {spot_id} — {len(cells)} cells", fontsize=11)

        plots.plot_pie_chart(
            fig.add_subplot(grid[0, 1]), predictions.mean(axis=0), self.palette, title="mean predicted probability"
        )
        plots.plot_pie_chart(
            fig.add_subplot(grid[0, 2]),
            self.labels.loc[cells].value_counts(normalize=True),
            self.palette,
            title="predicted composition",
        )
        if spot_id in self.proportions.index:
            plots.plot_pie_chart(
                fig.add_subplot(grid[1, 1]), self.proportions.loc[spot_id], self.palette, title="deconvolution"
            )
        plots.plot_legend(
            self.palette,
            names=[ct for ct in self.ct_list if ct in set(self.labels.loc[cells])],
            ax=fig.add_subplot(grid[1, 2]),
            fontsize=9,
        )

        return plots.close(fig, savefig)

    def plot_colocalization_matrix(
        self, source: str = "predicted", method: str = "pearson", savefig: Optional[str] = None, **kwargs: Any
    ) -> Figure:
        """
        Correlation of cell-type proportions across spots: which types share spots.

        Args:
            source: ``"predicted"`` for the HEDeST proportions, ``"deconvolution"`` for the
                input ones.
            method: Correlation method, passed to ``DataFrame.corr``.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_matrix`.

        Returns:
            The figure.
        """

        if source == "predicted":
            frame = self.predicted_proportions
        elif source == "deconvolution":
            frame = self.proportions
        else:
            raise ValueError(f"source must be 'predicted' or 'deconvolution', got '{source}'.")

        matrix = frame.reindex(columns=self.ct_list).corr(method=method)
        kwargs.setdefault("annot", len(matrix) <= 20)
        kwargs.setdefault("title", f"colocalization across spots ({source})")

        return plots.plot_matrix(matrix, vmin=-1, vmax=1, cbar_label=f"{method} correlation", savefig=savefig, **kwargs)

    def plot_confusion_matrix(
        self, subset: str = "all", normalize: bool = True, savefig: Optional[str] = None, **kwargs: Any
    ) -> Figure:
        """
        Confusion matrix against the ground truth.

        Args:
            subset: ``"all"``, ``"train"`` or ``"held_out"``.
            normalize: Whether to normalise each true-class row to 1.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_matrix`.

        Returns:
            The figure.
        """

        metrics = self.cell_metrics(subset=subset, per_class=True)
        matrix = metrics["confusion_matrix"].astype("float64")
        if normalize:
            matrix = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0)

        kwargs.setdefault("cmap", "magma_r")
        kwargs.setdefault("annot", len(matrix) <= 20)
        kwargs.setdefault("title", f"balanced accuracy {metrics['balanced_accuracy']:.3f} ({subset})")
        kwargs.setdefault("xlabel", "predicted")
        kwargs.setdefault("ylabel", "ground truth")

        return plots.plot_matrix(
            matrix, vmin=0, vmax=1 if normalize else None, cbar_label="share of true class", savefig=savefig, **kwargs
        )

    # ------------------------------------------------------------------ neighbourhoods

    def _edges(self, max_distance: Optional[float] = None, compute_dist: str = "centroid") -> np.ndarray:
        """
        The Delaunay edges between predicted cells, as indices into :attr:`centroids`.

        Cached, and computed with array operations throughout: on a slide with 10^5 nuclei
        the triangulation holds of the order of 10^6 edges, which is too many for a Python
        loop.

        Args:
            max_distance: Drops edges longer than this, in pixels.
            compute_dist: ``"centroid"`` to measure between centroids, ``"contour"`` to
                measure between the closest points of the two contours, which is far slower.

        Returns:
            An ``(n_edges, 2)`` array of index pairs, each pair sorted.
        """

        key = (max_distance, compute_dist)
        if self._edge_cache is not None and self._edges_key == key:
            return self._edge_cache

        coordinates = self.centroids.to_numpy()
        simplices = Delaunay(coordinates).simplices
        pairs = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
        pairs = np.sort(pairs, axis=1)
        edges = np.unique(pairs, axis=0)

        if max_distance is not None:
            if compute_dist == "centroid":
                lengths = np.linalg.norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1)
            elif compute_dist == "contour":
                nuc = self.seg_dict["nuc"]  # type: ignore[index]
                cell_ids = list(self.centroids.index)
                lengths = np.array(
                    [
                        np.min(
                            np.linalg.norm(
                                np.asarray(nuc[cell_ids[a]]["contour"], dtype="float64")[:, None, :]
                                - np.asarray(nuc[cell_ids[b]]["contour"], dtype="float64")[None, :, :],
                                axis=-1,
                            )
                        )
                        for a, b in edges
                    ]
                )
            else:
                raise ValueError(f"compute_dist must be 'centroid' or 'contour', got '{compute_dist}'.")
            edges = edges[lengths <= max_distance]

        self._edge_cache = edges
        self._edges_key = key
        logger.info(f"Delaunay graph: {len(self.centroids)} cells, {len(edges)} edges.")

        return edges

    def neighbors(self, max_distance: Optional[float] = None, compute_dist: str = "centroid") -> Dict[str, List[str]]:
        """
        The Delaunay neighbours of every predicted cell.

        Args:
            max_distance: Drops edges longer than this, in pixels. Use it: without a cut-off
                the triangulation joins cells across empty tissue.
            compute_dist: ``"centroid"`` to measure between centroids, ``"contour"`` to
                measure the closest points of the two contours, which is much slower.

        Returns:
            A mapping from cell id to the list of its neighbours.
        """

        edges = self._edges(max_distance=max_distance, compute_dist=compute_dist)
        cell_ids = np.asarray(self.centroids.index)

        neighbors: Dict[str, List[str]] = defaultdict(list)
        for a, b in zip(cell_ids[edges[:, 0]], cell_ids[edges[:, 1]]):
            neighbors[a].append(b)
            neighbors[b].append(a)

        return dict(neighbors)

    def _edge_labels(self, max_distance: Optional[float] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        The predicted cell type at each end of every edge.

        Args:
            max_distance: Passed to :meth:`_edges`.
            **kwargs: Passed to :meth:`_edges`.

        Returns:
            Two arrays of cell type names, one per endpoint.
        """

        edges = self._edges(max_distance=max_distance, **kwargs)
        labels = self.labels.reindex(self.centroids.index).to_numpy()

        return labels[edges[:, 0]], labels[edges[:, 1]]

    def neighborhood_composition(
        self, max_distance: Optional[float] = None, normalize: bool = True, **kwargs: Any
    ) -> pd.DataFrame:
        """
        The average neighbourhood of each cell type.

        Args:
            max_distance: Passed to :meth:`neighbors`.
            normalize: True for the share of the neighbours of each type, False for the
                average count.
            **kwargs: Passed to :meth:`neighbors`.

        Returns:
            A frame whose rows are the cell type of the cell and whose columns are the cell
            types of its neighbours.
        """

        source, target = self._edge_labels(max_distance=max_distance, **kwargs)
        # Every edge counts in both directions, since it is a neighbour of each endpoint.
        counts = (
            pd.crosstab(
                pd.Series(np.concatenate([source, target]), name="cell"),
                pd.Series(np.concatenate([target, source]), name="neighbour"),
            )
            .reindex(index=self.ct_list, columns=self.ct_list)
            .fillna(0.0)
        )

        n_cells = self.labels.value_counts().reindex(self.ct_list).fillna(0)
        table = counts.div(n_cells.replace(0, np.nan), axis=0)
        if normalize:
            table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)

        return table

    def plot_neighborhood_composition(
        self, max_distance: Optional[float] = None, savefig: Optional[str] = None, **kwargs: Any
    ) -> Figure:
        """
        The neighbourhood composition as a heatmap.

        Args:
            max_distance: Passed to :meth:`neighbors`.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_matrix`.

        Returns:
            The figure.
        """

        matrix = self.neighborhood_composition(max_distance=max_distance)
        kwargs.setdefault("cmap", "magma_r")
        kwargs.setdefault("annot", len(matrix) <= 20)
        kwargs.setdefault("title", "neighbourhood composition")
        kwargs.setdefault("xlabel", "cell type of the neighbours")
        kwargs.setdefault("ylabel", "cell type of the cell")

        return plots.plot_matrix(matrix, vmin=0, cbar_label="share of neighbours", savefig=savefig, **kwargs)

    def neighbor_distances(self, max_distance: Optional[float] = None) -> pd.DataFrame:
        """
        Mean distance between neighbouring cells, per pair of cell types.

        Args:
            max_distance: Passed to :meth:`neighbors`.

        Returns:
            A symmetric frame, in µm when ``mpp`` is known and in pixels otherwise.
        """

        edges = self._edges(max_distance=max_distance)
        coordinates = self.centroids.to_numpy()
        lengths = np.linalg.norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1)
        lengths = lengths * (self.mpp if self.mpp is not None else 1.0)
        source, target = self._edge_labels(max_distance=max_distance)

        frame = pd.DataFrame({"a": source, "b": target, "distance": lengths}).dropna()
        # Unordered pairs: sort the two ends so (A, B) and (B, A) land in the same group.
        low = np.minimum(frame["a"], frame["b"])
        high = np.maximum(frame["a"], frame["b"])
        means = frame.groupby([low, high])["distance"].mean()

        matrix = pd.DataFrame(np.nan, index=self.ct_list, columns=self.ct_list, dtype="float64")
        for (type_a, type_b), value in means.items():
            matrix.loc[type_a, type_b] = value
            matrix.loc[type_b, type_a] = value

        return matrix

    def plot_neighbor_distances(
        self, max_distance: Optional[float] = None, savefig: Optional[str] = None, **kwargs: Any
    ) -> Figure:
        """
        Mean neighbour distance per pair of cell types, as a heatmap.

        Args:
            max_distance: Passed to :meth:`neighbors`.
            savefig: Path to write the figure to.
            **kwargs: Passed to :func:`hedest.analysis.plots.plot_matrix`.

        Returns:
            The figure.
        """

        matrix = self.neighbor_distances(max_distance=max_distance)
        unit = "µm" if self.mpp is not None else "px"
        kwargs.setdefault("mask_upper", True)
        kwargs.setdefault("annot", len(matrix) <= 20)
        kwargs.setdefault("fmt", ".0f" if self.mpp is not None else ".1f")
        kwargs.setdefault("title", "mean distance between neighbours")

        return plots.plot_matrix(matrix, cbar_label=f"mean distance ({unit})", savefig=savefig, **kwargs)

    def plot_colocalization_graph(
        self,
        max_distance: Optional[float] = None,
        min_threshold: float = 0.05,
        curvature: float = 0.25,
        self_loops: bool = True,
        fontsize: int = 11,
        figsize: Tuple[float, float] = (8.0, 8.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Neighbourhood as a circular graph: node size is the number of cells, an arrow from A
        to B means that cells of type A have B among their neighbours.

        A loop on a node is the share of a cell type's neighbours that are of its own type,
        which is usually the largest term by far: without it a cell type that sits among its
        own kind, such as a ciliated epithelium lining a lumen, looks disconnected when in
        fact it is the most clustered of all.

        Args:
            max_distance: Passed to :meth:`neighbors`.
            min_threshold: Hides arrows below this share of neighbours.
            curvature: Curvature of the arrows, so A to B and B to A stay distinguishable.
            self_loops: Whether to draw the diagonal of the composition as a loop on each
                node, labelled with its share.
            fontsize: Size of the cell-type labels.
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure.
        """

        from matplotlib.patches import Circle
        from matplotlib.patches import FancyArrowPatch

        composition = self.neighborhood_composition(max_distance=max_distance)
        counts = self.labels.value_counts()
        present = [ct for ct in self.ct_list if counts.get(ct, 0) > 0]
        if not present:
            raise ValueError("No predicted cell to draw.")

        angles = np.linspace(0, 2 * np.pi, len(present), endpoint=False)
        positions = {ct: np.array([np.cos(angle), np.sin(angle)]) for ct, angle in zip(present, angles)}
        largest = float(counts.reindex(present).max())

        # Nodes are drawn as patches rather than scatter markers, so their radius is known
        # in data units: the arrows, the loops and the labels are all placed from it.
        radii = {ct: 0.05 + 0.10 * float(np.sqrt(counts.get(ct, 0) / largest)) for ct in present}
        loop_radius = 0.13
        label_radius = 1.0 + max(radii.values()) + (2 * loop_radius if self_loops else 0.0) + 0.06
        labels = {ct: plots.wrap_name(ct) for ct in present}
        longest = max(len(line) for ct in present for line in labels[ct].splitlines())
        # A character is about 0.58 point wide per point of font size. Long cell-type names
        # get a smaller font, and the view is widened so that the labels of the left-most
        # and right-most nodes fit; its height only has to hold the labels of the top and
        # bottom ones, so the figure is as flat as the view and nothing is wasted.
        fontsize = float(np.clip(fontsize * 16.0 / max(longest, 1), 7.0, fontsize))
        text_inches = 0.58 * fontsize / 72.0 * longest
        limit = label_radius / max(1.0 - 2.0 * text_inches / figsize[0], 0.3)
        y_limit = label_radius + 0.55

        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0] * y_limit / limit))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_aspect("equal")
        ax.axis("off")
        points_per_unit = figsize[0] * 72.0 / (2.0 * limit)

        for source in present:
            for target in present:
                if source == target:
                    continue
                weight = composition.loc[source, target]
                if not np.isfinite(weight) or weight < min_threshold:
                    continue
                arrow = FancyArrowPatch(
                    posA=positions[source],
                    posB=positions[target],
                    arrowstyle="-|>",
                    connectionstyle=f"arc3,rad={curvature}",
                    color=self.palette.rgb(source),
                    linewidth=1.0 + 4.0 * float(weight),
                    alpha=0.75,
                    mutation_scale=14,
                    shrinkA=radii[source] * points_per_unit + 2.0,
                    shrinkB=radii[target] * points_per_unit + 2.0,
                )
                ax.add_patch(arrow)

        for cell_type in present:
            position = positions[cell_type]
            share = float(composition.loc[cell_type, cell_type])
            share = share if np.isfinite(share) else 0.0

            if self_loops and share >= min_threshold:
                # A circle tangent to the node, pointing away from the centre of the graph,
                # left open where it meets the node and closed by an arrow head.
                center = position * (1.0 + radii[cell_type] + loop_radius)
                back = np.arctan2(-position[1], -position[0])
                theta = np.linspace(back + 0.55, back + 2 * np.pi - 0.55, 120)
                curve = center + loop_radius * np.column_stack([np.cos(theta), np.sin(theta)])
                color = self.palette.rgb(cell_type)
                width = 1.0 + 4.0 * share
                ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=width, alpha=0.75, zorder=2)
                ax.add_patch(
                    FancyArrowPatch(
                        posA=curve[-3],
                        posB=curve[-1],
                        arrowstyle="-|>",
                        color=color,
                        linewidth=width,
                        alpha=0.75,
                        mutation_scale=14,
                    )
                )

            ax.add_patch(
                Circle(
                    position,
                    radii[cell_type],
                    facecolor=self.palette.rgb(cell_type),
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
            )

            x, y = position * label_radius
            ax.text(
                x,
                y,
                f"{labels[cell_type]}\n{int(counts.get(cell_type, 0)):,} cells · {share:.0%} self",
                ha="left" if position[0] > 0.15 else ("right" if position[0] < -0.15 else "center"),
                va="bottom" if position[1] > 0.15 else ("top" if position[1] < -0.15 else "center"),
                fontsize=fontsize,
            )

        ax.set_title("neighbourhood graph (arrow and loop width = share of neighbours)", fontsize=11, loc="left")

        return plots.close(fig, savefig)

    def compare_area(
        self,
        cell_types: Optional[Sequence[str]] = None,
        tests: Optional[Sequence[Sequence[str]]] = None,
        figsize: Tuple[float, float] = (9.0, 5.0),
        savefig: Optional[str] = None,
    ) -> Figure:
        """
        Nucleus area per predicted cell type, as box plots, with optional one-sided
        Mann-Whitney tests.

        Args:
            cell_types: Cell types to compare. Defaults to every type with cells.
            tests: Pairs ``[a, b]`` to test for "a greater than b".
            figsize: Figure size.
            savefig: Path to write the figure to.

        Returns:
            The figure. The test results are also logged.
        """

        import seaborn as sns

        areas = self.cell_areas()
        labels = self.labels.reindex(areas.index)
        available = [ct for ct in (cell_types or self.ct_list) if (labels == ct).any()]
        if not available:
            raise ValueError("No predicted cell has a contour for the requested cell types.")

        frame = pd.DataFrame({"cell_type": labels, "area": areas}).dropna()
        frame = frame[frame["cell_type"].isin(available)]
        unit = "µm²" if self.mpp is not None else "px²"

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            data=frame,
            x="cell_type",
            y="area",
            hue="cell_type",
            order=available,
            palette={ct: self.palette.rgb(ct) for ct in available},
            legend=False,
            fliersize=1,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(f"nucleus area ({unit})")
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(available, rotation=40, ha="right", fontsize=9)

        if tests:
            top = float(frame["area"].max())
            step = 0.12
            for position, (type_a, type_b) in enumerate(tests):
                if type_a not in available or type_b not in available:
                    logger.warning(f"Skipping the test {type_a} > {type_b}: one of them has no cell.")
                    continue
                areas_a = frame.loc[frame["cell_type"] == type_a, "area"].to_numpy()
                areas_b = frame.loc[frame["cell_type"] == type_b, "area"].to_numpy()
                statistic, p_value = mannwhitneyu(areas_a, areas_b, alternative="greater")
                stars = "***" if p_value <= 1e-3 else "**" if p_value <= 1e-2 else "*" if p_value <= 5e-2 else "ns"
                logger.info(f"{type_a} > {type_b}: U={statistic:.0f}, p={p_value:.3e} ({stars})")

                x1, x2 = available.index(type_a), available.index(type_b)
                height = top * (1.0 + step) ** (position + 1)
                ax.plot([x1, x1, x2, x2], [height, height * 1.05, height * 1.05, height], lw=1.1, c="black")
                ax.text(
                    (x1 + x2) / 2, height * 1.08, f"p={p_value:.1e} ({stars})", ha="center", va="bottom", fontsize=9
                )

        return plots.close(fig, savefig)

    # ------------------------------------------------------------------ slide views

    def visualizer(self, with_labels: bool = True, **kwargs: Any) -> Any:
        """
        Builds a :class:`~hedest.analysis.postseg.SlideVisualizer` for this run, with the
        predicted cell types and the palette already attached.

        Args:
            with_labels: Whether to colour the nuclei by predicted cell type.
            **kwargs: Overrides passed to the visualizer.

        Returns:
            The visualizer.
        """

        from hedest.analysis.postseg import SlideVisualizer

        self._require(slide_path=self.slide_path)
        arguments = dict(
            slide_path=self.slide_path,
            adata=self.adata,
            adata_name=self.adata_name,
            seg=self.seg_dict,
            labels=self.labels if with_labels else None,
            palette=self.palette,
            mpp=self.mpp,
        )
        arguments.update(kwargs)

        return SlideVisualizer(**arguments)

    # ------------------------------------------------------------------ exports

    def seg_dict_with_labels(self) -> Dict[str, Any]:
        """
        The segmentation dictionary with the HEDeST cell type written into each nucleus, in
        the format the overlays and the GeoJSON export expect.

        Returns:
            A new segmentation dictionary holding only the predicted cells.
        """

        self._require(seg=self.seg_dict)
        nuc = self.seg_dict["nuc"]  # type: ignore[index]
        labels = self.labels

        return {
            "nuc": {
                cell_id: {**value, "type": self.palette.index(labels[cell_id])}
                for cell_id, value in nuc.items()
                if cell_id in labels.index
            }
        }

    def export_predictions(self, path: Union[str, Path], with_probabilities: bool = True) -> Path:
        """
        Writes one row per cell: its predicted type, its confidence, its position and,
        optionally, the whole probability vector.

        Args:
            path: Destination CSV file.
            with_probabilities: Whether to append one column per cell type.

        Returns:
            The path written.
        """

        table = self.label_table()
        if with_probabilities:
            table = table.join(self.predictions.add_prefix("p_"))

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index_label="cell_id")
        logger.info(f"{len(table)} predictions written to {path}")

        return path

    def export_geojson(self, path: Union[str, Path]) -> Path:
        """
        Writes the predictions as a QuPath-compatible GeoJSON.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """

        from hedest.utils import seg_dict_to_geojson

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        seg_dict_to_geojson(self.seg_dict_with_labels(), str(path), color_dict=self.palette.as_geojson_dict())

        return path


def _figure_to_array(fig: Figure) -> np.ndarray:
    """
    Rasterises a figure so it can be embedded as an image in another figure.

    Args:
        fig: The figure to rasterise.

    Returns:
        An RGBA array.
    """

    import io

    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)

    return np.asarray(Image.open(buffer))


__all__ = ["PredAnalyzer"]
