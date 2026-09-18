from __future__ import annotations

import pickle
import random
from collections import Counter
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from hedest.analysis.plots import plot_grid_celltype
from hedest.analysis.plots import plot_history
from hedest.analysis.plots import plot_legend
from hedest.analysis.plots import plot_mosaic_cells
from hedest.analysis.plots import plot_pie_chart
from hedest.analysis.plots import plot_predicted_cell_labels_in_spot
from hedest.analysis.plots import polygon_area
from hedest.analysis.postseg import StdVisualizer
from hedest.utils import fig_to_array
from hedest.utils import generate_color_dict
from hedest.utils import require_attributes


class PredAnalyzer:
    """
    A class to analyze predictions made by a cell classifier.
    """

    EXPECTED_VARIABLES = {
        "model_name",
        "hidden_dims",
        "norm",
        "dropout",
        "spot_dict",
        "train_spot_dict",
        "proportions",
        "history",
        "preds",
        "image_dict",
        "image_path",
        "adata",
        "adata_name",
        "seg_dict",
        "ground_truth",
        "adjustment",
        "gated",
    }

    def __init__(self, adjusted: bool = True, model_info: Optional[Union[dict, str]] = None, **kwargs):
        """
        Initializes PredAnalyzer with variables from a dictionary or a pickle file containing model informations
        and predictions. All variables can be None, except for 'preds' which must be provided. You can add more
        attributes dynamically using the `add_attributes` method.

        Args:
            adjusted: Whether to use adjusted predictions.
            model_info: Model information provided as:
                - A dictionary with variable data.
                - A path to a pickle file.
            **kwargs: Additional variables to add dynamically.
        """

        self.seg_dict_w_class = None
        self.delaunay_neighbors = None
        self.neighborhood_aggregates = None
        self.adjusted = adjusted
        self.model_info = {}

        # Load data from pickle if provided
        if model_info:
            if isinstance(model_info, dict):
                self.model_info = model_info
            elif isinstance(model_info, str):
                with open(model_info, "rb") as file:
                    self.model_info = pickle.load(file)
            else:
                raise ValueError("Invalid model_info type. Must be a dictionary or a path to a pickle file.")

        # Update with kwargs
        self.model_info.update(kwargs)

        unexpected_variables = set(self.model_info.keys()) - self.EXPECTED_VARIABLES
        if unexpected_variables:
            raise ValueError(
                f"Unexpected keys: {unexpected_variables}. " f"Expected keys are: {self.EXPECTED_VARIABLES}"
            )

        # Dynamic attribute assignment
        for key in self.EXPECTED_VARIABLES:
            setattr(self, key, self.model_info.get(key, None))

        assert self.preds is not None, "The 'preds' attribute must be provided and cannot be None."
        assert self.spot_dict is not None, "The 'spot_dict' attribute must be provided and cannot be None."

        if self.adjusted:
            self.predictions = self.preds["pred_best_adjusted"]
        else:
            self.predictions = self.preds["pred_best"]

        self.ct_list = list(self.predictions.columns)
        self.color_dict = generate_color_dict(self.ct_list, format="special")

        print("Loading predicted labels...")
        self.predicted_labels = self._get_labels_slide(self.predictions)
        self.predicted_proportions = self._get_predicted_proportions()

        all_spots = set(self.spot_dict.keys())
        all_cells = {cell for cell_list in self.spot_dict.values() for cell in cell_list}
        if self.train_spot_dict is not None:
            self.train_spots = list(self.train_spot_dict.keys())
            self.train_cells = list({cell for cell_list in self.train_spot_dict.values() for cell in cell_list})
            self.no_train_spots = list(all_spots - set(self.train_spots))
            self.no_train_cells = list(all_cells - set(self.train_cells))

        if self.ground_truth is not None:
            print("Loading true labels...")
            self.true_labels = self._get_labels_slide(self.ground_truth)

        if self.history is not None:
            self.history_train = self.history["train"]
            self.history_val = self.history["val"]

        else:
            print("Warning : No history provided. You won't be able to plot the training and validation histories.")
            print("Use `add_attributes(history=your_history)` to add one.")

        if self.seg_dict is not None:
            self._generate_dicts_viz_pred(self.seg_dict)

        else:
            print("Warning : No segmentation provided. You won't be able to plot the segmentation.")
            print("Use `add_attributes(seg_dict=your_seg_dict)` to add one.")

    def __repr__(self) -> str:
        """
        Returns a string representation of the PredAnalyzer instance.

        Returns:
            A string containing all expected attributes and their values.
        """

        attrs = ", ".join(f"{k}={getattr(self, k, None)}" for k in self.EXPECTED_VARIABLES)
        return f"PredAnalyzer({attrs})"

    @classmethod
    def expected_variables(cls) -> Set[str]:
        """
        Gets the set of expected variable keys.

        Returns:
            Expected variable keys.
        """

        return cls.EXPECTED_VARIABLES

    def add_attributes(self, **kwargs) -> None:
        """
        Dynamically adds attributes to the instance if they are in EXPECTED_VARIABLES.

        Args:
            **kwargs: Attribute names and values to add.

        Raises:
            ValueError: If any key is not in EXPECTED_VARIABLES.
        """

        for key, value in kwargs.items():
            if key not in self.EXPECTED_VARIABLES:
                raise ValueError(
                    f"Cannot add attribute '{key}', " f"it is not in the expected keys: {self.EXPECTED_VARIABLES}"
                )

            elif key == "ground_truth":
                self.true_labels = self._get_labels_slide(value)

            elif key == "history":
                self.history_train = value["train"]
                self.history_val = value["val"]

            elif key == "seg_dict":
                self._generate_dicts_viz_pred(value)

            setattr(self, key, value)

    def list_attributes(self) -> Dict[str, Any]:
        """
        Returns all current attributes of the instance.

        Returns:
            Dictionary of current attributes.
        """

        return {key: getattr(self, key, None) for key in self.EXPECTED_VARIABLES}

    def extract_stats(self, metric: str = "predicted") -> pd.DataFrame:
        """
        Extracts statistics from predictions. You can choose to extract statistics based on either
        predicted labels or all predictions.

        Args:
            metric: Metric to use, either "predicted" or "all".

        Returns:
            pd.DataFrame: DataFrame containing class-wise statistics.

        Raises:
            ValueError: If an invalid metric is specified.
        """

        stats = {}
        ct_list = list(self.predictions.columns)

        if metric == "predicted":
            for cell_id, pred in self.predicted_labels.items():
                max_prob = self.predictions.loc[cell_id].max()

                if pred["cell_type"] not in stats:
                    stats[pred["cell_type"]] = {"probs": [], "count": 0}

                stats[pred["cell_type"]]["probs"].append(max_prob)
                stats[pred["cell_type"]]["count"] += 1

            data = []
            for ct, class_stats in stats.items():
                class_probs = class_stats["probs"]
                row = [
                    self.predictions.columns.get_loc(ct),
                    ct,
                    np.min(class_probs),
                    np.max(class_probs),
                    np.median(class_probs),
                    np.mean(class_probs),
                    class_stats["count"],
                ]
                data.append(row)

            columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob", "Cell Count"]
            df_stats = pd.DataFrame(data, columns=columns)

        elif metric == "all":
            for ct in ct_list:
                stats[ct] = []

            for _, prob_vector in self.predictions.iterrows():
                for class_id, prob in enumerate(prob_vector):
                    stats[ct_list[class_id]].append(prob)

            data = []
            for ct, class_probs in stats.items():
                row = [
                    self.predictions.columns.get_loc(ct),
                    ct,
                    np.min(class_probs),
                    np.max(class_probs),
                    np.median(class_probs),
                    np.mean(class_probs),
                ]
                data.append(row)

            columns = ["Class", "CT", "Min Prob", "Max Prob", "Median Prob", "Mean Prob"]
            df_stats = pd.DataFrame(data, columns=columns)

        else:
            raise ValueError("Invalid metric. Choose 'predicted' or 'all'.")

        df_stats = df_stats.sort_values(by="Class", ascending=True).reset_index(drop=True)
        df_stats = df_stats.set_index("Class")

        return df_stats

    @require_attributes("history_train", "history_val")
    def plot_history(self, show: bool = False, savefig: Optional[str] = None) -> None:
        """
        Plots training and validation history.

        Args:
            show: Whether to display the plot.
            savefig: File path to save the plot.

        Returns:
            Train and validation history of the model.
        """

        return plot_history(self.history_train, self.history_val, show=show, savefig=savefig)

    @require_attributes("spot_dict", "image_dict")
    def plot_mosaic_cells(
        self, spot_id: Optional[str] = None, num_cols: int = 8, display: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plots a grid of individual cell images for a given spot ID.

        Args:
            spot_id: Spot ID to plot. If None, spot ID will be random.
            num_cols: Number of columns in the grid.
            display: Whether to display the plot.

        Returns:
            Image grid with individual cell images of the same spot.
        """

        return plot_mosaic_cells(
            self.spot_dict,
            self.image_dict,
            spot_id=spot_id,
            predicted_labels=self.predicted_labels,
            true_labels=self.true_labels,
            num_cols=num_cols,
            display=display,
        )

    @require_attributes("spot_dict", "image_dict", "image_path", "adata", "adata_name")
    def plot_predicted_cell_labels_in_spot(
        self, spot_id: Optional[str] = None, show_labels: bool = True, display: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plots a spot's visualization with all cell images arranged in a grid.

        Args:
            spot_id: Spot ID to plot. If None, spot ID will be random.
            show_labels: Whether to show predicted labels.
            display: Whether to display the plot.

        Returns:
            Image spot with cell spots and potentially cell labels.
        """

        return plot_predicted_cell_labels_in_spot(
            spot_dict=self.spot_dict,
            adata=self.adata,
            adata_name=self.adata_name,
            image_path=self.image_path,
            image_dict=self.image_dict,
            predicted_labels=[None, self.predicted_labels][show_labels],
            true_labels=[None, self.true_labels][show_labels],
            spot_id=spot_id,
            display=display,
        )

    @require_attributes("spot_dict", "proportions", "image_path", "adata", "adata_name")
    def plot_spot_proportions(self, spot_id: Optional[str] = None, draw_seg: bool = False) -> None:
        """
        Plots true and predicted cell type proportions for a given spot.

        Args:
            spot_id: Spot ID to plot. If None, selects a random spot.
            draw_seg: Whether to draw segmentation overlays.
        """

        if draw_seg:
            if self.seg_dict_w_class is None:
                raise ValueError("You must run `_generate_dicts_viz_pred` before to be able to plot segmentation.")

        if spot_id is None:
            spot_id = random.choice(list(self.spot_dict.keys()))
            print(f"Randomly selected spot_id: {spot_id}")

        elif spot_id not in self.spot_dict:
            raise ValueError(f"Spot ID {spot_id} not found in spot_dict.")

        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])

        ax0 = fig.add_subplot(gs[:, 0])
        plotter = StdVisualizer(
            self.image_path,
            self.adata,
            self.adata_name,
            [None, self.seg_dict_w_class][draw_seg],
            [None, self.color_dict][draw_seg],
        )

        fig1 = plotter.plot_specific_spot(spot_id=spot_id, display=False)
        img1 = fig_to_array(fig1)
        ax0.imshow(img1)
        ax0.axis("off")

        list_cells = self.spot_dict[spot_id]

        pie_color_dict = generate_color_dict(self.ct_list, format="classic")

        # mean predicted probabilities
        ax1 = fig.add_subplot(gs[0, 1])
        mean_prob_ct = self.predictions[self.predictions.index.isin(list_cells)].mean(axis=0)
        plot_pie_chart(ax1, mean_prob_ct, color_dict=pie_color_dict)
        ax1.set_title("Mean Predicted Probabilities")

        # predicted cell type proportions
        ax2 = fig.add_subplot(gs[0, 2])
        prop_ct = self.predictions[self.predictions.index.isin(list_cells)].idxmax(axis=1).value_counts() / len(
            list_cells
        )
        plot_pie_chart(ax2, prop_ct, color_dict=pie_color_dict)
        ax2.set_title("Predicted Cell Type Proportions")

        # true cell type proportions
        ax3 = fig.add_subplot(gs[1, 1])
        true_prop = self.proportions.loc[spot_id]
        plot_pie_chart(ax3, true_prop, color_dict=pie_color_dict)
        ax3.set_title("True Cell Type Proportions")

        # legend
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis("off")
        plot_legend(pie_color_dict, ax4)
        ax4.set_title("Legend")

        plt.tight_layout()
        plt.show()

    @require_attributes("proportions", "spot_dict")
    def plot_colocalization_matrix(
        self, title: str = "", display: bool = True, figsize: tuple = (8, 6), cmap: str = "coolwarm"
    ) -> Optional[plt.Figure]:
        """
        Plots the Pearson correlation matrix of cell type proportions across spots.

        This visualizes cell type colocalization: how often cell types co-occur
        across spots based on proportion similarity.

        Args:
            title: Title of the plot.
            display: Whether to display the plot immediately.
            figsize: Size of the figure.
            cmap: Colormap for heatmap.

        Returns:
            The matplotlib figure (if display is False).
        """

        correlation_matrix = self.proportions.corr(method="pearson")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            correlation_matrix,
            annot=False,
            fmt=".2f",
            cmap=cmap,
            square=True,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "Pearson Correlation"},
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        ax.set_title(title, fontsize=14)
        plt.tight_layout()

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    @require_attributes("proportions", "spot_dict")
    def evaluate_prop_predictions(self, subset="all") -> Dict[str, float]:
        """
        Evaluates slide-level predictions using various metrics. With this function, you can compute a series of
        metrics to compare, for each cell-type, true vs predicted proportions over the histological slide.

        Args:
            subset: Subset of spots to evaluate ("train", "no_train", or "all").

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        if subset == "train":
            predicted_proportions = self.predicted_proportions.loc[self.train_spots]
        elif subset == "no_train":
            predicted_proportions = self.predicted_proportions.loc[self.no_train_spots]
        elif subset == "all":
            predicted_proportions = self.predicted_proportions.copy()
        else:
            raise ValueError("Invalid subset. Choose 'train', 'no_train', or 'all'.")

        true_proportions, predicted_proportions = self.proportions.align(predicted_proportions, join="inner", axis=0)

        if predicted_proportions.isna().any().any():
            predicted_proportions = predicted_proportions.dropna()
            true_proportions = true_proportions.loc[predicted_proportions.index]

        metrics = {}

        # Compute per-cell-type metrics
        pearson_list = []
        spearman_list = []
        mse_list = []
        mae_list = []

        for cell_type in true_proportions.columns:
            true_col = true_proportions[cell_type].values
            pred_col = predicted_proportions[cell_type].values

            if np.std(true_col) > 0 and np.std(pred_col) > 0:  # Ensure non-constant values
                pearson_corr = pearsonr(true_col, pred_col)[0]
                spearman_corr = spearmanr(true_col, pred_col)[0]
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan

            mse_value = mean_squared_error(true_col, pred_col)
            mae_value = mean_absolute_error(true_col, pred_col)

            metrics[f"Pearson Correlation {cell_type}"] = pearson_corr
            metrics[f"Spearman Correlation {cell_type}"] = spearman_corr
            metrics[f"MSE {cell_type}"] = mse_value
            metrics[f"MAE {cell_type}"] = mae_value

            pearson_list.append(pearson_corr)
            spearman_list.append(spearman_corr)
            mse_list.append(mse_value)
            mae_list.append(mae_value)

        # Compute global metrics (Averaged approach)
        metrics["Pearson Correlation global"] = np.nanmean(pearson_list)
        metrics["Spearman Correlation global"] = np.nanmean(spearman_list)
        metrics["MSE global"] = np.mean(mse_list)
        metrics["MAE global"] = np.mean(mae_list)

        return metrics

    def plot_predicted_probability_histograms(
        self,
        bins: int = 60,
        y_lim: Optional[tuple] = None,
        figsize: tuple = (16, 10),
        compare_to_gt: bool = False,
        savefig: Optional[str] = None,
    ):
        """
        Plots histograms of predicted probabilities for each cell type.

        If compare_to_gt is True, plots histograms separately for cells that are truly
        that type (based on ground truth) vs. those that are not.

        Args:
            bins: Number of bins for histogram.
            y_lim: Y-axis limit.
            figsize: Size of the figure.
            compare_to_gt: Whether to split by true/false labels.
            savefig: Path to save the figure.
        """

        sns.set(style="whitegrid")

        cell_types = self.predictions.columns.tolist()
        n_types = len(cell_types)
        n_cols = 3
        n_rows = (n_types + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        if compare_to_gt:
            if self.ground_truth is None:
                raise ValueError("Ground truth must be provided to compare predicted probabilities.")

            gt_df = self.ground_truth.copy()

            for i, cell_type in enumerate(cell_types):
                ax = axes[i]

                probs = self.predictions[cell_type]
                truth = gt_df[cell_type] >= 0.5

                plot_df = pd.DataFrame(
                    {
                        "Predicted Probability": probs,
                        "True Label": truth.map({True: f"{cell_type}", False: f"Not {cell_type}"}),
                    }
                )

                sns.histplot(
                    data=plot_df,
                    x="Predicted Probability",
                    hue="True Label",
                    bins=bins,
                    ax=ax,
                    palette="Set1",
                    element="step",
                    stat="count",
                    common_norm=False,
                )

                ax.set_title(f"{cell_type}", fontsize=12)
                ax.set_xlim(0, 1)
                if y_lim is not None:
                    ax.set_ylim(y_lim)

        else:
            for i, cell_type in enumerate(cell_types):
                ax = axes[i]
                sns.histplot(self.predictions[cell_type], bins=bins, kde=False, ax=ax, color="skyblue")
                ax.set_title(f"{cell_type}", fontsize=12)
                ax.set_xlabel("Predicted Probability")
                ax.set_ylabel("Count")
                ax.set_xlim(0, 1)
                if y_lim is not None:
                    ax.set_ylim(y_lim)

        for j in range(n_types, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")

        plt.show()

    def evaluate_cell_predictions(self, subset="all", per_class=True) -> Dict[str, float]:
        """
        Evaluates cell-level predictions using various metrics.

        Args:
            subset: Subset of cells to evaluate ("train", "no_train", or "all").
            per_class: Whether to compute metrics per class.

        Returns:
            A dictionary of computed metrics.
        """

        if self.true_labels is None:
            raise ValueError("True labels are not available. Please provide ground_truth.")

        if subset == "train":
            true_labels = {k: v for k, v in self.true_labels.items() if k in self.train_cells}
            predicted_labels = {k: v for k, v in self.predicted_labels.items() if k in self.train_cells}
        elif subset == "no_train":
            true_labels = {k: v for k, v in self.true_labels.items() if k in self.no_train_cells}
            predicted_labels = {k: v for k, v in self.predicted_labels.items() if k in self.no_train_cells}
        elif subset == "all":
            true_labels = self.true_labels.copy()
            predicted_labels = self.predicted_labels.copy()
        else:
            raise ValueError("Invalid subset. Choose 'train', 'no_train', or 'all'.")

        true_labels = pd.Series({k: v["cell_type"] for k, v in true_labels.items()})
        predicted_labels = pd.Series({k: v["cell_type"] for k, v in predicted_labels.items()})

        # Global accuracy
        global_accuracy = accuracy_score(true_labels, predicted_labels)

        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

        # Weighted metrics
        weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        weighted_precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        weighted_recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

        metrics = {
            "Global Accuracy": global_accuracy,
            "Balanced Accuracy": balanced_acc,
            "Weighted F1 Score": weighted_f1,
            "Weighted Precision": weighted_precision,
            "Weighted Recall": weighted_recall,
        }

        if per_class:
            unique_classes = np.unique(true_labels)
            f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
            precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
            recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
            cm = confusion_matrix(true_labels, predicted_labels)

            metrics.update(
                {
                    "F1 Score (Per Class)": dict(zip(unique_classes, f1_per_class)),
                    "Precision (Per Class)": dict(zip(unique_classes, precision_per_class)),
                    "Recall (Per Class)": dict(zip(unique_classes, recall_per_class)),
                    "Confusion Matrix": pd.DataFrame(cm, columns=list(unique_classes), index=list(unique_classes)),
                }
            )

        return metrics

    @require_attributes("image_dict")
    def plot_grid_celltype(
        self,
        cell_type: Optional[str] = None,
        n: int = 20,
        selection: str = "max",
        show_probs: bool = True,
        display: bool = False,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        fontsize: int = 20,
    ) -> Optional[plt.Figure]:
        """
        Plots a grid of cell images predicted as one or multiple cell types.

        Args:
            cell_type: Target cell type. If None, plot all cell types in a big grid.
            n: Number of images per cell type grid.
            selection: Selection mode ("max" or "random").
            show_probs: Whether to show probability labels.
            display: Whether to display the plot.
            nrows: Number of rows for the big grid (only used if cell_type is None).
            ncols: Number of cols for the big grid (only used if cell_type is None).
            fontsize: Font size for cell type titles (only used if cell_type is None).

        Returns:
            The generated matplotlib figure.
        """

        if cell_type is not None:
            # --- Single cell type: just call your original function ---
            return plot_grid_celltype(
                self.predictions,
                self.image_dict,
                cell_type,
                n=n,
                selection=selection,
                title=cell_type,
                show_probs=show_probs,
                display=display,
            )

        # --- All cell types in one big plot ---
        ct_list = [ct for ct in self.ct_list if (self.predictions.idxmax(axis=1) == ct).any()]
        n_ct = len(ct_list)

        if nrows is None or ncols is None:
            # auto square layout if not provided
            nrows = int(np.ceil(np.sqrt(n_ct)))
            ncols = int(np.ceil(n_ct / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6))
        axes = np.atleast_2d(axes)

        for idx, ct in enumerate(ct_list):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            # generate the mini-figure using your original function
            subfig = plot_grid_celltype(
                self.predictions,
                self.image_dict,
                ct,
                n=n,
                selection=selection,
                show_probs=show_probs,
                display=False,
            )

            # draw the mini-figure into the main subplot
            subfig.canvas.draw()
            img = np.frombuffer(subfig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(subfig.canvas.get_width_height()[::-1] + (3,))
            ax.imshow(img)
            ax.set_title(ct, fontsize=fontsize)
            ax.axis("off")

            plt.close(subfig)

        # turn off any extra empty slots
        for i in range(n_ct, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].axis("off")

        plt.tight_layout()

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    @require_attributes("adata", "adata_name")
    def compare_area(
        self,
        cell_types: List[str],
        mpp: Optional[float] = None,
        title: str = "",
        ct_utest: List[List[str]] = None,
        height_unit_factor: float = 0.08,
        y_offset_factor: float = 1.15,
        fontsize_utest: float = 12,
        savefig: Optional[str] = None,
    ) -> None:
        """
        Compares the area of predicted cells for specific cell types using box plots and optional statistical tests.

        Args:
            cell_types (List[str]): List of cell types to compare. Must be in self.ct_list.
            mpp (float, optional): Microns per pixel for area conversion.
            title (str, optional): Plot title.
            ct_utest (List[List[str]], optional): List of [cell_type_A, cell_type_B] pairs for statistical comparison.
                                                    Performs one-sided Mann–Whitney U test (A > B).
            height_unit_factor (float): Factor to adjust vertical spacing of statistical annotations.
            y_offset_factor (float): Vertical offset factor for statistical annotation brackets.
            fontsize_utest (float): Font size for statistical annotation text.
            savefig (str, optional): If provided, saves the plot to this path.
        """

        if mpp is None:
            mpp = 55 / self.adata.uns["spatial"][self.adata_name]["scalefactors"]["spot_diameter_fullres"]

        # --- Validation ---
        invalid_ct = [ct for ct in cell_types if ct not in self.ct_list]
        if invalid_ct:
            raise ValueError(f"Invalid cell types: {invalid_ct}. Available types: {self.ct_list}")

        if self.seg_dict_w_class is None:
            raise ValueError(
                "No segmentation with predicted classes found. Please run `_generate_dicts_viz_pred` first."
            )

        # --- Area collection ---
        areas_by_type = {ct: [] for ct in cell_types}

        for cell_id, info in self.seg_dict_w_class["nuc"].items():
            cell_label = self.predicted_labels.get(cell_id, {}).get("cell_type", None)
            if cell_label in cell_types:
                contour = info.get("contour")
                if contour and len(contour) >= 3:
                    area = polygon_area(contour) * (mpp**2)
                    areas_by_type[cell_label].append(area)

        data = [{"Cell Type": ct, "Area": area} for ct, areas in areas_by_type.items() for area in areas]
        df = pd.DataFrame(data)

        # --- Color map ---
        color_map = {}
        for k, (name, rgba) in self.color_dict.items():
            rgb = tuple(c / 255 for c in rgba[:3])
            color_map[name] = rgb
        palette = {ct: color_map.get(ct, (0.5, 0.5, 0.5)) for ct in cell_types}

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=df, x="Cell Type", y="Area", hue="Cell Type", palette=palette, dodge=False, legend=False)

        plt.title(title)
        plt.xlabel("")
        plt.ylabel("Nucleus area (µm²)")
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # --- Optional statistical comparison ---
        if ct_utest is not None:

            def p_to_star(p):
                if p <= 0.001:
                    return "***"
                elif p <= 0.01:
                    return "**"
                elif p <= 0.05:
                    return "*"
                else:
                    return "ns"

            # Store current max height for stacking brackets
            y_max = df["Area"].max()
            y_min = df["Area"].min()
            height_unit = (np.log10(y_max) - np.log10(y_min)) * height_unit_factor

            # For each pair, perform test and annotate
            for i, (ct_a, ct_b) in enumerate(ct_utest):
                if ct_a not in areas_by_type or ct_b not in areas_by_type:
                    print(f"Skipping invalid comparison: {ct_a} vs {ct_b}")
                    continue

                areas_a = np.array(areas_by_type[ct_a])
                areas_b = np.array(areas_by_type[ct_b])
                if len(areas_a) == 0 or len(areas_b) == 0:
                    print(f"Skipping empty comparison: {ct_a} vs {ct_b}")
                    continue

                # Mann–Whitney U test (one-sided: A > B)
                statistic, p_value = mannwhitneyu(areas_a, areas_b, alternative="greater")
                stars = p_to_star(p_value)

                # --- Plot annotation ---
                x1, x2 = cell_types.index(ct_a), cell_types.index(ct_b)
                y, _ = np.log10(y_max) + (i * height_unit), height_unit * 0.4
                y = 10**y  # back-transform to log scale space

                # Bracket
                ax.plot([x1, x1, x2, x2], [y, y * y_offset_factor, y * y_offset_factor, y], lw=1.2, c="black")

                # Annotation text
                ax.text(
                    (x1 + x2) / 2,
                    y * (y_offset_factor**1.3),
                    f"{ct_a} > {ct_b}\np={p_value:.2e} ({stars})",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize_utest,
                    color="black",
                )

                # Print results in console too
                print(f"{ct_a} > {ct_b}: U={statistic:.3f}, p={p_value:.4e} ({stars})")

        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, format=savefig.split(".")[-1], dpi=600, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def compute_neighborhood_composition(
        self, compute_dist: str = "centroid", max_distance: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes average neighbor composition per cell type.

        Args:
            compute_dist: The method to compute distances ("centroid" or "contour").
            max_distance: The maximum distance to consider for neighbors.

        Returns:
            Dictionnary cell_type -> neighbor_type -> average count
        """

        self.delaunay_neighbors = self._build_delaunay_graph(compute_dist=compute_dist, max_distance=max_distance)

        neighbor_counts = defaultdict(lambda: defaultdict(int))
        source_type_counts = defaultdict(int)

        for cell_id, neighbors in self.delaunay_neighbors.items():
            if cell_id not in self.predicted_labels:
                continue

            source_type = self.predicted_labels[cell_id]["cell_type"]
            source_type_counts[source_type] += 1

            for neighbor_id in neighbors:
                if neighbor_id not in self.predicted_labels:
                    continue
                neighbor_type = self.predicted_labels[neighbor_id]["cell_type"]
                neighbor_counts[source_type][neighbor_type] += 1

        aggregated = {
            src_type: {neigh_type: count / source_type_counts[src_type] for neigh_type, count in neigh_dict.items()}
            for src_type, neigh_dict in neighbor_counts.items()
        }

        self.neighborhood_aggregates = aggregated
        return aggregated

    @require_attributes("adata", "adata_name")
    def plot_mean_neighbor_distances(
        self,
        mpp: Optional[float] = None,
        max_distance: Optional[float] = None,
        cmap: str = "coolwarm",
        display: bool = True,
    ):
        """
        Plots a symmetric matrix of mean distances between neighboring cell types.

        Args:
            mpp: Microns per pixel for area conversion.
            max_distance: The maximum distance to consider for neighbors.
            cmap: The colormap to use for the plot.
            display: Whether to display the plot or not.
        """

        if mpp is None:
            mpp = 55 / self.adata.uns["spatial"][self.adata_name]["scalefactors"]["spot_diameter_fullres"]

        neighbors = self._build_delaunay_graph(max_distance=max_distance)
        nuc_dict = self.seg_dict_w_class["nuc"]

        pair_distances = defaultdict(list)
        coords = {cid: np.array(info["centroid"]) for cid, info in nuc_dict.items()}
        types = {cid: self.ct_list[info["type"]] for cid, info in nuc_dict.items()}

        for cell_a, neigh_list in neighbors.items():
            for cell_b in neigh_list:
                if cell_a >= cell_b:
                    continue
                type_a, type_b = types[cell_a], types[cell_b]
                pair_key = tuple(sorted((type_a, type_b)))
                dist = np.linalg.norm(coords[cell_a] - coords[cell_b])
                pair_distances[pair_key].append(dist)

        matrix = pd.DataFrame(np.nan, index=self.ct_list, columns=self.ct_list, dtype=float)

        for (type_a, type_b), dist_list in pair_distances.items():
            mean_dist = np.mean(dist_list) * mpp
            matrix.loc[type_a, type_b] = mean_dist
            matrix.loc[type_b, type_a] = mean_dist

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            cmap=cmap,
            annot=False,
            square=True,
            linewidths=0.5,
            fmt=".2f",
            mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
            cbar_kws={"orientation": "horizontal", "shrink": 0.55, "label": "Mean Distance (μm)"},
            ax=ax,
        )
        ax.set_xticklabels(self.ct_list, rotation=25, ha="right")
        ax.set_yticklabels(self.ct_list, rotation=0)
        plt.grid(False)
        plt.tight_layout()

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    def plot_colocalization_graph(
        self,
        display: bool = True,
        figsize: tuple = (8, 8),
        curvature: float = 0.3,
        min_threshold: float = 0.05,
        fontsize: int = 12,
    ) -> Optional[plt.Figure]:
        """
        Plots a circular graph where:
        - Node size is proportional to cell count
        - Arrows show directional neighborhood influence
        - Arrows are curved in opposite directions for A->B and B->A

        Args:
            display: Whether to display the plot.
            figsize: Size of the figure.
            curvature: Curvature of arrows between nodes.
            min_threshold: Minimum weight threshold to draw an arrow.
            fontsize: Font size for cell type labels.
        """

        if self.neighborhood_aggregates is None:
            text1 = "Attribute `neighborhood_aggregates` is None. "
            text2 = "Computing neighborhood composition with default parameters..."
            print(text1 + text2)
            self.compute_neighborhood_composition()

        # All unique cell types
        cell_types = sorted(
            set(self.neighborhood_aggregates) | {ct for d in self.neighborhood_aggregates.values() for ct in d}
        )

        # Count cells per type from predicted labels
        node_counts = Counter([v["cell_type"] for v in self.predicted_labels.values()])

        # Normalize node sizes for plotting
        min_size, max_size = 10, 500
        raw_counts = np.array([node_counts[ct] for ct in cell_types])
        norm_sizes = min_size + (raw_counts - raw_counts.min()) / (raw_counts.ptp() + 1e-6) * (max_size - min_size)
        node_size_dict = dict(zip(cell_types, norm_sizes))

        # Layout in circle
        G = nx.DiGraph()
        G.add_nodes_from(cell_types)
        pos = nx.circular_layout(G)

        # Track edges we've drawn already to apply symmetric curvature
        drawn_edges = set()

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        name_to_rgba = {
            v[0]: tuple(c / 255 for c in v[1]) for v in self.color_dict.values()  # normalize for matplotlib
        }

        # Draw nodes (circles)
        for ct in cell_types:
            x, y = pos[ct]
            ax.scatter(x, y, s=node_size_dict[ct], color=name_to_rgba[ct], zorder=3)
            label_offset = 0.16
            dx, dy = x, y
            norm = (dx**2 + dy**2) ** 0.5
            offset_x = x + label_offset * dx / norm
            offset_y = y + label_offset * dy / norm

            ax.text(offset_x, offset_y, ct, ha="center", va="center", fontsize=fontsize)

        # Draw all arrows with curvature handling
        for src in cell_types:
            targets = self.neighborhood_aggregates.get(src, {})
            for tgt, weight in targets.items():
                if weight < min_threshold:
                    continue

                if src == tgt and weight >= min_threshold:
                    x, y = pos[src]
                    color = name_to_rgba[src]

                    # Direction vector from center
                    dx, dy = x, y
                    norm = (dx**2 + dy**2) ** 0.5
                    ux, uy = dx / norm, dy / norm

                    # Perpendicular unit vector
                    px, py = -uy, ux

                    # Control points for smooth oval loop
                    loop_size = 0.3
                    offset = 0.4

                    # Control point 1 (move outwards and to one side)
                    ctrl1 = (x + offset * ux + loop_size * px, y + offset * uy + loop_size * py)
                    # Control point 2 (return from other side)
                    ctrl2 = (x + offset * ux - loop_size * px, y + offset * uy - loop_size * py)

                    # Path vertices
                    path_data = [
                        (Path.MOVETO, (x, y)),
                        (Path.CURVE4, ctrl1),
                        (Path.CURVE4, ctrl2),
                        (Path.CURVE4, (x, y)),
                    ]
                    codes, verts = zip(*path_data)
                    path = Path(verts, codes)

                    # Transparent body of loop
                    patch = PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=color,
                        lw=2 * weight + 0.3,
                        alpha=0.5,  # transparent line
                        zorder=1,
                    )
                    ax.add_patch(patch)

                    # Arrowhead at the end
                    arrow = FancyArrowPatch(
                        posA=ctrl2,
                        posB=(x, y),
                        arrowstyle="->",
                        color=color,
                        lw=0,
                        mutation_scale=15,
                        alpha=1.0,
                        zorder=2,
                    )
                    ax.add_patch(arrow)
                    continue

                if (src, tgt) in drawn_edges or (tgt, src) in drawn_edges:
                    # If both directions exist, use symmetric curvatures
                    direction = +1 if (src, tgt) not in drawn_edges else -1
                    rad = direction * curvature
                else:
                    # First appearance of this pair, check if reciprocal exists
                    rad = curvature
                    if tgt in self.neighborhood_aggregates and src in self.neighborhood_aggregates[tgt]:
                        drawn_edges.add((src, tgt))  # mark that one direction is drawn

                # Draw the arrow
                src_pos = pos[src]
                tgt_pos = pos[tgt]
                color = name_to_rgba[src]

                # Transparent arrow body
                arrow_line = FancyArrowPatch(
                    posA=src_pos,
                    posB=tgt_pos,
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={rad}",
                    color=color,
                    lw=2 * weight + 0.3,
                    alpha=0.5,
                    shrinkA=18,
                    shrinkB=18,
                    zorder=1,
                )
                ax.add_patch(arrow_line)

                # Opaque arrowhead
                arrow_head = FancyArrowPatch(
                    posA=src_pos,
                    posB=tgt_pos,
                    arrowstyle="-|>",
                    connectionstyle=f"arc3,rad={rad}",
                    color=color,
                    lw=0,  # no line
                    mutation_scale=15,
                    alpha=1.0,
                    shrinkA=11,
                    shrinkB=11,
                    zorder=2,
                )
                ax.add_patch(arrow_head)

        if display:
            plt.show()
            return None
        else:
            plt.close(fig)
            return fig

    def _build_delaunay_graph(
        self, compute_dist: str = "centroid", max_distance: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """
        Builds a Delaunay triangulation graph from segmentation centroids.
        Each cell is a node and neighbors are connected by edges, optionally filtered by a distance threshold.

        Args:
            compute_dist: The method to compute distances ("centroid" or "contour").
            max_distance: Maximum allowed distance between neighbors. If None, no filtering.

        Returns:
            Mapping of cell_id to list of neighboring cell_ids.
        """

        if self.seg_dict_w_class is None:
            raise ValueError(
                "No segmentation with predicted classes found. Please run `_generate_dicts_viz_pred` first."
            )

        nuc_dict = self.seg_dict_w_class["nuc"]
        coords = np.array([v["centroid"] for v in nuc_dict.values()])
        cell_ids = list(nuc_dict.keys())

        tri = Delaunay(coords)
        neighbors = defaultdict(set)

        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                cell_a, cell_b = cell_ids[a], cell_ids[b]

                # Distance filtering
                if max_distance is not None:
                    if compute_dist == "centroid":
                        dist = np.linalg.norm(coords[a] - coords[b])
                    elif compute_dist == "contour":
                        contour_a = np.array(nuc_dict[cell_a]["contour"])
                        contour_b = np.array(nuc_dict[cell_b]["contour"])
                        dist = np.min(np.linalg.norm(contour_a[:, None, :] - contour_b[None, :, :], axis=-1))
                    else:
                        raise ValueError(f"Unknown compute_dist method: {compute_dist}")

                    if dist > max_distance:
                        continue

                neighbors[cell_a].add(cell_b)
                neighbors[cell_b].add(cell_a)

        return {k: list(v) for k, v in neighbors.items()}

    @require_attributes("spot_dict")
    def _get_predicted_proportions(self) -> pd.DataFrame:
        """
        Computes predicted proportions of cell types for each spot.

        Returns:
            DataFrame with predicted proportions per spot.
        """

        spot_proportions = {}

        for spot_id, cell_ids in self.spot_dict.items():
            spot_cells = self.predictions.loc[self.predictions.index.isin(cell_ids)]
            spot_proportions[spot_id] = spot_cells.mean(axis=0)

        predicted_proportions_df = pd.DataFrame.from_dict(spot_proportions, orient="index")
        predicted_proportions_df.index.name = "spot"

        return predicted_proportions_df

    def _get_labels_slide(self, data) -> Dict[str, Dict[str, Any]]:
        """
        Generates a dictionary mapping cell IDs to predicted labels.

        Args:
            data: DataFrame with predicted probabilities per cell.

        Returns:
            Mapping of cell IDs to predicted labels.
        """

        predicted_classes = data.idxmax(axis=1)

        predicted_class_indices = predicted_classes.map(data.columns.get_loc)

        predicted_labels = {
            cell_id: {"class": cls_idx, "cell_type": cls}
            for cell_id, cls_idx, cls in zip(data.index, predicted_class_indices, predicted_classes)
        }

        return predicted_labels

    def _generate_dicts_viz_pred(self, seg_dict: Dict[str, Any]) -> None:
        """
        Updates the segmentation dictionary with predicted classes.

        Args:
            seg_dict: Dictionary containing segmentation data.
        """

        self.seg_dict_w_class = {
            "nuc": {
                key: {
                    **value,
                    "type": self.predicted_labels[key]["class"],
                }
                for key, value in seg_dict["nuc"].items()
                if key in self.predicted_labels
            }
        }
