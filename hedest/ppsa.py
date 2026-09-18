"""Prior Probability Shift Adjustment (PPSA) of per-cell type probabilities.

Both classes rescale the probabilities of a cell with the cell-type composition of its
*local* spot, relative to the global (slide-level) prior. They differ only in what "local"
means for a cell that does not lie inside a spot:

* :class:`PPSAdjustment` (``"interpolated"``) averages the <= 3 nearest spots found within
  ``2 x spot_diameter``, weighting each spot linearly with distance.
* :class:`PPSANearestNeighbors` (``"nearest"``) takes the proportions of the closest spot,
  whatever the distance.

By default both adjust **every** cell of ``cell_prob_df``. With ``gated=True`` they adjust
only the cells listed in ``spot_dict``, i.e. the cells inside spots.

``spot_dict`` always maps a spot to the cells **inside** it (``map_cells_to_spots`` with
``only_in=True``). It also defines the spots the model is trained on, so it must never be
widened just to make PPSA reach more cells: that is what ``gated=False`` is for. Locating
the cells outside spots needs the spot coordinates (``adata``) and the cell centroids
(``json_path`` or ``seg_dict``). When they are missing, the adjustment falls back to the
gated behaviour and says so; fully simulated datasets have no cell outside a spot, so the
fallback never costs them anything.
"""
from __future__ import annotations

import json
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from hedest.config import TqdmToLogger
from hedest.dataset import CellProbDataset
from hedest.utils import revert_dict

tqdm_out = TqdmToLogger(logger, level="INFO")

ADJUSTMENT_METHODS = ("interpolated", "nearest")


class BasePPSA:
    r"""
    Shared machinery of the PPSA variants: input checks, the local proportion vector of every
    cell, and the adjustment itself.

    For a cell :math:`x` with predicted probabilities :math:`p(c|x)`, local composition
    :math:`\tilde p(c)` and global prior :math:`p(c)`, the adjusted probabilities are
    :math:`\alpha \, p(c|x) \, \tilde p(c) / p(c)` with :math:`\alpha` the normalising factor,
    blended with the raw probabilities through ``beta`` (0 = fully adjusted, 1 = untouched).

    Args:
        cell_prob_df: *n_cells x n_types* matrix with raw model probabilities.
        spot_dict: Mapping *spot ID -> list(cell IDs inside that spot)*.
        spot_prop_df: *n_spots x n_types* matrix with spot-level cell-type fractions.
        global_prop: Global slide-wide cell-type fractions (same ordering as columns).
        adata: Visium AnnData object, needed to locate the cells outside spots.
        adata_name: Key under ``adata.uns['spatial']`` holding the scalefactors.
        json_path: Path to the segmentation JSON holding the cell centroids.
        seg_dict: Already-parsed segmentation, as an alternative to ``json_path``.
        gated: If True, adjust only the cells inside spots (needs no coordinates).
        beta: Interpolation between adjusted (0 -> full adj., 1 -> keep original).
        batch_size: Minibatch size for GPU throughput.
        eps: Numerical stability constant.
        device: Computation device. Defaults to "cuda" if available.
    """

    method = "base"

    def __init__(
        self,
        cell_prob_df: pd.DataFrame,
        spot_dict: Dict[str, List[str]],
        spot_prop_df: pd.DataFrame,
        global_prop: pd.Series,
        adata: Optional[AnnData] = None,
        adata_name: Optional[str] = None,
        json_path: Optional[str] = None,
        seg_dict: Optional[Dict] = None,
        gated: bool = False,
        beta: float = 0.0,
        batch_size: int = 256,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        if list(cell_prob_df.columns) != list(spot_prop_df.columns):
            raise ValueError("cell_prob_df and spot_prop_df must share identical columns order")

        # Basic attributes --------------------------------------------------
        self.beta_global = float(beta)
        self.batch_size = batch_size
        self.eps = eps
        self.gated = bool(gated)
        self.gated_requested = bool(gated)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Index maps --------------------------------------------------------
        self.cell_to_spot = revert_dict(spot_dict)
        self.cell_prob_df = cell_prob_df.copy()
        self.spot_prop_df = spot_prop_df.copy()
        self.global_prop = global_prop.copy()
        self.outside_cells = [c for c in self.cell_prob_df.index if str(c) not in self.cell_to_spot]

        # Geometry (only needed to reach the cells outside spots) -----------
        self.spot_coords = None
        self.spot_diameter = None
        self.kdtree = None
        self.nuc = None
        self._missing_centroids = 0
        self._setup_geometry(adata, adata_name, json_path, seg_dict)

        # Torch global prior -----------------------------------------------
        self.p_c = torch.tensor(self.global_prop.values, dtype=torch.float32, device=self.device).clamp(min=self.eps)

        # Build tensors needed for adjustment ------------------------------
        self.adjustable_cells, self.unadjustable_cells, p_local_np, beta_np = self._prepare_local_vectors()

        self.p_cell = torch.tensor(
            self.cell_prob_df.loc[self.adjustable_cells].values, dtype=torch.float32, device=self.device
        )
        self.p_local = torch.tensor(p_local_np, dtype=torch.float32, device=self.device)
        self.beta_cell = torch.tensor(beta_np, dtype=torch.float32, device=self.device)

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------

    def _setup_geometry(
        self,
        adata: Optional[AnnData],
        adata_name: Optional[str],
        json_path: Optional[str],
        seg_dict: Optional[Dict],
    ) -> None:
        """
        Prepares the lookup giving each cell inside a spot the proportions of that spot, and,
        when the cells outside spots have to be reached, the spot KD-tree and the cell
        centroids. Falls back to the gated behaviour when the inputs needed to locate those
        cells are missing.
        """

        # Cells inside spots always use the proportions of their own spot, whether or not that
        # spot is described by the AnnData object.
        self._spot_row = {sid: i for i, sid in enumerate(self.spot_prop_df.index)}
        self._spot_prop = self.spot_prop_df.to_numpy(dtype=np.float64)
        self._neighbour_prop = None  # proportions in KD-tree order, built only when needed

        if self.gated:
            logger.info(f"{self.method} PPSA is gated: only the cells inside spots are adjusted.")
            return

        if not self.outside_cells:
            # every cell sits in a spot (simulated datasets): nothing to locate
            logger.info(
                f"{self.method} PPSA: all {len(self.cell_prob_df)} cells are inside spots, "
                "gated and non-gated adjustment are equivalent here."
            )
            return

        missing = [
            name
            for name, value in (("adata", adata), ("adata_name", adata_name), ("segmentation", json_path or seg_dict))
            if value is None
        ]
        if missing:
            logger.warning(
                f"{len(self.outside_cells)} cell(s) lie outside the spots but {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing, so they cannot be located: falling back to "
                "gated adjustment (only the cells inside spots are adjusted). Provide the AnnData object, "
                "its name and the segmentation to adjust them, or ask for a gated adjustment explicitly."
            )
            self.gated = True
            return

        # Keep only spots that actually have proportions
        has_prop = adata.obs_names.isin(self.spot_prop_df.index)
        n_missing = int((~has_prop).sum())
        if n_missing:
            logger.warning(
                f"{n_missing}/{adata.n_obs} adata spot(s) have no proportions in "
                f"spot_prop_df and are ignored for the neighbour search."
            )
            adata = adata[has_prop]

        self.spot_coords = adata.obsm["spatial"].astype("float64")  # (n_spots, 2)
        self._neighbour_prop = self.spot_prop_df.reindex(adata.obs_names, copy=False).to_numpy(dtype=np.float64)

        self.spot_diameter = float(adata.uns["spatial"][adata_name]["scalefactors"]["spot_diameter_fullres"])
        self.kdtree = cKDTree(self.spot_coords)

        if seg_dict is None:
            with open(json_path) as json_file:
                seg_dict = json.load(json_file)
        self.nuc = seg_dict.get("nuc", seg_dict)

    def _centroid(self, cell_id: str) -> Optional[np.ndarray]:
        """Level-0 (x, y) centroid of a cell, or None when the segmentation does not hold it."""

        try:
            centroid = np.asarray(self.nuc[cell_id]["centroid"], dtype=np.float64)
            assert centroid.shape == (2,)
        except Exception:
            self._missing_centroids += 1
            return None
        return centroid

    def _local_vector_outside(self, cell_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Local proportion vector and beta of a cell outside any spot (None if unreachable)."""

        raise NotImplementedError

    def _prepare_local_vectors(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """
        Builds the local proportion vector and the beta of every cell.

        Returns:
            A tuple containing:
            - List of adjustable cell IDs.
            - List of unadjustable cell IDs.
            - Numpy array of local proportion vectors for adjustable cells.
            - Numpy array of beta values for adjustable cells.
        """

        adjustable, unadjustable = [], []
        local_vecs: List[np.ndarray] = []
        beta_list: List[float] = []

        for cell in self.cell_prob_df.index:
            cid = str(cell)

            # ---------------- inside a spot ----------------
            spot_id = self.cell_to_spot.get(cid)
            if spot_id is not None:
                s_idx = self._spot_row.get(spot_id)
                if s_idx is None:
                    raise ValueError(f"spot {spot_id!r} of spot_dict has no proportions in spot_prop_df")
                local_vecs.append(self._spot_prop[s_idx])
                beta_list.append(self.beta_global)
                adjustable.append(cell)
                continue

            # ---------------- outside a spot ---------------
            if self.gated or self.kdtree is None:
                unadjustable.append(cell)
                continue

            local = self._local_vector_outside(cid)
            if local is None:
                unadjustable.append(cell)
                continue

            vec, beta = local
            local_vecs.append(vec)
            beta_list.append(beta)
            adjustable.append(cell)

        if self._missing_centroids:
            logger.warning(
                f"{self._missing_centroids} cell(s) outside spots have no centroid in the segmentation "
                "and were left unadjusted."
            )

        n_types = self.cell_prob_df.shape[1]
        p_local = np.stack(local_vecs, axis=0) if local_vecs else np.empty((0, n_types), dtype=np.float64)

        return adjustable, unadjustable, p_local, np.array(beta_list, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Adjustment
    # ----------------------------------------------------------------------

    def _alpha(self, p_cell: torch.Tensor, p_local: torch.Tensor) -> torch.Tensor:
        """
        Computes the alpha adjustment factor for a cell.

        Args:
            p_cell: Predicted probability vector for a single cell.
            p_local: Local proportion vector for the cell.

        Returns:
            Alpha adjustment factor.
        """

        sim = torch.sum(p_cell * (p_local / self.p_c)).clamp(min=self.eps)
        return 1.0 / sim

    def adjust(self) -> pd.DataFrame:
        """
        Adjusts cell probabilities based on local proportions and beta values.

        Returns:
            DataFrame of adjusted cell probabilities, in the order of ``cell_prob_df``.
        """

        if not self.adjustable_cells:
            logger.warning("No cell could be adjusted: returning the raw probabilities.")
            return self.cell_prob_df.copy()

        dataset = CellProbDataset(self.p_cell, self.p_local, self.beta_cell)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        adj = torch.zeros_like(self.p_cell)

        for p_cell_batch, p_loc_batch, beta_batch, idx in tqdm(
            loader, file=tqdm_out, desc="Adjusting cell probabilities"
        ):
            # alpha per cell (vectorised)
            sim = torch.sum(p_cell_batch * (p_loc_batch / self.p_c), dim=1).clamp(min=self.eps)
            alpha = (1.0 / sim).clamp(max=1e6)

            p_adj = p_cell_batch * alpha.unsqueeze(1) * (p_loc_batch / self.p_c)
            p_final = (1.0 - beta_batch.unsqueeze(1)) * p_adj + beta_batch.unsqueeze(1) * p_cell_batch
            adj[idx] = p_final

        # Merge with untouched cells --------------------------------------
        adj_df = pd.DataFrame(adj.cpu().numpy(), index=self.adjustable_cells, columns=self.cell_prob_df.columns)
        if not self.unadjustable_cells:
            return adj_df.loc[self.cell_prob_df.index]

        untouched_df = self.cell_prob_df.loc[self.unadjustable_cells]
        out = pd.concat([adj_df, untouched_df])
        return out.loc[self.cell_prob_df.index]


class PPSAdjustment(BasePPSA):
    """
    PPSA whose cells outside spots are adjusted with an *interpolation* of their neighbouring
    spots: the <= 3 nearest spots within a radius of **2 x spot_diameter**, each weighted
    linearly with distance. A spot sitting exactly on the cell gets weight 1, a spot at the
    edge of the radius gets weight 0. A cell with a single spot in range keeps part of its raw
    probabilities (beta grows with the distance to that spot); a cell with no spot in range is
    left untouched.

    Cells inside spots use the proportions of their own spot. See :class:`BasePPSA` for the
    arguments.
    """

    method = "interpolated"

    @property
    def R(self) -> Optional[float]:
        """Search radius for the neighbouring spots."""

        return None if self.spot_diameter is None else 2.0 * self.spot_diameter

    def _local_vector_outside(self, cell_id: str) -> Optional[Tuple[np.ndarray, float]]:
        centroid = self._centroid(cell_id)
        if centroid is None:
            return None

        R = self.R
        dists, idxs = self.kdtree.query(centroid, k=3, distance_upper_bound=R)
        neighbours = [
            (int(i), float(d))
            for i, d in zip(np.atleast_1d(idxs), np.atleast_1d(dists))
            if i != self.kdtree.n and math.isfinite(d)
        ]

        if not neighbours:
            return None

        # ------- exactly one neighbour (distance-aware beta) ------
        if len(neighbours) == 1:
            idx0, d0 = neighbours[0]
            w = max((R - d0) / R, 0.0)  # 0..1
            return self._neighbour_prop[idx0], 1.0 - w  # far => beta ~ 1, close => beta ~ 0

        # ------- two or three neighbours --------------------------
        weights = np.clip(np.array([(R - d) / R for _, d in neighbours], dtype=np.float64), 0.0, 1.0)
        norm = weights.sum()
        if norm <= self.eps:
            return None

        vecs = np.stack([self._neighbour_prop[i] for i, _ in neighbours], axis=0)
        return (weights[:, None] * vecs).sum(axis=0) / norm, self.beta_global


class PPSANearestNeighbors(BasePPSA):
    """
    PPSA whose cells outside spots take the proportions of the **closest spot**, whatever the
    distance. Cells inside spots use the proportions of their own spot.

    See :class:`BasePPSA` for the arguments.
    """

    method = "nearest"

    def _local_vector_outside(self, cell_id: str) -> Optional[Tuple[np.ndarray, float]]:
        centroid = self._centroid(cell_id)
        if centroid is None:
            return None

        _, idx = self.kdtree.query(centroid, k=1)
        idx = int(idx)
        if idx >= self.kdtree.n:
            return None

        return self._neighbour_prop[idx], self.beta_global


ADJUSTMENT_CLASSES = {"interpolated": PPSAdjustment, "nearest": PPSANearestNeighbors}
