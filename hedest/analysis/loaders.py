"""
Loading what a HEDeST run wrote.

A run directory holds ``info.pickle``, which carries everything the analysis needs except
the slide, the segmentation and the cell crops (those are big, and the user already has
them). :func:`load_run` reads it, and also understands a directory of ``seed_*`` runs, which
it aggregates on the fly so a multi-seed study needs no extra step before it can be looked
at.

The pickle schema is versioned. It holds plain pandas and python objects only, so it does
not depend on the HEDeST classes and cannot be broken by a refactor of the model.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from loguru import logger

SCHEMA_VERSION = 2
INFO_NAME = "info.pickle"
AGGREGATED_NAME = "info_aggregated.pickle"


@dataclass
class HedestRun:
    """
    Everything one HEDeST run produced, as plain objects.

    Attributes:
        predictions_raw: Cells x cell types probabilities straight out of the model.
        predictions_adjusted: The same after Prior Probability Shift Adjustment, or None if
            no adjustment was applied.
        spot_dict: Spot id to the list of cell ids it holds.
        proportions: Spots x cell types, the deconvolution input.
        params: The settings of the run (hidden dims, learning rate, adjustment, seed...).
        history: Training and validation loss per epoch, or None.
        train_spot_dict: The spots used for training, or None.
        run_dir: Where the run was read from, or None for an in-memory run.
        n_seeds: Number of seeds behind the predictions; 1 for a single run.
        seeds: The seeds behind the predictions.
        predictions_std: Per-cell standard deviation across seeds, for an aggregate.
        agreement: Per-cell share of seeds agreeing with the majority cell type, for an
            aggregate.
        adjusted: Which of the two prediction tables :attr:`predictions` returns.
    """

    predictions_raw: pd.DataFrame
    spot_dict: Dict[str, List[str]]
    proportions: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)
    predictions_adjusted: Optional[pd.DataFrame] = None
    history: Optional[Dict[str, List[float]]] = None
    train_spot_dict: Optional[Dict[str, List[str]]] = None
    run_dir: Optional[Path] = None
    n_seeds: int = 1
    seeds: List[int] = field(default_factory=list)
    predictions_std: Optional[pd.DataFrame] = None
    agreement: Optional[pd.Series] = None
    adjusted: bool = True

    def __post_init__(self) -> None:
        if self.adjusted and self.predictions_adjusted is None:
            logger.warning("This run holds no adjusted predictions; falling back to the raw ones.")
            self.adjusted = False

    def __repr__(self) -> str:
        name = self.name
        seeds = f"{self.n_seeds} seeds" if self.n_seeds > 1 else f"seed {self.params.get('rs', '?')}"
        return (
            f"HedestRun({name}, {len(self.predictions_raw)} cells, {len(self.ct_list)} cell types, "
            f"{len(self.spot_dict)} spots, {seeds}, adjusted={self.adjusted})"
        )

    @property
    def name(self) -> str:
        """A short name for titles and file names."""

        return self.run_dir.name if self.run_dir is not None else "hedest-run"

    @property
    def predictions(self) -> pd.DataFrame:
        """The prediction table to analyse, adjusted or raw according to :attr:`adjusted`."""

        if self.adjusted and self.predictions_adjusted is not None:
            return self.predictions_adjusted

        return self.predictions_raw

    @property
    def ct_list(self) -> List[str]:
        """The cell types, in model-class order."""

        return list(self.predictions_raw.columns)

    @property
    def is_aggregate(self) -> bool:
        """True when the predictions are the mean over several seeds."""

        return self.n_seeds > 1

    def use(self, adjusted: bool) -> "HedestRun":
        """
        Returns the same run, reading the other prediction table.

        Args:
            adjusted: True for the adjusted predictions, False for the raw ones.

        Returns:
            A shallow copy of the run.
        """

        import copy

        other = copy.copy(self)
        other.adjusted = adjusted and self.predictions_adjusted is not None

        return other

    def to_dict(self) -> Dict[str, Any]:
        """Returns the serialisable form written to ``info.pickle``."""

        return {
            "version": SCHEMA_VERSION,
            "params": self.params,
            "spot_dict": self.spot_dict,
            "train_spot_dict": self.train_spot_dict,
            "proportions": self.proportions,
            "history": self.history,
            "predictions": self.predictions_raw,
            "predictions_adjusted": self.predictions_adjusted,
            "n_seeds": self.n_seeds,
            "seeds": self.seeds,
            "predictions_std": self.predictions_std,
            "agreement": self.agreement,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """
        Writes the run to a pickle file.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self.to_dict(), handle)
        logger.info(f"Run written to {path}")

        return path


def _from_dict(payload: Dict[str, Any], run_dir: Optional[Path], adjusted: bool) -> HedestRun:
    """
    Builds a run from the contents of a pickle file.

    Args:
        payload: The unpickled dictionary.
        run_dir: The directory it was read from.
        adjusted: Whether to analyse the adjusted predictions.

    Returns:
        The run.
    """

    version = payload.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"This file was written by another version of HEDeST (schema {version!r}, expected "
            f"{SCHEMA_VERSION}). Re-run the model, or re-aggregate the seeds, to refresh it."
        )

    return HedestRun(
        predictions_raw=payload["predictions"],
        predictions_adjusted=payload.get("predictions_adjusted"),
        spot_dict=payload["spot_dict"],
        proportions=payload["proportions"],
        params=payload.get("params", {}),
        history=payload.get("history"),
        train_spot_dict=payload.get("train_spot_dict"),
        run_dir=run_dir,
        n_seeds=payload.get("n_seeds", 1),
        seeds=payload.get("seeds", []),
        predictions_std=payload.get("predictions_std"),
        agreement=payload.get("agreement"),
        adjusted=adjusted,
    )


def load_run(path: Union[str, Path], adjusted: bool = True, aggregate: bool = True) -> HedestRun:
    """
    Loads a run, an aggregate of seeds, or a folder of seed runs.

    Args:
        path: Either a pickle file, or a directory holding ``info.pickle``,
            ``info_aggregated.pickle`` or a set of ``seed_*`` subdirectories.
        adjusted: Whether to analyse the adjusted predictions. The raw ones stay available.
        aggregate: For a folder of seeds with no ``info_aggregated.pickle``, whether to
            aggregate them in memory. With False, the first seed is returned instead.

    Returns:
        The run.

    Raises:
        FileNotFoundError: If nothing loadable is found at that path.
    """

    path = Path(path)

    if path.is_file():
        with open(path, "rb") as handle:
            return _from_dict(pickle.load(handle), path.parent, adjusted)

    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a directory.")

    for name in (AGGREGATED_NAME, INFO_NAME):
        candidate = path / name
        if candidate.exists():
            logger.info(f"Loading {candidate}")
            with open(candidate, "rb") as handle:
                return _from_dict(pickle.load(handle), path, adjusted)

    seed_runs = load_seed_runs(path, adjusted=adjusted)
    if not seed_runs:
        raise FileNotFoundError(f"No {INFO_NAME}, {AGGREGATED_NAME} or seed_*/{INFO_NAME} found under {path}.")

    if not aggregate or len(seed_runs) == 1:
        return seed_runs[0]

    from hedest.analysis.seeds import SeedEnsemble

    logger.info(f"Aggregating {len(seed_runs)} seed runs found under {path}")

    return SeedEnsemble(seed_runs).to_run(run_dir=path)


def load_seed_runs(path: Union[str, Path], adjusted: bool = True) -> List[HedestRun]:
    """
    Loads every ``seed_*`` run of a directory.

    Args:
        path: The directory holding the ``seed_*`` subdirectories.
        adjusted: Whether to analyse the adjusted predictions.

    Returns:
        The runs, ordered by seed number.
    """

    path = Path(path)
    runs = []

    for seed_dir in sorted(path.glob("seed_*"), key=lambda p: _seed_number(p.name)):
        info = seed_dir / INFO_NAME
        if not info.exists():
            logger.warning(f"[skip] no {INFO_NAME} in {seed_dir}")
            continue
        with open(info, "rb") as handle:
            run = _from_dict(pickle.load(handle), seed_dir, adjusted)
        if not run.seeds:
            run.seeds = [_seed_number(seed_dir.name)]
        runs.append(run)

    return runs


def _seed_number(name: str) -> int:
    """
    Extracts the seed number of a ``seed_*`` directory name, for sorting.

    Args:
        name: The directory name.

    Returns:
        The number, or -1 when there is none.
    """

    tail = name.split("_")[-1]

    return int(tail) if tail.isdigit() else -1


__all__ = ["HedestRun", "load_run", "load_seed_runs", "SCHEMA_VERSION", "INFO_NAME", "AGGREGATED_NAME"]
