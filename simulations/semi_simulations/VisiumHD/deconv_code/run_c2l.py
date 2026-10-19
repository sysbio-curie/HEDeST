"""
One cell2location run: a sample's reference and pseudo-Visium spots in, proportions out.

The reference is ``reference_full.h5ad``, whose cell types refine all three annotation levels, so
one run per sample is enough; ``aggregate.py`` sums its columns into each level afterwards.

Every method parameter is cell2location's own documented default except two:

* ``N_cells_per_location`` is read from the data. It is normally estimated by eye from histology,
  but these spots were built on segmented nuclei, so ``obs["n_cells"]`` holds the real count and
  its mean is the right value;
* ``detection_alpha=20``, which upstream recommends over the ``200`` default whenever within-slide
  technical variability is expected.

The reference is checked for a ``.raw`` before anything else. cell2location does
``adata.X = adata.raw.X.copy()`` inside a bare ``try``, so a reference that carries one is
silently deconvolved from it instead of from ``X`` — a failure with no error message.

    python run_c2l.py --sample P1

Writes into ``deconv_results/runs/<sample>/``: the posterior means and 5% quantiles of cell
abundance, the row-normalised proportions, the training curves, and a provenance JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

ABUNDANCE_KEY = "q05_cell_abundance_w_sf"


def load_spots(sample: paths.Sample) -> tuple[ad.AnnData, dict]:
    """
    The sample's spots with integer counts in ``X``, ready for the spatial model.

    ``X`` sums bin2cell's *destriped* 2 um bins, and destriping is a multiplicative correction, so
    every entry is fractional. cell2location's likelihood is a Gamma-Poisson and rejects anything
    that is not a non-negative integer, so the counts are rounded here. Reconstructing undestriped
    counts was the alternative, but that would deconvolve a different object from the one the
    ground truth and the rest of the study are defined on. The cost of rounding is reported and
    recorded: it moves well under 1% of the counts.
    """

    adata = ad.read_h5ad(sample.spots)
    adata.var_names_make_unique()
    adata.obs["n_cells"] = adata.obs["n_cells"].astype(float)

    before = float(adata.X.sum())
    data = adata.X.data
    zeroed = int((np.rint(data) == 0).sum())
    adata.X.data = np.rint(data)
    adata.X.eliminate_zeros()
    after = float(adata.X.sum())
    rounding = {
        "nnz_before": int(data.size),
        "nnz_zeroed_by_rounding": zeroed,
        "total_before": before,
        "total_after": after,
        "total_change_pct": 100.0 * (after - before) / before,
    }
    print(
        f"  rounded counts to integers: {zeroed:,} of {data.size:,} nonzeros became 0, "
        f"total {before:,.0f} -> {after:,.0f} ({rounding['total_change_pct']:+.3f}%)",
        flush=True,
    )

    return adata, rounding


def load_reference(sample: paths.Sample) -> ad.AnnData:
    """The reference, checked for the ``.raw`` that would silently replace ``X``."""

    adata = ad.read_h5ad(sample.reference)
    if adata.raw is not None:
        raise SystemExit(
            f"{sample.reference} carries a .raw; cell2location would deconvolve from it "
            "instead of from X. Drop it before running."
        )
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.var_names_make_unique()

    return adata


def signatures(reference: ad.AnnData, out: Path) -> pd.DataFrame:
    """
    Per-cell-type expression signatures, from cell2location's negative-binomial regression.

    ``filter_genes`` and the regression settings are upstream's documented defaults. The batch key
    is the Flex library, which is the technical unit here; patient is deliberately not treated as
    a batch, since for the tumour compartment it is the biology the reference is built around.
    """

    from cell2location.models import RegressionModel
    from cell2location.utils.filtering import filter_genes

    # No overrides: the cutoffs are the ones filter_genes itself defaults to.
    selected = filter_genes(reference)
    plt.savefig(out / "gene_filter.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    reference = reference[:, selected].copy()
    print(f"  reference after gene filter: {reference.n_obs:,} cells x {reference.n_vars:,} genes", flush=True)

    RegressionModel.setup_anndata(reference, labels_key=paths.CELLTYPE_KEY, batch_key=paths.BATCH_KEY)
    model = RegressionModel(reference)
    model.train(max_epochs=paths.REFERENCE_EPOCHS, batch_size=2500, train_size=1, lr=0.002)

    model.plot_history(20)
    plt.savefig(out / "reference_elbo.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    reference = model.export_posterior(
        reference, sample_kwargs={"num_samples": paths.POSTERIOR_SAMPLES, "batch_size": 2500}
    )
    model.save(str(out / "reference_model"), overwrite=True)

    # The per-cluster means are the signature the spatial model consumes. Upstream names those
    # columns by pasting the summary, site and factor together, so they are renamed here from
    # `factor_names` by position rather than by stripping a prefix that upstream could change.
    column = "means_per_cluster_mu_fg"
    frame = reference.varm[column] if column in reference.varm else reference.var
    kinds = list(reference.uns["mod"]["factor_names"])
    inferred = pd.DataFrame({kind: frame[f"{column}_{kind}"].to_numpy() for kind in kinds}, index=reference.var_names)
    inferred.to_csv(out / "signatures.csv")

    return inferred


def deconvolve(spots: ad.AnnData, inferred: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Map the signatures onto the spots and return row-normalised proportions."""

    from cell2location.models import Cell2location

    shared = [g for g in spots.var_names if g in inferred.index]
    print(f"  shared genes: {len(shared):,} of {spots.n_vars:,} spot genes", flush=True)
    spots = spots[:, shared].copy()
    inferred = inferred.loc[shared]

    # The spots were built on segmented nuclei, so the expected number of cells per spot is known
    # rather than guessed.
    per_location = float(spots.obs["n_cells"].mean())
    print(f"  N_cells_per_location = {per_location:.2f} (observed mean)", flush=True)

    Cell2location.setup_anndata(spots)
    model = Cell2location(
        spots,
        cell_state_df=inferred,
        N_cells_per_location=per_location,
        detection_alpha=paths.DETECTION_ALPHA,
    )
    model.train(max_epochs=paths.SPATIAL_EPOCHS, batch_size=None, train_size=1)

    model.plot_history(1000)
    plt.savefig(out / "spatial_elbo.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    spots = model.export_posterior(
        spots, sample_kwargs={"num_samples": paths.POSTERIOR_SAMPLES, "batch_size": spots.n_obs}
    )
    model.save(str(out / "spatial_model"), overwrite=True)

    # Upstream builds these column names as summary + prefix + site + factor, with no separator
    # after the summary ("q05cell_abundance_w_sf_Goblet"), so they are replaced from
    # `factor_names`, which is the order the columns were written in.
    abundance = spots.obsm[ABUNDANCE_KEY].copy()
    kinds = list(spots.uns["mod"]["factor_names"])
    if len(kinds) != abundance.shape[1]:
        raise SystemExit(f"{len(kinds)} factor names for {abundance.shape[1]} abundance columns")
    abundance.columns = kinds
    abundance.to_csv(out / "abundance.csv")

    # cell2location estimates absolute abundance; the ground truth is a composition, so the
    # comparable quantity is each spot's abundance divided by its total.
    totals = abundance.sum(axis=1).replace(0, np.nan)
    proportions = abundance.div(totals, axis=0).fillna(0.0)
    proportions.index.name = "spot_id"

    return proportions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, choices=paths.SAMPLES)
    arguments = parser.parse_args()

    name = arguments.sample
    sample = paths.sample(name)
    out = paths.run_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name} / {sample.reference.name} -> {out}", flush=True)

    spots, rounding = load_spots(sample)
    reference = load_reference(sample)
    print(
        f"  spots: {spots.n_obs:,} x {spots.n_vars:,} | reference: {reference.n_obs:,} x {reference.n_vars:,}",
        flush=True,
    )
    kinds = sorted(reference.obs[paths.CELLTYPE_KEY].astype(str).unique())
    print(f"  {len(kinds)} cell types: {', '.join(kinds)}", flush=True)

    inferred = signatures(reference, out)
    proportions = deconvolve(spots, inferred, out)
    proportions.to_csv(out / "proportions.csv")

    (out / "provenance.json").write_text(
        json.dumps(
            {
                "sample": name,
                "reference_file": str(sample.reference),
                "spots_file": str(sample.spots),
                "n_spots": int(proportions.shape[0]),
                "cell_types": list(proportions.columns),
                "n_cells_per_location": float(spots.obs["n_cells"].mean()),
                "count_rounding": rounding,
                "detection_alpha": paths.DETECTION_ALPHA,
                "reference_epochs": paths.REFERENCE_EPOCHS,
                "spatial_epochs": paths.SPATIAL_EPOCHS,
                "batch_key": paths.BATCH_KEY,
                "finished": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            },
            indent=2,
        )
    )
    print(f"  wrote {out / 'proportions.csv'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
