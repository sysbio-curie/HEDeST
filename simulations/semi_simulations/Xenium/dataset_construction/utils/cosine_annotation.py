from __future__ import annotations

import argparse

import anndata as ad
import scanpy as sc
import spatialdata as sd
from spatial_tools import transfer_annot_batched


def main(
    sc_adata_path: str,
    xenium_data_path: str,
    out_dir: str,
    min_counts: int = 0,
    cell_type_key: str = "cell_type",
    k_neighb: int = 20,
    batch_size: int = 5000,
) -> None:
    """
    Annotates Xenium spatial data using a reference scRNA-seq dataset via cosine similarity.

    Args:
        sc_adata_path: Path to the scRNA-seq AnnData file (h5ad).
        xenium_data_path: Path to the Xenium data (zarr).
        out_dir: Output directory for processed data.
        min_counts: Minimum number of counts for filtering cells.
        cell_type_key: Key for cell type annotations.
        k_neighb: Number of neighbors to consider for annotation.
        batch_size: Number of cells to process at a time.
    """

    # Load scRNA-seq data
    sc_adata = ad.read_h5ad(sc_adata_path)
    sc.pp.normalize_total(sc_adata)
    sc.pp.log1p(sc_adata)
    sc.pp.highly_variable_genes(sc_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]
    print("--> scRNA-seq data loaded.")

    # Load xenium data
    sdata = sd.read_zarr(xenium_data_path)
    xenium_adata = sdata.tables["table"]
    xenium_adata.var["SYMBOL"] = xenium_adata.var_names
    xenium_adata.var.set_index("gene_ids", drop=True, inplace=True)
    print("--> Xenium data loaded.")

    # Transfer annotations using batched processing
    print("--> Transferring annotations...")
    xenium_adata = transfer_annot_batched(
        sc_adata, xenium_adata, cell_type_key, min_counts, k_neighb, batch_size=batch_size
    )
    print("--> Annotations transferred.")

    xenium_adata.obs["cell_type"] = xenium_adata.obsm["annotation_confidence"].idxmax(axis=1)

    # Save results
    xenium_adata.write(out_dir)
    print("--> Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation of Xenium data with single-cell dataset.")
    parser.add_argument("sc_adata_path", type=str, help="Path to the scRNA-seq AnnData file (h5ad).")
    parser.add_argument("xenium_data_path", type=str, help="Path to the Xenium data (zarr).")
    parser.add_argument("out_dir", type=str, help="Output directory for processed data.")
    parser.add_argument("--min_counts", type=int, default=0, help="Minimum number of counts for filtering cells.")
    parser.add_argument("--cell_type_key", type=str, default="cell_type", help="Key for cell type annotations.")
    parser.add_argument("--k_neighb", type=int, default=20, help="Number of neighbors to consider for annotation.")
    parser.add_argument("--batch_size", type=int, default=5000, help="Number of cells to process at a time.")

    args = parser.parse_args()

    main(
        args.sc_adata_path,
        args.xenium_data_path,
        args.out_dir,
        args.min_counts,
        args.cell_type_key,
        args.k_neighb,
        args.batch_size,
    )
