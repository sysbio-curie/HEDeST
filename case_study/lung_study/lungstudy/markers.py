"""
Marker genes of tertiary lymphoid structures (TLS), read per spot from the Visium counts.

cell2location could not see the cell types that tell a mature TLS from a lymphoid aggregate
(follicular dendritic cells, high endothelial venules, mature dendritic cells): they are not in
its single-cell reference. Their marker genes are in the counts, though. The list follows the
compartments of a mature TLS:

| compartment | cell type | genes |
|---|---|---|
| B-cell follicle | naive / memory B cells | MS4A1 (CD20), CD19, CD79A |
| germinal centre | germinal-centre B cells | BCL6, MKI67 (Ki-67), AICDA (AID), MME (CD10) |
| plasma cells | plasma cells | SDC1 (CD138), MZB1, JCHAIN, IGHG1, IGHA1 |
| follicular stroma | follicular dendritic cells | CR2 (CD21), FCER2 (CD23), CR1 (CD35) |
| T-cell zone | CD4 T, Tfh, CD8 T, Treg | CD3E, CD4, PDCD1 (PD-1), CXCR5, ICOS, CD8A, FOXP3 |
| antigen presentation | mature dendritic cells | LAMP3 (DC-LAMP), CCR7, CD83 |
| vasculature | high endothelial venules | CHST4 (makes the MECA-79 / PNAd epitope), ACKR1, SELE |
| stromal scaffold | fibroblastic reticular cells | PDPN, CCL19, CCL21 |
| TLS chemokine | FDCs, Tfh | CXCL13 |
| other | macrophages | CD68, CD163 |

GLYCAM1, the other PNAd carrier, is a pseudogene in humans and is absent from the counts.
Expression is per spot, in counts per 10,000 (CP10k): the spot is the only resolution there is.
"""
from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np
import pandas as pd

COMPARTMENTS: Dict[str, List[str]] = {
    "B-cell follicle": ["MS4A1", "CD19", "CD79A"],
    "germinal centre": ["BCL6", "MKI67", "AICDA", "MME"],
    "plasma cells": ["SDC1", "MZB1", "JCHAIN", "IGHG1", "IGHA1"],
    "follicular dendritic cells": ["CR2", "FCER2", "CR1"],
    "T-cell zone": ["CD3E", "CD4", "PDCD1", "CXCR5", "ICOS", "CD8A", "FOXP3"],
    "mature dendritic cells": ["LAMP3", "CCR7", "CD83"],
    "high endothelial venules": ["CHST4", "ACKR1", "SELE"],
    "stromal scaffold": ["PDPN", "CCL19", "CCL21"],
    "TLS chemokine": ["CXCL13"],
    "macrophages": ["CD68", "CD163"],
}
GENES = [g for genes in COMPARTMENTS.values() for g in genes]
COMPARTMENT_OF = {g: c for c, genes in COMPARTMENTS.items() for g in genes}


def read_counts(adata_path: str, genes: List[str] = GENES) -> pd.DataFrame:
    """Raw counts of ``genes`` per spot, plus each spot's total counts (``total_counts``)."""

    import anndata as ad

    adata = ad.read_h5ad(adata_path)
    present = [g for g in genes if g in adata.var_names]
    counts = adata[:, present].X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    total = np.asarray(adata.X.sum(axis=1)).ravel()
    table = pd.DataFrame(counts, index=adata.obs_names.astype(str), columns=present)
    table["total_counts"] = total

    return table


def cp10k(counts: pd.DataFrame) -> pd.DataFrame:
    """Counts per 10,000 of each spot's total (genes only)."""

    genes = [c for c in counts.columns if c != "total_counts"]
    total = counts["total_counts"].replace(0, np.nan)

    return counts[genes].div(total, axis=0) * 1e4
