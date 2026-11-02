"""
LungScape case study: what HEDeST shows that the deconvolution proportions cannot.

Modules:

- :mod:`lungstudy.paths` — the samples, the settings and where every file lives;
- :mod:`lungstudy.cache` — per-sample tables extracted once from the LungScape outputs
  (cells, predictions at every level, spots, per-seed spot predictions, touching cells,
  marker genes);
- :mod:`lungstudy.style` — one colour per cell type, shared by pies and cells;
- :mod:`lungstudy.views` — slide views: H&E, spots, pies, cells, density maps;
- :mod:`lungstudy.validation` — held-out spots, for the pre-check notebooks;
- :mod:`lungstudy.contacts` — which cells touch each other;
- :mod:`lungstudy.coloc` — colocalization from proportions and from cells;
- :mod:`lungstudy.tls` — lymphoid aggregates from cells and from spots, their B and T zones;
- :mod:`lungstudy.markers` — TLS marker genes per spot;
- :mod:`lungstudy.infiltration` — the epithelial compartment and the immune cells inside it;
- :mod:`lungstudy.geometry` — distances to outlines and densities by distance;
- :mod:`lungstudy.figures` — the colocalization zone plot, labels, H&E crops.

Nothing here writes outside ``case_study/lung_study``.
"""
from __future__ import annotations

import sys

# Importing hedest must not write byte-code into the HEDeST package.
sys.dont_write_bytecode = True
