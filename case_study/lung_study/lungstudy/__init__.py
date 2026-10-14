"""
LungScape case study: what HEDeST shows that the deconvolution proportions cannot.

Modules:

- :mod:`lungstudy.paths` — the samples, the settings and where every file lives;
- :mod:`lungstudy.cache` — per-sample tables extracted once from the LungScape outputs
  (cells, predictions at every level, spots, per-seed spot predictions);
- :mod:`lungstudy.views` — slide views: H&E, spots, pies, cells, density maps;
- :mod:`lungstudy.validation` — checks of HEDeST that need no ground truth;
- :mod:`lungstudy.coloc` — colocalization from proportions and from cells;
- :mod:`lungstudy.tls` — lymphoid aggregates from cells and from spots;
- :mod:`lungstudy.niches` — cellular neighbourhoods and immune infiltration.

Nothing here writes outside ``case_study/lung_study``.
"""
from __future__ import annotations

import sys

# Importing hedest must not write byte-code into the HEDeST package.
sys.dont_write_bytecode = True
