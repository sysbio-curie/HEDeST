"""Cell-level feature extraction for HEDeST.

The extractors need heavier dependencies than the rest of the package (timm, openslide),
so they are imported lazily: ``from hedest.features import extract_hoptimus_embeddings``
works, but merely importing ``hedest`` does not pull timm in.
"""
from __future__ import annotations

from typing import Any

__all__ = ["extract_hoptimus_embeddings"]


def __getattr__(name: str) -> Any:
    if name == "extract_hoptimus_embeddings":
        from hedest.features.hoptimus import extract_hoptimus_embeddings

        return extract_hoptimus_embeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
