"""
Downloads the single-cell reference of the Visium HD colon cancer paper.

The reference is the paper's own Chromium Fixed RNA Profiling (Flex) scFFPE-seq atlas, run on
the same eight FFPE blocks as the spatial data (de Oliveira et al., Nat Genet 57:1512-1523,
2025, https://doi.org/10.1038/s41588-025-02193-3). Two pieces live in two places:

- the counts, as 32 per-library matrices on GEO (GSE280318), 8 samples x 4 Flex libraries;
- the cell type labels, as one table in the paper's GitHub repository.

Downloading the 32 matrices individually avoids the 56 GB ``GSE280318_RAW.tar``; together they
are about 560 MB. Everything is cached, so re-running only fetches what is missing.

Usage: python fetch_flex.py [--cache DIR]
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CACHE = Path("/cluster/CBIO/data1/lgortana/.cache/visiumhd_flex")

GEO_SAMPLE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8594nnn/{gsm}/suppl/{name}"
METADATA_URL = (
    "https://raw.githubusercontent.com/10XGenomics/HumanColonCancer_VisiumHD/main/"
    "MetaData/SingleCell_MetaData.csv.gz"
)
METADATA_NAME = "SingleCell_MetaData.csv.gz"

# The three multiplexed Flex pools, each sequenced as four libraries. POOL_SAMPLES maps a probe
# barcode to the FFPE block carrying it; sorting those barcodes reproduces the order cellranger
# aggr used, which is what fixes the barcode suffixes (see ``libraries`` below).
POOL_GSMS: dict[int, list[int]] = {
    1: [8594533, 8594534, 8594535, 8594536],
    2: [8594537, 8594538, 8594539, 8594540],
    3: [8594541, 8594542, 8594543, 8594544],
}
POOL_SAMPLES: dict[int, dict[str, str]] = {
    1: {"BC1": "P2CRC", "BC2": "P4CRC"},
    2: {"BC1": "P1CRC", "BC2": "P3CRC", "BC3": "P5CRC", "BC4": "P2NAT"},
    3: {"BC3": "P5NAT", "BC4": "P3NAT"},
}


def libraries() -> list[dict]:
    """
    The 32 Flex libraries, each with the GEO file to fetch and the aggr suffix it carries.

    ``cellranger aggr`` numbered its 32 rows pool by pool, library by library, and within a
    library by probe barcode. That ordering is the only link between the labels (whose barcodes
    end in ``-1`` .. ``-32``) and these files (whose barcodes all end in ``-1``), so it is
    recorded here and verified against the cell counts at build time.
    """

    out, offset = [], 0
    for pool in sorted(POOL_GSMS):
        gsms, samples = POOL_GSMS[pool], POOL_SAMPLES[pool]
        for lane, gsm in enumerate(gsms, start=1):
            for rank, barcode in enumerate(sorted(samples), start=1):
                sample = samples[barcode]
                library = f"P{pool}L{lane}"
                name = f"GSM{gsm}_{library}_{sample}_{barcode}_count_filtered_feature_bc_matrix.h5"
                out.append(
                    {
                        "pool": pool,
                        "lane": lane,
                        "library": library,
                        "sample": sample,
                        "barcode": barcode,
                        "suffix": offset + (lane - 1) * len(samples) + rank,
                        "gsm": f"GSM{gsm}",
                        "name": name,
                        "url": GEO_SAMPLE_URL.format(gsm=f"GSM{gsm}", name=name),
                    }
                )
        offset += len(gsms) * len(samples)

    return out


def download(url: str, path: Path) -> bool:
    """Fetches ``url`` to ``path`` unless it is already there. True when it had to download."""

    if path.exists() and path.stat().st_size > 0:
        return False

    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError) as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"failed to download {url}: {error}") from error

    # Rename only once the body is complete, so an interrupted run cannot leave a short file
    # that the next run would mistake for a finished download.
    partial.rename(path)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="where to keep the downloads")
    arguments = parser.parse_args()

    cache = arguments.cache
    cache.mkdir(parents=True, exist_ok=True)

    fetched = download(METADATA_URL, cache / METADATA_NAME)
    print(f"{'fetched' if fetched else 'cached '} {METADATA_NAME}")

    for entry in libraries():
        fetched = download(entry["url"], cache / entry["name"])
        size = (cache / entry["name"]).stat().st_size / 1e6
        print(
            f"{'fetched' if fetched else 'cached '} suffix {entry['suffix']:>2}  "
            f"{entry['library']}  {entry['sample']:<6} {entry['barcode']}  {size:>6.1f} MB"
        )

    total = sum(f.stat().st_size for f in cache.iterdir() if f.is_file()) / 1e9
    print(f"\n{cache}: {total:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
