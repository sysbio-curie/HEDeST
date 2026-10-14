from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from spatialdata_io import xenium


def main(path: str) -> None:
    """
    Parses Xenium data from the given path and writes it to a Zarr format.

    Args:
        path: The base path where the data folder is located.
    """

    path_read = Path(path) / "data"
    path_write = Path(path) / "data.zarr"

    print("Parsing the data... ", end="")
    sdata = xenium(
        path=str(path_read),
        n_jobs=8,
        cell_boundaries=True,
        nucleus_boundaries=True,
        morphology_focus=True,
        cells_as_circles=True,
    )
    print("done")

    print("Writing the data... ", end="")
    if path_write.exists():
        shutil.rmtree(path_write)
    sdata.write(path_write)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and write Xenium data.")
    parser.add_argument(
        "path",
        type=str,
        help="The base path where the data folder is located.",
    )

    args = parser.parse_args()
    main(args.path)
