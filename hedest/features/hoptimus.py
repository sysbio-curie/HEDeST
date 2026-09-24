"""H-Optimus-0 applied as intended: tile-level inference + cell-level pooling.

Instead of feeding isolated cell crops to the model, whole H&E tiles are fed at the
resolution H-Optimus-0 was trained for, the *patch tokens* are kept, and they are pooled
per cell using the HoVer-Net segmentation. Each cell embedding therefore carries the
context of its tile.

Model card (https://huggingface.co/bioptimus/H-optimus-0):
  "H-optimus-0 expects images of size 224x224 that were extracted at 0.5 microns
   per pixel."
=> tile field of view = 224 * 0.5 = 112 um. A slide at 0.274 um/px (40x) therefore has
tiles of round(112/mpp) ~= 409 native px, resized to 224x224 (bicubic, antialiased).

Tiling / assignment:
  Tiles sit on a half-tile-stride grid (spacing = tile_native/2). Every cell is assigned
  to the tile whose centre is nearest its centroid, so the cell always falls inside the
  central half of its tile: the whole nucleus is inside the tile and there are >= 28 um
  of context on every side. Only tiles that own at least one cell are read and run
  through the model.

Cell pooling:
  ViT-g/14 at 224 px -> 16x16 = 256 patch tokens (each covering 14 px = 7 um), plus
  prefix tokens (1 CLS + 4 registers) which are dropped. For each cell its HoVer-Net
  contour is rasterised in tile-local 224-space, the nucleus pixels falling in each of
  the 256 patches are counted, and the area-weighted mean of the corresponding patch
  tokens is taken -> a 1536-d cell embedding. Cells too small to rasterise fall back to
  the single patch containing their centroid.

Coordinates: the HoVer-Net JSON 'centroid' and 'contour' are (x, y) = (column, row) in
level-0 slide pixels. 'bbox' is in a different frame and is NOT used.

Cell ids follow the convention of the rest of HEDeST (``extract_images_hn`` and
``map_cells_to_spots``): cells are numbered by their position in the segmentation file,
as strings, so the embeddings line up with the spot dictionary and the image dictionary.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import openslide
import torch
import typer
from loguru import logger
from PIL import Image
from skimage.draw import polygon as sk_polygon
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from hedest.config import TqdmToLogger
from hedest.slide import read_mpp

tqdm_out = TqdmToLogger(logger, level="INFO")

HOPTIMUS_MODEL = "hf-hub:bioptimus/H-optimus-0"
HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)
TARGET_MPP = 0.5  # model card
TILE_PX = 224  # model card
PATCH = 14  # ViT-g/14
GRID = TILE_PX // PATCH  # 16x16 patch tokens
EMBED_DIM = 1536
TILE_UM = TILE_PX * TARGET_MPP  # 112 um


class TileDataset(Dataset):
    """Reads tiles from the WSI and returns normalised 224x224 tensors."""

    def __init__(self, slide_path: str, origins: np.ndarray, tile_native: int) -> None:
        self.slide_path = slide_path
        self.origins = origins  # (T, 2) int, level-0 (x, y) top-left
        self.tile_native = tile_native
        self._slide = None
        self.mean = torch.tensor(HOPTIMUS_MEAN).view(3, 1, 1)
        self.std = torch.tensor(HOPTIMUS_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.origins)

    def _get_slide(self) -> openslide.OpenSlide:
        if self._slide is None:  # open lazily, per worker process
            self._slide = openslide.OpenSlide(self.slide_path)
        return self._slide

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        x0, y0 = (int(v) for v in self.origins[i])
        img = self._get_slide().read_region((x0, y0), 0, (self.tile_native, self.tile_native)).convert("RGB")
        # native px (112 um) -> 224 px == 0.5 um/px, the model's training scale.
        # PIL BICUBIC scales its kernel when downsampling, so this is antialiased.
        img = img.resize((TILE_PX, TILE_PX), resample=Image.BICUBIC)
        tensor = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor, i


def build_tiles(centroids: np.ndarray, tile_native: int, width: int, height: int) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Assigns every cell to the nearest half-stride grid tile.

    Args:
        centroids: (N, 2) array of level-0 (x, y) cell centroids.
        tile_native: Tile size in level-0 pixels.
        width: Slide width in level-0 pixels.
        height: Slide height in level-0 pixels.

    Returns:
        The (T, 2) array of level-0 top-left (x, y) tile origins, and for each tile the
        list of cell positions it owns.
    """

    g = max(tile_native // 2, 1)
    half = tile_native // 2
    gi = np.rint(centroids[:, 0] / g).astype(np.int64)
    gj = np.rint(centroids[:, 1] / g).astype(np.int64)
    cx, cy = gi * g, gj * g
    x0 = np.clip(cx - half, 0, max(width - tile_native, 0))
    y0 = np.clip(cy - half, 0, max(height - tile_native, 0))
    key = np.stack([x0, y0], axis=1)
    origins, inverse = np.unique(key, axis=0, return_inverse=True)
    per_tile: List[List[int]] = [[] for _ in range(len(origins))]
    for cell_pos, t in enumerate(inverse):
        per_tile[t].append(cell_pos)

    return origins.astype(np.int64), per_tile


def pool_cell(
    contour_xy: np.ndarray,
    x0: int,
    y0: int,
    scale: float,
    tokens: np.ndarray,
    centroid_xy: np.ndarray,
) -> np.ndarray:
    """
    Area-weighted mean of the patch tokens covered by one nucleus.

    Args:
        contour_xy: (M, 2) contour in level-0 (x, y).
        x0: Level-0 x of the tile origin.
        y0: Level-0 y of the tile origin.
        scale: Ratio between the 224 px tile and its native size.
        tokens: (256, 1536) patch tokens of the tile.
        centroid_xy: Level-0 (x, y) centroid, used when the contour cannot be rasterised.

    Returns:
        The (1536,) cell embedding.
    """

    # to tile-local 224-space
    cx = (contour_xy[:, 0] - x0) * scale
    cy = (contour_xy[:, 1] - y0) * scale
    rr, cc = sk_polygon(cy, cx, (TILE_PX, TILE_PX))  # rows=y, cols=x, clipped to the tile
    if rr.size == 0:
        px = int(np.clip((centroid_xy[0] - x0) * scale, 0, TILE_PX - 1))
        py = int(np.clip((centroid_xy[1] - y0) * scale, 0, TILE_PX - 1))
        return tokens[(py // PATCH) * GRID + (px // PATCH)]
    idx = (rr // PATCH) * GRID + (cc // PATCH)
    w = np.bincount(idx, minlength=GRID * GRID).astype(np.float32)

    return (w @ tokens) / w.sum()


def _load_segmentation(json_path: str) -> Tuple[List[dict], Optional[float]]:
    """
    Loads the nuclei of a segmentation file, in file order.

    Args:
        json_path: Path to the HoVer-Net segmentation file.

    Returns:
        The list of nucleus records and the microns per pixel recorded in the file, if any.
    """

    with open(json_path) as f:
        seg = json.load(f)

    nuclei = list(seg["nuc"].values())

    mpp = seg.get("mpp")
    try:
        mpp = float(mpp) if mpp is not None else None
    except (TypeError, ValueError):
        mpp = None

    return nuclei, mpp


@torch.inference_mode()
def extract_hoptimus_embeddings(
    slide_path: str,
    json_path: str,
    out_path: Optional[str] = None,
    mpp: Optional[float] = None,
    batch_size: int = 32,
    num_workers: int = 6,
    limit_tiles: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes a tile-context H-Optimus-0 embedding for every segmented cell of a slide.

    Args:
        slide_path: Path to the whole-slide image.
        json_path: Path to the HoVer-Net segmentation file.
        out_path: Where to save the embedding dictionary. If None, nothing is saved.
        mpp: Microns per pixel of the slide. If None, it is read from the segmentation
             file, then from the slide itself.
        batch_size: Number of tiles per forward pass.
        num_workers: Number of workers reading tiles from the slide.
        limit_tiles: Only process the first N tiles. For debugging.
        device: Device to run the model on. Defaults to CUDA when available.

    Returns:
        A dictionary mapping each cell id to its (1536,) float32 embedding, in the order
        the cells appear in the segmentation file.

    Raises:
        ValueError: If the resolution of the slide cannot be determined.
    """

    import timm  # imported here so the rest of HEDeST does not need it

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    nuclei, seg_mpp = _load_segmentation(json_path)
    cell_ids = [str(i) for i in range(len(nuclei))]

    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions
    slide_mpp = read_mpp(slide)
    slide.close()

    mpp = mpp if mpp is not None else (seg_mpp if seg_mpp is not None else slide_mpp)
    if mpp is None or mpp <= 0:
        raise ValueError(
            f"The resolution of {slide_path} is unknown: neither {json_path} nor the slide carries an mpp. "
            "Pass it explicitly."
        )

    tile_native = int(round(TILE_UM / mpp))
    scale = TILE_PX / tile_native

    centroids = np.array([n["centroid"] for n in nuclei], dtype=np.float64)
    origins, per_tile = build_tiles(centroids, tile_native, width, height)
    if limit_tiles is not None:
        keep = min(limit_tiles, len(origins))
        origins, per_tile = origins[:keep], per_tile[:keep]

    logger.info(
        f"{len(cell_ids)} cells | slide {width}x{height} @ {mpp:.4f} um/px | "
        f"tile {tile_native}px -> {TILE_PX} ({TARGET_MPP} um/px) | {len(origins)} tiles | "
        f"setup {time.time() - start:.1f}s"
    )

    model = timm.create_model(HOPTIMUS_MODEL, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.eval().to(device)
    n_prefix = int(getattr(model, "num_prefix_tokens", 5))
    if device == "cuda":
        model.half()
    logger.info(f"-> Model ready (prefix tokens={n_prefix}, device={device}).")

    contours = [np.asarray(n["contour"], dtype=np.float64) for n in nuclei]
    del nuclei

    out = np.zeros((len(cell_ids), EMBED_DIM), dtype=np.float32)
    filled = np.zeros(len(cell_ids), dtype=bool)

    loader = DataLoader(
        TileDataset(slide_path, origins, tile_native),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    start = time.time()
    for tiles, tile_idx in tqdm(loader, total=len(loader), desc="Embedding tiles", unit="batch", file=tqdm_out):
        tiles = tiles.to(device, non_blocking=True)
        if device == "cuda":
            tiles = tiles.half()
        feats = model.forward_features(tiles)  # (B, n_prefix+256, 1536)
        patch = feats[:, n_prefix:, :].float().cpu().numpy()  # (B, 256, 1536)
        for b, ti in enumerate(tile_idx.tolist()):
            toks = patch[b]
            x0, y0 = int(origins[ti][0]), int(origins[ti][1])
            for cp in per_tile[ti]:
                out[cp] = pool_cell(contours[cp], x0, y0, scale, toks, centroids[cp])
                filled[cp] = True

    elapsed = time.time() - start
    logger.info(
        f"-> Inference and pooling done in {elapsed:.1f}s ({len(origins) / max(elapsed, 1e-9):.1f} tiles/s), "
        f"cells filled {filled.sum()}/{len(cell_ids)}."
    )
    if limit_tiles is None and not filled.all():
        raise RuntimeError(f"{(~filled).sum()} cells got no embedding.")

    embeddings = {c: torch.from_numpy(out[i].copy()) for i, c in enumerate(cell_ids)}

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        tmp = out_path + ".tmp"
        torch.save(embeddings, tmp)
        os.replace(tmp, out_path)
        logger.info(f"-> Embeddings saved to {out_path} (dim={EMBED_DIM}, dtype=float32).")

    return embeddings


app = typer.Typer(help="Compute tile-context H-Optimus-0 embeddings for every segmented cell of a slide.")


@app.command()
def main(
    slide_path: str = typer.Argument(..., help="Path to the whole-slide image."),
    json_path: str = typer.Argument(..., help="Path to the HoVer-Net segmentation file."),
    out_path: str = typer.Argument(..., help="Where to save the {cell_id: embedding} dictionary (.pt)."),
    mpp: Optional[float] = typer.Option(None, help="Microns per pixel. Read from the file or the slide if omitted."),
    batch_size: int = typer.Option(32, help="Number of tiles per forward pass."),
    num_workers: int = typer.Option(6, help="Number of workers reading tiles from the slide."),
    limit_tiles: Optional[int] = typer.Option(None, help="Only process the first N tiles (debug)."),
) -> None:
    """Computes the cell embeddings of one slide."""

    extract_hoptimus_embeddings(
        slide_path=slide_path,
        json_path=json_path,
        out_path=out_path,
        mpp=mpp,
        batch_size=batch_size,
        num_workers=num_workers,
        limit_tiles=limit_tiles,
    )


if __name__ == "__main__":
    app()
