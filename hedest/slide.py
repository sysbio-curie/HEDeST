from __future__ import annotations

import os
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import openslide
from loguru import logger
from PIL import Image


class SlideError(RuntimeError):
    """Raised when a slide cannot be used by HEDeST and cannot be repaired."""


def _mpp_from_resolution(props: Dict[str, str]) -> Optional[float]:
    """
    Reads the microns per pixel from the TIFF resolution tags.

    Args:
        props: Slide properties, as given by OpenSlide.

    Returns:
        The microns per pixel, or None if the tags are missing or unusable.
    """

    try:
        xres = float(props["tiff.XResolution"])
    except (KeyError, TypeError, ValueError):
        return None

    if xres <= 0:
        return None

    unit = str(props.get("tiff.ResolutionUnit", "")).lower()
    if unit.startswith("cent"):
        return 1e4 / xres
    if unit.startswith("inch"):
        return 25400.0 / xres

    return None


def _mpp_from_ome_comment(props: Dict[str, str]) -> Optional[float]:
    """
    Reads the microns per pixel from an OME-XML comment.

    Args:
        props: Slide properties, as given by OpenSlide.

    Returns:
        The microns per pixel, or None if the comment is missing or unusable.
    """

    comment = props.get("openslide.comment")
    if not comment:
        return None

    match = re.search(r'PhysicalSizeX="([^"]*)"', comment)
    if match is None:
        return None

    try:
        value = float(match.group(1))
    except ValueError:
        return None

    return value if value > 0 else None


def _mpp_from_tifffile(slide_path: str) -> Optional[float]:
    """
    Reads the microns per pixel straight from the TIFF tags.

    Used for the slides OpenSlide refuses to open, typically striped TIFFs: their
    resolution is still readable, so they can be converted without asking for it.

    Args:
        slide_path: Path to the slide.

    Returns:
        The microns per pixel, or None if the tags are missing or unusable.
    """

    try:
        import tifffile

        with tifffile.TiffFile(slide_path) as handle:
            tags = handle.pages[0].tags
            xres = tags["XResolution"].value
            unit = int(getattr(tags.get("ResolutionUnit"), "value", 2))
    except Exception:
        return None

    try:
        xres = float(xres[0]) / float(xres[1]) if isinstance(xres, tuple) else float(xres)
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return None

    if xres <= 0:
        return None

    if unit == 3:  # centimetre
        return 1e4 / xres
    if unit == 2:  # inch
        return 25400.0 / xres

    return None


def read_mpp(slide: openslide.OpenSlide) -> Optional[float]:
    """
    Reads the microns per pixel of a slide, trying every source we know of.

    The order is: the OpenSlide property (set by vendor formats), then the TIFF
    resolution tags (the only source for a generic pyramidal TIFF), then an OME-XML
    comment.

    Args:
        slide: An open slide.

    Returns:
        The microns per pixel, or None if no source gives one.
    """

    props = dict(slide.properties)

    try:
        value = float(props[openslide.PROPERTY_NAME_MPP_X])
        if value > 0:
            return value
    except (KeyError, TypeError, ValueError):
        pass

    return _mpp_from_resolution(props) or _mpp_from_ome_comment(props)


def inspect_slide(slide_path: str) -> Dict[str, Any]:
    """
    Describes a slide without judging it.

    Args:
        slide_path: Path to the slide.

    Returns:
        A dictionary with the keys 'path', 'readable', 'error', 'dimensions',
        'level_count', 'pyramidal', 'mpp' and 'vendor'.
    """

    info: Dict[str, Any] = {
        "path": slide_path,
        "readable": False,
        "error": None,
        "dimensions": None,
        "level_count": None,
        "pyramidal": False,
        "mpp": None,
        "vendor": None,
    }

    if not os.path.exists(slide_path):
        info["error"] = "file does not exist"
        return info

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as exc:  # openslide raises several unrelated types
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["mpp"] = _mpp_from_tifffile(slide_path)  # still useful to drive the conversion
        return info

    try:
        info["readable"] = True
        info["dimensions"] = slide.dimensions
        info["level_count"] = slide.level_count
        info["pyramidal"] = slide.level_count > 1
        info["mpp"] = read_mpp(slide)
        info["vendor"] = slide.properties.get(openslide.PROPERTY_NAME_VENDOR)
    finally:
        slide.close()

    return info


def _tile_iterator(
    read_region,
    dimensions: Tuple[int, int],
    downsample: int,
    tile: int,
) -> object:
    """
    Yields the tiles of one pyramid level, in the row-major order TiffWriter expects.

    Each output tile is produced from the matching region of the full-resolution image,
    so memory stays bounded by one region (tile * downsample pixels per side).

    Args:
        read_region: Callable (x, y, w, h) -> RGB PIL image at full resolution.
        dimensions: Full-resolution (width, height).
        downsample: Downsampling factor of the level being written.
        tile: Tile size in pixels, at the level being written.

    Yields:
        RGB tile arrays of shape (tile, tile, 3).
    """

    width = max(dimensions[0] // downsample, 1)
    height = max(dimensions[1] // downsample, 1)
    src_tile = tile * downsample

    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            region = read_region(x0 * downsample, y0 * downsample, src_tile, src_tile)
            if downsample > 1:
                region = region.resize((tile, tile), resample=Image.BILINEAR)
            yield np.asarray(region, dtype=np.uint8)


def to_pyramidal_tiff(
    slide_path: str,
    out_path: str,
    mpp: Optional[float] = None,
    tile: int = 512,
    max_levels: int = 6,
    min_level_size: int = 512,
) -> str:
    """
    Converts a slide to a tiled, pyramidal TIFF that OpenSlide (and therefore HoVer-Net)
    can read.

    Uses pyvips when it is installed, which is much faster, and falls back to a
    tifffile writer otherwise. The pyramid is written as consecutive top-level
    directories, which is what the OpenSlide generic-tiff driver reads as levels.

    Args:
        slide_path: Path to the slide to convert.
        out_path: Path of the pyramidal TIFF to write.
        mpp: Microns per pixel to record in the resolution tags. If None, it is read
             from the source slide.
        tile: Tile size in pixels.
        max_levels: Maximum number of pyramid levels, including full resolution.
        min_level_size: Stop adding levels once both sides are below this size.

    Returns:
        The path of the written file.

    Raises:
        SlideError: If the source cannot be read at all.
    """

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    try:
        import pyvips  # noqa: F401

        has_pyvips = True
    except ImportError:
        has_pyvips = False

    if has_pyvips:
        return _to_pyramidal_tiff_pyvips(slide_path, out_path, mpp=mpp, tile=tile)

    return _to_pyramidal_tiff_tifffile(
        slide_path,
        out_path,
        mpp=mpp,
        tile=tile,
        max_levels=max_levels,
        min_level_size=min_level_size,
    )


def _to_pyramidal_tiff_pyvips(slide_path: str, out_path: str, mpp: Optional[float], tile: int) -> str:
    """
    Converts a slide with pyvips.

    Args:
        slide_path: Path to the slide to convert.
        out_path: Path of the pyramidal TIFF to write.
        mpp: Microns per pixel to record, or None to keep the source resolution.
        tile: Tile size in pixels.

    Returns:
        The path of the written file.
    """

    import pyvips

    logger.info(f"Converting {slide_path} to a pyramidal TIFF with pyvips...")
    image = pyvips.Image.new_from_file(slide_path, access="sequential")

    kwargs = dict(tile=True, tile_width=tile, tile_height=tile, pyramid=True, bigtiff=True, compression="jpeg", Q=90)
    if mpp is not None:
        kwargs["xres"] = 1e3 / mpp  # pyvips resolution is in pixels per mm
        kwargs["yres"] = 1e3 / mpp
        kwargs["resunit"] = "cm"

    image.tiffsave(out_path, **kwargs)
    logger.info(f"-> Written to {out_path}")

    return out_path


def _to_pyramidal_tiff_tifffile(
    slide_path: str,
    out_path: str,
    mpp: Optional[float],
    tile: int,
    max_levels: int,
    min_level_size: int,
) -> str:
    """
    Converts a slide with tifffile, streaming one tile at a time.

    Args:
        slide_path: Path to the slide to convert.
        out_path: Path of the pyramidal TIFF to write.
        mpp: Microns per pixel to record, or None to read it from the source.
        tile: Tile size in pixels.
        max_levels: Maximum number of pyramid levels, including full resolution.
        min_level_size: Stop adding levels once both sides are below this size.

    Returns:
        The path of the written file.

    Raises:
        SlideError: If the source cannot be read at all.
    """

    import tifffile

    slide = None
    memmap = None

    try:
        slide = openslide.OpenSlide(slide_path)
        dimensions = slide.dimensions
        source_mpp = read_mpp(slide)

        def read_region(x, y, w, h):
            return slide.read_region((x, y), 0, (w, h)).convert("RGB")

    except Exception as open_error:
        logger.warning(f"OpenSlide cannot read {slide_path} ({type(open_error).__name__}), falling back to tifffile.")
        try:
            memmap = tifffile.imread(slide_path, out="memmap")
        except Exception as exc:
            raise SlideError(
                f"{slide_path} can be read neither by OpenSlide ({open_error}) nor by tifffile ({exc}). "
                "Convert it to a tiled pyramidal TIFF yourself, then run HEDeST again."
            ) from exc

        if memmap.ndim != 3 or memmap.shape[2] < 3:
            raise SlideError(f"{slide_path} is not an RGB image (shape {memmap.shape}), cannot convert it.")

        dimensions = (memmap.shape[1], memmap.shape[0])
        source_mpp = None

        def read_region(x, y, w, h):
            patch = np.zeros((h, w, 3), dtype=np.uint8)
            y1, x1 = min(y + h, memmap.shape[0]), min(x + w, memmap.shape[1])
            if y1 > y and x1 > x:
                patch[: y1 - y, : x1 - x] = memmap[y:y1, x:x1, :3]
            return Image.fromarray(patch)

    mpp = mpp if mpp is not None else source_mpp
    if mpp is None:
        raise SlideError(
            f"The resolution of {slide_path} is unknown and no mpp was given. "
            "Pass the microns per pixel so it can be recorded in the converted slide."
        )

    width, height = dimensions
    downsamples = [1]
    while (
        len(downsamples) < max_levels
        and max(width // (downsamples[-1] * 2), height // (downsamples[-1] * 2)) >= min_level_size
    ):
        downsamples.append(downsamples[-1] * 2)

    logger.info(
        f"Converting {slide_path} to a pyramidal TIFF with tifffile: "
        f"{width}x{height} @ {mpp:.4f} um/px, {len(downsamples)} levels, tile {tile}px."
    )

    tmp_path = out_path + ".tmp.tif"
    try:
        with tifffile.TiffWriter(tmp_path, bigtiff=True) as writer:
            for level, downsample in enumerate(downsamples):
                level_shape = (max(height // downsample, 1), max(width // downsample, 1), 3)
                writer.write(
                    _tile_iterator(read_region, dimensions, downsample, tile),
                    shape=level_shape,
                    dtype=np.uint8,
                    tile=(tile, tile),
                    photometric="rgb",
                    compression="zlib",
                    resolution=(1e4 / mpp, 1e4 / mpp),
                    resolutionunit="CENTIMETER",
                    subfiletype=1 if level > 0 else 0,
                )
                logger.info(f"-> Level {level} written ({level_shape[1]}x{level_shape[0]}).")
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if slide is not None:
            slide.close()

    logger.info(f"-> Written to {out_path}")

    return out_path


def ensure_pyramidal_slide(
    slide_path: str,
    mpp: Optional[float] = None,
    convert: bool = True,
    converted_path: Optional[str] = None,
) -> Tuple[str, float]:
    """
    Makes sure a slide is usable by HEDeST: readable by OpenSlide, pyramidal, and with
    a known resolution.

    If the slide is already fine it is returned untouched. Otherwise it is converted to
    a pyramidal TIFF, unless conversion is disabled or impossible, in which case the
    problem is raised.

    Args:
        slide_path: Path to the slide.
        mpp: Microns per pixel. If None, it is read from the slide and an error is
             raised when the slide does not carry it.
        convert: Whether an unusable slide may be converted to a pyramidal TIFF.
        converted_path: Where to write the converted slide. Defaults to the source path
                        with a '_pyramidal.tif' suffix.

    Returns:
        The path of the slide to use, and its microns per pixel.

    Raises:
        SlideError: If the slide is unusable and cannot be converted.
    """

    info = inspect_slide(slide_path)

    if info["error"] == "file does not exist":
        raise SlideError(f"{slide_path} does not exist.")

    effective_mpp = mpp if mpp is not None else info["mpp"]

    if info["readable"] and info["pyramidal"] and effective_mpp is not None:
        logger.info(
            f"Slide OK: {info['dimensions'][0]}x{info['dimensions'][1]}, "
            f"{info['level_count']} levels, {effective_mpp:.4f} um/px ({info['vendor']})."
        )
        return slide_path, float(effective_mpp)

    reasons = []
    if not info["readable"]:
        reasons.append(f"OpenSlide cannot read it ({info['error']})")
    elif not info["pyramidal"]:
        reasons.append("it has a single resolution level (not pyramidal)")
    if effective_mpp is None:
        reasons.append("its resolution (mpp) is unknown")

    problem = f"{slide_path} cannot be used as is: " + ", and ".join(reasons) + "."

    if effective_mpp is None:
        raise SlideError(problem + " Pass the microns per pixel of the slide so HEDeST knows its scale.")

    if not convert:
        raise SlideError(problem + " Convert it to a tiled pyramidal TIFF, or allow the conversion.")

    if converted_path is None:
        base, _ = os.path.splitext(slide_path)
        converted_path = f"{base}_pyramidal.tif"

    if os.path.exists(converted_path):
        logger.info(f"{problem} A converted slide already exists at {converted_path}, using it.")
    else:
        logger.warning(problem + " Converting it to a pyramidal TIFF.")
        to_pyramidal_tiff(slide_path, converted_path, mpp=float(effective_mpp))

    converted_info = inspect_slide(converted_path)
    if not converted_info["readable"] or not converted_info["pyramidal"]:
        raise SlideError(
            f"The conversion of {slide_path} did not produce a usable slide "
            f"({converted_path}: {converted_info['error'] or 'not pyramidal'}). "
            "Convert it to a tiled pyramidal TIFF yourself, then run HEDeST again."
        )

    return converted_path, float(effective_mpp)
