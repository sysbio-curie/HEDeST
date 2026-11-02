"""
Times the H-Optimus-0 half of HEDeST on one slide.

HEDeST reaches a cell type in two steps -- H-Optimus-0 turns the slide into one embedding
per cell, then the classifier turns embeddings plus spot proportions into a cell type -- and
the runtime comparison against HistoCell and PanoSpace has to charge it for both. This
script measures the first step; ``time_hedest.py`` measures the second.

What is counted, and why:

* ``t_model``   loading H-Optimus-0 onto the device. Charged: PanoSpace is charged for
                loading DINOv2 and HistoCell for loading its ImageNet ResNet-18.
* ``t_infer``   reading tiles, the forward passes, and pooling patch tokens into cells.
                This is the whole of the image work.
* ``t_setup``   parsing the HoVer-Net JSON and laying out the tile grid. Reported but NOT
                charged, because the segmentation is shared by every method and PanoSpace's
                own ``segmentation`` stage is excluded on the same grounds.
* the 1.1 GB ``.pt`` write is not performed at all. It is how this repository caches
                embeddings between stages, not part of computing them; run as one program
                the embeddings would go straight to the classifier in memory.

The component times are recovered from the function's own log lines rather than by
instrumenting ``hedest.features.hoptimus``, which stays untouched.

Usage:
    python benchmark/comparison_code/time_hoptimus.py SAMPLE OUT_JSON [--batch-size 32]
"""
from __future__ import annotations

import json
import os
import platform
import re
import socket
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import torch
import typer
from loguru import logger

BENCH = Path("/cluster/CBIO/data1/lgortana/STHELAR/bench_data")

SETUP_RE = re.compile(
    r"(?P<cells>\d+) cells \| slide (?P<w>\d+)x(?P<h>\d+) @ (?P<mpp>[\d.]+) um/px \| "
    r"tile (?P<tile>\d+)px -> \d+ \([\d.]+ um/px\) \| (?P<tiles>\d+) tiles \| "
    r"setup (?P<setup>[\d.]+)s"
)
INFER_RE = re.compile(
    r"Inference and pooling done in (?P<elapsed>[\d.]+)s \((?P<rate>[\d.]+) tiles/s\), "
    r"cells filled (?P<filled>\d+)/(?P<total>\d+)"
)


def _gpu_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    return torch.cuda.get_device_name(0)


def main(
    sample: str = typer.Argument(..., help="bench_data sample name, e.g. prostate_s0."),
    out_json: str = typer.Argument(..., help="Where to write the timing record."),
    batch_size: int = typer.Option(32, help="Tiles per forward pass (the study default)."),
    num_workers: int = typer.Option(6, help="Tile reader workers (the study default)."),
    limit_tiles: Optional[int] = typer.Option(None, help="Calibration only: stop after N tiles."),
) -> None:
    """Times H-Optimus-0 inference over one slide and writes a JSON record."""

    slide = BENCH / sample / "he.tiff"
    seg = BENCH / sample / "hovernet.json"
    for path in (slide, seg):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    # Import before touching the logger: hedest.config replaces loguru's default handler on
    # import, and removing handlers first makes that call fail.
    from hedest.features.hoptimus import extract_hoptimus_embeddings

    # The log lines carry the component times; capture them as they are emitted, as an extra
    # sink so the job log still shows everything.
    captured: list[str] = []
    logger.add(captured.append, level="INFO", format="{message}")

    record: Dict[str, Any] = {
        "sample": sample,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "limit_tiles": limit_tiles,
        "host": socket.gethostname(),
        "gpu": _gpu_name(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "status": "started",
    }
    out = Path(out_json)

    # The record is written from a finally block. The function assembles its return value
    # -- one tensor per cell -- after the timed work is over, and on the largest slide that
    # allocation is the likeliest thing to fail on a node whose memory other jobs have
    # already taken. Writing early means a failure there still leaves the measurement.
    try:
        t0 = time.time()
        embeddings = extract_hoptimus_embeddings(
            slide_path=str(slide),
            json_path=str(seg),
            out_path=None,  # the .pt write is a cache, not computation
            batch_size=batch_size,
            num_workers=num_workers,
            limit_tiles=limit_tiles,
        )
        record["t_total_call_s"] = round(time.time() - t0, 2)
        record["n_cells"] = len(embeddings)
        record["status"] = "ok"
        del embeddings
    except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised
        record["status"] = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        text = "\n".join(captured)
        if (m := SETUP_RE.search(text)) is not None:
            record.update(
                t_setup_s=float(m["setup"]),
                n_tiles=int(m["tiles"]),
                slide_w=int(m["w"]),
                slide_h=int(m["h"]),
                mpp=float(m["mpp"]),
                tile_native_px=int(m["tile"]),
            )
        if (m := INFER_RE.search(text)) is not None:
            record.update(
                t_infer_s=float(m["elapsed"]),
                tiles_per_s=float(m["rate"]),
                cells_filled=int(m["filled"]),
            )

        # Whatever the call spent outside setup and inference is the model coming up: the
        # Hugging Face read, building the ViT and moving 1.1 B parameters to the device.
        if {"t_setup_s", "t_infer_s", "t_total_call_s"} <= record.keys():
            record["t_model_s"] = round(record["t_total_call_s"] - record["t_setup_s"] - record["t_infer_s"], 2)
            record["t_charged_s"] = round(record["t_model_s"] + record["t_infer_s"], 2)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        logger.info(f"-> {out} ({record['status']})")
        if "t_charged_s" in record:
            logger.info(
                f"{sample}: charged {record['t_charged_s']}s "
                f"(model {record['t_model_s']}s + inference {record['t_infer_s']}s), "
                f"setup {record.get('t_setup_s')}s not charged."
            )


if __name__ == "__main__":
    typer.run(main)
