"""Builds the per-sample cache (see lungstudy/cache.py). Usage: python scripts/build_cache.py [SAMPLE ...] [--overwrite]"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lungstudy import cache  # noqa: E402
from lungstudy import paths  # noqa: E402

if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in arguments or paths.SAMPLES:
        cache.build(name, overwrite="--overwrite" in sys.argv)
