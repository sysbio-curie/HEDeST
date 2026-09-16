"""Copy kept runs into benchmark/results/PanoSpace/deconv and extend its index files.

Takes a JSON plan of the runs to keep:

    [{"sample": "lung_s3", "level": 3,
      "staged": "<dir of the run>", "ref": "<transformed reference .h5ad>",
      "method": "direct"}, ...]

For each entry it copies the run's files, pulls the three per-backend
checkpoints out of the PanoSpace cache, installs the reference under
refs/{sample}_level{L}.h5ad, and appends to summary.csv, metrics.json,
palette.json and label_map.json. Existing rows are never rewritten.

    python integrate.py --plan kept.json [--deconv-dir ...] [--cache ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score

DECONV = "/cluster/CBIO/home/lgortana/HEDeST/benchmark/results/PanoSpace/deconv"
CACHE = os.path.expanduser("~/.panospace_cache/deconv")
FILES = ("proportions.csv", "run_info.json", "run.log", "deconv_vs_truth.png", "spot_pies_true_vs_endecon.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--deconv-dir", default=DECONV)
    p.add_argument("--cache", default=CACHE)
    a = p.parse_args()
    D = a.deconv_dir
    keep = json.load(open(a.plan))

    scores = {}
    for e in keep:
        s, L, staged = e["sample"], e["level"], e["staged"]
        method = e.get("method", "direct")
        tag = os.path.basename(staged.rstrip("/"))
        # scoring also writes the two figures into the staged dir, which is
        # where they are copied from below
        scores[tag] = score(s, L, staged, staged, method)
        dst = f"{D}/{s}/level{L}"
        os.makedirs(dst, exist_ok=True)
        for f in FILES:
            shutil.copy2(f"{staged}/{f}", f"{dst}/{f}")
        for f in glob.glob(f"{a.cache}/{tag}/deconv_*.csv.gz"):
            shutil.copy2(f, f"{dst}/{os.path.basename(f)}")
        shutil.copy2(e["ref"], f"{D}/refs/{s}_level{L}.h5ad")
        print(f"  {s}/level{L}: {len(os.listdir(dst))} files")

    rows = pd.read_csv(f"{D}/summary.csv").to_dict("records")
    have = {(r["sample"], r["level"]) for r in rows}
    metrics = json.load(open(f"{D}/metrics.json"))
    palette = json.load(open(f"{D}/palette.json"))
    lm = json.load(open(f"{D}/label_map.json"))
    for e in keep:
        s, L = e["sample"], e["level"]
        tag = os.path.basename(e["staged"].rstrip("/"))
        r = scores[tag]
        o = r["overall"]
        if (s, L) not in have:
            rows.append(
                dict(
                    sample=s,
                    level=L,
                    n_spots=r["n_spots"],
                    n_types=len(r["cell_types"]),
                    method=r["method"],
                    mean_pearson=round(o["mean_pearson"], 4),
                    mae=round(o["mae"], 4),
                    rmse=round(o["rmse"], 4),
                    mean_jsd=round(o["mean_jsd"], 4),
                )
            )
        metrics[f"{s}/level{L}"] = {
            k: r[k] for k in ("sample", "level", "n_spots", "cell_types", "method", "overall", "per_cell_type")
        }
        palette.setdefault(s, {})[f"level{L}"] = r["palette"]
        meta_path = e["ref"].replace(".h5ad", ".map.json")
        if os.path.exists(meta_path):
            m = json.load(open(meta_path))
            lm.setdefault(s, {}).setdefault("levels", {})[f"level{L}"] = {
                k: m[k]
                for k in (
                    "reference",
                    "granulosa_as_epithelial",
                    "n_cells_atlas",
                    "n_cells_used",
                    "per_class",
                    "map",
                    "dropped",
                )
            }

    df = pd.DataFrame(rows).sort_values(["sample", "level"])
    df.to_csv(f"{D}/summary.csv", index=False)
    json.dump(metrics, open(f"{D}/metrics.json", "w"), indent=2)
    json.dump(palette, open(f"{D}/palette.json", "w"), indent=2)
    json.dump(lm, open(f"{D}/label_map.json", "w"), indent=2)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
