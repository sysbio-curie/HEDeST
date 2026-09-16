#!/bin/bash
# Rebuild every reference and submit every deconvolution in plan.json.
#
#   bash reproduce.sh <scratch dir> [--refs-only]
#
# 31 EnDecon runs, ~50-90 min each on one GPU, producing the 27 result
# directories of benchmark/results/PanoSpace/deconv. Extract references/prostate.tar.gz into
# the scratch dir first (the MTX is converted once and cached).
set -euo pipefail
TMP=${1:?usage: reproduce.sh <scratch dir> [--refs-only]}
ONLY_REFS=${2:-}
CODE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$TMP/refs" "$TMP/out"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate panospace-env

python - "$CODE" "$TMP" <<'PY' > "$TMP/_jobs.txt"
import json, sys, subprocess, os
code, tmp = sys.argv[1], sys.argv[2]
plan = json.load(open(os.path.join(code, "plan.json")))
for d in plan["deconvolutions"]:
    cmd = [sys.executable, os.path.join(code, "build_refs.py"),
           "--sample", d["sample"], "--level", str(d["level"]),
           "--atlas", d["atlas"], "--out", os.path.join(tmp, "refs"), "--tmp", tmp,
           "--tag", "_" + d["id"]]
    if d.get("fine"):
        cmd.append("--fine")
    if d.get("via_level2"):
        cmd.append("--via-level2")
    if d.get("granulosa_as_epithelial") is False:
        cmd.append("--no-granulosa")
    subprocess.run(cmd, check=True)
    ref = os.path.join(tmp, "refs", f"{d['sample']}_level{d['level']}_{d['id']}.h5ad")
    print(f"{d['sample']}\t{d['level']}\t{ref}\t{d['id']}")
PY

echo "references built: $(wc -l < "$TMP/_jobs.txt")"
[ "$ONLY_REFS" = "--refs-only" ] && exit 0

while IFS=$'\t' read -r S L REF ID; do
    sbatch "$CODE/slurm_deconv.sh" "$S" "$L" "$REF" "$TMP/out/$ID" "$ID"
done < "$TMP/_jobs.txt"

cat <<'MSG'

Submitted. When they finish:
  * direct outputs        -> score.py, then integrate.py
  * fine outputs (6)      -> aggregate_fine.py on <sample>_fine/proportions.csv
                             for levels 0, 1 and 2, then score.py --method fine
  * granulosa variants    -> score.py only; they are compared, not installed
See plan.json ("outputs") for which is which.
MSG
