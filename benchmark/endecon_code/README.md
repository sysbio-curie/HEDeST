# EnDecon deconvolution — the code behind `benchmark/results/PanoSpace/deconv`

PanoSpace's deconvolution path (`--deconv-only`): **RCTD + cell2location +
spatialDWLS, fused by EnDecon**, run against the STHELAR benchmark's spot data
and scored on `bench_data/{sample}/sim/level{L}/proportions.csv`.

Every method parameter is PanoSpace's own default. Nothing here changes the
method; it builds the reference the method needs, launches it, and scores it.

Results live in `/cluster/CBIO/home/lgortana/HEDeST/benchmark/results/PanoSpace/deconv`.

## Files

| | |
|---|---|
| `celltype_rules.py` | atlas cell type → canonical tag → the class names of a level's vocabulary |
| `build_refs.py` | one transformed reference per (sample, level) |
| `slurm_deconv.sh` | one EnDecon run on the cluster |
| `score.py` | metrics + the two figures |
| `integrate.py` | copy kept runs into `benchmark/results/PanoSpace/deconv`, extend its index files |

## End to end

```bash
conda activate panospace-env
CODE=/cluster/CBIO/home/lgortana/HEDeST/benchmark/endecon_code
TMP=/path/to/scratch          # 3CA prostate is extracted and cached here
REFS=$TMP/refs

# 1. the reference for one (sample, level)
python $CODE/build_refs.py --sample lung_s3 --level 3 --atlas lung \
       --out $REFS --tmp $TMP
#    ovary atlas, granulosa excluded (see below):
python $CODE/build_refs.py --sample ovary_s1 --level 2 --atlas ovary \
       --out $REFS --tmp $TMP --no-granulosa

# 2. the deconvolution (~50-90 min on one GPU)
sbatch $CODE/slurm_deconv.sh lung_s3 3 $REFS/lung_s3_level3.h5ad $TMP/out/lung_s3_level3

# 3. score + figures
python $CODE/score.py --sample lung_s3 --level 3 \
       --run-dir $TMP/out/lung_s3_level3 --out-dir $TMP/out/lung_s3_level3

# 4. install the ones you keep
cat > kept.json <<'JSON'
[{"sample":"lung_s3","level":3,"staged":"'$TMP'/out/lung_s3_level3",
  "ref":"'$REFS'/lung_s3_level3.h5ad","method":"direct"}]
JSON
python $CODE/integrate.py --plan kept.json
```

`build_refs.py` refuses to build a reference when any class of the level has no
atlas cell type behind it. That guard is what decides which (sample, level)
pairs are attempted at all — see the exclusion table in the results README.

For the 3CA prostate reference, extract `references/prostate.tar.gz` into
`$TMP` first; the MTX is converted once and cached as `prostate_X.npz`.

## What was run

27 (sample, level) pairs across 7 samples, in two rounds.

| sample | levels | reference |
|---|---|---|
| breast_s6 | 0, 1, 2 | DISCO breast |
| lung_s3 | 0, 1, 2, **3, 4** | DISCO lung |
| skin_s4 | 0, 1, 2 | DISCO skin |
| prostate_s0 | 0, 1, 2, **3** | 3CA prostate (Song 2022) |
| **ovary_s1** | **0, 1, 2, 3** | DISCO ovary |
| **cervix_s0_0** | **0, 1, 2, 3** | DISCO ovary |
| **cervix_s0_1** | **0, 1, 2, 3** | DISCO ovary |

Bold entries are the second round, which this code produced. `lymph_node_s0` is
excluded at every level: DISCO bone marrow contains no endothelial, pericyte or
smooth-muscle cell among its 54 types, so `Blood_vessel` cannot be built.

## The granulosa decision

The DISCO ovary atlas is 31% granulosa cells (38,522 of 124,128). They are
follicular epithelium, so whether they belong in a sample's `Epithelial` class is
a real choice that changes that signature completely. It was decided by running
both at level 0, per sample, rather than assumed:

| sample | granulosa in | granulosa out | kept |
|---|---|---|---|
| ovary_s1 | r 0.755 | **r 0.810** | `--no-granulosa` |
| cervix_s0_0 | **r 0.821** | r 0.771 | keep them |

`cervix_s0_1` follows `cervix_s0_0`, being the same tissue. Numbers in
`granulosa_test.json` beside the results.

## Inputs are raw counts — do not pre-process

Each backend does its own preprocessing and they disagree, which is why they all
take raw counts:

* **RCTD** uses counts directly and derives `nUMI` from `X.sum(-1)`;
* **cell2location** casts `.X` to `int`, then filters genes/cells itself;
* **spatialDWLS** does its own `normalize_total(1e4)` → `log1p` →
  `highly_variable_genes(2000)` → PCA → neighbours.

Normalising beforehand breaks the first two. The benchmark's
`pseudovisium.h5ad` is already correct: raw counts in `.X`, the normalised copy
parked in `layers['log_norm']`.

One trap: cell2location tries `adata.X = adata.raw.X.copy()` inside a bare
`try`, so a reference carrying a `.raw` is silently deconvolved from it — and
the DISCO atlases ship a malformed one (its X has a single row against ~10^5
obs). `build_refs.py` drops `.raw` for exactly this reason.

## Reproducing the whole directory

`plan.json` lists **31 deconvolutions producing the 27 output directories**, and
`reproduce.sh` walks it:

```bash
bash reproduce.sh /path/to/scratch           # build all references, submit all runs
bash reproduce.sh /path/to/scratch --refs-only
```

31 against 27 because of two things:

* **6 outputs are `fine`.** lung and prostate at levels 0-2 report the
  fine-grained fit. That is *one* deconvolution per sample, on the split
  vocabulary, aggregated three ways -- so 2 runs cover 6 outputs. The direct run
  of each of those levels is also kept, as `proportions_direct.csv`, hence still
  6 direct runs behind them.
* **2 granulosa variants** are run, scored and discarded; they exist so
  `granulosa_test.json` is reproducible.

### The fine-grained fit

`Blood_vessel` lumps endothelium with perivascular cells, and the atlas's blend
is not the tissue's -- lung 27.6% perivascular in the reference against 48.9% in
the truth, prostate 31.9% against 76.0%. One averaged signature is therefore
wrong for the tissue, the class is under-called, and the missing mass lands on
`Fibroblast`, smooth muscle being transcriptionally much closer to a fibroblast
than to an endothelial cell.

`build_refs.py --fine` builds the level-2 vocabulary with `Blood_vessel` split
into `Endothelial` + `Perivascular` and `Fibroblast_Myofibroblast` reduced to
`Fibroblast`; `aggregate_fine.py` sums the result back into any of levels 0-2.
Each signature is then homogeneous and the model decides how much smooth muscle
the tissue holds. No ground truth enters the reference -- only the fit's
granularity changes. (Reweighting the reference to match the truth's composition
*would* leak: that composition is the answer.)

```bash
python build_refs.py --sample lung_s3 --level 2 --atlas lung --fine --out $REFS --tmp $TMP
sbatch slurm_deconv.sh lung_s3 2 $REFS/lung_s3_level2.h5ad $TMP/out/lung_s3_fine lung_s3_fine
for L in 0 1 2; do
  python aggregate_fine.py --sample lung_s3 --level $L \
      --fine-proportions $TMP/out/lung_s3_fine/proportions.csv \
      --out $TMP/out/lung_s3_level${L}_fine/proportions.csv
  python score.py --sample lung_s3 --level $L --method fine \
      --run-dir $TMP/out/lung_s3_level${L}_fine --out-dir $TMP/out/lung_s3_level${L}_fine
done
```

The aggregation map is derived, never hard-coded: fine -> level 2 from
`FINE_SPLIT`, and level 2 -> 1 -> 0 read off the proportions files, since
annotation levels are nested. Verified against the six stored
`proportions_finegrained.csv`: **max difference 7.8e-16**, i.e. float summation
order.

Every second-round level (lung 3-4, prostate 3, ovary, cervix) already separates
`Endothelial` from `Smooth_muscle` in its own vocabulary, so there is nothing to
split there; those runs are `direct` by construction, not by a change of method.

## Provenance

Written after the first round had already been run and its scripts deleted, so
this is a reconstruction. It was checked against what is on disk and matches:

| | |
|---|---|
| cell-type assignments vs the recorded `label_map.json` | **405/405** |
| figure colours vs the recorded `palette.json` | **138/138** |
| fine-grained aggregation vs the stored `proportions_finegrained.csv` | max diff **7.8e-16** |

The assignment check only reaches 405/405 with `via_level2`, which is why the
round-1 entries in `plan.json` carry it: round 1 mapped every atlas type to the
level-2 vocabulary once and rolled up, so prostate `B_cell` (178 cells, no level-2
home) is absent at levels 0 and 1 too. Mapping per level instead would place it in
`Immune` there. Levels 3 and 4 are finer than 2 and can only be mapped directly,
which is what the round-2 entries do.
