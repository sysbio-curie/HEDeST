#!/bin/bash

# MHAST on the fully simulated datasets, 10 seeds each, one array task per (dataset, seed).
#
# Only the 30-spot datasets are run: the global stage of the hierarchical permutation
# enumerates the Cartesian product of the spots it cannot settle locally (about 1e6
# candidates here), which is already ~20 min for 136 cells and would not finish on the
# 200-spot ones. One task per seed keeps the wall time at one repetition instead of ten.
#
# sim.py writes, per call, the metrics before (the random permutation respecting the spot
# composition) and after the permutation; the seeds are aggregated by
# benchmark/mhast_benchmark.py.
#
#   sbatch external/mhast/run_sim.sh

#SBATCH --job-name=mhast_sim
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/mhast_%A_%a.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/mhast_%A_%a.err
#SBATCH --array=0-19%10
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

echo "Found a place on $(hostname)!"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate mhast

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/mhast/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=2

DATA_PATH=/cluster/CBIO/data1/lgortana/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/sim
OUT_DIR=/cluster/CBIO/home/lgortana/HEDeST/benchmark/results/MHAST
TAGS=(4_hoptimus_clusters_30spots_balanced_5mean_5var 4_hoptimus_clusters_30spots_imbalanced_5mean_5var)

TAG=${TAGS[$((SLURM_ARRAY_TASK_ID / 10))]}
SEED=$((SLURM_ARRAY_TASK_ID % 10))
mkdir -p "${OUT_DIR}/${TAG}"

echo "=> ${TAG}, seed ${SEED}"

cd /cluster/CBIO/home/lgortana/HEDeST/external/mhast

python3 -u sim.py \
    --data_path "${DATA_PATH}" \
    --gt_filename "${TAG}_gt.csv" \
    --spot_dict_filename "${TAG}_spot_dict.json" \
    --embeddings_filename "${TAG}_emb_dict.pt" \
    --n_iter 1 \
    --seed "${SEED}" \
    --output_xlsx "${OUT_DIR}/${TAG}/seed_${SEED}.xlsx"
