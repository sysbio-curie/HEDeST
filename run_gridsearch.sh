#!/bin/bash

#SBATCH --job-name=HEDeST
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/gridsearch_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/gridsearch_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --exclude=node005,node006,node009
#SBATCH --cpus-per-task=8

echo "Found a place!"

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

FEATURES=$1
SIM_CSV=$2
JSON_PATH=$3
ADATA_PATH=$4
ADATA_NAME=$5
SPOT_DICT=$6
shift 6

EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

python3 -u hedest/gridsearch.py \
    "$FEATURES" \
    "$SIM_CSV" \
    "$JSON_PATH" \
    "$ADATA_PATH" \
    "$ADATA_NAME" \
    "$SPOT_DICT" \
    "${EXTRA_ARGS[@]}"
