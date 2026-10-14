#!/bin/bash

#SBATCH --job-name=xenium_annot
#SBATCH --output=/cluster/CBIO/home/lgortana/data-simulation-xenium/log/annot_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/data-simulation-xenium/log/annot_%j.err
#SBATCH -p cbio-cpu
#SBATCH --cpus-per-task=4

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate xenium-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/xenium-env/lib:$LD_LIBRARY_PATH

python3 -u cosine_annotation.py \
        /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sc/TME_atlas_lung_cancer_raw.h5ad \
        /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/data.zarr \
        /cluster/CBIO/data1/lgortana/Xenium_V1_humanLung_Cancer_FFPE/sim/cosine_annotation_TME_atlas.h5ad \
        --min_counts 10 \
        --cell_type_key cell_type \
        --batch_size 5000
