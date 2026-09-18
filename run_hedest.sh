#!/bin/bash

#SBATCH --job-name=HEDeST
#SBATCH --output=/cluster/CBIO/home/lgortana/HEDeST/log/hedest_%j.log
#SBATCH --error=/cluster/CBIO/home/lgortana/HEDeST/log/hedest_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p cbio-gpu
#SBATCH --cpus-per-task=8

echo 'Found a place!'

source /cluster/CBIO/home/lgortana/anaconda3/etc/profile.d/conda.sh
conda activate plugin-env

export LD_LIBRARY_PATH=/cluster/CBIO/home/lgortana/anaconda3/envs/plugin-env/lib:$LD_LIBRARY_PATH

for seed in {0..9}; do
  python3 -u hedest/main.py \
    /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/moco_embed_BRCA_64px_20um.pt \
    case_study/DestVI_BRCA_prop.csv \
    --json-path /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/seg_json/pannuke_fast_mask_lvl3.json \
    --path-st-adata /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/adata.h5ad \
    --adata-name Visium_FFPE_Human_Breast_Cancer \
    --spot-dict-file /cluster/CBIO/data1/lgortana/Visium_FFPE_Human_Breast_Cancer/spot_dict.json \
    --model-name default \
    --norm \
    --dropout 0.0 \
    --batch-size 64 \
    --lr 1e-4 \
    --divergence l2 \
    --alpha 0 \
    --beta 0.0 \
    --adjustment interpolated \
    --no-gated \
    --epochs 100 \
    --train-size 0.8 \
    --val-size 0.1 \
    --out-dir models/BRCA-test/DestVI_4000_hvg_squash_06_02_large/model_default_hidden_dim_512-256_norm_True_dropout_0.0_alpha_0.0_lr_0.0001_divergence_l2_beta_0.0/seed_${seed} \
    --save-geojson \
    --rs $seed
done
