# Fully simulated datasets, v2 (H-Optimus-0)

Same construction as `simulations/full_simulations/` (Methods of the article), with the MoCo
cell embeddings replaced by **tile-level H-Optimus-0** embeddings, plus a series with an
increasing number of clusters.

Slide: `CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma` (352,093 HoVerNet nuclei).
Construction: K-means on the cell embeddings -> keep the 2,000 cells closest to each centroid
-> distribute them into pseudo-spots whose size follows a normal distribution -> per-spot
cluster proportions. HEDeST then sees only the embeddings and the proportions and has to
recover each cell's cluster.

## Reproduce

```bash
# 0. embeddings (done once, see "Embeddings" below)
# 1-3. clustering, datasets, checks (CPU node, ~1-2 h)
sbatch simulations/full_simulations_v2/run_pipeline.sh
```

| step | script | output |
|---|---|---|
| 1 | `01_cluster.py` | `sim_v2/construction/` (K-means per K, retained cells, UMAP, centroids) + `figures/construction/K{K}/` |
| 2 | `02_simulate.py` | `sim_v2/{tag}_*` + `figures/datasets/{tag}/` |
| 3 | `03_verify.py` | `figures/verification_report.txt`, `figures/datasets_summary.csv` |

Everything is configured in `config.py` (paths, seeds, weights, dataset list). Step 1 caches
its results; rerun with `--force` to recompute. Environment: conda `CellST`.

## Embeddings

`CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma/hoptimus_tile_embed_OC.pt`, `{cell_id: float32 (1536,)}`,
same cell ids and order as `image_dict_64px_20um.pt`.

Generated with the tile strategy of `review/feature_extraction/TILE_STRATEGY.md`, using the
`build_tiles` / `TileDataset` / `pool_cell` functions of `review/feature_extraction/infer_hoptimus_tile.py`:
224 px tiles at 0.5 um/px (112 um), half-tile-stride grid, each cell assigned to the nearest tile,
256 patch tokens pooled per cell as the area-weighted mean over its HoVerNet contour.
Log: `logs/hoptimus_tile_embed_OC_5184577.log`.

**mpp = 0.2754 um/px** (tile = 407 native px). The TIFF carries no usable resolution (96 dpi)
and the HoVerNet json only `mag: 40`. 0.2754 comes from the Visium spot pitch
(100 um = 363.08 full-res px in `tissue_positions_list.csv`); the existing 64 px / 20 um crops
of `image_dict_64px_20um.pt` are reproduced from the WSI at this scale (pixel correlation 0.9998,
centroids are `(x, y)`). `spot_diameter_fullres` in `scalefactors_json.json` would imply 0.251,
which does not reproduce the crops (0.944).

## Datasets

All 200-spot datasets: mean 15 / variance 15 cells per spot. 30-spot datasets: 5 / 5. Seed 42.
Tags follow `sim/` with `moco` replaced by `hoptimus`.

| # | tag | proportion columns | weights | perturbed props |
|---|---|---|---|---|
| 1 | `4_hoptimus_clusters_30spots_balanced_5mean_5var` | 4 | uniform | - |
| 2 | `4_hoptimus_clusters_30spots_imbalanced_5mean_5var` | 4 | 4.5 : 0.5 : 1 : 2 | - |
| 3 | `6_hoptimus_clusters_200spots_balanced_15mean_15var` | 6 | uniform | 0.05, 0.1, 0.25, 0.5 |
| 4 | `6_hoptimus_clusters_200spots_imbalanced_15mean_15var` | 6 | 3.5 : 0.5 : 22 : 2 : 0.5 : 4.5 | 0.05, 0.1, 0.25, 0.5 |
| 5 | `6_hoptimus_clusters_200spots_balanced_15mean_15var_dup` | 7 (cluster 0 copied as 6) | uniform | 0.05, 0.1, 0.25, 0.5 |
| 6 | `6_hoptimus_clusters_200spots_imbalanced_15mean_15var_dup` | 7 (cluster 0 copied as 6) | 3.5 : 0.5 : 22 : 2 : 0.5 : 4.5 : 2 | 0.05, 0.1, 0.25, 0.5 |
| 7 | `6_hoptimus_clusters_200spots_balanced_15mean_15var_dup_not_mixed` | 7, clusters 0 and 6 never share a spot | uniform | - |
| 8 | `6_hoptimus_clusters_200spots_imbalanced_15mean_15var_dup_not_mixed` | 7, clusters 0 and 6 never share a spot | 3.5 : 0.5 : 22 : 2 : 0.5 : 4.5 : 2 | - |
| 9-12 | `{3,10,15,20}_hoptimus_clusters_200spots_balanced_15mean_15var` | K | uniform | - |
| 13-16 | `{3,10,15,20}_hoptimus_clusters_200spots_imbalanced_15mean_15var` | K | drawn (below) | - |

K=6 of the number-of-clusters series is datasets 3 and 4.

**Imbalanced weights.** K=4 and K=6 are the legacy weights (recovered from the cell counts of
`sim/`, also in the Methods). K=3, 10, 15 and 20 were drawn once by
`draw_imbalanced_weights(K, seed=42)` to mimic the legacy K=6 profile, and are hardcoded in
`config.py`: 1 dominant cluster about 6-10x a mid-sized one (capped at 60 %), `round(K/4)` small
clusters about 0.15-0.35x a mid-sized one (at least 30 cells), the rest mid-sized.

| K | dominant | mid-sized | small |
|---|---|---|---|
| 3 | cluster 0: 60.0 % | 1 x 36.1 % | cluster 1: 3.9 % |
| 10 | cluster 3: 48.1 % | 6 x 5.5-9.3 % | clusters 2, 6, 9: 1.2-1.6 % |
| 15 | cluster 0: 40.0 % | 10 x 3.2-8.3 % | clusters 8, 9, 11, 13: 1.0-1.5 % |
| 20 | cluster 2: 35.9 % | 14 x 2.3-6.2 % | clusters 6, 8, 10, 15, 18: 1.0-1.3 % |

### Files per dataset (`sim_v2/`, same layout as `sim/`)

| file | content |
|---|---|
| `{tag}_spot_dict.json` | spot id -> cell ids |
| `{tag}_prop.csv` | cluster proportions per spot (= perturbation 0) |
| `{tag}_perturb_{s}_prop.csv` | `(1-s) * prop + s * U(0,1)`, renormalised per spot |
| `{tag}_gt.csv` | one-hot cluster of every cell present in a spot (`nucleus_id` index) |
| `{tag}_emb_dict.pt` | H-Optimus-0 embeddings of those cells |
| `{tag}_image_dict_64px_20um.pt` | 64 px / 20 um images of those cells (visualisation) |

Duplicated cells are named `<id>-1` and share the embedding and image of `<id>`.

## Construction details and differences with `sim/`

The bag algorithm is the legacy one (`utils/data_simulation.py`, copied): cells are drawn per
cluster according to the weights (`int(weight x mean x n_spots)`), spot sizes are
`int(Normal(mean, var))`, and each spot picks uniformly among the clusters that still have
cells. Consequences kept from the legacy datasets: spot sizes average slightly below the target
mean, about 2,900 of the 3,000 drawn cells end up in spots, and imbalanced datasets end with
spots made only of the dominant cluster. Running the copied and the legacy `create_bags` /
`get_bag_proportions` on the same inputs gives identical spots and proportions for all the
legacy configurations.

Differences:
* **One K-means per K shared by every dataset with that K**, so cluster numbers mean the same
  thing across balanced / imbalanced / dup datasets. In `sim/` the balanced and imbalanced
  6-cluster datasets came from two runs of the same clustering with permuted cluster numbers
  (balanced 0-5 = imbalanced 1, 5, 4, 3, 0, 2): the imbalanced dup duplicated what is cluster 4
  in the balanced numbering.
* A cluster asked for more cells than it holds raises an error (the legacy code silently sampled
  with replacement, which never happened in `sim/`); an empty spot raises an error too.
* Perturbation noise is seeded.

## Figures (png + svg)

| folder | figures |
|---|---|
| `figures/construction/K{K}/` | `umap_K{K}` (all cells / retained cells), `cluster_sizes_K{K}`, `slide_map_K{K}` (retained cells on the tissue), `gallery_K{K}_closest` and `gallery_K{K}_random` (64 cells per cluster) |
| `figures/datasets/{tag}/` | `spot_sizes`, `cluster_distribution` (cells drawn from the weights vs cells in spots), `bags`, `cooccurrence` (spots containing both clusters), `perturbation` (perturbed datasets only) |
| `figures/` | `datasets_summary.csv`, `verification_report.txt` |

## Results (run of 2026-09-14)

* Embeddings: 13,781 tiles, 24 min on a P100, all 352,093 cells embedded.
* Clustering (`sim_v2/construction/clusters_summary.csv`): every cluster at every K holds more than
  2,000 cells (smallest: 4,367 at K=20), so every retained pool is a full 2,000 cells. The stroma
  cluster (cluster 5 at K=6, cluster 0 at K=20) is concentrated in the stromal band on the right of
  the slide; the other clusters are spread over the tumour.
* Verification: 302 checks, 0 failures. Pipeline: 32 min on node008 (K-means 30 min of it).
* The 8 datasets mirroring `sim/` have exactly the same proportion matrices and spot sizes as
  `sim/` (same algorithm, seed and 2,000-cell pools); only the cells differ (at most 31 cells in
  common). Comparing `sim/` and `sim_v2/` therefore compares MoCo and H-Optimus-0 on the same spot
  design. The perturbed proportions differ (the legacy noise was unseeded).
