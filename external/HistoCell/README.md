# HistoCell<img src="image/README/logo.png" width = 150 align = right>

**HistoCell** is a **weakly-supervised deep learning framework** to elucidate the **hierarchical spatial cellular information** including **tissue compartments, single cell types and cell states** with **histopathology images only**. This tutorial implements HistoCell to predict super-resolution spatial cellular information and illustrates the representative applications. The link to the HistoCell method will be presented soon. \
Our website: http://histocell.qhdyr.net/index/index/index.html

<img src="image/README/Intro.jpg" alt="Image" width="800" style="display: block; margin: 0 auto;">

## Environments
```sh
pip install -r requirements.txt
```

## Data Format and Preprocessing

### Data Preparation

* For model pre-train, HistoCell takes the **scRNA-seq** and **spatial transcriptomics** data with paired **high-resolution histopathology images** as input.
  * For histopathology images, we cut the paired images according to the pixel coordinates from ST data. The preprocessing code can be found in **./tutorial/tutorial.ipynb**
  * For transcriptomics data, we apply deconvolution methods to get the cell composition as the supervision. Applicable methods are listed as [CARD](https://www.nature.com/articles/s41587-022-01273-7), [RCTD](https://www.nature.com/articles/s41587-021-00830-w), [Tangram](https://www.nature.com/articles/s41592-021-01264-7), [Cell2location](https://www.nature.com/articles/s41587-021-01139-4) etc.
* For model inference, HistoCell only requires the histopathology images including tiles and WSIs.
  * WSIs should be formatted as .svs or .tif. In order to convert the WSIs into the tiles can be processed in a high-throughput manner, we apply the toolbox from [CLAM](https://github.com/mahmoodlab/CLAM) for image segmentation, stitching and patching. As for TCGA diagnostic images, patch of 256x256 pixels is recommended.
  * Tiles can be storaged in any format of image.

### Data Preprocessing

Using GPUs is highly recommended for whole slide images (WSIs) processing.

Before we begin the model inference process, cell segmentation is required. Here, we apply [HoVerNet](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841519X00079/1-s2.0-S1361841519301045/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDd2OMed8Quier79h6hhXShEPV1a3lwQv%2BMd%2B99MpGolQIgHbvYbhoiHEe1uT1QjfLGEOEMBSTSCAhJmoThBfbNll4quwUIpv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDMf%2B5OgldA9yxfGLPyqPBY%2FB1K2p3P7IvTS1hGh8akwWej0tXrgiaex2nNOJT6jwHbFCml3oXhsoNvr5bQ%2BmhFL3hNeKRbOXowl1RfkkrM9Mo8W7VB6L6a3faDuy3R9FmYg9OniS%2F2l1pasqKf%2Bk3es0ZtkBZYJhIpRcxWIogcRB1WPWE9WGuBRfN2qp7xf7NNkq4ZmbNaU3ysqx%2FMFFZWGP1DoLhVUeP18olpZstHpJ5rrKvMEJ4bUhOnN4WkA4wflhpJAKy6dv10PJIbCGYWReuhcTFO%2FNoSqCRDUUnQZD5zRfaCfsjNO943WEECuHreEcGSfsGwH16ncrE6deBpvS9x5f7qFSzLkM01th0ZwonFL0zXSGN6qaPnZ0wBzO2Lbe0OtzPBeHG3BrPl3VxL9qYSKNDITFNW%2BVRAO3CckWm%2Bt%2FEQqBGRbX%2FArLKvT7NS12jx%2FhEhj%2B%2Ba3yYQQjFUJMDPfLpbXsLpl8IOsNKBbQAqsT09iN0an0zA8q7oh%2F9HgfL8KCZdRuPWp9HkLLjSRbdH01i7ctSMTbehrkjiVMnXz8f3B9%2BVHfZR%2B3xQYH2YOH67UE87JCofjKJkWhroXKSkS1c53ye%2FOCEyF9gp2ChrWnKG8o95jTccF%2BInoECXr0Ymc5QiotpLF2es6pUQcGi2mq5rcjY1P6vJ9x4i4DDR2e%2FM718BZaM8zQCUmYm4XOBu%2B22Wtf8GgAymn8pz7uTgKnh9jlZhmTZ4YAnzVaqW%2FjTovCSRCUucEzklibwjVJFHw8urGagWh77nG0Qv9wGnO4PQYYOTJqs%2F2WZo9raeKVwxMPkcoXOVRM0Pkphg4bDnVZG63R1xL89urfYK1PPKDGtH08o9UvREJM6ugGTjDHpRZPeS29NYXNZ7kwxtHOrQY6sQFFrXdmd2FpmYFJtlfg2DJxJa7SwFYBVdj3Db6HD%2FLftOy%2B%2FyGCiRWyB%2FC%2FDxPPU0WDWvUzAq72HoaEV87cVsLz6Q7446UGQ1HAGMKRnAhCALhSZIp7WfyC1gTuAthi5QVJvr78GXQAinqnBaAlrMaHnLTtbiSNPNykRQrrEvhBDIq21Ffy%2Fid%2F9pPW0IURXrvj1end6m7dZT7ZtEIYcA%2BWVa5%2FYz6%2FOu5Tvta6zaDkE0k%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240126T134143Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5OQRHCPI%2F20240126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=608d9d59c577a23f89bb7aaa69c7f8c2163e3911b0cfcdc16987fb1603f50428&hash=48169297d0c9fa366d3cb8bd120add89965ee3917df478d735f708fb300c8168&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841519301045&tid=spdf-4ba996cc-271e-4fff-93e1-559385625235&sid=31ab0cf595ec25414989848882e6352d1712gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f135a56045e56020451&rr=84b92cd0cc162287&cc=us)[[code](https://github.com/vqdang/hover_net)], a well-developed nuclei instance segmentation method, to process cell segmentation from image tiles. It is worth mentioned that other segmentation methods (e.g. [MaskRCNN](https://arxiv.org/abs/1703.06870), [Cerbebrus](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841522X00078/1-s2.0-S1361841522003139/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDWnVS2nFNswtWtc37yxAthvAru8F%2Bi9sXObOpOSp4fcQIgPLxoDxjXB2%2Bn5LDajvayiIU5Ev5%2FZSwXH65%2FSCvHpTQqvAUIpf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDGqoAHUKPFM7LE85KCqQBRI1MjQTxc06bBJ2fe8tBSgL1PqcV3eXn3SDFwCsfPX48S61JVtwG9g91MxurBEUNhdwrjWdNBwI3pDUNKp7MQIH0AJVRXhwJ5mKktV96eZd5dizuGWGAicjTD%2Fy5zusW3nLiEhx0ka61bVUzKU7ZGdA%2Bs%2BT1egxuN3LK9fHKGVF63NVJt2sbDgNigMWEEiPdlXbWtvqvB0EylrEcVT7Rqo2a7ccsmPMCyDQlKyMzWJpNBOOu1KI2BK2KE0y0jLELSTeOvF9iHfwmBOf3Q1Azq7uBElKYA2jQPPFm6O%2Fi7b6UsJQtg5RLxB2HbOcRK1htkMmnOvGARy1y%2BbwQ26nYJJW5C85m%2F%2Bc3%2FXjBg%2BrsLeodZYEWQEpVgY%2BTUf%2Bdi4%2BCYmok0ra8kTrShb2CvgVK4pVrk8L0AO33WVGNokeOp%2F8xV0bz7O1DAsqN2ykAvvPk0wuprohZxBm2BGhA%2BLY7lox3f6v6Rv544wpvF8RHreBS%2B2jkVPae8%2FZER6WaRl9WuFuBzJlgY0xdW1V2iU8quvRxltyx%2BSS%2BVbdgMMBmX9H41LLVQgTYNnciL7hTcHRzS70pXQa2fb%2BfhFjQ0dXv3eH%2BCz3JeKgpU6wInT5Ax4GKKU9o3QqJMh9OsFmMvKjVHL8bzhpllg77aZEVPHbXj%2F2oUB6xN3BhSjl30wKiDKT0y1RGt2QuKWOqd%2FW5nEo0Nw8YA4kS%2FIkq6v7XQuyJnuPx6orceBGrHbBb0mLvHV5XubTTAmmvGmXojPqBT6cW%2FmCdjwHnGoAb5HG%2BAPuc%2FwCJ8pDR9ZxDgzry%2F4TNCPICSIP7yj%2B9tyfleO3i1J1mhnzCa%2BINadYcFBBJcVN4PW6abUmLq2zakWegZOr%2FjW2MNrHzq0GOrEBslcHzW1k%2BVNCvsQQ9R4j1edYgD16Ya5im3eeb4DyfkHwxF1eODT4NU1SsSI3uXG%2BnfmMEP0qPsq6khy%2FK9RXojWq0Qn0lsTw9ilEvPN97m192kvnNcD0dkbN32Jpjl%2Bfg08nRzORU9wzjJfwvSjwx2RIMdnBtJGENnzNNt%2FuEg%2BWdHon%2F6%2FG3%2FvmkWweeuQw9hnbO6X2HV0BGyMPXcMq9ouE8eIDqG38lbE3hm9kUFm1&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240126T134736Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY72FFPE6G%2F20240126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f81b493864396df90b2dc5a4bc19a720f03415ca06d0e42870d9bbc34adb93ef&hash=e4bf9d21b009b09db822df8fbcdc06443d696a664e097e20f414559b75f39f7d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841522003139&tid=spdf-1b0d3161-dd23-45aa-bfe3-aa725b637c50&sid=31ab0cf595ec25414989848882e6352d1712gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f135a56045e57545753&rr=84b935728ba1239d&cc=us)etc.) are also applicable here.

With cell segmentation by HoVerNet, you will get json file as the segmentation results illustrated as below:

```json
{"mag": null, "nuc": {"YOUR_CELL_ID": {"bbox": [[1, 2], [3, 4]], "centroid": [2, 3], "contour": [[0, 1]], "type_prob": 0.9, "type": 0}}}
```

### Input data format
The tiled images, segmentation results(json file) and cell type/state proportion given by deconvolution are required during training process.

## Data Accession
Demo data and tutorial data could be assessed on the Google drive with [Demo](https://drive.google.com/file/d/1DzwsFvD4wWKNN1wJCe3YsQZlk9xIp8DV/view?usp=sharing) and [Tutorial](https://drive.google.com/file/d/1iwXzxafdUF7SF6IAEyWluN728WbxAMEi/view?usp=sharing), respectively. To run the demo, you need to download the data to the directory **./demo/data**.

## Model pre-training with ST data
Run **train.py** for model pre-training on cell type prediction. You need to change the parameters in **configs.py** including the model information, dataset directory and training details. Besides, choose the proper tissue compartment file in **./tcs**. To develop your own model, run the command below.
```python
python train.py \
--model YOUR_MODEL_DESC \
--tissue TISSUE_TYPE \
--deconv DECONVOLUTION_METHOD \
--prefix SAMPLE_PREFIX \
--k_class NUM_OF_CLASS \
--tissue_compartment PATH_TO_TCS_FILE
```
A simple demo on subject H1 from HER2ST dataset:
```python
python train.py \
--model Breast_Benchmark_H1 \
--tissue BRCA \
--deconv RCTD \
--prefix H1 \
--k_class 6 \
--tissue_compartment ./tcs/tissue_compartment_BRCA.json
```

For the benchmark analysis on the PanNuke dataset, run the leave-one-out cross-validation command:
```
python train_oneout_pannuke.py \
--model PanNuke_cross_validation \
--tissue Breast \
--reso 1 \
--deconv Mix \
--prefix fold1 fold2 fold3 \
--k_class 5 \
--tissue_compartment ./tcs/tissue_compartment_Mix.json
```

As for cell state, you can run **train_state.py** to develop the cell state prediction model. For example, develop the cell state prediction on the breast cancer ST samples:
```python
python train_state.py \
--model brca_state \
--tissue BRCA \
--deconv RCTD \
--prefix 10x_BRCA A_1938345_11 B_1938529_9 C_2000752_23 D_2000910_33 \
--tissue_compartment ./tcs/tissue_compartment_state.json
```

## Quick inference with histopathology images only
With the pretrained model, you can infer the cellular spatial profile with **infer.py**.
```python
python infer.py \
--model Breast_Benchmark_H1 \
--epoch 30 \
--tissue BRCA \
--deconv RCTD \
--prefix H1_hires \
--k_class 6 \
--tissue_compartment ./tcs/tissue_compartment_BRCA.json \
--omit_gt
```

## Representative Results
Here we only illustrate the demo and representative results corrsponding to the paper in the tutorial.ipynb. The predicted hierarchical spatial cellular information is storaged as a dict in a pickle file for each slide. For more results, you can directly jump to our [HistoCell website](http://histocell.qhdyr.net/index/index/index.html).

### Benchmark results

* **Tissue Compartment**
  <div align = center>
  <img src="image/README/tissue_compartment.jpg" alt="Image" width="400" align="center">
  </div>
* **Single-cell Type**
<div align = center>
  <img src="image/README/cell_type2.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>
<div align = center>
  <img src="image/README/cell_type1.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>

  Red, blue and green scatters represent cancer epithelial cells, stromal cells and macrophage cells.
* **Cell States**
<div align = center>
  <img src="image/README/cell_state.jpg" alt="Image" width="800" style="display: block; margin: auto auto;">
</div>

### Representative Application: Tissue architecture annotations

With a histopathology image, HistoCell could first infer pixel-level cell types and then cluster cells as tissue regions, which exhibit high accuracy and allow users to further identify the small foci within tissue regions at pixel-level resolution.
<div align = center>
<img src="image/README/segmentation.jpg" alt="Image" width="500"  style="display: block; margin: auto auto;">
</div>

### Representative Application: Cell Type Deconvolution

Since HistoCell Integrates spot-level cellular compositions deconvoluted from expression data and those based on histologic morphologic features, it could produce a more precise and robust deconvolution result.
<div align = center>
<img src="image/README/deconvolution1.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>
<div align = center>
<img src="image/README/deconvolution2.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>

### Representative Application: Spatial organization indicators identification
<div align = center>
<img src="image/README/SOI.jpg" alt="Image" width="400" style="display: block; margin: auto auto;">
</div>
The histopathology image is coverted to **spatial cellular map** with HistoCell and the cells are accumulated as clusters. Through the correlation analysis between clinical outcomes and cellular spatial clustering features, we identify spatial biomarkers for prognosis. Demo results can be found in Tutorial.ipynb. The representative spatial features for prognosis stratification is visualized as below.
<div align = center>
<img src="image/README/biomarker.jpg" alt="Image" width="1000" style="display: block; margin: auto auto;">
</div>





---

# Benchmark adaptation (STHELAR / HEDeST)

This section documents a fork of HistoCell wired into the STHELAR benchmark. It
adds an I/O layer and a driver script; **the method itself is untouched** — same
architecture, same losses, same hyper-parameters. The list of what is and is not
the same is spelled out under [What is unchanged](#what-is-unchanged) and
[What was changed, and why](#what-was-changed-and-why).

## What this fork is for

HistoCell was published as a family of pre-trained models, one per cancer type,
but **the weights were never released**. Upstream's `infer.py` therefore cannot
run: it raises `FileNotFoundError("No trained model exits")` unless you have
already trained something.

So each run here trains a model from scratch **on the slide it is about to
annotate**, using that slide's own spot proportions as the weak supervision
label, and then predicts a cell type for every nucleus of the same slide:

```
     spot proportions  ─┐
                        ├──▶  train (50 epochs, one tile per spot)
     H&E + segmentation ┘                    │
                                             ▼
     H&E + segmentation ──────────▶  predict every nucleus of the slide
```

Training and inference happen in one command; there is no separate pre-training
stage and no single-cell reference is needed, because the proportions are given.

## Quick start

```bash
# once
bash external/HistoCell/setup_env.sh          # creates the histocell-env conda env

# one slide, one annotation level (a Slurm wrapper lives at HEDeST/run_histocell.sh)
sbatch run_histocell.sh \
    --he          .../bench_data/lung_s3/he.tiff \
    --st          .../bench_data/lung_s3/pseudovisium.h5ad \
    --seg-dict    .../bench_data/lung_s3/hovernet.json \
    --proportions .../bench_data/lung_s3/sim/level2/proportions.csv \
    --output      .../histocell/lung_s3/level2
```

`python external/HistoCell/run_histocell.py --help` lists every option. A
run takes roughly 1–3 h on one GPU, almost all of it training; re-running only
the inference is a matter of `--resume <output>/model/epoch_49.ckpt`.

## Inputs

The same four files PanoSpace takes, in the same formats:

| flag | file | used for |
|---|---|---|
| `--he` | `he.tiff` | the pixels; tiles are cut from it |
| `--st` | `pseudovisium.h5ad` | spot centres (`obsm['spatial']`) and the spot diameter |
| `--seg-dict` | `hovernet.json` | nucleus centroids, contours and PanNuke classes |
| `--proportions` | `sim/level{L}/proportions.csv` | the weak supervision label |

No single-cell reference is required — HistoCell consumes proportions that have
already been deconvolved, it does not deconvolve anything itself.

Two optional inputs:

* `--tissue-compartment` — an upstream-style `./tcs/*.json`. Generated
  automatically when omitted (see [Tissue compartments](#tissue-compartments)).
* `--spot-dict` — `spot_dict.json`, i.e. `spot_id -> [cell_id, ...]`. Restricts
  each training tile to the spot's own member cells and defines spot membership
  when clumping. **Off by default**, because upstream keeps every nucleus that
  falls inside the square tile, including the corners that lie outside the round
  spot; real Visium data has exactly the same mismatch.

## Outputs

Written to `--output`:

| file | shape |
|---|---|
| `histocell_predictions.csv` | `cell_id` x cell type, softmax probabilities. Same shape as HEDeST's `pred_best_adjusted`, so `df.idxmax(axis=1)` is the call. Row ids are the ids of `hovernet.json`. |
| `histocell_proportions.csv` | `spot_id` x cell type. Same shape as `sim/level{L}/proportions.csv`. |
| `tissue_compartment.json` | the `tcs` file actually used |
| `model/epoch_*.ckpt` | the trained weights |
| `01_annotation_slide.png` | every predicted nucleus on the slide |
| `02_spot_proportion_fit.png` | predicted vs deconvolved spot proportions, per cell type, with the PCC the paper uses as its metric |
| `03_spot_pies_true_vs_pred.png` | the spots at their slide positions drawn as pie charts — ground truth on the left, HistoCell on the right |
| `run_info.json`, `run.log` | provenance, counts, timings |

### Colours

The figures use the benchmark's hierarchically-consistent colour code: each
level-0 category is a colour family (Epithelial blue, Immune green, Structural
orange, Melanocyte purple) and finer types are shades within their family, so a
cell type keeps its colour across levels.

The scheme is **reproduced inside this package** (`bench.level_palette`) rather
than imported, so nothing outside this repository is needed to run it. The
family-to-leaf grouping it depends on is derived from the slide's own
`sim/level*/proportions.csv` files: annotation levels are nested, so a coarse
category's proportion is the sum of its children's spot by spot, which makes the
tree recoverable from the proportions alone — and keeps it from drifting out of
date. Checked against the benchmark's own palette on 39 (dataset, level) pairs:
identical to the last bit. If the levels cannot be read, the plots fall back to
`tab20` with a warning; the result files are unaffected either way.

The prediction index is written as bare cell ids, so `pd.read_csv` will type a
numeric id as `int64`. Cast to `str` before joining against `hovernet.json` keys.

## How a slide becomes HistoCell tiles

Upstream reads a directory of pre-cut tile images plus one HoVer-Net `.json`
per tile. The benchmark stores one whole-slide image and one whole-slide
segmentation, so the cutting happens in memory (`bench.py`, and
`data.SlideTileDataset`). No intermediate tiles are written to disk.

### Training tiles — one per spot

The paper cuts tiles "according to the size of spots (e.g. 55 μm for 10x
Visium)" and resizes them to 256 x 256. Here the tile side is
`uns['spatial'][lib]['scalefactors']['spot_diameter_fullres']` — 200.9 px, i.e.
the same 55 μm — and the tile is centred on the spot. Override with `--tile-px`.

Keeping the tile at the spot diameter is what makes the rest of the method
transferable unchanged. In particular the graph radius: nuclei are linked when
they are closer than 40 px *in tile space*, and since the tile is always
stretched to 256 px, 40 tile-px is 55/256 x 40 = **8.6 μm** here exactly as it
is on Visium.

### Inference tiles — a partition of the slide

The spots cover under a third of the tissue, so predicting only inside them
would leave most nuclei unannotated. Instead the slide is partitioned into a
regular grid at the *same* tile size, and every tile holding at least one
nucleus is predicted. This is the paper's own recipe for data without spots
(equations 1–5, where the Xenium area is "uniformly partitioned into grids to
simulate the Visium ST spots"), and it means each nucleus is seen exactly once,
at exactly the magnification the model was trained on. Verified on `skin_s4`:
98,678 nuclei, 98,678 assignments, 98,678 distinct — full coverage, no double
counting. `--infer-tiling spots` restricts inference to the spot tiles instead.

Spot-level proportions are then obtained the way the paper does it — "the
predicted single-nucleus-level cells in each spot were clumped to mimic
spot-level cell proportions" — by averaging the per-nucleus probabilities over
the cells of each spot.

### Mapping predictions back to cells

Upstream stores each nucleus's tile-local coordinates and recovers cells
afterwards by matching coordinates. Here the dataset carries the nucleus's row
number in the slide-wide table through the batch (`cell_index`, padded with
-1), so a prediction returns to its `hovernet.json` cell id exactly, with no
nearest-neighbour step. Nothing the model sees changes.

Two checks back this up.

*The crops land on the right pixels.* Cropping a nucleus out of its resized tile
and cropping it straight from the whole-slide image agree to a median 2.3/255
per pixel, against 17.5/255 when the box is deliberately shifted by 3 px.

*The model sees exactly what upstream would have fed it.* Before this adapter was
accepted, the same tiles were written to disk in the layout `TileBatchDataset`
expects and both datasets were run over them, with augmentation disabled on each
side. Every field was compared elementwise:

```
compared 24 tiles
  max |adapter - upstream|  tissue  image  mask  size  adj  cell_coords  cell_types
                            0.0e+00 for all seven
```

Bit-identical, so the model cannot tell which loader fed it. The throwaway script
that produced this was removed once it had served its purpose.

## Tissue compartments

The auxiliary cross-entropy head needs upstream's `tcs` file: `dict` maps each
cell-type index to a compartment, `list` is the compartment vocabulary, and
`HoVerNet` maps the six PanNuke classes to compartments.

Upstream ships one hand-written file per tissue under `./tcs`. The benchmark's
vocabulary changes with the annotation level, so the file is derived from the
cell-type names instead, with the same meaning and the same three compartments
`["Epi", "TME", "Stromal"]`. The PanNuke row is upstream's constant, copied
verbatim: `[2, 0, 1, 2, 2, 0]`. On the benchmark's levels this gives, e.g.

```
level 0  Immune -> TME    Structural -> Stromal   Epithelial -> Epi   Melanocyte -> Epi
level 2  B_Plasma -> TME  T_NK -> TME  Myeloid -> TME
         Fibroblast_Myofibroblast -> Stromal   Blood_vessel -> Stromal   Epithelial -> Epi
```

A handful of labels at the deeper levels are too short to match on a substring
without catching unrelated names (`B`, `T`, `DC`, `T_CD8`) and are resolved by
exact name instead. Across all 218 cell-type names of every sample and every
level of this benchmark, all 218 resolve. Names matched by no rule fall back to
`Stromal` — upstream's own catch-all for the PanNuke "nolabel" class — and are
logged loudly. The file used is always
written to `tissue_compartment.json` next to the results, and
`--tissue-compartment` overrides the whole thing.

The 0.75 rule that turns proportions into a compartment label, and the extra
"mixture" class, are upstream's and are untouched.

## What is unchanged

Everything that constitutes the method:

* **Architecture** (`model/arch.py`, not modified): ImageNet ResNet-18 encoder
  for both the tile and the nucleus crops, 16-d cell-size embedding fused by a
  linear layer, one graph-attention layer, a 2-step unidirectional LSTM, a
  linear + softmax cell-type head and the tile compartment head. On modern
  torchvision, `resnet18(pretrained=True)` still resolves to the very same
  `resnet18-f37072fd.pth` (IMAGENET1K_V1) checkpoint upstream used.
* **Losses**: symmetric KL between the tile-mean of the per-cell softmax and the
  deconvolved proportions, plus cross-entropy on the compartment — copied
  literally from `train.py`, `torch.nn.KLDivLoss()` default reduction included.
  The cell-state consistency term does not apply (see below).
* **Nucleus crops**: taken from the *normalised* 256 x 256 tile at the bounding
  box, resized to 30 x 30. Cell-size feature = bounding-box side / 256.
* **Graph**: binary adjacency at 40 tile-px, symmetrised, plus the identity.
* **`max_cell_num = 256`**. Note this is a *method* parameter, not just a batch
  shape: padded slots keep a non-zero attention weight in the GAT
  (`softmax(dim=1)` over a fully-masked column returns a uniform row), so their
  features do reach the real cells. Left at 256.
* **Augmentation**: ColorJitter + RandomGrayscale when training, ImageNet
  normalisation always.
* **Batch size 32** for training, 16 for inference, seed 47 — all as released.
* **Inference module mode**: upstream's `infer.py` calls `model.train()` before
  predicting, so batch-norm uses batch statistics and dropout stays on. That is
  the default here — and it is measurably the better one, see below.
  `--eval-mode eval` switches to `model.eval()`; note that two dropouts inside
  `arch.py` are written `training=True` and stay active either way — untouched.

### Why the odd inference mode is kept

Predicting with the module in train mode looks like a bug: batch-norm
normalises by whatever tiles share the batch rather than by the running
statistics, and dropout stays on, so the answer for a cell is not a fixed
function of that cell. Measured on the delivered 50-epoch checkpoints:

| | argmax agreement | mean abs. prob. difference |
|---|---|---|
| skin_s4 L0, train vs eval mode | 50.0% | 0.211 |
| skin_s4 L0, train vs train (re-run) | 83.4% | 0.054 |
| lung_s3 L0, train vs eval mode | 61.1% | 0.182 |
| lung_s3 L0, train vs train (re-run) | 78.0% | 0.088 |

`eval` mode is nonetheless far *less* accurate. On skin_s4 level 0 the true
composition is Immune 0.307 / Structural 0.172 / Melanocyte 0.487 / Epithelial
0.035; train mode returns 0.314 / 0.168 / 0.502 / 0.015 and eval mode returns
0.642 / 0.121 / 0.217 / 0.019. The running batch-norm statistics never catch up
because training runs on colour-jittered tiles and the two hard-coded dropouts
keep perturbing the activations. So upstream's default is kept, and the price
is that re-running inference relabels roughly 20% of cells. Averaging several
passes would stabilise it; HistoCell does not, so neither does this.

## What was changed, and why

Only I/O:

1. **Tiles are cut in memory** from `he.tiff` instead of read from a directory
   of `.jpg`/`.png` files, and nuclei come from one whole-slide `hovernet.json`
   instead of one `.json` per tile. New class `SlideTileDataset` in `data.py`;
   `TileBatchDataset` is left exactly as it was.
2. **Bounding boxes are recomputed from the contour** — see the next section.
   This is a fix for the input data, not a change to the method.
3. **`cell_index` is carried through the batch** so predictions map back to the
   benchmark's cell ids exactly.
4. **The `tcs` file is generated** from the level's cell-type names when not
   supplied, as described above.
5. **Inference tiling is a grid** covering the whole slide, so every nucleus is
   annotated.
6. **Paths and parameters come from the command line** rather than from the
   `./demo/data/...` layout hard-coded in `configs.py`. The new
   `_get_bench_config` carries only method settings.
7. **Environment**: python 3.10 / torch 2.5.1 instead of python 3.7 / torch
   1.12.1, because the pinned stack cannot read `.h5ad` or pyramidal `.tiff`.
   The encoder weights are identical, as noted above.

Not adapted, because the benchmark has no cell states: `train_state.py`,
`TileBatchStateDataset` and the `HistoState` model. They are left untouched and
unused, along with `train_oneout_pannuke.py` (the PanNuke cross-validation).

## Where the paper and the released code disagree

Two hyper-parameters differ between the article and `configs.py`. The article
wins by default, and both are one flag away:

| | article | released `configs.py` | default here |
|---|---|---|---|
| epochs (cell-type stage) | 50 | 41 | **50** (`--epochs`) |
| Adam learning rate | 1e-4 | 5e-4 | **1e-4** (`--lr`) |

The article is explicit: "we trained the model for 50 epochs with the
supervision of cell compartment loss and KL divergence loss" and "the model
parameters were updated via the Adam optimizer with a learning rate of
1 x 10⁻⁴". Use `--epochs 41 --lr 5e-4` for the released-code behaviour.

## A data problem found on the way: the `bbox` field

In the benchmark's `hovernet.json` files the `bbox` entry does not describe the
nucleus it is attached to. It is the contour's box with the two per-tile offsets
exchanged:

```
stored bbox = [[ymin - K, xmin + K], [ymax - K, xmax + K]],  K = (tile_row - tile_col) * tile_step
```

so it is only correct for the few percent of nuclei sitting on the tile
diagonal, where `K = 0` — measured at 3.6 % (skin_s4) to 9.2 % (lung_s3) of
nuclei. `centroid` and `contour` are consistent with each other and with the
image; only `bbox` is affected.

This matters here because HistoCell crops each nucleus at its bounding box:
taken at face value, the model would be shown a patch of tissue hundreds to
tens of thousands of pixels away from the cell it is meant to classify.
`bench.load_seg_dict` therefore derives the box from the contour — which is what
HoVer-Net's own `get_bounding_box` computes, and what PanoSpace's adapter
already did — and logs the disagreement rate on every run. Any other consumer
of these files that reads `bbox` directly is affected and should do the same.

## Not implemented: the tutorial's Bayesian re-weighting

`tutorial/tutorial.ipynb` re-weights the per-cell probabilities by a
`conditional_prob_*.tsv` matrix before taking the argmax:

```python
new_prob[idx] = prob * normalized_conditional_mat[:, prior_type]
```

This step appears in no equation of the paper, the matrix is shipped as an
opaque data file, and no code estimating it was released — so reproducing it
would mean inventing it. The predictions here are the model's own softmax, which
is what the paper's equations (18)–(19) define. The HoVer-Net class of every
nucleus is still available (it is what the `tcs` `HoVerNet` row maps), so the
step can be added later if the matrix is ever published.

## Files added / modified

| | |
|---|---|
| added | `bench.py`, `run_histocell.py`, `setup_env.sh`, and `HEDeST/run_histocell.sh` |
| modified | `data.py` (added `SlideTileDataset`), `configs.py` (added `_get_bench_config`), this `README.md` |
| removed | `model/__pycache__/`, `utils/__pycache__/` (stale python-3.7 bytecode) |

Nothing else in the repository was touched.
