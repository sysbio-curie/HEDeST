<img src="references/HEDeST_logo.png" align="left" width="150"/>

<h1 style="margin-top: -90px;">
    HEDeST: An Integrative Approach to Enhance Spatial Transcriptomic Deconvolution with Histology
</h1>

**HEDeST** is a deep-learning framework for assigning cell types to single cells on H&E slides using **deconvoluted spatial transcriptomics** data.

![figure](references/method.png)

## Installation

HEDeST uses two conda environments: **hedest-env** for the pipeline, the feature extraction, the model and the analysis, and **hovernet-env** for the tissue mask and the nuclei segmentation. HoVer-Net is kept apart because it needs Python 3.8 and its own torch build; the pipeline simply calls it as a subprocess.

```
git clone git@github.com:lucagortana/HEDeST.git
cd HEDeST
./setup_env.sh
```

The script creates both environments, installs openslide with conda and the rest with pip, records ``PYTHONPATH`` and ``LD_LIBRARY_PATH`` in them so the repository works as a source tree, and checks that the imports work. Useful options: ``--hedest-only``, ``--hovernet-only``, ``--force`` to rebuild an existing environment, and ``HEDEST_ENV=my-env ./setup_env.sh`` to change the names.

To install by hand instead, follow the header of ``requirements.txt`` and ``requirements-hovernet.txt``.

Then point the HoVer-Net stages at the second environment, in your pipeline configuration:
```yaml
mask:
  python: /path/to/envs/hovernet-env/bin/python
segmentation:
  python: /path/to/envs/hovernet-env/bin/python
```

The pinned builds target linux-x86_64 with a CUDA GPU (cu116 for hedest-env, cu118 for hovernet-env). Everything except the mask and the segmentation runs in ``hedest-env``:
```
conda activate hedest-env
```

### The model weights

Two sets of weights are needed, and neither ships with the repository.

**HoVer-Net** (nuclei segmentation). Download the PanNuke checkpoint
``hovernet_fast_pannuke_type_tf2pytorch.tar`` from the [HoVer-Net repository](https://github.com/vqdang/hover_net#model-weights),
put it wherever you like and give its path as ``segmentation.model_path`` in the pipeline configuration.

**H-Optimus-0** (cell embeddings). It is a **gated** model on the Hugging Face hub, so it cannot simply be downloaded:

1. Ask for access on the model page, https://huggingface.co/bioptimus/H-optimus-0, and wait for it to be granted.
2. Authenticate the machine once, either with ``huggingface-cli login`` or by exporting a token with ``export HF_TOKEN=hf_...``.
3. The first run downloads about 4.3 GB into ``~/.cache/huggingface/hub/models--bioptimus--H-optimus-0``; every later run reads the cache.

Three precautions are worth knowing:

- **The cache is the weights.** Once it is there, HEDeST reads it first and never calls the hub, so the embeddings keep working with no network, with an expired token, or on a cluster node that has no outbound access. To move the model to such a machine, copy that cache directory over — nothing else is needed.
- **Do not let the cache be split across machines.** Set ``HF_HOME`` (or ``HF_HUB_CACHE``) to a shared path if the home directory is not shared, otherwise every node downloads its own copy.
- **A 401 means the gate, not the network.** ``GatedRepoError: 401 ... Access to model bioptimus/H-optimus-0 is restricted`` means the token is missing, expired, or belongs to an account whose request was never granted. HEDeST turns this into an explicit message telling you which of the two to fix; it is never a reason to re-download anything.

Nothing has to be configured for any of this: ``hedest/features/hoptimus.py`` tries the cache, then the hub.

## Code structure
The code is structured as follows :
```
hedest/              → Source code for HEDeST and analysis tools
  pipeline.py        → End-to-end pipeline (slide → predictions)
  main.py            → HEDeST training alone
  features/          → Cell feature extraction (H-Optimus-0)
  slide.py           → Slide checking and conversion to pyramidal TIFF
  spots.py           → Spot geometry, cell-to-spot mapping and spot pies for QuPath
  aggregate_seeds.py → Combines the seeds of one study
  analysis/          → Visualization and analysis of the predictions
gridsearch/          → Parameter gridsearch against ground truth cell types
benchmark/           → Benchmarking notebooks
case_study/          → Notebook for the case study (tutorial)
external/            → External tools (some modified for HEDeST)
simulations/         → Code for generating and analyzing simulated data

setup_env.sh              → Creates the two conda environments
requirements.txt          → hedest-env (pipeline, features, model, analysis)
requirements-hovernet.txt → hovernet-env (mask, segmentation)
```

## Usage

![figure](references/pp_train.png)

To use HEDeST, you need an H&E slide (pyramidal .tif) and a CSV file with the cell-type proportions per spatial transcriptomics spot. Everything else is produced by the pipeline: tissue mask → nuclei segmentation → cell embeddings → cell types.

### Full pipeline

```
python hedest/pipeline.py init-config my_run.yaml   # writes a documented template
# fill in the slide, the output directory, the proportions and the HoVer-Net checkpoint
python hedest/pipeline.py run my_run.yaml
```

This runs the five stages in order and writes everything under ``out_dir`` (``slide/``, ``mask/``, ``segmentation/``, ``features/``, ``model/``), plus a ``pipeline.json`` recording what was produced.

- **check** — verifies the slide opens, is pyramidal and has a known resolution, and converts it to a pyramidal TIFF otherwise. If it cannot, it stops and says why.
- **mask** — computes the tissue mask. HoVer-Net's automatic mask is never used: the segmentation stage refuses to run without a mask of ours.
- **segment** — HoVer-Net nuclei segmentation. Cells are numbered by their position in the JSON, and that numbering ties the crops, the embeddings and the spot dictionary together.
- **features** — H-Optimus-0 tile embeddings pooled per nucleus.
- **train** — HEDeST itself.

HoVer-Net usually needs its own environment: set ``segmentation.python`` to that interpreter in the config. Set ``segmentation.image_dict_path`` to a ``.pt`` path if you also want the cell crops for later plotting (they are large, so the default is to skip them). Set ``train.save_spot_pies`` to also draw the spot proportions as pies for QuPath (see [Analysing the results](#analysing-the-results)).

Run only part of it with ``--stages``, for instance ``--stages mask,segment``. Each stage checks the outputs of the previous one and reuses what is already there.

### Running a step on its own

```
# --- in hovernet-env ---

# tissue mask
python external/hovernet/run_mask.py slide.tif mask/slide.png --mask-level 3

# nuclei segmentation (add --image_dict_path cells.pt instead of --skip_image_dict to also save the crops)
python external/hovernet/run_infer.py --gpu=0 --nr_types=6 --model_mode=fast \
    --model_path=hovernet_fast_pannuke_type_tf2pytorch.tar --mpp=0.2754 \
    wsi --input_dir=slide_dir/ --output_dir=seg/ --input_mask_dir=mask/ --skip_image_dict

# --- in hedest-env ---

# check the slide, and convert it if OpenSlide cannot read it
python hedest/pipeline.py check slide.tif --convert

# cell embeddings
python hedest/features/hoptimus.py slide.tif seg/slide.json features/slide_hoptimus.pt --mpp 0.2754

# HEDeST
python hedest/main.py features/slide_hoptimus.pt proportions.csv \
    --json-path seg/slide.json --path-st-adata adata.h5ad --mpp 0.2754 --out-dir results/
```

### Adjustment and spot geometry

Prior Probability Shift Adjustment (highly recommended) is applied automatically, and two options control it:
- ``--adjustment interpolated|nearest``: for the cells outside spots, ``interpolated`` (default) uses a distance-weighted mean of the ≤3 nearest spots, ``nearest`` uses the proportions of the closest spot.
- ``--gated/--no-gated``: ``--no-gated`` (default) adjusts every cell, ``--gated`` adjusts only the cells inside spots.

Adjusting the cells outside spots requires the HoverNet segmentation .json file, the AnnData object for your slide and the slide name. Without them only the cells inside spots are adjusted, and a warning tells you so. Gated runs and fully simulated datasets need none of the three.

The spot diameter in AnnData objects is for visualization purposes only. Pass ``--mpp`` (microns per pixel) so the real diameter is used instead (diameter = 55 / mpp); the pipeline does it for you. To find your mpp, use ``external/hovernet/get_tiff_resolution.py``, ``python hedest/pipeline.py check slide.tif``, or open the image in QuPath.

### Several seeds

HEDeST is supervised at the spot level, so different seeds can assign a given cell differently while fitting the spots equally well. Running a few seeds and averaging them gives a better prediction, and how much they agree is a confidence measure that needs no ground truth:
```
for seed in 0 1 2 3 4; do python hedest/main.py ... --out-dir results/seed_${seed} --rs $seed; done
python hedest/aggregate_seeds.py results/ --json-path seg/slide.json
```
This writes ``info_aggregated.pickle`` (mean predictions, per-cell spread and agreement) and ``stats_aggregated.xlsx`` next to the seed folders.

## Analysing the results

Each run writes ``info.pickle`` (everything the analysis needs) and ``stats.xlsx`` (the same numbers as a spreadsheet: composition per cell type, and how well the per-spot means reproduce the deconvolution, on all spots and on the held-out ones).

For anything more, open the run with ``hedest.analysis``. **A ground truth is never required** — it is assumed you do not have one:
```python
from hedest.analysis import PredAnalyzer, load_run

run = load_run("results")                 # a run, or a folder of seed_* runs
analyzer = PredAnalyzer(run, seg="seg/slide.json", slide_path="slide.tif",
                        adata=adata, adata_name="sample", mpp=0.2754)

analyzer.describe()                       # what this run is
analyzer.spot_metrics(subset="held_out")  # does it reproduce the deconvolution?
analyzer.plot_proportion_scatter()
analyzer.visualizer().plot_celltype_map() # every cell, coloured, over the whole slide
analyzer.export_predictions("cells.csv")  # or .export_geojson(...) for QuPath
```

The package is organised as:
- ``analysis.loaders`` — ``load_run`` / ``load_seed_runs``, which read a run directory.
- ``analysis.pred_analyzer`` — ``PredAnalyzer``: composition, spot-level metrics, confidence, the effect of the adjustment, cell crops, morphology, neighbourhoods, exports. With a ground truth (``set_ground_truth``) it also gives cell-level metrics and a confusion matrix.
- ``analysis.postseg`` — ``SlideVisualizer``: the slide with the segmentation, the spots or the predictions drawn on it, zoomable through plotly. It needs no run, so it is also the way to check a segmentation before training. Windows are given in full-resolution coordinates and large ones are downsampled automatically.
- ``analysis.seeds`` — ``SeedEnsemble``: agreement between seeds, per cell and per cell type.
- ``analysis.plots`` and ``analysis.palette`` — the stateless helpers and the colour of each cell type.

Every plotting function returns a closed matplotlib figure: display it as the value of a notebook cell, or pass ``savefig=``.

Colours come from a single ``Palette``, so a cell type keeps its colour across every plot, overlay and export. One is built from the cell types when you do not give one; to impose your own, pass ``palette=`` to ``PredAnalyzer`` and ``SlideVisualizer``:
```python
from hedest.analysis.palette import Palette

palette = Palette.from_colors({"T": "#d62728", "B": "tab:blue"}, names=run.ct_list)
analyzer = PredAnalyzer(run, palette=palette, ...)
```
Any matplotlib colour works, the cell types you leave out keep their default colour, and ``palette_from_yaml`` rebuilds the palette a run was exported with from the colour dictionary written next to its GeoJSON.

### In QuPath

With ``--save-geojson``, a run writes its cells with their predicted type (``hedest_predictions_adj.geojson``, and ``hedest_predictions_aggregated_adj.geojson`` for an aggregate): drag it onto the slide. With ``--save-spot-pies`` (``train.save_spot_pies`` in the pipeline, which writes a single file whatever the number of seeds, or ``hedest.spots.export_spot_pies`` from Python at any time, no model needed), ``spot_pies.geojson`` draws every spot at its centre with its real diameter, as a pie of the proportions HEDeST is trained on, with the colours of the cells (give the same ``--color-dict-file`` to both if you use one). Load the two files together to switch between the two scales: the cells are *detections* and the wedges *annotations*, so ``D`` shows or hides the cells, ``A`` the pies, ``F`` and ``Shift+F`` fill them. The wedges are locked; each one holds its proportion as a measurement and its spot barcode as metadata.

## Tutorial

Download the dataset **'Human Breast Cancer: Ductal Carcinoma In Situ, Invasive Carcinoma (FFPE)'**:

```
wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_image.tif
wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz
wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
```

Then, create a folder named ``Visium_FFPE_Human_Breast_Cancer``. Inside it, make sure you have a ST folder containing :
- a **spatial/** folder with low resolution images, scale factors and the list of tissue positions,
- the file ``filtered_feature_bc_matrix.h5``

You can find the proportion file in the **case_study** directory.

After running preprocessing and training, open the notebook ``case_study/DCIS_study.ipynb`` for analysis and visualization. The cell annotations performed by HEDeST can also be visualized in QuPath with the geojson output.

## Gridsearch

``gridsearch/`` trains HEDeST for every combination of parameter lists, on several datasets, annotation levels and feature types, and scores each model against ground truth cell types (balanced accuracy, with and without PPSA). A run is exactly what ``hedest/main.py`` trains with the same parameters and seed: it uses the same training and PPSA code, but each dataset is loaded only once for all its runs, with its embeddings kept on the GPU.

A study is described by one YAML file. Relative paths are relative to that file:

```yaml
out_dir: ../models/my_gridsearch   # models, predictions and scores of every run
results_dir: results/my_gridsearch # tables
plots_dir: plots/my_gridsearch     # figures

grid:                              # every combination is trained (any hedest/main.py option)
  lr: [0.0001, 0.001]
  alpha: [0.0, 0.01]
  divergence: [kl, l2]
fixed: {epochs: 100, batch_size: 64}   # the other options keep the hedest/main.py defaults
seeds: [0, 1, 2]

datasets:
  - name: my_slide
    root: /data/my_slide           # paths may use {level} and any entry of the dataset
    levels: [lv0, lv1]             # annotation levels, one proportion file each
    features: {hoptimus0: "{root}/hoptimus_embed.pt"}
    proportions: "{root}/proportions_{level}.csv"
    spot_dict: "{root}/spot_dict.json"
    adata: "{root}/adata.h5ad"
    adata_name: my_slide
    segmentation: "{root}/hovernet.json"
    ground_truth: {path: "{root}/cell_types.csv", index_col: cell_id, column: "{level}"}
```

The ground truth gives one label per segmented cell: a CSV with a label column (as above), a JSON ``{cell_id: label}`` (``{path: gt.json, key: gt}`` if it is nested under a key), or a CSV with one column per cell type (the label is the argmax). If your labels come from another segmentation (e.g. DAPI), match its cells to the HoVer-Net cells first.

Then:
```
python gridsearch/gridsearch.py check   my_gridsearch.yaml          # checks the inputs (no training)
python gridsearch/gridsearch.py plan    my_gridsearch.yaml          # lists the units (feature x dataset x level) and the progress
python gridsearch/gridsearch.py run     my_gridsearch.yaml --unit 0 # trains every run of a unit (index or feature/dataset/level)
python gridsearch/gridsearch.py collect my_gridsearch.yaml          # all scores -> {results_dir}/runs.csv
python gridsearch/plots.py              my_gridsearch.yaml          # figures -> {plots_dir}
```

Each run writes ``best_model.pth``, ``history.png``, ``metrics.json`` (parameters, scores, loss history, confusion matrices) and ``predictions.npz`` (raw and PPSA probabilities) to ``{out_dir}/{feature}/{dataset}/{level}/{combination}/seed_{seed}/``. The figures go in three steps, for each feature type:
- all the results: the balanced accuracy of every combination (over all datasets, per group of datasets, per number of cell types, and per dataset);
- the selection: the combinations are ranked two ways, by mean balanced accuracy and by mean rank (ranked within each dataset-level, then averaged, so that every dataset-level weighs the same whatever its number of cell types). The best mean rank is elected with a Friedman test, and each other combination is compared with it (Wilcoxon signed-rank test, Holm correction), which gives the combinations that are not significantly worse (``{results_dir}/ranking_*.csv`` and ``ranking_tests.csv``);
- the analysis of the selected combination on every dataset: confusion matrices, effect of PPSA (everywhere, inside or outside spots, on the training or test spots) and comparison of the feature types. All the bars of a dataset come from the same runs. The selected combination is the best mean rank over all datasets without PPSA (``plots: {select: ppsa}`` in the YAML to rank with PPSA, ``plots: {combination: {hoptimus0: ...}}`` to choose it), see ``{results_dir}/selected.csv``.

## Some classic errors
During segmentation, you can get the ``OSError: [Errno 39] Directory not empty: 'cache'`` error. Make sure to delete everything you have in this repository and apply chmod 777. \
Also, you can get the Openslide's error ``openslide.lowlevel.OpenSlideUnsupportedFormatError: Unsupported or missing image file``. The pipeline detects this and converts the slide for you; ``python hedest/pipeline.py check your_file.tif --convert`` does it on its own. To convert it yourself instead:
```
vips tiffsave your_file.tif output-pyramidal.tif --tile --pyramid --bigtiff --compression jpeg --Q 90
```

## Preprint
https://www.biorxiv.org/content/10.64898/2026.01.06.697922v1

```
@article{gortana_hedest_2026,
	title = {{HEDeST}: {An} {Integrative} {Approach} to {Enhance} {Spatial} {Transcriptomic} {Deconvolution} with {Histology}},
	doi = {10.64898/2026.01.06.697922},
	journal = {bioRxiv},
	author = {Gortana, Luca and Chadoutaud, Loic and Bourgade, Raphael and Barillot, Emmanuel and Walter, Thomas},
	year = {2026}
}
```
## Credits
We would like to thank the authors of HoVerNet (https://github.com/vqdang/hover_net), CellViT (https://github.com/TIO-IKIM/CellViT), MoCo-v3 (https://github.com/facebookresearch/moco-v3) and H-Optimus-0 (https://huggingface.co/bioptimus/H-optimus-0), whose work we adapted for this project.
