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

## Code structure
The code is structured as follows :
```
hedest/              → Source code for HEDeST and analysis tools
  pipeline.py        → End-to-end pipeline (slide → predictions)
  main.py            → HEDeST training alone
  features/          → Cell feature extraction (H-Optimus-0)
  slide.py           → Slide checking and conversion to pyramidal TIFF
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

HoVer-Net usually needs its own environment: set ``segmentation.python`` to that interpreter in the config. Set ``segmentation.image_dict_path`` to a ``.pt`` path if you also want the cell crops for later plotting (they are large, so the default is to skip them).

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

``hedest/main.py`` takes any ``{cell_id: embedding}`` dictionary, so MoCo-v3 embeddings (``run_moco_ssl.sh``) work as well as H-Optimus-0.

### Adjustment and spot geometry

Prior Probability Shift Adjustment (highly recommended) is applied automatically, and two options control it:
- ``--adjustment interpolated|nearest``: for the cells outside spots, ``interpolated`` (default) uses a distance-weighted mean of the ≤3 nearest spots, ``nearest`` uses the proportions of the closest spot.
- ``--gated/--no-gated``: ``--no-gated`` (default) adjusts every cell, ``--gated`` adjusts only the cells inside spots.

Adjusting the cells outside spots requires the HoverNet segmentation .json file, the AnnData object for your slide and the slide name. Without them only the cells inside spots are adjusted, and a warning tells you so. Gated runs and fully simulated datasets need none of the three.

The spot diameter in AnnData objects is for visualization purposes only. Pass ``--mpp`` (microns per pixel) so the real diameter is used instead (diameter = 55 / mpp); the pipeline does it for you. To find your mpp, use ``external/hovernet/get_tiff_resolution.py``, ``python hedest/pipeline.py check slide.tif``, or open the image in QuPath.

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
We would like to thank the authors of HoVerNet (https://github.com/vqdang/hover_net) and MoCo-v3 (https://github.com/facebookresearch/moco-v3), whose work we adapted for this project.
