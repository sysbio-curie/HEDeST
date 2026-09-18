<img src="references/HEDeST_logo.png" align="left" width="150"/>

<h1 style="margin-top: -90px;">
    HEDeST: An Integrative Approach to Enhance Spatial Transcriptomic Deconvolution with Histology
</h1>

**HEDeST** is a deep-learning framework for assigning cell types to single cells on H&E slides using **deconvoluted spatial transcriptomics** data.

![figure](references/method.png)

## Installation
To create your conda environment and install the requirements using ``pip``
```
git clone git@github.com:lucagortana/HEDeST.git
cd HEDeST
conda create -y -n hedest-env python=3.9
conda activate hedest-env
pip install -r requirements.txt
CODEPATH=$(realpath .)
conda env config vars set PYTHONPATH=$PYTHONPATH:$CODEPATH:$CODEPATH/external:$CODEPATH/external/hovernet:$CODEPATH/external/mocov3
conda deactivate
conda activate hedest-env
```
To install pytorch and torch-scatter
```
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.1.1%2Bpt113cu116-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.1+pt113cu116-cp39-cp39-linux_x86_64.whl
```
To install openslide
```
conda install -c conda-forge openslide=3.4.1
```

## Code structure
The code is structured as follows :
```
hedest/        → Source code for HEDeST and analysis tools
benchmark/     → Benchmarking notebooks
case_study/    → Notebook for the case study (tutorial)
external/      → External tools (some modified for HEDeST)
simulations/   → Code for generating and analyzing simulated data

run_mask.sh
run_hovernet.sh
run_moco_ssl.sh
run_hedest.sh   → Shell scripts to run the full HEDeST workflow (in this order)
```

## Usage

![figure](references/pp_train.png)

### Pre-processing

To use HEDeST, you need:
- An H&E slide in .tif format
- A CSV file containing cell-type proportions per spatial transcriptomics spot

Then follow these preprocessing steps:
- use ``run_mask.sh`` to get a binary mask of your H&E slide
- use ``run_hovernet.sh`` to segment the slide and extract single-cell images in a dictionnary
- use ``run_moco_ssl.sh`` to train moco-v3 and infer the embeddings

### Training

Once you have the embeddings and the proportions, you can run HEDeST using ``run_hedest.sh``. Prior Probability Shift Adjustment (highly recommended) is applied automatically, and two options control it:
- ``--adjustment interpolated|nearest``: for the cells outside spots, ``interpolated`` (default) uses a distance-weighted mean of the ≤3 nearest spots, ``nearest`` uses the proportions of the closest spot.
- ``--gated/--no-gated``: ``--no-gated`` (default) adjusts every cell, ``--gated`` adjusts only the cells inside spots.

Adjusting the cells outside spots requires the HoverNet segmentation .json file, the AnnData object for your slide and the slide name. Without them only the cells inside spots are adjusted, and a warning tells you so. Gated runs and fully simulated datasets need none of the three.

The spot diameter in AnnData objects is for visualization purposes only. If you want to get the real diameter (in pixels) of ST spots, we recommend you to change it with your image resolution (diameter = 55 / mpp). To know your mpp (microns per pixel), you can use ``external/hovernet/get_tiff_resolution.py`` or open the image in QuPath.

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
Also, you can get the Openslide's error ``openslide.lowlevel.OpenSlideUnsupportedFormatError: Unsupported or missing image file``. In that case, we recommend to check the properties of your file with the command ``openslide-show-properties your_file.tif``. If your file is indeed incompatible with Openslide, then you can try :
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
