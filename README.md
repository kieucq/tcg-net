# Deep-learning reconstruction and prediction of tropical cyclogenesis 

This TCG-Net repo is a research pipeline for detecting favorable environments for tropical cyclone genesis (TCG) directly from gridded reanalysis, climate-model, and forecast data. The repository covers data standardization, construction of supervised positive/negative samples, training and fine-tuning of an adapted ResNet-18 classifier, sliding-window inference, and visualization of TCG probability maps and climatological frequency. The scientific method and model design are described in Le et al. (2026), Kieu et al. (2026).

## Contents

- [Scientific overview](#scientific-overview)
- [Model and data](#model-and-data)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [How to run](#how-to-run)
- [Outputs](#outputs)
- [Directory structure](#directory-structure)
- [Reproducibility notes](#reproducibility-notes)
- [References](#references)
- [Contact](#contact)

## 1. Scientific overview

TCG-Net formulates tropical cyclogenesis detection as binary image classification. Each sample is a multichannel meteorological window centered on a candidate location and time. A positive label represents the first recorded genesis time of a storm; negative examples are constructed using two complementary strategies:

- **Past Domain (PD):** samples at the same location but at earlier times. This strategy emphasizes the temporal question of why genesis occurs at a particular time rather than earlier.
- **Dynamic Domain (DD):** samples in the eight neighboring directions around the positive domain, optionally shifted backward in time. This strategy emphasizes the spatial question of why genesis occurs in one region rather than a nearby region.

The pipeline addresses the rarity of TCG events through temporal enrichment, random undersampling of negative cases, and positive-class weighting. During inference, the model is applied to windows across the Western North Pacific to produce spatial fields of positive-class probability.

The associated paper trains primarily with NASA MERRA-2 data and IBTrACS genesis labels. It reports that an adapted ResNet-18 can reproduce major features of Western North Pacific TCG climatology, including seasonality and spatial distribution, and that a selected subset of environmental channels is sufficient for this task. The workflow for this TCG-Net is given as below:

```text
Raw atmospheric data + storm tracks
              |
              v
Dataset-specific preprocessing and coordinate/variable standardization
              |
              +-------------------------------+
              |                               |
              v                               v
     Past Domain samples             Dynamic Domain samples
              |                               |
              +---------------+---------------+
                              v
                  CSV train/validation/test splits
                              |
                              v
                 ResNet-18 pretraining/fine-tuning
                              |
                              v
                 Sliding-window probability inference
                              |
                              v
                    Maps and climatological plots
```

## 2. Directory structure

```text
tcg-net/
├── README.md
├── config.json                      # Shared paths, domains, model inputs, and plotting options
├── config_loader.py                 # Attribute-style JSON loader
├── input/                           # Raw data; ignored by Git
│   ├── tracks/
│   ├── merra2/
│   ├── era5/
│   ├── ncep/
│   ├── cmip6/
│   ├── gfs/
│   └── ifs/
├── preprocess/
│   ├── merra2/                      # MERRA-2 NetCDF preprocessing
│   ├── era5/                        # ERA5 NetCDF preprocessing
│   ├── ncep/                        # NCEP FNL GRIB preprocessing
│   ├── gfs/                         # GFS preprocessing and valid-time renaming
│   ├── cmip6/                       # WRF-style CMIP6 preprocessing
│   ├── domain_extract/              # PD/DD sample extraction and dataset adapters
│   ├── scripts/                     # Older legacy extraction job wrappers (no use)
│   └── backup/                      # Historical preprocessing/analysis implementation
├── models/
│   ├── lib/
│   │   ├── Dataset/                 # NetCDF loading, normalization, and Lightning DataModule
│   │   ├── Model/PointOut/          # Adapted ResNet-18 classifier
│   │   ├── Progress/                # Lightning training loop and checkpoint callbacks
│   │   └── Utils/                   # Metrics, derived features, seeds, and evaluation helpers
│   ├── resnet18/                    # CSV preparation and pretraining entry points
│   ├── finetune/                    # Fine-tuning entry points and local checkpoint outputs
│   ├── prediction/                  # Sliding-window generation and multi-step inference
│   └── pre-trained/                 # Local pretrained checkpoints; ignored by Git
├── operation/
│   └── job_main.sh                  # End-to-end operational orchestration
├── postprocess/                     # Map and monthly-frequency plotting
└── output/                          # Generated data and figures; ignored by Git
```

Several workflow directories contain symlinks to the root `config.json`, `input/`, and `output/` so that scripts can use a shared configuration and data tree.

## 3. Model and data

### 3.1 Supported atmospheric inputs

The current preprocessing code includes readers for:

| Dataset/source | Expected raw format | Preprocessor | Notes |
| --- | --- | --- | --- |
| NASA MERRA-2 | NetCDF4 (`*.nc4`) | `preprocess/merra2/merra2_preprocess.py` | Primary reanalysis used in the paper |
| ERA5 | NetCDF (`*.nc`) | `preprocess/era5/era5_preprocess.py` | Used for reconstruction and fine-tuning |
| NCEP FNL | GRIB1 or GRIB2 | `preprocess/ncep/ncep_preprocess.py` | Referred to as `fnl` by domain-extraction and map-slicing commands |
| GFS | GRIB1, GRIB2, or `*.f000` | `preprocess/gfs/gfs_preprocess.py` | Implemented operational forecast/detection path |
| CMIP6/WRF-style output | NetCDF | `preprocess/cmip6/cmip6_preprocess.py` | Expects the WRF-style dimensions and variables used by the code |

`config.json` also contains placeholder paths for IFS and JMA. This checkout does not contain corresponding preprocessors. The operational wrapper mentions IFS, but its comments state that IFS has not been checked; the end-to-end operational path currently implemented is GFS.

### 3.2 Storm labels

The supervised extraction workflow expects IBTrACS CSV data in `input/tracks/`. The IBTrACS reader selects the Western North Pacific basin (`BASIN == "WP"`) and uses the first retained record for each storm as the positive genesis event. CMIP6 experiments can instead use the text track format handled by `Cmip6Tracks`.

### 3.3 Spatial and temporal conventions

The default root configuration uses:

- preprocessing domain: 50°S–70°N, 60°E–220°E;
- inference area: 0°–30°N, 100°–150°E;
- child window: 33 × 33 grid points at 0.5° spacing;
- sliding stride: 10 grid points in latitude and longitude;
- maximum backward sampling count: 20 steps.

MERRA-2 is natively handled at 0.5° latitude × 0.625° longitude and 3-hour intervals. ERA5 and the configured inference windows are sampled onto the 0.5° conventions used by this checkout. The effective number of hours represented by a `Step_*` model must be interpreted using the temporal resolution and enrichment setup used to build that model.

### 3.4 Input channels

The model input is controlled by `DYNAMIC_MODEL_DATASET` in `config.json`. The current configuration uses:

- surface/single-level fields: `PHIS`, `SLP`;
- pressure-level fields: `H`, `OMEGA`, `QI`, `QL`, `QV`, `RH`, `T`, `U`, `V`;
- derived fields: `VOR`, `DIV`;
- 25 pressure levels from 1000 to 100 hPa.

This produces 277 input channels: 2 single-level channels plus 25 levels × 11 three-dimensional/derived channels. Samples are normalized using the means and standard deviations in `models/resnet18/data_train_statistics.xlsx`; missing normalized values are replaced with zero.

Input variables, pressure levels, the normalization table, and model checkpoints must remain mutually consistent. Changing channels or levels requires retraining or fine-tuning a model with the same configuration.

### 3.5 ResNet-18 implementation

The classifier in `models/lib/Model/PointOut/ResNet_classification.py` contains:

- a 7 × 7 input convolution, batch normalization, ReLU, and max pooling;
- four residual stages with `[2, 2, 2, 2]` basic blocks and 64–512 channels;
- adaptive global average pooling;
- a two-output fully connected classification layer.

Training uses weighted cross-entropy and AdamW. The command-line defaults are a learning rate of `1e-4`, weight decay of `1e-2`, 100 epochs, and batch size 64; the supplied Slurm wrappers override some values, including batch size. Precision, recall, and F1 score are logged for the positive class. Checkpoints include the last model, best validation loss, best validation F1 score, and periodic 20-epoch snapshots.

Inference loads separate checkpoints for step indices 2, 4, …, 18 and writes their positive-class softmax probabilities to `Score_step2`, `Score_step4`, …, `Score_step18` columns.

## 4. Requirements

### 4.1 Platform

- Linux with Bash;
- Python 3.10 or newer (the code uses structural pattern matching);
- an NVIDIA GPU is strongly recommended for training and large inference runs;
- Slurm is optional but assumed by the supplied `job_*.sh` wrappers;
- sufficient storage for reanalysis data, extracted windows, and checkpoints.

The repository does not currently provide a pinned `requirements.txt`, Conda environment, or Python package definition. The principal third-party dependencies used by the active pipeline are:

- PyTorch and Lightning;
- NumPy, pandas, SciPy, scikit-learn, and openpyxl;
- xarray and netCDF4;
- cfgrib and ecCodes for GRIB input;
- tqdm;
- Matplotlib and Basemap;
- PyYAML.

Historical analysis utilities under `preprocess/backup/` additionally reference `alive-progress`, `xESMF`, PyTables, and h5netcdf.

### 4.2 Indiana University HPC environment

On systems that provide the corresponding module, initialize Python with:

```bash
conda deactivate
module load python/gpu/3.12.5
python --version
```

The existing Slurm scripts currently contain `module load python/gpu/3.10.10`. Either use the version available at your site or update your local job-script copies consistently. The source files in this checkout parse successfully with Python 3.12.5, but scientific-library and CUDA compatibility still depends on the installed environment.

For a non-module environment, install a PyTorch build appropriate for the system’s CUDA version, followed by the remaining dependencies. For example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch/CUDA build first.
python -m pip install lightning numpy pandas scipy scikit-learn openpyxl \
  xarray netCDF4 cfgrib eccodes tqdm matplotlib basemap PyYAML
```

### 4.3 Clone and site setup

```bash
git clone https://github.com/kieucq/tcg-net.git
cd tcg-net
```

Before running a wrapper script, review its Slurm account, partition, memory request, Python module, and site paths. Several wrappers currently set:

```text
src=/N/slate/ckieu/tcg-net/
output_dir=/N/scratch/ckieu/tcg-net/output
```

These values are installation-specific. Manual Python commands can be used without Slurm, provided that the paths in `config.json` and the working directory are correct.

## 5. Configuration

`config.json` is the central runtime configuration. Symlinks named `config.json` in workflow subdirectories point back to this file, so changing the root file affects all stages.

| Section/key | Purpose |
| --- | --- |
| `PRE_DOMAIN` | Geographic crop applied by dataset preprocessors |
| `IPATH` | Raw atmospheric and track input locations |
| `OPATH` | Preprocessed, positive, Past Domain, and Dynamic Domain output locations |
| `STEP_BACK_COUNT` | Number of earlier time steps considered during domain extraction |
| `DOMAIN_EXTRACTION_BATCH_SIZE` | Number of track records assigned to an extraction batch |
| `SLICING_WINDOW` | Inference area, child-window dimensions/resolution, stride, aggregation, and input/output paths |
| `DYNAMIC_DOMAIN` | Sliding-window directory, checkpoint path template, and prediction CSV path |
| `TCG_FREQUENCY` | Prediction/IBTrACS inputs and map or frequency-plot settings |
| `DYNAMIC_MODEL_DATASET` | Model variables, pressure levels, normalization statistics, and dataset CSV paths |

Relative paths are resolved from the process working directory. The supplied wrappers change into a workflow directory whose `input`, `output`, and `config.json` entries are symlinked to the repository-level locations. When invoking Python manually, the examples below run from the repository root or explicitly change to the expected model directory.

> **Destructive-path warning:** preprocessors call `CleanDir` on their configured output directory before writing new files. Confirm every `OPATH.*_PREP` value before starting a run.

## 6. How to run

### 6.1. Prepare the input layout

Raw data are intentionally excluded from Git. The expected high-level layout is:

```text
input/
├── tracks/  # IBTrACS CSV files
├── merra2/{pretrain,finetune,forecast}/
├── era5/{pretrain,finetune,forecast}/
├── ncep/{pretrain,finetune,forecast}/
├── cmip6/{pretrain,finetune,forecast}/
└── gfs/
    ├── pretrain/
    ├── finetune/
    ├── forecast/YYYYMMDDHH/
    └── detection/YYYYMMDDHH/
```

Set the relevant `IPATH` and `OPATH` entries in `config.json` before processing. Preprocessed files are standardized to the coordinate and variable names expected downstream and are normally written to `output/<DATASET>_extend/`, where DATASET denotes ERA5, GFS, MERRA-2,... or any data type you want to use. Note that for each dataset, there 3 sub-dirs `pretrain, finetune, forecast`, which are used for different purposes. 

### 6.2. Preprocess atmospheric data

Set domain size under "PRE_DOMAIN" in config.json and run the required dataset preprocessor from the repository root as below:

```bash
python preprocess/merra2/merra2_preprocess.py
python preprocess/era5/era5_preprocess.py
python preprocess/ncep/ncep_preprocess.py
python preprocess/cmip6/cmip6_preprocess.py
python preprocess/gfs/gfs_preprocess.py
```

Only run the command for the data configured in `config.json`. The preprocessors recursively find their supported input extensions, clear the configured output directory, process files in parallel, and write timestamped NetCDF files.

On Slurm, MERRA-2 and ERA5 wrappers should be submitted after adapting their site settings in case the preprocess run takes a long time:

```bash
sbatch preprocess/merra2/job_sbatch.sh merra2
sbatch preprocess/era5/job_sbatch.sh era5
```

### 6.3. Construct supervised domains for training

This step is required for pretraining or fine-tuning, but not when applying an existing model to already preprocessed operational/climate data.

From the repository root:

```bash
python preprocess/domain_extract/Extract_DynamicDomain.py --dataset merra2
python preprocess/domain_extract/Extract_PastDomain.py --dataset merra2
```

Valid extraction choices are `fnl`, `merra2`, `cmip6`, `era5`, and `gfs`. Replace `merra2` by your DATASET as needed. The commands generate:

- `output/<DATASET>_positive/` for genesis-centered positive windows;
- `output/<DATASET>_dynamic/` for eight-direction Dynamic Domain negatives;
- `output/<DATASET>_past/` for same-location earlier-time negatives;
- `output/tracks_preprocess/FIRST_<DATASET>_*.csv` for processed genesis records.

Both extraction commands recreate the dataset-specific positive directory; therefore, the second run replaces the positive files written by the first. These files represent the same time-zero positive cases when the configuration and input coverage are unchanged.

`preprocess/domain_extract/` and `preprocess/backup/` are present in the full working tree used to prepare this README but are currently matched by `.gitignore`. A fresh GitHub clone may not contain them. If they are absent, obtain the complete pipeline from the associated archive listed in [References](#references) or contact the maintainers.

### 6.4. Build CSV splits

The CSV generator expects generic `output/POSITIVE`, `output/PastDomain`, and `output/DynamicDomain` paths. Point these names to the dataset being trained. For MERRA-2, for example:

```bash
ln -sfn "$PWD/output/MERRA2_positive" output/POSITIVE
ln -sfn "$PWD/output/MERRA2_past" output/PastDomain
ln -sfn "$PWD/output/MERRA2_dynamic" output/DynamicDomain

python models/resnet18/generate_csv_all.py \
  --inp_dir "$PWD/output" \
  --out_dir "$PWD/output/csv"

python models/resnet18/split_data_training.py \
  --inp_dir "$PWD/output/csv" \
  --out_dir "$PWD/output/csv" \
  --train_start_year 1980 \
  --train_end_year 2016 \
  --test_start_year 2017 \
  --test_end_year 2023 \
  --rus_ratio 30
```

The generator writes `output/csv/all.csv`. The splitter creates `train.csv`, `val.csv`, `test.csv`, and an undersampled `test2.csv` under `output/csv/DynamicRemain_rus<RATIO>/Step_<N>/` for step indices 2–18. `split_data_training_enrichment.py` is the alternative enrichment splitter that includes time-zero through the selected step as positive cases.

The paper’s main protocol uses 1980–2016 for training/validation and 2017–2023 for testing. Some supplied job wrappers were later customized to different year ranges; inspect and set the ranges appropriate for the experiment being reproduced.

### 6.5. Pretrain ResNet-18

To train one step manually:

```bash
cd models/resnet18
python train_job.py \
  --project ResNet \
  --seed 45 \
  --ratio 30 \
  --step 2 \
  --weight 6 \
  --batch_size 64 \
  --inp_dir ../../output/csv/DynamicRemain_rus30/Step_2 \
  --out_dir ../pre-trained/dynamic/ResNet_r30_w6/Step_2
cd ../..
```

The output directory receives a version suffix such as `_v0`. Repeat with steps `2 4 6 8 10 12 14 16 18` to build the complete checkpoint set used by inference.

After adapting the hard-coded site paths and Slurm directives, the combined CSV/split/train wrapper can instead be submitted with:

```bash
sbatch models/resnet18/job_sbatch.sh 2
```

Expected checkpoints are stored under:

```text
models/pre-trained/dynamic/ResNet_r30_w6/Step_<N>_v0/checkpoints/
```

### 6.6. Fine-tune on another dataset

Prepare and link the target dataset’s positive/Past/Dynamic outputs, regenerate the CSV splits, and supply the matching pretrained checkpoint:

```bash
cd models/finetune
python finetune_job.py \
  --project ResNet \
  --pretrained ../pre-trained/dynamic/ResNet_r30_w6/Step_2_v0/checkpoints/last.ckpt \
  --seed 45 \
  --ratio 30 \
  --step 2 \
  --weight 6 \
  --batch_size 64 \
  --inp_dir ../../output/csv/DynamicRemain_rus30/Step_2 \
  --out_dir ResNet_r30_w6/Step_2
cd ../..
```

The supplied fine-tuning wrapper defaults to ERA5 and can be submitted after its paths, years, resources, and module version are reviewed:

```bash
sbatch models/finetune/job_sbatch.sh 2
```

Fine-tuned checkpoints are expected under:

```text
models/finetune/ResNet_r30_w6/Step_<N>_v0/checkpoints/
```

### 6.7. Reconstruct TCG fields from climate or reanalysis data

For an existing preprocessed dataset and checkpoint set, configure:

- `SLICING_WINDOW.INPUT_PATH` and `OUTPUT_PATH`;
- `SLICING_WINDOW.AREA`, `CHILD_AREA`, and `NUM_STEP` if the grid differs;
- `DYNAMIC_DOMAIN.SLIDE_DIR`;
- `DYNAMIC_DOMAIN.MODEL_TEMP`, retaining `{0}` where the step number is substituted;
- `DYNAMIC_DOMAIN.PREDICTION_CSV`;
- `DYNAMIC_MODEL_DATASET` variables, levels, and statistics to match the checkpoints.

Then run, for example, ERA5 inference:

```bash
cd models/prediction
python map_slide.py --dataset era5
python main.py
cd ../..
```

Allowed map-slicing dataset identifiers are `fnl`, `merra2`, `cmip6`, `era5`, and `gfs`. `map_slide.py` creates the spatial windows and `data.csv`; `main.py` runs `dynamic.py`, loads step-specific checkpoints, and writes the combined probability CSV.

To resume only the prediction phase after sliding windows already exist:

```bash
sbatch models/prediction/job_dynamics.sh
```

### 6.8. Run the end-to-end GFS operational workflow

Place one cycle under either:

```text
input/gfs/forecast/YYYYMMDDHH/
input/gfs/detection/YYYYMMDDHH/
```

Ensure that the complete fine-tuned checkpoint set exists, then run from the repository root:

```bash
sbatch operation/job_main.sh 2025112500 forecast gfs
# or
bash operation/job_main.sh 2025112506 detection gfs
```

The three arguments are:

1. initialization cycle in `YYYYMMDDHH` format;
2. run mode: `forecast` or `detection`;
3. data type: currently `gfs` for the checked end-to-end path.

The wrapper performs GFS preprocessing, sliding-window inference, and map generation. Results are written to `output/postprocess/<YYYYMMDDHH>/`.

Operational scripts update several root `config.json` values in place with `sed`, including GFS input, slicing paths, checkpoint template, prediction CSV, and plotting cycle. Commit or copy the desired configuration before running if those edits must be preserved. In detection mode, `rename_file_datetime.sh` may also rename source GFS files to valid-time filenames.

### 6.9. Postprocess predictions

Set `TCG_FREQUENCY.PREDICT_CSV_FILE`, `TCG_CYCLE`, and related plotting options in `config.json`, then create maps with:

```bash
python postprocess/main.py
```

To plot monthly TCG frequency, including optional IBTrACS comparison:

```bash
python postprocess/plot_TCGfrequency_monthly.py
```

The map routines use a probability threshold of 0.5 where binary counts are required. `plot_TCGmap.py` produces per-time/per-step PDF maps; `plot_TCGmap_stat.py` produces time-aggregated maps; and `plot_TCGfrequency_monthly.py` produces yearly or multi-year EPS/PDF seasonality plots.

## 7. Outputs

Important generated products include:

| Path | Contents |
| --- | --- |
| `output/<DATASET>_extend/` | Standardized timestamped atmospheric NetCDF files |
| `output/<DATASET>_positive/` | Positive genesis-centered samples |
| `output/<DATASET>_past/` | Past Domain negative samples |
| `output/<DATASET>_dynamic/` | Dynamic Domain negative samples |
| `output/<DATASET>_slide/` | Inference windows, `data.csv`, and `dynamic_prediction.csv` |
| `output/csv/` | Master sample index and train/validation/test splits |
| `models/pre-trained/` | Pretrained model versions and checkpoints |
| `models/finetune/` | Fine-tuned model versions and checkpoints |
| `output/postprocess/<CYCLE>/` | Per-cycle TCG probability maps |
| `output/TCG_frequency_maps/` | Monthly climatology plots |

Training-index CSV files use fields such as `Path`, `FileName`, `ID`, `Year`, `Domain`, `Position`, `Step`, and `Label`. Inference CSV files use `Datetime`, `Point`, `Path`, `Step`, `Label`, and one `Score_step<N>` probability column per checkpoint.


## 8. Reproducibility notes

- This is research code, not an operational warning system. TCG probabilities should not be used for safety-critical decisions without independent validation.
- Raw atmospheric datasets, local `input/` and `output/` trees, pretrained/fine-tuned checkpoints, and large generated artifacts are excluded by `.gitignore` and are not guaranteed to be present in a fresh clone.
- The working checkout contains site-specific absolute paths and Slurm settings. Review them before use on another system.
- No dependency lockfile or automated test suite is included. Record package versions, CUDA versions, configuration, data periods, random seed, and checkpoint provenance for each experiment.
- Preprocessing and extraction stages clear configured output directories. Use experiment-specific paths to avoid overwriting earlier results.
- The current root configuration and stored run metadata reflect experiments beyond the paper’s primary MERRA-2 split. Reproduce the paper with its stated periods and labeling/enrichment choices rather than assuming every local wrapper retains publication defaults.
- No license file is currently included. Public availability alone does not define reuse or redistribution terms; contact the maintainers before reuse when licensing permission is required.

## 9. References

### Primary model paper
- Kieu, C., Nguyen, T.T., Le, D.T., Hoang, D.G.A., Luu, Q.L., Dang, B.T., Ngo, T.X., Luu, Q.T., Du, T.D. and Mai, K.V., 2025. Reconstructing Pre-Satellite Tropical Cyclogenesis Climatology Using Deep Learning. arXiv preprint arXiv:2512.17711.
- Kieu, C., and Q. Nguyen 2024: Binary dataset for machine learning applications to tropical cyclone formation prediction. Nature Scientific Data. 11, 446. https://doi.org/10.1038/s41597-024-03281-5.
- Le, D.T., Dang, T.B., Hoang Gia, A.D., Nguyen, D.H., Tien, M.H., Ngo, X.T., Luu, Q.T., Luu, Q.L., Nguyen, T.H., Nguyen, T.T. and Kieu, C., 2026. From reanalysis to climatology: deep learning reconstruction of tropical cyclogenesis in the western North Pacific. Geoscientific Model Development, 19(10), pp.4009-4030. https://doi.org/10.5194/gmd-19-4009-2026.
- Nguyen, Q, and C. Kieu, 2023: Predicting Tropical Cyclone Formation with Deep Learning. Wea. Forecasting, 39, 241-258.

### Software and data resources

- Project repository: [https://github.com/kieucq/tcg-net](https://github.com/kieucq/tcg-net)
- TCG-Net software archive cited by the paper: [https://doi.org/10.5281/zenodo.16741501](https://doi.org/10.5281/zenodo.16741501)
- Data-processing archive cited by the paper: [https://doi.org/10.5281/zenodo.15640334](https://doi.org/10.5281/zenodo.15640334)
- NASA MERRA-2: [https://disc.gsfc.nasa.gov/datasets?project=MERRA-2](https://disc.gsfc.nasa.gov/datasets?project=MERRA-2)
- NOAA International Best Track Archive for Climate Stewardship (IBTrACS): [https://www.ncei.noaa.gov/products/international-best-track-archive](https://www.ncei.noaa.gov/products/international-best-track-archive)

## 10. Contact

For scientific questions about TCG-Net and the associated study, please contact Chanh Kieu, Indiana University Bloomington, email: ckieu@iu.edu.

For reproducible bug reports or code questions, open an issue in the [GitHub repository](https://github.com/kieucq/tcg-net/issues) and include the dataset, configuration, command, environment/module versions, and complete error message.
