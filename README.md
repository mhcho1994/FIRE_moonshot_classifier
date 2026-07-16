# FIRE Moonshot Classifier

FIRE Moonshot Classifier identifies a drone's flight stack/autopilot from
kinematic features extracted from its flight trajectory. The current pipeline
primarily distinguishes **PX4** from **ArduPilot** and supports Sim-to-Real
experiments in which models trained on SITL logs are evaluated on real-flight
data.

## Overview

The processing pipeline performs the following steps:

1. Read PX4 ULog (`.ulg`), ArduPilot DataFlash (`.bin`), or real-flight CSV data.
2. Derive kinematic features such as acceleration, jerk, and curvature from
   position and velocity.
3. Use HMM-based flight segmentation to extract left- and right-turn segments.
4. Select `XY-Accel`, `XY-Jerk`, and `Curvature` as classifier inputs.
5. Use DWT statistics for SVM training and time-series features for 1D-CNN-based
   models.

The repository provides the following models and experiments:

- **SVM**: a baseline classifier that uses DWT statistical features
- **1D-CNN / CNN-LSTM**: neural classifiers that use padded time-series features
- **DIVERSIFY-based 1D-CNN**: a Sim-to-Real classifier that discovers latent
  domains and applies domain confusion and an out-of-distribution (OOD) gate

## Project Structure

```text
.
├── data/                              # Raw SITL and real-flight data
├── docker/
│   ├── Dockerfile                     # Multi-stage dev/release image
│   └── compose.yaml                   # Currently unused
├── scripts/
│   ├── setup_local.sh                 # Local virtual environment helper
│   └── setup_docker.sh                # Docker build/run management
├── src/fire_moonshot_classifier/
│   ├── datamanager/                   # Log parsing and dataset assembly
│   ├── processor/                     # Kinematics and flight segmentation
│   ├── training/                      # DIVERSIFY training
│   ├── evaluation/                    # Checkpoint evaluation
│   ├── postprocessor/                 # Result visualization
│   ├── cache/                         # Preprocessed .npz feature caches
│   └── model/                         # DIVERSIFY checkpoints
├── tools/
│   ├── preprocessing/build_features.py
│   └── training/                      # SVM, CNN, and CNN-LSTM training
└── results/                           # Statistics and generated figures
```

## Requirements

### General

- Linux is recommended
- Python `3.12`
- Git

The supported Python version declared in `pyproject.toml` is `>=3.12,<3.13`.

### GPU Support

GPU execution requires:

- An NVIDIA GPU
- A compatible NVIDIA driver
- NVIDIA Container Toolkit when using Docker

The Docker image uses CUDA 13.2 and a CUDA-enabled PyTorch wheel by default. Use
the `USE_GPU=0` option described below if no compatible GPU is available.

## Local Environment Setup

### Recommended Setup

Run the following commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Verify the installation:

```bash
python -c "import fire_moonshot_classifier; print('installation OK')"
```

To use a CPU-only PyTorch build, install the CPU wheel before installing the
project:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch
python -m pip install -e .
```

### Setup Script

The repository also provides a helper for creating a CPU-based development
environment:

```bash
./scripts/setup_local.sh
source .venv/bin/activate
python -m pip install -e .
```

Recreate the virtual environment:

```bash
./scripts/setup_local.sh --recreate
```

Skip PyTorch installation when it is already installed:

```bash
./scripts/setup_local.sh --skip-torch
```

> `setup_local.sh` installs the main scientific-computing dependencies. Running
> `pip install -e .` afterward is recommended to install the project itself and
> synchronize all dependencies declared in `pyproject.toml`, including W&B.

## Data Layout

Default dataset directory names are configured in
`src/fire_moonshot_classifier/datamanager/config.py`.

```text
data/
├── 260615_sitl_logs/
│   └── run_XXX/
│       ├── px4_logs/
│       │   └── raw/        # PX4 .ulg files
│       └── ardu_logs/
│           └── raw/logs/   # ArduPilot .bin files
├── 260527_flight_logs_1/
│   └── run_XXX/
│       ├── px4_logs/processed/    # Real-flight .csv files
│       └── ardu_logs/processed/   # Real-flight .csv files
└── 260527_flight_logs_2/
    └── run_XXX/
        └── ...
```

The SITL pipeline reads PX4 `.ulg` and ArduPilot `.bin` files. The real-flight
pipeline reads motion-capture coordinates from processed CSV files by default.
The following CSV column names are supported:

- Time: `time_s` or `timestamp`
- Motion-capture position: `gtx`, `gty`, `gtz` or `gt_x`, `gt_y`, `gt_z`
- Vision position: `xsmooth`, `ysmooth`, `zsmooth` or
  `x_smooth`, `y_smooth`, `z_smooth`

Edit `src/fire_moonshot_classifier/datamanager/config.py` to change dataset
names or the selected feature set.

## Preprocessing and Training

Run all commands from the repository root.

### 1. Build Feature Caches

```bash
python tools/preprocessing/build_features.py
```

Limit the number of processed runs for a quick smoke test:

```bash
python tools/preprocessing/build_features.py --max-runs 2
```

Generated caches are stored at:

```text
src/fire_moonshot_classifier/cache/<dataset>_features.npz
```

Each cache contains DWT features in `X`, time-series features in `X_seq`, class
labels in `y`, and source run information in `runs`.

### 2. Train Baseline Models

SVM:

```bash
python tools/training/train_svm.py
```

1D-CNN:

```bash
python tools/training/train_cnn.py
```

CNN-LSTM:

```bash
python tools/training/train_cnn_lstm.py
```

Add `--no-real` to any of these commands to skip real-flight evaluation.

### 3. Train the DIVERSIFY Model

```bash
python -m fire_moonshot_classifier.training.train_diversify
```

Example with explicit hyperparameters:

```bash
python -m fire_moonshot_classifier.training.train_diversify \
  --epochs 100 \
  --local-epochs 3 \
  --batch-size 128 \
  --lr 0.001
```

View all available options:

```bash
python -m fire_moonshot_classifier.training.train_diversify --help
```

### 4. Evaluate a DIVERSIFY Checkpoint

Evaluate a saved checkpoint on real-flight data:

```bash
python -m fire_moonshot_classifier.evaluation.eval_diversify \
  src/fire_moonshot_classifier/model/<checkpoint>.pt \
  --no-wandb
```

If the checkpoint path is omitted, the evaluator displays the checkpoints under
`src/fire_moonshot_classifier/model/` and prompts for a selection:

```bash
python -m fire_moonshot_classifier.evaluation.eval_diversify
```

Use `--wandb` to log evaluation results to Weights & Biases:

```bash
wandb login
python -m fire_moonshot_classifier.evaluation.eval_diversify \
  src/fire_moonshot_classifier/model/<checkpoint>.pt \
  --wandb
```

## Running with Docker

The `scripts/setup_docker.sh` script manages development and release images and
their container lifecycle. The `docker/compose.yaml` file is currently empty,
so use this script instead of Docker Compose.

Ensure that the script is executable:

```bash
chmod +x scripts/setup_docker.sh
```

### GPU Development Environment

```bash
./scripts/setup_docker.sh dev-build
./scripts/setup_docker.sh dev-run
./scripts/setup_docker.sh dev-shell
```

- Image name: `fire_moonshot_classifier:dev`
- Container name: `fire_moonshot_classifier_dev`
- The host repository is bind-mounted into the container.
- `dev-run` installs the project in editable mode.
- The host UID and GID are used so that files created in the container retain
  the correct host ownership.

Preprocessing and training commands can be run directly inside the container:

```bash
python tools/preprocessing/build_features.py --max-runs 2
python tools/training/train_svm.py
```

### CPU-Only Development Environment

```bash
USE_GPU=0 ./scripts/setup_docker.sh dev-build
USE_GPU=0 ./scripts/setup_docker.sh dev-run
./scripts/setup_docker.sh dev-shell
```

Using `USE_GPU=0` for both `dev-build` and `dev-run` is recommended. The first
build can take some time because it downloads the CUDA base image and Python
dependencies. The base image remains CUDA-based even when the CPU-only PyTorch
wheel is selected.

### Development Container Management

```bash
./scripts/setup_docker.sh dev-stop     # Stop the container
./scripts/setup_docker.sh dev-run      # Recreate and start the container
./scripts/setup_docker.sh dev-remove   # Remove the container
./scripts/setup_docker.sh dev-clean    # Remove the container and image
```

### Release Image

Build and run a release image containing both the project source and its
dependencies:

```bash
./scripts/setup_docker.sh release-build
./scripts/setup_docker.sh release-run
./scripts/setup_docker.sh release-shell
```

CPU-only build and execution:

```bash
USE_GPU=0 ./scripts/setup_docker.sh release-build
USE_GPU=0 ./scripts/setup_docker.sh release-run
./scripts/setup_docker.sh release-shell
```

The same Python module and tool commands can be run inside the release
container:

```bash
python -m fire_moonshot_classifier.evaluation.eval_diversify --no-wandb
```

> The `main.py` module connected to the `fire-classifier` console entry point
> does not currently contain executable application logic. Use
> `release-run`/`release-shell` and invoke the appropriate Python module
> directly instead of running the release image's default entry point.

### Docker Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `USE_GPU` | `1` | Set to `0` to disable GPU access and use CPU-only PyTorch |
| `DOCKER_GPUS` | `all` | Value passed to `docker run --gpus` |
| `TORCH_INDEX_URL` | Auto-selected | Overrides the PyTorch wheel index |
| `USER_UID`, `USER_GID` | Host UID/GID | User IDs for the dev container |
| `MATCH_HOST_ID` | `0` | Use the host IDs for the release image when set to `1` |
| `RELEASE_UID`, `RELEASE_GID` | `1000` | User IDs for the release image |
| `NO_CACHE` | `0` | Disable the Docker build cache when set to `1` |

To expose a specific GPU:

```bash
DOCKER_GPUS='"device=0"' ./scripts/setup_docker.sh dev-run
```

## Outputs

- Preprocessed caches: `src/fire_moonshot_classifier/cache/`
- Model checkpoints: `src/fire_moonshot_classifier/model/` or `models/`
- Evaluation statistics and figures: `results/`
- W&B offline/run data: `wandb/`

## Troubleshooting

### `Feature cache not found`

Build the feature cache before training:

```bash
python tools/preprocessing/build_features.py
```

### Docker: `could not select device driver`

NVIDIA Container Toolkit is unavailable or the host cannot expose a compatible
GPU. Rebuild and run the container with `USE_GPU=0`:

```bash
USE_GPU=0 ./scripts/setup_docker.sh dev-build
USE_GPU=0 ./scripts/setup_docker.sh dev-run
```

### Bind-Mount Permission Errors

The development image uses the host UID and GID by default. Specify them
explicitly before building if needed:

```bash
USER_UID="$(id -u)" USER_GID="$(id -g)" \
  ./scripts/setup_docker.sh dev-build
```

### Running Without W&B

Add `--no-wandb` to the evaluation command. To run training without network
access, enable W&B offline mode:

```bash
WANDB_MODE=offline \
  python -m fire_moonshot_classifier.training.train_diversify
```
