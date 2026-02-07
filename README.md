# TCGA Breast Cancer GLN Experiment

Classifies Tumor vs Normal tissue using mRNA expression profiles from TCGA Breast Invasive Carcinoma data.

## Data Source

Data is automatically downloaded from [cBioPortal DataHub](https://github.com/cBioPortal/datahub) - specifically the [Breast Invasive Carcinoma (TCGA, PanCancer Atlas)](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018) study.

## Requirements

- Python 3.11+
- [Git LFS](https://git-lfs.github.com/) (for downloading data from cBioPortal)

### Install Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Then initialize
git lfs install
```

## Usage

### Project Layout

- src/gln_tcga/ - library modules (dataset, training, analysis, experiments)
- src/gln_tcga/cli/ - CLI entry points (gln-train, gln-analyze, gln-experiment)
- experiments/ - experiment outputs and snapshots

This repository is a uv workspace with a src-based package and CLI entry points.
The GLN library is provided by the workspace member in gated-linear-networks.

### Install

```bash
uv sync
```

### Option 1: Full Pipeline

```bash
# Run training + analysis in one command
uv run gln-experiment --experiment exp01

# Pass training options through to the training step
uv run gln-experiment --experiment exp01 --train-args -- --epochs 20 --lr 0.005
```

### Option 2: Separate Steps (recommended)

```bash
# Step 1: Train models with different seeds
uv run gln-train --experiment exp01

# Step 2: Run biomarker analysis on best model
uv run gln-analyze --experiment exp01
```

### Command Line Options

```bash
# Train with custom parameters
uv run gln-train --experiment exp01 --seeds 42 43 --epochs 20 --lr 0.005

# List available experiments
uv run gln-analyze --list-experiments

# List available trained models for an experiment
uv run gln-analyze --experiment exp01 --list

# Analyze specific model
uv run gln-analyze --experiment exp01 --seed 42

# Analyze all models
uv run gln-analyze --experiment exp01 --all
```

On first run, the data will be automatically downloaded from cBioPortal DataHub (~150MB for required files).

## Output

Experiments are stored under experiments/NAME with a latest alias and snapshots:

- experiments/NAME/latest/models/ - Saved model checkpoints (gln_seed*.pt)
- experiments/NAME/latest/results/ - Biomarker analysis plots and gene importance CSV
- experiments/NAME/latest/results.db - SQLite database with experiment results
- experiments/NAME/runs/TIMESTAMP/ - Snapshot of each training run

## Query Results

```python
from gln_tcga.results import get_results_df, init_database

# Load all results as DataFrame
conn = init_database("experiments/exp01/latest/results.db")
df = get_results_df(conn)
print(df)
conn.close()
```

## Load Saved Model

```python
from gln_tcga.train import load_model

model, transformer, config = load_model("experiments/exp01/latest/models/gln_seed42.pt")
```

## Biomarker Analysis

The analysis uses **Integrated Gradients** to compute gene importance - measuring how much each gene contributes to the model's tumor vs normal predictions. This is more meaningful than extracting weights, as it captures the actual learned decision-making process.

## Reproducing the Report (TCGA-BRCA)

All commands below are run from the repository root after `uv sync`.

### 1) Local OGD run + attributions (used for IG/saliency figures)

```bash
# Paper-faithful local OGD (single pass, batch size 1)
uv run gln-train \
 --experiment report-ogd \
 --online-ogd \
 --epochs 1 \
 --batch-size 1 \
 --lr 0.01 \
 --lr-schedule sqrt \
 --layers 20 40 20 \
 --context-dim 4 \
 --seeds 4

# Integrated Gradients (Captum) attribution report
uv run gln-analyze \
 --experiment report-ogd \
 --method ig \
 --use-captum \
 --n-steps 50 \
 --batch-size 10 \
 --baseline mean

# Saliency attribution report
uv run gln-analyze \
 --experiment report-ogd \
 --method saliency \
 --saliency-samples 3
```

Outputs are written under `experiments/report-ogd/latest/results/`.

### 2) Training-curve panels (local OGD vs backprop)

Use a separate experiment name per configuration so each run produces its own `results/training_curves.png`.

```bash
# Local OGD, decay (sqrt) schedule
uv run gln-train --experiment curves-local-decay --online-ogd --epochs 1 --batch-size 1 --lr 0.01 --lr-schedule sqrt --layers 20 40 20 --context-dim 4 --seeds 4
uv run gln-curves --experiment curves-local-decay --retrain --online-ogd

# Local OGD, constant schedule
uv run gln-train --experiment curves-local-const --online-ogd --epochs 1 --batch-size 1 --lr 0.01 --lr-schedule constant --layers 20 40 20 --context-dim 4 --seeds 4
uv run gln-curves --experiment curves-local-const --retrain --online-ogd

# Backprop, batch size 1, linear decay
uv run gln-train --experiment curves-bp-decay --epochs 1 --batch-size 1 --lr 0.01 --lr-schedule linear --layers 20 40 20 --context-dim 4 --seeds 4
uv run gln-curves --experiment curves-bp-decay --retrain

# Backprop, batch size 1, constant
uv run gln-train --experiment curves-bp-const --epochs 1 --batch-size 1 --lr 0.01 --lr-schedule constant --layers 20 40 20 --context-dim 4 --seeds 4
uv run gln-curves --experiment curves-bp-const --retrain

# Backprop, batch size 10, 10 epochs (smooth curves)
uv run gln-train --experiment curves-bp-10 --epochs 10 --batch-size 10 --lr 0.01 --lr-schedule linear --layers 20 40 20 --context-dim 4 --seeds 4
uv run gln-curves --experiment curves-bp-10 --retrain
```

### 3) Baseline comparisons and cross-validation boxplots

```bash
# Local OGD GLN vs baselines (CV)
uv run gln-baselines --experiment report-ogd --cv-folds 5 --n-repeats 3 --compare --online-ogd

# Backprop GLN vs baselines (CV)
uv run gln-baselines --experiment report-ogd --cv-folds 5 --n-repeats 3 --compare
```

The comparison plots are saved under `experiments/<experiment>/latest/results/`.
