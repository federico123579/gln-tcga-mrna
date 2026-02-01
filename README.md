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
