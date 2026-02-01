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

### Option 1: Full Pipeline

```bash
# Run training + analysis in one command
uv run python run_experiment.py
```

### Option 2: Separate Steps (recommended)

```bash
# Step 1: Train models with different seeds
uv run python train_models.py

# Step 2: Run biomarker analysis on best model
uv run python run_analysis.py
```

### Command Line Options

```bash
# Train with custom parameters
uv run python train_models.py --seeds 42 43 --epochs 20 --lr 0.005

# List available trained models
uv run python run_analysis.py --list

# Analyze specific model
uv run python run_analysis.py --seed 42

# Analyze all models
uv run python run_analysis.py --all
```

On first run, the data will be automatically downloaded from cBioPortal DataHub (~150MB for required files).

## Output

After running, you'll find:

- `models/` - Saved model checkpoints (`gln_seed*.pt`)
- `results/` - Biomarker analysis plots and gene importance CSV
- `results.db` - SQLite database with experiment results

## Query Results

```python
from results import get_results_df, init_database

# Load all results as DataFrame
conn = init_database("results.db")
df = get_results_df(conn)
print(df)
conn.close()
```

## Load Saved Model

```python
from train import load_model

model, transformer, config = load_model("models/gln_seed42.pt")
```

## Biomarker Analysis

The analysis uses **Integrated Gradients** to compute gene importance - measuring how much each gene contributes to the model's tumor vs normal predictions. This is more meaningful than extracting weights, as it captures the actual learned decision-making process.
