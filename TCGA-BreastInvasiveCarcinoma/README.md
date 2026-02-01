# TCGA Breast Cancer GLN Experiment

Classifies Tumor vs Normal tissue using mRNA expression profiles from TCGA Breast Invasive Carcinoma data.

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

## Run Experiment

```bash
cd TCGA-BreastInvasiveCarcinoma
uv run python run_experiment.py
```

On first run, the data will be automatically downloaded from cBioPortal DataHub (~150MB).

## Output

After running, you'll find:

- `models/` - Saved model checkpoints (`gln_seed*.pt`)
- `results/` - Biomarker analysis plots and gene importance CSV
- `results.db` - SQLite database with experiment results

## Query Results

```python
from results import query_results, get_summary_stats

# Load all results as DataFrame
df = query_results("results.db")
print(df)

# Get summary statistics
stats = get_summary_stats("results.db")
print(f"Mean accuracy: {stats['mean_accuracy']:.2%}")
```

## Load Saved Model

```python
from train import load_model

model, transformer, config = load_model("models/gln_seed42.pt")
```
