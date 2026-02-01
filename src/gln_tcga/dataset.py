"""
TCGA Breast Invasive Carcinoma dataset loading.

Loads mRNA expression data for Tumor vs Normal classification.
Downloads data from cBioPortal DataHub with caching.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Memory
from torch.utils.data import TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[2]

# Cache directory for downloaded data
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)

# cBioPortal DataHub GitHub repository (uses Git LFS for large files)
DATAHUB_REPO = "https://github.com/cBioPortal/datahub.git"
STUDY_NAME = "brca_tcga_pan_can_atlas_2018"


@dataclass
class Dataset:
    """Dataset container for TCGA mRNA expression data."""

    X: torch.Tensor  # (n_samples, n_genes)
    y: torch.Tensor  # (n_samples,) binary labels
    gene_names: list[str]  # Gene symbols for biomarker analysis

    def __repr__(self):
        return (
            f"Dataset(n_samples={self.X.shape[0]}, "
            f"input_size={self.input_size}, "
            f"n_classes=2)"
        )

    @property
    def input_size(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    def train_test_split(
        self,
        *,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_seed: int = 42,
    ) -> tuple[TensorDataset, TensorDataset]:
        """Split dataset into train and test sets.

        Args:
            test_size: Fraction of data to use for testing.
            shuffle: Whether to shuffle before splitting.
            random_seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset) as TensorDatasets.
        """
        n_samples = self.X.shape[0]
        indices = torch.arange(n_samples)

        if shuffle:
            generator = torch.Generator().manual_seed(random_seed)
            indices = indices[torch.randperm(n_samples, generator=generator)]

        split_idx = int(n_samples * (1 - test_size))

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_ds = TensorDataset(self.X[train_indices], self.y[train_indices])
        test_ds = TensorDataset(self.X[test_indices], self.y[test_indices])

        return train_ds, test_ds


@memory.cache
def _download_tcga_brca_study() -> Path:
    """Download TCGA BRCA study files directly via HTTP.

    Downloads only the required mRNA expression files (~150MB total) directly
    from GitHub's raw file URLs. This bypasses Git entirely, avoiding the
    overhead of cloning the massive cBioPortal DataHub repository structure.

    GitHub automatically redirects LFS file URLs to their storage location,
    making this approach as fast as your internet connection allows.

    Results are cached using joblib.Memory.

    Returns:
        Path to the extracted data directory.
    """
    import requests

    data_dir = CACHE_DIR / STUDY_NAME

    if data_dir.exists():
        return data_dir

    print("Downloading TCGA BRCA data from cBioPortal DataHub...")
    print("  Using direct HTTP download (bypassing Git for speed)")

    # Base URL for raw files from GitHub (LFS files are auto-redirected)
    base_url = "https://github.com/cBioPortal/datahub/raw/master"

    # Files to download (relative paths within the repo)
    files_to_download = [
        f"public/{STUDY_NAME}/data_mrna_seq_v2_rsem.txt",
        f"public/{STUDY_NAME}/normals/data_mrna_seq_v2_rsem_normal_samples.txt",
    ]

    # Create target directory structure
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "normals").mkdir(exist_ok=True)

    for file_path in files_to_download:
        url = f"{base_url}/{file_path}"

        # Map repo path to local path (strip public/STUDY_NAME prefix)
        relative_path = file_path.replace(f"public/{STUDY_NAME}/", "")
        local_path = data_dir / relative_path

        print(f"  Fetching: {relative_path}")

        try:
            # Stream download for large LFS files
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()

                # Get file size for progress info
                total_size = int(r.headers.get("content-length", 0))
                if total_size > 0:
                    print(f"    Size: {total_size / 1024 / 1024:.1f} MB")

                # Write to disk in chunks
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print("    ✓ Success")

        except requests.exceptions.RequestException as e:
            # Clean up partial download
            if data_dir.exists():
                import shutil

                shutil.rmtree(data_dir)
            raise RuntimeError(f"Failed to download {file_path}: {e}") from e

    print(f"  Downloaded to: {data_dir}")
    return data_dir


def download_tcga_brca_study() -> Path:
    """Public wrapper for downloading TCGA BRCA study.

    Returns:
        Path to the extracted data directory.
    """
    return _download_tcga_brca_study()


def load_tcga_tumor_vs_normal(
    data_dir: str | Path | None = None,
    auto_download: bool = True,
) -> Dataset:
    """Load TCGA Breast Cancer data for Tumor vs Normal classification.

    Args:
        data_dir: Path to the brca_tcga_pan_can_atlas_2018 directory.
                  If None and auto_download=True, downloads from cBioPortal.
                  If None and auto_download=False, uses local directory.
        auto_download: Whether to automatically download data if not found.

    Returns:
        Dataset with mRNA expression features and binary labels.
    """
    if data_dir is None:
        # Check for local directory first
        local_dir = ROOT_DIR / "brca_tcga_pan_can_atlas_2018"
        if local_dir.exists():
            data_dir = local_dir
        elif auto_download:
            data_dir = download_tcga_brca_study()
        else:
            data_dir = local_dir  # Will fail with helpful error
    else:
        data_dir = Path(data_dir)

    # --- Load tumor samples ---
    tumor_file = data_dir / "data_mrna_seq_v2_rsem.txt"

    # Check if file is an LFS pointer (corrupted download)
    with open(tumor_file) as f:
        first_line = f.readline()
    if first_line.startswith("version https://git-lfs"):
        raise RuntimeError(
            f"Data file {tumor_file} is a Git LFS pointer, not actual data.\n"
            "The LFS files were not properly downloaded. Please:\n"
            "  1. Delete the cache directory: rm -rf .cache\n"
            "  2. Ensure Git LFS is installed: brew install git-lfs && git lfs install\n"
            "  3. Re-run the experiment"
        )

    tumor_df = pd.read_csv(tumor_file, sep="\t", index_col=0)

    # Drop Entrez_Gene_Id column if present
    if "Entrez_Gene_Id" in tumor_df.columns:
        tumor_df = tumor_df.drop("Entrez_Gene_Id", axis=1)

    # Store gene names before transposing
    gene_names = tumor_df.index.tolist()

    # Transpose: samples as rows, genes as columns
    tumor_df = tumor_df.T

    # --- Load normal samples ---
    normal_file = data_dir / "normals" / "data_mrna_seq_v2_rsem_normal_samples.txt"
    normal_df = pd.read_csv(normal_file, sep="\t", index_col=0)

    # Drop Entrez_Gene_Id column if present
    if "Entrez_Gene_Id" in normal_df.columns:
        normal_df = normal_df.drop("Entrez_Gene_Id", axis=1)

    # Transpose: samples as rows, genes as columns
    normal_df = normal_df.T

    # --- Align genes between tumor and normal ---
    # Filter out NaN gene symbols (genes without Hugo symbols)
    tumor_df = tumor_df.loc[:, tumor_df.columns.notna()]
    normal_df = normal_df.loc[:, normal_df.columns.notna()]

    # Remove duplicate gene columns (keep first occurrence)
    tumor_df = tumor_df.loc[:, ~tumor_df.columns.duplicated()]
    normal_df = normal_df.loc[:, ~normal_df.columns.duplicated()]

    # Get common genes (some genes may only be in one file)
    common_genes = tumor_df.columns.intersection(normal_df.columns).tolist()

    # Filter both dataframes to common genes in the same order
    tumor_df = tumor_df[common_genes]
    normal_df = normal_df[common_genes]
    gene_names = common_genes

    # --- Combine and create labels ---
    # Normal = 0, Tumor = 1
    X_tumor = tumor_df.values.astype(np.float64)
    X_normal = normal_df.values.astype(np.float64)

    X = np.vstack([X_normal, X_tumor])
    y = np.concatenate(
        [
            np.zeros(len(X_normal), dtype=np.int64),
            np.ones(len(X_tumor), dtype=np.int64),
        ]
    )

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)

    return Dataset(X=X, y=y, gene_names=gene_names)
