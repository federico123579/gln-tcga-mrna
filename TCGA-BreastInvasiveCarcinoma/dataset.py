"""
TCGA Breast Invasive Carcinoma dataset loading.

Loads mRNA expression data for Tumor vs Normal classification.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


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


def load_tcga_tumor_vs_normal(
    data_dir: str | Path | None = None,
) -> Dataset:
    """Load TCGA Breast Cancer data for Tumor vs Normal classification.

    Args:
        data_dir: Path to the brca_tcga_pan_can_atlas_2018 directory.
                  Defaults to the directory relative to this script.

    Returns:
        Dataset with mRNA expression features and binary labels.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "brca_tcga_pan_can_atlas_2018"
    else:
        data_dir = Path(data_dir)

    # --- Load tumor samples ---
    tumor_file = data_dir / "data_mrna_seq_v2_rsem.txt"
    tumor_df = pd.read_csv(tumor_file, sep="\t", index_col=0)

    # Drop Entrez_Gene_Id column
    tumor_df = tumor_df.drop("Entrez_Gene_Id", axis=1)

    # Store gene names before transposing
    gene_names = tumor_df.index.tolist()

    # Transpose: samples as rows, genes as columns
    tumor_df = tumor_df.T

    # --- Load normal samples ---
    normal_file = data_dir / "normals" / "data_mrna_seq_v2_rsem_normal_samples.txt"
    normal_df = pd.read_csv(normal_file, sep="\t", index_col=0)

    # Drop Entrez_Gene_Id column
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
    y = np.concatenate([
        np.zeros(len(X_normal)),  # Normal = 0
        np.ones(len(X_tumor)),    # Tumor = 1
    ])

    # --- Preprocessing: log2(x + 1) transform ---
    # Stabilizes variance for RSEM values
    X = np.log2(X + 1)

    # Handle any NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Convert to tensors ---
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return Dataset(X=X_tensor, y=y_tensor, gene_names=gene_names)


if __name__ == "__main__":
    # Quick test
    dataset = load_tcga_tumor_vs_normal()
    print(dataset)
    print(f"X shape: {dataset.X.shape}")
    print(f"y shape: {dataset.y.shape}")
    print(f"Class distribution: Normal={int((dataset.y == 0).sum())}, Tumor={int((dataset.y == 1).sum())}")
