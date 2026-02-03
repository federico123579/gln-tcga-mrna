"""
Captum-based attribution methods for GLN models.

Provides optimized Integrated Gradients and Permutation Importance
using Captum library, with correlation analysis between methods.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import gln
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import FeaturePermutation, IntegratedGradients
from scipy import stats
from tqdm import tqdm

if TYPE_CHECKING:
    pass


class GLNWithTransform(nn.Module):
    """Wrapper combining InputTransformer + GLN for Captum compatibility.

    Captum requires a single nn.Module that takes raw inputs and produces outputs.
    This wrapper combines the sigmoid transformation (InputTransformer) with the
    GLN model, allowing Captum to properly compute attributions through the full
    input space rather than the pre-transformed space.
    """

    def __init__(self, model: gln.GLN, transformer: gln.InputTransformer):
        """Initialize wrapper.

        Args:
            model: Trained GLN model.
            transformer: Fitted InputTransformer from training.
        """
        super().__init__()
        self.model = model
        self.transformer = transformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer and model.

        Args:
            x: Raw input tensor (before transformation).

        Returns:
            Model predictions.
        """
        x_transformed = self.transformer.transform(x)
        return self.model(x_transformed)


def compute_integrated_gradients(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    gene_names: list[str],
    *,
    baseline_type: str = "mean",
    n_steps: int = 50,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Compute Integrated Gradients attribution using Captum.

    Uses Captum's optimized IntegratedGradients implementation for computing
    feature attributions. The GLNWithTransform wrapper ensures gradients flow
    through the full input transformation.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        gene_names: List of gene names corresponding to input features.
        baseline_type: Type of baseline to use:
            - "mean": Dataset mean (transforms to ~0.5 for all genes) - RECOMMENDED
            - "zero": Zero baseline in raw space
        n_steps: Number of steps for Riemann approximation.
        batch_size: Batch size for processing samples.

    Returns:
        Tuple of:
            - DataFrame with gene names, importance scores, and direction
            - Raw attribution tensor (n_samples, n_genes)
    """
    # Wrap model for Captum
    wrapped_model = GLNWithTransform(model, transformer)
    wrapped_model.eval()

    # Initialize Captum's IntegratedGradients
    ig = IntegratedGradients(wrapped_model)

    # Choose baseline based on type
    if baseline_type == "mean":
        # Use dataset mean as baseline - transforms to ~0.5 for all genes
        baseline = transformer.means.unsqueeze(0).expand(X.shape[0], -1)
    else:
        # Zero baseline in raw space
        baseline = torch.zeros_like(X)

    # Process in batches
    all_attributions = []
    n_samples = X.shape[0]

    batch_iter = range(0, n_samples, batch_size)
    batch_iter = tqdm(batch_iter, desc="Computing IG attributions", unit="batch")

    for i in batch_iter:
        batch_X = X[i : i + batch_size]
        batch_baseline = baseline[i : i + batch_size]

        # Compute attributions for this batch
        attributions = ig.attribute(
            batch_X,
            baselines=batch_baseline,
            n_steps=n_steps,
            internal_batch_size=min(batch_size, len(batch_X)),
        )
        all_attributions.append(attributions.detach())

    # Concatenate all batches
    attributions = torch.cat(all_attributions, dim=0)

    # Aggregate across samples: mean absolute attribution
    mean_abs_attr = attributions.abs().mean(dim=0).numpy()

    # Compute signed mean to see direction of effect
    mean_signed_attr = attributions.mean(dim=0).numpy()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names[: len(mean_abs_attr)],
            "importance": mean_abs_attr,
            "direction": np.sign(mean_signed_attr),  # +1 = tumor, -1 = normal
            "signed_importance": mean_signed_attr,
        }
    )

    # Sort by absolute importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df, attributions


def compute_permutation_importance(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    gene_names: list[str],
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Compute Permutation Importance using Captum.

    Model-agnostic feature importance that measures the decrease in model
    performance when a feature's values are randomly shuffled.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        y: Target labels.
        gene_names: List of gene names corresponding to input features.
        batch_size: Not used directly but kept for API consistency.

    Returns:
        DataFrame with gene names and importance scores.
    """
    # Wrap model for Captum
    wrapped_model = GLNWithTransform(model, transformer)
    wrapped_model.eval()

    # Initialize Captum's FeaturePermutation
    perm = FeaturePermutation(wrapped_model)

    print("Computing permutation importance...")
    print(f"  n_features: {X.shape[1]}")

    # Compute permutation importance
    # FeaturePermutation measures the change in output when permuting each feature
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributions = perm.attribute(
            X,
            perturbations_per_eval=2,
            n_perturb_samples=10,
            show_progress=True,
        )

    # Aggregate: mean absolute change per feature
    mean_importance = attributions.abs().mean(dim=0).numpy()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names[: len(mean_importance)],
            "importance": mean_importance,
        }
    )

    # Sort by importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def compute_rank_correlation(
    ig_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute Spearman rank correlation between IG and Permutation rankings.

    Args:
        ig_df: DataFrame from compute_integrated_gradients with 'gene' and 'rank'.
        perm_df: DataFrame from compute_permutation_importance with 'gene' and 'rank'.

    Returns:
        Dictionary with keys:
            - spearman_r: Spearman correlation coefficient
            - p_value: Two-sided p-value
            - n_genes: Number of genes compared
    """
    # Merge on gene names to align rankings
    merged = ig_df[["gene", "rank"]].merge(
        perm_df[["gene", "rank"]],
        on="gene",
        suffixes=("_ig", "_perm"),
    )

    if merged.empty:
        return {
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "n_genes": 0,
        }

    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(merged["rank_ig"], merged["rank_perm"])

    return {
        "spearman_r": float(correlation),
        "p_value": float(p_value),
        "n_genes": len(merged),
    }


def generate_attributions_report(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    gene_names: list[str],
    output_dir,
    *,
    method: str = "both",
    compute_correlation: bool = True,
    n_steps: int = 50,
    batch_size: int = 64,
    baseline_type: str = "mean",
) -> dict:
    """Generate comprehensive attribution report.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        y: Target labels.
        gene_names: List of gene names corresponding to input features.
        output_dir: Directory to save outputs.
        method: Attribution method(s) to use: "ig", "permutation", or "both".
        compute_correlation: Whether to compute rank correlation (requires both methods).
        n_steps: Number of steps for IG.
        batch_size: Batch size for processing.
        baseline_type: Baseline type for IG.

    Returns:
        Dictionary containing results (DataFrames, correlation stats, etc.)
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Compute Integrated Gradients
    if method in ("ig", "both"):
        print("\n" + "-" * 40)
        print("Computing Integrated Gradients (Captum)...")
        print("-" * 40)

        ig_df, _ = compute_integrated_gradients(
            model,
            transformer,
            X,
            gene_names,
            baseline_type=baseline_type,
            n_steps=n_steps,
            batch_size=batch_size,
        )

        # Save to CSV
        ig_path = output_dir / "ig_importance.csv"
        ig_df.to_csv(ig_path, index=False)
        print(f"Saved IG importance to: {ig_path}")

        results["ig_df"] = ig_df

        # Print top genes
        print("\nTop 10 genes by IG:")
        for _, row in ig_df.head(10).iterrows():
            direction = "+" if row["direction"] > 0 else "-"
            print(
                f"  {row['rank']:3d}. {row['gene']:12s} ({direction}) importance={row['importance']:.4f}"
            )

    # Compute Permutation Importance
    if method in ("permutation", "both"):
        print("\n" + "-" * 40)
        print("Computing Permutation Importance (Captum)...")
        print("-" * 40)

        perm_df = compute_permutation_importance(
            model,
            transformer,
            X,
            y,
            gene_names,
            batch_size=batch_size,
        )

        # Save to CSV
        perm_path = output_dir / "permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        print(f"Saved permutation importance to: {perm_path}")

        results["perm_df"] = perm_df

        # Print top genes
        print("\nTop 10 genes by Permutation Importance:")
        for _, row in perm_df.head(10).iterrows():
            print(
                f"  {row['rank']:3d}. {row['gene']:12s} importance={row['importance']:.4f}"
            )

    # Compute correlation between methods
    if compute_correlation and method == "both":
        print("\n" + "-" * 40)
        print("Computing Rank Correlation...")
        print("-" * 40)

        correlation = compute_rank_correlation(results["ig_df"], results["perm_df"])

        # Save correlation results
        corr_path = output_dir / "rank_correlation.json"
        with open(corr_path, "w") as f:
            json.dump(correlation, f, indent=2)
        print(f"Saved correlation results to: {corr_path}")

        results["correlation"] = correlation

        print(f"\nSpearman correlation: r={correlation['spearman_r']:.4f}")
        print(f"P-value: {correlation['p_value']:.2e}")
        print(f"N genes: {correlation['n_genes']}")

    return results
