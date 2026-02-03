"""
Biomarker analysis for GLN on TCGA data.

Uses Integrated Gradients to compute gene importance - measuring how much
each gene contributes to the model's tumor vs normal predictions.

.. deprecated::
    This module contains the legacy implementation of Integrated Gradients.
    For new code, use :mod:`gln_tcga.attributions` which provides optimized
    Captum-based implementations of Integrated Gradients, Permutation Importance,
    and rank correlation analysis.

    Example migration::

        # Old (this module)
        from gln_tcga.analyze import extract_gene_importance

        # New (recommended)
        from gln_tcga.attributions import compute_integrated_gradients
"""

import warnings
from pathlib import Path

import gln
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Known breast cancer-related genes for validation
KNOWN_CANCER_GENES = [
    "BRCA1",  # Breast cancer gene 1
    "BRCA2",  # Breast cancer gene 2
    "ERBB2",  # HER2/neu (growth factor receptor)
    "TP53",  # Tumor suppressor
    "ESR1",  # Estrogen receptor alpha
    "PGR",  # Progesterone receptor
    "MKI67",  # Proliferation marker (Ki-67)
    "EGFR",  # Epidermal growth factor receptor
    "MYC",  # Proto-oncogene
    "PIK3CA",  # PI3K pathway
    "CDH1",  # E-cadherin (invasion)
    "PTEN",  # Tumor suppressor
    "GATA3",  # Transcription factor (luminal)
    "FOXA1",  # Transcription factor (hormone response)
]


def integrated_gradients(
    model: gln.GLN,
    inputs: torch.Tensor,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients attribution for each input feature.

    .. deprecated::
        Use :func:`gln_tcga.attributions.compute_integrated_gradients` instead,
        which uses Captum's optimized implementation.

    Integrated Gradients (Sundararajan et al., 2017) attributes importance
    by integrating gradients along the path from a baseline to the input.
    This satisfies key axioms: sensitivity and implementation invariance.

    Args:
        model: Trained GLN model (already expects transformed inputs).
        inputs: Input tensor of shape (n_samples, n_features).
        baseline: Baseline tensor (default: zeros, representing "no expression").
        n_steps: Number of steps for approximating the integral.

    Returns:
        Attribution tensor of shape (n_samples, n_features).
    """
    warnings.warn(
        "integrated_gradients() is deprecated. Use gln_tcga.attributions.compute_integrated_gradients() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # Scale inputs along the path from baseline to input
    # shape: (n_steps, n_samples, n_features)
    alphas = torch.linspace(0, 1, n_steps, device=inputs.device)
    scaled_inputs = baseline + alphas.view(-1, 1, 1) * (inputs - baseline)

    # Flatten for batch processing: (n_steps * n_samples, n_features)
    batch_size = inputs.shape[0]
    scaled_inputs = scaled_inputs.view(-1, inputs.shape[1])
    scaled_inputs.requires_grad_(True)

    # Forward pass
    model.eval()
    outputs = model(scaled_inputs)

    # Backward pass to get gradients
    grad_outputs = torch.ones_like(outputs)
    grads = torch.autograd.grad(outputs, scaled_inputs, grad_outputs)[0]

    # Reshape back: (n_steps, n_samples, n_features)
    grads = grads.view(n_steps, batch_size, -1)

    # Approximate integral using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2

    # Multiply by (input - baseline) to get attributions
    attributions = avg_grads * (inputs - baseline)

    return attributions


def extract_gene_importance(
    model: gln.GLN,
    transf: gln.InputTransformer,
    X: torch.Tensor,
    gene_names: list[str],
    y: torch.Tensor | None = None,
    n_steps: int = 50,
    batch_size: int = 64,
    baseline_type: str = "mean",
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Extract gene importance using Integrated Gradients.

    .. deprecated::
        Use :func:`gln_tcga.attributions.compute_integrated_gradients` instead,
        which uses Captum's optimized implementation.

    Computes how much each gene contributes to predicting "tumor" (class 1)
    across all samples. Genes with high positive attribution push predictions
    toward tumor; genes with high negative attribution push toward normal.

    Args:
        model: Trained GLN model.
        transf: Fitted input transformer.
        X: Raw input data tensor (before transformation).
        gene_names: List of gene names corresponding to input features.
        y: Labels tensor (optional, used for "normal" baseline type).
        n_steps: Number of integration steps (higher = more accurate).
        batch_size: Batch size for processing samples.
        baseline_type: Type of baseline to use:
            - "mean": Dataset mean (transforms to ~0.5 for all genes) - RECOMMENDED
            - "normal": Mean of normal samples (biologically meaningful)
            - "zero": Raw zeros (problematic due to sigmoid saturation)

    Returns:
        Tuple of:
            - DataFrame with gene names, importance scores, and direction
            - Raw attribution tensor (n_samples, n_genes)
    """
    warnings.warn(
        "extract_gene_importance() is deprecated. Use gln_tcga.attributions.compute_integrated_gradients() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    model.eval()

    # Transform inputs
    X_transformed = transf.transform(X)

    # Choose baseline based on type
    if baseline_type == "mean":
        # Use dataset mean as baseline - this transforms to ~0.5 for all genes,
        # avoiding the sigmoid saturation problem with zeros
        baseline_raw = transf.means.unsqueeze(0).expand(X.shape[0], -1)
        baseline = transf.transform(baseline_raw)
    elif baseline_type == "normal" and y is not None:
        # Use mean of normal samples as baseline (biologically meaningful)
        normal_mask = y == 0
        if normal_mask.sum() > 0:
            baseline_raw = (
                X[normal_mask].mean(dim=0, keepdim=True).expand(X.shape[0], -1)
            )
            baseline = transf.transform(baseline_raw)
        else:
            # Fallback to mean if no normal samples
            baseline_raw = transf.means.unsqueeze(0).expand(X.shape[0], -1)
            baseline = transf.transform(baseline_raw)
    else:
        # Zero baseline (original behavior - NOT recommended)
        baseline = torch.zeros_like(X_transformed)

    # Process in batches to manage memory
    all_attributions = []
    n_samples = X_transformed.shape[0]

    batch_iter = range(0, n_samples, batch_size)
    batch_iter = tqdm(batch_iter, desc="Computing attributions", unit="batch")

    for i in batch_iter:
        batch_X = X_transformed[i : i + batch_size]
        batch_baseline = baseline[i : i + batch_size]

        with torch.enable_grad():
            batch_attr = integrated_gradients(
                model, batch_X, batch_baseline, n_steps=n_steps
            )
        all_attributions.append(batch_attr.detach())

    # Concatenate all batches
    attributions = torch.cat(all_attributions, dim=0)

    # Aggregate across samples: mean absolute attribution
    # This captures genes that strongly influence predictions (either direction)
    mean_abs_attr = attributions.abs().mean(dim=0).numpy()

    # Also compute signed mean to see direction of effect
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


def plot_top_genes(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: str | Path | None = None,
    title: str = "Top Genes by Integrated Gradients Attribution",
) -> plt.Figure:
    """Create bar chart of top important genes.

    Args:
        importance_df: DataFrame with 'gene', 'importance', and 'direction' columns.
        top_n: Number of top genes to plot.
        output_path: Path to save the figure (optional).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    top_df = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by direction: red = tumor-associated, blue = normal-associated
    colors = []
    for _, row in top_df.iloc[::-1].iterrows():
        if "direction" in row and row["direction"] > 0:
            colors.append("indianred")  # Tumor-associated
        else:
            colors.append("steelblue")  # Normal-associated

    ax.barh(
        top_df["gene"][::-1],
        top_df["importance"][::-1],
        color=colors,
        edgecolor="black",
    )

    ax.set_xlabel("Importance (mean |attribution|)")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Add legend for direction
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="indianred", edgecolor="black", label="Tumor-associated (+)"),
        Patch(facecolor="steelblue", edgecolor="black", label="Normal-associated (-)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def plot_known_cancer_genes(
    importance_df: pd.DataFrame,
    known_genes: list[str] | None = None,
    output_path: str | Path | None = None,
    title: str = "Known Breast Cancer Genes - Attribution Importance",
) -> plt.Figure | None:
    """Highlight known breast cancer genes in the importance ranking.

    Args:
        importance_df: DataFrame with 'gene', 'importance', and 'rank' columns.
        known_genes: List of known cancer genes to highlight.
        output_path: Path to save the figure (optional).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if known_genes is None:
        known_genes = KNOWN_CANCER_GENES

    # Filter to known genes that are in the dataset
    gene_set = set(importance_df["gene"])
    found_genes = [g for g in known_genes if g in gene_set]
    missing_genes = [g for g in known_genes if g not in gene_set]

    if missing_genes:
        print(
            f"Warning: {len(missing_genes)} known genes not found in dataset: {missing_genes}"
        )

    # Get importance for found genes
    known_df = importance_df[importance_df["gene"].isin(found_genes)].copy()
    known_df = known_df.sort_values("importance", ascending=True)

    if known_df.empty:
        print("No known cancer genes found in dataset")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color based on rank (top 100 = green, 100-1000 = yellow, >1000 = red)
    colors = []
    for rank in known_df["rank"]:
        if rank <= 100:
            colors.append("forestgreen")
        elif rank <= 1000:
            colors.append("gold")
        else:
            colors.append("tomato")

    bars = ax.barh(
        known_df["gene"],
        known_df["importance"],
        color=colors,
        edgecolor="black",
    )

    # Add rank annotations
    for bar, rank in zip(bars, known_df["rank"]):
        ax.text(
            bar.get_width() + 0.01 * ax.get_xlim()[1],
            bar.get_y() + bar.get_height() / 2,
            f"Rank: {rank}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Importance (mean |attribution|)")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="forestgreen", edgecolor="black", label="Top 100"),
        Patch(facecolor="gold", edgecolor="black", label="Rank 100-1000"),
        Patch(facecolor="tomato", edgecolor="black", label="Rank > 1000"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    return fig


def generate_report(
    model: gln.GLN,
    transf: gln.InputTransformer,
    X: torch.Tensor,
    gene_names: list[str],
    output_dir: str | Path,
    y: torch.Tensor | None = None,
    n_steps: int = 50,
    batch_size: int = 64,
    baseline_type: str = "mean",
) -> pd.DataFrame:
    """Generate full biomarker analysis report with plots and CSV.

    Args:
        model: Trained GLN model.
        transf: Fitted input transformer.
        X: Raw input data tensor (before transformation).
        gene_names: List of gene names corresponding to input features.
        output_dir: Directory to save outputs.
        y: Labels tensor (optional, needed for "normal" baseline type).
        n_steps: Number of integration steps for IG.
        batch_size: Batch size for IG computation.
        baseline_type: Baseline for IG ("mean", "normal", or "zero").

    Returns:
        DataFrame with gene importance rankings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Biomarker Analysis Report")
    print("=" * 60)

    # Extract importance using Integrated Gradients
    print("\nComputing gene importance via Integrated Gradients...")
    print(f"  Baseline type: {baseline_type}")
    print("  (Measuring actual contribution to tumor vs normal prediction)")
    importance_df, _ = extract_gene_importance(
        model,
        transf,
        X,
        gene_names,
        y=y,
        n_steps=n_steps,
        batch_size=batch_size,
        baseline_type=baseline_type,
    )

    # Save full rankings to CSV
    csv_path = output_dir / "gene_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  Saved rankings to: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")

    plot_top_genes(
        importance_df,
        top_n=30,
        output_path=output_dir / "top_genes.png",
        title="Top 30 Genes by Integrated Gradients Attribution",
    )

    plot_known_cancer_genes(
        importance_df,
        output_path=output_dir / "known_cancer_genes.png",
    )

    # Print summary
    print("\n" + "-" * 40)
    print("Top 10 Most Important Genes:")
    print("-" * 40)
    for _, row in importance_df.head(10).iterrows():
        print(
            f"  {row['rank']:3d}. {row['gene']:12s}  (importance: {row['importance']:.4f})"
        )

    # Check known cancer genes
    print("\n" + "-" * 40)
    print("Known Breast Cancer Genes Rankings:")
    print("-" * 40)
    for gene in KNOWN_CANCER_GENES:
        matches = importance_df[importance_df["gene"] == gene]
        if not matches.empty:
            row = matches.iloc[0]
            print(
                f"  {gene:10s}: Rank {row['rank']:5d} (importance: {row['importance']:.4f})"
            )
        else:
            print(f"  {gene:10s}: NOT FOUND")

    print("\n" + "=" * 60)
    print(f"Report saved to: {output_dir}")
    print("=" * 60)

    return importance_df
