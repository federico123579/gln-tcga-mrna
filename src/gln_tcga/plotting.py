"""
Plotting utilities for GLN experiments.

Provides functions for visualizing training progress, model comparisons,
and confusion matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

if TYPE_CHECKING:
    import pandas as pd

    from gln_tcga.baselines import CVResult


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


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path | None = None,
    title: str = "GLN Training Curves",
) -> plt.Figure:
    """Plot training loss and test accuracy over epochs.

    Args:
        history: Dictionary with 'epoch_losses' and 'epoch_test_accs' lists.
        output_path: Path to save the figure (optional).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    batch_losses = history.get("batch_losses", [])
    batch_test_accs = history.get("batch_test_accs", [])

    batch = range(1, len(batch_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(batch, batch_losses, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    if batch_test_accs:
        ax2.plot(
            batch,
            [acc * 100 for acc in batch_test_accs],
            "g-",
            linewidth=2,
            label="Test Accuracy",
        )
        ax2.set_xlabel("Batch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Test Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add final accuracy annotation
        final_acc = batch_test_accs[-1] * 100
        ax2.axhline(y=final_acc, color="r", linestyle="--", alpha=0.5)
        ax2.annotate(
            f"Final: {final_acc:.2f}%",
            xy=(len(batch), final_acc),
            xytext=(-50, 10),
            textcoords="offset points",
            fontsize=10,
            color="red",
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to: {output_path}")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    output_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    class_names: list[str] | None = None,
) -> plt.Figure:
    """Plot confusion matrix using sklearn.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path to save the figure (optional).
        title: Plot title.
        class_names: Names for classes (default: ["Normal", "Tumor"]).

    Returns:
        Matplotlib Figure object.
    """
    if class_names is None:
        class_names = ["Normal", "Tumor"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add accuracy annotation
    accuracy = np.trace(cm) / cm.sum() * 100
    ax.text(
        0.5,
        -0.15,
        f"Accuracy: {accuracy:.2f}%",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to: {output_path}")

    return fig


def plot_model_comparison(
    results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
    title: str = "Model Comparison",
    metric: str = "accuracy",
) -> plt.Figure:
    """Plot bar chart comparing model performance.

    Args:
        results: Dictionary mapping model names to result dicts.
            Each result dict should have 'accuracy' key.
        output_path: Path to save the figure (optional).
        title: Plot title.
        metric: Metric to compare (default: "accuracy").

    Returns:
        Matplotlib Figure object.
    """
    model_names = list(results.keys())
    accuracies = [results[name].get(metric, 0) * 100 for name in model_names]

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars: highlight the best model
    colors = ["forestgreen" if i == 0 else "steelblue" for i in range(len(model_names))]

    bars = ax.bar(model_names, accuracies, color=colors, edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved model comparison to: {output_path}")

    return fig


def plot_model_comparison_cv(
    results: dict[str, "CVResult"],
    output_path: str | Path | None = None,
    title: str = "Model Comparison (K-Fold CV)",
) -> plt.Figure:
    """Plot bar chart comparing model performance with error bars.

    Args:
        results: Dictionary mapping model names to CVResult objects.
        output_path: Path to save the figure (optional).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    model_names = list(results.keys())
    mean_accs = [results[name].mean_accuracy * 100 for name in model_names]
    std_accs = [results[name].std_accuracy * 100 for name in model_names]

    # Sort by mean accuracy
    sorted_indices = np.argsort(mean_accs)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    mean_accs = [mean_accs[i] for i in sorted_indices]
    std_accs = [std_accs[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color bars: highlight the best model
    colors = ["forestgreen" if i == 0 else "steelblue" for i in range(len(model_names))]

    bars = ax.bar(
        model_names,
        mean_accs,
        yerr=std_accs,
        color=colors,
        edgecolor="black",
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, mean_accs, std_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 1,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved CV model comparison to: {output_path}")

    return fig


def plot_rank_correlation(
    ig_df: "pd.DataFrame",
    perm_df: "pd.DataFrame",
    output_path: str | Path | None = None,
    top_n: int | None = None,
    title: str = "IG vs Permutation Gene Ranks",
    x_label: str = "Integrated Gradients Rank",
    y_label: str = "Permutation Importance Rank",
) -> plt.Figure:
    """Plot scatter plot comparing gene ranks between two methods.

    Args:
        ig_df: DataFrame with 'gene' and 'rank' from Integrated Gradients.
        perm_df: DataFrame with 'gene' and 'rank' from Permutation Importance.
        output_path: Path to save the figure (optional).
        top_n: If set, only plot top N genes by IG rank.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    from scipy import stats

    # Merge on gene names
    merged = ig_df[["gene", "rank"]].merge(
        perm_df[["gene", "rank"]],
        on="gene",
        suffixes=("_ig", "_perm"),
    )

    if top_n is not None:
        merged = merged[merged["rank_ig"] <= top_n]

    # Compute correlation
    correlation, p_value = stats.spearmanr(merged["rank_ig"], merged["rank_perm"])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        merged["rank_ig"],
        merged["rank_perm"],
        alpha=0.5,
        s=20,
        c="steelblue",
        edgecolor="none",
    )

    # Add diagonal line (perfect correlation)
    max_rank = max(merged["rank_ig"].max(), merged["rank_perm"].max())
    ax.plot([0, max_rank], [0, max_rank], "r--", alpha=0.7, label="Perfect correlation")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nSpearman r={correlation:.3f}, p={p_value:.2e}", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Make axes equal
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved rank correlation plot to: {output_path}")

    return fig


def plot_sample_saliency(
    gene_names: list[str],
    weights: np.ndarray,
    output_path: str | Path | None = None,
    top_n: int = 30,
    title: str = "Sample Saliency Map",
) -> plt.Figure:
    """Plot a saliency map for a single sample.

    Args:
        gene_names: List of gene names.
        weights: Per-gene saliency weights for the sample.
        output_path: Path to save the figure (optional).
        top_n: Number of genes to display (by absolute weight).
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array for a single sample")

    df = pd.DataFrame(
        {
            "gene": gene_names[: len(weights)],
            "weight": weights,
        }
    )
    df["abs_weight"] = df["weight"].abs()
    df = df.sort_values("abs_weight", ascending=False).head(top_n)
    df = df.sort_values("weight", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["indianred" if w > 0 else "steelblue" for w in df["weight"]]

    ax.barh(df["gene"], df["weight"], color=colors, edgecolor="black")

    ax.set_xlabel("Saliency Weight")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="indianred", edgecolor="black", label="Positive weight"),
        Patch(facecolor="steelblue", edgecolor="black", label="Negative weight"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample saliency plot to: {output_path}")

    return fig


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
