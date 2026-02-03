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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

if TYPE_CHECKING:
    import pandas as pd
    from gln_tcga.baselines import CVResult


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

    ax.set_xlabel("Integrated Gradients Rank")
    ax.set_ylabel("Permutation Importance Rank")
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
