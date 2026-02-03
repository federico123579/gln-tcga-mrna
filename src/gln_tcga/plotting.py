"""
Plotting utilities for GLN experiments.

Provides functions for visualizing training progress, model comparisons,
and confusion matrices.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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
    epoch_losses = history.get("epoch_losses", [])
    epoch_test_accs = history.get("epoch_test_accs", [])

    epochs = range(1, len(epoch_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, epoch_losses, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy curve
    if epoch_test_accs:
        ax2.plot(
            epochs,
            [acc * 100 for acc in epoch_test_accs],
            "g-",
            linewidth=2,
            label="Test Accuracy",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Test Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add final accuracy annotation
        final_acc = epoch_test_accs[-1] * 100
        ax2.axhline(y=final_acc, color="r", linestyle="--", alpha=0.5)
        ax2.annotate(
            f"Final: {final_acc:.2f}%",
            xy=(len(epochs), final_acc),
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
