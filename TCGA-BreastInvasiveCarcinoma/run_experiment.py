#!/usr/bin/env python3
"""
GLN experiment on TCGA Breast Invasive Carcinoma data.

Classifies Tumor vs Normal tissue using mRNA expression profiles.
All 20,472 genes are used as features - the GLN's gating mechanism
handles feature selection internally.

Usage:
    python run_experiment.py
"""

import torch

from dataset import load_tcga_tumor_vs_normal
from train import train_gln

# =============================================================================
# Experiment Configuration
# =============================================================================

CONFIG = {
    # Model architecture
    "layer_sizes": [50, 25],  # Two hidden layers
    "context_dimension": 8,   # Context for gating

    # Training parameters
    "learning_rate": 1e-2,
    "num_epochs": 10,
    "batch_size": 32,

    # Regularization
    "weight_clamp_min": -10.0,
    "weight_clamp_max": 10.0,

    # Reproducibility
    "seeds": [42, 43, 44, 45, 46],
}

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 60)
    print("TCGA Breast Cancer: Tumor vs Normal Classification")
    print("=" * 60)
    print()

    # Load dataset
    print("Loading TCGA mRNA expression data...")
    dataset = load_tcga_tumor_vs_normal()
    print(f"  {dataset}")
    print(f"  Normal samples: {int((dataset.y == 0).sum())}")
    print(f"  Tumor samples:  {int((dataset.y == 1).sum())}")
    print(f"  Features (genes): {dataset.input_size}")
    print()

    # Run experiments with different seeds
    print(f"Training GLN (device={DEVICE})...")
    print(f"  Layer sizes: {CONFIG['layer_sizes']}")
    print(f"  Context dimension: {CONFIG['context_dimension']}")
    print(f"  Epochs: {CONFIG['num_epochs']}, Batch size: {CONFIG['batch_size']}")
    print()

    accuracies = []
    for seed in CONFIG["seeds"]:
        # Split data with this seed
        train_ds, test_ds = dataset.train_test_split(
            test_size=0.2,
            random_seed=seed,
        )

        # Create config for this run
        run_config = {**CONFIG, "seed": seed}

        # Train model
        model, transf, accuracy = train_gln(
            train_ds,
            test_ds,
            run_config,
            device=DEVICE,
            verbose=False,
        )

        accuracies.append(accuracy)
        print(f"  Seed {seed}: Accuracy = {accuracy:.2%}")

    # Summary statistics
    print()
    print("-" * 40)
    avg_acc = sum(accuracies) / len(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    print(f"Results across {len(CONFIG['seeds'])} seeds:")
    print(f"  Average accuracy: {avg_acc:.2%}")
    print(f"  Min accuracy:     {min_acc:.2%}")
    print(f"  Max accuracy:     {max_acc:.2%}")
    print()


if __name__ == "__main__":
    main()
