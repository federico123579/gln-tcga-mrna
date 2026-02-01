#!/usr/bin/env python3
"""
GLN experiment on TCGA Breast Invasive Carcinoma data.

Classifies Tumor vs Normal tissue using mRNA expression profiles.
All 20,472 genes are used as features - the GLN's gating mechanism
handles feature selection internally.

Features:
    - Automated data download from cBioPortal with caching
    - Results stored in SQLite database
    - Model checkpoints saved for reproducibility
    - Biomarker analysis report with gene importance rankings

Usage:
    python run_experiment.py
"""

import time
from pathlib import Path

import torch

from analyze import generate_report
from dataset import load_tcga_tumor_vs_normal
from results import init_database, save_result, get_summary_stats
from train import save_model, train_gln

# =============================================================================
# Paths
# =============================================================================

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DB = EXPERIMENT_DIR / "results.db"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"

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
    "bias": True,
    "eps": 1e-6,
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

    # Initialize output directories
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Initialize database
    conn = init_database(RESULTS_DB)
    print(f"Results database: {RESULTS_DB}")
    print()

    # Load dataset (auto-downloads if needed)
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
    best_model = None
    best_transf = None
    best_accuracy = 0.0

    for seed in CONFIG["seeds"]:
        # Split data with this seed
        train_ds, test_ds = dataset.train_test_split(
            test_size=0.2,
            random_seed=seed,
        )

        # Create config for this run
        run_config = {**CONFIG, "seed": seed}

        # Train model with timing
        start_time = time.time()
        model, transf, accuracy = train_gln(
            train_ds,
            test_ds,
            run_config,
            device=DEVICE,
            verbose=False,
        )
        training_time = time.time() - start_time

        accuracies.append(accuracy)
        print(f"  Seed {seed}: Accuracy = {accuracy:.2%} ({training_time:.1f}s)")

        # Save model checkpoint
        model_path = MODELS_DIR / f"gln_seed{seed}.pt"
        save_model(model, transf, run_config, model_path)

        # Save result to database
        save_result(conn, {
            "seed": seed,
            "accuracy": accuracy,
            "training_time": training_time,
            "layer_sizes": run_config["layer_sizes"],
            "context_dimension": run_config["context_dimension"],
            "learning_rate": run_config["learning_rate"],
            "num_epochs": run_config["num_epochs"],
            "batch_size": run_config["batch_size"],
            "n_train_samples": len(train_ds),
            "n_test_samples": len(test_ds),
            "n_genes": dataset.input_size,
            "model_path": str(model_path),
        })

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_transf = transf

    conn.close()

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

    # Generate biomarker analysis using best model
    print("Generating biomarker analysis report...")
    generate_report(
        best_model,
        dataset.gene_names,
        RESULTS_DIR,
        CONFIG,
    )

    print()
    print("=" * 60)
    print("Experiment complete!")
    print(f"  Models saved to:  {MODELS_DIR}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Database:         {RESULTS_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()
