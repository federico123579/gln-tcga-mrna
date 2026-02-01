#!/usr/bin/env python3
"""
Train GLN models on TCGA Breast Invasive Carcinoma data.

Trains multiple models with different seeds, saves checkpoints,
and stores metrics in the SQLite database.

Usage:
    python train_models.py
    python train_models.py --seeds 42 43 44
    python train_models.py --epochs 20 --lr 0.005
"""

import argparse
import time
from pathlib import Path

import torch
from dataset import load_tcga_tumor_vs_normal
from results import init_database, save_result
from tqdm import tqdm
from train import save_model, train_gln

# =============================================================================
# Paths
# =============================================================================

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_DB = EXPERIMENT_DIR / "results.db"
MODELS_DIR = EXPERIMENT_DIR / "models"

# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Model architecture
    "layer_sizes": [10],
    "context_dimension": 8,
    # Training parameters
    "learning_rate": 1e-3,
    "num_epochs": 10,
    "batch_size": 16,
    # Regularization
    "bias": True,
    "eps": 1e-6,
    "weight_clamp_min": -10.0,
    "weight_clamp_max": 10.0,
    # Reproducibility
    "seeds": [42, 43, 44],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GLN models on TCGA BRCA data")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["seeds"],
        help="Random seeds for training (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["layer_sizes"],
        help="Hidden layer sizes (default: 50 25)",
    )
    parser.add_argument(
        "--context-dim",
        type=int,
        default=DEFAULT_CONFIG["context_dimension"],
        help="Context dimension for gating (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine the device to use for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def main():
    args = parse_args()

    # Build config from arguments
    config = {
        "layer_sizes": args.layers,
        "context_dimension": args.context_dim,
        "learning_rate": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "bias": DEFAULT_CONFIG["bias"],
        "eps": DEFAULT_CONFIG["eps"],
        "weight_clamp_min": DEFAULT_CONFIG["weight_clamp_min"],
        "weight_clamp_max": DEFAULT_CONFIG["weight_clamp_max"],
        "seeds": args.seeds,
    }

    device = get_device(args.device)

    print("=" * 60)
    print("TCGA Breast Cancer: Model Training")
    print("=" * 60)
    print()

    # Initialize output directories
    MODELS_DIR.mkdir(exist_ok=True)

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
    print(f"Training GLN (device={device})...")
    print(f"  Layer sizes: {config['layer_sizes']}")
    print(f"  Context dimension: {config['context_dimension']}")
    print(f"  Epochs: {config['num_epochs']}, Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print()

    accuracies = []

    for seed in tqdm(config["seeds"], desc="Training models", unit="seed"):
        # Split data with this seed
        train_ds, test_ds = dataset.train_test_split(
            test_size=0.2,
            random_seed=seed,
        )

        # Create config for this run
        run_config = {**config, "seed": seed}

        # Train model with timing
        start_time = time.time()
        model, transf, accuracy = train_gln(
            train_ds,
            test_ds,
            run_config,
            device=device,
            verbose=False,
        )
        training_time = time.time() - start_time

        accuracies.append(accuracy)
        tqdm.write(f"  Seed {seed}: Accuracy = {accuracy:.2%} ({training_time:.1f}s)")

        # Save model checkpoint
        model_path = MODELS_DIR / f"gln_seed{seed}.pt"
        save_model(model, transf, run_config, model_path)

        # Save result to database
        save_result(
            conn,
            {
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
            },
        )

    conn.close()

    # Summary statistics
    print()
    print("-" * 40)
    avg_acc = sum(accuracies) / len(accuracies)
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    print(f"Results across {len(config['seeds'])} seeds:")
    print(f"  Average accuracy: {avg_acc:.2%}")
    print(f"  Min accuracy:     {min_acc:.2%}")
    print(f"  Max accuracy:     {max_acc:.2%}")
    print()

    print("=" * 60)
    print("Training complete!")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Database:        {RESULTS_DB}")
    print()
    print("Run 'python run_analysis.py' to generate biomarker reports.")
    print("=" * 60)


if __name__ == "__main__":
    main()
