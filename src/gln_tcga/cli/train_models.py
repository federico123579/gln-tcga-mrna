#!/usr/bin/env python3
"""
Train GLN models on TCGA Breast Invasive Carcinoma data.

Trains multiple models with different seeds, saves checkpoints,
and stores metrics in the SQLite database.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm

from gln_tcga.dataset import load_tcga_tumor_vs_normal
from gln_tcga.experiments import resolve_experiment, snapshot_latest, write_metadata
from gln_tcga.results import init_database, save_result
from gln_tcga.train import save_model, train_gln

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GLN models on TCGA BRCA data")
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["seeds"],
        help="Random seeds for training (default: 42 43 44)",
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
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_CONFIG["layer_sizes"],
        help="Hidden layer sizes (default: 10)",
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
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable snapshotting the latest run into a timestamped folder",
    )
    parser.add_argument(
        "--no-overwrite-latest",
        action="store_true",
        help="Do not clear the latest directory before training",
    )
    return parser.parse_args(argv)


def get_device(device_arg: str) -> str:
    """Determine the device to use for training."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def run_training(args: argparse.Namespace) -> None:
    paths = resolve_experiment(args.experiment, create=True)

    if not args.no_overwrite_latest and paths.latest_dir.exists():
        shutil.rmtree(paths.latest_dir)
        paths.latest_dir.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        paths.results_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Experiment: {paths.name}")
    print()

    # Initialize database
    conn = init_database(paths.db_path)
    print(f"Results database: {paths.db_path}")
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

    accuracies: list[float] = []
    per_seed: list[dict[str, float]] = []

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
        per_seed.append({"seed": seed, "accuracy": accuracy})
        tqdm.write(f"  Seed {seed}: Accuracy = {accuracy:.2%} ({training_time:.1f}s)")

        # Save model checkpoint
        model_path = paths.models_dir / f"gln_seed{seed}.pt"
        save_model(model, transf, run_config, model_path)
        model_path_rel = str(model_path.relative_to(paths.latest_dir))

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
                "model_path": model_path_rel,
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

    (paths.latest_dir / "train_model_config.json").write_text(
        json.dumps(config, indent=2)
    )
    (paths.latest_dir / "accuracy.json").write_text(
        json.dumps({"average_accuracy": avg_acc, "per_seed": per_seed}, indent=2)
    )

    write_metadata(
        paths,
        config=config,
        metrics={"average_accuracy": avg_acc, "per_seed": per_seed},
    )

    if not args.no_snapshot:
        snapshot_latest(paths)

    print("=" * 60)
    print("Training complete!")
    print(f"  Models saved to: {paths.models_dir}")
    print(f"  Database:        {paths.db_path}")
    print("=" * 60)


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
