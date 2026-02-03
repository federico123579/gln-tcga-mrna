#!/usr/bin/env python3
"""
Generate training curves for GLN experiments.

Either loads saved training history or retrains the model to
capture epoch-by-epoch loss and accuracy metrics.
"""

from __future__ import annotations

import argparse
import json

from gln_tcga.dataset import load_tcga_tumor_vs_normal
from gln_tcga.experiments import (
    load_legacy_config,
    resolve_experiment,
    resolve_run_dir,
)
from gln_tcga.plotting import plot_training_curves
from gln_tcga.results import get_results_df, init_database
from gln_tcga.train import train_gln


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate training curves for GLN experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="latest",
        help="Run id (timestamp) or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from experiment config or best model)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain model to capture training history",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training progress",
    )
    return parser.parse_args()


def run_curves(args: argparse.Namespace) -> None:
    """Generate training curves."""
    print("=" * 60)
    print("TCGA Breast Cancer: Training Curves")
    print("=" * 60)

    paths = resolve_experiment(args.experiment, create=False)

    # Handle legacy experiments
    is_legacy = not paths.latest_dir.exists() and (paths.root / "model.pt").exists()

    if is_legacy:
        config = load_legacy_config(paths.root)
        results_dir = paths.root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = resolve_run_dir(paths, args.run)
        results_db = run_dir / "results.db"
        results_dir = run_dir / "results"

        if not results_db.exists():
            print("Error: No results database found.")
            print("Run training first.")
            return

        conn = init_database(results_db)
        results_df = get_results_df(conn)
        conn.close()

        if results_df.is_empty():
            print("Error: No trained models found.")
            return

        # Load metadata for config
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            config = metadata.get("config", {})
        else:
            config = {}

    # Get seed from args or config
    seed = args.seed or config.get("seed", config.get("seeds", 42))
    if isinstance(seed, list):
        seed = seed[0]

    print(f"Experiment: {args.experiment}")
    print(f"Seed: {seed}")

    # Check for saved training history
    history_file = results_dir / "training_history.json"

    if history_file.exists() and not args.retrain:
        print(f"\nLoading saved training history from: {history_file}")
        with open(history_file) as f:
            history = json.load(f)
    else:
        if not args.retrain:
            print("\nNo saved training history found.")
            print("Use --retrain to train model and capture history.")
            return

        print("\nRetraining model to capture training history...")

        # Load dataset
        print("\nLoading TCGA mRNA expression data...")
        dataset = load_tcga_tumor_vs_normal()
        print(f"  {dataset}")

        train_ds, test_ds = dataset.train_test_split(
            test_size=config.get("test_size", 0.2),
            random_seed=seed,
        )
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Test: {len(test_ds)} samples")

        # Prepare config for training
        train_config = {
            "layer_sizes": config.get("layer_sizes", [64, 32]),
            "context_dimension": config.get("context_dimension", 4),
            "learning_rate": config.get("learning_rate", 0.001),
            "num_epochs": config.get("num_epochs", 100),
            "batch_size": config.get("batch_size", 32),
            "seed": seed,
            "bias": config.get("bias", True),
            "eps": config.get("eps", 1e-6),
            "weight_clamp_min": config.get("weight_clamp_min", -10.0),
            "weight_clamp_max": config.get("weight_clamp_max", 10.0),
        }

        print("\nTraining config:")
        print(f"  Layer sizes: {train_config['layer_sizes']}")
        print(f"  Context dimension: {train_config['context_dimension']}")
        print(f"  Learning rate: {train_config['learning_rate']}")
        print(f"  Epochs: {train_config['num_epochs']}")
        print(f"  Batch size: {train_config['batch_size']}")

        # Train with history tracking
        print("\nTraining GLN...")
        _, _, test_acc, history = train_gln(
            train_ds,
            test_ds,
            train_config,
            device=args.device,
            verbose=args.verbose,
            return_history=True,
        )

        print(f"\nFinal test accuracy: {test_acc * 100:.2f}%")

        # Save training history
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved training history to: {history_file}")

    # Generate training curves plot
    print("\nGenerating training curves plot...")
    output_path = results_dir / "training_curves.png"
    plot_training_curves(
        history,
        output_path=output_path,
        title=f"GLN Training Curves - {args.experiment}",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_batches = len(history.get("batch_losses", []))
    final_loss = history["batch_losses"][-1] if history.get("batch_losses") else None
    final_acc = (
        history["batch_test_accs"][-1] if history.get("batch_test_accs") else None
    )

    print(f"  Batches: {n_batches}")
    if final_loss is not None:
        print(f"  Final loss: {final_loss:.4f}")
    if final_acc is not None:
        print(f"  Final accuracy: {final_acc * 100:.2f}%")
    print(f"\nOutput saved to: {results_dir}")
    print("=" * 60)


def main() -> None:
    run_curves(parse_args())


if __name__ == "__main__":
    main()
