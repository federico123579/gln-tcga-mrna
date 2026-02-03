#!/usr/bin/env python3
"""
Run baseline model comparisons for GLN experiments.

Trains Logistic Regression, MLP, and GLN from scratch using the same
seed and data split for fair, reproducible comparison.
"""

from __future__ import annotations

import argparse
import json

from gln_tcga.baselines import train_logistic_regression, train_mlp, train_with_cv
from gln_tcga.dataset import load_tcga_tumor_vs_normal
from gln_tcga.experiments import (
    load_legacy_config,
    resolve_experiment,
    resolve_run_dir,
)
from gln_tcga.plotting import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_model_comparison_cv,
)
from gln_tcga.train import train_gln


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline comparisons for GLN experiments"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name to load config from (default: default)",
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
        default=42,
        help="Random seed for data split and training (default: 42)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of CV folds (default: None, uses single train/test split)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate model comparison chart",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Generate confusion matrices for all models",
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


def get_predictions(model, transf, test_ds, device: str = "cpu"):
    """Get predictions from a trained GLN model.

    Returns:
        Tuple of (y_true, y_pred) as numpy arrays.
    """
    import torch
    from torch.utils.data import DataLoader

    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)
    y_true = test_ds.tensors[1].numpy()

    model.eval()
    with torch.no_grad():
        for X, _ in test_dl:
            # InputTransformer applies: sigmoid((x - means) / stds)
            X = transf.transform(X).to(device)
            out = model(X)
            y_pred = (out.squeeze() >= 0.5).long().cpu().numpy()

    return y_true, y_pred


def run_baselines(args: argparse.Namespace) -> None:
    """Run baseline comparison experiments."""
    print("=" * 60)
    print("TCGA Breast Cancer: Baseline Comparison")
    print("=" * 60)

    paths = resolve_experiment(args.experiment, create=False)

    # Load experiment config for model architecture
    is_legacy = not paths.latest_dir.exists() and (paths.root / "model.pt").exists()

    if is_legacy:
        config = load_legacy_config(paths.root)
        results_dir = paths.root / "results"
    else:
        run_dir = resolve_run_dir(paths, args.run)
        results_dir = run_dir / "results"

        # Load metadata for config
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            config = metadata.get("config", {})
        else:
            config = {}

    results_dir.mkdir(parents=True, exist_ok=True)

    # Use provided seed for everything
    seed = args.seed

    print(f"Experiment: {args.experiment}")
    print(f"Seed: {seed}")

    # Load dataset
    print("\nLoading TCGA mRNA expression data...")
    dataset = load_tcga_tumor_vs_normal()
    print(f"  {dataset}")

    # Check if we're in CV mode
    if args.cv_folds is not None:
        run_baselines_cv(args, dataset, config, results_dir, seed)
    else:
        run_baselines_single(args, dataset, config, results_dir, seed)


def run_baselines_cv(args, dataset, config, results_dir, seed) -> None:
    """Run baselines with k-fold cross-validation."""
    n_folds = args.cv_folds

    print(f"\nRunning {n_folds}-fold cross-validation...")

    # Build configs
    mlp_config = {
        "layer_sizes": config.get("layer_sizes", [64, 32]),
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 32,
    }

    gln_config = {
        "layer_sizes": config.get("layer_sizes", [64, 32]),
        "context_dimension": config.get("context_dimension", 4),
        "learning_rate": config.get("learning_rate", 0.001),
        "num_epochs": config.get("num_epochs", 100),
        "batch_size": config.get("batch_size", 32),
        "bias": config.get("bias", True),
        "eps": config.get("eps", 1e-6),
        "weight_clamp_min": config.get("weight_clamp_min", -10.0),
        "weight_clamp_max": config.get("weight_clamp_max", 10.0),
    }

    # Train all models with CV
    cv_results = {}

    logreg_cv = train_with_cv(
        dataset, "logreg", {}, n_folds=n_folds, seed=seed, verbose=args.verbose
    )
    cv_results["Logistic Regression"] = logreg_cv

    mlp_cv = train_with_cv(
        dataset,
        "mlp",
        mlp_config,
        n_folds=n_folds,
        seed=seed,
        device=args.device,
        verbose=args.verbose,
    )
    cv_results["MLP"] = mlp_cv

    gln_cv = train_with_cv(
        dataset,
        "gln",
        gln_config,
        n_folds=n_folds,
        seed=seed,
        device=args.device,
        verbose=args.verbose,
    )
    cv_results["GLN"] = gln_cv

    # Save CV results to JSON
    cv_results_json = {
        name: {
            "model_name": result.model_name,
            "fold_accuracies": result.fold_accuracies,
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
        }
        for name, result in cv_results.items()
    }
    results_file = results_dir / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump(cv_results_json, f, indent=2)
    print(f"\nSaved CV results to: {results_file}")

    # Generate comparison chart with error bars
    if args.compare:
        print("\nGenerating CV model comparison chart...")
        plot_model_comparison_cv(
            cv_results,
            output_path=results_dir / "model_comparison_cv.png",
            title=f"Model Comparison - {args.experiment} ({n_folds}-fold CV)",
        )

    # Print summary
    print("\n" + "=" * 60)
    print(f"Summary ({n_folds}-fold Cross-Validation)")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>20}")
    print("-" * 50)
    for name, res in sorted(cv_results.items(), key=lambda x: -x[1].mean_accuracy):
        print(
            f"{name:<25} {res.mean_accuracy * 100:>8.2f}% +/- {res.std_accuracy * 100:.2f}%"
        )
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")


def run_baselines_single(args, dataset, config, results_dir, seed) -> None:
    """Run baselines with single train/test split (original behavior)."""
    train_ds, test_ds = dataset.train_test_split(
        test_size=config.get("test_size", 0.2),
        random_seed=seed,
    )
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")

    results: dict[str, dict] = {}

    # === Train Logistic Regression ===
    print("\n" + "-" * 40)
    print("Training Logistic Regression...")
    logreg_result = train_logistic_regression(train_ds, test_ds, seed=seed)
    print(f"  Accuracy: {logreg_result.accuracy * 100:.2f}%")
    results["Logistic Regression"] = {
        "accuracy": logreg_result.accuracy,
        "y_true": logreg_result.y_true.tolist(),
        "y_pred": logreg_result.y_pred.tolist(),
    }

    # === Train MLP ===
    print("\n" + "-" * 40)
    print("Training MLP...")

    mlp_config = {
        "layer_sizes": config.get("layer_sizes", [64, 32]),
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 32,
        "seed": seed,
    }
    print(f"  Config: {mlp_config}")

    mlp_result = train_mlp(
        train_ds,
        test_ds,
        mlp_config,
        device=args.device,
        verbose=args.verbose,
    )
    print(f"  Accuracy: {mlp_result.accuracy * 100:.2f}%")
    results["MLP"] = {
        "accuracy": mlp_result.accuracy,
        "y_true": mlp_result.y_true.tolist(),
        "y_pred": mlp_result.y_pred.tolist(),
    }

    # === Train GLN ===
    print("\n" + "-" * 40)
    print("Training GLN...")

    gln_config = {
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
    print(
        f"  Config: layer_sizes={gln_config['layer_sizes']}, "
        f"context_dim={gln_config['context_dimension']}, "
        f"epochs={gln_config['num_epochs']}"
    )

    model, transf, gln_accuracy = train_gln(
        train_ds,
        test_ds,
        gln_config,
        device=args.device,
        verbose=args.verbose,
    )

    # Get GLN predictions
    gln_y_true, gln_y_pred = get_predictions(model, transf, test_ds, device=args.device)
    print(f"  Accuracy: {gln_accuracy * 100:.2f}%")

    results["GLN"] = {
        "accuracy": gln_accuracy,
        "y_true": gln_y_true.tolist(),
        "y_pred": gln_y_pred.tolist(),
    }

    # Save results
    results_file = results_dir / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_file}")

    # Generate comparison chart
    if args.compare:
        print("\nGenerating model comparison chart...")
        plot_model_comparison(
            results,
            output_path=results_dir / "model_comparison.png",
            title=f"Model Comparison - {args.experiment} (seed={seed})",
        )

    # Generate confusion matrices
    if args.confusion_matrix:
        print("\nGenerating confusion matrices...")

        plot_confusion_matrix(
            gln_y_true,
            gln_y_pred,
            output_path=results_dir / "confusion_matrix.png",
            title=f"GLN Confusion Matrix (seed={seed})",
        )

        plot_confusion_matrix(
            logreg_result.y_true,
            logreg_result.y_pred,
            output_path=results_dir / "logreg_confusion_matrix.png",
            title=f"Logistic Regression Confusion Matrix (seed={seed})",
        )

        plot_confusion_matrix(
            mlp_result.y_true,
            mlp_result.y_pred,
            output_path=results_dir / "mlp_confusion_matrix.png",
            title=f"MLP Confusion Matrix (seed={seed})",
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>12}")
    print("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{name:<25} {res['accuracy'] * 100:>11.2f}%")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}")


def main() -> None:
    run_baselines(parse_args())


if __name__ == "__main__":
    main()
