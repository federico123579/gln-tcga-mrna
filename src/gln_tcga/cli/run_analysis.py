#!/usr/bin/env python3
"""
Run biomarker analysis on trained GLN models.

Loads saved models from the database/checkpoints and generates
gene importance reports using Integrated Gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl

from gln_tcga.analyze import generate_report
from gln_tcga.dataset import load_tcga_tumor_vs_normal
from gln_tcga.experiments import (
    ExperimentInfo,
    list_experiments,
    load_legacy_accuracy,
    load_legacy_config,
    resolve_experiment,
    resolve_run_dir,
)
from gln_tcga.results import get_results_df, init_database
from gln_tcga.train import load_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run biomarker analysis on trained GLN models"
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
        help="Analyze model trained with specific seed",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all trained models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and their metrics",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Analyze the best model (default behavior)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of integration steps for Integrated Gradients (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for analysis (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["mean", "normal", "zero"],
        default="mean",
        help="Baseline type for Integrated Gradients: 'mean' (recommended), 'normal', or 'zero' (default: mean)",
    )
    return parser.parse_args()


def list_models(results_df: pl.DataFrame) -> None:
    """Print available models and their metrics."""
    print("\n" + "=" * 70)
    print("Available Trained Models")
    print("=" * 70)

    if results_df.is_empty():
        print("\nNo trained models found. Run training first.")
        return

    # Sort by accuracy descending
    results_df = results_df.sort("accuracy", descending=True)

    print(f"\n{'Seed':<8} {'Accuracy':<12} {'Time (s)':<12} {'Model Path':<40}")
    print("-" * 70)

    for row in results_df.iter_rows(named=True):
        seed = row["seed"]
        accuracy = row["accuracy"]
        time_s = row.get("training_time", 0) or 0
        model_path = (
            Path(row.get("model_path", "")).name if row.get("model_path") else "N/A"
        )
        print(f"{seed:<8} {accuracy:<12.4f} {time_s:<12.1f} {model_path:<40}")

    print("-" * 70)
    avg_acc = results_df["accuracy"].mean()
    print(f"{'Average':<8} {avg_acc:<12.4f}")
    print()


def list_experiments_cli(experiments: list[ExperimentInfo]) -> None:
    print("\n" + "=" * 70)
    print("Available Experiments")
    print("=" * 70)

    if not experiments:
        print("\nNo experiments found in the experiments directory.")
        return

    for info in experiments:
        kind = "workspace" if info.kind == "workspace" else "legacy"
        latest = info.latest_dir.name if info.latest_dir else "-"
        print(f"- {info.name} ({kind}) latest={latest}")


def analyze_model(
    seed: int,
    model_path: Path,
    dataset,
    output_dir: Path,
    n_steps: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    baseline_type: str = "mean",
) -> None:
    """Run analysis on a single model."""
    print(f"\nAnalyzing model (seed={seed})...")
    print(f"  Loading from: {model_path}")

    # Load model
    model, transf, _ = load_model(model_path, device=device)

    # Create output directory for this seed
    seed_output_dir = output_dir / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report (pass y for baseline computation)
    generate_report(
        model,
        transf,
        dataset.X,
        dataset.gene_names,
        seed_output_dir,
        y=dataset.y,
        n_steps=n_steps,
        batch_size=batch_size,
        baseline_type=baseline_type,
    )


def resolve_model_path(model_value: str, run_dir: Path, models_dir: Path) -> Path:
    path = Path(model_value)
    if path.is_absolute() and path.exists():
        return path

    if not path.is_absolute():
        candidate = run_dir / path
        if candidate.exists():
            return candidate

    candidate = models_dir / path.name
    if candidate.exists():
        return candidate

    return path


def analyze_legacy_experiment(
    experiment_root: Path,
    args: argparse.Namespace,
) -> None:
    config = load_legacy_config(experiment_root)
    accuracy = load_legacy_accuracy(experiment_root)
    model_path = experiment_root / "model.pt"

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    seed = config.get("seeds", 0)
    print("=" * 60)
    print("TCGA Breast Cancer: Biomarker Analysis (Legacy)")
    print("=" * 60)
    print(f"Experiment: {experiment_root.name}")
    if accuracy is not None:
        print(f"Accuracy: {accuracy}")

    dataset = load_tcga_tumor_vs_normal()

    results_dir = experiment_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    analyze_model(
        int(seed) if isinstance(seed, int) else 0,
        model_path,
        dataset,
        results_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        baseline_type=args.baseline,
    )


def run_analysis(args: argparse.Namespace) -> None:
    if args.list_experiments:
        list_experiments_cli(list_experiments())
        return

    if not (args.all or args.seed is not None or args.list or args.best):
        args.best = True

    paths = resolve_experiment(args.experiment, create=False)

    # Legacy experiment fallback
    if not paths.latest_dir.exists() and (paths.root / "model.pt").exists():
        analyze_legacy_experiment(paths.root, args)
        return

    run_dir = resolve_run_dir(paths, args.run)
    results_db = run_dir / "results.db"
    models_dir = run_dir / "models"
    results_dir = run_dir / "results"

    # Check if database exists
    if not results_db.exists():
        print("Error: No results database found.")
        print("Run training first or check the experiment/run selection.")
        return

    # Load results from database
    conn = init_database(results_db)
    results_df = get_results_df(conn)
    conn.close()

    if results_df.is_empty():
        print("Error: No trained models found in database.")
        print("Run training first.")
        return

    if args.list:
        list_models(results_df)
        return

    print("=" * 60)
    print("TCGA Breast Cancer: Biomarker Analysis")
    print("=" * 60)
    print(f"Experiment: {paths.name}")
    print(f"Run: {args.run}")

    # Load dataset for analysis
    print("\nLoading TCGA mRNA expression data...")
    dataset = load_tcga_tumor_vs_normal()
    print(f"  {dataset}")

    # Ensure output directory exists
    results_dir.mkdir(exist_ok=True)

    if args.all:
        # Analyze all models
        print(f"\nAnalyzing all {len(results_df)} models...")
        for row in results_df.iter_rows(named=True):
            seed = row["seed"]
            model_path = resolve_model_path(row["model_path"], run_dir, models_dir)
            if model_path.exists():
                analyze_model(
                    seed,
                    model_path,
                    dataset,
                    results_dir,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    device=args.device,
                    baseline_type=args.baseline,
                )
            else:
                print(f"  Warning: Model file not found: {model_path}")

    elif args.seed is not None:
        matches = results_df.filter(pl.col("seed") == args.seed)
        if matches.is_empty():
            print(f"Error: No model found with seed {args.seed}")
            print("Use --list to see available models.")
            return

        row = matches.row(0, named=True)
        model_path = resolve_model_path(row["model_path"], run_dir, models_dir)

        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return

        analyze_model(
            args.seed,
            model_path,
            dataset,
            results_dir,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            baseline_type=args.baseline,
        )

    else:
        # Analyze best model (default)
        best_row = results_df.sort("accuracy", descending=True).row(0, named=True)
        seed = best_row["seed"]
        accuracy = best_row["accuracy"]
        model_path = resolve_model_path(best_row["model_path"], run_dir, models_dir)

        print(f"\nBest model: seed={seed}, accuracy={accuracy:.4f}")

        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            return

        analyze_model(
            seed,
            model_path,
            dataset,
            results_dir,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            device=args.device,
            baseline_type=args.baseline,
        )

    print()
    print("=" * 60)
    print("Analysis complete!")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
