#!/usr/bin/env python3
"""
GLN experiment on TCGA Breast Invasive Carcinoma data.

Convenience wrapper that runs both training and analysis.
"""

from __future__ import annotations

import argparse

from gln_tcga.cli.run_analysis import run_analysis
from gln_tcga.cli.train_models import parse_args as parse_train_args
from gln_tcga.cli.train_models import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training + analysis for a GLN TCGA experiment"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Experiment name (default: default)",
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
    parser.add_argument("--train-args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("TCGA Breast Cancer: Full Experiment Pipeline")
    print("=" * 60)
    print()
    print("This script runs both training and analysis.")
    print("For more control, use gln-train and gln-analyze separately.")
    print()

    train_args = parse_train_args(args.train_args or [])
    train_args.experiment = args.experiment
    run_training(train_args)

    analysis_args = argparse.Namespace(
        experiment=args.experiment,
        run="latest",
        seed=None,
        all=False,
        list=False,
        list_experiments=False,
        best=True,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
        baseline=args.baseline,
    )
    run_analysis(analysis_args)


if __name__ == "__main__":
    main()
