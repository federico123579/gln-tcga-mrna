#!/usr/bin/env python3
"""
GLN experiment on TCGA Breast Invasive Carcinoma data.

This is a convenience wrapper that runs both training and analysis.
For more control, use the separate scripts:
    - train_models.py: Train models and save to database
    - run_analysis.py: Analyze saved models

Usage:
    python run_experiment.py
"""

import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent


def main():
    print("=" * 60)
    print("TCGA Breast Cancer: Full Experiment Pipeline")
    print("=" * 60)
    print()
    print("This script runs both training and analysis.")
    print("For more control, use train_models.py and run_analysis.py separately.")
    print()

    # Run training
    print("Step 1: Training models...")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "train_models.py")],
        cwd=EXPERIMENT_DIR,
    )
    if result.returncode != 0:
        print("Training failed!")
        sys.exit(1)

    print()

    # Run analysis
    print("Step 2: Running biomarker analysis...")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "run_analysis.py"), "--best"],
        cwd=EXPERIMENT_DIR,
    )
    if result.returncode != 0:
        print("Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
