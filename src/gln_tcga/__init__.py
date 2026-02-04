"""TCGA Breast Cancer GLN experiment package."""

import gln

from .dataset import Dataset, download_tcga_brca_study, load_tcga_tumor_vs_normal
from .experiments import (
    ExperimentInfo,
    ExperimentPaths,
    list_experiments,
    resolve_experiment,
    snapshot_latest,
)
from .results import get_results_df, init_database, query_results
from .train import load_model, save_model, train_gln

# Re-export useful GLN classes for convenience
LearningRateScheduler = gln.LearningRateScheduler

__all__ = [
    "Dataset",
    "download_tcga_brca_study",
    "load_tcga_tumor_vs_normal",
    "save_model",
    "load_model",
    "train_gln",
    "init_database",
    "query_results",
    "get_results_df",
    "ExperimentInfo",
    "ExperimentPaths",
    "list_experiments",
    "resolve_experiment",
    "snapshot_latest",
    "LearningRateScheduler",
]
