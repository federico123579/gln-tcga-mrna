"""Experiment registry and snapshot utilities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"


@dataclass
class ExperimentPaths:
    name: str
    root: Path
    latest_dir: Path
    runs_dir: Path
    models_dir: Path
    results_dir: Path
    db_path: Path


@dataclass
class ExperimentInfo:
    name: str
    root: Path
    kind: str  # "workspace" | "legacy"
    latest_dir: Path | None
    runs_dir: Path | None
    metadata_path: Path | None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_experiment(name: str, *, create: bool = True) -> ExperimentPaths:
    """Resolve experiment paths and optionally create directories."""
    root = EXPERIMENTS_DIR / name
    latest_dir = root / "latest"
    runs_dir = root / "runs"
    models_dir = latest_dir / "models"
    results_dir = latest_dir / "results"
    db_path = latest_dir / "results.db"

    if create:
        results_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        name=name,
        root=root,
        latest_dir=latest_dir,
        runs_dir=runs_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        db_path=db_path,
    )


def write_metadata(
    paths: ExperimentPaths,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Write experiment metadata to latest directory."""
    metadata_path = paths.latest_dir / "metadata.json"
    payload = {
        "name": paths.name,
        "updated_at": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics or {},
        "paths": {
            "root": str(paths.root),
            "latest_dir": str(paths.latest_dir),
            "models_dir": str(paths.models_dir),
            "results_dir": str(paths.results_dir),
            "db_path": str(paths.db_path),
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2))
    return metadata_path


def snapshot_latest(paths: ExperimentPaths) -> Path:
    """Snapshot the latest run into a timestamped folder and update latest marker."""
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = _timestamp()
    run_dir = paths.runs_dir / run_id

    if run_dir.exists():
        shutil.rmtree(run_dir)

    shutil.copytree(paths.latest_dir, run_dir)

    latest_marker = paths.root / "latest.json"
    latest_marker.write_text(
        json.dumps({"latest_run": run_id, "latest_dir": str(paths.latest_dir)}, indent=2)
    )

    return run_dir


def list_experiments() -> list[ExperimentInfo]:
    """List experiments found under the experiments directory."""
    if not EXPERIMENTS_DIR.exists():
        return []

    experiments: list[ExperimentInfo] = []
    for item in sorted(EXPERIMENTS_DIR.iterdir()):
        if not item.is_dir():
            continue

        latest_dir = item / "latest"
        runs_dir = item / "runs"
        metadata_path = latest_dir / "metadata.json"

        if latest_dir.exists():
            experiments.append(
                ExperimentInfo(
                    name=item.name,
                    root=item,
                    kind="workspace",
                    latest_dir=latest_dir,
                    runs_dir=runs_dir if runs_dir.exists() else None,
                    metadata_path=metadata_path if metadata_path.exists() else None,
                )
            )
            continue

        # Legacy layout: model.pt + train_model_config.json
        legacy_model = item / "model.pt"
        legacy_config = item / "train_model_config.json"
        if legacy_model.exists() and legacy_config.exists():
            experiments.append(
                ExperimentInfo(
                    name=item.name,
                    root=item,
                    kind="legacy",
                    latest_dir=None,
                    runs_dir=None,
                    metadata_path=None,
                )
            )

    return experiments


def resolve_run_dir(paths: ExperimentPaths, run_id: str | None = None) -> Path:
    """Resolve a run directory for a given run id or latest."""
    if run_id is None or run_id == "latest":
        return paths.latest_dir

    run_dir = paths.runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    return run_dir


def load_legacy_config(experiment_root: Path) -> dict[str, Any]:
    """Load legacy experiment config."""
    config_path = experiment_root / "train_model_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def load_legacy_accuracy(experiment_root: Path) -> float | None:
    """Load legacy experiment accuracy if present."""
    acc_path = experiment_root / "accuracy.json"
    if not acc_path.exists():
        return None
    payload = json.loads(acc_path.read_text())
    return payload.get("accuracy")
