"""
SQLite database for storing experiment results.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


def init_database(db_path: str | Path) -> sqlite3.Connection:
    """Create results database and table if not exists.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Database connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            training_time REAL,
            layer_sizes TEXT NOT NULL,
            context_dimension INTEGER NOT NULL,
            learning_rate REAL NOT NULL,
            num_epochs INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            n_train_samples INTEGER,
            n_test_samples INTEGER,
            n_genes INTEGER,
            model_path TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def save_result(conn: sqlite3.Connection, result: dict[str, Any]) -> int:
    """Insert experiment result into database.

    Args:
        conn: Database connection.
        result: Dict with experiment results. Expected keys:
            - seed: Random seed used
            - accuracy: Test accuracy
            - training_time: Training time in seconds (optional)
            - layer_sizes: List of layer sizes
            - context_dimension: Context dimension
            - learning_rate: Learning rate
            - num_epochs: Number of epochs
            - batch_size: Batch size
            - n_train_samples: Number of training samples (optional)
            - n_test_samples: Number of test samples (optional)
            - n_genes: Number of genes/features (optional)
            - model_path: Path to saved model (optional)

    Returns:
        Row ID of inserted result.
    """
    cursor = conn.cursor()

    # Convert layer_sizes list to JSON string
    layer_sizes_json = json.dumps(result["layer_sizes"])
    timestamp = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO results (
            seed, accuracy, training_time, layer_sizes, context_dimension,
            learning_rate, num_epochs, batch_size, n_train_samples,
            n_test_samples, n_genes, model_path, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["seed"],
            result["accuracy"],
            result.get("training_time"),
            layer_sizes_json,
            result["context_dimension"],
            result["learning_rate"],
            result["num_epochs"],
            result["batch_size"],
            result.get("n_train_samples"),
            result.get("n_test_samples"),
            result.get("n_genes"),
            result.get("model_path"),
            timestamp,
        ),
    )

    conn.commit()
    return cursor.lastrowid


def query_results(db_path: str | Path) -> pl.DataFrame:
    """Load all results from database as a Polars DataFrame.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Polars DataFrame with all results.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM results")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()

    conn.close()

    if not rows:
        return pl.DataFrame(schema={col: pl.Utf8 for col in columns})

    df = pl.DataFrame(rows, schema=columns, orient="row")

    # Parse layer_sizes JSON back to list representation
    df = df.with_columns(pl.col("layer_sizes").alias("layer_sizes"))

    return df


def get_summary_stats(db_path: str | Path) -> dict[str, Any]:
    """Get summary statistics from results database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Dict with summary statistics.
    """
    df = query_results(db_path)

    if df.is_empty():
        return {"n_experiments": 0}

    return {
        "n_experiments": len(df),
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std(),
        "min_accuracy": df["accuracy"].min(),
        "max_accuracy": df["accuracy"].max(),
        "seeds": df["seed"].unique().to_list(),
    }


if __name__ == "__main__":
    # Quick test
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = init_database(db_path)

    # Insert test result
    result = {
        "seed": 42,
        "accuracy": 0.95,
        "training_time": 10.5,
        "layer_sizes": [50, 25],
        "context_dimension": 8,
        "learning_rate": 0.01,
        "num_epochs": 10,
        "batch_size": 32,
        "n_train_samples": 800,
        "n_test_samples": 200,
        "n_genes": 20000,
        "model_path": "models/gln_seed42.pt",
    }

    row_id = save_result(conn, result)
    print(f"Inserted row with ID: {row_id}")

    conn.close()

    # Query results
    df = query_results(db_path)
    print(df)

    # Get summary
    summary = get_summary_stats(db_path)
    print(f"Summary: {summary}")

    # Cleanup
    Path(db_path).unlink()
