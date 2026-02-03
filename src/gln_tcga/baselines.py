"""
Baseline models for comparison with GLN.

Provides Logistic Regression and MLP baselines using the same data splits.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class BaselineResult:
    """Standardized result container for baseline models."""

    model_name: str
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    model: Any = None  # The trained model object


class MLP(nn.Module):
    """Multi-layer perceptron with same architecture as GLN for fair comparison."""

    def __init__(self, input_size: int, layer_sizes: list[int]):
        """Initialize MLP.

        Args:
            input_size: Number of input features.
            layer_sizes: List of hidden layer sizes (same as GLN config).
        """
        super().__init__()

        layers = []
        prev_size = input_size

        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        # Output layer (binary classification)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_logistic_regression(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 42,
) -> BaselineResult:
    """Train Logistic Regression baseline.

    Args:
        train_ds: Training dataset (TensorDataset).
        test_ds: Test dataset (TensorDataset).
        max_iter: Maximum iterations for solver.
        C: Inverse of regularization strength.
        seed: Random seed for reproducibility.

    Returns:
        BaselineResult with accuracy and predictions.
    """
    # Extract numpy arrays
    X_train = train_ds.tensors[0].numpy()
    y_train = train_ds.tensors[1].numpy()
    X_test = test_ds.tensors[0].numpy()
    y_test = test_ds.tensors[1].numpy()

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    accuracy = (y_pred == y_test).mean()

    return BaselineResult(
        model_name="Logistic Regression",
        accuracy=float(accuracy),
        y_true=y_test,
        y_pred=y_pred,
        model=model,
    )


def train_mlp(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> BaselineResult:
    """Train MLP baseline with same architecture as GLN.

    Args:
        train_ds: Training dataset (TensorDataset).
        test_ds: Test dataset (TensorDataset).
        config: Training configuration dict with keys:
            - layer_sizes: List of hidden layer sizes (same as GLN)
            - learning_rate: Learning rate for Adam optimizer
            - num_epochs: Number of training epochs
            - batch_size: Batch size for training
            - seed: Random seed for reproducibility
        device: Device to train on ("cpu", "cuda", "mps").
        verbose: Whether to print training progress.

    Returns:
        BaselineResult with accuracy and predictions.
    """
    seed = config.get("seed", 42)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )
    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    # Standardize inputs (like GLN's InputTransformer)
    X_train = train_ds.tensors[0]
    means = X_train.mean(dim=0)
    stds = X_train.std(dim=0)
    stds = torch.where(stds < 1e-6, torch.ones_like(stds), stds)

    def transform(x: torch.Tensor) -> torch.Tensor:
        return (x - means) / stds

    # Get input size from data
    input_size = train_ds.tensors[0].shape[1]

    # Create MLP model
    model = MLP(input_size=input_size, layer_sizes=config["layer_sizes"]).to(device)

    # Optimizer and scheduler (same as GLN training)
    optim = Adam(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_dl) * config["num_epochs"]
    scheduler = LinearLR(
        optim,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    model.train()
    epoch_iter = range(config["num_epochs"])
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training MLP", unit="epoch")

    for _ in epoch_iter:
        epoch_loss = 0.0
        n_batches = 0

        for X, y in train_dl:
            X = transform(X)
            X, y = X.to(device), y.float().unsqueeze(1).to(device)

            optim.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if verbose and n_batches > 0:
            avg_loss = epoch_loss / n_batches
            tqdm.write(f"MLP Epoch loss: {avg_loss:.4f}")

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dl:
            X = transform(X).to(device)
            out = model(X)
            preds = (out.squeeze() >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    accuracy = (y_pred == y_true).mean()

    return BaselineResult(
        model_name="MLP",
        accuracy=float(accuracy),
        y_true=y_true,
        y_pred=y_pred,
        model=model,
    )
