"""
Training utilities for GLN on TCGA data.
"""

import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path for gln imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gln


def save_model(
    model: gln.GLN,
    transf: gln.InputTransformer,
    config: dict[str, Any],
    path: str | Path,
) -> None:
    """Save trained model, transformer, and config to a checkpoint file.

    Args:
        model: Trained GLN model.
        transf: Input transformer fitted on training data.
        config: Training configuration dict.
        path: Path to save the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "transformer_state": {
            "means": transf.means,
            "stds": transf.stds,
            "eps": transf.eps,
        },
        "config": config,
        "input_size": model.input_size,
        "layer_sizes": model.layer_sizes,
        "context_dimension": model.context_dimension,
    }
    torch.save(checkpoint, path)


def load_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[gln.GLN, gln.InputTransformer, dict[str, Any]]:
    """Load trained model and transformer from a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, transformer, config).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct model
    model = gln.GLN(
        input_size=checkpoint["input_size"],
        layer_sizes=checkpoint["layer_sizes"],
        context_dimension=checkpoint["context_dimension"],
        bias=checkpoint["config"].get("bias", True),
        eps=checkpoint["config"].get("eps", 1e-6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct transformer
    transf = gln.InputTransformer(eps=checkpoint["transformer_state"]["eps"])
    transf.means = checkpoint["transformer_state"]["means"]
    transf.stds = checkpoint["transformer_state"]["stds"]

    return model, transf, checkpoint["config"]


def binary_accuracy(
    model: gln.GLN,
    transf: gln.InputTransformer,
    test_dl: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Compute binary classification accuracy.

    Args:
        model: Trained GLN model.
        transf: Input transformer fitted on training data.
        test_dl: Test data loader.
        device: Device to run inference on.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for X, y in test_dl:
            X = transf.transform(X)
            X, y = X.to(device), y.to(device)
            out = model(X)
            predicted = (out.squeeze() >= 0.5).long()
            correct += (predicted == y.long()).sum().item()
            total += y.size(0)

    return correct / total


def train_gln(
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    config: dict,
    *,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> tuple[gln.GLN, gln.InputTransformer, float]:
    """Train a GLN model on the given data.

    Args:
        train_ds: Training dataset (TensorDataset).
        test_ds: Test dataset (TensorDataset).
        config: Training configuration dict with keys:
            - layer_sizes: List of hidden layer sizes
            - context_dimension: Context dimension for gating
            - learning_rate: Learning rate for Adam optimizer
            - num_epochs: Number of training epochs
            - batch_size: Batch size for training
            - seed: Random seed for reproducibility
        device: Device to train on ("cpu", "cuda", "mps").
        verbose: Whether to print training progress.

    Returns:
        Tuple of (trained_model, input_transformer, test_accuracy).
    """
    seed = config.get("seed", 42)
    generator = torch.Generator().manual_seed(seed)

    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
    )
    test_dl = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    # Fit input transformer on training data
    transf = gln.InputTransformer()
    transf.fit(train_ds.tensors[0])

    # Get input size from data
    input_size = train_ds.tensors[0].shape[1]

    # Create GLN model
    model = gln.GLN(
        input_size=input_size,
        layer_sizes=config["layer_sizes"],
        context_dimension=config["context_dimension"],
        bias=config.get("bias", True),
        eps=config.get("eps", 1e-6),
        generator=generator,
    ).to(device)

    # Optimizer and scheduler
    optim = Adam(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_dl) * config["num_epochs"]
    scheduler = LinearLR(
        optim,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,
    )

    # Weight clamping bounds
    weight_clamp_min = config.get("weight_clamp_min", -10.0)
    weight_clamp_max = config.get("weight_clamp_max", 10.0)

    # Training loop
    model.train()
    epoch_iter = range(config["num_epochs"])
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    for epoch in epoch_iter:
        epoch_loss = 0.0
        n_batches = 0

        for X, y in train_dl:
            X = transf(X)
            X, y = X.to(device), y.unsqueeze(1).to(device)

            optim.zero_grad()
            out = model(X)
            loss = model.loss(out, y)
            loss.backward()
            optim.step()
            scheduler.step()

            # Weight clamping
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(weight_clamp_min, weight_clamp_max)

            epoch_loss += loss.item()
            n_batches += 1

        if verbose:
            avg_loss = epoch_loss / n_batches
            epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")

    # Evaluate accuracy
    accuracy = binary_accuracy(model, transf, test_dl, device=device)

    return model, transf, accuracy
