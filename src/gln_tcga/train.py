"""
Training utilities for GLN on TCGA data.
"""

from pathlib import Path
from typing import Any

import gln
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
        "w_min": model.w_min,
        "w_max": model.w_max,
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

    # Reconstruct model (handle both old and new checkpoint formats)
    model = gln.GLN(
        input_size=checkpoint["input_size"],
        layer_sizes=checkpoint["layer_sizes"],
        context_dimension=checkpoint["context_dimension"],
        bias=checkpoint["config"].get("bias", True),
        eps=checkpoint["config"].get("eps", 1e-6),
        w_min=checkpoint.get("w_min", 0.0),
        w_max=checkpoint.get("w_max", 1.0),
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
    return_history: bool = False,
    use_online_ogd: bool = False,
) -> (
    tuple[gln.GLN, gln.InputTransformer, float]
    | tuple[gln.GLN, gln.InputTransformer, float, dict[str, list[float]]]
):
    """Train a GLN model on the given data.

    Args:
        train_ds: Training dataset (TensorDataset).
        test_ds: Test dataset (TensorDataset).
        config: Training configuration dict with keys:
            - layer_sizes: List of hidden layer sizes
            - context_dimension: Context dimension for gating
            - learning_rate: Learning rate (for Adam or initial OGD lr)
            - num_epochs: Number of training epochs
            - batch_size: Batch size for training
            - seed: Random seed for reproducibility
            - w_min: (optional) Min weight for OGD projection, default 0.0
            - w_max: (optional) Max weight for OGD projection, default 1.0
            - lr_schedule: (optional) "sqrt", "linear", or "constant" for OGD
        device: Device to train on ("cpu", "cuda", "mps").
        verbose: Whether to print training progress.
        return_history: Whether to return training history (epoch losses/accuracies).
        use_online_ogd: If True, use paper-faithful local Online Gradient Descent
            instead of standard end-to-end backprop with Adam. This implements
            the GLN learning rule from the original paper.

    Returns:
        Tuple of (trained_model, input_transformer, test_accuracy).
        If return_history=True, returns (trained_model, input_transformer,
        test_accuracy, history) where history contains 'epoch_losses' and
        'epoch_test_accs' lists.
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

    # Weight bounds for OGD projection
    w_min = config.get("w_min", 0.0)
    w_max = config.get("w_max", 1.0)

    # Create GLN model
    model = gln.GLN(
        input_size=input_size,
        layer_sizes=config["layer_sizes"],
        context_dimension=config["context_dimension"],
        bias=config.get("bias", True),
        eps=config.get("eps", 1e-6),
        w_min=w_min,
        w_max=w_max,
        generator=generator,
    ).to(device)

    # Training mode selection
    if use_online_ogd:
        # Paper-faithful local Online Gradient Descent
        lr_schedule = config.get("lr_schedule", "sqrt")
        lr_scheduler = gln.LearningRateScheduler(
            initial_lr=config["learning_rate"],
            schedule=lr_schedule,
            min_lr=1e-6,
        )
    else:
        # Standard end-to-end backprop with Adam
        optim = Adam(model.parameters(), lr=config["learning_rate"])
        total_steps = len(train_dl) * config["num_epochs"]
        scheduler = LinearLR(
            optim,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps,
        )
        # scheduler = ConstantLR(
        #     optim,
        #     factor=1,
        #     total_iters=total_steps,
        # )

        # Weight clamping bounds (for backprop mode)
        weight_clamp_min = config.get("weight_clamp_min", -10.0)
        weight_clamp_max = config.get("weight_clamp_max", 10.0)

    # Training loop
    model.train()
    epoch_iter = range(config["num_epochs"])
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

    # Track history if requested
    batch_losses: list[float] = []
    batch_test_accs: list[float] = []

    for _ in epoch_iter:
        epoch_loss = 0.0
        n_batches = 0

        for X, y in tqdm(train_dl, desc="Batches", unit="batch", leave=False):
            X = transf(X)
            X, y = X.to(device), y.float().to(device)

            if use_online_ogd:
                # Paper-faithful local OGD update
                # Process samples one at a time for true online learning
                # (or use batch with vectorized update)
                lr = lr_scheduler.step()
                out = model.online_update(X, y, lr=lr, vectorized=True)
                # Compute loss for logging (no gradient needed)
                with torch.no_grad():
                    loss = model.loss(out, y.unsqueeze(1))
            else:
                # Standard end-to-end backprop
                y = y.unsqueeze(1)
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

            if return_history:
                batch_losses.append(loss.item())
                # Evaluate on test set at end of each batch
                batch_acc = binary_accuracy(model, transf, test_dl, device=device)
                batch_test_accs.append(batch_acc)
                model.train()  # Switch back to training mode

        if verbose and n_batches > 0:
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            tqdm.write(f"Epoch loss: {avg_loss:.4f}")

    # Evaluate accuracy on test set
    test_acc = binary_accuracy(model, transf, test_dl, device=device)

    if return_history:
        history = {
            "batch_losses": batch_losses,
            "batch_test_accs": batch_test_accs,
        }
        return model, transf, test_acc, history

    return model, transf, test_acc
