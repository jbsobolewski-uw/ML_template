"""
models/neural.py
----------------
PyTorch-based neural network training with automatic device dispatch.

Device priority:
  NVIDIA_GPU -> 'cuda'
  INTEL_GPU  -> 'xpu' (Intel Extension for PyTorch) or 'cpu' fallback
  CPU        -> 'cpu'

The module ships a generic MLP (MLPRegressor / MLPClassifier) usable
as a drop-in for most tabular deep learning tasks.
For custom architectures, pass any nn.Module to train_neural().

Dependencies
------------
  - torch (always required)
  - intel_extension_for_pytorch (optional, Intel XPU support)
"""

import time
import logging
from typing import Callable, Optional

import numpy as np
from src.executor.accelerator import Backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_torch_device(
    backend: Backend,
    explicit_device: Optional[str] = None,
) -> "torch.device":
    """
    Map a Backend enum to a torch.device.

    Parameters
    ----------
    backend : Backend
    explicit_device : str or None
        If set, bypasses auto-resolution (e.g. 'cuda:1', 'cpu').

    Returns
    -------
    torch.device
    """
    import torch

    if explicit_device is not None:
        return torch.device(explicit_device)

    if backend == Backend.NVIDIA_GPU and torch.cuda.is_available():
        return torch.device("cuda")

    if backend == Backend.INTEL_GPU:
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
        except ImportError:
            logger.warning(
                "intel_extension_for_pytorch not installed. "
                "Intel GPU backend unavailable for PyTorch; using CPU."
            )

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generic MLP definitions
# ---------------------------------------------------------------------------

def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    task: str = "regression",
) -> "torch.nn.Module":
    """
    Build a fully-connected MLP.

    Parameters
    ----------
    input_dim : int
    output_dim : int
    hidden_dims : tuple of int
        Hidden layer widths.
    task : str
        'regression' or 'classification'.

    Returns
    -------
    torch.nn.Sequential
    """
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
        prev = h

    layers.append(nn.Linear(prev, output_dim))

    if task == "classification" and output_dim == 1:
        layers.append(nn.Sigmoid())
    # Multi-class softmax is applied via CrossEntropyLoss; no explicit layer needed.

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_neural(
    model: "torch.nn.Module",
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend = Backend.CPU,
    explicit_device: Optional[str] = None,
    task: str = "regression",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    loss_fn: Optional[Callable] = None,
    callbacks: Optional[list[Callable]] = None,
) -> dict:
    """
    Training loop for an arbitrary nn.Module.

    Parameters
    ----------
    model : nn.Module
        Uninitialized or pre-built PyTorch model.
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
    backend : Backend
    explicit_device : str or None
    task : str
        'regression' or 'classification'. Determines default loss.
    epochs : int
    batch_size : int
    lr : float
        Learning rate for Adam.
    loss_fn : callable or None
        Custom loss. If None, MSELoss (regression) or CrossEntropyLoss.
    callbacks : list of callable or None
        Each callback receives (epoch, loss_value) after every epoch.

    Returns
    -------
    dict with keys: 'model', 'device', 'history', 'duration_s'
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = resolve_torch_device(backend, explicit_device)
    logger.info("Neural training on device: %s", str(device).upper())

    # Cast data.
    X_t = torch.tensor(X, dtype=torch.float32)
    if task == "classification":
        y_t = torch.tensor(y, dtype=torch.long)
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    model = model.to(device)

    # Intel XPU optimisation (optional).
    if str(device) == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
        except ImportError:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn is None:
        loss_fn = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    history: list[float] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)

        if epoch % max(1, epochs // 10) == 0:
            logger.info("Epoch %d/%d | loss=%.6f", epoch, epochs, avg_loss)

        if callbacks:
            for cb in callbacks:
                cb(epoch, avg_loss)

    duration = time.perf_counter() - start
    logger.info(
        "Neural training complete | device=%s | epochs=%d | duration=%.2fs",
        device,
        epochs,
        duration,
    )

    return {
        "model": model.cpu(),
        "device": str(device),
        "history": history,
        "duration_s": round(duration, 4),
    }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_mlp_regressor(
    input_dim: int,
    output_dim: int = 1,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
) -> "torch.nn.Module":
    """Construct a MLP suitable for regression."""
    return _build_mlp(input_dim, output_dim, hidden_dims, task="regression")


def build_mlp_classifier(
    input_dim: int,
    n_classes: int,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
) -> "torch.nn.Module":
    """Construct a MLP suitable for classification."""
    return _build_mlp(input_dim, n_classes, hidden_dims, task="classification")