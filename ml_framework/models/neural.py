"""
models/neural.py
----------------
PyTorch-based neural network training with automatic device dispatch.

Device priority:
  NVIDIA_GPU -> 'cuda'
  INTEL_GPU  -> 'xpu' (Intel Extension for PyTorch) or 'cpu' fallback
  CPU        -> 'cpu'

resolve_torch_device() raises DeviceUnavailableError when an explicitly
requested accelerator cannot be satisfied, rather than silently falling
back to CPU. Silent fallback masks misconfiguration and makes profiling
results misleading.

Dependencies
------------
  - torch (always required)
  - intel_extension_for_pytorch (optional, Intel XPU support)
"""

import time
import logging
from typing import Callable, Optional

import numpy as np
from ml_framework.executor.accelerator import Backend

logger = logging.getLogger(__name__)


class DeviceUnavailableError(RuntimeError):
    """
    Raised when the torch device required by a Backend cannot be acquired.
    Distinct from BackendUnavailableError (accelerator-layer) so callers
    can distinguish detection-time vs. runtime failures.
    """


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_torch_device(
    backend: Backend,
    explicit_device: Optional[str] = None,
    strict: bool = True,
) -> "torch.device":
    """
    Map a Backend enum to a torch.device.

    Parameters
    ----------
    backend : Backend
    explicit_device : str or None
        If set, returns torch.device(explicit_device) directly without
        any availability check. Caller assumes full responsibility.
    strict : bool
        If True (default), raises DeviceUnavailableError when the backend
        device cannot be acquired. If False, silently falls back to CPU.
        Use strict=False only in testing or exploratory contexts.

    Returns
    -------
    torch.device

    Raises
    ------
    DeviceUnavailableError
        When strict=True and the requested backend device is unavailable.
    """
    import torch

    if explicit_device is not None:
        return torch.device(explicit_device)

    if backend == Backend.NVIDIA_GPU:
        if torch.cuda.is_available():
            return torch.device("cuda")
        msg = (
            "Backend.NVIDIA_GPU was selected but torch.cuda.is_available() "
            "returned False at training time. This indicates a mismatch between "
            "hardware detection (accelerator.py) and the runtime CUDA context. "
            "Check driver initialisation order and CUDA_VISIBLE_DEVICES."
        )
        if strict:
            raise DeviceUnavailableError(msg)
        logger.warning("%s Falling back to CPU.", msg)
        return torch.device("cpu")

    if backend == Backend.INTEL_GPU:
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return torch.device("xpu")
        except ImportError:
            pass
        # Intel XPU not available: this is a soft fallback (ipex is optional).
        logger.warning(
            "Backend.INTEL_GPU requested for PyTorch but XPU is unavailable "
            "(intel_extension_for_pytorch missing or no XPU device). "
            "Falling back to CPU for this neural task. "
            "sklearn tasks will still use sklearnex iGPU offload."
        )
        return torch.device("cpu")

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
    Build a fully-connected MLP with BatchNorm and ReLU activations.

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
    # Multi-class: softmax applied implicitly via CrossEntropyLoss.

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
        Unfitted PyTorch model instance.
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
    backend : Backend
    explicit_device : str or None
        Passed directly to resolve_torch_device; bypasses all checks.
    task : str
        'regression' or 'classification'. Determines default loss function.
    epochs : int
    batch_size : int
    lr : float
        Learning rate for Adam.
    loss_fn : callable or None
        Custom loss. Defaults to MSELoss (regression) or CrossEntropyLoss.
    callbacks : list of callable or None
        Each callback receives (epoch: int, avg_loss: float).

    Returns
    -------
    dict
        Keys: 'model' (moved to CPU), 'device' (str), 'history' (list[float]),
        'duration_s' (float), 'backend_used' (str).

    Raises
    ------
    DeviceUnavailableError
        If backend=NVIDIA_GPU but CUDA is not available at runtime.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # strict=True: fail loudly if the selected backend cannot be honoured.
    device = resolve_torch_device(backend, explicit_device, strict=True)
    logger.info("Neural training on device: %s", str(device).upper())

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

    # Intel XPU: apply ipex optimisation pass when available.
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
        str(device).upper(),
        epochs,
        duration,
    )

    # Resolve backend_used string from actual device, not from the Backend enum.
    actual = str(device).lower()
    if "cuda" in actual:
        backend_used = Backend.NVIDIA_GPU.name
    elif "xpu" in actual:
        backend_used = Backend.INTEL_GPU.name
    else:
        backend_used = Backend.CPU.name

    return {
        "model": model.cpu(),
        "device": str(device),
        "history": history,
        "duration_s": round(duration, 4),
        "backend_used": backend_used,
    }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_mlp_regressor(
    input_dim: int,
    output_dim: int = 1,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
) -> "torch.nn.Module":
    """Construct an MLP suitable for regression."""
    return _build_mlp(input_dim, output_dim, hidden_dims, task="regression")


def build_mlp_classifier(
    input_dim: int,
    n_classes: int,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
) -> "torch.nn.Module":
    """Construct an MLP suitable for classification."""
    return _build_mlp(input_dim, n_classes, hidden_dims, task="classification")
