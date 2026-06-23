"""
models/neural.py
----------------
PyTorch neural network wrapper implementing the MLModel ABC.

Device priority:
  NVIDIA_GPU -> 'cuda'
  INTEL_GPU  -> 'xpu' (intel_extension_for_pytorch) or CPU fallback
  CPU        -> 'cpu'

DeviceUnavailableError is raised (not swallowed) when NVIDIA_GPU is selected
but CUDA is absent at training time — see resolve_torch_device().

Usage
-----
    from ml_framework.models.neural import NeuralModel, build_mlp_regressor

    task = WorkerTask(
        task_id="mlp",
        X_handle=...,
        y_handle=...,
        model=NeuralModel(
            build_mlp_regressor(input_dim=32),
            epochs=50,
            batch_size=256,
            lr=1e-3,
            task="regression",
        ),
        config=Config(),
    )
"""

import time
import logging
from typing import Callable, Optional

import numpy as np

from ml_framework.executor.accelerator import Backend
from ml_framework.executor.config import Config
from ml_framework.models.base import MLModel

logger = logging.getLogger(__name__)


class DeviceUnavailableError(RuntimeError):
    """
    Raised when the torch device required by a Backend cannot be acquired at
    training time. Distinct from BackendUnavailableError (detection-layer).
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
        If set, returns torch.device(explicit_device) with no availability check.
    strict : bool
        True (default): raises DeviceUnavailableError if CUDA is requested but absent.
        False: silently returns CPU. Use False only for testing.

    Returns
    -------
    torch.device

    Raises
    ------
    DeviceUnavailableError
    """
    import torch

    if explicit_device is not None:
        return torch.device(explicit_device)

    if backend == Backend.NVIDIA_GPU:
        if torch.cuda.is_available():
            return torch.device("cuda")
        msg = (
            "Backend.NVIDIA_GPU selected but torch.cuda.is_available() is False "
            "at training time. Check CUDA_VISIBLE_DEVICES and driver state."
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
        logger.warning(
            "Backend.INTEL_GPU for PyTorch: XPU unavailable "
            "(intel_extension_for_pytorch missing or no XPU device). "
            "Using CPU for this neural task."
        )
        return torch.device("cpu")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLModel implementation
# ---------------------------------------------------------------------------

class NeuralModel(MLModel):
    """
    Wraps an nn.Module and training hyperparameters for dispatch via WorkerTask.

    Parameters
    ----------
    module : torch.nn.Module
        Unfitted model. Must be picklable.
    task : str
        'regression' or 'classification'.
    epochs : int
    batch_size : int
    lr : float
    loss_fn : callable or None
        Custom loss. None = MSELoss (regression) or CrossEntropyLoss.
    callbacks : list of callable or None
        Each receives (epoch: int, avg_loss: float) after each epoch.
    """

    def __init__(
        self,
        module: "torch.nn.Module",
        task: str = "regression",
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[list[Callable]] = None,
    ) -> None:
        self.module = module
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []

    def run(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        backend: Backend,
        config: Config,
    ) -> dict:
        """
        Train the module and return results dict.

        Returns
        -------
        dict with keys: 'model', 'backend_used', 'device', 'history', 'duration_s'
        """
        return train_neural(
            model=self.module,
            X=X,
            y=y,
            backend=backend,
            explicit_device=config.torch_device,
            task=self.task,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            loss_fn=self.loss_fn,
            callbacks=self.callbacks,
        )


# ---------------------------------------------------------------------------
# Core training loop (used by NeuralModel.run and callable directly)
# ---------------------------------------------------------------------------

def train_neural(
    model: "torch.nn.Module",
    X: np.ndarray,
    y: Optional[np.ndarray],
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
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray or None
    backend : Backend
    explicit_device : str or None
    task : str
    epochs : int
    batch_size : int
    lr : float
    loss_fn : callable or None
    callbacks : list of callable or None

    Returns
    -------
    dict: 'model' (on CPU), 'device', 'history', 'duration_s', 'backend_used'

    Raises
    ------
    DeviceUnavailableError
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

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
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=False
    )

    model = model.to(device)

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

        for cb in callbacks or []:
            cb(epoch, avg_loss)

    duration = time.perf_counter() - start
    logger.info(
        "Neural training complete | device=%s | epochs=%d | duration=%.2fs",
        str(device).upper(),
        epochs,
        duration,
    )

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

def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    task: str,
) -> "torch.nn.Module":
    import torch.nn as nn

    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if task == "classification" and output_dim == 1:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


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
