"""
ml_framework
============
Public API. Everything a caller needs is importable directly from this package.

Typical usage::

    from ml_framework import Config, WorkerTask, run_parallel, Backend
    from ml_framework import build_mlp_regressor, fit_sklearn
    from ml_framework import setup_logging
"""

# Executor layer
from ml_framework.executor import (
    Backend,
    HardwareProfile,
    detect_hardware,
    select_backend,
    Config,
    WorkerTask,
    WorkerResult,
    run_worker,
    run_parallel,
)

# Model runners
from ml_framework.models import (
    fit_sklearn,
    train_neural,
    resolve_torch_device,
    build_mlp_regressor,
    build_mlp_classifier,
)

# Logging helpers
from ml_framework.utils import setup_logging, get_worker_logger

__all__ = [
    # executor
    "Backend",
    "HardwareProfile",
    "detect_hardware",
    "select_backend",
    "Config",
    "WorkerTask",
    "WorkerResult",
    "run_worker",
    "run_parallel",
    # models
    "fit_sklearn",
    "train_neural",
    "resolve_torch_device",
    "build_mlp_regressor",
    "build_mlp_classifier",
    # utils
    "setup_logging",
    "get_worker_logger",
]
