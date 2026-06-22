"""
executor package public API.

Imports here define what is considered stable and user-facing.
Internal submodule paths (accelerator, pool, worker) are an
implementation detail and should not be imported directly by callers.
"""

from ml_framework.executor.accelerator import Backend, HardwareProfile, detect_hardware, select_backend
from ml_framework.executor.config import Config
from ml_framework.executor.worker import WorkerTask, WorkerResult, run_worker
from ml_framework.executor.pool import run_parallel

__all__ = [
    "Backend",
    "HardwareProfile",
    "detect_hardware",
    "select_backend",
    "Config",
    "WorkerTask",
    "WorkerResult",
    "run_worker",
    "run_parallel"
]
