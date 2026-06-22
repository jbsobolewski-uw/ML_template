"""executor package."""
from .accelerator import (
    Backend,
    HardwareProfile,
    detect_hardware,
    select_backend,
    LARGE_WORKLOAD_THRESHOLD,
)
from .config import Config
from .pool import run_parallel
from .worker import WorkerTask, WorkerResult, run_worker

__all__ = [
    "Backend",
    "HardwareProfile",
    "detect_hardware",
    "select_backend",
    "LARGE_WORKLOAD_THRESHOLD",
    "Config",
    "run_parallel",
    "WorkerTask",
    "WorkerResult",
    "run_worker",
]