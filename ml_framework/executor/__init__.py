""" executor package public API. """

from ml_framework.executor.accelerator import (
    Backend,
    HardwareProfile,
    BackendUnavailableError,
    detect_hardware,
    select_backend,
)
from ml_framework.executor.config import Config
from ml_framework.executor.shared_memory import SharedArrayHandle
from ml_framework.executor.worker import WorkerTask, WorkerResult, run_worker
from ml_framework.executor.pool import run_parallel, run_parallel_simple, make_task
from ml_framework.executor.scheduler import SchedulingPolicy, ResourceRegistry

__all__ = [
    "Backend",
    "HardwareProfile",
    "BackendUnavailableError",
    "detect_hardware",
    "select_backend",
    "Config",
    "SharedArrayHandle",
    "WorkerTask",
    "WorkerResult",
    "run_worker",
    "run_parallel",
    "run_parallel_simple",
    "make_task",
    "SchedulingPolicy",
    "ResourceRegistry",
]


