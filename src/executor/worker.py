"""
executor/worker.py
---------
Worker function executed in each child process spawned by pool.py.

Design constraints
------------------
- Must be importable on Windows (spawn method requires top-level picklability).
- No module-level side effects (no patch_sklearn() at import time).
- Receives a WorkerTask dataclass; returns a WorkerResult dataclass.
- Backend selection is re-confirmed inside the worker because child processes
  inherit no state from the parent under 'spawn'.
"""

import os
import time
import traceback
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .accelerator import (
    Backend,
    HardwareProfile,
    detect_hardware,
    select_backend,
    LARGE_WORKLOAD_THRESHOLD,
)
from .config import Config
from src.utils.logging import get_worker_logger


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class WorkerTask:
    """
    Encapsulates all inputs a worker process needs.

    Attributes
    ----------
    task_id : int or str
        Unique identifier for logging and result correlation.
    X : np.ndarray
    y : np.ndarray
    model_type : str
        'sklearn' or 'neural'.
    estimator_or_model : Any
        Unfitted sklearn estimator OR nn.Module instance.
    config : Config
    extra_kwargs : dict
        Forwarded verbatim to the model runner (e.g. epochs, batch_size).
    """
    task_id: Any
    X: np.ndarray
    y: np.ndarray
    model_type: str
    estimator_or_model: Any
    config: Config
    extra_kwargs: dict = field(default_factory=dict)


@dataclass
class WorkerResult:
    """
    Result returned from a worker process.

    Attributes
    ----------
    task_id : Any
    success : bool
    backend_used : str
    duration_s : float
    payload : dict
        Model, history, or other artefacts from the runner.
    error : str or None
        Traceback string if success=False.
    """
    task_id: Any
    success: bool
    backend_used: str
    duration_s: float
    payload: dict = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------

def run_worker(task: WorkerTask) -> WorkerResult:
    """
    Execute a single ML training task inside a child process.

    Parameters
    ----------
    task : WorkerTask

    Returns
    -------
    WorkerResult
    """
    log = get_worker_logger(__name__, task.config.log_level)
    log.info("[Task %s] Worker started | PID=%d", task.task_id, os.getpid())

    wall_start = time.perf_counter()

    try:
        # Re-detect hardware inside the spawned process.
        profile: HardwareProfile = detect_hardware()

        n_samples, n_features = task.X.shape
        backend: Backend = select_backend(
            profile=profile,
            n_samples=n_samples,
            n_features=n_features,
            force_backend=task.config.force_backend,
        )

        # Override large_workload_threshold from config.
        import accelerator as _acc
        _acc.LARGE_WORKLOAD_THRESHOLD = task.config.large_workload_threshold

        log.info(
            "[Task %s] backend=%s | shape=(%d, %d)",
            task.task_id,
            backend.name,
            n_samples,
            n_features,
        )

        if task.model_type == "sklearn":
            payload = _run_sklearn(task, backend, log)
        elif task.model_type == "neural":
            payload = _run_neural(task, backend, log)
        else:
            raise ValueError(f"Unknown model_type: {task.model_type!r}")

        duration = time.perf_counter() - wall_start
        log.info("[Task %s] Done in %.2fs", task.task_id, duration)

        return WorkerResult(
            task_id=task.task_id,
            success=True,
            backend_used=payload.get("backend_used", backend.name),
            duration_s=round(duration, 4),
            payload=payload,
        )

    except Exception:
        duration = time.perf_counter() - wall_start
        tb = traceback.format_exc()
        log.error("[Task %s] FAILED:\n%s", task.task_id, tb)
        return WorkerResult(
            task_id=task.task_id,
            success=False,
            backend_used="UNKNOWN",
            duration_s=round(duration, 4),
            error=tb,
        )


# ---------------------------------------------------------------------------
# Internal dispatch helpers (not part of public API)
# ---------------------------------------------------------------------------

def _run_sklearn(task: WorkerTask, backend: Backend, log: logging.Logger) -> dict:
    from src.models.sklearn_models import fit_sklearn
    result = fit_sklearn(
        estimator=task.estimator_or_model,
        X=task.X,
        y=task.y,
        backend=backend,
        sklearn_n_jobs=task.config.sklearn_n_jobs,
    )
    return result


def _run_neural(task: WorkerTask, backend: Backend, log: logging.Logger) -> dict:
    from src.models.neural import train_neural

    # Extract any optional training parameters passed via extra_kwargs
    extra_kwargs = getattr(task, "extra_kwargs", {}) or {}
    if extra_kwargs is None:
        extra_kwargs = {}

    result = train_neural(
        model=task.estimator_or_model,
        X=task.X,
        y=task.y,
        backend=backend,
        **extra_kwargs
    )

    # Translate the actual torch hardware target back into the reporting enum string
    actual_device = str(result.get("device", "CPU")).lower()
    if "cuda" in actual_device:
        result["backend_used"] = Backend.NVIDIA_GPU.name
    elif "xpu" in actual_device:
        result["backend_used"] = Backend.INTEL_GPU.name
    else:
        result["backend_used"] = Backend.CPU.name

    return result