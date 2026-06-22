"""
executor/worker.py
------------------
Worker function executed in each child process spawned by pool.py.

Design constraints
------------------
- Must be importable on Windows (spawn requires top-level picklability).
- No module-level side effects.
- Receives WorkerTask; returns WorkerResult.
- Backend selection re-runs inside the worker: child processes under 'spawn'
  inherit no runtime state from the parent.
- BackendUnavailableError from accelerator and DeviceUnavailableError from
  neural.py are both caught and surfaced as a failed WorkerResult, not a
  pool-level crash, so remaining tasks continue unaffected.
"""

import os
import time
import traceback
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ml_framework.executor.accelerator import (
    Backend,
    HardwareProfile,
    BackendUnavailableError,
    detect_hardware,
    select_backend,
)
from ml_framework.executor.config import Config
from ml_framework.utils.logging import get_worker_logger


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

    BackendUnavailableError (detection-layer) and DeviceUnavailableError
    (runtime-layer) are both caught here and returned as a failed WorkerResult
    so the pool continues processing remaining tasks.

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
        profile: HardwareProfile = detect_hardware()

        n_samples, n_features = task.X.shape

        # Pass large_workload_threshold from Config so the worker's threshold
        # matches what the user configured, not the module-level default.
        backend: Backend = select_backend(
            profile=profile,
            n_samples=n_samples,
            n_features=n_features,
            force_backend=task.config.force_backend,
            large_workload_threshold=task.config.large_workload_threshold,
        )

        log.info(
            "[Task %s] backend=%s | shape=(%d, %d)",
            task.task_id,
            backend.name,
            n_samples,
            n_features,
        )

        if task.model_type == "sklearn":
            payload = _run_sklearn(task, backend)
        elif task.model_type == "neural":
            payload = _run_neural(task, backend)
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

    except BackendUnavailableError as exc:
        duration = time.perf_counter() - wall_start
        msg = f"BackendUnavailableError: {exc}"
        log.error("[Task %s] %s", task.task_id, msg)
        return WorkerResult(
            task_id=task.task_id,
            success=False,
            backend_used="UNAVAILABLE",
            duration_s=round(duration, 4),
            error=msg,
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
# Internal dispatch helpers
# ---------------------------------------------------------------------------

def _run_sklearn(task: WorkerTask, backend: Backend) -> dict:
    from ml_framework.models.sklearn_models import fit_sklearn
    return fit_sklearn(
        estimator=task.estimator_or_model,
        X=task.X,
        y=task.y,
        backend=backend,
        sklearn_n_jobs=task.config.sklearn_n_jobs,
    )


def _run_neural(task: WorkerTask, backend: Backend) -> dict:
    from ml_framework.models.neural import train_neural
    return train_neural(
        model=task.estimator_or_model,
        X=task.X,
        y=task.y,
        backend=backend,
        explicit_device=task.config.torch_device,
        **(task.extra_kwargs or {}),
    )
