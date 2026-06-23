"""
executor/worker.py
------------------
Worker entry point executed in each spawned child process.

Changes from previous version
------------------------------
- WorkerTask.model_type (str) removed. Replaced by WorkerTask.model (MLModel).
  Workers call task.model.run(...) directly — no string dispatch.
- WorkerTask.X / .y replaced by WorkerTask.X_handle / .y_handle
  (SharedArrayHandle). Arrays are reconstructed zero-copy inside the worker
  via to_array(). No array data crosses the pickle pipe.
- extra_kwargs removed — training hyperparameters now live inside the MLModel
  subclass (e.g. NeuralModel.epochs, .batch_size, .lr).

Design constraints (unchanged)
-------------------------------
- Importable on Windows (spawn picklability requirement).
- No module-level side effects.
- BackendUnavailableError and DeviceUnavailableError surface as failed
  WorkerResult, not pool-level crash.
"""

import os
import time
import traceback
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ml_framework.executor.accelerator import (
    Backend,
    HardwareProfile,
    BackendUnavailableError,
    detect_hardware,
    select_backend,
)
from ml_framework.executor.config import Config
from ml_framework.executor.shared_memory import SharedArrayHandle
from ml_framework.models.base import MLModel
from ml_framework.utils.logging import get_worker_logger


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class WorkerTask:
    """
    Encapsulates all inputs a spawned worker needs.

    Attributes
    ----------
    task_id : Any
        Unique identifier for logging and result correlation.
    X_handle : SharedArrayHandle
        Feature matrix, zero-copy shared memory descriptor.
    y_handle : SharedArrayHandle
        Target vector. May be a none-sentinel for unsupervised tasks.
    model : MLModel
        Carries the estimator/module and all training hyperparameters.
        Dispatches itself via model.run().
    config : Config
    """
    task_id: Any
    X_handle: SharedArrayHandle
    y_handle: SharedArrayHandle
    model: MLModel
    config: Config


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
        Model, history, or other artefacts from MLModel.run().
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
        # Reconstruct arrays from shared memory — zero copy.
        X = task.X_handle.to_array()
        y = task.y_handle.to_array()

        # Re-detect hardware inside the spawned process (spawn inherits no state).
        profile: HardwareProfile = detect_hardware()

        n_samples, n_features = X.shape
        backend: Backend = select_backend(
            profile=profile,
            n_samples=n_samples,
            n_features=n_features,
            force_backend=task.config.force_backend,
            large_workload_threshold=task.config.large_workload_threshold,
        )

        log.info(
            "[Task %s] backend=%s | shape=(%d, %d) | model=%s",
            task.task_id,
            backend.name,
            n_samples,
            n_features,
            type(task.model).__name__,
        )

        payload = task.model.run(X, y, backend, task.config)

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
