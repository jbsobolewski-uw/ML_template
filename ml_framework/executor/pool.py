"""
executor/pool.py
----------------
Main dispatch entry point: run_parallel().

Responsibilities
----------------
1. Allocate SharedMemory blocks for X and y arrays (one block per unique array
   object; deduplication by id() avoids double-allocation when multiple tasks
   share the same dataset).
2. Build SharedArrayHandle descriptors and attach them to WorkerTask copies.
3. Invoke run_scheduled() (GPU-aware scheduler) or fall back to a simple
   pool.map if no GPU is present.
4. Unlink all shared memory blocks after results are collected.

Shared memory lifetime
----------------------
Blocks are allocated in this function, kept alive until after the pool/executor
joins, then unlinked. Workers attach and detach within their run() calls.
"""

import os
import sys
import logging
import multiprocessing as mp
import time
from typing import Sequence

from ml_framework.executor.accelerator import detect_hardware, HardwareProfile
from ml_framework.executor.config import Config
from ml_framework.executor.scheduler import (
    SchedulingPolicy,
    run_scheduled,
    _resolve_pool_size,
)
from ml_framework.executor.shared_memory import SharedArrayHandle
from ml_framework.executor.worker import WorkerTask, WorkerResult, run_worker

logger = logging.getLogger(__name__)


def run_parallel(
    tasks: Sequence[WorkerTask],
    config: Config,
    policy: SchedulingPolicy = SchedulingPolicy.FIFO,
) -> list[WorkerResult]:
    """
    Execute a list of WorkerTask instances with shared-memory data transfer
    and GPU-aware scheduling.

    Parameters
    ----------
    tasks : sequence of WorkerTask
        Tasks must already have X_handle and y_handle set (use make_task()
        or build handles manually via SharedArrayHandle.from_array()).
    config : Config
        Pool-level configuration. Per-task Config is read from task.config.
    policy : SchedulingPolicy
        Task ordering policy for the scheduler.

    Returns
    -------
    list of WorkerResult, order matches input tasks.
    """
    if not tasks:
        return []

    # Ensure child processes use the correct venv interpreter.
    mp.set_executable(sys.executable)

    profile: HardwareProfile = detect_hardware()

    pool_size = _resolve_pool_size(len(tasks), config.pool_size)
    logger.info(
        "run_parallel | tasks=%d | pool_size=%d | policy=%s | interpreter=%s",
        len(tasks),
        pool_size,
        policy.name,
        sys.executable,
    )

    wall_start = time.perf_counter()
    results = run_scheduled(tasks, config, profile, policy)
    wall_duration = time.perf_counter() - wall_start

    successes = sum(1 for r in results if r.success)
    logger.info(
        "run_parallel complete | %d/%d succeeded | wall_time=%.2fs",
        successes,
        len(results),
        wall_duration,
    )
    return results


def make_task(
    task_id: object,
    X,
    y,
    model,
    config: Config,
) -> tuple["WorkerTask", list[SharedArrayHandle]]:
    """
    Convenience constructor: allocates shared memory for X and y, returns
    a WorkerTask and the list of handles to unlink after run_parallel().

    Parameters
    ----------
    task_id : Any
    X : np.ndarray
    y : np.ndarray or None
    model : MLModel
    config : Config

    Returns
    -------
    (WorkerTask, [X_handle, y_handle])
        Caller must call handle.unlink() on each handle after run_parallel().

    Example
    -------
        task, handles = make_task("rf", X, y, SklearnModel(rf), Config())
        try:
            results = run_parallel([task], Config())
        finally:
            for h in handles:
                h.unlink()
    """
    import numpy as np
    X_handle = SharedArrayHandle.from_array(np.asarray(X))
    y_handle = SharedArrayHandle.from_array(np.asarray(y) if y is not None else None)

    task = WorkerTask(
        task_id=task_id,
        X_handle=X_handle,
        y_handle=y_handle,
        model=model,
        config=config,
    )
    return task, [X_handle, y_handle]


def run_parallel_simple(
    task_specs: Sequence[tuple],
    config: Config,
    policy: SchedulingPolicy = SchedulingPolicy.FIFO,
) -> list[WorkerResult]:
    """
    High-level wrapper: accepts raw (task_id, X, y, model) tuples, handles
    shared memory allocation and cleanup internally.

    Parameters
    ----------
    task_specs : sequence of (task_id, X, y, model) tuples
    config : Config
    policy : SchedulingPolicy

    Returns
    -------
    list of WorkerResult

    Example
    -------
        results = run_parallel_simple([
            ("rf", X_train, y_train, SklearnModel(RandomForestRegressor())),
            ("mlp", X_train, y_train, NeuralModel(build_mlp_regressor(32))),
        ], Config())
    """
    tasks = []
    all_handles: list[SharedArrayHandle] = []

    for spec in task_specs:
        task_id, X, y, model = spec
        task, handles = make_task(task_id, X, y, model, config)
        tasks.append(task)
        all_handles.extend(handles)

    try:
        return run_parallel(tasks, config, policy)
    finally:
        for handle in all_handles:
            handle.unlink()
