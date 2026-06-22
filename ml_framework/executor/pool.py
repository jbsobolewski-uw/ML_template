"""
executor/pool.py
-------
Cross-OS multiprocessing pool management.

Key design decisions
--------------------
- Uses 'spawn' start method on all platforms for consistency and to avoid
  CUDA/OpenCL context fork issues on Linux.
- Sets mp.set_executable() to the current venv interpreter so that
  child processes use the same environment (critical on Windows).
- Pool size defaults to min(n_tasks, floor(cpu_count * 2/3), 1).
- Exposes a single public function: run_parallel(tasks, config).
"""

import os
import sys
import logging
import multiprocessing as mp
import time
from typing import Sequence

from .config import Config
from .worker import WorkerTask, WorkerResult, run_worker

logger = logging.getLogger(__name__)


def _resolve_pool_size(n_tasks: int, requested: int | None) -> int:
    """
    Compute the actual pool size.

    Parameters
    ----------
    n_tasks : int
    requested : int or None
        User-specified pool size. None triggers auto-calculation.

    Returns
    -------
    int
    """
    cpu_count = os.cpu_count() or 1
    auto = max(1, cpu_count * 2 // 3)
    size = requested if requested is not None else auto
    return max(1, min(size, n_tasks))


def run_parallel(
    tasks: Sequence[WorkerTask],
    config: Config,
) -> list[WorkerResult]:
    """
    Execute a list of WorkerTask instances concurrently.

    Parameters
    ----------
    tasks : sequence of WorkerTask
    config : Config

    Returns
    -------
    list of WorkerResult, order matches input tasks.
    """
    if not tasks:
        return []

    pool_size = _resolve_pool_size(len(tasks), config.pool_size)

    # Ensure child processes use the correct interpreter.
    mp.set_executable(sys.executable)

    # 'spawn' is safe across Linux and Windows; avoids fork + CUDA issues.
    ctx = mp.get_context("spawn")

    logger.info(
        "Launching pool | tasks=%d | pool_size=%d | interpreter=%s",
        len(tasks),
        pool_size,
        sys.executable,
    )

    wall_start = time.perf_counter()

    with ctx.Pool(processes=pool_size) as pool:
        results: list[WorkerResult] = pool.map(run_worker, tasks)

    wall_duration = time.perf_counter() - wall_start
    successes = sum(1 for r in results if r.success)
    logger.info(
        "Pool complete | %d/%d succeeded | wall_time=%.2fs",
        successes,
        len(results),
        wall_duration,
    )

    return results