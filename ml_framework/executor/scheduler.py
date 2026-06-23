"""
executor/scheduler.py
---------------------
GPU-aware task scheduler with per-device mutex locking and pluggable
ordering policies.

Problem
-------
pool.map dispatches tasks in FIFO order without regard for GPU contention.
With N tasks targeting the same GPU and a pool_size > 1, multiple workers
attempt simultaneous GPU use, causing CUDA context-switching overhead that
can exceed the compute time itself for small/medium tasks.

Solution
--------
A ResourceRegistry holds one threading.Lock per detected GPU device index.
The scheduler dispatches tasks through a managed thread loop:

  1. Order the pending queue by SchedulingPolicy.
  2. For the next task:
     - If it targets a GPU backend: try-acquire the device lock.
       - Lock acquired  → submit to ProcessPoolExecutor on the GPU-capable config.
       - Lock not free  → downgrade task to CPU temporarily and submit immediately.
         When the GPU lock becomes free, remaining GPU tasks will be submitted
         without downgrade.
  3. On task completion, release the device lock (if held).

GPU slot count
--------------
By default, one slot per detected CUDA device is registered. Multi-GPU
machines get one lock per device; tasks can target specific devices via
Config.torch_device ('cuda:0', 'cuda:1', ...) or use the generic 'cuda'
(binds to device 0 slot).

Policies
--------
FIFO           - submission order (default, deterministic)
BIGGEST_FIRST  - largest workload (n_samples * n_features) first;
                 maximises GPU utilisation by front-loading heavy tasks.
SMALLEST_FIRST - smallest workload first; minimises queue wait for light tasks.

Workload size is taken from SharedArrayHandle.shape when available, falling
back to MLModel.workload_elements.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence, Optional

from ml_framework.executor.accelerator import Backend, HardwareProfile
from ml_framework.executor.config import Config
from ml_framework.executor.worker import WorkerTask, WorkerResult, run_worker

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    FIFO = auto()
    BIGGEST_FIRST = auto()
    SMALLEST_FIRST = auto()


# ---------------------------------------------------------------------------
# Resource registry
# ---------------------------------------------------------------------------

@dataclass
class _DeviceSlot:
    """One GPU device slot: a lock + bookkeeping."""
    device_index: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_task_id: Optional[object] = None


class ResourceRegistry:
    """
    Tracks available GPU device slots.

    One slot is registered per CUDA device detected. Intel GPU is treated as
    a single shared slot (index -1) since sklearnex does not expose multi-GPU
    addressing.

    Parameters
    ----------
    profile : HardwareProfile
        Result of detect_hardware() run in the parent process.
    """

    def __init__(self, profile: HardwareProfile) -> None:
        self._slots: dict[int, _DeviceSlot] = {}

        if profile.cuda_available:
            try:
                import torch
                n = torch.cuda.device_count()
            except ImportError:
                n = 1
            for i in range(max(n, 1)):
                self._slots[i] = _DeviceSlot(device_index=i)
            logger.debug("ResourceRegistry: registered %d CUDA slot(s).", n)

        if profile.intel_gpu_available:
            # Intel iGPU treated as a single slot, index -1.
            self._slots[-1] = _DeviceSlot(device_index=-1)
            logger.debug("ResourceRegistry: registered Intel GPU slot.")

    @property
    def has_gpu_slots(self) -> bool:
        return bool(self._slots)

    def try_acquire(self, backend: Backend, torch_device: Optional[str]) -> Optional[int]:
        """
        Non-blocking attempt to acquire a GPU slot matching backend/device.

        Parameters
        ----------
        backend : Backend
        torch_device : str or None
            Explicit device string from Config (e.g. 'cuda:1').

        Returns
        -------
        int or None
            Device index if acquired, None if no slot is free.
        """
        if backend == Backend.NVIDIA_GPU:
            target_idx = _parse_cuda_index(torch_device)
            if target_idx is not None and target_idx in self._slots:
                candidates = [target_idx]
            else:
                candidates = [i for i in self._slots if i >= 0]
            for idx in candidates:
                if self._slots[idx].lock.acquire(blocking=False):
                    return idx
            return None

        if backend == Backend.INTEL_GPU:
            if -1 in self._slots and self._slots[-1].lock.acquire(blocking=False):
                return -1
            return None

        return None  # CPU tasks do not use slots.

    def release(self, slot_index: int) -> None:
        """Release a previously acquired slot."""
        if slot_index in self._slots:
            try:
                self._slots[slot_index].lock.release()
            except RuntimeError:
                pass  # already released


def _parse_cuda_index(device_str: Optional[str]) -> Optional[int]:
    """Parse 'cuda:N' → N, 'cuda' → 0, None → None."""
    if device_str is None:
        return None
    if device_str.startswith("cuda:"):
        try:
            return int(device_str.split(":")[1])
        except (IndexError, ValueError):
            return 0
    if device_str == "cuda":
        return 0
    return None


# ---------------------------------------------------------------------------
# Workload size estimation
# ---------------------------------------------------------------------------

def _workload_size(task: WorkerTask) -> int:
    """
    Estimate task workload in elements for scheduling purposes.
    Uses SharedArrayHandle.shape if X_handle is present, else MLModel hint.
    """
    if hasattr(task, "X_handle") and task.X_handle is not None:
        shape = task.X_handle.shape
        if len(shape) >= 2:
            return shape[0] * shape[1]
        if len(shape) == 1:
            return shape[0]
    if hasattr(task, "model") and hasattr(task.model, "workload_elements"):
        return task.model.workload_elements
    return 0


# ---------------------------------------------------------------------------
# Ordered queue
# ---------------------------------------------------------------------------

def _ordered_tasks(
    tasks: list[WorkerTask],
    policy: SchedulingPolicy,
) -> list[WorkerTask]:
    if policy == SchedulingPolicy.BIGGEST_FIRST:
        return sorted(tasks, key=_workload_size, reverse=True)
    if policy == SchedulingPolicy.SMALLEST_FIRST:
        return sorted(tasks, key=_workload_size)
    return list(tasks)  # FIFO


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def run_scheduled(
    tasks: Sequence[WorkerTask],
    config: Config,
    profile: HardwareProfile,
    policy: SchedulingPolicy = SchedulingPolicy.FIFO,
) -> list[WorkerResult]:
    """
    Execute tasks via a GPU-aware scheduler.

    GPU tasks compete for per-device locks. A task that cannot immediately
    acquire its target GPU device is submitted with a CPU-overridden config
    so it does not stall the queue. GPU slots are released as tasks complete,
    allowing subsequent GPU tasks to proceed without downgrade.

    Parameters
    ----------
    tasks : sequence of WorkerTask
    config : Config
        Pool-level config. Per-task Config is preserved for backend selection.
    profile : HardwareProfile
    policy : SchedulingPolicy

    Returns
    -------
    list of WorkerResult, order matches input tasks.
    """
    if not tasks:
        return []

    registry = ResourceRegistry(profile)
    ordered = _ordered_tasks(list(tasks), policy)

    # Map future → (original_task_index, acquired_slot_index_or_None)
    future_meta: dict[Future, tuple[int, Optional[int]]] = {}
    results: dict[int, WorkerResult] = {}

    # Index map: task_id → original position for result ordering.
    task_index = {id(t): i for i, t in enumerate(tasks)}

    max_workers = _resolve_pool_size(len(tasks), config.pool_size)
    logger.info(
        "Scheduler starting | tasks=%d | workers=%d | policy=%s",
        len(tasks),
        max_workers,
        policy.name,
    )

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending = list(ordered)

        while pending or future_meta:
            # Submit as many tasks as executor slots allow.
            while pending and len(future_meta) < max_workers:
                task = pending.pop(0)
                orig_idx = task_index[id(task)]
                submit_task, slot = _prepare_submission(task, registry, profile)
                fut = executor.submit(run_worker, submit_task)
                future_meta[fut] = (orig_idx, slot)
                logger.debug(
                    "Submitted task %s | slot=%s | backend=%s",
                    task.task_id,
                    slot,
                    submit_task.config.force_backend,
                )

            # Wait for at least one completion.
            done_futures = []
            for fut in list(future_meta.keys()):
                if fut.done():
                    done_futures.append(fut)

            if not done_futures:
                # Brief sleep to avoid busy-wait; tune as needed.
                time.sleep(0.05)
                continue

            for fut in done_futures:
                orig_idx, slot = future_meta.pop(fut)
                if slot is not None:
                    registry.release(slot)
                try:
                    result = fut.result()
                except Exception as exc:
                    import traceback
                    result = WorkerResult(
                        task_id=ordered[orig_idx].task_id,
                        success=False,
                        backend_used="UNKNOWN",
                        duration_s=0.0,
                        error=traceback.format_exc(),
                    )
                results[orig_idx] = result
                logger.debug("Completed task | orig_idx=%d | success=%s", orig_idx, result.success)

    wall_duration = time.perf_counter() - wall_start
    successes = sum(1 for r in results.values() if r.success)
    logger.info(
        "Scheduler complete | %d/%d succeeded | wall_time=%.2fs",
        successes, len(tasks), wall_duration,
    )

    return [results[i] for i in range(len(tasks))]


def _prepare_submission(
    task: WorkerTask,
    registry: ResourceRegistry,
    profile: HardwareProfile,
) -> tuple[WorkerTask, Optional[int]]:
    """
    Attempt to acquire a GPU slot for task. If unavailable, override config
    to CPU so the task runs immediately without stalling.

    Returns
    -------
    (task_to_submit, acquired_slot_or_None)
    """
    import dataclasses

    # Determine what backend this task would target.
    target_backend = task.config.force_backend
    if target_backend is None:
        # Infer from profile (mirrors select_backend logic without running it).
        if profile.cuda_available and profile.torch_available:
            target_backend = Backend.NVIDIA_GPU
        elif profile.intel_gpu_available and profile.sklearnex_available:
            target_backend = Backend.INTEL_GPU
        else:
            target_backend = Backend.CPU

    if target_backend == Backend.CPU:
        return task, None

    slot = registry.try_acquire(target_backend, task.config.torch_device)

    if slot is not None:
        # GPU slot acquired — submit as-is.
        return task, slot

    # GPU busy — temporarily downgrade to CPU.
    logger.info(
        "Task %s: GPU slot busy, downgrading to CPU for this submission.",
        task.task_id,
    )
    cpu_config = dataclasses.replace(task.config, force_backend=Backend.CPU)
    cpu_task = dataclasses.replace(task, config=cpu_config)
    return cpu_task, None


def _resolve_pool_size(n_tasks: int, requested: Optional[int]) -> int:
    import os
    cpu_count = os.cpu_count() or 1
    auto = max(1, cpu_count * 2 // 3)
    size = requested if requested is not None else auto
    return max(1, min(size, n_tasks))
