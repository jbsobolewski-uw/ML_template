"""
executor/scheduler.py
---------------------
GPU-aware task scheduler with per-device mutex locking and pluggable
ordering policies.

Problem
-------
pool.map dispatches tasks in FIFO order without regard for GPU contention.
With N tasks targeting the same GPU, multiple workers attempt simultaneous
GPU use, causing CUDA context-switching overhead that can dominate compute
time on small/medium tasks.

Solution
--------
ResourceRegistry holds one threading.Lock per detected GPU device index.
The scheduler loop:

  1. Order the pending queue by SchedulingPolicy.
  2. For each task that fits in an available executor slot:
     - CPU tasks: submit immediately, no lock needed.
     - GPU tasks: non-blocking try_acquire on the target device slot.
         Acquired  → submit with GPU config, hold lock until task completes.
         Not free  → leave task at front of pending queue; do not downgrade.
                     Scheduler will retry on next iteration after a completion.
  3. On task completion, release the device lock (if held).

This ensures GPU tasks are never silently demoted to CPU due to contention.
They wait in the pending queue until the slot is genuinely free.

Exception: if the registry has NO slot for the requested backend at all
(e.g. force_backend=NVIDIA_GPU on a machine with no NVIDIA GPU), the task
is flagged at submission time with a clear warning and submitted as CPU
rather than looping forever.

GPU slot count
--------------
One slot per detected CUDA device (by torch.cuda.device_count()).
Intel iGPU: one shared slot at index -1.
Multi-GPU: tasks with Config.torch_device='cuda:1' bind to slot 1, etc.

Policies
--------
FIFO           - submission order (default)
BIGGEST_FIRST  - largest workload first; front-loads heavy GPU tasks.
SMALLEST_FIRST - smallest workload first; minimises latency for light tasks.

Workload size derives from SharedArrayHandle.shape, falling back to
MLModel.workload_elements.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, Future
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
    """One GPU device slot: a mutex + identity."""
    device_index: int
    lock: threading.Lock = field(default_factory=threading.Lock)


class ResourceRegistry:
    """
    Tracks available GPU device slots.

    One slot per CUDA device; one shared slot (index -1) for Intel iGPU.
    CPU tasks never interact with the registry.

    Parameters
    ----------
    profile : HardwareProfile
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
            logger.debug("ResourceRegistry: %d CUDA slot(s) registered.", len(self._slots))

        if profile.intel_gpu_available:
            self._slots[-1] = _DeviceSlot(device_index=-1)
            logger.debug("ResourceRegistry: Intel GPU slot registered.")

        if not self._slots:
            logger.debug("ResourceRegistry: no GPU slots (CPU-only system).")

    @property
    def has_gpu_slots(self) -> bool:
        return bool(self._slots)

    def has_slot_for(self, backend: Backend) -> bool:
        """Return True if at least one slot exists for this backend type."""
        if backend == Backend.NVIDIA_GPU:
            return any(i >= 0 for i in self._slots)
        if backend == Backend.INTEL_GPU:
            return -1 in self._slots
        return False  # CPU needs no slot

    def try_acquire(self, backend: Backend, torch_device: Optional[str]) -> Optional[int]:
        """
        Non-blocking acquisition of a matching slot.

        Returns device index on success, None if all matching slots are busy.
        Callers must first check has_slot_for(); this method assumes a slot
        exists and returns None only when all are currently locked.
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

        return None

    def release(self, slot_index: int) -> None:
        """Release a previously acquired slot."""
        if slot_index in self._slots:
            try:
                self._slots[slot_index].lock.release()
            except RuntimeError:
                pass


def _parse_cuda_index(device_str: Optional[str]) -> Optional[int]:
    """'cuda:N' -> N, 'cuda' -> 0, None -> None."""
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
    """Elements in X; used for BIGGEST/SMALLEST ordering."""
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

def _ordered_tasks(tasks: list[WorkerTask], policy: SchedulingPolicy) -> list[WorkerTask]:
    if policy == SchedulingPolicy.BIGGEST_FIRST:
        return sorted(tasks, key=_workload_size, reverse=True)
    if policy == SchedulingPolicy.SMALLEST_FIRST:
        return sorted(tasks, key=_workload_size)
    return list(tasks)


# ---------------------------------------------------------------------------
# Target backend inference
# ---------------------------------------------------------------------------

def _infer_target_backend(task: WorkerTask, profile: HardwareProfile) -> Backend:
    """
    Determine what backend a task will request, without running select_backend
    (which requires X to be loaded). Mirrors auto-select priority.
    """
    if task.config.force_backend is not None:
        return task.config.force_backend
    if profile.cuda_available and profile.torch_available:
        return Backend.NVIDIA_GPU
    if profile.intel_gpu_available and profile.sklearnex_available:
        workload = _workload_size(task)
        if workload < task.config.large_workload_threshold:
            return Backend.INTEL_GPU
    return Backend.CPU


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# Sentinel: task should be held at front of pending queue (GPU slot busy).
_HOLD = object()


def run_scheduled(
    tasks: Sequence[WorkerTask],
    config: Config,
    profile: HardwareProfile,
    policy: SchedulingPolicy = SchedulingPolicy.FIFO,
) -> list[WorkerResult]:
    """
    Execute tasks via GPU-aware scheduler.

    GPU tasks wait in the pending queue until their device slot is free.
    They are never silently downgraded to CPU due to contention.
    Tasks targeting a backend with no registered slot (e.g. NVIDIA_GPU on
    a CPU-only or Intel-only machine) are logged as warnings and submitted
    as CPU immediately rather than blocking indefinitely.

    Parameters
    ----------
    tasks : sequence of WorkerTask
    config : Config
        Pool-level config (pool_size, log_level). Per-task Config on
        task.config governs backend selection inside each worker.
    profile : HardwareProfile
    policy : SchedulingPolicy

    Returns
    -------
    list of WorkerResult, order matches input tasks.
    """
    # Materialise once — guards against generators and ensures id() stability.
    tasks_list: list[WorkerTask] = list(tasks)
    if not tasks_list:
        return []

    registry = ResourceRegistry(profile)

    # Build orig-index map before sorting so result order matches caller's list.
    task_orig_idx: dict[int, int] = {id(t): i for i, t in enumerate(tasks_list)}
    ordered = _ordered_tasks(tasks_list, policy)

    # future -> (original_index, acquired_slot_or_None, original_requested_backend_or_None)
    # The third element preserves the backend the user asked for on tasks that were
    # downgraded before submission (force_backend replaced with CPU). Without it,
    # run_worker sees only the overridden config and reports requested=CPU.
    future_meta: dict[Future, tuple[int, Optional[int], Optional[str]]] = {}
    results: dict[int, WorkerResult] = {}

    max_workers = _resolve_pool_size(len(tasks_list), config.pool_size)
    logger.info(
        "Scheduler starting | tasks=%d | workers=%d | policy=%s",
        len(tasks_list), max_workers, policy.name,
    )

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending: list[WorkerTask] = list(ordered)

        while pending or future_meta:

            # --- submission pass -------------------------------------------
            # Walk pending in order. Submit CPU tasks and GPU tasks that can
            # acquire a slot. GPU tasks whose slot is busy are skipped (i += 1)
            # so tasks further in the queue (CPU or different slot) can proceed.
            # They are retried on the next iteration after a completion frees
            # a slot. Tasks are never silently downgraded due to contention.
            i = 0
            while i < len(pending) and len(future_meta) < max_workers:
                task = pending[i]
                target = _infer_target_backend(task, profile)

                if target == Backend.CPU:
                    pending.pop(i)
                    fut = executor.submit(run_worker, task)
                    future_meta[fut] = (task_orig_idx[id(task)], None, None)
                    continue  # don't increment i; next task slides into i

                # GPU task: check whether any slot exists for this backend.
                if not registry.has_slot_for(target):
                    # Hardware absent or library missing for this backend.
                    # Downgrade once with an explicit warning; never loop on it.
                    logger.warning(
                        "Task %s: %s requested but no slot registered "
                        "(hardware absent or library missing). "
                        "Submitting as CPU. requested_backend=%s actual_backend=CPU",
                        task.task_id, target.name, target.name,
                    )
                    pending.pop(i)
                    cpu_task = dataclasses.replace(
                        task,
                        config=dataclasses.replace(
                            task.config, force_backend=Backend.CPU
                        ),
                    )
                    fut = executor.submit(run_worker, cpu_task)
                    # Preserve the original target so WorkerResult.requested_backend
                    # reflects what the user asked for, not the downgraded config.
                    future_meta[fut] = (task_orig_idx[id(task)], None, target.name)
                    continue

                slot = registry.try_acquire(target, task.config.torch_device)
                if slot is not None:
                    pending.pop(i)
                    fut = executor.submit(run_worker, task)
                    future_meta[fut] = (task_orig_idx[id(task)], slot, None)
                    logger.debug(
                        "Task %s acquired slot %d for %s.",
                        task.task_id, slot, target.name,
                    )
                    continue

                # Slot exists but currently locked. Skip — do not downgrade.
                # Other tasks (CPU or targeting a free slot) may still proceed.
                logger.debug(
                    "Task %s: slot for %s busy, will retry after next completion.",
                    task.task_id, target.name,
                )
                i += 1

            # --- wait for completion ---------------------------------------
            done_futures = [f for f in future_meta if f.done()]

            if not done_futures:
                time.sleep(0.05)
                continue

            for fut in done_futures:
                orig_idx, slot, scheduler_requested = future_meta.pop(fut)
                if slot is not None:
                    registry.release(slot)
                    logger.debug("Released slot %d.", slot)
                try:
                    result = fut.result()
                    # If the scheduler downgraded this task, the worker only saw
                    # the overridden config (force_backend=CPU). Restore the true
                    # requested backend so the caller sees what was originally asked.
                    if scheduler_requested is not None:
                        result = dataclasses.replace(
                            result, requested_backend=scheduler_requested
                        )
                except Exception:
                    result = WorkerResult(
                        task_id="unknown",
                        success=False,
                        backend_used="UNKNOWN",
                        duration_s=0.0,
                        error=traceback.format_exc(),
                    )
                results[orig_idx] = result

    wall_duration = time.perf_counter() - wall_start
    successes = sum(1 for r in results.values() if r.success)
    logger.info(
        "Scheduler complete | %d/%d succeeded | wall_time=%.2fs",
        successes, len(tasks_list), wall_duration,
    )

    return [results[i] for i in range(len(tasks_list))]


def _resolve_pool_size(n_tasks: int, requested: Optional[int]) -> int:
    import os
    cpu_count = os.cpu_count() or 1
    auto = max(1, cpu_count * 2 // 3)
    size = requested if requested is not None else auto
    return max(1, min(size, n_tasks))