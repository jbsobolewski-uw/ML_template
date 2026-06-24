# USER GUIDE — ml_framework

---

## Contents

1. [Configuration](#1-configuration)
2. [Backend selection](#2-backend-selection)
3. [Building models](#3-building-models)
4. [Building tasks](#4-building-tasks)
5. [Running the pool](#5-running-the-pool)
6. [Consuming results](#6-consuming-results)
7. [Model export and inference](#7-model-export-and-inference)
8. [Pool size and parallelism](#8-pool-size-and-parallelism)
9. [Task decomposition — user responsibility](#9-task-decomposition--user-responsibility)
10. [Memory management](#10-memory-management)
11. [Error handling and pool behaviour](#11-error-handling-and-pool-behaviour)
12. [Logging](#12-logging)

---

## 1. Configuration

All tunable parameters are in `Config`. A single instance can be passed to
`run_parallel` as a pool-level default, or attached per `WorkerTask` for
per-task overrides.

```python
from ml_framework import Config, Backend

Config(
    pool_size=None,
    # int or None.
    # None = number of physical CPU cores (detected via psutil, falls back
    # to os.cpu_count() // 2). See section 8 for how to set this correctly.

    force_backend=None,
    # Backend.CPU | Backend.INTEL_GPU | Backend.NVIDIA_GPU
    # If the requested backend is absent from hardware, raises
    # BackendUnavailableError. Never silently falls back.

    large_workload_threshold=500_000,
    # n_samples * n_features above which iGPU is bypassed for sklearn tasks.
    # iGPU memory bandwidth saturates on large dense matrices; CPU with
    # n_jobs=-1 outperforms above this threshold.
    # PyTorch tasks are not subject to this threshold.

    log_level="INFO",
    # "DEBUG" | "INFO" | "WARNING" | "ERROR"

    torch_device=None,
    # Explicit torch device string: "cuda:0", "xpu", "cpu".
    # None = resolved automatically from backend selection.

    sklearn_n_jobs=-1,
    # Passed to sklearn estimators on the CPU path. -1 = all logical cores.
    # Set to 1 if running many workers in parallel to avoid core contention.
    # See section 8.

    extra={},
    # Arbitrary key-value pairs; not used by the framework internally.
)
```

---

## 2. Backend selection

### Auto-selection priority

| Condition | Selected backend |
|---|---|
| NVIDIA GPU present + CUDA torch build | `NVIDIA_GPU` |
| Intel GPU + sklearnex + workload < threshold | `INTEL_GPU` |
| Intel GPU + workload ≥ threshold | `CPU` |
| No accelerator available | `CPU` |

NVIDIA takes priority over Intel unconditionally when both are present.

### PyTorch on Intel iGPU

`INTEL_GPU` for `NeuralModel` tasks requires `intel-extension-for-pytorch` and
routes to the `xpu` device. If `intel-extension-for-pytorch` is not installed,
PyTorch falls back to CPU with a WARNING log line. `SklearnModel` tasks still
use sklearnex iGPU offload regardless of ipex availability.

### Forcing a backend

```python
from ml_framework import Config, Backend

config = Config(force_backend=Backend.NVIDIA_GPU)
# Raises BackendUnavailableError at task execution time if CUDA is absent.

config = Config(force_backend=Backend.INTEL_GPU)
# Raises BackendUnavailableError if Intel GPU or sklearnex is absent.
```

When `force_backend` targets hardware with no registered slot (e.g. NVIDIA on a
machine with no NVIDIA GPU), the scheduler downgrades the task to CPU before
submission with a WARNING. `WorkerResult.requested_backend` records the original
request; `WorkerResult.backend_used` records what actually ran. See section 6.

### Hardware detection

```python
from ml_framework import detect_hardware

profile = detect_hardware()
print(profile.cuda_available)
print(profile.intel_gpu_available)
print(profile.sklearnex_available)
print(profile.torch_available)
print(profile.extra)   # device names, cpu_count, platform string
```

Detection runs in the calling process. Workers re-run detection independently
because spawned processes do not inherit parent runtime state.

---

## 3. Building models

Models are instances of `MLModel` subclasses. They carry the estimator or module
and all training hyperparameters. The `model_type` string and `extra_kwargs` dict
from older versions are removed.

### SklearnModel

```python
from ml_framework import SklearnModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans

model = SklearnModel(RandomForestRegressor(n_estimators=100, random_state=42))
model = SklearnModel(GradientBoostingRegressor(n_estimators=200))
model = SklearnModel(KMeans(n_clusters=10))
```

Any unfitted sklearn estimator that is picklable works. `n_jobs` is propagated
from `Config.sklearn_n_jobs` automatically on the CPU path.

### NeuralModel

```python
from ml_framework import NeuralModel, build_mlp_regressor, build_mlp_classifier

model = NeuralModel(
    build_mlp_regressor(input_dim=32, output_dim=1, hidden_dims=(256, 128, 64)),
    task="regression",      # "regression" | "classification"
    epochs=50,
    batch_size=256,
    lr=1e-3,
    loss_fn=None,           # nn.Module or None — defaults to MSELoss / CrossEntropyLoss
    callbacks=[],           # list of callable(epoch: int, avg_loss: float)
)
```

### Custom architectures

Pass any `nn.Module` to `NeuralModel`. The training loop is architecture-agnostic.

```python
import torch.nn as nn
from ml_framework import NeuralModel

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return x + self.net(x)

custom = nn.Sequential(nn.Linear(32, 128), ResBlock(128), nn.Linear(128, 1))
model = NeuralModel(custom, epochs=50, batch_size=128)
```

The model instance must be picklable. Standard `nn.Module` subclasses are
picklable by default. Avoid lambda layers or closures over unpicklable objects.

### Custom MLModel subclass

```python
from ml_framework.models.base import MLModel
from ml_framework import Backend, Config
import numpy as np
from typing import Optional

class MyModel(MLModel):
    def run(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        backend: Backend,
        config: Config,
    ) -> dict:
        # ... fit logic ...
        return {
            "model": fitted_object,
            "backend_used": backend.name,
            "duration_s": 0.0,
        }
```

The returned dict must contain at minimum `model`, `backend_used`, `duration_s`.

---

## 4. Building tasks

### run_parallel_simple — recommended for most use cases

Accepts raw `(task_id, X, y, model)` tuples. Handles shared memory allocation
and cleanup internally. All tasks share the pool-level `Config`.

```python
from ml_framework import run_parallel_simple, Config, SchedulingPolicy

results = run_parallel_simple(
    [
        ("rf",  X_train, y_train, SklearnModel(RandomForestRegressor())),
        ("mlp", X_train, y_train, NeuralModel(build_mlp_regressor(32), epochs=30)),
    ],
    Config(pool_size=2),
    policy=SchedulingPolicy.BIGGEST_FIRST,
)
```

### make_task — for per-task Config overrides

Use when different tasks need different backends, thresholds, or log levels.
Shared memory handles must be unlinked by the caller after results are collected.

```python
from ml_framework import make_task, run_parallel, Config, Backend

tasks = []
handles = []

specs = [
    ("task_a", X_small, y_small, SklearnModel(rf_small), Config()),
    ("task_b", X_large, y_large, NeuralModel(mlp), Config(force_backend=Backend.NVIDIA_GPU)),
]

for task_id, X, y, model, cfg in specs:
    task, task_handles = make_task(task_id, X, y, model, cfg)
    tasks.append(task)
    handles.extend(task_handles)

try:
    results = run_parallel(tasks, Config(pool_size=2))
finally:
    for h in handles:
        h.unlink()
```

### WorkerTask — direct construction

For full control, construct `WorkerTask` directly with pre-built
`SharedArrayHandle` objects.

```python
from ml_framework import WorkerTask, SharedArrayHandle, Config
import numpy as np

X_handle = SharedArrayHandle.from_array(np.asarray(X))
y_handle = SharedArrayHandle.from_array(np.asarray(y))

task = WorkerTask(
    task_id="manual",
    X_handle=X_handle,
    y_handle=y_handle,
    model=SklearnModel(RandomForestRegressor()),
    config=Config(),
)

# After run_parallel:
X_handle.unlink()
y_handle.unlink()
```

`SharedArrayHandle.from_array(None)` produces a none-sentinel for unsupervised
tasks where `y` is not required.

---

## 5. Running the pool

```python
from ml_framework import run_parallel, Config, SchedulingPolicy

results = run_parallel(tasks, config, policy=SchedulingPolicy.FIFO)
```

`run_parallel` blocks until all tasks complete and returns results in the same
order as the input task list.

### Scheduling policies

| Policy | Behaviour |
|---|---|
| `FIFO` | Submission order. Default. Deterministic. |
| `BIGGEST_FIRST` | Largest workload (n_samples × n_features) submitted first. Front-loads heavy tasks onto GPU while lighter tasks follow. |
| `SMALLEST_FIRST` | Smallest workload first. Minimises queue wait time for light tasks. |

Workload size is read from `SharedArrayHandle.shape` before any worker starts.

### GPU slot behaviour

One `threading.Lock` per GPU device slot is held in the scheduler. GPU tasks
compete for their device slot via non-blocking `try_acquire`. A task that cannot
acquire its slot is skipped — not downgraded — and retried after the next task
completion frees the slot. CPU tasks and tasks targeting different slots proceed
in the meantime.

Tasks whose requested backend has no registered slot (hardware absent) are
downgraded to CPU once at submission time with a WARNING. They do not block the
queue.

### Spawn requirements

The pool uses `ProcessPoolExecutor` with the `spawn` start method on both platforms.

- Entrypoint must be protected: `if __name__ == "__main__":`.
- All objects in `WorkerTask` (model instances, config) must be picklable.
- Array data is in shared memory and does not need to be picklable.
- `ml_framework` must be importable by child processes.

---

## 6. Consuming results

```python
for res in results:
    res.task_id           # matches task_id passed to make_task / run_parallel_simple
    res.success           # bool
    res.backend_used      # "CPU" | "INTEL_GPU" | "NVIDIA_GPU" | "UNAVAILABLE" | "UNKNOWN"
    res.requested_backend # str or None — what Config.force_backend requested;
                          # None means auto-selected.
                          # Differs from backend_used when the scheduler downgraded
                          # the task (e.g. "NVIDIA_GPU" requested, "CPU" used).
    res.duration_s        # wall time inside the worker, including hardware detection
    res.payload           # dict — contents depend on model type (see below)
    res.error             # str traceback if success=False, else None
```

Detecting a downgrade:

```python
if res.requested_backend and res.requested_backend != res.backend_used:
    print(f"Task {res.task_id} was downgraded: "
          f"requested {res.requested_backend}, ran on {res.backend_used}")
```

### payload — SklearnModel tasks

```python
res.payload["model"]        # fitted sklearn estimator
res.payload["backend_used"] # str — "CPU" or "INTEL_GPU"
res.payload["duration_s"]   # float — fit time only, excludes detection overhead
```

### payload — NeuralModel tasks

```python
res.payload["model"]        # nn.Module moved to CPU
res.payload["device"]       # str — "cuda", "xpu", or "cpu"
res.payload["history"]      # list[float] — per-epoch average loss
res.payload["duration_s"]   # float — training loop time only
res.payload["backend_used"] # str — derived from actual torch device used
```

---

## 7. Model export and inference

### sklearn

```python
import joblib

model = res.payload["model"]

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")

preds = model.predict(X_test)
score = model.score(X_test, y_test)
```

### PyTorch

```python
import torch

model = res.payload["model"]   # already on CPU

# Recommended: weights only (portable across moves/renames)
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))

# Alternative: full model (less portable — pickles class path)
torch.save(model, "model_full.pt")
model = torch.load("model_full.pt")

# Inference
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32))
```

Prefer `state_dict` for anything that will be moved, renamed, or shared.
Full model serialisation breaks if the module definition moves.

### Learning curve

```python
history = res.payload["history"]   # list[float], one value per epoch
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel("epoch")
plt.ylabel("avg loss")
```

---

## 8. Pool size and parallelism

Understanding the two levels of parallelism is necessary to set `pool_size`
correctly.

### Two independent parallelism levels

**Level 1 — worker processes** (controlled by `Config.pool_size`):
`pool_size` caps the number of worker processes alive simultaneously. Each
process runs one task. This is the only level the framework controls.

**Level 2 — internal threading** (not controlled by the framework):
Each worker process runs sklearn with `n_jobs=-1` (all logical cores) and
PyTorch with its default thread pool (also all logical cores). This is
internal to the library and independent of `pool_size`.

The GPU slot lock operates at level 1 only. It serialises access to a GPU
device across worker processes. It has no effect on CPU parallelism — CPU
tasks always submit immediately with no lock interaction.

### Why `pool_size` and internal threading interact

With `pool_size=2` and two CPU-bound workers both using all cores internally,
both workers compete for the same physical cores simultaneously. You get ~full
CPU utilisation but each task takes longer than if it ran alone. Adding more
workers beyond 2 in this configuration increases contention without proportional
throughput gain.

### Recommended pool sizes by workload type

**sklearn on Intel iGPU (`INTEL_GPU`)**
The worker process is mostly waiting while the iGPU executes. CPU overhead per
worker is low. Multiple workers can coexist without competing for CPU, but the
GPU slot lock serialises actual iGPU use to one task at a time regardless.
Recommended: `pool_size=2–4`. Extra workers handle CPU-side orchestration while
one holds the iGPU slot.

**NVIDIA GPU (PyTorch CUDA), 1 GPU**
The GPU slot lock limits actual GPU execution to one task at a time. Additional
workers either wait for the slot or run as CPU-downgraded tasks.
Recommended: `pool_size=2` — one GPU task plus one CPU fallback task running
concurrently. Higher values only help if you have many CPU-downgraded tasks.

**NVIDIA GPU, N GPUs**
N tasks can hold GPU slots simultaneously.
Recommended: `pool_size=N` to `pool_size=N + n_physical_cores` depending on how
many CPU tasks you also want running concurrently.

**CPU with internal parallelism (`sklearn_n_jobs=-1`, PyTorch default threads)**
Each worker saturates available cores internally. Workers compete directly.
Recommended: `pool_size=1–2`. Beyond 2 increases context switching and slows
individual tasks without meaningful throughput gain.

**CPU with disabled internal parallelism (`sklearn_n_jobs=1`,
`torch.set_num_threads(1)`)**
Each worker uses one thread. No internal competition.
Recommended: `pool_size=n_physical_cores`. Each worker maps cleanly to one core.
This is the only scenario where the physical-core-based default of `pool_size=None`
is genuinely optimal.

### Summary table

| Backend | Internal parallelism | Recommended pool_size |
|---|---|---|
| Intel iGPU (sklearn) | Low (GPU does the work) | 2–4 |
| NVIDIA GPU (PyTorch) 1× | Low (GPU does the work) | 2 |
| NVIDIA GPU (PyTorch) N× | Low | N to N + n_physical_cores |
| CPU, `n_jobs=-1` / default threads | High | 1–2 |
| CPU, `n_jobs=1` / `set_num_threads(1)` | None | n_physical_cores |
| Mixed iGPU + CPU tasks | Mixed | 3–4 |

### Controlling PyTorch thread count

The framework does not currently cap PyTorch's internal thread pool. To run
more workers in parallel without core competition, set the thread count manually
at the start of `main.py` before spawning workers:

```python
import torch
torch.set_num_threads(1)          # one thread per worker
# then set pool_size=n_physical_cores in Config
```

---

## 9. Task decomposition — user responsibility

The framework does not decompose tasks internally. A single task runs sequentially
inside one worker process. Concurrency comes only from submitting multiple tasks.

### Grid search

```python
from itertools import product
from ml_framework import run_parallel_simple, Config, SklearnModel
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [4, 8, 16],
}

specs = [
    (
        f"rf_ne{ne}_md{md}",
        X, y,
        SklearnModel(RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=0)),
    )
    for ne, md in product(param_grid["n_estimators"], param_grid["max_depth"])
]

results = run_parallel_simple(specs, Config(pool_size=2))
```

### Large enumeration — batched submission

For enumerations over many large arrays (e.g. 130 images × 20 k-values × 5 seeds),
constructing all tasks upfront holds all arrays resident in the parent process
simultaneously. Batch by the outer dimension instead:

```python
from ml_framework import run_parallel_simple, Config, SklearnModel
from sklearn.cluster import KMeans

config = Config(pool_size=2)

for i, image in enumerate(load_images_lazily()):   # generator — one image at a time
    specs = [
        (
            f"img{i}_k{k}_s{seed}",
            image, None,
            SklearnModel(KMeans(n_clusters=k, random_state=seed)),
        )
        for k in [5, 10, 20]
        for seed in range(5)
    ]
    batch_results = run_parallel_simple(specs, config)
    process_results(batch_results)
    # image array released here before next iteration
```

`run_parallel_simple` allocates shared memory per batch and frees it after each
call returns. Only `pool_size` workers are alive at once; remaining tasks in a
batch queue internally and dispatch as workers finish.

---

## 10. Memory management

### Shared memory

Arrays are placed in `multiprocessing.shared_memory` blocks by the parent.
Workers attach to the existing block — no array data crosses the pickle pipe.
Each worker calls `.copy()` internally to get a writable local array, then
detaches from the shared block. The parent unlinks blocks after the pool returns.

Peak memory per array: shared block (1×) + one worker-local copy per active
worker process. For an 80,000 × 128 float32 array (~40 MB):
- Shared block: 40 MB
- 2 active workers: 2 × 40 MB copies
- Total: ~120 MB for that array

### Shared memory on Windows

On Windows, shared memory blocks persist until all processes that opened them
release their handles and the creator calls `.unlink()`. A worker that exits
abnormally before detaching leaves the block allocated until the parent's cleanup
runs. Under heavy memory pressure, use `try/finally` around `run_parallel` when
managing handles manually.

### Large model results

`res.payload["model"]` holds the full fitted model object in the parent process.
For runs with many tasks, release models you no longer need or hold only the
`state_dict`:

```python
state = res.payload["model"].state_dict()
del res   # release the full result including the model object
```

---

## 11. Error handling and pool behaviour

### Per-task failures

Exceptions inside a worker return a `WorkerResult` with `success=False` and
the full traceback in `res.error`. The pool continues processing remaining tasks.

```python
for res in results:
    if not res.success:
        print(f"Task {res.task_id} failed ({res.backend_used}):")
        print(res.error)
```

`res.backend_used` values on failure:

| Value | Meaning |
|---|---|
| `"UNAVAILABLE"` | `BackendUnavailableError` — forced backend hardware absent |
| `"UNKNOWN"` | Any other unhandled exception inside the worker |

### BackendUnavailableError

Raised when `force_backend` is set and the required hardware or libraries are
absent at worker startup. Caught by `run_worker` and returned as a failed result,
not a pool crash.

| Forced backend | Trigger |
|---|---|
| `Backend.NVIDIA_GPU` | `torch` not installed, or `torch.cuda.is_available()` False |
| `Backend.INTEL_GPU` | No Intel GPU detected, or `scikit-learn-intelex` not installed |

### Scheduler downgrade vs BackendUnavailableError

These are two distinct paths:

- **Scheduler downgrade** (WARNING at scheduler layer, before worker starts):
  triggered when `force_backend` targets a backend with no registered slot in
  `ResourceRegistry` (e.g. NVIDIA_GPU on a machine with no NVIDIA GPU). The task
  is submitted as CPU. `res.success=True`, `res.requested_backend` shows original
  request, `res.backend_used="CPU"`.

- **BackendUnavailableError** (ERROR inside worker): triggered when a slot exists
  but `select_backend` raises because the hardware failed validation at detection
  time inside the worker. `res.success=False`, `res.backend_used="UNAVAILABLE"`.

### DeviceUnavailableError

Raised inside `neural.py` if `Backend.NVIDIA_GPU` was selected by `select_backend`
but `torch.cuda.is_available()` returns False at training time. This is a
runtime divergence — CUDA was present at detection but absent at training.
Causes: driver reinitialisation, `CUDA_VISIBLE_DEVICES=""`, or context conflicts
between spawned processes.

### Pool-level crashes

A worker killed by OOM, SIGKILL, or a C-extension segfault causes
`ProcessPoolExecutor` to raise in the parent, propagating out of `run_parallel`
as an unhandled exception. Tasks in flight at crash time produce no result.

```python
from concurrent.futures import ProcessError

try:
    results = run_parallel(tasks, config)
except ProcessError as e:
    print(f"Worker killed at OS level: {e}")
```

There is no automatic retry.

### Intel iGPU soft fallback for PyTorch

If `Backend.INTEL_GPU` is selected for a `NeuralModel` task but
`intel-extension-for-pytorch` is absent, `resolve_torch_device` returns CPU
with a WARNING rather than raising. This is intentional: ipex is optional, and
`SklearnModel` tasks on the same worker still use sklearnex iGPU offload.
`res.backend_used` will be `"CPU"` and `res.payload["device"]` will be `"cpu"`.

---

## 12. Logging

Call `setup_logging` once in the main process before `run_parallel`:

```python
from ml_framework import setup_logging
setup_logging("INFO")   # "DEBUG" | "INFO" | "WARNING" | "ERROR"
```

Worker processes configure their own handlers on startup. The log level
propagates via `Config.log_level` per task. Different tasks can use different
verbosity by attaching different `Config` instances.

Log format:

```
[HH:MM:SS] [LEVEL] [ProcessName/module.name] message
```

Process names (`SpawnProcess-1`, `SpawnProcess-2`, etc.) appear in every line,
making it straightforward to trace which worker produced which output and
correlate with task timings.

Key log lines to watch:

| Source | Level | Meaning |
|---|---|---|
| `scheduler` | WARNING | Task downgraded — backend absent, running as CPU |
| `scheduler` | DEBUG | GPU slot busy, task will retry after next completion |
| `worker` | INFO | `backend=X \| requested=Y` — actual vs requested backend |
| `accelerator` | INFO | `Workload exceeds threshold` — iGPU bypassed for large task |
| `neural` | WARNING | XPU unavailable — PyTorch falling back to CPU |
| `accelerator` | INFO | `Backend forced to: CPU` — downgraded task inside worker |