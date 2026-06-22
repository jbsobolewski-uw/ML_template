# USER GUIDE — ml_framework

---

## Contents

1. [Configuration](#1-configuration)
2. [Backend selection](#2-backend-selection)
3. [Building tasks](#3-building-tasks)
4. [Running the pool](#4-running-the-pool)
5. [Consuming results](#5-consuming-results)
6. [Model export and inference](#6-model-export-and-inference)
7. [Task decomposition — user responsibility](#7-task-decomposition--user-responsibility)
8. [Memory management](#8-memory-management)
9. [Error handling and pool behaviour](#9-error-handling-and-pool-behaviour)
10. [Logging](#10-logging)

---

## 1. Configuration

All tunable parameters are in `Config`. Pass one instance per call to `run_parallel`, or per `WorkerTask` for per-task overrides.

```python
from ml_framework import Config, Backend

Config(
    pool_size=None,             # int or None. None = floor(cpu_count * 2/3), min 1
    force_backend=None,         # Backend.CPU | Backend.INTEL_GPU | Backend.NVIDIA_GPU
    large_workload_threshold=500_000,  # n_samples * n_features above which iGPU is bypassed
    log_level="INFO",           # DEBUG | INFO | WARNING | ERROR
    torch_device=None,          # str: 'cuda:0', 'cpu', 'xpu'. None = auto
    sklearn_n_jobs=-1,          # passed to sklearn estimators on CPU path. -1 = all cores
    extra={},                   # arbitrary kwargs forwarded to worker (user-defined use)
)
```

`force_backend` raises `BackendUnavailableError` at task execution time if the requested hardware is absent. It does not fall back silently.

`large_workload_threshold` applies only to Intel iGPU. Above this element count (`n_samples * n_features`) the framework routes sklearn tasks to CPU regardless of iGPU availability. This exists because iGPU memory bandwidth saturates on large dense matrices and CPU with `n_jobs=-1` outperforms. PyTorch tasks are not subject to this threshold; they follow CUDA > XPU > CPU priority unconditionally.

---

## 2. Backend selection

### Auto-selection priority

| Condition | Selected backend |
|---|---|
| NVIDIA GPU + CUDA torch build | `NVIDIA_GPU` |
| Intel GPU + sklearnex + workload < threshold | `INTEL_GPU` |
| Intel GPU + workload ≥ threshold | `CPU` |
| No accelerator | `CPU` |

NVIDIA takes priority over Intel unconditionally. If both are present and CUDA is available, `NVIDIA_GPU` is selected.

### PyTorch on Intel iGPU

`INTEL_GPU` backend for neural tasks uses `intel-extension-for-pytorch` and routes to `xpu` device. If `intel-extension-for-pytorch` is not installed, the neural runner falls back to CPU with a warning. Sklearn tasks still use sklearnex iGPU offload regardless of ipex availability.

### Forcing a backend

```python
from ml_framework import Config, Backend

# Raises BackendUnavailableError if CUDA is absent — never silently uses CPU.
config = Config(force_backend=Backend.NVIDIA_GPU)

# Raises BackendUnavailableError if Intel GPU or sklearnex is absent.
config = Config(force_backend=Backend.INTEL_GPU)
```

### Hardware detection

```python
from ml_framework import detect_hardware

profile = detect_hardware()
print(profile.cuda_available)
print(profile.intel_gpu_available)
print(profile.extra)  # device names, cpu_count, platform
```

Detection runs in the calling process. Workers re-run detection independently because spawned processes do not inherit parent state.

---

## 3. Building tasks

```python
from ml_framework import WorkerTask, Config, build_mlp_regressor
from sklearn.ensemble import GradientBoostingRegressor

WorkerTask(
    task_id="job_01",           # any hashable; used for logging and result lookup
    X=X_train,                  # np.ndarray, shape (n_samples, n_features)
    y=y_train,                  # np.ndarray
    model_type="sklearn",       # "sklearn" | "neural"
    estimator_or_model=GradientBoostingRegressor(n_estimators=200),
    config=Config(),
    extra_kwargs={},            # neural only: epochs, batch_size, lr, task, loss_fn, callbacks
)
```

For neural tasks `extra_kwargs` maps directly to `train_neural` keyword arguments:

```python
WorkerTask(
    task_id="mlp_01",
    X=X, y=y,
    model_type="neural",
    estimator_or_model=build_mlp_regressor(input_dim=32),
    config=Config(),
    extra_kwargs={
        "epochs": 100,
        "batch_size": 512,
        "lr": 5e-4,
        "task": "regression",   # "regression" | "classification"
        "loss_fn": None,        # nn.Module or None for default
        "callbacks": [lambda epoch, loss: print(epoch, loss)],
    },
)
```

### Custom architectures

Pass any `nn.Module` instance as `estimator_or_model`. The training loop in `train_neural` is architecture-agnostic; it calls `.forward()` and backpropagates the loss.

```python
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return x + self.net(x)

model = nn.Sequential(nn.Linear(32, 128), ResBlock(128), nn.Linear(128, 1))

task = WorkerTask(
    task_id="resnet_tabular",
    X=X, y=y,
    model_type="neural",
    estimator_or_model=model,
    config=Config(),
    extra_kwargs={"epochs": 50},
)
```

The model instance must be picklable. Standard `nn.Module` subclasses are picklable by default. Lambda layers, closures capturing unpicklable objects, or models with open file handles are not.

---

## 4. Running the pool

```python
from ml_framework import run_parallel, Config

results = run_parallel(tasks, config)
```

`run_parallel` blocks until all tasks complete. It returns a list of `WorkerResult` in the same order as the input task list.

Pool size is resolved as `min(n_tasks, config.pool_size or floor(cpu_count * 2/3))`. At most `pool_size` worker processes are alive simultaneously. Remaining tasks queue internally and are dispatched as workers finish.

The pool uses `multiprocessing.get_context("spawn")`, which is safe on both Linux and Windows. The spawn context requires:

- The entrypoint must be protected by `if __name__ == "__main__":`.
- All objects in `WorkerTask` (arrays, estimators, models, config) must be picklable.
- The `ml_framework` package must be importable by child processes (i.e., on `PYTHONPATH`).

---

## 5. Consuming results

```python
for res in results:
    res.task_id        # matches WorkerTask.task_id
    res.success        # bool
    res.backend_used   # "CPU" | "INTEL_GPU" | "NVIDIA_GPU" | "UNAVAILABLE" | "UNKNOWN"
    res.duration_s     # wall time of the worker including hardware detection
    res.payload        # dict — contents depend on model_type (see below)
    res.error          # str traceback if success=False, else None
```

### payload contents

**sklearn tasks:**

```python
res.payload["model"]        # fitted sklearn estimator
res.payload["backend_used"] # str
res.payload["duration_s"]   # float — fit time only, excludes detection overhead
```

**neural tasks:**

```python
res.payload["model"]        # nn.Module on CPU
res.payload["device"]       # str: "cuda", "xpu", "cpu"
res.payload["history"]      # list[float] — per-epoch average loss
res.payload["duration_s"]   # float — training loop time only
res.payload["backend_used"] # str — derived from actual torch device, not Backend enum
```

---

## 6. Model export and inference

### sklearn

```python
import joblib

model = res.payload["model"]

# Persist
joblib.dump(model, "model.joblib")

# Reload
model = joblib.load("model.joblib")

# Inference
preds = model.predict(X_test)
scores = model.score(X_test, y_test)
```

### PyTorch

```python
import torch

model = res.payload["model"]  # already on CPU

# Persist weights only (requires architecture to reload)
torch.save(model.state_dict(), "model.pt")

# Persist entire model (architecture + weights, less portable)
torch.save(model, "model_full.pt")

# Reload weights
model.load_state_dict(torch.load("model.pt"))

# Reload full model
model = torch.load("model_full.pt")

# Inference
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32))
```

Prefer `state_dict` for portability. Full model serialisation with `torch.save(model)` pickles the class definition path; if the module is moved or renamed, reload fails.

### Learning curve

```python
history = res.payload["history"]  # list of avg loss per epoch
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel("epoch")
plt.ylabel("loss")
```

---

## 7. Task decomposition — user responsibility

The framework does not decompose tasks internally. A single `WorkerTask` runs sequentially inside one worker process. There is no automatic chunking of large grid searches, hyperparameter sweeps, or dataset partitions.

**The user is responsible for task granularity.**

A `WorkerTask` is the atomic unit of parallelism. Concurrency is achieved only by submitting multiple tasks to `run_parallel`. The framework schedules them across the pool; it does not subdivide any individual task.

### Practical decomposition patterns

**Grid search — explicit decomposition:**

```python
from itertools import product
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [4, 8, 16],
}

tasks = [
    WorkerTask(
        task_id=f"rf_ne{ne}_md{md}",
        X=X, y=y,
        model_type="sklearn",
        estimator_or_model=RandomForestRegressor(
            n_estimators=ne, max_depth=md, random_state=0
        ),
        config=Config(),
    )
    for ne, md in product(param_grid["n_estimators"], param_grid["max_depth"])
]

results = run_parallel(tasks, Config(pool_size=4))
```

**Image clustering — batched enumeration:**

For 130 images × 20 k-values × 5 seeds = 13,000 combinations, creating all tasks upfront holds all 130 image arrays in parent-process memory simultaneously. Instead, batch by image:

```python
from ml_framework import run_parallel, Config, WorkerTask

def make_image_tasks(image, image_id, k_values, seeds):
    tasks = []
    for k in k_values:
        for seed in seeds:
            tasks.append(WorkerTask(
                task_id=f"img{image_id}_k{k}_s{seed}",
                X=image,
                y=None,        # clustering — handle in a custom worker or adapt fit_sklearn
                model_type="sklearn",
                estimator_or_model=KMeans(n_clusters=k, random_state=seed),
                config=Config(),
            ))
    return tasks

config = Config(pool_size=4)

for i, image in enumerate(load_images_lazily()):   # generator, not list
    batch_tasks = make_image_tasks(image, i, k_values=[5,10,20], seeds=range(5))
    batch_results = run_parallel(batch_tasks, config)
    process_results(batch_results)
    # image array is released here before the next iteration
```

This keeps only one image resident in the parent at a time. `pool_size` worker processes each hold one task's copy of the image data; on task completion the worker's memory is freed before the next task is dispatched.

**Key principle:** the pool size cap ensures at most `pool_size` tasks are executing simultaneously. Tasks not yet dispatched remain serialised in the pool's internal queue. However, `pool.map` requires the full task list to be constructed upfront and pickled into the queue. For large enumerations over large arrays, construct tasks in batches as above rather than building the full 13,000-task list at once.

---

## 8. Memory management

`run_parallel` calls `pool.map`, which pickles the entire task list into the worker queue before any task starts. For `N` tasks each containing an array of size `S`, parent-process peak memory is `N * S` before any result is returned.

Mitigations:

- Batch task submission as shown above.
- Use memory-mapped arrays (`np.memmap`) for very large datasets. Workers receive the memmap path and load slices independently.
- For neural tasks, `res.payload["model"]` is the full model on CPU. If running hundreds of tasks, hold only the state dict and discard the model object.

---

## 9. Error handling and pool behaviour

### Per-task failures

A task that raises any exception inside the worker returns a `WorkerResult` with `success=False` and the full traceback in `res.error`. The pool continues processing remaining tasks.

```python
for res in results:
    if not res.success:
        print(f"Task {res.task_id} failed:")
        print(res.error)
```

`res.backend_used` is `"UNAVAILABLE"` for `BackendUnavailableError` (forced backend absent from hardware) and `"UNKNOWN"` for all other exceptions.

### BackendUnavailableError

Raised when `force_backend` is set in `Config` and the required hardware or libraries are absent. This surfaces as a failed `WorkerResult`, not a pool crash.

Conditions that trigger it:

| Forced backend | Trigger condition |
|---|---|
| `Backend.NVIDIA_GPU` | `torch` not installed, or `torch.cuda.is_available()` is False |
| `Backend.INTEL_GPU` | No Intel GPU detected, or `scikit-learn-intelex` not installed |

### DeviceUnavailableError

Raised inside `neural.py` at training time if `Backend.NVIDIA_GPU` was selected by `select_backend` (passed hardware detection) but `torch.cuda.is_available()` returns False at the moment of training. This is a runtime divergence — CUDA was present at detection but absent at execution. Causes include driver reinitialisation, `CUDA_VISIBLE_DEVICES=""`, or context conflicts between spawned processes.

Both error types are caught in `run_worker` and returned as failed `WorkerResult` objects.

### Pool-level crashes

A pool-level crash (worker process killed by OOM, SIGKILL, or a C-extension segfault) is not recoverable per task. `pool.map` raises `WorkerLostError` in the parent, which propagates out of `run_parallel` as an unhandled exception. Wrap `run_parallel` in a try/except if this is a concern:

```python
from multiprocessing import ProcessError

try:
    results = run_parallel(tasks, config)
except ProcessError as e:
    # At least one worker was killed at the OS level.
    print(e)
```

Tasks in flight at the time of the crash do not return results. Tasks not yet dispatched are lost. There is no automatic retry.

### Silent CPU fallback — what does and does not happen

The framework does not silently fall back from a requested accelerator to CPU at the backend-selection layer. `force_backend` either succeeds or raises.

The one exception is Intel iGPU for PyTorch neural tasks: if `Backend.INTEL_GPU` is selected (or forced) but `intel-extension-for-pytorch` is absent, `resolve_torch_device` returns CPU with a warning rather than raising. This is intentional: ipex is an optional dependency, and sklearn tasks on the same worker still use sklearnex iGPU offload. The warning is logged at WARNING level and `res.payload["backend_used"]` will reflect `"CPU"`.

---

## 10. Logging

Call `setup_logging` once in the main process before `run_parallel`:

```python
from ml_framework import setup_logging
setup_logging("INFO")   # DEBUG | INFO | WARNING | ERROR
```

Worker processes configure their own handlers independently. The log level propagates via `Config.log_level` inside each `WorkerTask`. If different tasks require different verbosity, set `log_level` per `Config` instance.

Log output format:

```
[HH:MM:SS] [LEVEL] [ProcessName/module.name] message
```

Worker process names are assigned by `multiprocessing.Pool` (`SpawnPoolWorker-1`, etc.) and appear in every log line, making it straightforward to trace which worker produced which output.