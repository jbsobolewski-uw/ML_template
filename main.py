"""
main.py
-------
Entrypoint demonstrating the ml_framework multiprocessing pool
and automatic hardware backend dispatch.

All imports come from the top-level package. No internal paths exposed.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml_framework import (
    Backend,
    Config,
    WorkerTask,
    run_parallel,
    build_mlp_regressor,
    setup_logging
)


def main() -> None:
    """Execute a sample parallel training workflow across heterogeneous models."""
    setup_logging(level="INFO")

    np.random.seed(42)
    X_small = np.random.randn(1000, 20).astype(np.float32)
    y_small = np.random.randn(1000, 1).astype(np.float32)

    X_heavy = np.random.randn(80_000, 128).astype(np.float32)
    y_heavy = np.random.randn(80_000, 1).astype(np.float32)

    X_med = np.random.randn(15_000, 32).astype(np.float32)
    y_med = np.random.randn(15_000).astype(np.float32)

    config = Config(
        pool_size=2,
        large_workload_threshold=500_000,
        log_level="INFO",
        sklearn_n_jobs=-1,
    )

    # force_backend=NVIDIA_GPU on hardware without CUDA will now produce a
    # failed WorkerResult with BackendUnavailableError, not a silent CPU run.
    config_force_nvidia = Config(
        pool_size=1,
        force_backend=Backend.NVIDIA_GPU,
        log_level="INFO",
    )

    config_force_intel = Config(
        pool_size=1,
        force_backend=Backend.INTEL_GPU,
        large_workload_threshold=2_000_000,
        log_level="INFO",
    )

    tasks = [
        WorkerTask(
            task_id="sklearn_rf_cpu",
            X=X_small,
            y=y_small.ravel(),
            model_type="sklearn",
            estimator_or_model=RandomForestRegressor(n_estimators=20, random_state=42),
            config=config,
        ),
        WorkerTask(
            task_id="pytorch_mlp_standard",
            X=X_small,
            y=y_small,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(input_dim=20, output_dim=1),
            config=config,
            extra_kwargs={"epochs": 10, "batch_size": 128, "lr": 1e-3},
        ),
        WorkerTask(
            task_id="pytorch_nvidia_cuda_stress",
            X=X_heavy,
            y=y_heavy,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(
                input_dim=128, output_dim=1, hidden_dims=(1024, 512, 256)
            ),
            config=config_force_nvidia,
            extra_kwargs={"epochs": 40, "batch_size": 256, "lr": 1e-3},
        ),
        WorkerTask(
            task_id="sklearn_intel_gpu_patched",
            X=X_med,
            y=y_med,
            model_type="sklearn",
            estimator_or_model=RandomForestRegressor(n_estimators=100, random_state=42),
            config=config_force_intel,
        ),
        WorkerTask(
            task_id="pytorch_auto_gpu_heavy",
            X=X_heavy,
            y=y_heavy,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(
                input_dim=128, output_dim=1, hidden_dims=(512, 512, 512)
            ),
            config=config,
            extra_kwargs={"epochs": 30, "batch_size": 512, "lr": 2e-3},
        ),
    ]

    print("\n=======================================================")
    print("      LAUNCHING WORKLOADS VIA COMPUTE POOL             ")
    print("=======================================================\n")

    results = run_parallel(tasks, config)

    print("\n=======================================================")
    print("              PROCESSING SUMMARY                       ")
    print("=======================================================\n")

    for res in results:
        if res.success:
            print(f"Task [{res.task_id}] -> SUCCESS")
            print(f"  Backend Used : {res.backend_used}")
            print(f"  Duration     : {res.duration_s}s")
            if "model" in res.payload:
                print(f"  Model Type   : {type(res.payload['model']).__name__}")
        else:
            print(f"Task [{res.task_id}] -> FAILED")
            print(f"  Error        : {res.error}")
        print("-" * 55)


if __name__ == "__main__":
    main()
