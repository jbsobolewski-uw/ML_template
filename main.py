"""
main.py
-------
Demonstrates the updated ml_framework API:
  - run_parallel_simple() for minimal boilerplate (handles shm lifecycle).
  - SklearnModel / NeuralModel replacing model_type string.
  - SchedulingPolicy for GPU-aware ordering.
  - Direct make_task() + run_parallel() for manual shm control.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml_framework import (
    Backend,
    Config,
    SchedulingPolicy,
    SklearnModel,
    NeuralModel,
    build_mlp_regressor,
    run_parallel_simple,
    make_task,
    run_parallel,
    setup_logging,
)


def main() -> None:
    setup_logging("INFO")

    np.random.seed(42)
    X_small = np.random.randn(1000, 20).astype(np.float32)
    y_small = np.random.randn(1000).astype(np.float32)
    X_heavy = np.random.randn(80_000, 128).astype(np.float32)
    y_heavy = np.random.randn(80_000).astype(np.float32)

    config = Config(pool_size=2, log_level="INFO", sklearn_n_jobs=-1)
    config_nvidia = Config(pool_size=1, force_backend=Backend.NVIDIA_GPU)
    config_intel = Config(
        pool_size=1,
        force_backend=Backend.INTEL_GPU,
        large_workload_threshold=2_000_000,
    )

    # ------------------------------------------------------------------
    # Option A: run_parallel_simple — shm allocation handled internally.
    # Accepts (task_id, X, y, MLModel) tuples directly.
    # ------------------------------------------------------------------

    task_specs = [
        (
            "sklearn_rf_cpu",
            X_small,
            y_small,
            SklearnModel(RandomForestRegressor(n_estimators=20, random_state=42)),
        ),
        (
            "pytorch_mlp_standard",
            X_small,
            y_small,
            NeuralModel(
                build_mlp_regressor(input_dim=20, output_dim=1),
                epochs=10,
                batch_size=128,
                lr=1e-3,
            ),
        ),
        (
            "pytorch_nvidia_stress",
            X_heavy,
            y_heavy,
            NeuralModel(
                build_mlp_regressor(input_dim=128, output_dim=1, hidden_dims=(1024, 512, 256)),
                epochs=40,
                batch_size=256,
                lr=1e-3,
            ),
        ),
        (
            "sklearn_intel_gpu",
            X_small,
            y_small,
            SklearnModel(RandomForestRegressor(n_estimators=100, random_state=42)),
        ),
        (
            "pytorch_auto_heavy",
            X_heavy,
            y_heavy,
            NeuralModel(
                build_mlp_regressor(input_dim=128, output_dim=1, hidden_dims=(512, 512, 512)),
                epochs=30,
                batch_size=512,
                lr=2e-3,
            ),
        ),
    ]

    # Override per-task config by attaching it inside a WorkerTask manually,
    # or use the simple interface with a single pool config.
    # For per-task config overrides, use make_task() directly (Option B below).

    print("\n=======================================================")
    print("   LAUNCHING VIA run_parallel_simple (BIGGEST_FIRST)   ")
    print("=======================================================\n")

    results = run_parallel_simple(
        task_specs,
        config,
        policy=SchedulingPolicy.BIGGEST_FIRST,
    )

    _print_summary(results)

    # ------------------------------------------------------------------
    # Option B: make_task() — manual shm control, per-task Config.
    # Use when different tasks need different Config instances.
    # ------------------------------------------------------------------

    print("\n=======================================================")
    print("   LAUNCHING VIA make_task() WITH PER-TASK CONFIG       ")
    print("=======================================================\n")

    tasks = []
    handles = []

    specs_with_config = [
        ("rf_per_task_config", X_small, y_small,
         SklearnModel(RandomForestRegressor(n_estimators=50)), config),
        ("mlp_forced_nvidia", X_heavy, y_heavy,
         NeuralModel(build_mlp_regressor(128, 1, (256, 128)), epochs=5), config_nvidia),
        ("rf_intel", X_small, y_small,
         SklearnModel(RandomForestRegressor(n_estimators=50)), config_intel),
    ]

    for task_id, X, y, model, task_config in specs_with_config:
        task, task_handles = make_task(task_id, X, y, model, task_config)
        tasks.append(task)
        handles.extend(task_handles)

    try:
        results_b = run_parallel(tasks, config, policy=SchedulingPolicy.SMALLEST_FIRST)
    finally:
        for h in handles:
            h.unlink()

    _print_summary(results_b)


def _print_summary(results) -> None:
    print("\n-------------------------------------------------------")
    for res in results:
        status = "SUCCESS" if res.success else "FAILED"
        print(f"Task [{res.task_id}] -> {status}")
        if res.success:
            print(f"  Backend : {res.backend_used}")
            print(f"  Duration: {res.duration_s}s")
            if "model" in res.payload:
                print(f"  Model   : {type(res.payload['model']).__name__}")
        else:
            print(f"  Error   : {res.error}")
        print("-" * 55)


if __name__ == "__main__":
    main()
