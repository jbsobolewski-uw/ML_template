"""
main.py
-------
Sample entrypoint to demonstrate the ML framework's multiprocessing pool
and automatic hardware backend dispatch.
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent package structure is resolvable if executing directly via `python src/main.py`
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from utils.logging import setup_logging
from executor import Config, WorkerTask, run_parallel
from models.neural import build_mlp_regressor


def main() -> None:
    """Execute a sample parallel training workflow across heterogeneous models."""
    # 1. Setup centralized framework logging
    setup_logging(level="INFO")

    # =======================================================================
    # DATA & CONFIGURATION PREPARATION FOR GPU DEMONSTRATION
    # =======================================================================

    # 1. Standard small dataset (from original example)
    np.random.seed(42)
    X_small = np.random.randn(1000, 20).astype(np.float32)
    y_small = np.random.randn(1000, 1).astype(np.float32)

    # 2. Heavy dataset to maximize GPU tensor cores and notice compute utilization
    X_heavy = np.random.randn(80000, 128).astype(np.float32)
    y_heavy = np.random.randn(80000, 1).astype(np.float32)

    # 3. Medium dataset optimized for Intel graphics extension scaling limits
    X_med = np.random.randn(15000, 32).astype(np.float32)
    y_med = np.random.randn(15000).astype(np.float32)

    # Import Backend enum for explicit routing overrides
    from src.executor.accelerator import Backend

    config = Config(
        pool_size=2,
        large_workload_threshold=500_000,
        log_level="INFO",
        sklearn_n_jobs=-1
    )

    # Explicit configuration profiles to force hardware contexts
    config_force_nvidia = Config(
        pool_size=1,
        force_backend=Backend.NVIDIA_GPU,
        log_level="INFO"
    )

    config_force_intel = Config(
        pool_size=1,
        force_backend=Backend.INTEL_GPU,
        large_workload_threshold=2_000_000,  # Prevent CPU fallback bypass
        log_level="INFO"
    )

    # =======================================================================
    # CONCURRENT TASK DEFINITIONS
    # =======================================================================

    tasks = [
        # --- Task 1: Vanilla Sklearn (CPU Bound) ---
        WorkerTask(
            task_id="sklearn_rf_cpu",
            X=X_small,
            y=y_small.ravel(),
            model_type="sklearn",
            estimator_or_model=RandomForestRegressor(n_estimators=20, random_state=42),
            config=config,
        ),

        # --- Task 2: Standard PyTorch (Auto-route) ---
        WorkerTask(
            task_id="pytorch_mlp_standard",
            X=X_small,
            y=y_small,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(input_dim=20, output_dim=1),
            config=config,
            extra_kwargs={"epochs": 10, "batch_size": 128, "lr": 1e-3}
        ),

        # --- Task 3: FORCED NVIDIA CUDA STRESS TASK ---
        # Bypasses profile validation to stream dense matrix math directly to CUDA.
        # High hidden dimensions (1024x512x256) ensure persistent GPU core saturation.
        WorkerTask(
            task_id="pytorch_nvidia_cuda_stress",
            X=X_heavy,
            y=y_heavy,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(
                input_dim=128,
                output_dim=1,
                hidden_dims=(1024, 512, 256)
            ),
            config=config_force_nvidia,
            extra_kwargs={"epochs": 40, "batch_size": 256, "lr": 1e-3}
        ),

        # --- Task 4: FORCED INTEL XPU/SKLEARNEX TARGET ---
        # Explicitly targets Intel Graphics (iGPU/dGPU) runtimes using the
        # optimized daal4py/sklearnex execution engine context context.
        WorkerTask(
            task_id="sklearn_intel_gpu_patched",
            X=X_med,
            y=y_med,
            model_type="sklearn",
            estimator_or_model=RandomForestRegressor(n_estimators=100, random_state=42),
            config=config_force_intel,
        ),

        # --- Task 5: HEAVY AUTO-DISPATCH NEURAL WORKLOAD ---
        # Uses the default auto-detecting config. Because the model type is neural
        # and hardware contains a valid GPU, the framework automatically maps
        # this massive matrix operation directly to the available accelerator device.
        WorkerTask(
            task_id="pytorch_auto_gpu_heavy",
            X=X_heavy,
            y=y_heavy,
            model_type="neural",
            estimator_or_model=build_mlp_regressor(
                input_dim=128,
                output_dim=1,
                hidden_dims=(512, 512, 512)
            ),
            config=config,
            extra_kwargs={"epochs": 30, "batch_size": 512, "lr": 2e-3}
        )
    ]

    # 5. Execute tasks concurrently via cross-OS safe spawn pool
    print("\n=======================================================")
    print("      LAUNCHING WORKLOADS VIA COMPUTE POOL             ")
    print("=======================================================\n")

    results = run_parallel(tasks, config)

    print("\n=======================================================")
    print("              PROCESSING SUMMARY                       ")
    print("=======================================================\n")

    # 6. Parse and print results
    for res in results:
        if res.success:
            print(f"Task [{res.task_id}] -> SUCCESS")
            print(f"  - Target Backend Used : {res.backend_used}")
            print(f"  - Execution Duration  : {res.duration_s}s")
            if "model" in res.payload:
                print(f"  - Output Object Type  : {type(res.payload['model']).__name__}")
        else:
            print(f"Task [{res.task_id}] -> FAILED")
            print(f"  - Traceback Details   :\n{res.error}")
        print("-" * 55)


if __name__ == "__main__":
    # The 'spawn' start method requires protecting the entrypoint on Windows and Linux
    main()