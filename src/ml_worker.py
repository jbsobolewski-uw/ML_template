import os
import time
import numpy as np
from sklearnex import patch_sklearn, config_context

# 1. Patch scikit-learn IMMEDIATELY before importing algorithms
patch_sklearn()

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor


def train_housing_model(worker_id):
    """Fetches a real dataset, inflates it to stress the iGPU, and trains a model."""
    print(f"[Worker {worker_id}] Initializing on Process ID: {os.getpid()}...")

    # 2. Fetch the open-source California Housing dataset
    housing = fetch_california_housing()
    X, y = housing.data.astype(np.float32), housing.target.astype(np.float32)

    # 3. Inflate the dataset size significantly so the training doesn't instantly blink out.
    # This gives you enough time to check your Task Manager / Resource Monitor.
    X_large = np.repeat(X, 35, axis=0)
    y_large = np.repeat(y, 35, axis=0)

    print(f"[Worker {worker_id}] Dataset inflated to {X_large.shape[0]} samples. Preparing GPU context...")

    # 4. Bind execution context tightly to the Intel iGPU compute layers
    start_time = time.time()
    with config_context(target_offload="gpu"):
        print(f"[Worker {worker_id}] CRITICAL: Training RandomForest on Intel iGPU now...")

        # High n_estimators and max_depth keep the iGPU working for several seconds
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=1  # Let the GPU backend handle execution parallelism
        )
        model.fit(X_large, y_large)

    duration = time.time() - start_time
    print(f"[Worker {worker_id}] Complete! Training took {duration:.2f} seconds.")
    return f"Success from worker {worker_id}"
