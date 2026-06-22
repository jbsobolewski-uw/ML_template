import os
import time
import numpy as np
from sklearnex import patch_sklearn, config_context

patch_sklearn()
from sklearn.ensemble import RandomForestRegressor


def train_housing_model(X, y, worker_id):
    """Processes a training workload on the Intel iGPU."""
    print(f"[Worker {worker_id}] Processing on PID: {os.getpid()}...")

    X_large = np.repeat(X.astype(np.float32), 5, axis=0)
    y_large = np.repeat(y.astype(np.float32), 5, axis=0)

    print(f"[Worker {worker_id}] Inflated to {X_large.shape}. Routing to hardware engine...")

    start_time = time.time()

    with config_context(target_offload="gpu"):
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            n_jobs=1
        )

        model.fit(X_large, y_large)

    duration = time.time() - start_time
    print(f"[Worker {worker_id}] Task completed in {duration:.2f} seconds.")

    return f"Success - Worker {worker_id}"
