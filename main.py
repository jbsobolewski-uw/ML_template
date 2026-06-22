import sys
import os
import multiprocessing as mp
import time
from sklearn.datasets import fetch_california_housing
from ml_worker import train_housing_model

if __name__ == '__main__':
    venv_python = sys.executable
    mp.set_executable(venv_python)
    mp.set_start_method('spawn', force=True)

    print("===============================================================")
    print(f"Using Interpreter: {venv_python}")
    print("Pre-fetching dataset to prevent file locking conflicts...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print("Dataset loaded successfully.")
    print("===============================================================\n")

    tasks = [
        (X, y, 1),
        (X, y, 2)
    ]

    POOL_SIZE: int = len(tasks)
    if POOL_SIZE > (os.cpu_count() * 2 // 3 or 1):
        POOL_SIZE = os.cpu_count() * 2 // 3 or 1

    print("Starting Concurrent Intel iGPU Model Training...")
    start_all = time.time()

    with mp.Pool(processes=POOL_SIZE) as pool:
        results = pool.starmap(train_housing_model, tasks)

    end_all = time.time()

    print("\n===============================================================")
    print(f"All parallel training jobs resolved in {end_all - start_all:.2f} seconds.")
    print("Results:", results)
    print("===============================================================")
