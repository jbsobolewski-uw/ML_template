import multiprocessing as mp
import time
from ml_worker import train_housing_model

if __name__ == '__main__':
    # Force 'spawn' for safe memory initialization on both Windows and Linux
    mp.set_start_method('spawn', force=True)

    print("===============================================================")
    print("Starting Concurrent Intel iGPU Model Training")
    print("Open Windows Task Manager (Performance -> GPU) or Linux 'intel_gpu_top'")
    print("===============================================================\n")

    # Define tasks to pass to workers
    tasks = [1, 2]

    start_all = time.time()

    # Keep pool size low (e.g. 2 processes) because iGPUs share system memory channels
    with mp.Pool(processes=2) as pool:
        results = pool.map(train_housing_model, tasks)

    end_all = time.time()

    print("\n===============================================================")
    print(f"All parallel training jobs resolved in {end_all - start_all:.2f} seconds.")
    print("Results:", results)
    print("===============================================================")
