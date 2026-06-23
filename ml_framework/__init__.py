"""
ml_framework
============
Full public API — import everything from here.

    from ml_framework import (
        Config, Backend, SchedulingPolicy,
        WorkerTask, WorkerResult,
        run_parallel, run_parallel_simple, make_task,
        SklearnModel, NeuralModel, MLModel,
        build_mlp_regressor, build_mlp_classifier,
        setup_logging,
    )
"""

from ml_framework.executor import (
    Backend,
    HardwareProfile,
    BackendUnavailableError,
    detect_hardware,
    select_backend,
    Config,
    SharedArrayHandle,
    WorkerTask,
    WorkerResult,
    run_worker,
    run_parallel,
    run_parallel_simple,
    make_task,
    SchedulingPolicy,
    ResourceRegistry,
)

from ml_framework.models import (
    MLModel,
    SklearnModel,
    fit_sklearn,
    NeuralModel,
    train_neural,
    resolve_torch_device,
    build_mlp_regressor,
    build_mlp_classifier,
    DeviceUnavailableError,
)

from ml_framework.utils import setup_logging, get_worker_logger

__all__ = [
    # executor
    "Backend",
    "HardwareProfile",
    "BackendUnavailableError",
    "detect_hardware",
    "select_backend",
    "Config",
    "SharedArrayHandle",
    "WorkerTask",
    "WorkerResult",
    "run_worker",
    "run_parallel",
    "run_parallel_simple",
    "make_task",
    "SchedulingPolicy",
    "ResourceRegistry",
    # models
    "MLModel",
    "SklearnModel",
    "fit_sklearn",
    "NeuralModel",
    "train_neural",
    "resolve_torch_device",
    "build_mlp_regressor",
    "build_mlp_classifier",
    "DeviceUnavailableError",
    # utils
    "setup_logging",
    "get_worker_logger",
]