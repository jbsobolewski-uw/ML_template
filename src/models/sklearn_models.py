"""
models/sklearn_models.py
------------------------
Scikit-learn model wrappers with automatic backend dispatch.

Supports:
  - CPU:       plain sklearn (n_jobs=-1 for parallelism)
  - INTEL_GPU: sklearnex patched sklearn with target_offload="gpu"
  - NVIDIA_GPU: falls back to CPU sklearn (sklearn has no CUDA backend;
                use neural.py for GPU-accelerated deep models on NVIDIA).

Usage
-----
from models.sklearn_models import fit_sklearn

result = fit_sklearn(
    estimator=RandomForestRegressor(n_estimators=100),
    X=X_train,
    y=y_train,
    backend=Backend.INTEL_GPU,
    sklearn_n_jobs=-1
)
"""

import time
import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from src.executor.accelerator import Backend

logger = logging.getLogger(__name__)


def _patch_intel() -> None:
    """Apply sklearnex patch inside the current process."""
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        logger.debug("sklearnex patch applied.")
    except ImportError:
        logger.warning("sklearnex not available; falling back to vanilla sklearn.")


def fit_sklearn(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend = Backend.CPU,
    sklearn_n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Fit a scikit-learn estimator with the specified backend.

    Parameters
    ----------
    estimator : BaseEstimator
        Unfitted sklearn estimator instance.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    backend : Backend
        Compute backend to use.
    sklearn_n_jobs : int
        n_jobs value for CPU-bound parallel fitting.

    Returns
    -------
    dict with keys: 'model', 'backend_used', 'duration_s'
    """
    # Propagate n_jobs to estimator if it supports it (CPU path).
    if backend != Backend.INTEL_GPU and hasattr(estimator, "n_jobs"):
        estimator.set_params(n_jobs=sklearn_n_jobs)

    start = time.perf_counter()

    if backend == Backend.INTEL_GPU:
        _patch_intel()
        try:
            from sklearnex import config_context
            with config_context(target_offload="gpu:0"):
                estimator.fit(X, y)
            backend_used = Backend.INTEL_GPU
        except Exception as exc:
            logger.warning("Intel GPU fit failed (%s); falling back to CPU.", exc)
            estimator.fit(X, y)
            backend_used = Backend.CPU
    else:
        # CPU or NVIDIA (sklearn has no CUDA path; neural.py handles that).
        estimator.fit(X, y)
        backend_used = Backend.CPU if backend == Backend.CPU else Backend.CPU

    duration = time.perf_counter() - start
    logger.info(
        "sklearn fit complete | backend=%s | duration=%.2fs | estimator=%s",
        backend_used.name,
        duration,
        type(estimator).__name__,
    )
    return {
        "model": estimator,
        "backend_used": backend_used.name,
        "duration_s": round(duration, 4),
    }