"""
models/sklearn_models.py
------------------------
Scikit-learn model wrapper implementing the MLModel ABC.

Supports:
  - CPU:       plain sklearn, n_jobs propagated from Config.sklearn_n_jobs.
  - INTEL_GPU: sklearnex-patched sklearn with target_offload="gpu".
  - NVIDIA_GPU: no sklearn CUDA path; falls back to CPU (use NeuralModel for CUDA).

Usage
-----
    from ml_framework.models.sklearn_models import SklearnModel
    from sklearn.ensemble import RandomForestRegressor

    task = WorkerTask(
        task_id="rf",
        X_handle=...,
        y_handle=...,
        model=SklearnModel(RandomForestRegressor(n_estimators=100)),
        config=Config(),
    )
"""

import time
import logging
from typing import Optional, Any

import numpy as np
from sklearn.base import BaseEstimator

from ml_framework.executor.accelerator import Backend
from ml_framework.executor.config import Config
from ml_framework.models.base import MLModel

logger = logging.getLogger(__name__)


def _patch_intel() -> None:
    """Apply sklearnex monkey-patch inside the current process."""
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        logger.debug("sklearnex patch applied.")
    except ImportError:
        logger.warning("sklearnex not available; falling back to vanilla sklearn.")


class SklearnModel(MLModel):
    """
    Wraps an unfitted sklearn estimator for dispatch through WorkerTask.

    Parameters
    ----------
    estimator : BaseEstimator
        Unfitted sklearn estimator instance. Must be picklable.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    @property
    def workload_elements(self) -> int:
        # No X available at construction time; scheduler uses SharedArrayHandle
        # shape if present. Return 0 as sentinel.
        return 0

    def run(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        backend: Backend,
        config: Config,
    ) -> dict:
        """
        Fit the estimator with backend-appropriate execution context.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        backend : Backend
        config : Config

        Returns
        -------
        dict with keys: 'model', 'backend_used', 'duration_s'
        """
        estimator = self.estimator

        # Propagate parallelism hint on CPU path.
        if backend != Backend.INTEL_GPU and hasattr(estimator, "n_jobs"):
            estimator.set_params(n_jobs=config.sklearn_n_jobs)

        start = time.perf_counter()

        if backend == Backend.INTEL_GPU:
            _patch_intel()
            try:
                from sklearnex import config_context
                with config_context(target_offload="gpu:0"):
                    estimator.fit(X, y)
                backend_used = Backend.INTEL_GPU
            except Exception as exc:
                logger.warning(
                    "Intel GPU fit failed (%s); falling back to CPU.", exc
                )
                estimator.fit(X, y)
                backend_used = Backend.CPU
        else:
            # CPU and NVIDIA_GPU both use plain sklearn; CUDA is for NeuralModel.
            estimator.fit(X, y)
            backend_used = Backend.CPU

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


# ---------------------------------------------------------------------------
# Backward-compatible functional interface
# ---------------------------------------------------------------------------

def fit_sklearn(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend = Backend.CPU,
    sklearn_n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Functional wrapper around SklearnModel.run() for direct calls outside
    the WorkerTask/pool system.
    """
    from ml_framework.executor.config import Config
    cfg = Config(sklearn_n_jobs=sklearn_n_jobs)
    return SklearnModel(estimator).run(X, y, backend, cfg)
