"""
models/base.py
--------------
Abstract base class for all model wrappers in ml_framework.

Eliminates the model_type string field from WorkerTask. Workers call
task.model.run(...) directly; the model object carries its own dispatch
logic. Adding a new model type requires only a new MLModel subclass —
no changes to worker.py or pool.py.

Subclasses
----------
- SklearnModel  (models/sklearn_models.py)
- NeuralModel   (models/neural.py)

Custom model types
------------------
Subclass MLModel and implement run(). The instance must be picklable
(standard Python requirement for multiprocessing spawn).

    class MyCustomModel(MLModel):
        def run(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray],
            backend: Backend,
            config: Config,
        ) -> dict:
            ...
            return {"model": fitted, "backend_used": backend.name, "duration_s": 0.0}
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from ml_framework.executor.accelerator import Backend
from ml_framework.executor.config import Config


class MLModel(abc.ABC):
    """
    Abstract wrapper for a trainable model.

    The instance is created in the parent process, pickled into a WorkerTask,
    and reconstructed inside the spawned worker. run() is called there.

    All subclasses must be picklable.
    """

    @abc.abstractmethod
    def run(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        backend: Backend,
        config: Config,
    ) -> dict:
        """
        Execute training and return a result payload dict.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray or None
        backend : Backend
            Already-selected backend (detection happened in worker before this call).
        config : Config

        Returns
        -------
        dict
            Must contain at minimum:
              'model'        : fitted model object
              'backend_used' : str (Backend.name)
              'duration_s'   : float
        """

    @property
    def workload_elements(self) -> int:
        """
        Estimated workload size in elements, used by scheduler for BIGGEST/SMALLEST
        ordering before X is available in the parent process.

        Override to provide a meaningful hint. Default returns 0 (unknown).
        """
        return 0
