"""
executor/config.py
---------
User-facing configuration for the ML framework.
All tuneable parameters live here; workers receive a Config instance.
"""

from dataclasses import dataclass, field
from typing import Optional
from .accelerator import Backend


@dataclass
class Config:
    """
    Top-level framework configuration.

    Attributes
    ----------
    pool_size : int or None
        Number of parallel worker processes.
        None = auto (2/3 of logical CPU count, min 1).
    force_backend : Backend or None
        Override automatic backend detection.
    large_workload_threshold : int
        Element count above which iGPU is skipped.
        Overrides the module-level constant in accelerator.py.
    log_level : str
        Python logging level string ('DEBUG', 'INFO', 'WARNING', ...).
    torch_device : str or None
        Explicit torch device string, e.g. 'cuda:0', 'cpu'.
        None = resolved automatically from backend selection.
    sklearn_n_jobs : int
        n_jobs passed to sklearn estimators when running on CPU.
        -1 = use all cores.
    extra : dict
        Arbitrary key-value pairs forwarded to worker functions.
    """

    pool_size: Optional[int] = None
    force_backend: Optional[Backend] = None
    large_workload_threshold: int = 500_000
    log_level: str = "INFO"
    torch_device: Optional[str] = None
    sklearn_n_jobs: int = -1
    extra: dict = field(default_factory=dict)