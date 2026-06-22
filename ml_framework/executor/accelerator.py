"""
executor/accelerator.py
-----------------------
Hardware detection and compute backend selection.

Priority logic:
  - Large workloads (> LARGE_WORKLOAD_THRESHOLD elements): CPU-only (iGPU bypassed).
  - NVIDIA GPU present + torch available: CUDA.
  - Intel GPU present + sklearnex available: iGPU (small/medium workloads only).
  - Fallback: CPU.

force_backend behaviour:
  - If the forced backend is *unavailable* on this hardware, raises BackendUnavailableError.
    The caller (worker.py) catches this and returns a failed WorkerResult.
  - Silent fallback on force failure is intentionally NOT supported: it masks misconfiguration.
"""

import os
import logging
import platform
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

# Workload size above which iGPU is bypassed in favour of CPU.
LARGE_WORKLOAD_THRESHOLD: int = 500_000  # total elements (n_samples * n_features)


class Backend(Enum):
    CPU = auto()
    INTEL_GPU = auto()
    NVIDIA_GPU = auto()


class BackendUnavailableError(RuntimeError):
    """
    Raised when a forced backend cannot be satisfied by available hardware.
    Prefer this over silent fallback so misconfiguration is never hidden.
    """


@dataclass
class HardwareProfile:
    backend: Backend
    device_name: str
    cuda_available: bool = False
    intel_gpu_available: bool = False
    sklearnex_available: bool = False
    torch_available: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_nvidia() -> tuple[bool, str]:
    """Return (available, device_name) for NVIDIA CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return False, ""


def _detect_intel_gpu() -> tuple[bool, str]:
    """
    Return (available, device_name) for Intel GPU.
    Tries dpctl first (most reliable), then a sklearnex probe fit as fallback.
    """
    try:
        import dpctl
        devices = dpctl.get_devices(device_type="gpu")
        intel_devices = [d for d in devices if "Intel" in d.name]
        if intel_devices:
            return True, intel_devices[0].name
    except (ImportError, Exception):
        pass

    try:
        from sklearnex import config_context, patch_sklearn
        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression

        patch_sklearn()
        X_tiny, y_tiny = make_regression(n_samples=10, n_features=2, random_state=0)
        with config_context(target_offload="gpu:0"):
            LinearRegression().fit(X_tiny, y_tiny)
        return True, "Intel GPU (sklearnex probe)"
    except Exception:
        pass

    return False, ""


def _sklearnex_available() -> bool:
    try:
        import sklearnex  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """
    Probe available accelerators and return a HardwareProfile.
    Does NOT make backend selection decisions — see select_backend().
    """
    cuda_ok, cuda_name = _detect_nvidia()
    igpu_ok, igpu_name = _detect_intel_gpu()
    sklex_ok = _sklearnex_available()
    torch_ok = _torch_available()

    return HardwareProfile(
        backend=Backend.CPU,
        device_name="CPU",
        cuda_available=cuda_ok,
        intel_gpu_available=igpu_ok,
        sklearnex_available=sklex_ok,
        torch_available=torch_ok,
        extra={
            "cuda_device": cuda_name,
            "intel_gpu_device": igpu_name,
            "platform": platform.system(),
            "cpu_count": os.cpu_count(),
        },
    )


def select_backend(
    profile: HardwareProfile,
    n_samples: int = 0,
    n_features: int = 0,
    force_backend: Optional[Backend] = None,
    large_workload_threshold: Optional[int] = None,
) -> Backend:
    """
    Choose compute backend given hardware profile and workload size.

    Parameters
    ----------
    profile : HardwareProfile
    n_samples : int
    n_features : int
    force_backend : Backend or None
        Bypass auto-selection. Raises BackendUnavailableError if the requested
        backend cannot be satisfied — never silently falls back.
    large_workload_threshold : int or None
        Override module-level LARGE_WORKLOAD_THRESHOLD for this call.

    Returns
    -------
    Backend

    Raises
    ------
    BackendUnavailableError
        If force_backend is set but the required hardware/libraries are absent.
    """
    threshold = large_workload_threshold if large_workload_threshold is not None \
        else LARGE_WORKLOAD_THRESHOLD

    if force_backend is not None:
        _validate_forced_backend(force_backend, profile)
        logger.info("Backend forced to: %s", force_backend.name)
        return force_backend

    workload_size = n_samples * n_features

    if profile.cuda_available and profile.torch_available:
        logger.info("Backend selected: NVIDIA_GPU (%s)", profile.extra["cuda_device"])
        return Backend.NVIDIA_GPU

    if (
        profile.intel_gpu_available
        and profile.sklearnex_available
        and workload_size < threshold
    ):
        logger.info(
            "Backend selected: INTEL_GPU (%s) | workload=%d",
            profile.extra["intel_gpu_device"],
            workload_size,
        )
        return Backend.INTEL_GPU

    if profile.intel_gpu_available and workload_size >= threshold:
        logger.info(
            "Workload (%d elements) exceeds threshold (%d). "
            "Bypassing iGPU, using CPU.",
            workload_size,
            threshold,
        )

    logger.info("Backend selected: CPU")
    return Backend.CPU


def _validate_forced_backend(backend: Backend, profile: HardwareProfile) -> None:
    """
    Raise BackendUnavailableError if the requested backend cannot be used.

    Parameters
    ----------
    backend : Backend
    profile : HardwareProfile

    Raises
    ------
    BackendUnavailableError
    """
    if backend == Backend.NVIDIA_GPU:
        missing = []
        if not profile.torch_available:
            missing.append("torch not installed")
        if not profile.cuda_available:
            missing.append("CUDA unavailable (no NVIDIA GPU or driver missing)")
        if missing:
            raise BackendUnavailableError(
                f"Backend.NVIDIA_GPU requested but cannot be satisfied: "
                f"{'; '.join(missing)}."
            )

    elif backend == Backend.INTEL_GPU:
        missing = []
        if not profile.intel_gpu_available:
            missing.append("no Intel GPU detected")
        if not profile.sklearnex_available:
            missing.append("scikit-learn-intelex not installed")
        if missing:
            raise BackendUnavailableError(
                f"Backend.INTEL_GPU requested but cannot be satisfied: "
                f"{'; '.join(missing)}."
            )
