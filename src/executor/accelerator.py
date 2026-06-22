"""
executor/accelerator.py
--------------
Hardware detection and compute backend selection.

Priority logic:
  - Large workloads (> LARGE_WORKLOAD_THRESHOLD samples): CPU-only.
  - NVIDIA GPU present + torch available: CUDA.
  - Intel GPU present + sklearnex available: iGPU (small/medium workloads only).
  - Fallback: CPU.
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


@dataclass
class HardwareProfile:
    backend: Backend
    device_name: str
    cuda_available: bool = False
    intel_gpu_available: bool = False
    sklearnex_available: bool = False
    torch_available: bool = False
    extra: dict = field(default_factory=dict)


def _detect_nvidia() -> tuple[bool, str]:
    """Return (available, device_name) for NVIDIA CUDA."""
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
    except ImportError:
        pass
    return False, ""


def _detect_intel_gpu() -> tuple[bool, str]:
    """
    Return (available, device_name) for Intel GPU via dpctl or sklearnex.
    Works on both Linux and Windows.
    """
    try:
        import dpctl
        devices = dpctl.get_devices(device_type="gpu")
        intel_devices = [d for d in devices if "Intel" in d.name]
        if intel_devices:
            return True, intel_devices[0].name
    except (ImportError, Exception):
        pass

    # Fallback: try sklearnex GPU context probe.
    try:
        from sklearnex import config_context
        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression
        from sklearnex import patch_sklearn
        patch_sklearn()
        X_tiny, y_tiny = make_regression(n_samples=10, n_features=2, random_state=0)
        with config_context(target_offload="gpu:0"):
            m = LinearRegression()
            m.fit(X_tiny, y_tiny)
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


def detect_hardware() -> HardwareProfile:
    """
    Probe available accelerators and return a HardwareProfile.
    Does NOT make backend selection decisions — see select_backend().
    """
    cuda_ok, cuda_name = _detect_nvidia()
    igpu_ok, igpu_name = _detect_intel_gpu()
    sklex_ok = _sklearnex_available()
    torch_ok = _torch_available()

    profile = HardwareProfile(
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
    return profile


def select_backend(
    profile: HardwareProfile,
    n_samples: int = 0,
    n_features: int = 0,
    force_backend: Optional[Backend] = None,
) -> Backend:
    """
    Choose compute backend given hardware profile and workload size.

    Parameters
    ----------
    profile : HardwareProfile
    n_samples : int
    n_features : int
    force_backend : Backend or None
        Bypass auto-selection.

    Returns
    -------
    Backend
    """
    if force_backend is not None:
        # Sanitize forced NVIDIA requests against environment capabilities
        if force_backend == Backend.NVIDIA_GPU and not (
                profile.cuda_available and profile.torch_available
        ):
            logger.warning(
                "Backend forced to NVIDIA_GPU, but CUDA is unavailable "
                "on this hardware. Falling back to automatic backend choice."
            )

        # Sanitize forced Intel requests against environment capabilities
        elif force_backend == Backend.INTEL_GPU and not (
                profile.intel_gpu_available and profile.sklearnex_available
        ):
            logger.warning(
                "Backend forced to INTEL_GPU, but Intel extensions "
                "are unavailable on this hardware. Falling back to "
                "automatic backend choice."
            )

        else:
            logger.info("Backend successfully forced to: %s", force_backend)
            return force_backend

    workload_size = n_samples * n_features

    # NVIDIA GPU — highest priority for neural nets / large models.
    if profile.cuda_available and profile.torch_available:
        logger.info("Backend selected: NVIDIA_GPU (%s)", profile.extra["cuda_device"])
        return Backend.NVIDIA_GPU

    # Intel iGPU — only for small/medium classical ML workloads.
    if (
        profile.intel_gpu_available
        and profile.sklearnex_available
        and workload_size < LARGE_WORKLOAD_THRESHOLD
    ):
        logger.info(
            "Backend selected: INTEL_GPU (%s), workload=%d",
            profile.extra["intel_gpu_device"],
            workload_size,
        )
        return Backend.INTEL_GPU

    if profile.intel_gpu_available and workload_size >= LARGE_WORKLOAD_THRESHOLD:
        logger.info(
            "Workload (%d elements) exceeds threshold (%d). "
            "Bypassing iGPU, using CPU.",
            workload_size,
            LARGE_WORKLOAD_THRESHOLD,
        )

    logger.info("Backend selected: CPU")
    return Backend.CPU