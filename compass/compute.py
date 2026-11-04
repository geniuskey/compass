"""Portable compute-device selection for COMPASS solver adapters."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _torch_cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except ImportError:
        return 0


def cuda_device_index(device: str) -> int:
    """Return the GPU index encoded in a device string (``"cuda:1"`` -> 1).

    Plain ``"cuda"`` and non-CUDA devices map to index 0.
    """
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def _cuda_device(gpu_id: int) -> str:
    if gpu_id < 0:
        raise ValueError(f"compute.gpu_id must be >= 0, got {gpu_id}")
    if gpu_id == 0:
        return "cuda"
    count = _torch_cuda_device_count()
    if gpu_id >= count:
        raise RuntimeError(
            f"compute.gpu_id={gpu_id} was requested, but only {count} CUDA device(s) are visible"
        )
    return f"cuda:{gpu_id}"


def resolve_compute_device(
    backend: str = "auto", solver_name: str | None = None, gpu_id: int = 0
) -> str:
    """Return a device supported by the installed runtime and solver.

    When CUDA is selected, ``gpu_id`` picks the GPU: index 0 yields ``"cuda"``
    and any other index yields ``"cuda:<gpu_id>"``.

    CUDA is selected automatically when available. The current solver adapters
    do not have a complete MPS-safe path: for example, torcwa needs complex
    eigendecomposition, which PyTorch's MPS backend does not implement. MPS
    requests therefore run on CPU instead of returning failed or NaN results.
    """
    if backend == "cpu":
        return "cpu"

    if backend == "auto":
        if _torch_cuda_available():
            return _cuda_device(gpu_id)
        return "cpu"

    if backend == "cuda":
        # An explicit device choice is preserved rather than silently
        # downgraded. Auto mode is the portable option for machines without
        # an NVIDIA GPU.
        if not _torch_cuda_available():
            raise RuntimeError(
                "compute.backend='cuda' was requested, but CUDA is not available "
                "(no NVIDIA GPU or a CPU-only PyTorch build). Use compute=auto "
                "or compute=cpu instead."
            )
        return _cuda_device(gpu_id)

    if backend == "mps":
        solver_label = f" for {solver_name}" if solver_name else ""
        if _torch_mps_available():
            logger.warning(
                "MPS was requested%s, but the current solver adapters do not "
                "support their required operations on MPS; using CPU",
                solver_label,
            )
        else:
            logger.warning("MPS was requested%s, but is unavailable; using CPU", solver_label)
        return "cpu"

    raise ValueError(f"Unsupported compute backend: {backend!r}")
