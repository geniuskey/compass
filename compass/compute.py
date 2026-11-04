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


def resolve_compute_device(backend: str = "auto", solver_name: str | None = None) -> str:
    """Return a device supported by the installed runtime and solver.

    CUDA is selected automatically when available. The current solver adapters
    do not have a complete MPS-safe path: for example, torcwa needs complex
    eigendecomposition, which PyTorch's MPS backend does not implement. MPS
    requests therefore run on CPU instead of returning failed or NaN results.
    """
    if backend == "cpu":
        return "cpu"

    if backend == "auto":
        if _torch_cuda_available():
            return "cuda"
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
        return "cuda"

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
