"""RCWA solver adapters."""

import contextlib

with contextlib.suppress(ImportError):
    from compass.solvers.rcwa import fmmax_solver as _fmmax  # noqa: F401
