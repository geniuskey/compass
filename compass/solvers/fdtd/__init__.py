"""FDTD solver adapters."""

import contextlib

from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

with contextlib.suppress(ImportError):
    from compass.solvers.fdtd.fdtdz_solver import FdtdzSolver

with contextlib.suppress(ImportError):
    from compass.solvers.fdtd.meep_solver import MeepSolver

__all__ = [
    "FdtdzSolver",
    "FlaportFdtdSolver",
    "MeepSolver",
]
