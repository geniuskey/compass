"""FDTD solver adapters."""

from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

try:
    from compass.solvers.fdtd.fdtdz_solver import FdtdzSolver
except ImportError:
    pass

try:
    from compass.solvers.fdtd.meep_solver import MeepSolver
except ImportError:
    pass

__all__ = [
    "FlaportFdtdSolver",
    "FdtdzSolver",
    "MeepSolver",
]
