"""Transfer Matrix Method (TMM) solver for 1D planar thin-film stacks.

TMM is an analytical solver that computes reflection, transmission, and
absorption for multilayer planar structures using 2x2 transfer matrices.
It is extremely fast (~1000x faster than RCWA) but limited to 1D stacks
without lateral patterning.
"""

from compass.solvers.tmm.tmm_core import (
    tmm_field_profile,
    tmm_spectrum,
    transfer_matrix_1d,
)
from compass.solvers.tmm.tmm_solver import TMMSolver

__all__ = [
    "TMMSolver",
    "transfer_matrix_1d",
    "tmm_spectrum",
    "tmm_field_profile",
]
