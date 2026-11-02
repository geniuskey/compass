"""Unsupported incidence must fail before normal-incidence FDTD can run."""

import pytest

from compass.solvers.fdtd.fdtdx_solver import FdtdxSolver
from compass.solvers.fdtd.fdtdz_solver import FdtdzSolver
from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver


@pytest.mark.parametrize("solver_class", [FdtdxSolver, FdtdzSolver, FlaportFdtdSolver])
def test_oblique_incidence_is_rejected(solver_class):
    # Source validation needs no optional backend or geometry initialization.
    solver = solver_class.__new__(solver_class)
    with pytest.raises(NotImplementedError, match="normal incidence only"):
        solver.setup_source({"angle": {"theta_deg": 15.0}})
