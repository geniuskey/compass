"""Actual FDTDX slab regressions; mocks cannot validate loss or source power."""

import numpy as np
import pytest

from compass.core.types import Layer
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverFactory
from compass.solvers.tmm.tmm_core import transfer_matrix_1d


@pytest.mark.parametrize("extinction", [0.0, 0.005, 0.05])
def test_real_fdtdx_slab_matches_tmm(extinction):
    pytest.importorskip("fdtdx")
    stack = PixelStack({"pixel": {"pitch": 0.05, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("glass", n=1.5, k=extinction)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "glass")]
    stack.photodiodes = []
    solver = SolverFactory.create(
        "fdtdx",
        {
            "name": "fdtdx",
            "params": {
                "grid_spacing": 0.0125,
                "pml_layers": 28,
                "duration": 40.0,
                "dtype": "float64",
            },
        },
        "cpu",
    )
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    result = solver.run()
    assert not result.metadata.get("failed_runs")
    assert result.metadata["actual_device"] == "cpu:0"
    expected = transfer_matrix_1d(
        np.array([1.0, 1.5 + 1j * extinction, 1.0]),
        np.array([np.inf, 0.1, np.inf]),
        0.55,
        0.0,
        "TE",
    )
    np.testing.assert_allclose(
        [result.reflection[0], result.transmission[0], result.absorption[0]], expected, atol=0.004
    )
