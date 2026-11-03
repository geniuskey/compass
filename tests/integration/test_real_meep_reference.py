"""Optical regressions using MIT Meep, including weak loss and live DFT fields."""

# ruff: noqa: E402
import numpy as np
import pytest

mp = pytest.importorskip("meep")
if not hasattr(mp, "Simulation"):
    pytest.skip("MIT photonics Meep required", allow_module_level=True)

from compass.core.types import Layer, PhotodiodeSpec
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.fdtd.meep_solver import MeepSolver
from compass.solvers.tmm.tmm_core import transfer_matrix_1d


@pytest.mark.parametrize("extinction", [0.0, 0.005, 0.05])
def test_real_meep_slab_matches_tmm_and_keeps_dft_alive(extinction):
    mp.verbosity(0)
    stack = PixelStack({"pixel": {"pitch": 0.05, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("test_silicon", n=1.5, k=extinction)
    stack.layers = [Layer("silicon", 0.0, 0.1, 0.1, "test_silicon")]
    stack.photodiodes = [PhotodiodeSpec((0, 0, 0.05), (0.05, 0.05, 0.1), (0, 0), "R")]
    solver = MeepSolver(
        {
            "name": "meep",
            "params": {
                "resolution": 160,
                "runtime_periods": 80,
                "use_dispersive": False,
            },
        }
    )
    solver.setup_geometry(stack)
    solver.setup_source({"wavelength": {"mode": "single", "value": 0.55}, "polarization": "TE"})
    result = solver.run()
    assert not result.metadata.get("failed_runs")
    expected = transfer_matrix_1d(
        np.array([1, 1.5 + 1j * extinction, 1]), np.array([np.inf, 0.1, np.inf]), 0.55, 0.0, "TE"
    )
    np.testing.assert_allclose(
        [result.reflection[0], result.transmission[0], result.absorption[0]], expected, atol=0.004
    )
    assert solver._last_sim.fields is not None
    field = solver._last_sim.get_dft_array(solver._last_dft_fields, mp.Ey, 0)
    assert np.max(np.abs(field)) > 0
    if extinction:
        # A whole-cell PD must measure absorption relative to incident power,
        # rather than return its fraction of all absorption (which would be 1).
        assert 0 < result.qe_per_pixel["R_0_0"][0] < 0.2
        assert result.qe_per_pixel["R_0_0"][0] == pytest.approx(expected[2], abs=0.012)
    solver._last_sim.reset_meep()


@pytest.mark.parametrize("n,k", [(1.5, 0.005), (1.5, 0.05), (0.2, 3.0)])
def test_meep_material_matches_target_complex_epsilon(n, k):
    stack = PixelStack({"pixel": {"pitch": 0.1, "unit_cell": [1, 1]}})
    stack.material_db.register_constant("target", n=n, k=k)
    stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "target")]
    solver = MeepSolver({"name": "meep"})
    solver.setup_geometry(stack)
    medium = solver._get_meep_material(mp, "target", 0.55, True)
    np.testing.assert_allclose(np.diag(medium.epsilon(1 / 0.55)), (n + 1j * k) ** 2, atol=1e-12)
