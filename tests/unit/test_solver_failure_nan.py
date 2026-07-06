"""Tests for solver failure handling.

A failed (wavelength, polarization) run must contribute NaN to the
spectral outputs — not a silent 0.0 that deflates polarization averages —
and must be recorded in result metadata under "failed_runs".
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from compass.geometry.pixel_stack import PixelStack
from compass.solvers.rcwa.grcwa_solver import GrcwaSolver

FAIL_WAVELENGTH = 0.65


class _FakeGrcwaObj:
    """Minimal grcwa.obj stand-in that fails at FAIL_WAVELENGTH."""

    def __init__(self, nG, L1, L2, freq, theta, phi, verbose=0):
        self._freq = freq

    def Add_LayerUniform(self, thickness, eps):
        pass

    def Add_LayerGrid(self, thickness, nx, ny):
        pass

    def Init_Setup(self, Gmethod=0):
        pass

    def GridLayer_geteps(self, eps_flat):
        pass

    def MakeExcitationPlanewave(self, p_amp, s_amp, p_phase, s_phase, order=0):
        pass

    def RT_Solve(self, normalize=1):
        if abs(1.0 / self._freq - FAIL_WAVELENGTH) < 1e-9:
            raise RuntimeError("synthetic eigendecomposition failure")
        return 0.1, 0.3


@pytest.fixture
def solver_with_stack() -> GrcwaSolver:
    solver = GrcwaSolver({"name": "grcwa", "params": {"fourier_order": [3, 3]}})
    solver.setup_geometry(
        PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [1, 2], "bayer_map": [["R", "G"]]}})
    )
    solver.setup_source(
        {
            "wavelength": {"mode": "list", "values": [0.55, FAIL_WAVELENGTH]},
            "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
            "polarization": "TE",
        }
    )
    return solver


def _run_with_fake_grcwa(solver: GrcwaSolver):
    fake = types.ModuleType("grcwa")
    fake.obj = _FakeGrcwaObj
    with patch.dict(sys.modules, {"grcwa": fake}), pytest.warns(UserWarning, match="NaN"):
        return solver.run()


def test_failed_run_yields_nan_not_zero(solver_with_stack):
    result = _run_with_fake_grcwa(solver_with_stack)

    # Successful wavelength keeps finite values
    assert result.reflection[0] == pytest.approx(0.1)
    assert result.transmission[0] == pytest.approx(0.3)

    # Failed wavelength must be NaN, not 0.0
    assert np.isnan(result.reflection[1])
    assert np.isnan(result.transmission[1])
    assert np.isnan(result.absorption[1])
    for qe in result.qe_per_pixel.values():
        assert np.isfinite(qe[0])
        assert np.isnan(qe[1])


def test_failed_run_recorded_in_metadata(solver_with_stack):
    result = _run_with_fake_grcwa(solver_with_stack)

    failed = result.metadata["failed_runs"]
    assert len(failed) == 1
    assert failed[0]["wavelength"] == pytest.approx(FAIL_WAVELENGTH)
    assert failed[0]["polarization"] == "TE"
    assert "synthetic eigendecomposition failure" in failed[0]["error"]


def test_successful_run_has_no_failure_metadata(solver_with_stack):
    fake = types.ModuleType("grcwa")

    class _AlwaysOk(_FakeGrcwaObj):
        def RT_Solve(self, normalize=1):
            return 0.1, 0.3

    fake.obj = _AlwaysOk
    with patch.dict(sys.modules, {"grcwa": fake}):
        result = solver_with_stack.run()

    assert "failed_runs" not in result.metadata
    assert np.all(np.isfinite(result.reflection))
