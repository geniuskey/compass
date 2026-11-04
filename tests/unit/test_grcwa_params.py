"""Tests for grcwa parameter semantics.

grcwa truncates by TOTAL plane-wave count (nG), unlike the per-axis
Fourier order used by torcwa/meent/fmmax. The adapter must prefer an
explicit params.nG and only fall back to fourier_order[0] (with a
warning) for legacy configs.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import patch

import pytest

from compass.geometry.pixel_stack import PixelStack
from compass.solvers.rcwa.grcwa_solver import GrcwaSolver


class _RecordingGrcwaObj:
    """Fake grcwa.obj that records the nG it was constructed with."""

    last_nG: int | None = None
    last_angles: tuple[float, float] | None = None

    def __init__(self, nG, L1, L2, freq, theta, phi, verbose=0):
        type(self).last_nG = nG
        type(self).last_angles = (theta, phi)

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
        return 0.1, 0.3


def _run(solver: GrcwaSolver, theta_deg=0.0, phi_deg=0.0):
    solver.setup_geometry(PixelStack({"pixel": {"pitch": 1.0, "unit_cell": [2, 2]}}))
    solver.setup_source(
        {
            "wavelength": {"mode": "single", "value": 0.55},
            "angle": {"theta_deg": theta_deg, "phi_deg": phi_deg},
            "polarization": "TE",
        }
    )
    fake = types.ModuleType("grcwa")
    fake.obj = _RecordingGrcwaObj
    with patch.dict(sys.modules, {"grcwa": fake}):
        return solver.run()


def test_explicit_nG_param_is_used():
    result = _run(GrcwaSolver({"name": "grcwa", "params": {"nG": 25}}))

    assert _RecordingGrcwaObj.last_nG == 25
    assert result.metadata["nG"] == 25


def test_incident_angles_are_converted_to_radians():
    import math

    _run(GrcwaSolver({"name": "grcwa", "params": {"nG": 9}}), 30.0, 45.0)
    assert _RecordingGrcwaObj.last_angles == pytest.approx((math.pi / 6, math.pi / 4))


def test_fourier_order_fallback_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="compass.solvers.rcwa.grcwa_solver"):
        result = _run(GrcwaSolver({"name": "grcwa", "params": {"fourier_order": [49, 49]}}))

    assert _RecordingGrcwaObj.last_nG == 49
    assert result.metadata["nG"] == 49
    assert any("NOT a per-axis order" in rec.message for rec in caplog.records)


def test_explicit_nG_wins_over_fourier_order():
    result = _run(
        GrcwaSolver({"name": "grcwa", "params": {"nG": 25, "fourier_order": [49, 49]}})
    )

    assert _RecordingGrcwaObj.last_nG == 25
    assert result.metadata["nG"] == 25


def test_qe_method_recorded_in_metadata():
    # The recording mock has no field API, so field-integrated QE must fall
    # back to the eps''-weight approximation and say so in the metadata.
    result = _run(GrcwaSolver({"name": "grcwa", "params": {"nG": 9}}))

    assert result.metadata["qe_method"] == "eps_imag_weight_fallback"


def test_shipped_grcwa_configs_use_nG():
    """The tuned grcwa presets must carry explicit nG, not fourier_order."""
    from pathlib import Path

    import yaml

    for name in ("grcwa", "grcwa_fast", "grcwa_converged"):
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "solver" / f"{name}.yaml"
        params = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["solver"]["params"]
        assert "nG" in params, f"{name}.yaml must set params.nG"
        assert "fourier_order" not in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
