"""Unit tests for TorcwaSolver field-distribution plumbing (no backend needed)."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from compass.core.types import LayerSlice
from compass.solvers.rcwa.torcwa_solver import TorcwaSolver


def _solver_with_state():
    solver = TorcwaSolver({"name": "torcwa"})
    stack = MagicMock()
    stack.domain_size = (1.0, 1.0)
    solver._pixel_stack = stack
    eps = np.full((8, 8), 2.25 + 0.1j)
    solver._last_layer_info = [
        LayerSlice(z_start=0.5, z_end=1.0, thickness=0.5, eps_grid=eps, name="top"),
        LayerSlice(z_start=0.0, z_end=0.5, thickness=0.5, eps_grid=eps, name="bottom"),
    ]
    solver._last_wavelength = 0.55
    solver._last_source_amp = [0.0, 1.0]
    solver._last_sim = MagicMock()
    return solver


class TestValidation:
    def test_unknown_component(self):
        with pytest.raises(ValueError, match="component"):
            _solver_with_state().get_field_distribution("Hx", "xz", 0.0)

    def test_unknown_plane(self):
        with pytest.raises(ValueError, match="plane"):
            _solver_with_state().get_field_distribution("|E|2", "zz", 0.0)

    def test_xy_position_outside_stack(self):
        with pytest.raises(ValueError, match="outside the stack"):
            _solver_with_state().get_field_distribution("|E|2", "xy", 2.0)

    def test_no_simulation_returns_zeros(self):
        solver = TorcwaSolver({"name": "torcwa"})
        out = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert out.shape == (64, 64) and not out.any()


class TestExtraction:
    def test_xy_selects_layer_and_depth(self):
        solver = _solver_with_state()
        n = solver.FIELD_N_XY
        ones = MagicMock()
        ones.detach.return_value.cpu.return_value.numpy.return_value = np.ones((n, n))
        solver._last_sim.field_xy.return_value = ([ones] * 3, [ones] * 3)
        out = solver.get_field_distribution("|E|2", "xy", 0.2)
        np.testing.assert_allclose(out, 3.0)
        args, kwargs = solver._last_sim.field_xy.call_args
        assert args[0] == 1  # bottom slice is torcwa layer 1 (top-first order)
        assert kwargs["z_prop"] == pytest.approx(0.3)  # depth below that slice's top
        solver._last_sim.source_planewave.assert_called_once_with(
            amplitude=[0.0, 1.0], direction="forward", notation="ps"
        )

    def test_xz_maps_compass_z_to_torcwa_depth(self):
        solver = _solver_with_state()
        captured = {}

        def field_xz(x_axis, z_axis, y):
            captured["z"] = z_axis.numpy()
            captured["y"] = y
            arr = MagicMock()
            arr.detach.return_value.cpu.return_value.numpy.return_value = np.zeros(
                (len(x_axis), len(z_axis))
            )
            return [arr] * 3, [arr] * 3

        solver._last_sim.field_xz.side_effect = field_xz
        out = solver.get_field_distribution("Ex", "xz", 0.25)
        assert out.shape == (solver.FIELD_N_XY, solver.FIELD_N_Z)
        # first column is the stack bottom (torcwa depth 1.0), last the top (depth 0)
        assert captured["z"][0] == pytest.approx(1.0)
        assert captured["z"][-1] == pytest.approx(0.0)
        # COMPASS samples eps at cell centres; torcwa at cell corners (8x8 grid)
        assert captured["y"] == pytest.approx(0.25 - 0.5 / 8)

    def test_failure_falls_back_to_approximation(self, caplog):
        solver = _solver_with_state()
        solver._last_sim.field_yz.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.WARNING):
            out = solver.get_field_distribution("|E|2", "yz", 0.5)
        assert "not a solved field" in caplog.text
        assert out.shape == (64, 2)


class TestComponentFromField:
    E = [np.array([1 + 1j]), np.array([2.0]), np.array([3j])]
    H = [np.array([0.5]), np.array([1.0]), np.array([0.0])]

    @pytest.mark.parametrize("comp,expected", [("Ex", 2.0), ("Ey", 4.0), ("Ez", 9.0),
                                               ("|E|2", 15.0)])
    def test_intensities(self, comp, expected):
        assert TorcwaSolver._component_from_field(self.E, self.H, comp)[0] == pytest.approx(expected)

    def test_sz_is_compass_frame(self):
        # +x E with +y H carries power down (+z in torcwa) -> negative in COMPASS
        E = [np.array([1.0]), np.array([0.0]), np.array([0.0])]
        H = [np.array([0.0]), np.array([1.0]), np.array([0.0])]
        assert TorcwaSolver._component_from_field(E, H, "Sz")[0] == pytest.approx(-1.0)

    def test_unknown(self):
        with pytest.raises(ValueError):
            TorcwaSolver._component_from_field(self.E, self.H, "Hz")
