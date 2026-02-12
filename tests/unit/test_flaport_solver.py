"""Unit tests for the flaport FDTD solver adapter."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pixel_stack_mock():
    """Create a mock PixelStack with minimal attributes."""
    ps = MagicMock()
    ps.layers = [MagicMock()]
    ps.unit_cell = (2, 2)
    ps.domain_size = (2.0, 2.0)
    ps.z_range = (0.0, 5.0)
    ps.bayer_map = [["R", "G"], ["G", "B"]]
    ps.photodiodes = []
    ps.get_permittivity_grid.return_value = np.ones((51, 51, 51), dtype=complex) * 1.5
    return ps


def _make_source_config(wavelength: float = 0.55):
    """Create a minimal source config."""
    return {
        "wavelength": {"mode": "single", "value": wavelength},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "TE",
    }


def _make_fdtd_mock():
    """Build a comprehensive fdtd module mock for solver tests."""
    fdtd = MagicMock()

    # Grid mock — supports item assignment for boundaries/source
    # inverse_permittivity is a MagicMock to avoid shape mismatch errors
    grid = MagicMock()

    fdtd.Grid.return_value = grid
    fdtd.PeriodicBoundary.return_value = MagicMock()
    fdtd.PML.return_value = MagicMock()
    fdtd.PlaneSource.return_value = MagicMock()

    return fdtd, grid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlaportPMLSliceIndexing:
    """Verify PML uses slice indexing, not integer indexing."""

    @patch.dict("sys.modules", {"fdtd": MagicMock()})
    def test_pml_slice_indexing_no_exception(self):
        """_build_grid should use slice indexing for PML (no TypeError)."""
        import sys

        fdtd_mod = sys.modules["fdtd"]
        fdtd_mod.Grid.return_value = MagicMock()
        fdtd_mod.PeriodicBoundary.return_value = MagicMock()
        fdtd_mod.PML.return_value = MagicMock()
        fdtd_mod.PlaneSource.return_value = MagicMock()
        fdtd_mod.BlockDetector.return_value = MagicMock()

        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        # Should not raise — slice PML indexing
        grid, damping = solver._build_grid(
            fdtd_mod,
            nx=51,
            ny=51,
            nz=81,
            grid_spacing_m=2e-8,
            pml_layers=15,
            wavelength_m=5.5e-7,
            pol="TE",
            eps_3d=None,
        )
        assert grid is not None
        assert damping is None  # No damping for vacuum run

    @patch.dict("sys.modules", {"fdtd": MagicMock()})
    def test_pml_called_twice(self):
        """PML is created for both z_low and z_high."""
        import sys

        fdtd_mod = sys.modules["fdtd"]
        fdtd_mod.Grid.return_value = MagicMock()
        fdtd_mod.PeriodicBoundary.return_value = MagicMock()
        fdtd_mod.PML.return_value = MagicMock()
        fdtd_mod.PlaneSource.return_value = MagicMock()
        fdtd_mod.BlockDetector.return_value = MagicMock()

        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        solver._build_grid(
            fdtd_mod,
            nx=51,
            ny=51,
            nz=81,
            grid_spacing_m=2e-8,
            pml_layers=15,
            wavelength_m=5.5e-7,
            pol="TE",
            eps_3d=None,
        )
        assert fdtd_mod.PML.call_count == 2


class TestFlaportVacuumEnergyConservation:
    """Vacuum run should give T~1, R~0, A~0."""

    def test_vacuum_energy_conservation(self):
        """With vacuum (no absorption), R~0, T~1, A~0."""
        fdtd_mod, _grid = _make_fdtd_mock()

        # Two-pass normalization: both reference and structure runs return same fluxes
        # Sz_above = +0.1 (small upward), Sz_below = -0.5 (downward), Sz_trans = -0.5
        # Reference: P_inc = abs(Sz_ref_below) = 0.5
        # Structure (vacuum): R = (0.1 - 0.1) / 0.5 = 0.0, T = 0.5/0.5 = 1.0, A = 0.0
        poynting_vacuum = (0.1, -0.5, -0.5)

        with patch.dict("sys.modules", {"fdtd": fdtd_mod}):
            from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

            solver = FlaportFdtdSolver(
                {"params": {"grid_spacing": 0.04, "runtime": 100, "pml_layers": 10}},
                device="cpu",
            )
            ps = _make_pixel_stack_mock()
            ps.get_permittivity_grid.return_value = np.ones((26, 26, 26), dtype=complex)

            solver.setup_geometry(ps)
            solver.setup_source(_make_source_config())

            with patch.object(
                FlaportFdtdSolver, "_build_grid", return_value=(MagicMock(), None)
            ), patch.object(
                FlaportFdtdSolver,
                "_run_and_poynting",
                return_value=poynting_vacuum,
            ):
                result = solver.run()

        assert result.reflection is not None
        assert result.transmission is not None
        assert result.absorption is not None
        np.testing.assert_allclose(result.reflection[0], 0.0, atol=0.01)
        np.testing.assert_allclose(result.transmission[0], 1.0, atol=0.01)
        np.testing.assert_allclose(result.absorption[0], 0.0, atol=0.02)


class TestFlaportEnergyBalance:
    """R + T + A should be close to 1.0 for structure runs."""

    def test_energy_balance(self):
        """R + T + A should be within 2% of 1.0."""
        fdtd_mod, _grid = _make_fdtd_mock()

        # Two-pass normalization:
        # Reference run: Sz_ref_above=0.1, Sz_ref_below=-0.5 → P_inc = 0.5
        # Structure run: Sz_above=0.15, Sz_below=-0.40, Sz_trans=-0.22
        # R = (0.15 - 0.1) / 0.5 = 0.10
        # T = 0.22 / 0.5 = 0.44
        # A = 1 - 0.10 - 0.44 = 0.46
        poynting_ref = (0.1, -0.5, -0.5)
        poynting_structure = (0.15, -0.40, -0.22)

        with patch.dict("sys.modules", {"fdtd": fdtd_mod}):
            from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

            solver = FlaportFdtdSolver(
                {"params": {"grid_spacing": 0.04, "runtime": 100, "pml_layers": 10}},
                device="cpu",
            )
            ps = _make_pixel_stack_mock()
            solver.setup_geometry(ps)
            solver.setup_source(_make_source_config())

            with patch.object(
                FlaportFdtdSolver, "_build_grid", return_value=(MagicMock(), None)
            ), patch.object(
                FlaportFdtdSolver,
                "_run_and_poynting",
                side_effect=[poynting_ref, poynting_structure],
            ):
                result = solver.run()

        R = result.reflection[0]
        T = result.transmission[0]
        A = result.absorption[0]
        total = R + T + A
        assert abs(total - 1.0) < 0.02, f"R+T+A = {total:.4f}, expected ~1.0"


class TestFlaportTimeAvgIntensity:
    """Tests for the _time_avg_intensity static method."""

    def test_time_avg_uses_last_third(self):
        """Averaging uses only the last 1/3 of timesteps."""
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        E = np.zeros((90, 5, 5, 3))
        # First 60 timesteps: value 0
        # Last 30 timesteps: value 1
        E[60:, :, :, :] = 1.0

        intensity = FlaportFdtdSolver._time_avg_intensity(E)
        # Last 30 out of 90 → all 1.0 → |E|² = 3.0 per point
        expected = 3.0
        np.testing.assert_allclose(intensity, expected, atol=0.01)

    def test_time_avg_zero_field(self):
        """Zero field should give zero intensity."""
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        E = np.zeros((30, 5, 5, 3))
        assert FlaportFdtdSolver._time_avg_intensity(E) == pytest.approx(0.0)


class TestFlaportSolverSetup:
    """Basic setup and error handling tests."""

    def test_setup_geometry_rejects_none(self):
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        with pytest.raises(ValueError, match="must not be None"):
            solver.setup_geometry(None)

    def test_setup_geometry_rejects_empty_layers(self):
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        ps = MagicMock()
        ps.layers = []
        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        with pytest.raises(ValueError, match="at least one layer"):
            solver.setup_geometry(ps)

    def test_run_without_setup_raises(self):
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        with pytest.raises(RuntimeError, match="setup_geometry"):
            solver.run()

    def test_run_without_source_raises(self):
        from compass.solvers.fdtd.flaport_solver import FlaportFdtdSolver

        solver = FlaportFdtdSolver({"params": {}}, device="cpu")
        solver._pixel_stack = _make_pixel_stack_mock()
        with pytest.raises(RuntimeError, match="setup_source"):
            solver.run()
