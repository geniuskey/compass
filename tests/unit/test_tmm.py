"""Comprehensive tests for the Transfer Matrix Method (TMM) solver.

Tests cover:
- Analytical Fresnel validation
- Anti-reflection coating design
- Energy conservation
- Normal and oblique incidence
- Brewster's angle
- Absorbing layers
- Field profiles
- Spectrum computation
- Solver adapter integration
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.solvers.tmm.tmm_core import (
    transfer_matrix_1d,
    tmm_field_profile,
    tmm_spectrum,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _air_glass_stack(n_glass: float = 1.5):
    """Create a simple air-glass interface stack."""
    n_layers = np.array([1.0, n_glass], dtype=complex)
    d_layers = np.array([np.inf, np.inf])
    return n_layers, d_layers


def _arc_on_silicon(wavelength: float = 0.55):
    """Create quarter-wave ARC on silicon for given wavelength.

    Optimal ARC: n_arc = sqrt(n_air * n_si), d_arc = lambda / (4 * n_arc).
    """
    n_si = 4.0  # approximate silicon n at 550nm (no absorption for simplicity)
    n_arc = np.sqrt(1.0 * n_si)  # ~2.0
    d_arc = wavelength / (4.0 * n_arc)
    n_layers = np.array([1.0, n_arc, n_si], dtype=complex)
    d_layers = np.array([np.inf, d_arc, np.inf])
    return n_layers, d_layers


# ---------------------------------------------------------------------------
# Test: Single interface Fresnel equations
# ---------------------------------------------------------------------------


class TestFresnelSingleInterface:
    """Validate TMM against analytical Fresnel equations for a single interface."""

    def test_air_glass_normal_incidence_te(self):
        """Air-glass interface at normal incidence should give ~4% reflection (TE)."""
        n_layers, d_layers = _air_glass_stack(n_glass=1.5)
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")

        # Analytical: R = ((n1 - n2)/(n1 + n2))^2 = ((1-1.5)/(1+1.5))^2 = 0.04
        R_analytical = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        assert abs(R - R_analytical) < 1e-10
        assert abs(T - (1.0 - R_analytical)) < 1e-10
        assert abs(A) < 1e-10

    def test_air_glass_normal_incidence_tm(self):
        """Air-glass interface at normal incidence: TE and TM should be identical."""
        n_layers, d_layers = _air_glass_stack(n_glass=1.5)
        R_te, T_te, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")
        R_tm, T_tm, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TM")
        assert abs(R_te - R_tm) < 1e-10
        assert abs(T_te - T_tm) < 1e-10

    def test_air_glass_reflection_value(self):
        """Air-glass gives approximately 4% reflection."""
        n_layers, d_layers = _air_glass_stack(n_glass=1.5)
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")
        assert abs(R - 0.04) < 1e-10
        assert abs(A) < 1e-10

    def test_high_index_contrast(self):
        """Higher index contrast gives more reflection."""
        n_lo, d_lo = _air_glass_stack(n_glass=1.5)
        n_hi, d_hi = _air_glass_stack(n_glass=3.0)

        R_lo, _, _ = transfer_matrix_1d(n_lo, d_lo, 0.55, 0.0, "TE")
        R_hi, _, _ = transfer_matrix_1d(n_hi, d_hi, 0.55, 0.0, "TE")

        assert R_hi > R_lo
        # n=3.0: R = ((1-3)/(1+3))^2 = 0.25
        assert abs(R_hi - 0.25) < 1e-10


# ---------------------------------------------------------------------------
# Test: Quarter-wave anti-reflection coating
# ---------------------------------------------------------------------------


class TestQuarterWaveARC:
    """Test quarter-wave anti-reflection coating on silicon."""

    def test_arc_minimizes_reflection(self):
        """Quarter-wave ARC should nearly eliminate reflection at design wavelength."""
        wavelength = 0.55
        n_layers, d_layers = _arc_on_silicon(wavelength)
        R, T, A = transfer_matrix_1d(n_layers, d_layers, wavelength, 0.0, "TE")

        # With ideal n_arc = sqrt(n_air * n_si) and d = lambda/(4*n_arc),
        # reflection should be very close to zero for non-absorbing materials
        assert R < 1e-10

    def test_arc_off_design_has_higher_reflection(self):
        """ARC designed for 550nm should have higher R at 450nm."""
        n_layers, d_layers = _arc_on_silicon(0.55)

        R_design, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")
        R_off, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.45, 0.0, "TE")

        assert R_off > R_design


# ---------------------------------------------------------------------------
# Test: Energy conservation R + T + A = 1
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Verify R + T + A = 1 for various configurations."""

    @pytest.mark.parametrize("polarization", ["TE", "TM"])
    def test_dielectric_interface(self, polarization):
        """R + T + A = 1 for lossless dielectric interface."""
        n_layers, d_layers = _air_glass_stack(1.5)
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, polarization)
        assert abs(R + T + A - 1.0) < 1e-10

    @pytest.mark.parametrize("theta", [0.0, 0.2, 0.5, 1.0])
    def test_oblique_incidence(self, theta):
        """R + T + A = 1 at various incidence angles."""
        n_layers, d_layers = _air_glass_stack(1.5)
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, theta, "TE")
        assert abs(R + T + A - 1.0) < 1e-10

    def test_absorbing_single_layer(self):
        """R + T + A = 1 for a stack with an absorbing layer."""
        # Air / absorbing film / glass
        n_layers = np.array([1.0, 2.0 + 0.5j, 1.5], dtype=complex)
        d_layers = np.array([np.inf, 0.1, np.inf])
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")
        assert abs(R + T + A - 1.0) < 1e-10
        assert A > 0  # Must have some absorption

    def test_multi_layer_stack(self):
        """R + T + A = 1 for a multi-layer stack."""
        n_layers = np.array([1.0, 1.4, 2.0, 1.5 + 0.01j, 3.5], dtype=complex)
        d_layers = np.array([np.inf, 0.1, 0.05, 0.2, np.inf])
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.3, "TM")
        assert abs(R + T + A - 1.0) < 1e-10

    @pytest.mark.parametrize("polarization", ["TE", "TM"])
    def test_thick_absorbing_layer(self, polarization):
        """Thick absorbing layer: T -> 0, A -> 1-R."""
        # Very thick silicon-like absorbing layer
        n_layers = np.array([1.0, 4.0 + 0.5j, 1.5], dtype=complex)
        d_layers = np.array([np.inf, 100.0, np.inf])
        R, T, A = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, polarization)
        assert abs(R + T + A - 1.0) < 1e-10
        assert T < 1e-10  # Essentially zero transmission
        assert A > 0.5  # Most light is absorbed


# ---------------------------------------------------------------------------
# Test: Oblique incidence and Brewster's angle
# ---------------------------------------------------------------------------


class TestObliqueIncidence:
    """Test behavior at oblique incidence including Brewster's angle."""

    def test_brewster_angle_tm_near_zero(self):
        """At Brewster's angle, TM reflection should approach zero."""
        n1, n2 = 1.0, 1.5
        # Brewster's angle: tan(theta_B) = n2/n1
        theta_brewster = np.arctan(n2 / n1)

        n_layers = np.array([n1, n2], dtype=complex)
        d_layers = np.array([np.inf, np.inf])

        R_tm, _, _ = transfer_matrix_1d(
            n_layers, d_layers, 0.55, theta_brewster, "TM"
        )
        assert R_tm < 1e-10

    def test_te_higher_than_tm_below_brewster(self):
        """Below Brewster's angle, TE reflection < TM reflection is NOT always true.
        Actually, for external reflection, TE > TM between normal and Brewster.
        """
        n_layers = np.array([1.0, 1.5], dtype=complex)
        d_layers = np.array([np.inf, np.inf])
        theta = 0.3  # ~17 degrees, below Brewster (~56 degrees)

        R_te, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, theta, "TE")
        R_tm, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, theta, "TM")

        # For external reflection (n1 < n2), R_TE > R_TM between 0 and Brewster
        assert R_te > R_tm

    def test_reflection_increases_near_grazing(self):
        """Reflection should increase for both TE and TM near grazing incidence."""
        n_layers = np.array([1.0, 1.5], dtype=complex)
        d_layers = np.array([np.inf, np.inf])

        R_normal, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, 0.0, "TE")
        R_grazing, _, _ = transfer_matrix_1d(n_layers, d_layers, 0.55, 1.4, "TE")

        assert R_grazing > R_normal


# ---------------------------------------------------------------------------
# Test: Field profile
# ---------------------------------------------------------------------------


class TestFieldProfile:
    """Test |E|^2 field profile computation."""

    def test_field_at_interface_incident_side(self):
        """At z=0 (first interface) from incident side, |E|^2 should be finite."""
        n_layers = np.array([1.0, 1.5], dtype=complex)
        d_layers = np.array([np.inf, np.inf])
        z_points = np.array([0.0])

        intensity = tmm_field_profile(n_layers, d_layers, 0.55, 0.0, "TE", z_points)
        assert np.all(np.isfinite(intensity))
        assert intensity[0] > 0

    def test_field_decays_in_absorbing_medium(self):
        """Field intensity should decrease through an absorbing layer."""
        # Air / thick absorbing film / glass
        n_layers = np.array([1.0, 2.0 + 1.0j, 1.5], dtype=complex)
        d_layers = np.array([np.inf, 2.0, np.inf])

        # Sample points through the absorbing film
        z_points = np.linspace(0.01, 1.9, 50)
        intensity = tmm_field_profile(n_layers, d_layers, 0.55, 0.0, "TE", z_points)

        # On average the field should decay (allow for some oscillation
        # due to interference, but the trend should be decreasing)
        # Check that the last quarter average is less than the first quarter average
        n = len(intensity)
        first_quarter = np.mean(intensity[: n // 4])
        last_quarter = np.mean(intensity[3 * n // 4:])
        assert last_quarter < first_quarter

    def test_field_correct_number_of_points(self):
        """Output should have same number of points as z_points input."""
        n_layers = np.array([1.0, 1.5, 2.0], dtype=complex)
        d_layers = np.array([np.inf, 0.5, np.inf])
        z_points = np.linspace(-0.5, 1.0, 100)

        intensity = tmm_field_profile(n_layers, d_layers, 0.55, 0.0, "TE", z_points)
        assert len(intensity) == 100

    def test_field_all_positive(self):
        """|E|^2 should always be non-negative."""
        n_layers = np.array([1.0, 2.0, 1.5 + 0.1j, 3.5], dtype=complex)
        d_layers = np.array([np.inf, 0.1, 0.3, np.inf])
        z_points = np.linspace(-0.2, 0.6, 200)

        intensity = tmm_field_profile(n_layers, d_layers, 0.55, 0.0, "TE", z_points)
        assert np.all(intensity >= 0)


# ---------------------------------------------------------------------------
# Test: Spectrum computation
# ---------------------------------------------------------------------------


class TestTMMSpectrum:
    """Test multi-wavelength spectrum computation."""

    def test_spectrum_correct_length(self):
        """Output arrays should match number of wavelengths."""
        wavelengths = np.linspace(0.4, 0.7, 31)
        n_layers_func = lambda wl: np.array([1.0, 1.5], dtype=complex)
        d_layers = np.array([np.inf, np.inf])

        R, T, A = tmm_spectrum(n_layers_func, d_layers, wavelengths, 0.0, "TE")
        assert len(R) == 31
        assert len(T) == 31
        assert len(A) == 31

    def test_spectrum_energy_conservation(self):
        """R + T + A = 1 at every wavelength."""
        wavelengths = np.linspace(0.4, 0.7, 20)

        def n_func(wl):
            # Dispersive film
            n_film = 1.45 + 0.01 / wl**2
            return np.array([1.0, n_film, 1.5], dtype=complex)

        d_layers = np.array([np.inf, 0.2, np.inf])
        R, T, A = tmm_spectrum(n_func, d_layers, wavelengths, 0.0, "TE")

        total = R + T + A
        assert np.allclose(total, 1.0, atol=1e-10)

    def test_single_wavelength_spectrum(self):
        """Spectrum with single wavelength should work."""
        wavelengths = np.array([0.55])
        n_func = lambda wl: np.array([1.0, 1.5], dtype=complex)
        d_layers = np.array([np.inf, np.inf])

        R, T, A = tmm_spectrum(n_func, d_layers, wavelengths)
        assert len(R) == 1
        assert abs(R[0] - 0.04) < 1e-10


# ---------------------------------------------------------------------------
# Test: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test error handling for invalid inputs."""

    def test_too_few_layers(self):
        """Must have at least 2 layers."""
        with pytest.raises(ValueError, match="at least 2 layers"):
            transfer_matrix_1d(np.array([1.0]), np.array([np.inf]), 0.55)

    def test_mismatched_lengths(self):
        """n_layers and d_layers must have same length."""
        with pytest.raises(ValueError, match="same length"):
            transfer_matrix_1d(
                np.array([1.0, 1.5]),
                np.array([np.inf, np.inf, np.inf]),
                0.55,
            )

    def test_zero_wavelength(self):
        """Wavelength must be positive."""
        with pytest.raises(ValueError, match="positive"):
            transfer_matrix_1d(np.array([1.0, 1.5]), np.array([np.inf, np.inf]), 0.0)

    def test_negative_wavelength(self):
        """Negative wavelength is rejected."""
        with pytest.raises(ValueError, match="positive"):
            transfer_matrix_1d(np.array([1.0, 1.5]), np.array([np.inf, np.inf]), -0.55)

    def test_invalid_polarization(self):
        """Invalid polarization raises ValueError."""
        with pytest.raises(ValueError, match="polarization"):
            transfer_matrix_1d(
                np.array([1.0, 1.5]),
                np.array([np.inf, np.inf]),
                0.55,
                polarization="XY",
            )


# ---------------------------------------------------------------------------
# Test: Solver adapter (TMMSolver)
# ---------------------------------------------------------------------------


class TestTMMSolverAdapter:
    """Test TMMSolver integration with SolverBase/SolverFactory."""

    def test_factory_registration(self):
        """TMM should be creatable via SolverFactory."""
        from compass.solvers.base import SolverFactory

        solver = SolverFactory.create(
            "tmm",
            {"name": "tmm", "type": "tmm", "params": {}},
        )
        assert solver.name == "tmm"
        assert solver.solver_type == "tmm"

    def test_factory_list_includes_tmm(self):
        """SolverFactory.list_solvers should include 'tmm'."""
        from compass.solvers.base import SolverFactory

        # Trigger import
        SolverFactory.create(
            "tmm", {"name": "tmm", "type": "tmm", "params": {}}
        )
        assert "tmm" in SolverFactory.list_solvers()

    def test_setup_geometry_accepts_pixel_stack(self):
        """setup_geometry should accept a PixelStack."""
        from compass.solvers.tmm.tmm_solver import TMMSolver

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})

        # Create minimal PixelStack
        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {
                    "silicon": {"thickness": 3.0},
                },
            }
        }
        from compass.geometry.pixel_stack import PixelStack

        ps = PixelStack(config)
        solver.setup_geometry(ps)

        assert solver._pixel_stack is ps
        assert len(solver._layer_materials) > 0

    def test_setup_geometry_rejects_none(self):
        """setup_geometry raises on None input."""
        from compass.solvers.tmm.tmm_solver import TMMSolver

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})
        with pytest.raises(ValueError, match="must not be None"):
            solver.setup_geometry(None)

    def test_setup_source(self):
        """setup_source configures wavelength and angle."""
        from compass.solvers.tmm.tmm_solver import TMMSolver

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})
        source_cfg = {
            "wavelength": {"mode": "sweep", "sweep": {"start": 0.4, "stop": 0.7, "step": 0.05}},
            "angle": {"theta_deg": 0.0},
            "polarization": "unpolarized",
        }
        solver.setup_source(source_cfg)
        assert solver._source is not None
        assert solver._source.n_wavelengths > 0

    def test_full_run(self):
        """Full TMM solver run produces valid SimulationResult."""
        from compass.solvers.tmm.tmm_solver import TMMSolver
        from compass.core.types import SimulationResult

        solver = TMMSolver(
            {"name": "tmm", "type": "tmm", "params": {"field_resolution": 100}}
        )

        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {
                    "silicon": {"thickness": 3.0},
                },
            }
        }
        from compass.geometry.pixel_stack import PixelStack

        ps = PixelStack(config)
        solver.setup_geometry(ps)

        source_cfg = {
            "wavelength": {"mode": "list", "values": [0.45, 0.55, 0.65]},
            "angle": {"theta_deg": 0.0},
            "polarization": "unpolarized",
        }
        solver.setup_source(source_cfg)

        result = solver.run()
        assert isinstance(result, SimulationResult)
        assert len(result.wavelengths) == 3
        assert result.reflection is not None
        assert result.transmission is not None
        assert result.absorption is not None

        # Energy conservation
        total = result.reflection + result.transmission + result.absorption
        assert np.allclose(total, 1.0, atol=0.01)

        # QE per pixel should have entries
        assert len(result.qe_per_pixel) > 0

    def test_run_without_setup_geometry_raises(self):
        """run() before setup_geometry() raises RuntimeError."""
        from compass.solvers.tmm.tmm_solver import TMMSolver

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})
        solver.setup_source({
            "wavelength": {"mode": "single", "value": 0.55},
            "polarization": "TE",
        })
        with pytest.raises(RuntimeError, match="setup_geometry"):
            solver.run()

    def test_run_without_setup_source_raises(self):
        """run() before setup_source() raises RuntimeError."""
        from compass.solvers.tmm.tmm_solver import TMMSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})
        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {"silicon": {"thickness": 3.0}},
            }
        }
        ps = PixelStack(config)
        solver.setup_geometry(ps)
        with pytest.raises(RuntimeError, match="setup_source"):
            solver.run()

    def test_get_field_distribution(self):
        """get_field_distribution returns 2D array after run."""
        from compass.solvers.tmm.tmm_solver import TMMSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = TMMSolver(
            {"name": "tmm", "type": "tmm", "params": {"field_resolution": 50}}
        )
        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {"silicon": {"thickness": 3.0}},
            }
        }
        ps = PixelStack(config)
        solver.setup_geometry(ps)
        solver.setup_source({
            "wavelength": {"mode": "single", "value": 0.55},
            "polarization": "TE",
        })
        solver.run()

        field = solver.get_field_distribution("|E|2", "xz", 0.0)
        assert field.ndim == 2
        assert field.shape[0] == 64  # lateral points
        assert np.all(np.isfinite(field))

    def test_validate_energy_balance(self):
        """validate_energy_balance should pass for TMM results."""
        from compass.solvers.tmm.tmm_solver import TMMSolver
        from compass.geometry.pixel_stack import PixelStack

        solver = TMMSolver({"name": "tmm", "type": "tmm", "params": {}})
        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {"silicon": {"thickness": 3.0}},
            }
        }
        ps = PixelStack(config)
        solver.setup_geometry(ps)
        solver.setup_source({
            "wavelength": {"mode": "single", "value": 0.55},
            "polarization": "TE",
        })
        result = solver.run()
        assert solver.validate_energy_balance(result) is True
