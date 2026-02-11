"""Unit tests for stability diagnostics (re-exported from compass.solvers.rcwa.stability).

Tests cover:
- StabilityDiagnostics pre/post simulation checks
- AdaptivePrecisionRunner fallback strategy
- PrecisionManager configuration
- Additional edge cases not covered in test_stability.py
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from compass.core.types import Layer, SimulationResult
from compass.diagnostics.stability_diagnostics import (
    AdaptivePrecisionRunner,
    PrecisionManager,
    StabilityDiagnostics,
)
from compass.solvers.rcwa.stability import (
    EigenvalueStabilizer,
    LiFactorization,
    StableSMatrixAlgorithm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_layer(layer_name, thickness, is_patterned=False):
    """Create a mock Layer with the given attributes."""
    layer = MagicMock(spec=Layer)
    layer.name = layer_name
    layer.thickness = thickness
    layer.is_patterned = is_patterned
    return layer


def _make_pixel_stack(layers=None, has_patterned=False):
    """Create a mock PixelStack with specified layers."""
    stack = MagicMock()
    if layers is None:
        layers = [
            _make_mock_layer("oxide", 0.5, is_patterned=False),
            _make_mock_layer("silicon", 1.0, is_patterned=has_patterned),
        ]
    stack.layers = layers
    return stack


def _make_result(
    qe_range=(0.0, 1.0),
    reflection=None,
    transmission=None,
    absorption=None,
    n_wl=5,
):
    """Create a SimulationResult for diagnostics testing."""
    wavelengths = np.linspace(0.4, 0.7, n_wl)
    qe = np.linspace(qe_range[0], qe_range[1], n_wl)
    return SimulationResult(
        qe_per_pixel={"R_0_0": qe},
        wavelengths=wavelengths,
        reflection=reflection,
        transmission=transmission,
        absorption=absorption,
        metadata={},
    )


# ---------------------------------------------------------------------------
# StabilityDiagnostics.pre_simulation_check Tests
# ---------------------------------------------------------------------------


class TestPreSimulationCheck:
    """Tests for StabilityDiagnostics.pre_simulation_check."""

    def test_no_warnings_for_safe_config(self):
        """Safe configuration produces no warnings."""
        stack = _make_pixel_stack()
        config = {
            "params": {"fourier_order": [5, 5]},
            "stability": {
                "precision_strategy": "mixed",
                "fourier_factorization": "li_inverse",
            },
        }
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert len(warnings) == 0

    def test_warns_large_matrix_float32(self):
        """Warns when large matrix size combined with float32 strategy."""
        stack = _make_pixel_stack()
        # order [15, 15] -> matrix = (2*15+1)^2 = 961
        config = {
            "params": {"fourier_order": [15, 15]},
            "stability": {"precision_strategy": "float32"},
        }
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert any("float32" in w for w in warnings)

    def test_no_warn_large_matrix_mixed_precision(self):
        """No warning for large matrix with mixed precision strategy."""
        stack = _make_pixel_stack()
        config = {
            "params": {"fourier_order": [15, 15]},
            "stability": {"precision_strategy": "mixed"},
        }
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert not any("float32" in w for w in warnings)

    def test_warns_thick_layer(self):
        """Warns about layers thicker than 2um."""
        thick_layer = _make_mock_layer("bulk_si", 3.0, is_patterned=False)
        stack = _make_pixel_stack(layers=[thick_layer])
        config = {"params": {"fourier_order": [5, 5]}, "stability": {}}
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert any("Thick layer" in w for w in warnings)

    def test_no_warn_normal_thickness(self):
        """No warning for layers at or below 2um thickness."""
        normal_layer = _make_mock_layer("oxide", 2.0, is_patterned=False)
        stack = _make_pixel_stack(layers=[normal_layer])
        config = {"params": {"fourier_order": [5, 5]}, "stability": {}}
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert not any("Thick layer" in w for w in warnings)

    def test_warns_patterned_with_naive_factorization(self):
        """Warns when patterned layers use naive factorization."""
        stack = _make_pixel_stack(has_patterned=True)
        config = {
            "params": {"fourier_order": [5, 5]},
            "stability": {"fourier_factorization": "naive"},
        }
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert any("naive" in w.lower() for w in warnings)

    def test_no_warn_patterned_with_inverse_rule(self):
        """No warning for patterned layers with li_inverse factorization."""
        stack = _make_pixel_stack(has_patterned=True)
        config = {
            "params": {"fourier_order": [5, 5]},
            "stability": {"fourier_factorization": "li_inverse"},
        }
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        assert not any("naive" in w.lower() for w in warnings)

    def test_default_fourier_order_used(self):
        """Default fourier_order [9, 9] is used when not specified."""
        stack = _make_pixel_stack()
        # With no params, default order = [9,9], matrix = (2*9+1)^2 = 361
        # With default strategy "mixed", no float32 warning
        config = {"stability": {}}
        warnings = StabilityDiagnostics.pre_simulation_check(stack, config)
        # Should not crash; warnings depend on other checks
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# StabilityDiagnostics.post_simulation_check Tests
# ---------------------------------------------------------------------------


class TestPostSimulationCheck:
    """Tests for StabilityDiagnostics.post_simulation_check."""

    def test_valid_result_empty_report(self):
        """Physically plausible result produces empty report."""
        result = _make_result(
            qe_range=(0.1, 0.8),
            reflection=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            transmission=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            absorption=np.array([0.7, 0.7, 0.7, 0.7, 0.7]),
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert len(report) == 0

    def test_detects_qe_out_of_range_negative(self):
        """Detects QE values significantly below zero."""
        result = _make_result(qe_range=(-0.1, 0.5))
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "R_0_0" in report
        assert report["R_0_0"]["status"] == "FAILED"

    def test_detects_qe_out_of_range_above_one(self):
        """Detects QE values significantly above 1.0."""
        result = _make_result(qe_range=(0.5, 1.2))
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "R_0_0" in report
        assert report["R_0_0"]["status"] == "FAILED"

    def test_qe_slightly_out_of_range_passes(self):
        """QE values within [-0.01, 1.01] pass the check."""
        result = SimulationResult(
            qe_per_pixel={"px": np.array([-0.005, 0.5, 1.005])},
            wavelengths=np.array([0.4, 0.55, 0.7]),
            metadata={},
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "px" not in report

    def test_detects_nan_in_reflection(self):
        """Detects NaN in reflection array."""
        result = _make_result(
            reflection=np.array([0.1, np.nan, 0.1, 0.1, 0.1]),
            transmission=np.array([0.3] * 5),
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "reflection" in report
        assert report["reflection"]["status"] == "FAILED"

    def test_detects_inf_in_transmission(self):
        """Detects Inf in transmission array."""
        result = _make_result(
            reflection=np.array([0.1] * 5),
            transmission=np.array([0.3, np.inf, 0.3, 0.3, 0.3]),
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "transmission" in report

    def test_detects_energy_violation(self):
        """Detects R + T + A significantly different from 1."""
        result = _make_result(
            reflection=np.array([0.5] * 5),
            transmission=np.array([0.5] * 5),
            absorption=np.array([0.5] * 5),  # total = 1.5
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "energy_balance" in report
        assert report["energy_balance"]["status"] == "WARNING"

    def test_no_energy_check_when_rta_missing(self):
        """No energy check when reflection or transmission is None."""
        result = _make_result(reflection=None, transmission=None)
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "energy_balance" not in report

    def test_multiple_issues_all_reported(self):
        """Multiple issues are all reported simultaneously."""
        result = SimulationResult(
            qe_per_pixel={"bad_px": np.array([-0.5, 1.5, 0.5])},
            wavelengths=np.array([0.4, 0.55, 0.7]),
            reflection=np.array([np.nan, 0.5, 0.5]),
            transmission=np.array([0.5, 0.5, 0.5]),
            absorption=np.array([0.5, 0.5, 0.5]),
            metadata={},
        )
        report = StabilityDiagnostics.post_simulation_check(result)
        assert "bad_px" in report
        assert "reflection" in report


# ---------------------------------------------------------------------------
# AdaptivePrecisionRunner Tests
# ---------------------------------------------------------------------------


class TestAdaptivePrecisionRunner:
    """Tests for AdaptivePrecisionRunner.run_with_fallback."""

    def test_first_strategy_succeeds(self):
        """Returns result from first strategy when energy is conserved."""
        runner = AdaptivePrecisionRunner(tolerance=0.02)
        mock_solver = MagicMock()
        mock_solver.run_single_wavelength.return_value = {"R": 0.3, "T": 0.7, "data": "ok"}

        result = runner.run_with_fallback(mock_solver, 0.55, {})

        assert result["data"] == "ok"
        assert mock_solver.run_single_wavelength.call_count == 1

    def test_falls_back_on_energy_violation(self):
        """Falls back to next strategy when energy conservation fails."""
        runner = AdaptivePrecisionRunner(tolerance=0.02)
        mock_solver = MagicMock()
        # First call: energy violation
        # Second call: valid
        mock_solver.run_single_wavelength.side_effect = [
            {"R": 0.6, "T": 0.6},  # over-unity
            {"R": 0.3, "T": 0.7, "data": "fallback"},
        ]

        result = runner.run_with_fallback(mock_solver, 0.55, {})

        assert result["data"] == "fallback"
        assert mock_solver.run_single_wavelength.call_count == 2

    def test_falls_back_on_exception(self):
        """Falls back when solver raises an exception."""
        runner = AdaptivePrecisionRunner(tolerance=0.02)
        mock_solver = MagicMock()
        mock_solver.run_single_wavelength.side_effect = [
            RuntimeError("GPU OOM"),
            {"R": 0.2, "T": 0.8, "data": "cpu_result"},
        ]

        result = runner.run_with_fallback(mock_solver, 0.55, {})

        assert result["data"] == "cpu_result"

    def test_raises_when_all_strategies_fail(self):
        """Raises RuntimeError when all three strategies fail."""
        runner = AdaptivePrecisionRunner(tolerance=0.02)
        mock_solver = MagicMock()
        mock_solver.run_single_wavelength.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All precision strategies failed"):
            runner.run_with_fallback(mock_solver, 0.55, {})

        assert mock_solver.run_single_wavelength.call_count == 3

    def test_custom_tolerance(self):
        """Custom tolerance affects energy validation."""
        runner = AdaptivePrecisionRunner(tolerance=0.5)
        mock_solver = MagicMock()
        # R + T = 1.3, which exceeds default but within 0.5 tolerance
        mock_solver.run_single_wavelength.return_value = {"R": 0.6, "T": 0.7}

        # Should NOT fall back since total < 1.0 + 0.5 = 1.5
        result = runner.run_with_fallback(mock_solver, 0.55, {})
        assert mock_solver.run_single_wavelength.call_count == 1

    def test_strategy_order(self):
        """Strategies are tried in order: GPU-f32, GPU-f64, CPU-f64."""
        runner = AdaptivePrecisionRunner(tolerance=0.02)
        mock_solver = MagicMock()

        call_params = []
        def record_call(**kwargs):
            call_params.append(kwargs)
            raise RuntimeError("fail")

        mock_solver.run_single_wavelength.side_effect = record_call

        with pytest.raises(RuntimeError):
            runner.run_with_fallback(mock_solver, 0.55, {})

        assert call_params[0]["dtype"] == "complex64"
        assert call_params[0]["device"] == "cuda"
        assert call_params[1]["dtype"] == "complex128"
        assert call_params[1]["device"] == "cuda"
        assert call_params[2]["dtype"] == "complex128"
        assert call_params[2]["device"] == "cpu"


# ---------------------------------------------------------------------------
# PrecisionManager Tests (supplementary to test_stability.py)
# ---------------------------------------------------------------------------


class TestPrecisionManagerConfigure:
    """Tests for PrecisionManager.configure."""

    def test_configure_with_empty_config(self):
        """Does not raise with empty config."""
        PrecisionManager.configure({})

    def test_configure_with_stability_section(self):
        """Accepts config with stability section."""
        PrecisionManager.configure({"stability": {"allow_tf32": False}})

    def test_mixed_precision_eigen_preserves_shape(self):
        """Output eigenvalues/vectors have correct shapes."""
        A = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
        vals, vecs = PrecisionManager.mixed_precision_eigen(A)
        assert vals.shape == (8,)
        assert vecs.shape == (8, 8)

    def test_mixed_precision_eigen_complex128_passthrough(self):
        """complex128 input keeps complex128 output dtype."""
        A = np.random.randn(4, 4).astype(np.complex128)
        vals, vecs = PrecisionManager.mixed_precision_eigen(A)
        assert vals.dtype == np.complex128
        assert vecs.dtype == np.complex128


# ---------------------------------------------------------------------------
# Additional EigenvalueStabilizer edge case tests
# ---------------------------------------------------------------------------


class TestEigenvalueStabilizerEdgeCases:
    """Additional edge case tests for EigenvalueStabilizer."""

    def test_fix_degenerate_no_degeneracy(self):
        """Non-degenerate eigenvalues are unchanged."""
        eigenvalues = np.array([1.0 + 0j, 5.0 + 0j, 10.0 + 0j])
        eigenvectors = np.eye(3, dtype=complex)
        vals, vecs = EigenvalueStabilizer.fix_degenerate_eigenvalues(
            eigenvalues, eigenvectors, broadening=1e-10
        )
        np.testing.assert_array_equal(vals, eigenvalues)

    def test_select_propagation_direction_pure_real(self):
        """Pure real positive eigenvalues give positive real sqrt."""
        eigenvalues = np.array([1.0, 4.0, 16.0])
        sqrt_ev = EigenvalueStabilizer.select_propagation_direction(eigenvalues)
        np.testing.assert_allclose(np.real(sqrt_ev), [1.0, 2.0, 4.0], atol=1e-10)
        assert np.all(np.real(sqrt_ev) >= 0)

    def test_select_propagation_direction_negative_eigenvalues(self):
        """Negative real eigenvalues give evanescent modes with correct sign."""
        eigenvalues = np.array([-1.0, -4.0])
        sqrt_ev = EigenvalueStabilizer.select_propagation_direction(eigenvalues)
        # For negative reals, sqrt gives imaginary, real part should be >= 0
        for s in sqrt_ev:
            assert np.real(s) >= -1e-10

    def test_energy_conservation_absorbing(self):
        """For absorbing media, R + T < 1 is valid."""
        R = np.array([0.1, 0.2])
        T = np.array([0.3, 0.2])
        # total = 0.4, 0.4 → far from 1 but below, which is valid for absorbing
        result = EigenvalueStabilizer.validate_energy_conservation(R, T, tolerance=0.05)
        assert result["valid"] is True
        assert result["max_violation"] == pytest.approx(0.6)

    def test_energy_conservation_empty_violations(self):
        """No over-unity means empty violation_indices."""
        R = np.array([0.5])
        T = np.array([0.5])
        result = EigenvalueStabilizer.validate_energy_conservation(R, T)
        assert result["violation_indices"] == []
