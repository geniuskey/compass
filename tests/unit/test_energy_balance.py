"""Unit tests for EnergyBalance."""

import numpy as np
import pytest

from compass.analysis.energy_balance import EnergyBalance
from compass.core.types import SimulationResult


def _make_result(
    reflection=None, transmission=None, absorption=None, n_wl=3
):
    """Helper to create a SimulationResult with given R/T/A arrays."""
    wavelengths = np.linspace(0.4, 0.7, n_wl)
    return SimulationResult(
        qe_per_pixel={},
        wavelengths=wavelengths,
        reflection=reflection,
        transmission=transmission,
        absorption=absorption,
    )


class TestEnergyBalanceCheck:
    """Tests for EnergyBalance.check."""

    def test_perfect_energy_conservation(self):
        """R + T + A = 1 exactly passes validation."""
        R = np.array([0.1, 0.2, 0.3])
        T = np.array([0.3, 0.3, 0.2])
        A = np.array([0.6, 0.5, 0.5])
        result = _make_result(reflection=R, transmission=T, absorption=A)
        check = EnergyBalance.check(result)
        assert check["valid"] is True
        assert check["max_error"] == pytest.approx(0.0, abs=1e-12)
        assert check["mean_error"] == pytest.approx(0.0, abs=1e-12)

    def test_small_violation_within_tolerance(self):
        """Small energy violation within default tolerance (1%) passes."""
        R = np.array([0.1, 0.2])
        T = np.array([0.3, 0.3])
        A = np.array([0.605, 0.495])  # total = 1.005, 0.995
        result = _make_result(reflection=R, transmission=T, absorption=A, n_wl=2)
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_large_violation_fails(self):
        """Large energy violation exceeds tolerance and fails."""
        R = np.array([0.5, 0.5])
        T = np.array([0.5, 0.5])
        A = np.array([0.5, 0.5])  # total = 1.5
        result = _make_result(reflection=R, transmission=T, absorption=A, n_wl=2)
        check = EnergyBalance.check(result)
        assert check["valid"] is False
        assert check["max_error"] == pytest.approx(0.5)

    def test_custom_tolerance(self):
        """Custom tolerance is respected."""
        R = np.array([0.4])
        T = np.array([0.4])
        A = np.array([0.22])  # total = 1.02
        result = _make_result(reflection=R, transmission=T, absorption=A, n_wl=1)
        # Default 1% tolerance: should pass
        assert EnergyBalance.check(result, tolerance=0.03)["valid"] is True
        # Very tight tolerance: should fail
        assert EnergyBalance.check(result, tolerance=0.01)["valid"] is False

    def test_missing_reflection_returns_valid(self):
        """Missing reflection data returns valid with message."""
        result = _make_result(reflection=None, transmission=np.array([0.5]))
        check = EnergyBalance.check(result)
        assert check["valid"] is True
        assert "Insufficient data" in check.get("message", "")

    def test_missing_transmission_returns_valid(self):
        """Missing transmission data returns valid with message."""
        result = _make_result(reflection=np.array([0.5]), transmission=None)
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_absorption_inferred_when_none(self):
        """When absorption is None, A = 1 - R - T is used."""
        R = np.array([0.3, 0.2])
        T = np.array([0.2, 0.3])
        result = _make_result(reflection=R, transmission=T, absorption=None, n_wl=2)
        check = EnergyBalance.check(result)
        # A = 1 - R - T, so total = R + T + (1 - R - T) = 1 exactly
        assert check["valid"] is True
        assert check["max_error"] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(check["A"], 1.0 - R - T)

    def test_per_wavelength_errors_returned(self):
        """per_wavelength field has correct shape and values."""
        R = np.array([0.3, 0.3, 0.3])
        T = np.array([0.3, 0.3, 0.3])
        A = np.array([0.4, 0.45, 0.35])  # errors: 0.0, 0.05, 0.05
        result = _make_result(reflection=R, transmission=T, absorption=A)
        check = EnergyBalance.check(result)
        assert check["per_wavelength"].shape == (3,)
        assert check["per_wavelength"][0] == pytest.approx(0.0)
        assert check["per_wavelength"][1] == pytest.approx(0.05)

    def test_rta_arrays_returned(self):
        """R, T, A arrays are included in the returned dict."""
        R = np.array([0.2])
        T = np.array([0.3])
        A = np.array([0.5])
        result = _make_result(reflection=R, transmission=T, absorption=A, n_wl=1)
        check = EnergyBalance.check(result)
        np.testing.assert_array_equal(check["R"], R)
        np.testing.assert_array_equal(check["T"], T)
        np.testing.assert_array_equal(check["A"], A)

    def test_single_wavelength(self):
        """Works with single wavelength data."""
        R = np.array([0.25])
        T = np.array([0.25])
        A = np.array([0.50])
        result = _make_result(reflection=R, transmission=T, absorption=A, n_wl=1)
        check = EnergyBalance.check(result)
        assert check["valid"] is True
        assert check["per_wavelength"].shape == (1,)
