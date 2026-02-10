"""Benchmark tests for energy conservation checks.

Verifies that the EnergyBalance checker correctly identifies valid and invalid
simulation results based on R + T + A = 1 conservation law.
"""

import numpy as np
import pytest

from compass.analysis.energy_balance import EnergyBalance
from compass.core.types import SimulationResult


def make_result(R, T, A=None, n_wavelengths=11):
    """Create a mock SimulationResult with given R, T, A spectra.

    Args:
        R: Scalar or array for reflection.
        T: Scalar or array for transmission.
        A: Scalar or array for absorption (None = computed as 1-R-T).
        n_wavelengths: Number of wavelengths if R/T/A are scalars.
    """
    wavelengths = np.linspace(0.4, 0.7, n_wavelengths)

    R_arr = np.full(n_wavelengths, R) if np.isscalar(R) else np.asarray(R)
    T_arr = np.full(n_wavelengths, T) if np.isscalar(T) else np.asarray(T)

    if A is not None:
        A_arr = np.full(n_wavelengths, A) if np.isscalar(A) else np.asarray(A)
    else:
        A_arr = None

    return SimulationResult(
        qe_per_pixel={"R_0_0": np.zeros(n_wavelengths)},
        wavelengths=wavelengths,
        reflection=R_arr,
        transmission=T_arr,
        absorption=A_arr,
    )


class TestEnergyBalanceValid:
    """Tests where R + T + A = 1 should hold."""

    def test_perfect_conservation(self):
        """R=0.3, T=0.5, A=0.2 -> sum=1.0 exactly."""
        result = make_result(R=0.3, T=0.5, A=0.2)
        check = EnergyBalance.check(result)
        assert check["valid"] is True
        assert check["max_error"] == pytest.approx(0.0, abs=1e-12)

    def test_conservation_no_absorption_explicit(self):
        """R=0.04, T=0.96, A=0.0 -> sum=1.0."""
        result = make_result(R=0.04, T=0.96, A=0.0)
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_conservation_absorption_inferred(self):
        """When A is None, EnergyBalance should infer A=1-R-T."""
        result = make_result(R=0.3, T=0.5, A=None)
        check = EnergyBalance.check(result)
        assert check["valid"] is True
        assert check["max_error"] == pytest.approx(0.0, abs=1e-12)

    def test_per_wavelength_varying(self):
        """Different R, T, A at each wavelength, all summing to 1."""
        n = 5
        R = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        T = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        A = 1.0 - R - T
        result = make_result(R=R, T=T, A=A, n_wavelengths=n)
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_within_tolerance(self):
        """Small numerical errors within tolerance should still be valid."""
        result = make_result(R=0.3, T=0.5, A=0.2 + 5e-3)
        check = EnergyBalance.check(result, tolerance=0.01)
        assert check["valid"] is True

    def test_full_reflection(self):
        """R=1, T=0, A=0 (total reflection)."""
        result = make_result(R=1.0, T=0.0, A=0.0)
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_full_absorption(self):
        """R=0, T=0, A=1 (total absorption)."""
        result = make_result(R=0.0, T=0.0, A=1.0)
        check = EnergyBalance.check(result)
        assert check["valid"] is True


class TestEnergyBalanceInvalid:
    """Tests where R + T + A != 1 should be flagged."""

    def test_exceeds_unity(self):
        """R+T+A > 1.01 should fail."""
        result = make_result(R=0.5, T=0.5, A=0.1)
        check = EnergyBalance.check(result, tolerance=0.01)
        assert check["valid"] is False
        assert check["max_error"] > 0.01

    def test_significantly_below_unity(self):
        """R+T+A << 1 should fail (energy lost)."""
        result = make_result(R=0.1, T=0.1, A=0.1)
        check = EnergyBalance.check(result, tolerance=0.01)
        assert check["valid"] is False
        assert check["max_error"] > 0.5

    def test_single_wavelength_violation(self):
        """Even one wavelength violating should fail."""
        n = 5
        R = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        T = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        A = np.array([0.2, 0.2, 0.2, 0.2, 0.4])  # last one sums to 1.2
        result = make_result(R=R, T=T, A=A, n_wavelengths=n)
        check = EnergyBalance.check(result, tolerance=0.01)
        assert check["valid"] is False

    def test_negative_values_exceed(self):
        """Negative reflection is unphysical and may cause total > 1."""
        result = make_result(R=-0.1, T=0.8, A=0.5)
        check = EnergyBalance.check(result, tolerance=0.01)
        # R+T+A = 1.2, should fail
        assert check["valid"] is False


class TestEnergyBalanceEdgeCases:
    """Edge cases for the energy balance checker."""

    def test_no_reflection_data(self):
        """Result with no reflection data should return valid (insufficient data)."""
        result = SimulationResult(
            qe_per_pixel={"R_0_0": np.zeros(5)},
            wavelengths=np.linspace(0.4, 0.7, 5),
            reflection=None,
            transmission=np.ones(5) * 0.5,
        )
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_no_transmission_data(self):
        """Result with no transmission data should return valid (insufficient data)."""
        result = SimulationResult(
            qe_per_pixel={"R_0_0": np.zeros(5)},
            wavelengths=np.linspace(0.4, 0.7, 5),
            reflection=np.ones(5) * 0.3,
            transmission=None,
        )
        check = EnergyBalance.check(result)
        assert check["valid"] is True

    def test_custom_tolerance(self):
        """Tighter tolerance should catch smaller errors."""
        result = make_result(R=0.3, T=0.5, A=0.2 + 0.005)
        check_loose = EnergyBalance.check(result, tolerance=0.01)
        check_tight = EnergyBalance.check(result, tolerance=0.001)
        assert check_loose["valid"] is True
        assert check_tight["valid"] is False
