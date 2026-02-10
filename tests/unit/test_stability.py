"""Unit tests for RCWA stability module."""

import numpy as np
import pytest

from compass.solvers.rcwa.stability import (
    EigenvalueStabilizer,
    LiFactorization,
    PrecisionManager,
    StableSMatrixAlgorithm,
)


class TestPrecisionManager:
    """Tests for precision management."""

    def test_mixed_precision_eigen(self):
        """Should compute eigendecomp in float64."""
        # Create a well-conditioned matrix
        A = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        A = A.astype(np.complex64)

        eigenvalues, eigenvectors = PrecisionManager.mixed_precision_eigen(A)
        assert eigenvalues.dtype == np.complex64
        assert eigenvectors.dtype == np.complex64
        assert len(eigenvalues) == 10

    def test_eigen_correctness(self):
        """Eigendecomp should satisfy A @ v = lambda * v."""
        A = np.random.randn(5, 5) + 1j * np.random.randn(5, 5)
        A = A.astype(np.complex128)

        vals, vecs = PrecisionManager.mixed_precision_eigen(A)

        # Check A @ v ≈ lambda * v for each eigenpair
        for i in range(5):
            lhs = A @ vecs[:, i]
            rhs = vals[i] * vecs[:, i]
            np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestStableSMatrixAlgorithm:
    """Tests for S-matrix Redheffer star product."""

    def test_identity_smatrix(self):
        """Identity S-matrix should have unit transmission."""
        S = StableSMatrixAlgorithm.identity_smatrix(4)
        np.testing.assert_array_almost_equal(S["S11"], np.zeros((4, 4)))
        np.testing.assert_array_almost_equal(S["S12"], np.eye(4))
        np.testing.assert_array_almost_equal(S["S21"], np.eye(4))
        np.testing.assert_array_almost_equal(S["S22"], np.zeros((4, 4)))

    def test_star_product_identity(self):
        """Star product with identity should return original."""
        n = 4
        I_S = StableSMatrixAlgorithm.identity_smatrix(n)

        # Create a random S-matrix
        S = {
            "S11": np.random.randn(n, n) * 0.1 + 1j * np.random.randn(n, n) * 0.1,
            "S12": np.eye(n) + np.random.randn(n, n) * 0.1,
            "S21": np.eye(n) + np.random.randn(n, n) * 0.1,
            "S22": np.random.randn(n, n) * 0.1 + 1j * np.random.randn(n, n) * 0.1,
        }

        result = StableSMatrixAlgorithm.redheffer_star_product(I_S, S)

        np.testing.assert_allclose(result["S11"], S["S11"], atol=1e-10)
        np.testing.assert_allclose(result["S12"], S["S12"], atol=1e-10)

    def test_star_product_two_identities(self):
        """Star product of two identities should give identity."""
        n = 3
        I1 = StableSMatrixAlgorithm.identity_smatrix(n)
        I2 = StableSMatrixAlgorithm.identity_smatrix(n)

        result = StableSMatrixAlgorithm.redheffer_star_product(I1, I2)
        np.testing.assert_allclose(result["S12"], np.eye(n), atol=1e-10)
        np.testing.assert_allclose(result["S21"], np.eye(n), atol=1e-10)


class TestLiFactorization:
    """Tests for Li's Fourier factorization."""

    def test_toeplitz_uniform(self):
        """Uniform permittivity should give diagonal Toeplitz."""
        eps = np.full(32, 4.0 + 0j)
        T = LiFactorization.toeplitz_from_fft(eps, n_harmonics=2)
        # DC component should be eps, off-diagonal near zero
        assert T.shape == (5, 5)
        assert T[2, 2] == pytest.approx(4.0, abs=1e-10)

    def test_inverse_rule_uniform(self):
        """Inverse rule on uniform medium should match naive."""
        eps = np.full(64, 2.25 + 0j)  # SiO2-like
        T_naive = LiFactorization.convolution_matrix_naive(eps, 3)
        T_inv = LiFactorization.convolution_matrix_inverse_rule(eps, 3)

        # For uniform medium, both should be identical
        np.testing.assert_allclose(T_naive, T_inv, atol=1e-8)

    def test_2d_convolution_shape(self):
        """2D convolution matrix should have correct shape."""
        eps_2d = np.full((32, 32), 4.0 + 0j)
        T = LiFactorization.convolution_matrix_2d(eps_2d, 2, 2, use_inverse_rule=False)
        # (2*2+1)*(2*2+1) = 25
        assert T.shape == (25, 25)


class TestEigenvalueStabilizer:
    """Tests for eigenvalue stabilization."""

    def test_propagation_direction(self):
        """Should select correct sqrt branch."""
        # Positive real eigenvalues
        eigenvalues = np.array([1.0, 4.0, 9.0, -1.0 + 0.1j])
        sqrt_ev = EigenvalueStabilizer.select_propagation_direction(eigenvalues)

        # All should have non-negative real part
        assert np.all(np.real(sqrt_ev[:3]) >= 0)

    def test_fix_degenerate(self):
        """Should handle degenerate eigenvalues."""
        eigenvalues = np.array([1.0 + 0j, 1.0 + 1e-15j, 2.0 + 0j])
        eigenvectors = np.eye(3, dtype=complex)

        vals, vecs = EigenvalueStabilizer.fix_degenerate_eigenvalues(
            eigenvalues, eigenvectors, broadening=1e-10,
        )
        assert len(vals) == 3
        assert vecs.shape == (3, 3)

    def test_energy_conservation_valid(self):
        """Should pass for valid R, T."""
        R = np.array([0.3, 0.25, 0.2])
        T = np.array([0.7, 0.75, 0.8])
        result = EigenvalueStabilizer.validate_energy_conservation(R, T)
        assert result["valid"]

    def test_energy_conservation_violation(self):
        """Should detect over-unity R + T."""
        R = np.array([0.6])
        T = np.array([0.6])
        result = EigenvalueStabilizer.validate_energy_conservation(R, T, tolerance=0.05)
        assert not result["valid"]
        assert len(result["violation_indices"]) > 0
