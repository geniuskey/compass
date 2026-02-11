"""Unit tests for ConeIllumination."""

import numpy as np
import pytest

from compass.sources.cone_illumination import ConeIllumination


class TestConeIlluminationConstructor:
    """Tests for ConeIllumination initialization."""

    def test_default_parameters(self):
        """Default constructor values are correct."""
        ci = ConeIllumination()
        assert ci.cra_deg == 0.0
        assert ci.f_number == 2.0
        assert ci.n_points == 37
        assert ci.sampling == "fibonacci"
        assert ci.weighting == "cosine"

    def test_half_cone_angle(self):
        """Half-cone angle computed from f-number."""
        ci = ConeIllumination(f_number=2.0)
        expected = np.arcsin(1.0 / (2.0 * 2.0))
        assert ci.half_cone_rad == pytest.approx(expected)

    @pytest.mark.parametrize("f_number", [1.4, 2.0, 2.8, 4.0, 5.6])
    def test_half_cone_decreases_with_f_number(self, f_number):
        """Higher f-number gives smaller cone angle."""
        ci = ConeIllumination(f_number=f_number)
        expected = np.arcsin(1.0 / (2.0 * f_number))
        assert ci.half_cone_rad == pytest.approx(expected)

    def test_custom_parameters(self):
        """Custom constructor parameters are stored."""
        ci = ConeIllumination(
            cra_deg=10.0, f_number=2.8, n_points=100,
            sampling="grid", weighting="uniform",
        )
        assert ci.cra_deg == 10.0
        assert ci.f_number == 2.8
        assert ci.n_points == 100
        assert ci.sampling == "grid"
        assert ci.weighting == "uniform"


class TestFibonacciSampling:
    """Tests for fibonacci sampling method."""

    def test_correct_number_of_points(self):
        """Fibonacci sampling returns n_points samples."""
        ci = ConeIllumination(n_points=37, sampling="fibonacci")
        points = ci.get_sampling_points()
        assert len(points) == 37

    def test_weights_sum_to_one(self):
        """Normalized weights sum to 1."""
        ci = ConeIllumination(n_points=50, sampling="fibonacci")
        points = ci.get_sampling_points()
        total = sum(w for _, _, w in points)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_all_weights_positive(self):
        """All weights are non-negative."""
        ci = ConeIllumination(n_points=37, sampling="fibonacci")
        points = ci.get_sampling_points()
        for _, _, w in points:
            assert w >= 0.0

    def test_theta_within_range(self):
        """All theta values are within valid range [0, 90] degrees."""
        ci = ConeIllumination(cra_deg=0.0, f_number=2.0, n_points=37, sampling="fibonacci")
        points = ci.get_sampling_points()
        for theta, _, _ in points:
            assert 0.0 <= theta <= 90.0

    def test_tuple_structure(self):
        """Each point is a 3-tuple (theta_deg, phi_deg, weight)."""
        ci = ConeIllumination(n_points=5, sampling="fibonacci")
        points = ci.get_sampling_points()
        for p in points:
            assert len(p) == 3
            theta, phi, weight = p
            assert isinstance(float(theta), float)
            assert isinstance(float(phi), float)
            assert isinstance(float(weight), float)


class TestGridSampling:
    """Tests for grid sampling method."""

    def test_weights_sum_to_one(self):
        """Grid sampling weights sum to 1."""
        ci = ConeIllumination(n_points=50, sampling="grid")
        points = ci.get_sampling_points()
        total = sum(w for _, _, w in points)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_returns_points(self):
        """Grid sampling returns non-empty list."""
        ci = ConeIllumination(n_points=20, sampling="grid")
        points = ci.get_sampling_points()
        assert len(points) > 0

    def test_all_weights_positive(self):
        """All grid weights are positive."""
        ci = ConeIllumination(n_points=30, sampling="grid")
        points = ci.get_sampling_points()
        for _, _, w in points:
            assert w >= 0.0


class TestGaussianQuadratureSampling:
    """Tests for Gauss-Legendre quadrature sampling."""

    def test_weights_sum_to_one(self):
        """Gauss quadrature weights sum to 1."""
        ci = ConeIllumination(n_points=50, sampling="gauss")
        points = ci.get_sampling_points()
        total = sum(w for _, _, w in points)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_returns_points(self):
        """Gauss sampling returns non-empty list."""
        ci = ConeIllumination(n_points=20, sampling="gauss")
        points = ci.get_sampling_points()
        assert len(points) > 0

    def test_all_weights_positive(self):
        """All Gauss quadrature weights are positive."""
        ci = ConeIllumination(n_points=30, sampling="gauss")
        points = ci.get_sampling_points()
        for _, _, w in points:
            assert w >= 0.0


class TestWeightingSchemes:
    """Tests for different weighting schemes."""

    @pytest.mark.parametrize("weighting", ["uniform", "cosine", "cos4", "gaussian", "custom"])
    def test_weights_sum_to_one(self, weighting):
        """All weighting schemes produce normalized weights."""
        ci = ConeIllumination(n_points=37, sampling="fibonacci", weighting=weighting)
        points = ci.get_sampling_points()
        total = sum(w for _, _, w in points)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_cosine_weight_decreases_with_angle(self):
        """Cosine weighting gives lower weight at larger angles."""
        ci = ConeIllumination(weighting="cosine")
        # Weight at theta=0 should be >= weight at theta>0
        w0 = ci._compute_weight(0.0)
        w1 = ci._compute_weight(0.2)
        assert w0 >= w1

    def test_uniform_weight_constant(self):
        """Uniform weighting returns constant 1.0 before normalization."""
        ci = ConeIllumination(weighting="uniform")
        assert ci._compute_weight(0.0) == 1.0
        assert ci._compute_weight(0.5) == 1.0

    def test_cos4_stronger_than_cosine(self):
        """cos4 weighting decays faster than cosine."""
        ci_cos = ConeIllumination(weighting="cosine")
        ci_cos4 = ConeIllumination(weighting="cos4")
        theta = 0.3
        w_cos = ci_cos._compute_weight(theta)
        w_cos4 = ci_cos4._compute_weight(theta)
        # cos^4 < cos for theta > 0
        assert w_cos4 < w_cos

    def test_gaussian_weight_peaks_at_center(self):
        """Gaussian weighting peaks at theta=0."""
        ci = ConeIllumination(weighting="gaussian")
        w_center = ci._compute_weight(0.0)
        w_off = ci._compute_weight(0.1)
        assert w_center >= w_off


class TestUnknownSampling:
    """Tests for fallback behavior with unknown sampling."""

    def test_unknown_sampling_falls_back_to_fibonacci(self):
        """Unknown sampling string defaults to fibonacci."""
        ci = ConeIllumination(n_points=10, sampling="unknown_method")
        points = ci.get_sampling_points()
        # Should produce same result as fibonacci
        ci_fib = ConeIllumination(n_points=10, sampling="fibonacci")
        points_fib = ci_fib.get_sampling_points()
        assert len(points) == len(points_fib)


class TestCRAEffect:
    """Tests for chief ray angle effects."""

    def test_nonzero_cra(self):
        """Non-zero CRA still produces valid sampling points."""
        ci = ConeIllumination(cra_deg=15.0, f_number=2.8, n_points=20)
        points = ci.get_sampling_points()
        assert len(points) == 20
        total = sum(w for _, _, w in points)
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_zero_cra(self):
        """Zero CRA produces symmetric-looking points."""
        ci = ConeIllumination(cra_deg=0.0, n_points=20)
        points = ci.get_sampling_points()
        assert len(points) == 20
        for theta, _, _ in points:
            assert theta >= 0.0
