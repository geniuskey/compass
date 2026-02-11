"""Unit tests for QECalculator."""

import numpy as np
import pytest

from compass.analysis.qe_calculator import QECalculator


class TestFromAbsorption:
    """Tests for QECalculator.from_absorption."""

    def test_basic_qe(self):
        """QE = absorbed / incident for each pixel."""
        absorption = {
            "R_0_0": np.array([0.3, 0.5, 0.2]),
            "G_0_1": np.array([0.1, 0.4, 0.6]),
        }
        incident = np.array([1.0, 1.0, 1.0])
        qe = QECalculator.from_absorption(absorption, incident)
        assert "R_0_0" in qe
        assert "G_0_1" in qe
        np.testing.assert_allclose(qe["R_0_0"], [0.3, 0.5, 0.2])
        np.testing.assert_allclose(qe["G_0_1"], [0.1, 0.4, 0.6])

    def test_qe_clipped_to_0_1(self):
        """QE values are clipped to [0, 1]."""
        absorption = {"pixel": np.array([1.5, -0.2, 0.5])}
        incident = np.array([1.0, 1.0, 1.0])
        qe = QECalculator.from_absorption(absorption, incident)
        assert np.all(qe["pixel"] >= 0)
        assert np.all(qe["pixel"] <= 1)

    def test_zero_incident_power(self):
        """Zero incident power does not cause division by zero."""
        absorption = {"pixel": np.array([0.5, 0.0])}
        incident = np.array([0.0, 0.0])
        qe = QECalculator.from_absorption(absorption, incident)
        assert np.all(np.isfinite(qe["pixel"]))

    def test_single_wavelength(self):
        """Works correctly with a single wavelength."""
        absorption = {"pixel": np.array([0.45])}
        incident = np.array([1.0])
        qe = QECalculator.from_absorption(absorption, incident)
        assert qe["pixel"][0] == pytest.approx(0.45)

    def test_fractional_incident(self):
        """Correct QE when incident power is not 1.0."""
        absorption = {"pixel": np.array([0.25, 0.50])}
        incident = np.array([0.5, 1.0])
        qe = QECalculator.from_absorption(absorption, incident)
        np.testing.assert_allclose(qe["pixel"], [0.5, 0.5])

    def test_empty_absorption_dict(self):
        """Empty pixel dict returns empty QE dict."""
        qe = QECalculator.from_absorption({}, np.array([1.0]))
        assert qe == {}

    def test_multiple_pixels_independent(self):
        """Each pixel is computed independently."""
        absorption = {
            "A": np.array([0.1]),
            "B": np.array([0.9]),
        }
        incident = np.array([1.0])
        qe = QECalculator.from_absorption(absorption, incident)
        assert qe["A"][0] == pytest.approx(0.1)
        assert qe["B"][0] == pytest.approx(0.9)


class TestFromPoyntingFlux:
    """Tests for QECalculator.from_poynting_flux."""

    def test_basic_flux_difference(self):
        """QE = (flux_top - flux_bottom) / incident."""
        flux_top = np.array([0.8, 0.6])
        flux_bottom = np.array([0.3, 0.1])
        incident = np.array([1.0, 1.0])
        qe = QECalculator.from_poynting_flux(flux_top, flux_bottom, incident)
        np.testing.assert_allclose(qe, [0.5, 0.5])

    def test_clipped_to_0_1(self):
        """Output is clipped to valid QE range."""
        flux_top = np.array([1.5, 0.1])
        flux_bottom = np.array([0.0, 0.5])
        incident = np.array([1.0, 1.0])
        qe = QECalculator.from_poynting_flux(flux_top, flux_bottom, incident)
        assert np.all(qe >= 0)
        assert np.all(qe <= 1)

    def test_zero_incident(self):
        """Zero incident power does not blow up."""
        flux_top = np.array([0.5])
        flux_bottom = np.array([0.2])
        incident = np.array([0.0])
        qe = QECalculator.from_poynting_flux(flux_top, flux_bottom, incident)
        assert np.all(np.isfinite(qe))

    def test_returns_ndarray(self):
        """Result is always a numpy ndarray."""
        qe = QECalculator.from_poynting_flux(
            np.array([0.5]), np.array([0.1]), np.array([1.0])
        )
        assert isinstance(qe, np.ndarray)


class TestComputeCrosstalk:
    """Tests for QECalculator.compute_crosstalk."""

    def test_shape(self):
        """Crosstalk matrix has shape (n_pixels, n_pixels, n_wavelengths)."""
        qe = {
            "A": np.array([0.4, 0.3]),
            "B": np.array([0.3, 0.4]),
        }
        ct = QECalculator.compute_crosstalk(qe, bayer_map=[])
        assert ct.shape == (2, 2, 2)

    def test_rows_sum_to_one(self):
        """Each row of the crosstalk matrix sums to 1 (light is redistributed)."""
        qe = {
            "A": np.array([0.4, 0.3, 0.5]),
            "B": np.array([0.3, 0.4, 0.2]),
            "C": np.array([0.1, 0.1, 0.1]),
        }
        ct = QECalculator.compute_crosstalk(qe, bayer_map=[])
        for i in range(ct.shape[0]):
            for wl in range(ct.shape[2]):
                assert ct[i, :, wl].sum() == pytest.approx(1.0, abs=1e-10)

    def test_single_pixel(self):
        """Single pixel means all 'light' stays in that pixel."""
        qe = {"only": np.array([0.5, 0.3])}
        ct = QECalculator.compute_crosstalk(qe, bayer_map=[])
        assert ct.shape == (1, 1, 2)
        np.testing.assert_allclose(ct[0, 0, :], [1.0, 1.0])

    def test_zero_qe_handled(self):
        """Zero total QE does not cause division by zero."""
        qe = {
            "A": np.array([0.0]),
            "B": np.array([0.0]),
        }
        ct = QECalculator.compute_crosstalk(qe, bayer_map=[])
        assert np.all(np.isfinite(ct))

    def test_dominant_pixel(self):
        """Pixel with much higher QE gets most of the crosstalk fraction."""
        qe = {
            "A": np.array([0.9]),
            "B": np.array([0.01]),
        }
        ct = QECalculator.compute_crosstalk(qe, bayer_map=[])
        # Column 0 (pixel A) should dominate
        assert ct[0, 0, 0] > 0.9


class TestSpectralResponse:
    """Tests for QECalculator.spectral_response."""

    def test_groups_by_color(self):
        """Pixels are grouped by color prefix."""
        qe = {
            "R_0_0": np.array([0.4, 0.3]),
            "R_1_0": np.array([0.6, 0.5]),
            "G_0_1": np.array([0.3, 0.7]),
        }
        wavelengths = np.array([0.45, 0.65])
        sr = QECalculator.spectral_response(qe, wavelengths)
        assert "G" in sr
        assert "R" in sr
        assert len(sr) == 2

    def test_averaged_over_same_color(self):
        """QE is averaged across same-color pixels."""
        qe = {
            "R_0_0": np.array([0.4]),
            "R_1_0": np.array([0.6]),
        }
        wavelengths = np.array([0.55])
        sr = QECalculator.spectral_response(qe, wavelengths)
        wl, mean_qe = sr["R"]
        assert mean_qe[0] == pytest.approx(0.5)
        np.testing.assert_array_equal(wl, wavelengths)

    def test_single_pixel_per_color(self):
        """Single pixel per color returns that pixel's QE directly."""
        qe = {"B_0_0": np.array([0.7, 0.8])}
        wavelengths = np.array([0.40, 0.45])
        sr = QECalculator.spectral_response(qe, wavelengths)
        _, mean_qe = sr["B"]
        np.testing.assert_allclose(mean_qe, [0.7, 0.8])

    def test_keys_sorted(self):
        """Output keys are alphabetically sorted."""
        qe = {
            "G_0_0": np.array([0.5]),
            "B_0_0": np.array([0.5]),
            "R_0_0": np.array([0.5]),
        }
        sr = QECalculator.spectral_response(qe, np.array([0.55]))
        assert list(sr.keys()) == ["B", "G", "R"]
