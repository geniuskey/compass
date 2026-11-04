"""Unit tests for CrosstalkAnalyzer (spectral crosstalk definition)."""

import numpy as np
import pytest

from compass.analysis.crosstalk import CrosstalkAnalyzer
from compass.analysis.qe_calculator import QECalculator
from compass.core.types import SimulationResult

BAYER = [["R", "G"], ["G", "B"]]
# sorted pixel order: B_1_1, G_0_1, G_1_0, R_0_0
WL = np.array([0.45, 0.55, 0.65, 0.75])


def _ideal_qe():
    """Each color collects only inside its own band; nothing at 0.75 um."""
    return {
        "B_1_1": np.array([0.8, 0.0, 0.0, 0.0]),
        "G_0_1": np.array([0.0, 0.4, 0.0, 0.0]),
        "G_1_0": np.array([0.0, 0.4, 0.0, 0.0]),
        "R_0_0": np.array([0.0, 0.0, 0.7, 0.0]),
    }


def _leaky_qe():
    """Out-of-color pixels pick up 0.1 everywhere."""
    qe = {k: v + 0.1 for k, v in _ideal_qe().items()}
    return qe


class TestComputeMatrix:
    def test_shape(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        assert ct.shape == (4, 4, 4)

    def test_rows_defined_only_inside_band(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        # B row (index 0) defined only at 0.45 um
        assert not np.isnan(ct[0, :, 0]).any()
        assert np.isnan(ct[0, :, 1:]).all()
        # R row (index 3) defined only at 0.65 um
        assert not np.isnan(ct[3, :, 2]).any()
        assert np.isnan(ct[3, :, [0, 1, 3]]).all()
        # Nobody owns 0.75 um with default bands
        assert np.isnan(ct[:, :, 3]).all()

    def test_defined_rows_sum_to_one(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        sums = np.nansum(ct, axis=1)
        defined = ~np.isnan(ct[:, 0, :])
        np.testing.assert_allclose(sums[defined], 1.0)

    def test_rows_differ_by_intended_pixel(self):
        """Regression: rows used to be identical regardless of the intended pixel."""
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        assert not np.allclose(np.nan_to_num(ct[0]), np.nan_to_num(ct[3]))

    def test_share_values(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        # at 0.45: B=0.9, others 0.1 each -> total 1.2
        np.testing.assert_allclose(ct[0, :, 0], [0.9 / 1.2, 0.1 / 1.2, 0.1 / 1.2, 0.1 / 1.2])

    def test_zero_signal_in_band_is_self_collection(self):
        qe = {k: np.zeros(4) for k in _ideal_qe()}
        ct = CrosstalkAnalyzer.compute_matrix(qe, BAYER, WL)
        np.testing.assert_allclose(ct[0, :, 0], [1.0, 0.0, 0.0, 0.0])

    def test_unknown_color_uses_full_range(self):
        qe = {"W_0_0": np.ones(4), "X_0_1": np.ones(4)}
        ct = CrosstalkAnalyzer.compute_matrix(qe, None, WL)
        assert not np.isnan(ct).any()
        np.testing.assert_allclose(ct, 0.5)

    def test_custom_bands(self):
        bands = {"B": (0.4, 0.8), "G": (0.4, 0.8), "R": (0.4, 0.8)}
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL, bands=bands)
        assert not np.isnan(ct).any()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="wavelengths"):
            CrosstalkAnalyzer.compute_matrix({"R_0_0": np.ones(3)}, None, WL)

    def test_empty(self):
        ct = CrosstalkAnalyzer.compute_matrix({}, None, WL)
        assert ct.shape == (0, 0, 4)


class TestSpectralCrosstalk:
    def test_ideal_sensor_has_zero_crosstalk(self):
        names = sorted(_ideal_qe())
        ct = CrosstalkAnalyzer.compute_matrix(_ideal_qe(), BAYER, WL)
        spec = CrosstalkAnalyzer.spectral_crosstalk(ct, WL, names)
        np.testing.assert_allclose(spec[:3], 0.0)
        assert np.isnan(spec[3])

    def test_same_color_pixels_are_not_crosstalk(self):
        names = sorted(_leaky_qe())
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        spec = CrosstalkAnalyzer.spectral_crosstalk(ct, WL, names)
        # at 0.55: G pixels 0.5 each, B and R 0.1 each -> total 1.2, off-color 0.2
        assert spec[1] == pytest.approx(0.2 / 1.2)
        # at 0.45: off-color share 0.3 / 1.2
        assert spec[0] == pytest.approx(0.3 / 1.2)

    def test_without_names_counts_all_off_diagonal(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        spec = CrosstalkAnalyzer.spectral_crosstalk(ct, WL)
        # at 0.55 each G row sees the other G pixel as off-diagonal: 1 - 0.5/1.2
        assert spec[1] == pytest.approx(1 - 0.5 / 1.2)

    def test_matches_qe_calculator_single_wavelength_bands(self):
        """Per-band row values agree with QECalculator.compute_crosstalk."""
        qe = _leaky_qe()
        ct3 = CrosstalkAnalyzer.compute_matrix(qe, BAYER, WL)
        ct2 = QECalculator.compute_crosstalk(qe, WL)
        # one sample per band, so the band-integrated row equals the 3D slice
        np.testing.assert_allclose(ct3[0, :, 0], ct2[0])
        np.testing.assert_allclose(ct3[3, :, 2], ct2[3])

    def test_name_length_mismatch_raises(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        with pytest.raises(ValueError, match="pixel_names"):
            CrosstalkAnalyzer.spectral_crosstalk(ct, WL, ["R_0_0"])


class TestPeakCrosstalk:
    def test_peak_location(self):
        names = sorted(_leaky_qe())
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        val, wl = CrosstalkAnalyzer.peak_crosstalk(ct, WL, names)
        # 0.45: 0.25, 0.55: 0.1667, 0.65: 0.3/1.1
        assert wl == pytest.approx(0.65)
        assert val == pytest.approx(0.3 / 1.1)

    def test_all_undefined_returns_nan(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, np.array([0.8, 0.9, 1.0, 1.1]))
        val, wl = CrosstalkAnalyzer.peak_crosstalk(ct, WL)
        assert np.isnan(val) and np.isnan(wl)

    def test_empty_spectrum(self):
        assert CrosstalkAnalyzer.peak_crosstalk(np.zeros((2, 2, 0)), np.array([])) == (0.0, 0.0)


class TestNeighborCrosstalk:
    def test_returns_every_pixel(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        nb = CrosstalkAnalyzer.neighbor_crosstalk(ct, BAYER)
        assert set(nb) == set(_leaky_qe())

    def test_values(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        nb = CrosstalkAnalyzer.neighbor_crosstalk(ct, BAYER)
        # R_0_0 neighbours G_0_1, G_1_0; at 0.65 each holds 0.1 / 1.1
        assert nb["R_0_0"] == pytest.approx(0.1 / 1.1)
        # G_0_1 neighbours R_0_0 and B_1_1; at 0.55 each holds 0.1 / 1.2
        assert nb["G_0_1"] == pytest.approx(0.1 / 1.2)

    def test_single_pixel_has_no_neighbours(self):
        ct = CrosstalkAnalyzer.compute_matrix({"G_0_0": np.ones(4)}, None, WL)
        assert CrosstalkAnalyzer.neighbor_crosstalk(ct, [["G"]]) == {"G_0_0": 0.0}

    def test_never_intended_pixel_is_nan(self):
        wl = np.array([0.45])
        qe = {k: v[:1] for k, v in _leaky_qe().items()}
        ct = CrosstalkAnalyzer.compute_matrix(qe, BAYER, wl)
        nb = CrosstalkAnalyzer.neighbor_crosstalk(ct, BAYER)
        assert np.isnan(nb["R_0_0"])
        assert not np.isnan(nb["B_1_1"])

    def test_mismatched_bayer_raises(self):
        ct = CrosstalkAnalyzer.compute_matrix(_leaky_qe(), BAYER, WL)
        with pytest.raises(ValueError, match="bayer_map"):
            CrosstalkAnalyzer.neighbor_crosstalk(ct, [["R"]])


class TestSummarize:
    def test_summary_keys_and_values(self):
        result = SimulationResult(qe_per_pixel=_leaky_qe(), wavelengths=WL)
        s = CrosstalkAnalyzer.summarize(result)
        assert s["n_pixels"] == 4
        assert s["n_wavelengths"] == 4
        assert s["peak_wavelength_um"] == pytest.approx(0.65)
        expected_mean = np.mean([0.3 / 1.2, 0.2 / 1.2, 0.3 / 1.1])
        assert s["mean_crosstalk"] == pytest.approx(expected_mean)
        assert len(s["spectral_crosstalk"]) == 4

    def test_ideal_sensor_summary_is_zero(self):
        s = CrosstalkAnalyzer.summarize(SimulationResult(qe_per_pixel=_ideal_qe(), wavelengths=WL))
        assert s["peak_crosstalk"] == pytest.approx(0.0)
        assert s["mean_crosstalk"] == pytest.approx(0.0)

    def test_empty_result(self):
        s = CrosstalkAnalyzer.summarize(SimulationResult(qe_per_pixel={}, wavelengths=WL))
        assert s["n_pixels"] == 0
        assert s["peak_crosstalk"] == 0.0
        assert s["n_wavelengths"] == 4
