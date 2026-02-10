"""Unit tests for the radiometric signal chain modules.

Tests cover:
- Illuminant models (blackbody, CIE A, D65, F11, LED, monochromatic, flat)
- Scene reflectance models (flat, Macbeth, colour targets)
- Module optics (IR cut filter, lens, combined)
- Signal calculator (integration, electrons, SNR, white balance, ratios)
"""

import numpy as np
import pytest

from compass.analysis.signal_calculator import SignalCalculator
from compass.optics.module_optics import IRCutFilter, LensTransmittance, ModuleOptics
from compass.sources.illuminant import Illuminant
from compass.sources.scene import SceneReflectance

# Common wavelength grids used across tests
WL_VIS = np.linspace(0.38, 0.78, 401)  # Visible range, 1 nm steps
WL_COARSE = np.linspace(0.40, 0.70, 31)  # Coarse grid for fast tests


class TestIlluminant:
    """Tests for Illuminant spectral models."""

    def test_blackbody_peak_wiens_law(self):
        """Blackbody peak wavelength should follow Wien's law: lam_max = 2898/T um."""
        for temp in [2856.0, 3500.0, 5000.0, 5500.0, 6500.0]:
            wl = np.linspace(0.2, 5.0, 5000)
            ill = Illuminant.blackbody(temp, wl)
            peak_idx = np.argmax(ill.spectrum)
            peak_wl = wl[peak_idx]
            expected_peak = 2898.0 / temp  # Wien's displacement law in um
            assert peak_wl == pytest.approx(expected_peak, abs=0.02), (
                f"Blackbody at {temp}K: peak={peak_wl:.4f} um, expected={expected_peak:.4f} um"
            )

    def test_blackbody_normalized(self):
        """Blackbody spectrum should be normalised with peak = 1."""
        ill = Illuminant.blackbody(5500.0, WL_VIS)
        assert np.max(ill.spectrum) == pytest.approx(1.0, abs=1e-10)

    def test_cie_a_matches_blackbody_2856(self):
        """CIE Illuminant A should closely match a blackbody at 2856K."""
        wl = np.linspace(0.38, 0.78, 201)
        ill_a = Illuminant.cie_a(wl)
        ill_bb = Illuminant.blackbody(2856.0, wl)
        # Both should be normalised, so shapes should be very close
        # Normalise both to peak = 1 for comparison
        a_norm = ill_a.spectrum / np.max(ill_a.spectrum)
        bb_norm = ill_bb.spectrum / np.max(ill_bb.spectrum)
        max_diff = np.max(np.abs(a_norm - bb_norm))
        assert max_diff < 0.01, f"CIE A vs blackbody 2856K max diff = {max_diff}"

    def test_cie_d65_is_named(self):
        """CIE D65 illuminant should have the correct name."""
        ill = Illuminant.cie_d65(WL_VIS)
        assert ill.name == "CIE_D65"

    def test_cie_f11_has_peaks(self):
        """CIE F11 should have distinct emission peaks near 430, 545, 610 nm."""
        wl = np.linspace(0.38, 0.78, 401)
        ill = Illuminant.cie_f11(wl)
        wl_nm = wl * 1000.0
        # Find local maxima
        spectrum = ill.spectrum
        # Check there are significant peaks at the expected locations
        idx_430 = np.argmin(np.abs(wl_nm - 430.0))
        idx_545 = np.argmin(np.abs(wl_nm - 545.0))
        idx_610 = np.argmin(np.abs(wl_nm - 610.0))
        # Each peak region should be higher than the surrounding valleys
        assert spectrum[idx_430] > 0.3
        assert spectrum[idx_545] > 0.5
        assert spectrum[idx_610] > 0.3

    def test_led_white_has_blue_peak(self):
        """White LED should have a blue peak near 450 nm."""
        ill = Illuminant.led_white(5000.0, WL_VIS)
        wl_nm = WL_VIS * 1000.0
        idx_450 = np.argmin(np.abs(wl_nm - 450.0))
        # Blue peak region should be significant
        assert ill.spectrum[idx_450] > 0.3

    def test_monochromatic_peak(self):
        """Monochromatic source should peak at the specified centre wavelength."""
        center = 0.532
        ill = Illuminant.monochromatic(center, WL_VIS, fwhm=0.01)
        peak_idx = np.argmax(ill.spectrum)
        peak_wl = WL_VIS[peak_idx]
        assert peak_wl == pytest.approx(center, abs=0.005)

    def test_flat_is_constant(self):
        """Flat illuminant should have equal energy at all wavelengths."""
        ill = Illuminant.flat(WL_VIS)
        assert np.all(ill.spectrum == 1.0)
        assert ill.name == "E"

    def test_normalized_method(self):
        """normalized() should produce peak = 1."""
        ill = Illuminant("test", WL_VIS, WL_VIS * 100)
        norm = ill.normalized()
        assert np.max(norm.spectrum) == pytest.approx(1.0, abs=1e-10)

    def test_interpolate(self):
        """Interpolation should work on a different grid."""
        ill = Illuminant.flat(WL_VIS)
        new_wl = np.linspace(0.40, 0.70, 31)
        result = ill.interpolate(new_wl)
        assert len(result) == 31
        assert np.allclose(result, 1.0)

    def test_invalid_negative_spectrum(self):
        """Negative spectrum values should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Illuminant("bad", WL_VIS, -np.ones_like(WL_VIS))

    def test_mismatched_arrays(self):
        """Mismatched array lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            Illuminant("bad", WL_VIS, np.ones(10))


class TestSceneReflectance:
    """Tests for SceneReflectance models."""

    def test_flat_returns_constant(self):
        """Flat reflectance should return the specified constant."""
        sr = SceneReflectance.flat(0.18, WL_VIS)
        assert np.allclose(sr.reflectance, 0.18)

    def test_flat_grey18(self):
        """18% grey card should have reflectance 0.18 everywhere."""
        sr = SceneReflectance.flat(0.18, WL_COARSE)
        interp = sr.interpolate(WL_VIS)
        assert np.allclose(interp, 0.18, atol=0.01)

    def test_macbeth_white_patch(self):
        """Macbeth white patch (18) should have high reflectance."""
        sr = SceneReflectance.macbeth_patch(18, WL_VIS)
        assert np.mean(sr.reflectance) > 0.7

    def test_macbeth_black_patch(self):
        """Macbeth black patch (23) should have low reflectance."""
        sr = SceneReflectance.macbeth_patch(23, WL_VIS)
        assert np.mean(sr.reflectance) < 0.1

    def test_macbeth_range(self):
        """All Macbeth patches should have reflectance in [0, 1]."""
        for pid in range(24):
            sr = SceneReflectance.macbeth_patch(pid, WL_VIS)
            assert np.all(sr.reflectance >= 0.0)
            assert np.all(sr.reflectance <= 1.0)

    def test_macbeth_invalid_id(self):
        """Invalid patch ID should raise ValueError."""
        with pytest.raises(ValueError, match="0-23"):
            SceneReflectance.macbeth_patch(25, WL_VIS)

    def test_color_target_red(self):
        """Red target should have higher reflectance at red wavelengths."""
        sr = SceneReflectance.color_target("red", WL_VIS)
        wl_nm = WL_VIS * 1000.0
        red_region = sr.reflectance[wl_nm > 600]
        blue_region = sr.reflectance[wl_nm < 480]
        assert np.mean(red_region) > np.mean(blue_region)

    def test_color_target_invalid(self):
        """Unknown colour target should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            SceneReflectance.color_target("ultraviolet", WL_VIS)

    def test_from_array(self):
        """from_array should create a valid SceneReflectance."""
        wl = np.array([0.4, 0.5, 0.6, 0.7])
        refl = np.array([0.1, 0.5, 0.3, 0.2])
        sr = SceneReflectance.from_array("custom", wl, refl)
        assert sr.name == "custom"
        assert len(sr.wavelengths) == 4

    def test_reflectance_out_of_range(self):
        """Reflectance values outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            SceneReflectance("bad", WL_VIS, np.full_like(WL_VIS, 1.5))


class TestIRCutFilter:
    """Tests for IRCutFilter model."""

    def test_passband_transmittance(self):
        """Passband (visible) should have high transmittance."""
        filt = IRCutFilter(cutoff_wl=0.65, passband_t=0.95)
        wl = np.array([0.45, 0.50, 0.55, 0.60])
        t = filt.transmittance(wl)
        assert np.all(t > 0.90)

    def test_stopband_transmittance(self):
        """Stopband (IR) should have low transmittance."""
        filt = IRCutFilter(cutoff_wl=0.65, stopband_t=0.01)
        wl = np.array([0.75, 0.80, 0.90, 1.00])
        t = filt.transmittance(wl)
        assert np.all(t < 0.10)

    def test_cutoff_wavelength(self):
        """Transmittance at cutoff should be approximately midpoint."""
        filt = IRCutFilter(cutoff_wl=0.65, passband_t=0.95, stopband_t=0.01)
        t_at_cutoff = filt.transmittance(np.array([0.65]))[0]
        midpoint = (0.95 + 0.01) / 2.0
        assert t_at_cutoff == pytest.approx(midpoint, abs=0.05)

    def test_monotonically_decreasing(self):
        """Transmittance should decrease monotonically through the transition."""
        filt = IRCutFilter(cutoff_wl=0.65)
        wl = np.linspace(0.55, 0.75, 100)
        t = filt.transmittance(wl)
        # Allow small numerical noise
        diffs = np.diff(t)
        assert np.all(diffs <= 1e-10)


class TestLensTransmittance:
    """Tests for LensTransmittance model."""

    def test_ar_coated_high_transmittance(self):
        """AR-coated lens should have high transmittance in visible."""
        lens = LensTransmittance(num_elements=6, ar_coating=True)
        t = lens.transmittance(np.array([0.55]))
        assert t[0] > 0.85

    def test_more_elements_lower_transmittance(self):
        """More lens elements should give lower transmittance."""
        lens_4 = LensTransmittance(num_elements=4)
        lens_8 = LensTransmittance(num_elements=8)
        t4 = lens_4.transmittance(np.array([0.55]))[0]
        t8 = lens_8.transmittance(np.array([0.55]))[0]
        assert t4 > t8

    def test_no_elements_unity(self):
        """Zero elements should give transmittance = 1."""
        lens = LensTransmittance(num_elements=0)
        t = lens.transmittance(WL_VIS)
        assert np.allclose(t, 1.0)

    def test_uncoated_lower_than_coated(self):
        """Uncoated lens should have lower transmittance than coated."""
        coated = LensTransmittance(num_elements=6, ar_coating=True)
        uncoated = LensTransmittance(num_elements=6, ar_coating=False)
        tc = coated.transmittance(np.array([0.55]))[0]
        tu = uncoated.transmittance(np.array([0.55]))[0]
        assert tc > tu


class TestModuleOptics:
    """Tests for ModuleOptics combined model."""

    def test_smartphone_module(self):
        """Smartphone module should have reasonable visible transmittance."""
        mod = ModuleOptics.smartphone_module()
        t = mod.total_transmittance(np.array([0.55]))
        # Should be moderately high in green
        assert 0.5 < t[0] < 1.0

    def test_industrial_module(self):
        """Industrial module should pass more light than smartphone in red/NIR."""
        sm = ModuleOptics.smartphone_module()
        ind = ModuleOptics.industrial_module()
        wl = np.array([0.70])  # Near IR cut edge
        t_sm = sm.total_transmittance(wl)[0]
        t_ind = ind.total_transmittance(wl)[0]
        assert t_ind > t_sm

    def test_no_optics_unity(self):
        """no_optics should give transmittance ~1.0 in visible."""
        mod = ModuleOptics.no_optics()
        t = mod.total_transmittance(WL_VIS)
        assert np.all(t > 0.99)

    def test_energy_conservation(self):
        """Transmittance must never exceed 1.0."""
        mod = ModuleOptics.smartphone_module()
        t = mod.total_transmittance(WL_VIS)
        assert np.all(t <= 1.0 + 1e-10)
        assert np.all(t >= 0.0)


class TestSignalCalculator:
    """Tests for SignalCalculator integration."""

    @pytest.fixture()
    def rgb_qe(self):
        """Simple RGB QE curves (Gaussian approximations)."""
        wl_nm = WL_VIS * 1000.0
        qe_r = 0.5 * np.exp(-0.5 * ((wl_nm - 620.0) / 40.0) ** 2)
        qe_g = 0.6 * np.exp(-0.5 * ((wl_nm - 530.0) / 35.0) ** 2)
        qe_b = 0.4 * np.exp(-0.5 * ((wl_nm - 450.0) / 30.0) ** 2)
        return {
            "R_0": qe_r,
            "G_0": qe_g,
            "G_1": qe_g,
            "B_0": qe_b,
        }

    @pytest.fixture()
    def bayer_map(self):
        """Standard RGGB Bayer pattern."""
        return [["R_0", "G_0"], ["G_1", "B_0"]]

    def test_spectral_irradiance_shape(self):
        """Spectral irradiance should have same length as wavelength grid."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        mod = ModuleOptics.no_optics()
        irr = calc.compute_spectral_irradiance(ill, scene, mod)
        assert irr.shape == WL_VIS.shape

    def test_flat_illuminant_flat_scene_no_optics(self):
        """Flat illuminant + flat scene + no optics = uniform irradiance."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        mod = ModuleOptics.no_optics()
        irr = calc.compute_spectral_irradiance(ill, scene, mod)
        # Should be close to 0.5 everywhere
        assert np.allclose(irr, 0.5, atol=0.01)

    def test_signal_integration_positive(self, rgb_qe, bayer_map):
        """All pixel signals should be positive under normal conditions."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        mod = ModuleOptics.smartphone_module()
        signals = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        for name, sig in signals.items():
            assert sig > 0, f"Signal for {name} should be positive"

    def test_signal_bounded(self, rgb_qe, bayer_map):
        """Signal should be bounded: not exceed integral of illuminant * QE_max."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(1.0, WL_VIS)  # Perfect reflector
        mod = ModuleOptics.no_optics()
        signals = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        # Maximum possible signal: integral of QE over wavelength range
        for name, qe in rgb_qe.items():
            max_signal = float(np.trapezoid(qe, WL_VIS))
            assert signals[name] <= max_signal + 1e-10

    def test_channel_signal(self, rgb_qe, bayer_map):
        """Channel signal should group pixels by colour."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        mod = ModuleOptics.no_optics()
        ch_sig = calc.compute_channel_signal(ill, scene, mod, rgb_qe, bayer_map)
        assert "R" in ch_sig
        assert "G" in ch_sig
        assert "B" in ch_sig
        # G channel should average G_0 and G_1 (which are identical)
        pixel_sig = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        assert ch_sig["G"] == pytest.approx(
            (pixel_sig["G_0"] + pixel_sig["G_1"]) / 2.0, rel=1e-6,
        )

    def test_white_balance_gains_flat_target(self, rgb_qe, bayer_map):
        """White balance gains with flat target should normalise channels."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        mod = ModuleOptics.smartphone_module()
        gains = calc.white_balance_gains(ill, mod, rgb_qe, bayer_map)
        # G gain should be 1.0
        assert gains["G"] == pytest.approx(1.0, abs=1e-6)
        # R and B gains should be positive
        assert gains["R"] > 0
        assert gains["B"] > 0

    def test_white_balance_applied_equalises(self, rgb_qe, bayer_map):
        """Applying WB gains to channel signals should equalise them."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        mod = ModuleOptics.smartphone_module()
        grey = SceneReflectance.flat(0.18, WL_VIS)
        gains = calc.white_balance_gains(ill, mod, rgb_qe, bayer_map)
        ch_sig = calc.compute_channel_signal(ill, grey, mod, rgb_qe, bayer_map)
        # After applying gains, all channels should be equal
        balanced = {ch: ch_sig[ch] * gains[ch] for ch in ch_sig}
        values = list(balanced.values())
        for v in values:
            assert v == pytest.approx(values[0], rel=1e-6)

    def test_compute_signal_electrons(self, rgb_qe, bayer_map):
        """Electron count should be positive and finite."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        mod = ModuleOptics.smartphone_module()
        pixel_sig = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        electrons = calc.compute_signal_electrons(
            pixel_sig, exposure_time=1.0 / 60, pixel_area=1.0, f_number=2.0,
        )
        for name, ne in electrons.items():
            assert ne > 0, f"Electrons for {name} should be positive"
            assert np.isfinite(ne), f"Electrons for {name} should be finite"

    def test_compute_snr_positive(self, rgb_qe, bayer_map):
        """SNR should be positive when signal is present."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        mod = ModuleOptics.smartphone_module()
        pixel_sig = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        electrons = calc.compute_signal_electrons(
            pixel_sig, exposure_time=1.0 / 60, pixel_area=1.0, f_number=2.0,
        )
        snr = calc.compute_snr(electrons)
        for name, s in snr.items():
            assert s > 0, f"SNR for {name} should be positive"

    def test_color_ratio(self, rgb_qe, bayer_map):
        """Colour ratios should be positive and finite."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        mod = ModuleOptics.no_optics()
        ch_sig = calc.compute_channel_signal(ill, scene, mod, rgb_qe, bayer_map)
        ratios = SignalCalculator.color_ratio(ch_sig)
        assert "R/G" in ratios
        assert "B/G" in ratios
        assert ratios["R/G"] > 0
        assert ratios["B/G"] > 0
        assert np.isfinite(ratios["R/G"])
        assert np.isfinite(ratios["B/G"])

    def test_color_ratio_missing_channel(self):
        """color_ratio should raise on missing channels."""
        with pytest.raises(ValueError, match="Missing"):
            SignalCalculator.color_ratio({"R": 1.0, "B": 0.5})

    def test_color_ratio_zero_green(self):
        """color_ratio should raise when green is zero."""
        with pytest.raises(ValueError, match="positive"):
            SignalCalculator.color_ratio({"R": 1.0, "G": 0.0, "B": 0.5})

    def test_energy_conservation_signal(self, rgb_qe, bayer_map):
        """Total signal should not exceed the total input energy."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(1.0, WL_VIS)
        mod = ModuleOptics.no_optics()
        signals = calc.compute_pixel_signal(ill, scene, mod, rgb_qe)
        # Each pixel signal should be <= integral of 1.0 * QE over wavelength
        wl_range = WL_VIS[-1] - WL_VIS[0]
        for name, sig in signals.items():
            assert sig <= wl_range + 1e-6, (
                f"Signal {name}={sig:.6f} exceeds wavelength range {wl_range:.6f}"
            )
