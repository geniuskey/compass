"""Integration tests for the signal chain pipeline.

Tests the full flow: Illuminant -> Scene -> Optics -> QE -> Signal.
Validates physical constraints, spectral relationships, and end-to-end
correctness of the radiometric signal chain.
"""

import numpy as np
import pytest

from compass.analysis.signal_calculator import SignalCalculator
from compass.materials.database import MaterialDB
from compass.optics.module_optics import IRCutFilter, LensTransmittance, ModuleOptics
from compass.sources.illuminant import Illuminant
from compass.sources.scene import SceneReflectance

# Common wavelength grids
WL_VIS = np.linspace(0.38, 0.78, 401)  # Visible range, 1 nm steps


def _make_rgb_qe(wl: np.ndarray) -> dict[str, np.ndarray]:
    """Create simple Gaussian RGB QE curves on the given wavelength grid."""
    wl_nm = wl * 1000.0
    qe_r = 0.5 * np.exp(-0.5 * ((wl_nm - 620.0) / 40.0) ** 2)
    qe_g = 0.6 * np.exp(-0.5 * ((wl_nm - 530.0) / 35.0) ** 2)
    qe_b = 0.4 * np.exp(-0.5 * ((wl_nm - 450.0) / 30.0) ** 2)
    return {"R_0": qe_r, "G_0": qe_g, "G_1": qe_g, "B_0": qe_b}


def _make_bayer_map() -> list[list[str]]:
    """Standard RGGB Bayer pattern."""
    return [["R_0", "G_0"], ["G_1", "B_0"]]


class TestEndToEndSignalChain:
    """End-to-end tests exercising the full signal chain."""

    def test_d65_grey_card_signal(self):
        """D65 + 18% grey + smartphone module should give positive signal for all channels.

        Under D65 daylight illumination, green channel should generally be the
        strongest due to QE peak alignment with the D65 spectral peak region.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        ch_sig = calc.compute_channel_signal(ill, scene, module, qe, bayer)

        assert ch_sig["R"] > 0, "Red channel signal must be positive under D65"
        assert ch_sig["G"] > 0, "Green channel signal must be positive under D65"
        assert ch_sig["B"] > 0, "Blue channel signal must be positive under D65"
        # Green should be strongest because QE peak is 0.6 and D65 has good
        # green content; the exact ordering also depends on optics transmittance
        assert ch_sig["G"] > ch_sig["B"], (
            "Green channel should exceed blue under D65 + grey card"
        )

    def test_illuminant_a_vs_d65_color_shift(self):
        """CIE A (2856K) should produce a higher R/G ratio than D65 (warmer light).

        Incandescent (A) has relatively more red power than daylight (D65),
        so the R/G ratio should be higher.
        """
        calc = SignalCalculator(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        ill_a = Illuminant.cie_a(WL_VIS)
        ill_d65 = Illuminant.cie_d65(WL_VIS)

        ch_a = calc.compute_channel_signal(ill_a, scene, module, qe, bayer)
        ch_d65 = calc.compute_channel_signal(ill_d65, scene, module, qe, bayer)

        ratio_a = SignalCalculator.color_ratio(ch_a)
        ratio_d65 = SignalCalculator.color_ratio(ch_d65)

        assert ratio_a["R/G"] > ratio_d65["R/G"], (
            f"Illuminant A R/G ({ratio_a['R/G']:.4f}) should exceed "
            f"D65 R/G ({ratio_d65['R/G']:.4f}) because A is warmer"
        )

    def test_ir_filter_effect(self):
        """Comparing signal with and without IR filter: red channel should be most affected.

        The IR cut filter attenuates wavelengths near and above 650nm, which
        overlaps the red QE curve most significantly.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        qe = _make_rgb_qe(WL_VIS)

        mod_with_ir = ModuleOptics.smartphone_module()  # Has IR filter at 650nm
        mod_no_ir = ModuleOptics.no_optics()  # No IR filter

        sig_with = calc.compute_pixel_signal(ill, scene, mod_with_ir, qe)
        sig_without = calc.compute_pixel_signal(ill, scene, mod_no_ir, qe)

        # Red channel should lose the most signal from the IR filter since
        # its QE peak at 620nm is closest to the 650nm cutoff
        r_reduction = 1.0 - sig_with["R_0"] / sig_without["R_0"]
        g_reduction = 1.0 - sig_with["G_0"] / sig_without["G_0"]
        b_reduction = 1.0 - sig_with["B_0"] / sig_without["B_0"]

        assert r_reduction > g_reduction, (
            f"Red reduction ({r_reduction:.4f}) should exceed green ({g_reduction:.4f})"
        )
        assert r_reduction > b_reduction, (
            f"Red reduction ({r_reduction:.4f}) should exceed blue ({b_reduction:.4f})"
        )

    def test_white_balance_corrects_color(self):
        """After applying WB gains, R/G and B/G should be exactly 1.0 for grey target.

        White balance gains are computed to equalize channel signals for a flat
        grey target, so applying them should bring all ratios to unity.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        gains = calc.white_balance_gains(ill, module, qe, bayer)
        grey = SceneReflectance.flat(0.18, WL_VIS)
        ch_sig = calc.compute_channel_signal(ill, grey, module, qe, bayer)

        balanced = {ch: ch_sig[ch] * gains[ch] for ch in ch_sig}
        ratios = SignalCalculator.color_ratio(balanced)

        assert ratios["R/G"] == pytest.approx(1.0, abs=1e-6), (
            f"After WB, R/G should be 1.0, got {ratios['R/G']:.6f}"
        )
        assert ratios["B/G"] == pytest.approx(1.0, abs=1e-6), (
            f"After WB, B/G should be 1.0, got {ratios['B/G']:.6f}"
        )

    def test_monochromatic_source_single_channel(self):
        """Monochromatic sources should primarily excite the expected channel.

        450nm -> Blue, 550nm -> Green, 650nm -> Red.
        """
        calc = SignalCalculator(WL_VIS)
        module = ModuleOptics.no_optics()
        scene = SceneReflectance.flat(1.0, WL_VIS)
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        test_cases = [
            (0.450, "B", "Blue at 450nm should dominate blue channel"),
            (0.530, "G", "Green at 530nm should dominate green channel"),
            (0.620, "R", "Red at 620nm should dominate red channel"),
        ]

        for center_wl, expected_ch, msg in test_cases:
            ill = Illuminant.monochromatic(center_wl, WL_VIS, fwhm=0.01)
            ch_sig = calc.compute_channel_signal(ill, scene, module, qe, bayer)
            dominant = max(ch_sig, key=ch_sig.get)
            assert dominant == expected_ch, (
                f"{msg}: dominant={dominant} with signals {ch_sig}"
            )

    def test_black_target_zero_signal(self):
        """A black target (R~0) should produce near-zero signal for all channels."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.0, WL_VIS)  # Perfect absorber
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)

        signals = calc.compute_pixel_signal(ill, scene, module, qe)

        for name, sig in signals.items():
            assert sig == pytest.approx(0.0, abs=1e-12), (
                f"Black target signal for {name} should be ~0, got {sig:.6e}"
            )

    def test_signal_scales_with_reflectance(self):
        """Doubling reflectance should roughly double the signal.

        Since signal = integral(L * R * T * QE) dlam, and R is a constant
        multiplier for flat reflectance, signal should scale linearly.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)

        scene_18 = SceneReflectance.flat(0.18, WL_VIS)
        scene_36 = SceneReflectance.flat(0.36, WL_VIS)

        sig_18 = calc.compute_pixel_signal(ill, scene_18, module, qe)
        sig_36 = calc.compute_pixel_signal(ill, scene_36, module, qe)

        for name in sig_18:
            ratio = sig_36[name] / sig_18[name]
            assert ratio == pytest.approx(2.0, rel=1e-6), (
                f"Doubling reflectance should double signal for {name}: "
                f"ratio={ratio:.6f}"
            )


class TestSignalChainPhysics:
    """Tests that validate physical constraints and relationships."""

    def test_energy_conservation_signal_bounded(self):
        """Total signal across all channels should not exceed the integrated irradiance.

        Each pixel's signal = integral(irradiance * QE) dlam, and QE <= 1,
        so signal <= integral(irradiance) dlam.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.flat(WL_VIS)
        scene = SceneReflectance.flat(1.0, WL_VIS)
        module = ModuleOptics.no_optics()
        qe = _make_rgb_qe(WL_VIS)

        irradiance = calc.compute_spectral_irradiance(ill, scene, module)
        total_irradiance = float(np.trapezoid(irradiance, WL_VIS))

        signals = calc.compute_pixel_signal(ill, scene, module, qe)
        for name, sig in signals.items():
            assert sig <= total_irradiance + 1e-10, (
                f"Signal for {name} ({sig:.6f}) exceeds total irradiance "
                f"({total_irradiance:.6f})"
            )

    def test_higher_temperature_bluer(self):
        """Higher CCT blackbody should give a higher B/G ratio (bluer spectrum).

        Wien's law: hotter blackbodies peak at shorter wavelengths, shifting
        spectral power toward blue.
        """
        calc = SignalCalculator(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        module = ModuleOptics.no_optics()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        ill_cool = Illuminant.blackbody(3000.0, WL_VIS)
        ill_hot = Illuminant.blackbody(10000.0, WL_VIS)

        ch_cool = calc.compute_channel_signal(ill_cool, scene, module, qe, bayer)
        ch_hot = calc.compute_channel_signal(ill_hot, scene, module, qe, bayer)

        ratio_cool = SignalCalculator.color_ratio(ch_cool)
        ratio_hot = SignalCalculator.color_ratio(ch_hot)

        assert ratio_hot["B/G"] > ratio_cool["B/G"], (
            f"10000K B/G ({ratio_hot['B/G']:.4f}) should exceed "
            f"3000K B/G ({ratio_cool['B/G']:.4f})"
        )

    def test_led_vs_blackbody_spectral_difference(self):
        """LED and equivalent-CCT blackbody should give different color ratios.

        An LED has a blue pump peak that a smooth blackbody lacks, so even
        at the same CCT the channel responses should differ.
        """
        calc = SignalCalculator(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        module = ModuleOptics.no_optics()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        cct = 5000.0
        ill_led = Illuminant.led_white(cct, WL_VIS)
        ill_bb = Illuminant.blackbody(cct, WL_VIS)

        ch_led = calc.compute_channel_signal(ill_led, scene, module, qe, bayer)
        ch_bb = calc.compute_channel_signal(ill_bb, scene, module, qe, bayer)

        ratio_led = SignalCalculator.color_ratio(ch_led)
        ratio_bb = SignalCalculator.color_ratio(ch_bb)

        # The ratios should differ because LED spectrum has a blue pump peak
        assert ratio_led["B/G"] != pytest.approx(ratio_bb["B/G"], rel=0.01), (
            f"LED B/G ({ratio_led['B/G']:.4f}) should differ from "
            f"blackbody B/G ({ratio_bb['B/G']:.4f})"
        )

    def test_lens_elements_reduce_signal(self):
        """More lens elements should reduce total signal due to transmission losses."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        qe = _make_rgb_qe(WL_VIS)

        mod_4 = ModuleOptics(
            lens=LensTransmittance(num_elements=4, ar_coating=True),
            ir_filter=IRCutFilter(cutoff_wl=10.0, passband_t=1.0, stopband_t=1.0),
        )
        mod_8 = ModuleOptics(
            lens=LensTransmittance(num_elements=8, ar_coating=True),
            ir_filter=IRCutFilter(cutoff_wl=10.0, passband_t=1.0, stopband_t=1.0),
        )

        sig_4 = calc.compute_pixel_signal(ill, scene, mod_4, qe)
        sig_8 = calc.compute_pixel_signal(ill, scene, mod_8, qe)

        for name in sig_4:
            assert sig_4[name] > sig_8[name], (
                f"4-element lens should transmit more for {name}: "
                f"4-elem={sig_4[name]:.6f}, 8-elem={sig_8[name]:.6f}"
            )

    def test_ir_cutoff_wavelength_effect(self):
        """Moving IR cut to 600nm should reduce red signal more than a 700nm cutoff.

        A 600nm cutoff clips deeper into the red QE band (peak at 620nm)
        compared to a 700nm cutoff.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.5, WL_VIS)
        qe = _make_rgb_qe(WL_VIS)

        mod_600 = ModuleOptics(
            lens=LensTransmittance(num_elements=0),
            ir_filter=IRCutFilter(cutoff_wl=0.60, transition_width=0.03),
        )
        mod_700 = ModuleOptics(
            lens=LensTransmittance(num_elements=0),
            ir_filter=IRCutFilter(cutoff_wl=0.70, transition_width=0.03),
        )

        sig_600 = calc.compute_pixel_signal(ill, scene, mod_600, qe)
        sig_700 = calc.compute_pixel_signal(ill, scene, mod_700, qe)

        assert sig_700["R_0"] > sig_600["R_0"], (
            f"700nm cutoff R signal ({sig_700['R_0']:.6f}) should exceed "
            f"600nm cutoff R signal ({sig_600['R_0']:.6f})"
        )
        # Green and blue should be less affected
        r_ratio = sig_600["R_0"] / sig_700["R_0"]
        g_ratio = sig_600["G_0"] / sig_700["G_0"]
        assert r_ratio < g_ratio, (
            "Red should be relatively more attenuated by the 600nm cutoff than green"
        )

    def test_snr_increases_with_exposure(self):
        """Longer exposure time should yield higher SNR (more signal electrons)."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)

        pixel_sig = calc.compute_pixel_signal(ill, scene, module, qe)

        electrons_short = calc.compute_signal_electrons(
            pixel_sig, exposure_time=1.0 / 120, pixel_area=1.0, f_number=2.0,
        )
        electrons_long = calc.compute_signal_electrons(
            pixel_sig, exposure_time=1.0 / 30, pixel_area=1.0, f_number=2.0,
        )

        snr_short = calc.compute_snr(electrons_short)
        snr_long = calc.compute_snr(electrons_long)

        for name in snr_short:
            assert snr_long[name] > snr_short[name], (
                f"Longer exposure should give higher SNR for {name}: "
                f"long={snr_long[name]:.2f}, short={snr_short[name]:.2f}"
            )

    def test_snr_decreases_with_read_noise(self):
        """Higher read noise should result in lower SNR."""
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)

        pixel_sig = calc.compute_pixel_signal(ill, scene, module, qe)
        electrons = calc.compute_signal_electrons(
            pixel_sig, exposure_time=1.0 / 60, pixel_area=1.0, f_number=2.0,
        )

        snr_low_noise = calc.compute_snr(electrons, read_noise=1.0)
        snr_high_noise = calc.compute_snr(electrons, read_noise=10.0)

        for name in snr_low_noise:
            assert snr_low_noise[name] > snr_high_noise[name], (
                f"Lower read noise should give higher SNR for {name}: "
                f"low_noise={snr_low_noise[name]:.2f}, "
                f"high_noise={snr_high_noise[name]:.2f}"
            )


class TestSignalChainWithPixelStack:
    """Tests combining the signal chain with material database data."""

    def test_signal_with_material_db_qe(self):
        """Build QE-like curves using MaterialDB silicon absorption, then compute signal.

        Silicon's extinction coefficient k determines absorption depth. We
        construct a crude QE by computing the fraction of light absorbed in
        a finite thickness of silicon, then use it to compute signal.
        """
        db = MaterialDB()
        calc = SignalCalculator(WL_VIS)

        # Build approximate QE from silicon absorption: QE ~ 1 - exp(-alpha * d)
        # where alpha = 4*pi*k / lambda
        thickness_um = 2.0
        qe_si = np.zeros_like(WL_VIS)
        for i, wl in enumerate(WL_VIS):
            _, k = db.get_nk("silicon", wl)
            alpha = 4.0 * np.pi * k / wl  # absorption coefficient in 1/um
            qe_si[i] = 1.0 - np.exp(-alpha * thickness_um)

        qe_dict = {"SI_0": qe_si}

        ill = Illuminant.cie_d65(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()

        signals = calc.compute_pixel_signal(ill, scene, module, qe_dict)
        assert signals["SI_0"] > 0, (
            "Signal with silicon-derived QE should be positive"
        )
        assert np.isfinite(signals["SI_0"]), (
            "Signal with silicon-derived QE should be finite"
        )

    def test_multiple_illuminants_comparison(self):
        """Different illuminants should produce different but plausible channel signals.

        D65, A, F11, and LED are physically distinct light sources that should
        produce different spectral responses.
        """
        calc = SignalCalculator(WL_VIS)
        scene = SceneReflectance.flat(0.18, WL_VIS)
        module = ModuleOptics.smartphone_module()
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        illuminants = {
            "D65": Illuminant.cie_d65(WL_VIS),
            "A": Illuminant.cie_a(WL_VIS),
            "F11": Illuminant.cie_f11(WL_VIS),
            "LED": Illuminant.led_white(5000.0, WL_VIS),
        }

        results = {}
        for name, ill in illuminants.items():
            ch_sig = calc.compute_channel_signal(ill, scene, module, qe, bayer)
            results[name] = ch_sig
            # All signals should be positive
            for ch, sig in ch_sig.items():
                assert sig > 0, (
                    f"Signal for {name}/{ch} must be positive, got {sig:.6e}"
                )

        # Each illuminant should produce a unique R/G ratio
        rg_ratios = {}
        for name, ch_sig in results.items():
            rg_ratios[name] = ch_sig["R"] / ch_sig["G"]

        # Verify they are all different (no two within 1% of each other)
        names = list(rg_ratios.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r_i = rg_ratios[names[i]]
                r_j = rg_ratios[names[j]]
                assert r_i != pytest.approx(r_j, rel=0.01), (
                    f"{names[i]} R/G ({r_i:.4f}) should differ from "
                    f"{names[j]} R/G ({r_j:.4f})"
                )

    def test_macbeth_patches_color_separation(self):
        """Red patch should have highest R signal, blue patch highest B, green patch highest G.

        Using Macbeth ColorChecker patches: 15=Red, 14=Green, 13=Blue.
        Each should produce the strongest signal in its corresponding channel.
        """
        calc = SignalCalculator(WL_VIS)
        ill = Illuminant.cie_d65(WL_VIS)
        module = ModuleOptics.no_optics()  # No optics to keep it clean
        qe = _make_rgb_qe(WL_VIS)
        bayer = _make_bayer_map()

        # Macbeth patches: 15=Red, 14=Green, 12=Blue
        # (Patch 12 is the purer blue; patch 13 has a secondary green peak)
        red_patch = SceneReflectance.macbeth_patch(15, WL_VIS)
        green_patch = SceneReflectance.macbeth_patch(14, WL_VIS)
        blue_patch = SceneReflectance.macbeth_patch(12, WL_VIS)

        ch_red = calc.compute_channel_signal(ill, red_patch, module, qe, bayer)
        ch_green = calc.compute_channel_signal(ill, green_patch, module, qe, bayer)
        ch_blue = calc.compute_channel_signal(ill, blue_patch, module, qe, bayer)

        # Red patch should have highest R signal relative to its other channels
        assert ch_red["R"] > ch_red["G"], (
            f"Red patch: R ({ch_red['R']:.6f}) should exceed G ({ch_red['G']:.6f})"
        )
        assert ch_red["R"] > ch_red["B"], (
            f"Red patch: R ({ch_red['R']:.6f}) should exceed B ({ch_red['B']:.6f})"
        )

        # Green patch should have highest G signal
        assert ch_green["G"] > ch_green["R"], (
            f"Green patch: G ({ch_green['G']:.6f}) should exceed R ({ch_green['R']:.6f})"
        )
        assert ch_green["G"] > ch_green["B"], (
            f"Green patch: G ({ch_green['G']:.6f}) should exceed B ({ch_green['B']:.6f})"
        )

        # Blue patch should have highest B signal
        assert ch_blue["B"] > ch_blue["R"], (
            f"Blue patch: B ({ch_blue['B']:.6f}) should exceed R ({ch_blue['R']:.6f})"
        )
        assert ch_blue["B"] > ch_blue["G"], (
            f"Blue patch: B ({ch_blue['B']:.6f}) should exceed G ({ch_blue['G']:.6f})"
        )
