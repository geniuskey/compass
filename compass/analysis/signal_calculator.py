"""Signal calculator: integrates illuminant x scene x optics x QE.

Computes pixel-level signal from scene illumination through the optical chain
to the sensor. Implements the core radiometric integral:

    Signal_i(lam) = integral L(lam) * R_scene(lam) * T_lens(lam) * T_IR(lam) * QE_i(lam) dlam

Also provides conversion to electron counts, SNR estimation, white balance
gain computation, and colour ratio analysis.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.optics.module_optics import ModuleOptics
from compass.sources.illuminant import C_LIGHT, H_PLANCK, Illuminant
from compass.sources.scene import SceneReflectance

logger = logging.getLogger(__name__)


class SignalCalculator:
    """Compute pixel signal from scene through optical chain to sensor.

    Attributes:
        wavelengths: Common wavelength grid in micrometers.
    """

    def __init__(self, wavelengths: np.ndarray) -> None:
        """Initialise the calculator with a common wavelength grid.

        Args:
            wavelengths: Common wavelength grid in micrometers.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        logger.debug(
            "SignalCalculator: %d wavelength points [%.3f, %.3f] um.",
            len(self.wavelengths),
            self.wavelengths[0],
            self.wavelengths[-1],
        )

    def compute_spectral_irradiance(
        self,
        illuminant: Illuminant,
        scene: SceneReflectance,
        module: ModuleOptics,
    ) -> np.ndarray:
        """Compute spectral irradiance at the sensor plane.

        E(lam) = L(lam) * R(lam) * T_module(lam)

        All inputs are interpolated onto the calculator's wavelength grid.

        Args:
            illuminant: Light source spectral power distribution.
            scene: Scene reflectance model.
            module: Camera module optics.

        Returns:
            Spectral irradiance array (same shape as wavelengths).
        """
        l_spectrum = illuminant.interpolate(self.wavelengths)
        r_spectrum = scene.interpolate(self.wavelengths)
        t_spectrum = module.total_transmittance(self.wavelengths)
        irradiance = l_spectrum * r_spectrum * t_spectrum
        logger.debug("Spectral irradiance: peak=%.6e", np.max(irradiance))
        return np.asarray(irradiance, dtype=np.float64)

    def compute_pixel_signal(
        self,
        illuminant: Illuminant,
        scene: SceneReflectance,
        module: ModuleOptics,
        qe_per_pixel: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute integrated signal per pixel.

        Signal_i = integral E(lam) * QE_i(lam) dlam

        Uses trapezoidal integration over the wavelength grid.

        Args:
            illuminant: Light source spectral power distribution.
            scene: Scene reflectance model.
            module: Camera module optics.
            qe_per_pixel: Dict mapping pixel name to QE spectrum array
                (must be on the same wavelength grid as the calculator).

        Returns:
            Dict mapping pixel name to relative signal value.
        """
        irradiance = self.compute_spectral_irradiance(illuminant, scene, module)
        signals: dict[str, float] = {}
        for name, qe in qe_per_pixel.items():
            qe_arr = np.asarray(qe, dtype=np.float64)
            integrand = irradiance * qe_arr
            signal = float(np.trapezoid(integrand, self.wavelengths))
            signals[name] = signal
            logger.debug("Pixel '%s': signal=%.6e", name, signal)
        return signals

    def compute_channel_signal(
        self,
        illuminant: Illuminant,
        scene: SceneReflectance,
        module: ModuleOptics,
        qe_per_pixel: dict[str, np.ndarray],
        bayer_map: list[list[str]],
    ) -> dict[str, float]:
        """Compute signal per colour channel (averaging same-colour pixels).

        Groups pixels by colour channel extracted from the Bayer map and
        averages the signal of all pixels in each channel.

        Args:
            illuminant: Light source spectral power distribution.
            scene: Scene reflectance model.
            module: Camera module optics.
            qe_per_pixel: Dict mapping pixel name to QE spectrum array.
            bayer_map: 2D list of pixel names defining the Bayer pattern.

        Returns:
            Dict mapping colour channel name to averaged signal
            (e.g., {'R': signal_R, 'G': signal_G, 'B': signal_B}).
        """
        pixel_signals = self.compute_pixel_signal(illuminant, scene, module, qe_per_pixel)

        # Group by colour channel (first part of name before '_')
        channel_sums: dict[str, float] = {}
        channel_counts: dict[str, int] = {}
        for row in bayer_map:
            for pixel_name in row:
                channel = pixel_name.split("_")[0]
                channel_sums[channel] = channel_sums.get(channel, 0.0) + pixel_signals.get(pixel_name, 0.0)
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

        channel_signals: dict[str, float] = {}
        for ch in sorted(channel_sums.keys()):
            channel_signals[ch] = channel_sums[ch] / max(channel_counts[ch], 1)
        logger.debug("Channel signals: %s", channel_signals)
        return channel_signals

    def compute_signal_electrons(
        self,
        pixel_signal: dict[str, float],
        exposure_time: float,
        pixel_area: float,
        f_number: float,
    ) -> dict[str, float]:
        """Convert relative signal to electron count estimate.

        N_e = Signal * t_exp * A_pixel / (4 * F#^2) * scale_factor

        This is a simplified order-of-magnitude estimate. The scale factor
        is derived from photon energy at the mean wavelength of the grid.

        Args:
            pixel_signal: From compute_pixel_signal.
            exposure_time: Exposure time in seconds.
            pixel_area: Pixel area in um^2.
            f_number: Lens f-number.

        Returns:
            Estimated electron count per pixel.
        """
        # Mean wavelength for photon energy estimate
        lam_mean_m = float(np.mean(self.wavelengths)) * 1e-6
        photon_energy = H_PLANCK * C_LIGHT / lam_mean_m  # Joules
        # Pixel area in m^2
        a_pixel_m2 = pixel_area * 1e-12
        # Geometric factor: sensor plane irradiance ~ pi / (4 * F#^2)
        geometric_factor = np.pi / (4.0 * f_number**2)

        electrons: dict[str, float] = {}
        for name, signal in pixel_signal.items():
            # Scale: relative signal -> photon flux -> electrons
            n_electrons = signal * exposure_time * a_pixel_m2 * geometric_factor / photon_energy
            electrons[name] = max(n_electrons, 0.0)
        logger.debug("Electron counts: %s", electrons)
        return electrons

    def compute_snr(
        self,
        signal_electrons: dict[str, float],
        read_noise: float = 2.0,
        dark_current: float = 0.1,
        exposure_time: float = 1.0 / 60.0,
    ) -> dict[str, float]:
        """Estimate signal-to-noise ratio.

        SNR = N_signal / sqrt(N_signal + N_dark + N_read^2)

        Args:
            signal_electrons: From compute_signal_electrons.
            read_noise: Read noise in electrons (rms).
            dark_current: Dark current in e/s.
            exposure_time: Exposure time in seconds.

        Returns:
            SNR per pixel.
        """
        n_dark = dark_current * exposure_time
        snr: dict[str, float] = {}
        for name, n_signal in signal_electrons.items():
            noise_variance = n_signal + n_dark + read_noise**2
            if noise_variance > 0:
                snr[name] = n_signal / np.sqrt(noise_variance)
            else:
                snr[name] = 0.0
        logger.debug("SNR: %s", snr)
        return snr

    def white_balance_gains(
        self,
        illuminant: Illuminant,
        module: ModuleOptics,
        qe_per_pixel: dict[str, np.ndarray],
        bayer_map: list[list[str]],
    ) -> dict[str, float]:
        """Compute white balance gains for a given illuminant.

        Uses the grey-world assumption: gains normalise R/G/B channels to
        equal response for a flat 18% grey target under the specified illuminant.

        Args:
            illuminant: Light source for white balance.
            module: Camera module optics.
            qe_per_pixel: Dict mapping pixel name to QE spectrum.
            bayer_map: 2D list of pixel names defining the Bayer pattern.

        Returns:
            Dict of gains (e.g., {'R': gain_R, 'G': gain_G, 'B': gain_B})
            normalised so that G = 1.0.
        """
        grey = SceneReflectance.flat(0.18, self.wavelengths)
        channel_signals = self.compute_channel_signal(
            illuminant, grey, module, qe_per_pixel, bayer_map,
        )

        # Find green channel signal for normalisation
        g_signal = channel_signals.get("G", None)
        if g_signal is None:
            # Fall back to first channel
            g_signal = next(iter(channel_signals.values()))
            logger.warning("No 'G' channel found; normalising to first channel.")

        gains: dict[str, float] = {}
        for ch, signal in channel_signals.items():
            if signal > 0:
                gains[ch] = g_signal / signal
            else:
                gains[ch] = 1.0
                logger.warning("Channel '%s' has zero signal; gain set to 1.0.", ch)
        logger.debug("White balance gains: %s", gains)
        return gains

    @staticmethod
    def color_ratio(channel_signal: dict[str, float]) -> dict[str, float]:
        """Compute R/G and B/G ratios.

        Args:
            channel_signal: Dict with at least 'R', 'G', 'B' keys.

        Returns:
            Dict with 'R/G' and 'B/G' ratio values.

        Raises:
            ValueError: If required channels are missing or G is zero.
        """
        for key in ("R", "G", "B"):
            if key not in channel_signal:
                raise ValueError(f"Missing required channel '{key}' in signal dict.")
        g_val = channel_signal["G"]
        if g_val <= 0:
            raise ValueError("Green channel signal must be positive for ratio computation.")
        return {
            "R/G": channel_signal["R"] / g_val,
            "B/G": channel_signal["B"] / g_val,
        }
