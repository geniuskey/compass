"""Spectral illuminant models for signal-level simulation.

Provides standard and custom light source spectral power distributions (SPDs)
for radiometric signal chain calculations. Supports blackbody, CIE standard
illuminants, LED, monochromatic, and flat (equal-energy) sources.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# CIE daylight basis functions S0, S1, S2 (CIE 015), 380-780 nm in 10 nm steps.
_CIE_D_WL_NM = np.arange(380.0, 781.0, 10.0)
_CIE_D_S0 = np.array([
    63.4, 65.8, 94.8, 104.8, 105.9, 96.8, 113.9, 125.6, 125.5, 121.3, 121.3,
    113.5, 113.1, 110.8, 106.5, 108.8, 105.3, 104.4, 100.0, 96.0, 95.1, 89.1,
    90.5, 90.3, 88.4, 84.0, 85.1, 81.9, 82.6, 84.9, 81.3, 71.9, 74.3, 76.4,
    63.3, 71.7, 77.0, 65.2, 47.7, 68.6, 65.0,
])
_CIE_D_S1 = np.array([
    38.5, 35.0, 43.4, 46.3, 43.9, 37.1, 36.7, 35.9, 32.6, 27.9, 24.3, 20.1,
    16.2, 13.2, 8.6, 6.1, 4.2, 1.9, 0.0, -1.6, -3.5, -3.5, -5.8, -7.2, -8.6,
    -9.5, -10.9, -10.7, -12.0, -14.0, -13.6, -12.0, -13.3, -12.9, -10.6,
    -11.6, -12.2, -10.2, -7.8, -11.2, -10.4,
])
_CIE_D_S2 = np.array([
    3.0, 1.2, -1.1, -0.5, -0.7, -1.2, -2.6, -2.9, -2.8, -2.6, -2.6, -1.8,
    -1.5, -1.3, -1.2, -1.0, -0.5, -0.3, 0.0, 0.2, 0.5, 2.1, 3.2, 4.1, 4.7,
    5.1, 6.7, 7.3, 8.6, 9.8, 10.2, 8.3, 9.6, 8.5, 7.0, 7.6, 8.0, 6.7, 5.2,
    7.4, 6.8,
])

# Physical constants
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 2.99792458e8  # m/s
K_BOLTZMANN = 1.380649e-23  # J/K


class Illuminant:
    """Spectral power distribution of a light source.

    Attributes:
        name: Illuminant identifier (e.g., 'D65', 'A', 'blackbody_5500K').
        wavelengths: Wavelength array in micrometers.
        spectrum: Relative spectral power distribution (arbitrary units).
    """

    def __init__(self, name: str, wavelengths: np.ndarray, spectrum: np.ndarray) -> None:
        """Initialise an illuminant from wavelength and spectral arrays.

        Args:
            name: Illuminant name (e.g., 'D65', 'A', 'blackbody_5500K').
            wavelengths: Wavelength array in micrometers.
            spectrum: Relative spectral power distribution (arbitrary units).

        Raises:
            ValueError: If array lengths mismatch or spectrum contains negatives.
        """
        if len(wavelengths) != len(spectrum):
            raise ValueError(
                f"Wavelength ({len(wavelengths)}) and spectrum ({len(spectrum)}) "
                "arrays must have the same length."
            )
        if np.any(spectrum < 0):
            raise ValueError("Spectrum values must be non-negative.")

        self.name = name
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.spectrum = np.asarray(spectrum, dtype=np.float64)
        logger.debug("Created illuminant '%s' with %d points.", name, len(wavelengths))

    def normalized(self) -> Illuminant:
        """Return a copy normalised so that the peak value equals 1.

        Returns:
            New Illuminant with peak-normalised spectrum.
        """
        peak = np.max(self.spectrum)
        if peak <= 0:
            logger.warning("Illuminant '%s' has zero peak; returning copy.", self.name)
            return Illuminant(self.name + "_norm", self.wavelengths.copy(), self.spectrum.copy())
        return Illuminant(
            self.name + "_norm",
            self.wavelengths.copy(),
            self.spectrum / peak,
        )

    def interpolate(self, wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate the spectrum onto a new wavelength grid.

        Uses linear interpolation with zero-fill outside the original range.

        Args:
            wavelengths: Target wavelength grid in micrometers.

        Returns:
            Interpolated spectrum values on the target grid.
        """
        f = interp1d(
            self.wavelengths,
            self.spectrum,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        return np.asarray(f(wavelengths), dtype=np.float64)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def blackbody(temperature: float, wavelengths: np.ndarray) -> Illuminant:
        """Create a blackbody illuminant using Planck's law.

        L(lam, T) = (2 h c^2 / lam^5) / (exp(h c / lam k T) - 1)

        The returned spectrum is normalised to a relative peak of 1.

        Args:
            temperature: Colour temperature in Kelvin.
            wavelengths: Wavelength array in micrometers.

        Returns:
            Illuminant with blackbody spectral shape.
        """
        lam_m = np.asarray(wavelengths, dtype=np.float64) * 1e-6  # um -> m
        numerator = 2.0 * H_PLANCK * C_LIGHT**2 / (lam_m**5)
        exponent = H_PLANCK * C_LIGHT / (lam_m * K_BOLTZMANN * temperature)
        # Clip exponent to avoid overflow
        exponent = np.clip(exponent, 0.0, 500.0)
        spectrum = numerator / (np.exp(exponent) - 1.0)
        # Normalise to relative units (peak = 1)
        peak = np.max(spectrum)
        if peak > 0:
            spectrum = spectrum / peak
        name = f"blackbody_{temperature:.0f}K"
        logger.debug("Created %s illuminant.", name)
        return Illuminant(name, np.asarray(wavelengths, dtype=np.float64), spectrum)

    @staticmethod
    def cie_d_illuminant(temperature: float, wavelengths: np.ndarray) -> Illuminant:
        """CIE daylight illuminant D_T (CIE 015 method).

        S(lam) = S0(lam) + M1 * S1(lam) + M2 * S2(lam), with the tabulated CIE
        basis functions (10 nm steps, 380-780 nm, linearly interpolated and
        held constant outside that range) and M1, M2 from the daylight-locus
        chromaticity x_D(T). Normalized to a peak of 1.

        Args:
            temperature: Correlated colour temperature in Kelvin (4000-25000).
            wavelengths: Wavelength array in micrometers.

        Returns:
            Daylight illuminant.
        """
        if not 4000.0 <= temperature <= 25000.0:
            raise ValueError(
                f"CIE daylight is defined for 4000-25000 K, got {temperature} K"
            )
        wl = np.asarray(wavelengths, dtype=np.float64)
        if temperature <= 7000:
            xd = (0.244063 + 0.09911e3 / temperature + 2.9678e6 / temperature**2
                  - 4.6070e9 / temperature**3)
        else:
            xd = (0.237040 + 0.24748e3 / temperature + 1.9018e6 / temperature**2
                  - 2.0064e9 / temperature**3)
        yd = -3.0 * xd**2 + 2.870 * xd - 0.275
        m = 0.0241 + 0.2562 * xd - 0.7341 * yd
        m1 = (-1.3515 - 1.7703 * xd + 5.9114 * yd) / m
        m2 = (0.0300 - 31.4424 * xd + 30.0717 * yd) / m

        wl_nm = wl * 1000.0
        s0 = np.interp(wl_nm, _CIE_D_WL_NM, _CIE_D_S0)
        s1 = np.interp(wl_nm, _CIE_D_WL_NM, _CIE_D_S1)
        s2 = np.interp(wl_nm, _CIE_D_WL_NM, _CIE_D_S2)
        spectrum = np.maximum(s0 + m1 * s1 + m2 * s2, 0.0)
        peak = np.max(spectrum)
        if peak > 0:
            spectrum = spectrum / peak

        name = f"CIE_D{temperature:.0f}"
        logger.debug("Created %s illuminant.", name)
        return Illuminant(name, wl, spectrum)

    @staticmethod
    def cie_a(wavelengths: np.ndarray) -> Illuminant:
        """CIE Standard Illuminant A (incandescent, 2856 K).

        This is essentially a Planckian radiator at 2856 K.

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Illuminant A.
        """
        ill = Illuminant.blackbody(2856.0, wavelengths)
        ill.name = "CIE_A"
        return ill

    @staticmethod
    def cie_d65(wavelengths: np.ndarray) -> Illuminant:
        """CIE Standard Illuminant D65 (average daylight, ~6504 K).

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Illuminant D65.
        """
        ill = Illuminant.cie_d_illuminant(6504.0, wavelengths)
        ill.name = "CIE_D65"
        return ill

    @staticmethod
    def cie_f11(wavelengths: np.ndarray) -> Illuminant:
        """CIE Fluorescent Illuminant F11 (narrow-band triband).

        Modelled as the sum of three Gaussians at approximately 430 nm,
        545 nm, and 610 nm, plus a weak continuum.

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Illuminant F11.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        wl_nm = wl * 1000.0
        # Three emission peaks
        peak1 = 1.0 * np.exp(-0.5 * ((wl_nm - 430.0) / 10.0) ** 2)
        peak2 = 2.0 * np.exp(-0.5 * ((wl_nm - 545.0) / 12.0) ** 2)
        peak3 = 1.5 * np.exp(-0.5 * ((wl_nm - 610.0) / 10.0) ** 2)
        # Weak continuum
        continuum = 0.1 * np.exp(-0.5 * ((wl_nm - 550.0) / 100.0) ** 2)
        spectrum = peak1 + peak2 + peak3 + continuum
        peak = np.max(spectrum)
        if peak > 0:
            spectrum = spectrum / peak
        return Illuminant("CIE_F11", wl, spectrum)

    @staticmethod
    def led_white(cct: float, wavelengths: np.ndarray) -> Illuminant:
        """White LED spectrum: blue pump plus phosphor emission.

        Models a phosphor-converted white LED with a narrow blue peak at
        ~450 nm and a broad phosphor emission centred between 550-600 nm
        depending on the correlated colour temperature.

        Args:
            cct: Correlated colour temperature in Kelvin.
            wavelengths: Wavelength array in micrometers.

        Returns:
            LED white illuminant.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        wl_nm = wl * 1000.0

        # Blue pump peak ~450 nm
        blue = 0.8 * np.exp(-0.5 * ((wl_nm - 450.0) / 12.0) ** 2)

        # Phosphor emission centre shifts with CCT
        # Lower CCT -> warmer -> phosphor centre at longer wavelength
        phosphor_center = 600.0 - (cct - 2700.0) / (6500.0 - 2700.0) * 50.0
        phosphor_center = np.clip(phosphor_center, 540.0, 610.0)
        phosphor = 1.0 * np.exp(-0.5 * ((wl_nm - phosphor_center) / 50.0) ** 2)

        # Blue-to-phosphor ratio: higher CCT -> relatively more blue
        blue_ratio = 0.3 + 0.5 * (cct - 2700.0) / (6500.0 - 2700.0)
        blue_ratio = np.clip(blue_ratio, 0.2, 0.9)
        spectrum = blue_ratio * blue + (1.0 - blue_ratio) * phosphor

        peak = np.max(spectrum)
        if peak > 0:
            spectrum = spectrum / peak

        name = f"LED_white_{cct:.0f}K"
        logger.debug("Created %s illuminant.", name)
        return Illuminant(name, wl, spectrum)

    @staticmethod
    def monochromatic(
        center_wl: float, wavelengths: np.ndarray, fwhm: float = 0.01
    ) -> Illuminant:
        """Narrow-band (laser/LED) source.

        Modelled as a Gaussian with specified centre wavelength and FWHM.

        Args:
            center_wl: Centre wavelength in micrometers.
            wavelengths: Wavelength array in micrometers.
            fwhm: Full width at half maximum in micrometers.

        Returns:
            Narrow-band illuminant.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        spectrum = np.exp(-0.5 * ((wl - center_wl) / sigma) ** 2)
        name = f"mono_{center_wl:.3f}um"
        return Illuminant(name, wl, spectrum)

    @staticmethod
    def flat(wavelengths: np.ndarray) -> Illuminant:
        """Flat (equal-energy) spectrum. CIE Illuminant E.

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Equal-energy illuminant.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        return Illuminant("E", wl, np.ones_like(wl))
