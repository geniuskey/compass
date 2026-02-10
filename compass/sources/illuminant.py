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
        """CIE daylight illuminant approximation.

        Uses CIE daylight model: S(lam) = S0(lam) + M1 * S1(lam) + M2 * S2(lam).
        Approximated here as a blackbody at the given temperature with a daylight
        correction that boosts the blue end and applies a UV roll-off.

        Args:
            temperature: Correlated colour temperature in Kelvin.
            wavelengths: Wavelength array in micrometers.

        Returns:
            Illuminant with CIE-daylight-like spectral shape.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        # Start from blackbody
        bb = Illuminant.blackbody(temperature, wl)
        spectrum = bb.spectrum.copy()

        # Daylight correction: slight blue boost, UV roll-off
        xd = 0.0 if temperature <= 4000 else 1.0
        if temperature <= 7000:
            xd = 0.244063 + 0.09911e3 / temperature + 2.9678e6 / temperature**2 - 4.6070e9 / temperature**3
        else:
            xd = 0.237040 + 0.24748e3 / temperature + 1.9018e6 / temperature**2 - 2.0064e9 / temperature**3
        yd = -3.0 * xd**2 + 2.870 * xd - 0.275

        # M coefficients
        m1 = (-1.3515 - 1.7703 * xd + 5.9114 * yd) / (0.0241 + 0.2562 * xd - 0.7341 * yd)
        m2 = (0.0300 - 31.4424 * xd + 30.0717 * yd) / (0.0241 + 0.2562 * xd - 0.7341 * yd)

        # Approximate S0, S1, S2 basis functions as smooth curves
        wl_nm = wl * 1000.0  # convert to nm for the model
        s0 = np.exp(-0.5 * ((wl_nm - 560.0) / 120.0) ** 2) * 100.0 + 40.0
        s1 = np.exp(-0.5 * ((wl_nm - 450.0) / 50.0) ** 2) * 20.0 - np.exp(
            -0.5 * ((wl_nm - 600.0) / 60.0) ** 2
        ) * 15.0
        s2 = np.exp(-0.5 * ((wl_nm - 430.0) / 30.0) ** 2) * 10.0 - np.exp(
            -0.5 * ((wl_nm - 560.0) / 40.0) ** 2
        ) * 8.0

        daylight_spd = s0 + m1 * s1 + m2 * s2
        daylight_spd = np.maximum(daylight_spd, 0.0)
        peak = np.max(daylight_spd)
        if peak > 0:
            daylight_spd = daylight_spd / peak

        # Blend blackbody and daylight model (favour daylight model)
        spectrum = 0.3 * spectrum + 0.7 * daylight_spd
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
