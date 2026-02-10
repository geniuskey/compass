"""Camera module optics: lens transmittance and IR cut filter.

Models the optical elements between the scene and the sensor, including
multi-element lens transmittance, IR cut-off filters, and arbitrary
additional filter stacks. Provides factory presets for smartphone and
industrial camera modules.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class IRCutFilter:
    """Infrared cut-off filter model.

    Uses a sigmoid (logistic) function to model the sharp transition
    between the visible passband and the IR stopband.

    Attributes:
        cutoff_wl: Cut-off wavelength in um (where T = 50%).
        transition_width: Transition region width in um.
        passband_t: Maximum transmittance in passband.
        stopband_t: Minimum transmittance in stopband.
    """

    def __init__(
        self,
        cutoff_wl: float = 0.65,
        transition_width: float = 0.03,
        passband_t: float = 0.95,
        stopband_t: float = 0.01,
    ) -> None:
        """Initialise IR cut filter parameters.

        Args:
            cutoff_wl: Cut-off wavelength in um (where T = 50%).
            transition_width: Transition region width in um.
            passband_t: Maximum transmittance in passband.
            stopband_t: Minimum transmittance in stopband.
        """
        self.cutoff_wl = cutoff_wl
        self.transition_width = transition_width
        self.passband_t = passband_t
        self.stopband_t = stopband_t
        logger.debug(
            "IRCutFilter: cutoff=%.3f um, width=%.3f um, Tpass=%.3f, Tstop=%.3f",
            cutoff_wl, transition_width, passband_t, stopband_t,
        )

    def transmittance(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute transmittance spectrum.

        Uses sigmoid: T = stopband + (passband - stopband) * sigmoid(-(lam - lam_c) / w)

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Transmittance array.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        # Steepness parameter: scale so transition_width corresponds to ~10-90% range
        steepness = 4.0 / self.transition_width  # ~4 sigmoid units per transition_width
        sigmoid = 1.0 / (1.0 + np.exp(steepness * (wl - self.cutoff_wl)))
        t = self.stopband_t + (self.passband_t - self.stopband_t) * sigmoid
        return np.asarray(t, dtype=np.float64)


class LensTransmittance:
    """Camera module lens transmittance model.

    Models transmission losses from glass absorption and surface
    reflections (with or without anti-reflection coatings).

    Attributes:
        num_elements: Number of lens elements.
        ar_coating: Whether AR coating is applied.
    """

    def __init__(self, num_elements: int = 6, ar_coating: bool = True) -> None:
        """Initialise lens transmittance model.

        Args:
            num_elements: Number of lens elements (more elements = more loss).
            ar_coating: Whether anti-reflection coating is applied.
        """
        self.num_elements = num_elements
        self.ar_coating = ar_coating
        logger.debug(
            "LensTransmittance: %d elements, AR=%s", num_elements, ar_coating,
        )

    def transmittance(self, wavelengths: np.ndarray) -> np.ndarray:
        """Compute lens transmittance.

        Model: T = T_glass^N * T_AR^(2N) where N = num_elements.
        T_glass is approximately 0.998 per element at peak, with slight
        wavelength dependence. T_AR per surface is approximately 0.995
        with coating, 0.96 without.

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Transmittance array.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        n = self.num_elements

        # Glass absorption: slight increase at UV/blue end
        # T_glass per element ~0.998 at 550nm, drops at shorter wavelengths
        wl_nm = wl * 1000.0
        t_glass_per_element = 0.998 - 0.001 * np.exp(-((wl_nm - 400.0) / 50.0) ** 2)
        t_glass = t_glass_per_element**n

        # Surface reflection loss: 2 surfaces per element
        if self.ar_coating:
            # AR-coated: ~0.5% loss per surface, slight wavelength dependence
            t_surface = 0.995 - 0.002 * np.exp(-0.5 * ((wl_nm - 550.0) / 150.0) ** 2)
        else:
            # Uncoated: ~4% Fresnel reflection per surface (n~1.5 glass)
            t_surface = np.full_like(wl, 0.96)

        t_surfaces = t_surface ** (2 * n)
        return np.asarray(t_glass * t_surfaces, dtype=np.float64)


class ModuleOptics:
    """Combined camera module optics chain.

    Combines lens, IR cut filter, and any additional filters into a
    single transmittance model.

    Attributes:
        lens: Lens transmittance model.
        ir_filter: IR cut filter model.
        additional_filters: List of (name, wavelengths, transmittance) tuples.
    """

    def __init__(
        self,
        lens: LensTransmittance | None = None,
        ir_filter: IRCutFilter | None = None,
        additional_filters: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Initialise combined optics module.

        Args:
            lens: Lens transmittance model (default: 6-element with AR).
            ir_filter: IR cut filter (default: 650 nm cutoff).
            additional_filters: List of (name, wavelengths, transmittance)
                for custom filters.
        """
        self.lens = lens if lens is not None else LensTransmittance()
        self.ir_filter = ir_filter if ir_filter is not None else IRCutFilter()
        self.additional_filters = additional_filters if additional_filters is not None else []
        logger.debug(
            "ModuleOptics: %d additional filters.", len(self.additional_filters),
        )

    def total_transmittance(self, wavelengths: np.ndarray) -> np.ndarray:
        """Combined transmittance of all optical elements.

        T_total = T_lens * T_IR * T_additional1 * T_additional2 * ...

        Args:
            wavelengths: Wavelength array in micrometers.

        Returns:
            Combined transmittance array.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        t_total = self.lens.transmittance(wl) * self.ir_filter.transmittance(wl)

        for name, filt_wl, filt_t in self.additional_filters:
            f = interp1d(
                filt_wl, filt_t,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            t_interp = np.asarray(f(wl), dtype=np.float64)
            t_total = t_total * t_interp
            logger.debug("Applied additional filter '%s'.", name)

        return np.asarray(t_total, dtype=np.float64)

    @staticmethod
    def smartphone_module(wavelengths: np.ndarray | None = None) -> ModuleOptics:
        """Typical smartphone camera module (6P lens, IR cut at 650 nm).

        Args:
            wavelengths: Optional; not used for construction but kept for
                API consistency.

        Returns:
            ModuleOptics configured as a smartphone module.
        """
        return ModuleOptics(
            lens=LensTransmittance(num_elements=6, ar_coating=True),
            ir_filter=IRCutFilter(cutoff_wl=0.65, transition_width=0.03),
        )

    @staticmethod
    def industrial_module(wavelengths: np.ndarray | None = None) -> ModuleOptics:
        """Industrial camera module (fewer elements, wider IR pass).

        Args:
            wavelengths: Optional; not used for construction but kept for
                API consistency.

        Returns:
            ModuleOptics configured as an industrial module.
        """
        return ModuleOptics(
            lens=LensTransmittance(num_elements=4, ar_coating=True),
            ir_filter=IRCutFilter(cutoff_wl=0.70, transition_width=0.05),
        )

    @staticmethod
    def no_optics() -> ModuleOptics:
        """No module optics (T = 1 everywhere). For bare sensor simulation.

        Returns:
            ModuleOptics with unity transmittance.
        """
        return ModuleOptics(
            lens=LensTransmittance(num_elements=0, ar_coating=True),
            ir_filter=IRCutFilter(
                cutoff_wl=10.0,  # Far beyond visible range
                transition_width=0.03,
                passband_t=1.0,
                stopband_t=1.0,
            ),
        )
