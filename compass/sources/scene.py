"""Scene spectral reflectance models.

Provides spectral reflectance data for scene elements used in radiometric
signal chain calculations. Includes flat targets, Macbeth ColorChecker
patch approximations, and simple colour targets.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class SceneReflectance:
    """Spectral reflectance of a scene element.

    Attributes:
        name: Patch or object name.
        wavelengths: Wavelength array in micrometers.
        reflectance: Reflectance values in the range [0, 1].
    """

    def __init__(self, name: str, wavelengths: np.ndarray, reflectance: np.ndarray) -> None:
        """Initialise from wavelength and reflectance arrays.

        Args:
            name: Patch/object name.
            wavelengths: Wavelength array in micrometers.
            reflectance: Reflectance values in the range [0, 1].

        Raises:
            ValueError: If array lengths mismatch or reflectance is out of range.
        """
        if len(wavelengths) != len(reflectance):
            raise ValueError(
                f"Wavelength ({len(wavelengths)}) and reflectance ({len(reflectance)}) "
                "arrays must have the same length."
            )
        if np.any(reflectance < 0) or np.any(reflectance > 1):
            raise ValueError("Reflectance values must be in [0, 1].")

        self.name = name
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.reflectance = np.asarray(reflectance, dtype=np.float64)
        logger.debug("Created scene reflectance '%s' with %d points.", name, len(wavelengths))

    def interpolate(self, wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate reflectance onto a new wavelength grid.

        Uses linear interpolation. Values outside the original range are
        clamped to the nearest boundary value.

        Args:
            wavelengths: Target wavelength grid in micrometers.

        Returns:
            Interpolated reflectance values.
        """
        f = interp1d(
            self.wavelengths,
            self.reflectance,
            kind="linear",
            bounds_error=False,
            fill_value=(self.reflectance[0], self.reflectance[-1]),
        )
        result = np.asarray(f(wavelengths), dtype=np.float64)
        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def flat(value: float, wavelengths: np.ndarray) -> SceneReflectance:
        """Uniform reflectance across all wavelengths.

        Args:
            value: Constant reflectance (e.g., 0.18 for an 18% grey card).
            wavelengths: Wavelength array in micrometers.

        Returns:
            Flat scene reflectance.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        return SceneReflectance(f"flat_{value:.2f}", wl, np.full_like(wl, value))

    @staticmethod
    def macbeth_patch(patch_id: int, wavelengths: np.ndarray) -> SceneReflectance:
        """Approximate Macbeth ColorChecker patch reflectance.

        Patches 0-23. Uses Gaussian mixture approximations for the spectral
        shape of each patch. Key patches include:
            0: Dark skin, 1: Light skin, 2: Blue sky, 3: Foliage,
            4: Blue flower, 5: Bluish green, 13: Blue, 14: Green,
            15: Red, 18: White, 19: Neutral 8, 20: Neutral 6.5,
            21: Neutral 5, 22: Neutral 3.5, 23: Black.

        Args:
            patch_id: Patch index (0-23).
            wavelengths: Wavelength array in micrometers.

        Returns:
            Approximate patch reflectance.

        Raises:
            ValueError: If patch_id is out of range.
        """
        if not 0 <= patch_id <= 23:
            raise ValueError(f"Patch ID must be 0-23, got {patch_id}.")

        wl = np.asarray(wavelengths, dtype=np.float64)
        wl_nm = wl * 1000.0

        # Gaussian mixture parameters: list of (amplitude, centre_nm, sigma_nm)
        # plus a baseline reflectance level for each patch.
        _patch_models: dict[int, tuple[float, list[tuple[float, float, float]]]] = {
            0: (0.08, [(0.15, 600.0, 50.0), (0.05, 500.0, 40.0)]),  # Dark skin
            1: (0.20, [(0.25, 590.0, 60.0), (0.10, 500.0, 50.0)]),  # Light skin
            2: (0.10, [(0.15, 480.0, 40.0), (0.05, 600.0, 30.0)]),  # Blue sky
            3: (0.05, [(0.12, 540.0, 30.0), (0.03, 430.0, 20.0)]),  # Foliage
            4: (0.12, [(0.18, 460.0, 35.0), (0.08, 600.0, 40.0)]),  # Blue flower
            5: (0.20, [(0.25, 500.0, 40.0), (0.10, 450.0, 30.0)]),  # Bluish green
            6: (0.15, [(0.30, 580.0, 40.0), (0.05, 500.0, 30.0)]),  # Orange
            7: (0.05, [(0.20, 450.0, 30.0), (0.03, 600.0, 20.0)]),  # Purplish blue
            8: (0.10, [(0.25, 600.0, 40.0), (0.05, 450.0, 25.0)]),  # Moderate red
            9: (0.03, [(0.07, 500.0, 30.0), (0.03, 600.0, 30.0)]),  # Purple
            10: (0.25, [(0.25, 550.0, 35.0), (0.05, 430.0, 20.0)]),  # Yellow green
            11: (0.25, [(0.30, 570.0, 40.0), (0.05, 500.0, 30.0)]),  # Orange yellow
            12: (0.02, [(0.15, 440.0, 25.0)]),  # Blue
            13: (0.05, [(0.20, 450.0, 30.0), (0.03, 550.0, 20.0)]),  # Blue (darker)
            14: (0.05, [(0.20, 530.0, 30.0), (0.03, 450.0, 20.0)]),  # Green
            15: (0.05, [(0.35, 620.0, 40.0)]),  # Red
            16: (0.10, [(0.25, 600.0, 40.0), (0.10, 450.0, 25.0)]),  # Magenta
            17: (0.08, [(0.20, 480.0, 30.0), (0.05, 550.0, 25.0)]),  # Cyan
            18: (0.90, []),  # White
            19: (0.59, []),  # Neutral 8
            20: (0.36, []),  # Neutral 6.5
            21: (0.20, []),  # Neutral 5
            22: (0.09, []),  # Neutral 3.5
            23: (0.03, []),  # Black
        }

        baseline, gaussians = _patch_models[patch_id]
        spectrum = np.full_like(wl_nm, baseline, dtype=np.float64)
        for amp, center, sigma in gaussians:
            spectrum += amp * np.exp(-0.5 * ((wl_nm - center) / sigma) ** 2)
        spectrum = np.clip(spectrum, 0.0, 1.0)

        _names = [
            "dark_skin", "light_skin", "blue_sky", "foliage",
            "blue_flower", "bluish_green", "orange", "purplish_blue",
            "moderate_red", "purple", "yellow_green", "orange_yellow",
            "blue_12", "blue_13", "green", "red",
            "magenta", "cyan", "white", "neutral8",
            "neutral6.5", "neutral5", "neutral3.5", "black",
        ]
        name = f"macbeth_{patch_id}_{_names[patch_id]}"
        logger.debug("Created Macbeth patch %d (%s).", patch_id, _names[patch_id])
        return SceneReflectance(name, wl, spectrum)

    @staticmethod
    def color_target(color: str, wavelengths: np.ndarray) -> SceneReflectance:
        """Simple colour targets.

        Supported colours: 'red', 'green', 'blue', 'cyan', 'magenta',
        'yellow', 'white', 'grey18', 'black'. Modelled as bandpass or
        broadband reflectance using Gaussian profiles.

        Args:
            color: Colour name string.
            wavelengths: Wavelength array in micrometers.

        Returns:
            Colour target reflectance.

        Raises:
            ValueError: If colour name is not recognised.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        wl_nm = wl * 1000.0

        # Definitions: (baseline, list of (amplitude, center_nm, sigma_nm))
        _targets: dict[str, tuple[float, list[tuple[float, float, float]]]] = {
            "red": (0.02, [(0.90, 640.0, 40.0)]),
            "green": (0.02, [(0.85, 530.0, 35.0)]),
            "blue": (0.02, [(0.85, 450.0, 30.0)]),
            "cyan": (0.02, [(0.80, 490.0, 40.0), (0.30, 530.0, 30.0)]),
            "magenta": (0.02, [(0.70, 440.0, 30.0), (0.70, 640.0, 40.0)]),
            "yellow": (0.02, [(0.85, 570.0, 40.0), (0.40, 620.0, 30.0)]),
            "white": (0.90, []),
            "grey18": (0.18, []),
            "black": (0.03, []),
        }

        color_lower = color.lower()
        if color_lower not in _targets:
            raise ValueError(
                f"Unknown colour target '{color}'. "
                f"Available: {sorted(_targets.keys())}"
            )

        baseline, gaussians = _targets[color_lower]
        spectrum = np.full_like(wl_nm, baseline, dtype=np.float64)
        for amp, center, sigma in gaussians:
            spectrum += amp * np.exp(-0.5 * ((wl_nm - center) / sigma) ** 2)
        spectrum = np.clip(spectrum, 0.0, 1.0)
        return SceneReflectance(f"target_{color_lower}", wl, spectrum)

    @staticmethod
    def from_array(name: str, wavelengths: np.ndarray, reflectance: np.ndarray) -> SceneReflectance:
        """Create from user-provided arrays.

        Args:
            name: Object name.
            wavelengths: Wavelength array in micrometers.
            reflectance: Reflectance values in [0, 1].

        Returns:
            SceneReflectance instance.
        """
        return SceneReflectance(name, wavelengths, reflectance)
