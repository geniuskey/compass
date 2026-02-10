"""Material property database for COMPASS.

Supports three material definition types:
1. Constant: Fixed n, k values (e.g., air n=1.0, k=0.0)
2. Tabulated: CSV file with wavelength-dependent n, k (cubic spline interpolation)
3. Analytical: Cauchy, Sellmeier, or Drude-Lorentz models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

logger = logging.getLogger(__name__)

# Default materials directory (relative to package root)
_MATERIALS_DIR = Path(__file__).parent.parent.parent / "materials"


@dataclass
class MaterialData:
    """Container for a single material's optical properties."""

    name: str
    mat_type: Literal["constant", "tabulated", "cauchy", "sellmeier"]
    # For constant
    n_const: float = 1.0
    k_const: float = 0.0
    # For tabulated
    wavelengths: np.ndarray | None = None
    n_data: np.ndarray | None = None
    k_data: np.ndarray | None = None
    _n_interp: object | None = field(default=None, repr=False)
    _k_interp: object | None = field(default=None, repr=False)
    interpolation: str = "cubic_spline"
    # For Cauchy
    cauchy_A: float = 1.0
    cauchy_B: float = 0.0
    cauchy_C: float = 0.0
    # For Sellmeier
    sellmeier_B: list[float] | None = None
    sellmeier_C: list[float] | None = None

    def _build_interpolators(self) -> None:
        """Build interpolation functions from tabulated data."""
        if self.wavelengths is None:
            return
        if self.interpolation == "cubic_spline" and len(self.wavelengths) >= 4:
            self._n_interp = CubicSpline(self.wavelengths, self.n_data, extrapolate=True)
            self._k_interp = CubicSpline(self.wavelengths, self.k_data, extrapolate=True)
        else:
            self._n_interp = interp1d(
                self.wavelengths, self.n_data,
                kind="linear", fill_value="extrapolate",
            )
            self._k_interp = interp1d(
                self.wavelengths, self.k_data,
                kind="linear", fill_value="extrapolate",
            )

    def get_nk(self, wavelength: float) -> tuple[float, float]:
        """Get refractive index (n, k) at a given wavelength in um."""
        if self.mat_type == "constant":
            return self.n_const, self.k_const

        elif self.mat_type == "tabulated":
            if self._n_interp is None:
                self._build_interpolators()
            assert self.wavelengths is not None
            wl_clamped = np.clip(
                wavelength,
                self.wavelengths.min(),
                self.wavelengths.max(),
            )
            if wavelength != wl_clamped:
                logger.warning(
                    f"Material '{self.name}': wavelength {wavelength:.4f} um "
                    f"outside data range [{self.wavelengths.min():.4f}, "
                    f"{self.wavelengths.max():.4f}], clamping."
                )
            assert self._n_interp is not None and callable(self._n_interp)
            assert self._k_interp is not None and callable(self._k_interp)
            n = float(self._n_interp(wl_clamped))
            k = float(self._k_interp(wl_clamped))
            return n, max(k, 0.0)

        elif self.mat_type == "cauchy":
            lam2 = wavelength ** 2
            n = self.cauchy_A + self.cauchy_B / lam2 + self.cauchy_C / (lam2 ** 2)
            return n, 0.0

        elif self.mat_type == "sellmeier":
            lam2 = wavelength ** 2
            n2 = 1.0
            assert self.sellmeier_B is not None
            assert self.sellmeier_C is not None
            for B, C in zip(self.sellmeier_B, self.sellmeier_C):
                n2 += B * lam2 / (lam2 - C)
            return np.sqrt(max(n2, 1.0)), 0.0

        raise ValueError(f"Unknown material type: {self.mat_type}")

    def get_epsilon(self, wavelength: float) -> complex:
        """Get complex permittivity epsilon = (n + ik)^2."""
        n, k = self.get_nk(wavelength)
        return (n + 1j * k) ** 2


class MaterialDB:
    """Central material property database."""

    def __init__(self, db_path: str | None = None):
        self._materials: dict[str, MaterialData] = {}
        self._db_path = Path(db_path) if db_path else _MATERIALS_DIR
        self._load_builtin()

    def _load_builtin(self) -> None:
        """Load built-in material definitions."""
        # Air
        self.register_constant("air", n=1.0, k=0.0)

        # Polymer microlens
        self.register_cauchy("polymer_n1p56", A=1.56, B=0.004, C=0.0)

        # SiO2 (approximate Sellmeier)
        self.register_sellmeier(
            "sio2",
            B=[0.6961663, 0.4079426, 0.8974794],
            C=[0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2],
        )

        # HfO2 (approximate Cauchy)
        self.register_cauchy("hfo2", A=1.90, B=0.02, C=0.0)

        # Si3N4 (approximate Sellmeier)
        self.register_sellmeier(
            "si3n4",
            B=[2.8939, 0.0],
            C=[0.13967 ** 2, 1.0],
        )

        # TiO2 (approximate Cauchy for anatase)
        self.register_cauchy("tio2", A=2.27, B=0.05, C=0.0)

        # Load tabulated materials from CSV files if available
        self._load_csv_materials()

    def _load_csv_materials(self) -> None:
        """Load tabulated materials from CSV files."""
        csv_mapping = {
            "silicon": ["silicon_green2008.csv", "silicon_palik.csv"],
            "tungsten": ["tungsten.csv"],
            "cf_red": ["color_filter_red.csv"],
            "cf_green": ["color_filter_green.csv"],
            "cf_blue": ["color_filter_blue.csv"],
        }

        for name, filenames in csv_mapping.items():
            for filename in filenames:
                csv_path = self._db_path / filename
                if csv_path.exists():
                    self.load_csv(name, str(csv_path))
                    break
            else:
                # If CSV not found, register fallback
                if name == "silicon":
                    self._register_silicon_fallback()
                elif name == "tungsten":
                    self._register_tungsten_fallback()
                elif name.startswith("cf_"):
                    self._register_color_filter_fallback(name)

    def _register_silicon_fallback(self) -> None:
        """Register silicon with approximate tabulated data (Green 2008)."""
        # Subset of Green 2008 data for visible range
        wl = np.array([
            0.350, 0.360, 0.370, 0.380, 0.390, 0.400, 0.410, 0.420, 0.430,
            0.440, 0.450, 0.460, 0.470, 0.480, 0.490, 0.500, 0.510, 0.520,
            0.530, 0.540, 0.550, 0.560, 0.570, 0.580, 0.590, 0.600, 0.620,
            0.640, 0.660, 0.680, 0.700, 0.720, 0.740, 0.760, 0.780, 0.800,
            0.850, 0.900, 0.950, 1.000, 1.050, 1.100,
        ])
        n = np.array([
            5.565, 5.827, 6.044, 5.976, 5.587, 5.381, 5.253, 5.103, 4.930,
            4.774, 4.641, 4.528, 4.432, 4.350, 4.279, 4.215, 4.159, 4.109,
            4.064, 4.024, 4.082, 3.979, 3.948, 3.921, 3.897, 3.876, 3.840,
            3.810, 3.785, 3.764, 3.746, 3.731, 3.718, 3.707, 3.697, 3.688,
            3.670, 3.655, 3.642, 3.632, 3.623, 3.616,
        ])
        k = np.array([
            3.004, 2.989, 2.823, 2.459, 2.025, 0.340, 0.296, 0.267, 0.244,
            0.224, 0.206, 0.189, 0.173, 0.158, 0.143, 0.130, 0.118, 0.107,
            0.098, 0.089, 0.028, 0.075, 0.069, 0.063, 0.058, 0.054, 0.047,
            0.041, 0.037, 0.033, 0.030, 0.027, 0.024, 0.022, 0.020, 0.018,
            0.014, 0.011, 0.008, 0.006, 0.004, 0.003,
        ])
        mat = MaterialData(
            name="silicon",
            mat_type="tabulated",
            wavelengths=wl, n_data=n, k_data=k,
        )
        mat._build_interpolators()
        self._materials["silicon"] = mat

    def _register_tungsten_fallback(self) -> None:
        """Register tungsten with approximate tabulated data."""
        wl = np.array([0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.78])
        n = np.array([3.39, 3.46, 3.55, 3.61, 3.65, 3.68, 3.70, 3.72, 3.74])
        k = np.array([2.66, 2.72, 2.86, 2.98, 3.08, 3.17, 3.25, 3.33, 3.44])
        mat = MaterialData(
            name="tungsten",
            mat_type="tabulated",
            wavelengths=wl, n_data=n, k_data=k,
        )
        mat._build_interpolators()
        self._materials["tungsten"] = mat

    def _register_color_filter_fallback(self, name: str) -> None:
        """Register color filter with approximate Lorentzian absorption profiles."""
        wl = np.linspace(0.38, 0.78, 41)

        # Generic absorption profiles for R, G, B filters
        if name == "cf_red":
            peak_wl, width = 0.62, 0.06
            n_base, k_max = 1.55, 0.15
        elif name == "cf_green":
            peak_wl, width = 0.53, 0.05
            n_base, k_max = 1.55, 0.12
        elif name == "cf_blue":
            peak_wl, width = 0.45, 0.05
            n_base, k_max = 1.55, 0.18
        else:
            n_base, k_max, peak_wl, width = 1.55, 0.1, 0.55, 0.05

        # Absorption: high k outside passband, low k in passband
        k = k_max * (1.0 - np.exp(-((wl - peak_wl) / width) ** 2))
        n = np.full_like(wl, n_base)

        mat = MaterialData(
            name=name,
            mat_type="tabulated",
            wavelengths=wl, n_data=n, k_data=k,
        )
        mat._build_interpolators()
        self._materials[name] = mat

    def register_constant(self, name: str, n: float, k: float = 0.0) -> None:
        """Register a material with constant n, k."""
        self._materials[name] = MaterialData(
            name=name, mat_type="constant", n_const=n, k_const=k,
        )

    def register_cauchy(self, name: str, A: float, B: float = 0.0, C: float = 0.0) -> None:
        """Register a material with Cauchy dispersion model: n(λ) = A + B/λ² + C/λ⁴."""
        self._materials[name] = MaterialData(
            name=name, mat_type="cauchy",
            cauchy_A=A, cauchy_B=B, cauchy_C=C,
        )

    def register_sellmeier(self, name: str, B: list[float], C: list[float]) -> None:
        """Register a material with Sellmeier dispersion model."""
        self._materials[name] = MaterialData(
            name=name, mat_type="sellmeier",
            sellmeier_B=B, sellmeier_C=C,
        )

    def load_csv(self, name: str, filepath: str, interpolation: str = "cubic_spline") -> None:
        """Load tabulated material from CSV file.

        CSV format: wavelength(um), n, k
        Lines starting with # are comments.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Material CSV not found: {filepath}")

        data = np.loadtxt(filepath, delimiter=",", comments="#")
        if data.shape[1] < 3:
            raise ValueError(f"CSV must have 3 columns (wavelength, n, k), got {data.shape[1]}")

        # Sort by wavelength
        sort_idx = np.argsort(data[:, 0])
        data = data[sort_idx]

        mat = MaterialData(
            name=name,
            mat_type="tabulated",
            wavelengths=data[:, 0],
            n_data=data[:, 1],
            k_data=data[:, 2],
            interpolation=interpolation,
        )
        mat._build_interpolators()
        self._materials[name] = mat

    def get_nk(self, name: str, wavelength: float) -> tuple[float, float]:
        """Get (n, k) for a material at a given wavelength (um)."""
        if name not in self._materials:
            raise KeyError(f"Unknown material: '{name}'. Available: {list(self._materials.keys())}")
        return self._materials[name].get_nk(wavelength)

    def get_epsilon(self, name: str, wavelength: float) -> complex:
        """Get complex permittivity at given wavelength. ε = (n + ik)²."""
        if name not in self._materials:
            raise KeyError(f"Unknown material: '{name}'. Available: {list(self._materials.keys())}")
        return self._materials[name].get_epsilon(wavelength)

    def get_epsilon_spectrum(self, name: str, wavelengths: np.ndarray) -> np.ndarray:
        """Get complex permittivity over wavelength array."""
        return np.array([self.get_epsilon(name, wl) for wl in wavelengths])

    def list_materials(self) -> list[str]:
        """List all available material names."""
        return sorted(self._materials.keys())

    def has_material(self, name: str) -> bool:
        """Check if a material exists in the database."""
        return name in self._materials
