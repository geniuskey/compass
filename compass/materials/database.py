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
    # Cache for n,k lookups keyed by wavelength (rounded to avoid float issues)
    _nk_cache: dict[float, tuple[float, float]] = field(default_factory=dict, repr=False)

    def _build_interpolators(self) -> None:
        """Build interpolation functions from tabulated data."""
        if self.wavelengths is None:
            return
        if self.interpolation == "cubic_spline" and len(self.wavelengths) >= 4:
            self._n_interp = CubicSpline(self.wavelengths, self.n_data, extrapolate=True)
            self._k_interp = CubicSpline(self.wavelengths, self.k_data, extrapolate=True)
        else:
            self._n_interp = interp1d(
                self.wavelengths,
                self.n_data,
                kind="linear",
                fill_value="extrapolate",
            )
            self._k_interp = interp1d(
                self.wavelengths,
                self.k_data,
                kind="linear",
                fill_value="extrapolate",
            )

    def get_nk(self, wavelength: float) -> tuple[float, float]:
        """Get refractive index (n, k) at a given wavelength in um.

        Results are cached to avoid redundant computation for repeated lookups
        at the same wavelength (common in multi-pixel RCWA sweeps).
        """
        if self.mat_type == "constant":
            return self.n_const, self.k_const

        # Check cache (round to 10 decimal places to avoid float precision issues)
        cache_key = round(wavelength, 10)
        cached = self._nk_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.mat_type == "tabulated":
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
            result = (n, max(k, 0.0))

        elif self.mat_type == "cauchy":
            lam2 = wavelength**2
            n = self.cauchy_A + self.cauchy_B / lam2 + self.cauchy_C / (lam2**2)
            result = (n, 0.0)

        elif self.mat_type == "sellmeier":
            lam2 = wavelength**2
            n2 = 1.0
            assert self.sellmeier_B is not None
            assert self.sellmeier_C is not None
            for B, C in zip(self.sellmeier_B, self.sellmeier_C):
                n2 += B * lam2 / (lam2 - C)
            result = (float(np.sqrt(max(n2, 1.0))), 0.0)

        else:
            raise ValueError(f"Unknown material type: {self.mat_type}")

        self._nk_cache[cache_key] = result
        return result

    def get_epsilon(self, wavelength: float) -> complex:
        """Get complex permittivity epsilon = (n + ik)^2."""
        n, k = self.get_nk(wavelength)
        return (n + 1j * k) ** 2


class MaterialDB:
    """Central material property database."""

    def __init__(self, db_path: str | None = None):
        self._materials: dict[str, MaterialData] = {}
        self._db_path = Path(db_path) if db_path else _MATERIALS_DIR
        # Cache for epsilon lookups: (material_name, rounded_wavelength) -> complex
        self._epsilon_cache: dict[tuple[str, float], complex] = {}
        self._load_builtin()

    def _load_builtin(self) -> None:
        """Load built-in material definitions."""
        # Air
        self.register_constant("air", n=1.0, k=0.0)

        # Polymer microlens (standard acrylate, n ~ 1.56 @ 550 nm)
        self.register_cauchy("polymer_n1p56", A=1.56, B=0.004, C=0.0)
        # High-refractive-index microlens (TiO2-doped polymer, n ~ 1.70 @ 550 nm).
        # Used in recent flagship sensors (e.g. Samsung ISOCELL HP9 "new material"
        # high-refractive microlens) to shorten focal length and improve light
        # collection at sub-um pitch.
        self.register_cauchy("polymer_hri_n1p70", A=1.70, B=0.008, C=0.0)
        # Very-high-RI microlens (n ~ 1.85, HRI inorganic-organic hybrid).
        self.register_cauchy("polymer_hri_n1p85", A=1.85, B=0.010, C=0.0)

        # SiO2 (approximate Sellmeier)
        self.register_sellmeier(
            "sio2",
            B=[0.6961663, 0.4079426, 0.8974794],
            C=[0.0684043**2, 0.1162414**2, 9.896161**2],
        )

        # HfO2 (approximate Cauchy)
        self.register_cauchy("hfo2", A=1.90, B=0.02, C=0.0)

        # Si3N4 (approximate Sellmeier)
        self.register_sellmeier(
            "si3n4",
            B=[2.8939, 0.0],
            C=[0.13967**2, 1.0],
        )

        # TiO2 (approximate Cauchy for anatase)
        self.register_cauchy("tio2", A=2.27, B=0.05, C=0.0)

        # Load tabulated materials from CSV files if available
        self._load_csv_materials()

        # Chemical-formula aliases so configs can use short names (e.g. DTI
        # liners spelled "al2o3"/"ta2o5") interchangeably with the descriptive
        # registered names.
        self._register_aliases()

    def _register_aliases(self) -> None:
        """Point common chemical-formula names at registered materials."""
        aliases = {
            "al2o3": "aluminum_oxide",
            "alumina": "aluminum_oxide",
            "ta2o5": "tantalum_pentoxide",
            "tantalum_oxide": "tantalum_pentoxide",
            "w": "tungsten",
            "si": "silicon",
            "mgf2": "magnesium_fluoride",
            "ito": "indium_tin_oxide",
            "sion": "silicon_oxynitride",
        }
        for alias, target in aliases.items():
            if target in self._materials and alias not in self._materials:
                self._materials[alias] = self._materials[target]

    def _load_csv_materials(self) -> None:
        """Load tabulated materials from CSV files."""
        # Original materials (top-level CSV files)
        csv_mapping: dict[str, list[str]] = {
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

        # Extended material library organized by category
        self._load_extended_csv_materials()

    def _load_extended_csv_materials(self) -> None:
        """Load extended material library from categorized subdirectories.

        Materials are organized under metals/, dielectrics/, polymers/,
        and semiconductors/ subdirectories within the materials directory.
        """
        # Metals (interconnects, metal grids, plasmonic structures)
        metals_mapping: dict[str, str] = {
            "aluminum": "metals/aluminum.csv",
            "gold": "metals/gold.csv",
            "silver": "metals/silver.csv",
            "copper": "metals/copper.csv",
            "titanium": "metals/titanium.csv",
            "titanium_nitride": "metals/titanium_nitride.csv",
        }

        # Dielectrics (ARC, passivation, transparent conductors)
        dielectrics_mapping: dict[str, str] = {
            "silicon_nitride": "dielectrics/silicon_nitride.csv",
            "aluminum_oxide": "dielectrics/aluminum_oxide.csv",
            "tantalum_pentoxide": "dielectrics/tantalum_pentoxide.csv",
            "magnesium_fluoride": "dielectrics/magnesium_fluoride.csv",
            "zinc_oxide": "dielectrics/zinc_oxide.csv",
            "indium_tin_oxide": "dielectrics/indium_tin_oxide.csv",
            "silicon_oxynitride": "dielectrics/silicon_oxynitride.csv",
        }

        # Polymers (microlens, planarization, photoresist)
        polymers_mapping: dict[str, str] = {
            "pmma": "polymers/pmma.csv",
            "polycarbonate": "polymers/polycarbonate.csv",
            "polyimide": "polymers/polyimide.csv",
            "benzocyclobutene": "polymers/benzocyclobutene.csv",
            "su8": "polymers/su8.csv",
        }

        # Semiconductors (beyond silicon)
        semiconductors_mapping: dict[str, str] = {
            "germanium": "semiconductors/germanium.csv",
            "gallium_arsenide": "semiconductors/gallium_arsenide.csv",
            "indium_phosphide": "semiconductors/indium_phosphide.csv",
        }

        all_mappings = {
            **metals_mapping,
            **dielectrics_mapping,
            **polymers_mapping,
            **semiconductors_mapping,
        }

        for name, rel_path in all_mappings.items():
            csv_path = self._db_path / rel_path
            if csv_path.exists():
                self.load_csv(name, str(csv_path))
                logger.debug(f"Loaded extended material '{name}' from {rel_path}")
            else:
                logger.debug(f"Extended material CSV not found for '{name}': {csv_path}")

    def _register_silicon_fallback(self) -> None:
        """Register silicon with approximate tabulated data.

        n after Aspnes & Studna (1983); k = alpha*lambda/4pi with alpha from
        Green 2008 (300K). Matches materials/silicon_green2008.csv.
        """
        wl = np.array(
            [
                0.350,
                0.360,
                0.370,
                0.380,
                0.390,
                0.400,
                0.410,
                0.420,
                0.430,
                0.440,
                0.450,
                0.460,
                0.470,
                0.480,
                0.490,
                0.500,
                0.510,
                0.520,
                0.530,
                0.540,
                0.550,
                0.560,
                0.570,
                0.580,
                0.590,
                0.600,
                0.620,
                0.640,
                0.660,
                0.680,
                0.700,
                0.720,
                0.740,
                0.760,
                0.780,
                0.800,
                0.850,
                0.900,
                0.950,
                1.000,
                1.050,
                1.100,
            ]
        )
        n = np.array(
            [
                5.44,
                6.01,
                6.86,
                6.55,
                6.04,
                5.57,
                5.28,
                5.10,
                4.95,
                4.82,
                4.69,
                4.59,
                4.50,
                4.42,
                4.36,
                4.30,
                4.25,
                4.18,
                4.14,
                4.11,
                4.08,
                4.05,
                4.02,
                4.00,
                3.97,
                3.94,
                3.90,
                3.87,
                3.84,
                3.81,
                3.78,
                3.76,
                3.74,
                3.72,
                3.71,
                3.69,
                3.65,
                3.62,
                3.60,
                3.58,
                3.57,
                3.55,
            ]
        )
        k = np.array(
            [
                2.90,
                2.91,
                2.06,
                0.880,
                0.440,
                0.303,
                0.220,
                0.167,
                0.134,
                0.109,
                0.0913,
                0.0775,
                0.0651,
                0.0563,
                0.0496,
                0.0441,
                0.0394,
                0.0364,
                0.0331,
                0.0303,
                0.0280,
                0.0258,
                0.0241,
                0.0225,
                0.0211,
                0.0198,
                0.0174,
                0.0155,
                0.0136,
                0.0120,
                0.0106,
                0.0095,
                0.0084,
                0.0073,
                0.0065,
                0.0054,
                0.0036,
                0.0022,
                0.0012,
                0.00051,
                0.00015,
                0.00003,
            ]
        )
        mat = MaterialData(
            name="silicon",
            mat_type="tabulated",
            wavelengths=wl,
            n_data=n,
            k_data=k,
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
            wavelengths=wl,
            n_data=n,
            k_data=k,
        )
        mat._build_interpolators()
        self._materials["tungsten"] = mat

    def _register_color_filter_fallback(self, name: str) -> None:
        """Register color filter with approximate Gaussian-passband absorption profiles.

        Pigment CFA dyes become nearly transparent in the NIR (the reason
        sensors need IR-cut filters), so k rolls off beyond ~0.75 um. The red
        filter is a long-pass dye: transparent from its passband through the NIR.
        """
        wl = np.linspace(0.38, 1.10, 73)

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
        k = k_max * (1.0 - np.exp(-(((wl - peak_wl) / width) ** 2)))
        if name == "cf_red":
            # Long-pass: transmissive from the passband through the NIR
            k = np.where(wl > peak_wl, 0.003, k)
        else:
            # NIR leakage: blocking dies off beyond ~0.78 um
            k = k / (1.0 + np.exp((wl - 0.80) / 0.05)) + 0.003
        n = np.full_like(wl, n_base)

        mat = MaterialData(
            name=name,
            mat_type="tabulated",
            wavelengths=wl,
            n_data=n,
            k_data=k,
        )
        mat._build_interpolators()
        self._materials[name] = mat

    def register_constant(self, name: str, n: float, k: float = 0.0) -> None:
        """Register a material with constant n, k."""
        self._materials[name] = MaterialData(
            name=name,
            mat_type="constant",
            n_const=n,
            k_const=k,
        )

    def register_cauchy(self, name: str, A: float, B: float = 0.0, C: float = 0.0) -> None:
        """Register a material with Cauchy dispersion model: n(λ) = A + B/λ² + C/λ⁴."""
        self._materials[name] = MaterialData(
            name=name,
            mat_type="cauchy",
            cauchy_A=A,
            cauchy_B=B,
            cauchy_C=C,
        )

    def register_sellmeier(self, name: str, B: list[float], C: list[float]) -> None:
        """Register a material with Sellmeier dispersion model."""
        self._materials[name] = MaterialData(
            name=name,
            mat_type="sellmeier",
            sellmeier_B=B,
            sellmeier_C=C,
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
        """Get complex permittivity at given wavelength. ε = (n + ik)².

        Results are cached to avoid redundant computation during wavelength sweeps.
        """
        cache_key = (name, round(wavelength, 10))
        cached = self._epsilon_cache.get(cache_key)
        if cached is not None:
            return cached
        if name not in self._materials:
            raise KeyError(f"Unknown material: '{name}'. Available: {list(self._materials.keys())}")
        eps = self._materials[name].get_epsilon(wavelength)
        self._epsilon_cache[cache_key] = eps
        return eps

    def get_epsilon_spectrum(self, name: str, wavelengths: np.ndarray) -> np.ndarray:
        """Get complex permittivity over wavelength array."""
        return np.array([self.get_epsilon(name, wl) for wl in wavelengths])

    def list_materials(self) -> list[str]:
        """List all available material names."""
        return sorted(self._materials.keys())

    def has_material(self, name: str) -> bool:
        """Check if a material exists in the database."""
        return name in self._materials
