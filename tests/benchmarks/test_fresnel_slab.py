"""Benchmark tests for Fresnel reflection calculations.

Validates optical physics using the Fresnel equations against textbook values,
using real material properties from the COMPASS material database.
"""

import numpy as np
import pytest

from compass.materials.database import MaterialDB


def fresnel_reflection(n1: complex, n2: complex) -> float:
    """Fresnel reflectance at normal incidence for the interface n1 -> n2.

    R = |r|^2 where r = (n1 - n2) / (n1 + n2) for complex refractive indices.
    """
    r = (n1 - n2) / (n1 + n2)
    return float(np.abs(r) ** 2)


def fresnel_r_s(n1: float, n2: float, theta_i: float) -> float:
    """Fresnel reflectance for s-polarization at angle theta_i (radians)."""
    cos_i = np.cos(theta_i)
    sin_t = (n1 / n2) * np.sin(theta_i)
    if np.abs(sin_t) > 1.0:
        return 1.0  # total internal reflection
    cos_t = np.sqrt(1.0 - sin_t**2)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    return float(r_s**2)


def fresnel_r_p(n1: float, n2: float, theta_i: float) -> float:
    """Fresnel reflectance for p-polarization at angle theta_i (radians)."""
    cos_i = np.cos(theta_i)
    sin_t = (n1 / n2) * np.sin(theta_i)
    if np.abs(sin_t) > 1.0:
        return 1.0  # total internal reflection
    cos_t = np.sqrt(1.0 - sin_t**2)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return float(r_p**2)


class TestFresnelAirGlass:
    """Fresnel reflection for Air -> Glass (n=1.5) interface."""

    def test_normal_incidence_reflectance(self):
        """R = ((1.0 - 1.5)/(1.0 + 1.5))^2 = 0.04 (4%)."""
        R = fresnel_reflection(1.0, 1.5)
        assert R == pytest.approx(0.04, abs=1e-6)

    def test_brewster_angle(self):
        """Brewster angle for Air->Glass: theta_B = arctan(n2/n1) ~ 56.31 degrees."""
        n1, n2 = 1.0, 1.5
        theta_brewster = np.arctan(n2 / n1)  # radians
        theta_brewster_deg = np.degrees(theta_brewster)

        assert theta_brewster_deg == pytest.approx(56.31, abs=0.01)

        # At Brewster's angle, p-polarization reflectance should vanish
        R_p = fresnel_r_p(n1, n2, theta_brewster)
        assert R_p == pytest.approx(0.0, abs=1e-10)

        # s-polarization should still have significant reflection
        R_s = fresnel_r_s(n1, n2, theta_brewster)
        assert R_s > 0.1

    def test_grazing_incidence(self):
        """Reflectance approaches 1 at grazing incidence (theta -> 90 deg)."""
        n1, n2 = 1.0, 1.5
        theta_graze = np.radians(89.9)
        R_s = fresnel_r_s(n1, n2, theta_graze)
        R_p = fresnel_r_p(n1, n2, theta_graze)
        assert R_s > 0.99
        assert R_p > 0.98

    def test_reciprocity(self):
        """Reflectance at normal incidence is the same from either side."""
        R_forward = fresnel_reflection(1.0, 1.5)
        R_reverse = fresnel_reflection(1.5, 1.0)
        assert R_forward == pytest.approx(R_reverse, abs=1e-10)


class TestFresnelAirSilicon:
    """Fresnel reflection for Air -> Silicon at 550 nm using MaterialDB."""

    @pytest.fixture
    def db(self):
        return MaterialDB()

    def test_silicon_refractive_index_550nm(self, db):
        """Silicon at 550nm should have n ~ 4.08, k ~ 0.028 (Green 2008)."""
        n, k = db.get_nk("silicon", 0.550)
        # The fallback data may differ slightly from exact Green 2008 values
        # due to interpolation, but should be in the right ballpark
        assert 3.5 < n < 4.5, f"Silicon n={n} out of expected range at 550nm"
        assert 0.0 <= k < 0.2, f"Silicon k={k} out of expected range at 550nm"

    def test_silicon_reflectance_550nm(self, db):
        """Air -> Silicon reflectance at 550nm should be approximately 36%."""
        n, k = db.get_nk("silicon", 0.550)
        n_complex = n + 1j * k
        R = fresnel_reflection(1.0, n_complex)
        # Accept a range due to interpolation variations in fallback data
        assert 0.30 < R < 0.42, f"Silicon R={R:.4f} outside expected range at 550nm"

    def test_silicon_high_reflectance_uv(self, db):
        """Silicon reflectance should be higher in UV due to higher n, k."""
        n_uv, k_uv = db.get_nk("silicon", 0.400)
        n_vis, k_vis = db.get_nk("silicon", 0.550)
        R_uv = fresnel_reflection(1.0, n_uv + 1j * k_uv)
        R_vis = fresnel_reflection(1.0, n_vis + 1j * k_vis)
        assert R_uv > R_vis, "UV reflectance should exceed visible for silicon"

    def test_silicon_permittivity_relation(self, db):
        """Verify epsilon = (n + ik)^2 relationship."""
        n, k = db.get_nk("silicon", 0.550)
        eps_from_nk = (n + 1j * k) ** 2
        eps_from_db = db.get_epsilon("silicon", 0.550)
        assert eps_from_nk == pytest.approx(eps_from_db, rel=1e-10)


class TestMaterialDBProperties:
    """Verify that MaterialDB provides physically reasonable properties."""

    @pytest.fixture
    def db(self):
        return MaterialDB()

    def test_air_refractive_index(self, db):
        """Air should have n=1, k=0 at all wavelengths."""
        for wl in [0.4, 0.55, 0.7, 1.0]:
            n, k = db.get_nk("air", wl)
            assert n == pytest.approx(1.0, abs=1e-10)
            assert k == pytest.approx(0.0, abs=1e-10)

    def test_sio2_refractive_index(self, db):
        """SiO2 should have n ~ 1.46 in visible, k = 0."""
        n, k = db.get_nk("sio2", 0.550)
        assert 1.40 < n < 1.50, f"SiO2 n={n} outside expected range"
        assert k == pytest.approx(0.0, abs=1e-10)

    def test_si3n4_refractive_index(self, db):
        """Si3N4 should have n ~ 2.0 in visible."""
        n, k = db.get_nk("si3n4", 0.550)
        assert 1.8 < n < 2.3, f"Si3N4 n={n} outside expected range"

    def test_all_builtin_materials_exist(self, db):
        """All expected built-in materials should be registered."""
        expected = ["air", "silicon", "sio2", "si3n4", "hfo2", "tio2",
                    "polymer_n1p56", "tungsten", "cf_red", "cf_green", "cf_blue"]
        available = db.list_materials()
        for mat in expected:
            assert mat in available, f"Material '{mat}' not found in database"

    def test_cauchy_dispersion_increases(self, db):
        """Cauchy materials should have increasing n at shorter wavelengths."""
        n_blue, _ = db.get_nk("polymer_n1p56", 0.450)
        n_red, _ = db.get_nk("polymer_n1p56", 0.650)
        assert n_blue > n_red, "Cauchy dispersion: n should increase at shorter wavelengths"

    def test_epsilon_spectrum_shape(self, db):
        """get_epsilon_spectrum should return array matching wavelength input."""
        wl = np.linspace(0.4, 0.7, 31)
        eps = db.get_epsilon_spectrum("silicon", wl)
        assert eps.shape == (31,)
        assert eps.dtype == complex
