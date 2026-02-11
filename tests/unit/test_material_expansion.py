"""Unit tests for the expanded MaterialDB material library.

Tests cover newly added metals, dielectrics, polymers, and semiconductors.
Verifies physical reasonableness of optical constants (n, k) and
correct integration with the existing MaterialDB infrastructure.
"""

import numpy as np
import pytest

from compass.materials.database import MaterialDB


@pytest.fixture
def db():
    """Create a MaterialDB instance with all built-in materials."""
    return MaterialDB()


# ---------------------------------------------------------------------------
# Material categories
# ---------------------------------------------------------------------------

METALS = ["aluminum", "gold", "silver", "copper", "titanium", "titanium_nitride"]
DIELECTRICS = [
    "silicon_nitride", "aluminum_oxide", "tantalum_pentoxide",
    "magnesium_fluoride", "zinc_oxide", "indium_tin_oxide",
    "silicon_oxynitride",
]
POLYMERS = ["pmma", "polycarbonate", "polyimide", "benzocyclobutene", "su8"]
SEMICONDUCTORS = ["germanium", "gallium_arsenide", "indium_phosphide"]

ALL_NEW_MATERIALS = METALS + DIELECTRICS + POLYMERS + SEMICONDUCTORS


class TestNewMaterialsRegistered:
    """Verify all new materials are loadable by name."""

    @pytest.mark.parametrize("name", ALL_NEW_MATERIALS)
    def test_material_exists(self, db, name):
        """Each new material should be registered in the database."""
        assert db.has_material(name), f"Material '{name}' not found in database"

    def test_total_material_count(self, db):
        """Database should contain both original and new materials."""
        materials = db.list_materials()
        # Original: air, polymer_n1p56, sio2, si3n4, hfo2, tio2, silicon,
        #           tungsten, cf_red, cf_green, cf_blue = 11
        # New: 6 metals + 7 dielectrics + 5 polymers + 3 semiconductors = 21
        # Total = 32
        assert len(materials) >= 32, (
            f"Expected at least 32 materials, got {len(materials)}"
        )


class TestPhysicalReasonableness:
    """Verify n and k values are physically reasonable for all materials."""

    @pytest.mark.parametrize("name", ALL_NEW_MATERIALS)
    def test_n_positive(self, db, name):
        """Refractive index n must be positive at all test wavelengths."""
        for wl in [0.40, 0.55, 0.70, 0.90]:
            n, _k = db.get_nk(name, wl)
            assert n > 0, f"{name}: n={n} at {wl} um is not positive"

    @pytest.mark.parametrize("name", ALL_NEW_MATERIALS)
    def test_k_non_negative(self, db, name):
        """Extinction coefficient k must be non-negative."""
        for wl in [0.40, 0.55, 0.70, 0.90]:
            _n, k = db.get_nk(name, wl)
            assert k >= 0, f"{name}: k={k} at {wl} um is negative"

    @pytest.mark.parametrize("name", ALL_NEW_MATERIALS)
    def test_no_nan_values(self, db, name):
        """No NaN values should appear at any wavelength in range."""
        wavelengths = np.linspace(0.35, 1.0, 20)
        for wl in wavelengths:
            n, k = db.get_nk(name, wl)
            assert np.isfinite(n), f"{name}: n is NaN/inf at {wl} um"
            assert np.isfinite(k), f"{name}: k is NaN/inf at {wl} um"


class TestMetalProperties:
    """Verify metals have characteristic optical properties."""

    @pytest.mark.parametrize("name", METALS)
    def test_metals_have_large_k_visible(self, db, name):
        """Metals should have k > 1 in the visible range."""
        _n, k = db.get_nk(name, 0.55)
        assert k > 1.0, (
            f"{name}: k={k} at 550nm -- metals should have k > 1 in visible"
        )

    def test_aluminum_high_reflectivity(self, db):
        """Aluminum should have low n and very high k in visible (broadband reflector)."""
        n, k = db.get_nk("aluminum", 0.55)
        assert n < 1.5, f"Al n={n} at 550nm, expected < 1.5"
        assert k > 4.0, f"Al k={k} at 550nm, expected > 4.0"

    def test_gold_plasmonic_transition(self, db):
        """Gold should have low n below 500nm and increasing k above it."""
        _n_blue, k_blue = db.get_nk("gold", 0.45)
        _n_red, k_red = db.get_nk("gold", 0.65)
        assert k_red > k_blue, "Au k should increase with wavelength above 500nm"

    def test_silver_best_visible_reflector(self, db):
        """Silver should have the lowest n among metals in visible."""
        n_ag, _k = db.get_nk("silver", 0.55)
        n_al, _ = db.get_nk("aluminum", 0.55)
        n_au, _ = db.get_nk("gold", 0.55)
        assert n_ag < n_al, "Ag should have lower n than Al at 550nm"
        assert n_ag < n_au, "Ag should have lower n than Au at 550nm"


class TestDielectricProperties:
    """Verify dielectrics have characteristic optical properties."""

    @pytest.mark.parametrize("name", DIELECTRICS)
    def test_dielectric_small_k_visible(self, db, name):
        """Dielectrics should have k near zero in visible range."""
        _n, k = db.get_nk(name, 0.55)
        assert k < 0.01, (
            f"{name}: k={k} at 550nm -- dielectrics should have k ~ 0 in visible"
        )

    def test_silicon_nitride_n_approx_2(self, db):
        """Si3N4 tabulated data should give n ~ 2.0 at 550nm."""
        n, k = db.get_nk("silicon_nitride", 0.55)
        assert 1.95 < n < 2.10, f"Si3N4 n={n} at 550nm, expected ~2.0"
        assert k < 0.001, f"Si3N4 k={k} at 550nm, expected ~0"

    def test_aluminum_oxide_n_approx_1p77(self, db):
        """Al2O3 should have n ~ 1.77 at 550nm."""
        n, k = db.get_nk("aluminum_oxide", 0.55)
        assert 1.72 < n < 1.82, f"Al2O3 n={n} at 550nm, expected ~1.77"
        assert k == 0.0, f"Al2O3 k={k} at 550nm, expected 0"

    def test_magnesium_fluoride_low_n(self, db):
        """MgF2 should have n ~ 1.38, one of the lowest index dielectrics."""
        n, k = db.get_nk("magnesium_fluoride", 0.55)
        assert 1.35 < n < 1.42, f"MgF2 n={n} at 550nm, expected ~1.38"
        assert k == 0.0, f"MgF2 k={k} at 550nm, expected 0"

    def test_tantalum_pentoxide_high_n(self, db):
        """Ta2O5 should have n ~ 2.1, a high-index dielectric."""
        n, _k = db.get_nk("tantalum_pentoxide", 0.55)
        assert 2.05 < n < 2.20, f"Ta2O5 n={n} at 550nm, expected ~2.1"

    def test_ito_transparent_conductor(self, db):
        """ITO should be transparent in visible (small k) with n ~ 1.9-2.0."""
        n, k = db.get_nk("indium_tin_oxide", 0.55)
        assert 1.85 < n < 2.05, f"ITO n={n} at 550nm, expected ~1.95"
        assert k < 0.01, f"ITO k={k} at 550nm, expected near-zero in visible"

    def test_sion_between_sio2_and_si3n4(self, db):
        """SiON refractive index should be between SiO2 and Si3N4."""
        n_sio2, _ = db.get_nk("sio2", 0.55)
        n_si3n4, _ = db.get_nk("silicon_nitride", 0.55)
        n_sion, _ = db.get_nk("silicon_oxynitride", 0.55)
        assert n_sio2 < n_sion < n_si3n4, (
            f"SiON n={n_sion} should be between SiO2 n={n_sio2} "
            f"and Si3N4 n={n_si3n4}"
        )


class TestPolymerProperties:
    """Verify polymers have characteristic optical properties."""

    @pytest.mark.parametrize("name", POLYMERS)
    def test_polymer_transparent_visible(self, db, name):
        """Polymers should be transparent (k ~ 0) in visible range."""
        _n, k = db.get_nk(name, 0.55)
        assert k < 0.001, (
            f"{name}: k={k} at 550nm -- polymers should be transparent in visible"
        )

    def test_pmma_n_approx_1p49(self, db):
        """PMMA should have n ~ 1.49 at 550nm."""
        n, _k = db.get_nk("pmma", 0.55)
        assert 1.45 < n < 1.53, f"PMMA n={n} at 550nm, expected ~1.49"

    def test_polycarbonate_n_approx_1p58(self, db):
        """Polycarbonate should have n ~ 1.58 at 550nm."""
        n, _k = db.get_nk("polycarbonate", 0.55)
        assert 1.55 < n < 1.63, f"PC n={n} at 550nm, expected ~1.58"

    def test_polyimide_n_approx_1p70(self, db):
        """Polyimide should have n ~ 1.70 at 550nm."""
        n, _k = db.get_nk("polyimide", 0.55)
        assert 1.65 < n < 1.75, f"PI n={n} at 550nm, expected ~1.70"

    def test_su8_n_approx_1p59(self, db):
        """SU-8 should have n ~ 1.59 at 550nm."""
        n, _k = db.get_nk("su8", 0.55)
        assert 1.55 < n < 1.63, f"SU-8 n={n} at 550nm, expected ~1.59"


class TestSemiconductorProperties:
    """Verify semiconductors have characteristic optical properties."""

    def test_germanium_high_n_visible(self, db):
        """Germanium should have high n (> 4) and significant k in visible."""
        n, k = db.get_nk("germanium", 0.55)
        assert n > 3.5, f"Ge n={n} at 550nm, expected > 3.5"
        assert k > 0.1, f"Ge k={k} at 550nm, expected > 0.1 (absorbing)"

    def test_germanium_high_absorption_visible(self, db):
        """Germanium should absorb strongly in UV-visible."""
        _n, k = db.get_nk("germanium", 0.40)
        assert k > 1.0, f"Ge k={k} at 400nm, expected > 1.0"

    def test_gaas_high_n(self, db):
        """GaAs should have high n > 3.5 in visible."""
        n, _k = db.get_nk("gallium_arsenide", 0.55)
        assert n > 3.5, f"GaAs n={n} at 550nm, expected > 3.5"

    def test_inp_high_n(self, db):
        """InP should have high n > 3.0 in visible."""
        n, _k = db.get_nk("indium_phosphide", 0.55)
        assert n > 3.0, f"InP n={n} at 550nm, expected > 3.0"


class TestInterpolation:
    """Verify interpolation works correctly across the wavelength range."""

    @pytest.mark.parametrize("name", ALL_NEW_MATERIALS)
    def test_interpolation_mid_range(self, db, name):
        """Interpolation at a mid-range wavelength should produce valid results."""
        n, k = db.get_nk(name, 0.575)
        assert np.isfinite(n), f"{name}: n not finite at 0.575 um"
        assert np.isfinite(k), f"{name}: k not finite at 0.575 um"
        assert n > 0, f"{name}: n={n} at 0.575 um is not positive"
        assert k >= 0, f"{name}: k={k} at 0.575 um is negative"

    def test_epsilon_spectrum_new_materials(self, db):
        """Complex permittivity computation should work for new materials."""
        wl = np.array([0.45, 0.55, 0.65, 0.80, 1.00])
        for name in ["aluminum", "silicon_nitride", "pmma", "germanium"]:
            eps = db.get_epsilon_spectrum(name, wl)
            assert len(eps) == len(wl), f"{name}: epsilon spectrum length mismatch"
            assert all(np.isfinite(eps)), f"{name}: non-finite epsilon values"

    def test_epsilon_complex_for_metals(self, db):
        """Metals should produce complex epsilon with significant imaginary part."""
        eps = db.get_epsilon("aluminum", 0.55)
        assert np.imag(eps) != 0, "Al epsilon should have nonzero imaginary part"
        # For metals: Re(eps) is typically negative
        assert np.real(eps) < 0, (
            f"Al Re(eps)={np.real(eps)} at 550nm, expected negative for metals"
        )


class TestOriginalMaterialsUnchanged:
    """Verify that adding new materials does not break existing ones."""

    def test_silicon_unchanged(self, db):
        """Silicon optical constants should remain unchanged."""
        n, k = db.get_nk("silicon", 0.55)
        assert 3.5 < n < 4.5, f"Si n={n} at 550nm changed unexpectedly"
        assert 0.0 <= k < 0.2, f"Si k={k} at 550nm changed unexpectedly"

    def test_sio2_unchanged(self, db):
        """SiO2 should still work via Sellmeier model."""
        n, k = db.get_nk("sio2", 0.55)
        assert 1.4 < n < 1.6, f"SiO2 n={n} unexpected"
        assert k == 0.0

    def test_air_unchanged(self, db):
        """Air should remain n=1, k=0."""
        n, k = db.get_nk("air", 0.55)
        assert n == 1.0
        assert k == 0.0

    def test_tungsten_unchanged(self, db):
        """Tungsten should still have metallic properties."""
        n, k = db.get_nk("tungsten", 0.55)
        assert n > 1.0
        assert k > 1.0
