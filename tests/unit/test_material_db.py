"""Unit tests for MaterialDB."""

import numpy as np
import pytest

from compass.materials.database import MaterialDB


@pytest.fixture
def db():
    """Create a MaterialDB instance."""
    return MaterialDB()


class TestMaterialDB:
    """Tests for material database."""

    def test_builtin_materials_registered(self, db):
        """All built-in materials should be available."""
        expected = ["air", "polymer_n1p56", "sio2", "si3n4", "tio2", "hfo2", "silicon", "tungsten"]
        for name in expected:
            assert db.has_material(name), f"Missing built-in material: {name}"

    def test_color_filters_registered(self, db):
        """Color filter materials should be available."""
        for name in ["cf_red", "cf_green", "cf_blue"]:
            assert db.has_material(name), f"Missing color filter: {name}"

    def test_air_refractive_index(self, db):
        """Air should have n=1.0, k=0.0."""
        n, k = db.get_nk("air", 0.55)
        assert n == 1.0
        assert k == 0.0

    def test_air_epsilon(self, db):
        """Air epsilon should be 1.0."""
        eps = db.get_epsilon("air", 0.55)
        assert np.isclose(eps, 1.0)

    def test_silicon_visible_range(self, db):
        """Silicon n, k at 550nm should be physically reasonable."""
        n, k = db.get_nk("silicon", 0.55)
        # Green 2008: n ≈ 4.08, k ≈ 0.028 at 550nm
        assert 3.5 < n < 4.5, f"Si n={n} at 550nm out of range"
        assert 0.0 <= k < 0.2, f"Si k={k} at 550nm out of range"

    def test_silicon_uv_high_absorption(self, db):
        """Silicon should have high k in UV."""
        _n, k = db.get_nk("silicon", 0.40)
        assert k > 0.1, f"Si k={k} at 400nm should be high"

    def test_silicon_nir_low_absorption(self, db):
        """Silicon should have low k in NIR."""
        _n, k = db.get_nk("silicon", 0.80)
        assert k < 0.05, f"Si k={k} at 800nm should be low"

    def test_sio2_transparent(self, db):
        """SiO2 should be transparent in visible (k≈0)."""
        n, k = db.get_nk("sio2", 0.55)
        assert 1.4 < n < 1.6, f"SiO2 n={n} unexpected"
        assert k == 0.0

    def test_polymer_cauchy(self, db):
        """Polymer should follow Cauchy model."""
        n, k = db.get_nk("polymer_n1p56", 0.55)
        assert 1.5 < n < 1.7, f"Polymer n={n} unexpected"
        assert k == 0.0

    def test_tungsten_metallic(self, db):
        """Tungsten should have both significant n and k."""
        n, k = db.get_nk("tungsten", 0.55)
        assert n > 1.0, f"W n={n} unexpected"
        assert k > 1.0, f"W k={k} should be metallic"

    def test_epsilon_complex(self, db):
        """Epsilon should be complex for absorbing materials."""
        eps = db.get_epsilon("silicon", 0.40)
        assert np.imag(eps) != 0, "Si epsilon should be complex in UV"

    def test_epsilon_spectrum(self, db):
        """Should compute epsilon over wavelength array."""
        wl = np.array([0.45, 0.55, 0.65])
        eps = db.get_epsilon_spectrum("silicon", wl)
        assert len(eps) == 3
        assert all(np.isfinite(eps))

    def test_register_constant(self, db):
        """Should register custom constant material."""
        db.register_constant("my_glass", n=1.52, k=0.0)
        n, k = db.get_nk("my_glass", 0.55)
        assert n == 1.52
        assert k == 0.0

    def test_register_cauchy(self, db):
        """Should register custom Cauchy material."""
        db.register_cauchy("my_polymer", A=1.60, B=0.005, C=0.001)
        n, _k = db.get_nk("my_polymer", 0.55)
        assert n > 1.6

    def test_unknown_material_raises(self, db):
        """Should raise KeyError for unknown material."""
        with pytest.raises(KeyError):
            db.get_nk("nonexistent", 0.55)

    def test_list_materials(self, db):
        """Should list all materials."""
        materials = db.list_materials()
        assert isinstance(materials, list)
        assert len(materials) >= 8
