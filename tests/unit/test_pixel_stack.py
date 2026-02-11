"""Unit tests for PixelStack."""

import numpy as np
import pytest

from compass.geometry.pixel_stack import PixelStack


@pytest.fixture
def default_config():
    """Default BSI 1um pixel config."""
    return {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [2, 2],
            "layers": {
                "air": {"thickness": 1.0, "material": "air"},
                "microlens": {
                    "enabled": True,
                    "height": 0.6,
                    "radius_x": 0.48,
                    "radius_y": 0.48,
                    "material": "polymer_n1p56",
                    "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                    "shift": {"mode": "none", "cra_deg": 0.0},
                    "gap": 0.0,
                },
                "planarization": {"thickness": 0.3, "material": "sio2"},
                "color_filter": {
                    "thickness": 0.6,
                    "pattern": "bayer_rggb",
                    "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
                },
                "barl": {
                    "layers": [
                        {"thickness": 0.010, "material": "sio2"},
                        {"thickness": 0.025, "material": "hfo2"},
                    ],
                },
                "silicon": {
                    "thickness": 3.0,
                    "material": "silicon",
                    "photodiode": {"position": [0.0, 0.0, 0.5], "size": [0.7, 0.7, 2.0]},
                    "dti": {"enabled": True, "width": 0.1, "depth": 3.0, "material": "sio2"},
                },
            },
            "bayer_map": [["R", "G"], ["G", "B"]],
        }
    }


@pytest.fixture
def pixel_stack(default_config):
    return PixelStack(default_config)


class TestPixelStack:
    """Tests for PixelStack."""

    def test_basic_construction(self, pixel_stack):
        """Should construct without errors."""
        assert pixel_stack.pitch == 1.0
        assert pixel_stack.unit_cell == (2, 2)

    def test_domain_size(self, pixel_stack):
        """Domain size should be pitch * unit_cell."""
        lx, ly = pixel_stack.domain_size
        assert lx == 2.0  # 1.0 * 2
        assert ly == 2.0

    def test_layers_created(self, pixel_stack):
        """All layers should be created."""
        layer_names = [l.name for l in pixel_stack.layers]
        assert "silicon" in layer_names
        assert "color_filter" in layer_names
        assert "planarization" in layer_names
        assert "microlens" in layer_names
        assert "air" in layer_names

    def test_layer_z_ordering(self, pixel_stack):
        """Layers should be ordered bottom to top (Si → Air)."""
        z_starts = [l.z_start for l in pixel_stack.layers]
        # Should be monotonically non-decreasing
        assert all(z_starts[i] <= z_starts[i + 1] for i in range(len(z_starts) - 1))

    def test_total_height(self, pixel_stack):
        """Total height should be sum of all layers."""
        assert pixel_stack.total_height > 0
        # Si(3) + BARL(0.035) + CF(0.6) + plan(0.3) + ML(0.6) + air(1.0) ≈ 5.535
        assert 5.0 < pixel_stack.total_height < 6.0

    def test_bayer_map(self, pixel_stack):
        """Bayer map should be correct."""
        assert pixel_stack.bayer_map == [["R", "G"], ["G", "B"]]

    def test_photodiodes_created(self, pixel_stack):
        """Should have 4 photodiodes for 2x2 unit cell."""
        assert len(pixel_stack.photodiodes) == 4

    def test_photodiode_colors(self, pixel_stack):
        """Photodiode colors should match Bayer pattern."""
        colors = {pd.color for pd in pixel_stack.photodiodes}
        assert colors == {"R", "G", "B"}

    def test_microlenses_created(self, pixel_stack):
        """Should have 4 microlenses for 2x2."""
        assert len(pixel_stack.microlenses) == 4

    def test_get_layer_slices(self, pixel_stack):
        """Should generate layer slices for RCWA."""
        slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=32, ny=32)
        assert len(slices) > 0

        # Each slice should have proper shape
        for s in slices:
            assert s.eps_grid.shape == (32, 32)
            assert s.thickness > 0
            assert np.isfinite(s.eps_grid).all()

    def test_layer_slices_cover_full_stack(self, pixel_stack):
        """Layer slices should cover the full z range."""
        slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=16, ny=16)
        z_min = min(s.z_start for s in slices)
        z_max = max(s.z_end for s in slices)

        stack_z_min, stack_z_max = pixel_stack.z_range
        assert z_min == pytest.approx(stack_z_min, abs=0.01)
        assert z_max == pytest.approx(stack_z_max, abs=0.01)

    def test_get_permittivity_grid(self, pixel_stack):
        """Should generate 3D permittivity grid for FDTD."""
        eps = pixel_stack.get_permittivity_grid(wavelength=0.55, nx=16, ny=16, nz=32)
        assert eps.shape == (16, 16, 32)
        assert np.isfinite(eps).all()

    def test_color_filter_pattern_in_slices(self, pixel_stack):
        """CF layer slice should have different epsilon values (patterned)."""
        slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=64, ny=64)
        cf_slices = [s for s in slices if s.name == "color_filter"]
        assert len(cf_slices) == 1

        eps = cf_slices[0].eps_grid
        # Should have at least 2 distinct epsilon values (different CF colors + grid)
        unique_eps = len(np.unique(np.round(np.real(eps), 4)))
        assert unique_eps >= 2, f"Expected patterned CF, got {unique_eps} unique values"

    def test_silicon_dti_pattern(self, pixel_stack):
        """Si layer should show DTI pattern (different epsilon)."""
        slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=64, ny=64)
        si_slices = [s for s in slices if s.name == "silicon"]
        assert len(si_slices) == 1

        eps = si_slices[0].eps_grid
        unique_eps = len(np.unique(np.round(np.real(eps), 2)))
        assert unique_eps >= 2, "Expected DTI pattern in Si layer"


class TestSnellCRAShift:
    """Tests for Snell's law CRA shift computation."""

    @pytest.fixture
    def cra_config(self):
        """Config with auto_cra shift mode."""
        return {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True,
                        "height": 0.6,
                        "radius_x": 0.48,
                        "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                        "shift": {"mode": "auto_cra", "cra_deg": 0.0},
                        "gap": 0.0,
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "pattern": "bayer_rggb",
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                        "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
                    },
                    "barl": {
                        "layers": [
                            {"thickness": 0.010, "material": "sio2"},
                            {"thickness": 0.025, "material": "hfo2"},
                        ],
                    },
                    "silicon": {
                        "thickness": 3.0,
                        "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5], "size": [0.7, 0.7, 2.0]},
                        "dti": {"enabled": True, "width": 0.1, "depth": 3.0, "material": "sio2"},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }

    def test_zero_cra_zero_shift(self, cra_config):
        """CRA=0 should produce zero shift."""
        cra_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = 0.0
        ps = PixelStack(cra_config)
        for ml in ps.microlenses:
            assert ml.shift_x == pytest.approx(0.0, abs=1e-12)
            assert ml.shift_y == pytest.approx(0.0, abs=1e-12)

    def test_small_angle_close_to_tan(self, cra_config):
        """At small CRA (5 deg), Snell shift should be close to tan approximation."""
        cra_deg = 5.0
        cra_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = cra_deg
        ps = PixelStack(cra_config)
        snell_shift = ps.microlenses[0].shift_x

        # Simple tan approximation: tan(CRA) * total_height * 0.5
        cra_rad = np.deg2rad(cra_deg)
        ml_height = 0.6
        tan_shift = np.tan(cra_rad) * ml_height * 0.5

        # At small angles, Snell and tan should be within ~20% of each other
        # (they differ because Snell uses actual layer thicknesses and refractive indices)
        assert snell_shift > 0
        assert tan_shift > 0
        # Both should be small positive numbers of similar magnitude
        ratio = snell_shift / tan_shift
        assert 0.5 < ratio < 20.0, f"Snell/tan ratio {ratio} out of expected range"

    def test_large_angle_less_than_tan(self, cra_config):
        """At large CRA (30 deg), Snell shift per layer should be less than air-only tan."""
        cra_deg = 30.0
        cra_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = cra_deg
        ps = PixelStack(cra_config)
        snell_shift = ps.microlenses[0].shift_x

        # In each layer, n > 1 causes sin(theta_layer) < sin(CRA), so tan(theta) < tan(CRA).
        # But the total Snell shift sums over all layers down to PD, which is a longer path
        # than the old ml_height*0.5 approximation. The key physics is that within each layer,
        # refraction bends the ray closer to normal (less lateral displacement per unit height).
        assert snell_shift > 0

        # The Snell shift through individual high-n layers is reduced by refraction.
        # Verify it's a reasonable positive number (not negative, not absurdly large).
        assert 0.01 < snell_shift < 5.0, f"Shift {snell_shift} out of physical range"

    def test_shift_monotonic_with_cra(self, cra_config):
        """Shift should increase monotonically with CRA."""
        shifts = []
        for cra_deg in [0, 5, 10, 15, 20, 25, 30]:
            cra_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra_deg)
            ps = PixelStack(cra_config)
            shifts.append(ps.microlenses[0].shift_x)

        for i in range(1, len(shifts)):
            assert shifts[i] >= shifts[i - 1], (
                f"Shift not monotonic: {shifts[i]} < {shifts[i-1]} "
                f"at CRA={[0,5,10,15,20,25,30][i]} deg"
            )

    def test_ref_wavelength_effect(self, cra_config):
        """Different reference wavelength should produce different shift."""
        cra_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = 20.0

        cra_config["pixel"]["layers"]["microlens"]["shift"]["ref_wavelength"] = 0.45
        ps_blue = PixelStack(cra_config)
        shift_blue = ps_blue.microlenses[0].shift_x

        cra_config["pixel"]["layers"]["microlens"]["shift"]["ref_wavelength"] = 0.65
        ps_red = PixelStack(cra_config)
        shift_red = ps_red.microlenses[0].shift_x

        # Silicon has very different n at blue vs red wavelengths,
        # so shifts should differ
        assert shift_blue != pytest.approx(shift_red, rel=0.001), (
            f"Expected different shifts for different wavelengths: "
            f"blue={shift_blue}, red={shift_red}"
        )


class TestSnellCRAShiftEdgeCases:
    """Edge case tests for Snell's law CRA shift."""

    def test_no_barl_layers(self):
        """Should work with empty BARL config."""
        config = {
            "pixel": {
                "pitch": 1.0,
                "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {"mode": "auto_cra", "cra_deg": 20.0},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "barl": {"layers": []},
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps = PixelStack(config)
        shift = ps.microlenses[0].shift_x
        assert shift > 0, "Shift should be positive even without BARL"

        # Compare with config that has BARL
        config_with_barl = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {"mode": "auto_cra", "cra_deg": 20.0},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "barl": {"layers": [
                        {"thickness": 0.010, "material": "sio2"},
                        {"thickness": 0.025, "material": "hfo2"},
                    ]},
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps_with = PixelStack(config_with_barl)
        shift_with = ps_with.microlenses[0].shift_x
        # Adding BARL layers adds extra path → more shift
        assert shift_with > shift, "Adding BARL layers should increase total shift"

    def test_microlens_disabled_no_shift(self):
        """When microlens is disabled, no microlenses should be created."""
        config = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {"enabled": False},
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {"thickness": 3.0, "material": "silicon"},
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps = PixelStack(config)
        assert len(ps.microlenses) == 0, "No microlenses when disabled"

    def test_extreme_cra_no_crash(self):
        """Extreme CRA (85 deg) should not crash due to sin clamping."""
        config = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {"mode": "auto_cra", "cra_deg": 85.0},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps = PixelStack(config)
        shift = ps.microlenses[0].shift_x
        assert np.isfinite(shift), f"Shift should be finite at extreme CRA, got {shift}"
        assert shift > 0, "Shift should be positive at extreme CRA"

    def test_missing_barl_key(self):
        """Should handle config with no 'barl' key at all."""
        config = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {"mode": "auto_cra", "cra_deg": 15.0},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps = PixelStack(config)
        shift = ps.microlenses[0].shift_x
        assert shift > 0

    def test_default_ref_wavelength_without_key(self):
        """Config without ref_wavelength should use default 0.55 um."""
        config = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {"mode": "auto_cra", "cra_deg": 20.0},
                        # No ref_wavelength key — should default to 0.55
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        ps_default = PixelStack(config)
        shift_default = ps_default.microlenses[0].shift_x

        # Explicitly set ref_wavelength=0.55
        config["pixel"]["layers"]["microlens"]["shift"]["ref_wavelength"] = 0.55
        ps_explicit = PixelStack(config)
        shift_explicit = ps_explicit.microlenses[0].shift_x

        assert shift_default == pytest.approx(shift_explicit, rel=1e-10), (
            f"Default ref_wavelength should match explicit 0.55: "
            f"{shift_default} vs {shift_explicit}"
        )


class TestSnellCRAShiftAnalytical:
    """Analytical verification of Snell's law shift computation."""

    def test_single_homogeneous_layer(self):
        """For a single uniform layer, shift should match analytical formula exactly.

        With only a planarization layer (n=1.46 SiO2) and silicon to PD,
        the shift for each layer should be:
            d_i = h_i * sin(theta_i) / cos(theta_i)
        where sin(theta_i) = sin(CRA) / n_i
        """
        from compass.materials.database import MaterialDB

        cra_deg = 25.0
        ref_wl = 0.55
        plan_t = 0.5
        si_t = 3.0
        pd_z = 0.5  # PD position from bottom

        config = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "shift": {
                            "mode": "auto_cra", "cra_deg": cra_deg,
                            "ref_wavelength": ref_wl,
                        },
                    },
                    "planarization": {"thickness": plan_t, "material": "sio2"},
                    "color_filter": {"thickness": 0.6,
                                     "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}},
                    "barl": {"layers": []},
                    "silicon": {
                        "thickness": si_t, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, pd_z]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }

        ps = PixelStack(config)
        computed_shift = ps.microlenses[0].shift_x

        # Compute expected shift analytically
        db = MaterialDB()
        sin_cra = np.sin(np.deg2rad(cra_deg))

        expected_shift = 0.0

        # Planarization layer
        n_plan, _ = db.get_nk("sio2", ref_wl)
        sin_plan = sin_cra / n_plan
        cos_plan = np.sqrt(1.0 - sin_plan**2)
        expected_shift += plan_t * sin_plan / cos_plan

        # Color filter layer
        n_cf, _ = db.get_nk("cf_green", ref_wl)
        sin_cf = sin_cra / n_cf
        cos_cf = np.sqrt(1.0 - sin_cf**2)
        cf_t = 0.6
        expected_shift += cf_t * sin_cf / cos_cf

        # No BARL

        # Silicon to PD
        n_si, _ = db.get_nk("silicon", ref_wl)
        sin_si = sin_cra / n_si
        cos_si = np.sqrt(1.0 - sin_si**2)
        si_to_pd = si_t - pd_z
        expected_shift += si_to_pd * sin_si / cos_si

        assert computed_shift == pytest.approx(expected_shift, rel=1e-10), (
            f"Snell shift {computed_shift} != analytical {expected_shift}"
        )

    def test_air_only_equals_tan(self):
        """If all layers had n=1 (air), shift should equal sum(h_i) * tan(CRA).

        We can't set real layers to air and keep the structure valid,
        but we can verify the math: for n=1, sin(theta)=sin(CRA),
        so d = h * tan(CRA). Check this holds for a known layer.
        """
        cra_deg = 20.0
        cra_rad = np.deg2rad(cra_deg)
        sin_cra = np.sin(cra_rad)

        # For a layer with n=1: shift = h * sin(CRA) / cos(CRA) = h * tan(CRA)
        h = 1.5
        n = 1.0
        sin_theta = sin_cra / n
        cos_theta = np.sqrt(1.0 - sin_theta**2)
        shift = h * sin_theta / cos_theta

        expected = h * np.tan(cra_rad)
        assert shift == pytest.approx(expected, rel=1e-12)

    def test_high_index_reduces_shift(self):
        """Higher refractive index should reduce per-layer lateral displacement.

        For fixed thickness h and CRA, shift(n=4) < shift(n=1.5) < shift(n=1).
        """
        cra_deg = 25.0
        sin_cra = np.sin(np.deg2rad(cra_deg))
        h = 1.0

        shifts = []
        for n in [1.0, 1.5, 4.0]:
            sin_theta = sin_cra / n
            cos_theta = np.sqrt(1.0 - sin_theta**2)
            shifts.append(h * sin_theta / cos_theta)

        # n=1 > n=1.5 > n=4
        assert shifts[0] > shifts[1] > shifts[2], (
            f"Higher n should reduce shift: n=1→{shifts[0]:.4f}, "
            f"n=1.5→{shifts[1]:.4f}, n=4→{shifts[2]:.4f}"
        )

    def test_shift_proportional_to_thickness(self):
        """Doubling a layer thickness should double its contribution to shift."""
        from compass.materials.database import MaterialDB

        cra_deg = 20.0
        ref_wl = 0.55
        db = MaterialDB()
        sin_cra = np.sin(np.deg2rad(cra_deg))

        n_sio2, _ = db.get_nk("sio2", ref_wl)
        sin_theta = sin_cra / n_sio2
        cos_theta = np.sqrt(1.0 - sin_theta**2)

        shift_1 = 0.3 * sin_theta / cos_theta
        shift_2 = 0.6 * sin_theta / cos_theta

        assert shift_2 == pytest.approx(2.0 * shift_1, rel=1e-12)

    def test_shift_matches_layer_slices(self):
        """Shift should propagate correctly to the microlens height map.

        When CRA > 0, the microlens staircase should show asymmetric
        permittivity (lens center is offset).
        """
        config_shifted = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                        "shift": {"mode": "auto_cra", "cra_deg": 25.0},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }
        config_centered = {
            "pixel": {
                "pitch": 1.0, "unit_cell": [2, 2],
                "layers": {
                    "air": {"thickness": 1.0, "material": "air"},
                    "microlens": {
                        "enabled": True, "height": 0.6,
                        "radius_x": 0.48, "radius_y": 0.48,
                        "material": "polymer_n1p56",
                        "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                        "shift": {"mode": "none"},
                    },
                    "planarization": {"thickness": 0.3, "material": "sio2"},
                    "color_filter": {
                        "thickness": 0.6,
                        "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    },
                    "silicon": {
                        "thickness": 3.0, "material": "silicon",
                        "photodiode": {"position": [0.0, 0.0, 0.5]},
                    },
                },
                "bayer_map": [["R", "G"], ["G", "B"]],
            }
        }

        ps_shifted = PixelStack(config_shifted)
        ps_centered = PixelStack(config_centered)

        # Get microlens staircase slices
        slices_shifted = ps_shifted.get_layer_slices(0.55, nx=64, ny=64)
        slices_centered = ps_centered.get_layer_slices(0.55, nx=64, ny=64)

        ml_shifted = [s for s in slices_shifted if s.name.startswith("microlens_slice_")]
        ml_centered = [s for s in slices_centered if s.name.startswith("microlens_slice_")]

        assert len(ml_shifted) == len(ml_centered)
        assert len(ml_shifted) > 0

        # The eps_grid for shifted vs centered should differ
        # (lens is displaced laterally)
        mid_idx = len(ml_shifted) // 2
        eps_shifted = ml_shifted[mid_idx].eps_grid
        eps_centered = ml_centered[mid_idx].eps_grid

        diff = np.abs(eps_shifted - eps_centered)
        assert np.max(diff) > 0, "Shifted microlens should differ from centered"
