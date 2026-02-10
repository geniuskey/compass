"""Performance benchmarks for core COMPASS operations.

Times CPU-only operations to track regressions. No GPU or optional solver
packages required.
"""

import time

import numpy as np
import pytest

from compass.geometry.builder import GeometryBuilder
from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB


def make_pixel_config(unit_cell=(2, 2), pitch=1.0):
    """Create a minimal pixel configuration for testing."""
    return {
        "pixel": {
            "pitch": pitch,
            "unit_cell": list(unit_cell),
            "bayer_map": GeometryBuilder.bayer_pattern(unit_cell, "bayer_rggb"),
            "layers": {
                "silicon": {
                    "thickness": 3.0,
                    "material": "silicon",
                    "photodiode": {
                        "position": [0.0, 0.0, 0.5],
                        "size": [0.7, 0.7, 2.0],
                    },
                    "dti": {"enabled": False},
                },
                "barl": {"layers": []},
                "color_filter": {
                    "thickness": 0.6,
                    "pattern": "bayer_rggb",
                    "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    "grid": {"enabled": False},
                },
                "planarization": {
                    "thickness": 0.3,
                    "material": "sio2",
                },
                "microlens": {
                    "enabled": True,
                    "height": 0.6,
                    "radius_x": 0.48,
                    "radius_y": 0.48,
                    "material": "polymer_n1p56",
                    "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                    "shift": {"mode": "manual", "shift_x": 0.0, "shift_y": 0.0},
                },
                "air": {"thickness": 1.0, "material": "air"},
            },
        }
    }


class TestMaterialDBPerformance:
    """Benchmark MaterialDB lookup operations."""

    def test_epsilon_spectrum_41_wavelengths(self):
        """Time MaterialDB.get_epsilon_spectrum for 41 wavelengths."""
        db = MaterialDB()
        wavelengths = np.linspace(0.38, 0.78, 41)

        t0 = time.perf_counter()
        eps = db.get_epsilon_spectrum("silicon", wavelengths)
        elapsed = time.perf_counter() - t0

        print(f"\n  MaterialDB.get_epsilon_spectrum(silicon, 41 wl): {elapsed*1000:.2f} ms")
        assert eps.shape == (41,)
        assert elapsed < 1.0, f"Epsilon spectrum took {elapsed:.3f}s, expected < 1s"

    def test_multiple_materials_spectrum(self):
        """Time epsilon spectrum lookup across all built-in materials."""
        db = MaterialDB()
        wavelengths = np.linspace(0.38, 0.78, 41)
        materials = db.list_materials()

        t0 = time.perf_counter()
        for mat in materials:
            db.get_epsilon_spectrum(mat, wavelengths)
        elapsed = time.perf_counter() - t0

        print(f"\n  All {len(materials)} materials x 41 wl: {elapsed*1000:.2f} ms")
        assert elapsed < 2.0, f"All materials took {elapsed:.3f}s, expected < 2s"

    def test_single_nk_lookup_speed(self):
        """Time 1000 individual get_nk calls."""
        db = MaterialDB()
        wavelengths = np.linspace(0.4, 0.7, 1000)

        t0 = time.perf_counter()
        for wl in wavelengths:
            db.get_nk("silicon", wl)
        elapsed = time.perf_counter() - t0

        print(f"\n  1000 x get_nk(silicon): {elapsed*1000:.2f} ms")
        assert elapsed < 2.0, f"1000 nk lookups took {elapsed:.3f}s, expected < 2s"


class TestPixelStackConstruction:
    """Benchmark PixelStack construction from config."""

    def test_construction_time_2x2(self):
        """Time PixelStack construction for 2x2 unit cell."""
        config = make_pixel_config(unit_cell=(2, 2))

        t0 = time.perf_counter()
        stack = PixelStack(config)
        elapsed = time.perf_counter() - t0

        print(f"\n  PixelStack(2x2) construction: {elapsed*1000:.2f} ms")
        assert len(stack.layers) > 0
        assert elapsed < 1.0, f"Construction took {elapsed:.3f}s, expected < 1s"

    def test_construction_time_4x4(self):
        """Time PixelStack construction for 4x4 unit cell."""
        config = make_pixel_config(unit_cell=(4, 4))

        t0 = time.perf_counter()
        stack = PixelStack(config)
        elapsed = time.perf_counter() - t0

        print(f"\n  PixelStack(4x4) construction: {elapsed*1000:.2f} ms")
        assert len(stack.layers) > 0
        assert elapsed < 2.0, f"Construction took {elapsed:.3f}s, expected < 2s"


class TestLayerSlicePerformance:
    """Benchmark get_layer_slices at various resolutions."""

    @pytest.mark.parametrize("nx", [32, 64, 128])
    def test_layer_slices_timing(self, nx):
        """Time get_layer_slices at resolution nx."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)

        t0 = time.perf_counter()
        slices = stack.get_layer_slices(wavelength=0.550, nx=nx, ny=nx,
                                        n_lens_slices=30)
        elapsed = time.perf_counter() - t0

        print(f"\n  get_layer_slices(nx={nx}): {elapsed*1000:.2f} ms, {len(slices)} slices")
        assert len(slices) > 0
        assert elapsed < 10.0, f"Layer slices at nx={nx} took {elapsed:.3f}s, expected < 10s"


class TestPermittivityGridPerformance:
    """Benchmark get_permittivity_grid at various resolutions."""

    @pytest.mark.parametrize("nx", [32, 64])
    def test_permittivity_grid_timing(self, nx):
        """Time get_permittivity_grid at resolution nx."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)

        t0 = time.perf_counter()
        eps_3d = stack.get_permittivity_grid(wavelength=0.550, nx=nx, ny=nx, nz=64)
        elapsed = time.perf_counter() - t0

        print(f"\n  get_permittivity_grid(nx={nx}, nz=64): {elapsed*1000:.2f} ms")
        assert eps_3d.shape == (nx, nx, 64)
        assert elapsed < 30.0, f"Permittivity grid at nx={nx} took {elapsed:.3f}s, expected < 30s"

    @pytest.mark.slow
    def test_permittivity_grid_high_res(self):
        """Time get_permittivity_grid at high resolution (nx=128)."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)

        t0 = time.perf_counter()
        eps_3d = stack.get_permittivity_grid(wavelength=0.550, nx=128, ny=128, nz=128)
        elapsed = time.perf_counter() - t0

        print(f"\n  get_permittivity_grid(128x128x128): {elapsed*1000:.2f} ms")
        assert eps_3d.shape == (128, 128, 128)
        assert elapsed < 60.0, f"High-res grid took {elapsed:.3f}s, expected < 60s"
