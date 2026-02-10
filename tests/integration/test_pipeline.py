"""Integration tests for the full simulation pipeline."""

import numpy as np
import pytest

from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.sources.planewave import PlanewaveSource


@pytest.fixture
def simple_config():
    """Minimal config for fast integration testing."""
    return {
        "pixel": {
            "pitch": 1.0,
            "unit_cell": [2, 2],
            "layers": {
                "air": {"thickness": 0.5, "material": "air"},
                "microlens": {
                    "enabled": True,
                    "height": 0.4,
                    "radius_x": 0.4,
                    "radius_y": 0.4,
                    "material": "polymer_n1p56",
                    "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                    "shift": {"mode": "none"},
                },
                "planarization": {"thickness": 0.2, "material": "sio2"},
                "color_filter": {
                    "thickness": 0.5,
                    "pattern": "bayer_rggb",
                    "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                    "grid": {"enabled": False},
                },
                "barl": {"layers": []},
                "silicon": {
                    "thickness": 2.0,
                    "material": "silicon",
                    "photodiode": {"position": [0.0, 0.0, 0.3], "size": [0.6, 0.6, 1.5]},
                    "dti": {"enabled": False},
                },
            },
            "bayer_map": [["R", "G"], ["G", "B"]],
        }
    }


class TestConfigToStackPipeline:
    """Test config → PixelStack → layer slices pipeline."""

    def test_config_to_pixel_stack(self, simple_config):
        """Config should build a valid PixelStack."""
        ps = PixelStack(simple_config)
        assert ps.pitch == 1.0
        assert ps.unit_cell == (2, 2)
        assert len(ps.layers) > 0

    def test_pixel_stack_to_slices(self, simple_config):
        """PixelStack should produce valid layer slices."""
        ps = PixelStack(simple_config)
        slices = ps.get_layer_slices(wavelength=0.55, nx=32, ny=32)
        assert len(slices) > 0

        for s in slices:
            assert s.eps_grid.shape == (32, 32)
            assert s.thickness > 0
            assert np.all(np.isfinite(s.eps_grid))

    def test_pixel_stack_to_3d_grid(self, simple_config):
        """PixelStack should produce valid 3D permittivity grid."""
        ps = PixelStack(simple_config)
        eps = ps.get_permittivity_grid(wavelength=0.55, nx=16, ny=16, nz=32)
        assert eps.shape == (16, 16, 32)
        assert np.all(np.isfinite(eps))

    def test_multiple_wavelengths(self, simple_config):
        """Pipeline should work for multiple wavelengths."""
        ps = PixelStack(simple_config)
        for wl in [0.45, 0.55, 0.65]:
            slices = ps.get_layer_slices(wavelength=wl, nx=16, ny=16)
            assert len(slices) > 0
            # Si permittivity should change with wavelength
            si_slices = [s for s in slices if s.name == "silicon"]
            assert len(si_slices) == 1

    def test_planewave_creation(self):
        """Should create planewave source from config."""
        config = {
            "wavelength": {"mode": "sweep", "sweep": {"start": 0.45, "stop": 0.65, "step": 0.05}},
            "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
            "polarization": "unpolarized",
        }
        src = PlanewaveSource.from_config(config)
        assert src.n_wavelengths == 5  # 0.45, 0.50, 0.55, 0.60, 0.65

    def test_material_db_wavelength_range(self):
        """MaterialDB should work across full visible range."""
        db = MaterialDB()
        for wl in np.arange(0.38, 0.79, 0.05):
            for mat in ["silicon", "sio2", "air", "polymer_n1p56"]:
                n, k = db.get_nk(mat, wl)
                assert np.isfinite(n), f"n not finite for {mat} at {wl}"
                assert np.isfinite(k), f"k not finite for {mat} at {wl}"
                assert n > 0, f"n <= 0 for {mat} at {wl}"
                assert k >= 0, f"k < 0 for {mat} at {wl}"
