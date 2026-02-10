"""Unit tests for PixelStack."""

import numpy as np
import pytest

from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB


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
