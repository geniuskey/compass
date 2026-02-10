"""Benchmark tests for geometry convergence behavior.

Tests that the PixelStack geometry builder produces consistent and convergent
results at different resolutions and discretization levels.
"""

import numpy as np
import pytest

from compass.geometry.builder import GeometryBuilder
from compass.geometry.pixel_stack import PixelStack


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


class TestMicrolensStaircaseConvergence:
    """Verify that microlens volume converges as staircase resolution increases."""

    def test_volume_converges_with_n_slices(self):
        """Microlens volume should converge as n_lens_slices increases."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)
        wavelength = 0.550

        volumes = []
        slice_counts = [5, 10, 20, 40, 80]

        for n_slices in slice_counts:
            slices = stack.get_layer_slices(
                wavelength=wavelength, nx=64, ny=64, n_lens_slices=n_slices
            )
            # Sum up lens material volume across microlens slices
            lens_eps = stack.material_db.get_epsilon("polymer_n1p56", wavelength)
            air_eps = stack.material_db.get_epsilon("air", wavelength)
            lx, ly = stack.domain_size

            total_lens_vol = 0.0
            for s in slices:
                if s.name.startswith("microlens_slice_"):
                    # Count pixels filled with lens material
                    lens_mask = np.abs(s.eps_grid - lens_eps) < np.abs(s.eps_grid - air_eps)
                    fill_fraction = np.mean(lens_mask)
                    slice_vol = fill_fraction * lx * ly * s.thickness
                    total_lens_vol += slice_vol

            volumes.append(total_lens_vol)

        # Volumes should converge: differences between consecutive levels decrease
        diffs = [abs(volumes[i + 1] - volumes[i]) for i in range(len(volumes) - 1)]
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] < diffs[i] + 1e-10, (
                f"Volume not converging: diff[{i}]={diffs[i]:.6f}, diff[{i+1}]={diffs[i+1]:.6f}"
            )

        # Fine and coarse volumes should agree within 10%
        assert volumes[-1] == pytest.approx(volumes[0], rel=0.10), (
            f"Volume mismatch: n=5 gives {volumes[0]:.4f}, n=80 gives {volumes[-1]:.4f}"
        )

    def test_n_slices_matches_output_count(self):
        """Number of microlens slices should match n_lens_slices parameter."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)

        for n_slices in [5, 10, 30]:
            slices = stack.get_layer_slices(wavelength=0.550, nx=32, ny=32,
                                            n_lens_slices=n_slices)
            ml_slices = [s for s in slices if s.name.startswith("microlens_slice_")]
            assert len(ml_slices) == n_slices


class TestLayerSliceConsistency:
    """Verify that layer slices produce consistent results at different resolutions."""

    def test_uniform_layer_resolution_independent(self):
        """Uniform layers should have identical permittivity at any resolution."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)
        wavelength = 0.550

        slices_lo = stack.get_layer_slices(wavelength=wavelength, nx=32, ny=32)
        slices_hi = stack.get_layer_slices(wavelength=wavelength, nx=128, ny=128)

        # Find uniform layers (same name, not microlens or color_filter)
        uniform_names = {"planarization", "air"}
        for name in uniform_names:
            lo = [s for s in slices_lo if s.name == name]
            hi = [s for s in slices_hi if s.name == name]
            assert len(lo) == 1 and len(hi) == 1, f"Missing layer {name}"

            # All values should be identical
            assert np.allclose(lo[0].eps_grid, lo[0].eps_grid[0, 0]), (
                f"Uniform layer {name} not constant at nx=32"
            )
            assert np.allclose(hi[0].eps_grid, hi[0].eps_grid[0, 0]), (
                f"Uniform layer {name} not constant at nx=128"
            )

            # The permittivity value should be the same regardless of resolution
            assert lo[0].eps_grid[0, 0] == pytest.approx(hi[0].eps_grid[0, 0], rel=1e-10)

    def test_z_coordinates_consistent(self):
        """Layer z-coordinates should be identical regardless of resolution."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)
        wavelength = 0.550

        slices_lo = stack.get_layer_slices(wavelength=wavelength, nx=32, ny=32)
        slices_hi = stack.get_layer_slices(wavelength=wavelength, nx=128, ny=128)

        assert len(slices_lo) == len(slices_hi)
        for lo, hi in zip(slices_lo, slices_hi):
            assert lo.z_start == pytest.approx(hi.z_start, abs=1e-12)
            assert lo.z_end == pytest.approx(hi.z_end, abs=1e-12)
            assert lo.thickness == pytest.approx(hi.thickness, abs=1e-12)

    def test_total_height_correct(self):
        """Total stack height should match sum of layer thicknesses."""
        config = make_pixel_config(unit_cell=(2, 2))
        stack = PixelStack(config)
        expected_total = 3.0 + 0.6 + 0.3 + 0.6 + 1.0  # si + cf + plan + ml + air
        assert stack.total_height == pytest.approx(expected_total, abs=1e-10)


class TestBayerPatternTiling:
    """Verify that Bayer patterns tile correctly at different unit cell sizes."""

    @pytest.mark.parametrize("unit_cell", [(2, 2), (4, 4), (8, 8)])
    def test_rggb_pattern_tiling(self, unit_cell):
        """RGGB Bayer pattern should tile correctly for NxN unit cells."""
        pattern = GeometryBuilder.bayer_pattern(unit_cell, "bayer_rggb")

        assert len(pattern) == unit_cell[0]
        assert len(pattern[0]) == unit_cell[1]

        # Check that the base 2x2 pattern repeats correctly
        for r in range(unit_cell[0]):
            for c in range(unit_cell[1]):
                expected_base = [["R", "G"], ["G", "B"]]
                expected = expected_base[r % 2][c % 2]
                assert pattern[r][c] == expected, (
                    f"Mismatch at ({r},{c}): got {pattern[r][c]}, expected {expected}"
                )

    @pytest.mark.parametrize("unit_cell", [(2, 2), (4, 4)])
    def test_color_count_balance(self, unit_cell):
        """For even unit cells, G should appear twice as often as R or B."""
        pattern = GeometryBuilder.bayer_pattern(unit_cell, "bayer_rggb")
        flat = [c for row in pattern for c in row]
        n_total = len(flat)
        n_R = flat.count("R")
        n_G = flat.count("G")
        n_B = flat.count("B")

        assert n_R + n_G + n_B == n_total
        assert n_R == n_B, "R and B should have equal count"
        assert n_G == 2 * n_R, "G should appear twice as often as R"

    def test_quad_bayer_pattern(self):
        """Quad Bayer 4x4 base pattern should tile correctly."""
        pattern = GeometryBuilder.bayer_pattern((4, 4), "quad_bayer")
        expected = [
            ["R", "R", "G", "G"],
            ["R", "R", "G", "G"],
            ["G", "G", "B", "B"],
            ["G", "G", "B", "B"],
        ]
        assert pattern == expected

    def test_invalid_pattern_raises(self):
        """Unknown pattern name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown Bayer pattern"):
            GeometryBuilder.bayer_pattern((2, 2), "unknown_pattern")

    @pytest.mark.parametrize("unit_cell", [(2, 2), (4, 4), (8, 8)])
    def test_bayer_in_pixel_stack(self, unit_cell):
        """PixelStack should correctly assign color filter from Bayer pattern."""
        config = make_pixel_config(unit_cell=unit_cell)
        stack = PixelStack(config)

        assert len(stack.bayer_map) == unit_cell[0]
        assert len(stack.bayer_map[0]) == unit_cell[1]
        assert len(stack.photodiodes) == unit_cell[0] * unit_cell[1]

        # Each photodiode should have a valid color
        for pd in stack.photodiodes:
            assert pd.color in ("R", "G", "B")
