"""Unit tests for GeometryBuilder."""

import numpy as np
import pytest

from compass.geometry.builder import GeometryBuilder


class TestSuperellipseLens:
    """Tests for superellipse microlens generation."""

    def test_basic_shape(self):
        """Lens should be highest at center, zero at edges."""
        x = np.linspace(-0.5, 0.5, 64)
        y = np.linspace(-0.5, 0.5, 64)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        h = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            center_x=0.0,
            center_y=0.0,
            rx=0.4,
            ry=0.4,
            height=0.6,
            n=2.5,
            alpha=1.0,
        )

        # Center should be maximum
        center_idx = 32
        assert h[center_idx, center_idx] == pytest.approx(0.6, abs=0.01)

        # Edges should be zero
        assert h[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_symmetry(self):
        """Symmetric parameters should give symmetric height map."""
        x = np.linspace(-0.5, 0.5, 65)
        y = np.linspace(-0.5, 0.5, 65)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        h = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            center_x=0.0,
            center_y=0.0,
            rx=0.4,
            ry=0.4,
            height=0.6,
            n=2.5,
            alpha=1.0,
        )

        # Should be symmetric in x and y
        np.testing.assert_allclose(h, np.flip(h, axis=0), atol=1e-10)
        np.testing.assert_allclose(h, np.flip(h, axis=1), atol=1e-10)

    def test_shift(self):
        """CRA shift should move the lens center."""
        x = np.linspace(-0.5, 0.5, 65)
        y = np.linspace(-0.5, 0.5, 65)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        h_centered = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            center_x=0.0,
            center_y=0.0,
            rx=0.3,
            ry=0.3,
            height=0.5,
            n=2.0,
            alpha=1.0,
        )
        h_shifted = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            center_x=0.0,
            center_y=0.0,
            rx=0.3,
            ry=0.3,
            height=0.5,
            n=2.0,
            alpha=1.0,
            shift_x=0.1,
        )

        # Max position should be different
        max_centered = np.unravel_index(h_centered.argmax(), h_centered.shape)
        max_shifted = np.unravel_index(h_shifted.argmax(), h_shifted.shape)
        assert max_centered != max_shifted

    def test_non_negative(self):
        """Height should never be negative."""
        x = np.linspace(-1.0, 1.0, 100)
        y = np.linspace(-1.0, 1.0, 100)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        h = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            center_x=0.0,
            center_y=0.0,
            rx=0.4,
            ry=0.4,
            height=0.6,
            n=2.5,
            alpha=1.0,
        )
        assert np.all(h >= 0)

    def test_squareness_parameter(self):
        """Higher n should make the lens more square."""
        x = np.linspace(-0.5, 0.5, 65)
        y = np.linspace(-0.5, 0.5, 65)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        h_round = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            0.0,
            0.0,
            0.4,
            0.4,
            0.6,
            n=2.0,
            alpha=1.0,
        )
        h_square = GeometryBuilder.superellipse_lens(
            xx,
            yy,
            0.0,
            0.0,
            0.4,
            0.4,
            0.6,
            n=10.0,
            alpha=1.0,
        )

        # Squarer lens should have more total volume
        assert np.sum(h_square) > np.sum(h_round)


class TestBayerPattern:
    """Tests for Bayer pattern generation."""

    def test_rggb_2x2(self):
        """Standard RGGB 2x2 pattern."""
        pattern = GeometryBuilder.bayer_pattern((2, 2), "bayer_rggb")
        assert pattern == [["R", "G"], ["G", "B"]]

    def test_rggb_4x4(self):
        """RGGB pattern tiled to 4x4."""
        pattern = GeometryBuilder.bayer_pattern((4, 4), "bayer_rggb")
        assert len(pattern) == 4
        assert len(pattern[0]) == 4
        assert pattern[0] == ["R", "G", "R", "G"]
        assert pattern[1] == ["G", "B", "G", "B"]
        assert pattern[2] == ["R", "G", "R", "G"]
        assert pattern[3] == ["G", "B", "G", "B"]

    def test_grbg(self):
        """GRBG pattern."""
        pattern = GeometryBuilder.bayer_pattern((2, 2), "bayer_grbg")
        assert pattern == [["G", "R"], ["B", "G"]]

    def test_quad_bayer(self):
        """Quad Bayer pattern should be 4x4 base."""
        pattern = GeometryBuilder.bayer_pattern((4, 4), "quad_bayer")
        assert pattern[0][0] == "R"
        assert pattern[0][1] == "R"
        assert pattern[2][2] == "B"

    def test_nonacell_6x6(self):
        """Nonacell: 3x3 same-color groups in 6x6 super-pixel."""
        pattern = GeometryBuilder.bayer_pattern((6, 6), "nonacell")
        assert len(pattern) == 6 and len(pattern[0]) == 6
        # Top-left 3x3 block should all be R, top-right G, bot-left G, bot-right B.
        for r in range(3):
            for c in range(3):
                assert pattern[r][c] == "R"
                assert pattern[r][c + 3] == "G"
                assert pattern[r + 3][c] == "G"
                assert pattern[r + 3][c + 3] == "B"
        flat = [c for row in pattern for c in row]
        assert flat.count("R") == 9
        assert flat.count("G") == 18
        assert flat.count("B") == 9

    def test_tetra2cell_8x8(self):
        """Tetra^2 / Hexadeca: 4x4 same-color groups in 8x8 super-pixel."""
        pattern = GeometryBuilder.bayer_pattern((8, 8), "tetra2cell")
        assert len(pattern) == 8 and len(pattern[0]) == 8
        flat = [c for row in pattern for c in row]
        assert flat.count("R") == 16
        assert flat.count("G") == 32
        assert flat.count("B") == 16
        assert pattern[0][0] == "R"
        assert pattern[0][7] == "G"
        assert pattern[7][7] == "B"

    def test_unknown_pattern_raises(self):
        """Should raise for unknown pattern."""
        with pytest.raises(ValueError):
            GeometryBuilder.bayer_pattern((2, 2), "unknown_pattern")

    def test_color_count(self):
        """2x2 RGGB should have 1R, 2G, 1B."""
        pattern = GeometryBuilder.bayer_pattern((2, 2), "bayer_rggb")
        flat = [c for row in pattern for c in row]
        assert flat.count("R") == 1
        assert flat.count("G") == 2
        assert flat.count("B") == 1


class TestDtiGrid:
    """Tests for DTI grid generation."""

    def test_basic_grid(self):
        """DTI grid should create lines at pixel boundaries."""
        mask = GeometryBuilder.dti_grid(
            nx=100,
            ny=100,
            pitch=1.0,
            unit_cell=(2, 2),
            dti_width=0.1,
        )
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float64

        # Should have DTI at boundaries
        assert np.sum(mask) > 0

    def test_grid_fraction(self):
        """DTI should cover a reasonable fraction of the area."""
        mask = GeometryBuilder.dti_grid(
            nx=200,
            ny=200,
            pitch=1.0,
            unit_cell=(2, 2),
            dti_width=0.1,
        )
        fraction = np.mean(mask)
        # Grid lines cover about 2*2*0.1/(2*1) ≈ 20% minus overlaps
        assert 0.05 < fraction < 0.4, f"DTI fraction {fraction} unexpected"

    def test_zero_width(self):
        """Zero-width DTI should have no lines."""
        mask = GeometryBuilder.dti_grid(
            nx=100,
            ny=100,
            pitch=1.0,
            unit_cell=(2, 2),
            dti_width=0.0,
        )
        assert np.sum(mask) == 0


class TestTrenchGrid:
    """Tests for the explicit-half-width trench grid (tapered DTI)."""

    def test_matches_dti_grid(self):
        """trench_grid(half) must equal dti_grid(width=2*half)."""
        a = GeometryBuilder.trench_grid(
            nx=120,
            ny=120,
            pitch=1.0,
            unit_cell=(2, 2),
            half_width=0.05,
        )
        b = GeometryBuilder.dti_grid(
            nx=120,
            ny=120,
            pitch=1.0,
            unit_cell=(2, 2),
            dti_width=0.10,
        )
        assert np.array_equal(a, b)

    def test_narrower_halfwidth_covers_less(self):
        """A smaller half-width (deeper in a tapered trench) covers less area."""
        wide = GeometryBuilder.trench_grid(
            nx=200,
            ny=200,
            pitch=1.0,
            unit_cell=(2, 2),
            half_width=0.06,
        )
        narrow = GeometryBuilder.trench_grid(
            nx=200,
            ny=200,
            pitch=1.0,
            unit_cell=(2, 2),
            half_width=0.02,
        )
        assert np.mean(narrow) < np.mean(wide)

    def test_nonpositive_halfwidth_empty(self):
        mask = GeometryBuilder.trench_grid(
            nx=64,
            ny=64,
            pitch=1.0,
            unit_cell=(2, 2),
            half_width=0.0,
        )
        assert np.sum(mask) == 0


class TestInvertedPyramidMask:
    """Tests for the inverted-pyramid (light-trapping texture) mask."""

    def test_full_width_at_surface(self):
        """At the surface (half_width = period/2) pits tessellate the plane."""
        mask = GeometryBuilder.inverted_pyramid_mask(
            nx=200,
            ny=200,
            lx=2.0,
            ly=2.0,
            period=1.0,
            half_width=0.5,
        )
        assert np.mean(mask) > 0.95

    def test_shrinks_toward_apex(self):
        """A smaller half-width (closer to the apex) covers less area."""
        wide = GeometryBuilder.inverted_pyramid_mask(
            nx=200,
            ny=200,
            lx=2.0,
            ly=2.0,
            period=1.0,
            half_width=0.4,
        )
        narrow = GeometryBuilder.inverted_pyramid_mask(
            nx=200,
            ny=200,
            lx=2.0,
            ly=2.0,
            period=1.0,
            half_width=0.1,
        )
        assert np.mean(narrow) < np.mean(wide)

    def test_apex_empty(self):
        mask = GeometryBuilder.inverted_pyramid_mask(
            nx=64,
            ny=64,
            lx=2.0,
            ly=2.0,
            period=1.0,
            half_width=0.0,
        )
        assert np.sum(mask) == 0


class TestMetalGridRounded:
    """Tests for the rounded-rectangle metal grid (corner_radius > 0)."""

    def test_sharp_fallback_matches_dti(self):
        """corner_radius=0 must be identical to dti_grid."""
        sharp = GeometryBuilder.metal_grid(
            nx=64,
            ny=64,
            pitch=1.0,
            unit_cell=(2, 2),
            grid_width=0.05,
        )
        dti = GeometryBuilder.dti_grid(
            nx=64,
            ny=64,
            pitch=1.0,
            unit_cell=(2, 2),
            dti_width=0.05,
        )
        assert np.array_equal(sharp, dti)

    def test_rounded_has_more_metal_than_sharp(self):
        """Rounding the CF corners must increase the metal area."""
        sharp = GeometryBuilder.metal_grid(
            nx=128,
            ny=128,
            pitch=1.0,
            unit_cell=(2, 2),
            grid_width=0.05,
        )
        rounded = GeometryBuilder.metal_grid(
            nx=128,
            ny=128,
            pitch=1.0,
            unit_cell=(2, 2),
            grid_width=0.05,
            corner_radius=0.1,
        )
        assert rounded.mean() > sharp.mean()

    def test_metal_area_monotonic_in_radius(self):
        """Metal fraction should grow monotonically with corner_radius."""
        fractions = []
        for r in [0.0, 0.05, 0.1, 0.2, 0.3]:
            mask = GeometryBuilder.metal_grid(
                nx=128,
                ny=128,
                pitch=1.0,
                unit_cell=(2, 2),
                grid_width=0.05,
                corner_radius=r,
            )
            fractions.append(mask.mean())
        for a, b in zip(fractions, fractions[1:]):
            assert b >= a - 1e-12

    def test_per_pixel_symmetry(self):
        """Each pixel's CF should be symmetric in x and y about its center."""
        mask = GeometryBuilder.metal_grid(
            nx=64,
            ny=64,
            pitch=1.0,
            unit_cell=(1, 1),
            grid_width=0.05,
            corner_radius=0.15,
        )
        assert np.array_equal(mask, mask[::-1, :])
        assert np.array_equal(mask, mask[:, ::-1])

    def test_radius_clamped_to_inner_half(self):
        """Oversized r must clamp; CF becomes the inscribed circle."""
        nx = ny = 256
        pitch = 1.0
        gw = 0.05
        inner_half = (pitch - gw) / 2.0
        mask = GeometryBuilder.metal_grid(
            nx=nx,
            ny=ny,
            pitch=pitch,
            unit_cell=(1, 1),
            grid_width=gw,
            corner_radius=10.0,
        )
        # CF area ≈ π r²; metal area ≈ pitch² - π r²
        cf_fraction = 1.0 - mask.mean()
        expected = np.pi * inner_half**2 / (pitch * pitch)
        assert abs(cf_fraction - expected) < 0.01

    def test_mask_dtype_and_range(self):
        """Mask must be float64 with values in {0, 1}."""
        mask = GeometryBuilder.metal_grid(
            nx=32,
            ny=32,
            pitch=1.0,
            unit_cell=(2, 2),
            grid_width=0.05,
            corner_radius=0.1,
        )
        assert mask.dtype == np.float64
        unique = np.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})
