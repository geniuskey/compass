"""Parametric geometry generation for COMPASS pixel structures."""

from __future__ import annotations

import numpy as np


class GeometryBuilder:
    """Parametric geometry generation for all solver types."""

    @staticmethod
    def superellipse_lens(
        x: np.ndarray,
        y: np.ndarray,
        center_x: float,
        center_y: float,
        rx: float,
        ry: float,
        height: float,
        n: float = 2.5,
        alpha: float = 1.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> np.ndarray:
        """Generate superellipse microlens height map.

        z(x,y) = h * (1 - r^2)^(1/(2*alpha))
        where r = (|x/rx|^n + |y/ry|^n)^(1/n)

        Args:
            x: 2D x-coordinate array.
            y: 2D y-coordinate array.
            center_x: Lens center x.
            center_y: Lens center y.
            rx: Semi-axis in x direction (um).
            ry: Semi-axis in y direction (um).
            height: Lens height (um).
            n: Superellipse squareness parameter (2=ellipse, >2=squarer).
            alpha: Curvature parameter.
            shift_x: CRA offset in x (um).
            shift_y: CRA offset in y (um).

        Returns:
            2D height map array.
        """
        dx = x - (center_x + shift_x)
        dy = y - (center_y + shift_y)

        # Avoid division by zero
        rx_safe = max(rx, 1e-10)
        ry_safe = max(ry, 1e-10)

        r = (np.abs(dx / rx_safe) ** n + np.abs(dy / ry_safe) ** n) ** (1.0 / n)
        r = np.clip(r, 0, 1.0 + 1e-10)

        z = np.zeros_like(r)
        mask = r <= 1.0
        z[mask] = height * (1.0 - r[mask] ** 2) ** (1.0 / (2.0 * alpha))

        return np.asarray(z)

    @staticmethod
    def bayer_pattern(unit_cell: tuple[int, int], pattern: str = "bayer_rggb") -> list[list[str]]:
        """Generate Bayer color filter map for NxM unit cell.

        Args:
            unit_cell: (rows, cols) in unit cell.
            pattern: Pattern type (bayer_rggb, bayer_grbg, bayer_gbrg, bayer_bggr, quad_bayer).

        Returns:
            2D list of color strings.
        """

        def _binned(group: int) -> list[list[str]]:
            tile_rggb = [["R", "G"], ["G", "B"]]
            size = 2 * group
            out = [["" for _ in range(size)] for _ in range(size)]
            for br in range(2):
                for bc in range(2):
                    color = tile_rggb[br][bc]
                    for r in range(group):
                        for c in range(group):
                            out[br * group + r][bc * group + c] = color
            return out

        base_patterns = {
            "bayer_rggb": [["R", "G"], ["G", "B"]],
            "bayer_grbg": [["G", "R"], ["B", "G"]],
            "bayer_gbrg": [["G", "B"], ["R", "G"]],
            "bayer_bggr": [["B", "G"], ["G", "R"]],
            # Quad Bayer (Tetracell): 2x2 same-color groups, 4x4 unit cell
            "quad_bayer": _binned(2),
            "tetracell": _binned(2),
            # Nonacell: 3x3 same-color groups, 6x6 unit cell (Samsung 108MP HM1)
            "nonacell": _binned(3),
            # Hexadeca / Tetra^2 / Tetra2pixel: 4x4 same-color groups, 8x8 unit cell (Samsung HP9)
            "hexadeca": _binned(4),
            "tetra2cell": _binned(4),
        }

        # Strip "bayer_" prefix if not in mapping
        pattern_key = pattern.lower()
        if pattern_key not in base_patterns:
            pattern_key = f"bayer_{pattern_key}"

        if pattern_key not in base_patterns:
            raise ValueError(
                f"Unknown Bayer pattern: {pattern}. Available: {list(base_patterns.keys())}"
            )

        tile = base_patterns[pattern_key]
        tile_rows = len(tile)
        tile_cols = len(tile[0])
        rows, cols = unit_cell

        full = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(tile[r % tile_rows][c % tile_cols])
            full.append(row)
        return full

    @staticmethod
    def dti_grid(
        nx: int,
        ny: int,
        pitch: float,
        unit_cell: tuple[int, int],
        dti_width: float,
    ) -> np.ndarray:
        """Generate DTI (Deep Trench Isolation) grid pattern as 2D binary mask.

        DTI forms a grid at pixel boundaries to optically isolate pixels.

        Args:
            nx: Grid resolution in x.
            ny: Grid resolution in y.
            pitch: Pixel pitch in um.
            unit_cell: (rows, cols) number of pixels.
            dti_width: DTI trench width in um.

        Returns:
            2D binary mask (1 = DTI, 0 = silicon).
        """
        rows, cols = unit_cell
        lx = pitch * cols
        ly = pitch * rows

        # Cell-centered sampling keeps discretized masks mirror-symmetric
        # about pixel centers (a global half-cell shift is irrelevant for
        # periodic structures but symmetry of each pixel is preserved).
        x = (np.arange(nx) + 0.5) * (lx / nx)
        y = (np.arange(ny) + 0.5) * (ly / ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        mask = np.zeros((ny, nx), dtype=bool)
        half_w = dti_width / 2.0

        # Vertical lines at pixel boundaries
        for c in range(cols + 1):
            x_pos = c * pitch
            mask = mask | (np.abs(xx - x_pos) < half_w)
            # Handle periodicity
            if c == 0:
                mask = mask | (np.abs(xx - lx) < half_w)

        # Horizontal lines at pixel boundaries
        for r in range(rows + 1):
            y_pos = r * pitch
            mask = mask | (np.abs(yy - y_pos) < half_w)
            if r == 0:
                mask = mask | (np.abs(yy - ly) < half_w)

        return mask.astype(np.float64)

    @staticmethod
    def trench_grid(
        nx: int,
        ny: int,
        pitch: float,
        unit_cell: tuple[int, int],
        half_width: float,
    ) -> np.ndarray:
        """Generate a pixel-boundary trench grid for an explicit half-width.

        Like :meth:`dti_grid` but parametrised by the trench *half-width* so
        that tapered (depth-dependent) and lined trenches can be built slice by
        slice. A non-positive ``half_width`` returns an all-zero mask.

        Args:
            nx: Grid resolution in x.
            ny: Grid resolution in y.
            pitch: Pixel pitch in um.
            unit_cell: (rows, cols) number of pixels.
            half_width: Half of the trench width in um at this z-slice.

        Returns:
            2D binary mask (1 = trench, 0 = silicon).
        """
        if half_width <= 0.0:
            return np.zeros((ny, nx), dtype=np.float64)
        return GeometryBuilder.dti_grid(nx, ny, pitch, unit_cell, 2.0 * half_width)

    @staticmethod
    def inverted_pyramid_mask(
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        period: float,
        half_width: float,
    ) -> np.ndarray:
        """Generate an inverted-pyramid (pit) array mask at one z-slice.

        Models a tessellated array of square pyramidal pits with the given
        ``period``. ``half_width`` is the pit half-width at the current depth;
        it shrinks linearly from ``period/2`` at the surface to 0 at the apex,
        producing the graded silicon fill fraction of a light-trapping texture.

        Args:
            nx: Grid resolution in x.
            ny: Grid resolution in y.
            lx: Domain size in x (um).
            ly: Domain size in y (um).
            period: Pyramid array period in um.
            half_width: Pit half-width at this z-slice in um.

        Returns:
            2D binary mask (1 = pit/fill, 0 = silicon).
        """
        if half_width <= 0.0 or period <= 0.0:
            return np.zeros((ny, nx), dtype=np.float64)

        # Cell-centered sampling keeps discretized masks mirror-symmetric
        # about pixel centers (a global half-cell shift is irrelevant for
        # periodic structures but symmetry of each pixel is preserved).
        x = (np.arange(nx) + 0.5) * (lx / nx)
        y = (np.arange(ny) + 0.5) * (ly / ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        # Distance to the nearest pyramid centre on the periodic lattice.
        dx = np.abs((xx % period) - period / 2.0)
        dy = np.abs((yy % period) - period / 2.0)
        mask = (dx <= half_width) & (dy <= half_width)
        return np.asarray(mask, dtype=np.float64)

    @staticmethod
    def metal_grid(
        nx: int,
        ny: int,
        pitch: float,
        unit_cell: tuple[int, int],
        grid_width: float,
        corner_radius: float = 0.0,
    ) -> np.ndarray:
        """Generate metal grid pattern between color filters.

        When ``corner_radius`` is 0, this delegates to :meth:`dti_grid` and
        produces a sharp-cornered grid (lines along pixel boundaries).
        When ``corner_radius > 0``, each pixel's CF region is modeled as a
        rounded rectangle with the same radius ``r`` at all four corners,
        and the metal grid is the complement within the unit cell.

        Args:
            nx: Grid resolution in x.
            ny: Grid resolution in y.
            pitch: Pixel pitch in um.
            unit_cell: (rows, cols) number of pixels.
            grid_width: Metal grid width in um (gap between adjacent CF rects).
            corner_radius: Corner rounding radius r (um) applied identically
                to all four corners of each CF rounded rectangle.

        Returns:
            2D mask (1 = metal, 0 = color filter).
        """
        if corner_radius <= 0.0:
            return GeometryBuilder.dti_grid(nx, ny, pitch, unit_cell, grid_width)

        rows, cols = unit_cell
        lx = pitch * cols
        ly = pitch * rows

        # Cell-centered sampling keeps discretized masks mirror-symmetric
        # about pixel centers (a global half-cell shift is irrelevant for
        # periodic structures but symmetry of each pixel is preserved).
        x = (np.arange(nx) + 0.5) * (lx / nx)
        y = (np.arange(ny) + 0.5) * (ly / ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        inner_half = max((pitch - grid_width) / 2.0, 0.0)
        r = min(corner_radius, inner_half)

        inside_any = np.zeros((ny, nx), dtype=bool)
        for ri in range(rows):
            for ci in range(cols):
                cx = (ci + 0.5) * pitch
                cy = (ri + 0.5) * pitch
                dx = np.abs(xx - cx)
                dy = np.abs(yy - cy)
                ex = np.maximum(dx - (inner_half - r), 0.0)
                ey = np.maximum(dy - (inner_half - r), 0.0)
                inside_any |= (dx <= inner_half) & (dy <= inner_half) & (ex * ex + ey * ey <= r * r)

        return (~inside_any).astype(np.float64)

    @staticmethod
    def photodiode_mask_3d(
        nx: int,
        ny: int,
        nz: int,
        pitch: float,
        unit_cell: tuple[int, int],
        pd_position: tuple[float, float, float],
        pd_size: tuple[float, float, float],
        si_z_start: float,
        si_z_end: float,
        bayer_map: list[list[str]],
    ) -> tuple[np.ndarray, dict]:
        """Generate 3D photodiode mask and per-pixel masks.

        Args:
            nx, ny, nz: Grid resolution.
            pitch: Pixel pitch in um.
            unit_cell: (rows, cols).
            pd_position: Photodiode center offset (x, y, z) relative to pixel center.
            pd_size: Photodiode size (dx, dy, dz) in um.
            si_z_start: Silicon bottom z.
            si_z_end: Silicon top z.
            bayer_map: 2D color assignment.

        Returns:
            Tuple of (full 3D mask, dict of per-pixel masks keyed by "{color}_{row}_{col}").
        """
        rows, cols = unit_cell
        lx = pitch * cols
        ly = pitch * rows

        # Cell-centered sampling keeps discretized masks mirror-symmetric
        # about pixel centers (a global half-cell shift is irrelevant for
        # periodic structures but symmetry of each pixel is preserved).
        x = (np.arange(nx) + 0.5) * (lx / nx)
        y = (np.arange(ny) + 0.5) * (ly / ny)
        z = np.linspace(si_z_start, si_z_end, nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")

        full_mask = np.zeros((ny, nx, nz), dtype=bool)
        per_pixel = {}

        px, py, pz = pd_position
        dx, dy, dz = pd_size

        for r in range(rows):
            for c in range(cols):
                cx = (c + 0.5) * pitch + px
                cy = (r + 0.5) * pitch + py
                cz = si_z_end - pz  # z from Si top

                pixel_mask = (
                    (np.abs(xx - cx) < dx / 2.0)
                    & (np.abs(yy - cy) < dy / 2.0)
                    & (np.abs(zz - cz) < dz / 2.0)
                )

                full_mask = full_mask | pixel_mask
                color = bayer_map[r][c]
                per_pixel[f"{color}_{r}_{c}"] = pixel_mask.astype(np.float64)

        return full_mask.astype(np.float64), per_pixel
