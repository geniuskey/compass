"""Solver-agnostic pixel stack representation.

Constructs the full 3D pixel structure from YAML configuration,
producing both layer-slice output (for RCWA) and voxel-grid output (for FDTD).
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import Layer, LayerSlice, MicrolensSpec, PhotodiodeSpec
from compass.core.units import deg_to_rad
from compass.geometry.builder import GeometryBuilder
from compass.materials.database import MaterialDB

logger = logging.getLogger(__name__)


class PixelStack:
    """Solver-agnostic pixel stack representation.

    Constructs the complete pixel structure from configuration dictionary,
    with methods to generate RCWA layer slices or FDTD 3D permittivity grids.
    """

    def __init__(self, config: dict, material_db: MaterialDB | None = None):
        """Initialize PixelStack from config dict.

        Args:
            config: Configuration dictionary with 'pixel' key.
            material_db: Material database instance (created if None).
        """
        pixel_cfg = config.get("pixel", config)

        self.pitch: float = pixel_cfg["pitch"]
        self.unit_cell: tuple[int, int] = tuple(pixel_cfg["unit_cell"])
        self.material_db = material_db or MaterialDB()

        self.layers: list[Layer] = []
        self.microlenses: list[MicrolensSpec] = []
        self.photodiodes: list[PhotodiodeSpec] = []
        self.bayer_map: list[list[str]] = []

        self._layer_configs: dict = pixel_cfg.get("layers", {})
        self._build_from_config(pixel_cfg)

    @property
    def domain_size(self) -> tuple[float, float]:
        """(Lx, Ly) simulation domain size in um."""
        return (self.pitch * self.unit_cell[1], self.pitch * self.unit_cell[0])

    @property
    def total_height(self) -> float:
        """Total stack height in um."""
        if not self.layers:
            return 0.0
        return self.layers[-1].z_end - self.layers[0].z_start

    @property
    def z_range(self) -> tuple[float, float]:
        """(z_min, z_max) of the stack."""
        if not self.layers:
            return (0.0, 0.0)
        return (self.layers[0].z_start, self.layers[-1].z_end)

    def _build_from_config(self, pixel_cfg: dict) -> None:
        """Construct full stack from YAML parameters.

        Layer order (bottom to top): silicon, BARL, color_filter, planarization, microlens, air
        z=0 is at bottom of silicon.
        """
        layers_cfg = pixel_cfg.get("layers", {})
        z_cursor = 0.0

        # 1. Silicon layer (bottom)
        si_cfg = layers_cfg.get("silicon", {})
        si_thickness = si_cfg.get("thickness", 3.0)
        self.layers.append(Layer(
            name="silicon",
            z_start=z_cursor,
            z_end=z_cursor + si_thickness,
            thickness=si_thickness,
            base_material=si_cfg.get("material", "silicon"),
            is_patterned=si_cfg.get("dti", {}).get("enabled", False),
        ))
        z_cursor += si_thickness

        # Build photodiode specs
        pd_cfg = si_cfg.get("photodiode", {})
        bayer_cfg = pixel_cfg.get("bayer_map", [["R", "G"], ["G", "B"]])
        pattern_type = layers_cfg.get("color_filter", {}).get("pattern", "bayer_rggb")
        self.bayer_map = bayer_cfg if bayer_cfg else GeometryBuilder.bayer_pattern(
            self.unit_cell, pattern_type
        )

        pd_pos = tuple(pd_cfg.get("position", [0.0, 0.0, 0.5]))
        pd_size = tuple(pd_cfg.get("size", [0.7, 0.7, 2.0]))
        for r in range(self.unit_cell[0]):
            for c in range(self.unit_cell[1]):
                color = self.bayer_map[r % len(self.bayer_map)][c % len(self.bayer_map[0])]
                self.photodiodes.append(PhotodiodeSpec(
                    position=pd_pos,
                    size=pd_size,
                    pixel_index=(r, c),
                    color=color,
                ))

        # 2. BARL layers
        barl_cfg = layers_cfg.get("barl", {})
        barl_layers = barl_cfg.get("layers", [])
        for i, bl in enumerate(barl_layers):
            t = bl.get("thickness", 0.01)
            self.layers.append(Layer(
                name=f"barl_{i}",
                z_start=z_cursor,
                z_end=z_cursor + t,
                thickness=t,
                base_material=bl.get("material", "sio2"),
            ))
            z_cursor += t

        # 3. Color filter layer
        cf_cfg = layers_cfg.get("color_filter", {})
        cf_thickness = cf_cfg.get("thickness", 0.6)
        _has_grid = cf_cfg.get("grid", {}).get("enabled", False)
        self.layers.append(Layer(
            name="color_filter",
            z_start=z_cursor,
            z_end=z_cursor + cf_thickness,
            thickness=cf_thickness,
            base_material="cf_green",  # base, actual is patterned
            is_patterned=True,
        ))
        z_cursor += cf_thickness

        # 4. Planarization (over-coat) layer
        plan_cfg = layers_cfg.get("planarization", {})
        plan_thickness = plan_cfg.get("thickness", 0.3)
        self.layers.append(Layer(
            name="planarization",
            z_start=z_cursor,
            z_end=z_cursor + plan_thickness,
            thickness=plan_thickness,
            base_material=plan_cfg.get("material", "sio2"),
        ))
        z_cursor += plan_thickness

        # 5. Microlens layer
        ml_cfg = layers_cfg.get("microlens", {})
        if ml_cfg.get("enabled", True):
            ml_height = ml_cfg.get("height", 0.6)
            profile = ml_cfg.get("profile", {})
            shift_cfg = ml_cfg.get("shift", {})

            # Compute CRA shift
            shift_x, shift_y = 0.0, 0.0
            if shift_cfg.get("mode") == "manual":
                shift_x = shift_cfg.get("shift_x", 0.0)
                shift_y = shift_cfg.get("shift_y", 0.0)
            elif shift_cfg.get("mode") == "auto_cra":
                cra_rad = deg_to_rad(shift_cfg.get("cra_deg", 0.0))
                # shift = tan(CRA) * height_above_PD (approximate)
                shift_x = np.tan(cra_rad) * ml_height * 0.5
                shift_y = 0.0

            self.layers.append(Layer(
                name="microlens",
                z_start=z_cursor,
                z_end=z_cursor + ml_height,
                thickness=ml_height,
                base_material=ml_cfg.get("material", "polymer_n1p56"),
                is_patterned=True,
            ))

            # Create microlens specs for each pixel
            for _r in range(self.unit_cell[0]):
                for _c in range(self.unit_cell[1]):
                    self.microlenses.append(MicrolensSpec(
                        height=ml_height,
                        radius_x=ml_cfg.get("radius_x", 0.48),
                        radius_y=ml_cfg.get("radius_y", 0.48),
                        material=ml_cfg.get("material", "polymer_n1p56"),
                        profile_type=profile.get("type", "superellipse"),
                        n_param=profile.get("n", 2.5),
                        alpha_param=profile.get("alpha", 1.0),
                        shift_x=shift_x,
                        shift_y=shift_y,
                    ))
            z_cursor += ml_height

        # 6. Air layer (top)
        air_cfg = layers_cfg.get("air", {})
        air_thickness = air_cfg.get("thickness", 1.0)
        self.layers.append(Layer(
            name="air",
            z_start=z_cursor,
            z_end=z_cursor + air_thickness,
            thickness=air_thickness,
            base_material="air",
        ))

    def get_layer_slices(
        self,
        wavelength: float,
        nx: int = 128,
        ny: int = 128,
        n_lens_slices: int = 30,
    ) -> list[LayerSlice]:
        """Get z-wise layer decomposition for RCWA solvers.

        Each slice contains a 2D permittivity grid eps(x,y) at that z-level.

        Args:
            wavelength: Wavelength in um for computing permittivity.
            nx: Grid resolution in x.
            ny: Grid resolution in y.
            n_lens_slices: Number of staircase slices for microlens.

        Returns:
            List of LayerSlice from bottom (z_min) to top (z_max).
        """
        slices = []
        _lx, _ly = self.domain_size
        cf_cfg = self._layer_configs.get("color_filter", {})
        si_cfg = self._layer_configs.get("silicon", {})

        for layer in self.layers:
            if layer.name == "microlens":
                # Staircase approximation for microlens
                ml_slices = self._microlens_staircase(
                    layer, wavelength, nx, ny, n_lens_slices
                )
                slices.extend(ml_slices)

            elif layer.name == "color_filter":
                # Patterned layer with Bayer color filter + optional metal grid
                eps_grid = self._build_cf_layer(wavelength, nx, ny, cf_cfg)
                slices.append(LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name="color_filter",
                    material="bayer_pattern",
                ))

            elif layer.name == "silicon":
                # Silicon with optional DTI
                eps_grid = self._build_si_layer(wavelength, nx, ny, si_cfg)
                slices.append(LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name="silicon",
                    material="silicon",
                ))

            else:
                # Uniform layer
                eps = self.material_db.get_epsilon(layer.base_material, wavelength)
                eps_grid = np.full((ny, nx), eps, dtype=complex)
                slices.append(LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name=layer.name,
                    material=layer.base_material,
                ))

        return slices

    def _microlens_staircase(
        self,
        layer: Layer,
        wavelength: float,
        nx: int,
        ny: int,
        n_slices: int,
    ) -> list[LayerSlice]:
        """Generate staircase approximation of microlens layer."""
        lx, ly = self.domain_size
        x = np.linspace(0, lx, nx, endpoint=False)
        y = np.linspace(0, ly, ny, endpoint=False)
        xx, yy = np.meshgrid(x, y, indexing="xy")

        # Build full height map for all lenses in unit cell
        height_map = np.zeros((ny, nx))
        for idx, ml in enumerate(self.microlenses):
            r = idx // self.unit_cell[1]
            c = idx % self.unit_cell[1]
            cx = (c + 0.5) * self.pitch
            cy = (r + 0.5) * self.pitch

            h = GeometryBuilder.superellipse_lens(
                xx, yy,
                center_x=cx, center_y=cy,
                rx=ml.radius_x, ry=ml.radius_y,
                height=ml.height,
                n=ml.n_param, alpha=ml.alpha_param,
                shift_x=ml.shift_x, shift_y=ml.shift_y,
            )
            height_map = np.maximum(height_map, h)

        # Slice the lens into thin layers
        eps_lens = self.material_db.get_epsilon(layer.base_material, wavelength)
        eps_air = self.material_db.get_epsilon("air", wavelength)

        slices = []
        slice_thickness = layer.thickness / n_slices

        for i in range(n_slices):
            z_lo = layer.z_start + i * slice_thickness
            z_hi = z_lo + slice_thickness
            z_mid = (z_lo + z_hi) / 2.0
            rel_z = z_mid - layer.z_start  # height within lens layer

            # Filling fraction: lens material where height_map > rel_z
            eps_grid = np.where(
                height_map > rel_z,
                eps_lens,
                eps_air,
            )

            slices.append(LayerSlice(
                z_start=z_lo,
                z_end=z_hi,
                thickness=slice_thickness,
                eps_grid=eps_grid,
                name=f"microlens_slice_{i}",
                material=layer.base_material,
            ))

        return slices

    def _build_cf_layer(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        cf_cfg: dict,
    ) -> np.ndarray:
        """Build color filter layer with Bayer pattern and optional metal grid."""
        lx, ly = self.domain_size
        eps_grid = np.zeros((ny, nx), dtype=complex)

        # Assign color filter materials per pixel
        cf_materials = cf_cfg.get("materials", {"R": "cf_red", "G": "cf_green", "B": "cf_blue"})
        x = np.linspace(0, lx, nx, endpoint=False)
        y = np.linspace(0, ly, ny, endpoint=False)

        for r in range(self.unit_cell[0]):
            for c in range(self.unit_cell[1]):
                color = self.bayer_map[r][c]
                mat_name = cf_materials.get(color, f"cf_{color.lower()}")
                eps = self.material_db.get_epsilon(mat_name, wavelength)

                # Pixel region
                x_lo = c * self.pitch
                x_hi = (c + 1) * self.pitch
                y_lo = r * self.pitch
                y_hi = (r + 1) * self.pitch

                x_mask = (x >= x_lo) & (x < x_hi)
                y_mask = (y >= y_lo) & (y < y_hi)
                mask_2d = np.outer(y_mask, x_mask)
                eps_grid[mask_2d] = eps

        # Metal grid overlay
        grid_cfg = cf_cfg.get("grid", {})
        if grid_cfg.get("enabled", False):
            grid_width = grid_cfg.get("width", 0.05)
            grid_material = grid_cfg.get("material", "tungsten")
            eps_grid_metal = self.material_db.get_epsilon(grid_material, wavelength)
            metal_mask = GeometryBuilder.metal_grid(
                nx, ny, self.pitch, self.unit_cell, grid_width
            )
            eps_grid[metal_mask > 0.5] = eps_grid_metal

        return eps_grid

    def _build_si_layer(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        si_cfg: dict,
    ) -> np.ndarray:
        """Build silicon layer with optional DTI."""
        eps_si = self.material_db.get_epsilon(
            si_cfg.get("material", "silicon"), wavelength
        )
        eps_grid = np.full((ny, nx), eps_si, dtype=complex)

        dti_cfg = si_cfg.get("dti", {})
        if dti_cfg.get("enabled", False):
            dti_width = dti_cfg.get("width", 0.1)
            dti_material = dti_cfg.get("material", "sio2")
            eps_dti = self.material_db.get_epsilon(dti_material, wavelength)
            dti_mask = GeometryBuilder.dti_grid(
                nx, ny, self.pitch, self.unit_cell, dti_width
            )
            eps_grid[dti_mask > 0.5] = eps_dti

        return eps_grid

    def get_permittivity_grid(
        self,
        wavelength: float,
        nx: int = 64,
        ny: int = 64,
        nz: int = 128,
    ) -> np.ndarray:
        """Generate 3D permittivity distribution for FDTD solvers.

        Args:
            wavelength: Wavelength in um.
            nx, ny, nz: Grid resolution.

        Returns:
            Complex permittivity array of shape (ny, nx, nz).
        """
        z_min, z_max = self.z_range
        z = np.linspace(z_min, z_max, nz)

        # Get all layer slices
        slices = self.get_layer_slices(wavelength, nx, ny)

        eps_3d = np.ones((ny, nx, nz), dtype=complex)

        for s in slices:
            z_mask = (z >= s.z_start) & (z < s.z_end)
            if not np.any(z_mask):
                continue
            # Broadcast 2D eps to matching z slices
            eps_3d[:, :, z_mask] = s.eps_grid[:, :, np.newaxis]

        return eps_3d

    def get_photodiode_mask(
        self,
        nx: int = 64,
        ny: int = 64,
        nz: int = 128,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generate 3D photodiode mask.

        Returns:
            Tuple of (full_mask, per_pixel_masks).
        """
        si_layer = None
        for layer in self.layers:
            if layer.name == "silicon":
                si_layer = layer
                break

        if si_layer is None:
            return np.zeros((ny, nx, nz)), {}

        si_cfg = self._layer_configs.get("silicon", {})
        pd_cfg = si_cfg.get("photodiode", {})

        return GeometryBuilder.photodiode_mask_3d(
            nx, ny, nz,
            self.pitch, self.unit_cell,
            tuple(pd_cfg.get("position", [0.0, 0.0, 0.5])),
            tuple(pd_cfg.get("size", [0.7, 0.7, 2.0])),
            si_layer.z_start, si_layer.z_end,
            self.bayer_map,
        )
