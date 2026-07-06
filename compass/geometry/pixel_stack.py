"""Solver-agnostic pixel stack representation.

Constructs the full 3D pixel structure from YAML configuration,
producing both layer-slice output (for RCWA) and voxel-grid output (for FDTD).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from compass.core.types import Layer, LayerSlice, MicrolensSpec, PhotodiodeSpec
from compass.core.units import deg_to_rad
from compass.geometry.builder import GeometryBuilder
from compass.materials.database import MaterialDB

logger = logging.getLogger(__name__)

_CF_CHANNEL_NAMES = {"R": "red", "G": "green", "B": "blue"}
_DEFAULT_CF_MATERIALS = {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}


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

        # Geometry caches (wavelength-independent, invalidated on config change)
        self._meshgrid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._height_map_cache: dict[tuple[int, int], np.ndarray] = {}
        self._dti_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self._metal_grid_cache: dict[tuple, np.ndarray] = {}

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
        self.layers.append(
            Layer(
                name="silicon",
                z_start=z_cursor,
                z_end=z_cursor + si_thickness,
                thickness=si_thickness,
                base_material=si_cfg.get("material", "silicon"),
                is_patterned=si_cfg.get("dti", {}).get("enabled", False),
            )
        )
        z_cursor += si_thickness

        # Build photodiode specs
        pd_cfg = si_cfg.get("photodiode", {})
        bayer_cfg = pixel_cfg.get("bayer_map", [["R", "G"], ["G", "B"]])
        pattern_type = layers_cfg.get("color_filter", {}).get("pattern", "bayer_rggb")
        self.bayer_map = (
            bayer_cfg if bayer_cfg else GeometryBuilder.bayer_pattern(self.unit_cell, pattern_type)
        )

        pd_pos = tuple(pd_cfg.get("position", [0.0, 0.0, 0.5]))
        pd_size = tuple(pd_cfg.get("size", [0.7, 0.7, 2.0]))
        for r in range(self.unit_cell[0]):
            for c in range(self.unit_cell[1]):
                color = self.bayer_map[r % len(self.bayer_map)][c % len(self.bayer_map[0])]
                self.photodiodes.append(
                    PhotodiodeSpec(
                        position=pd_pos,
                        size=pd_size,
                        pixel_index=(r, c),
                        color=color,
                    )
                )

        # 2. BARL layers
        barl_cfg = layers_cfg.get("barl", {})
        barl_layers = barl_cfg.get("layers", [])
        for i, bl in enumerate(barl_layers):
            t = bl.get("thickness", 0.01)
            self.layers.append(
                Layer(
                    name=f"barl_{i}",
                    z_start=z_cursor,
                    z_end=z_cursor + t,
                    thickness=t,
                    base_material=bl.get("material", "sio2"),
                )
            )
            z_cursor += t

        # 3. Color filter layer
        cf_cfg = layers_cfg.get("color_filter", {})
        cf_thickness = self._color_filter_stack_thickness(cf_cfg)
        self.layers.append(
            Layer(
                name="color_filter",
                z_start=z_cursor,
                z_end=z_cursor + cf_thickness,
                thickness=cf_thickness,
                base_material="cf_green",  # base, actual is patterned
                is_patterned=True,
            )
        )
        z_cursor += cf_thickness

        # 4. Planarization (over-coat) layer
        plan_cfg = layers_cfg.get("planarization", {})
        plan_thickness = plan_cfg.get("thickness", 0.3)
        self.layers.append(
            Layer(
                name="planarization",
                z_start=z_cursor,
                z_end=z_cursor + plan_thickness,
                thickness=plan_thickness,
                base_material=plan_cfg.get("material", "sio2"),
            )
        )
        z_cursor += plan_thickness

        # 5. Microlens layer
        ml_cfg = layers_cfg.get("microlens", {})
        if ml_cfg.get("enabled", True):
            ml_height = ml_cfg.get("height", 0.6)
            ml_base = max(0.0, float(ml_cfg.get("base_thickness", 0.0)))
            self._ml_base_thickness = ml_base
            profile = ml_cfg.get("profile", {})
            shift_cfg = ml_cfg.get("shift", {})

            # Compute CRA shift
            shift_x, shift_y = 0.0, 0.0
            if shift_cfg.get("mode") == "manual":
                shift_x = shift_cfg.get("shift_x", 0.0)
                shift_y = shift_cfg.get("shift_y", 0.0)
            elif shift_cfg.get("mode") == "auto_cra":
                ref_wl = shift_cfg.get("ref_wavelength", 0.55)
                shift_x = self._compute_snell_shift(
                    shift_cfg.get("cra_deg", 0.0), layers_cfg, ref_wl
                )
                shift_y = 0.0

            ml_layer_thickness = ml_height + ml_base
            self.layers.append(
                Layer(
                    name="microlens",
                    z_start=z_cursor,
                    z_end=z_cursor + ml_layer_thickness,
                    thickness=ml_layer_thickness,
                    base_material=ml_cfg.get("material", "polymer_n1p56"),
                    is_patterned=True,
                )
            )

            # Multi-pixel lens sharing (Sony 2x2 OCL, Samsung Hexadeca 4x4 OCL).
            # One microlens covers an N x N group of pixels; default radius
            # auto-scales to N * pitch / 2 when not explicitly set.
            sharing = max(1, int(ml_cfg.get("sharing", 1)))
            default_r = sharing * self.pitch / 2.0
            radius_x = ml_cfg.get("radius_x", default_r)
            radius_y = ml_cfg.get("radius_y", default_r)
            ml_material = ml_cfg.get("material", "polymer_n1p56")

            # Place one MicrolensSpec per N x N group, anchored at group center.
            rows, cols = self.unit_cell
            for _r0 in range(0, rows, sharing):
                for _c0 in range(0, cols, sharing):
                    self.microlenses.append(
                        MicrolensSpec(
                            height=ml_height,
                            radius_x=radius_x,
                            radius_y=radius_y,
                            material=ml_material,
                            profile_type=profile.get("type", "superellipse"),
                            n_param=profile.get("n", 2.5),
                            alpha_param=profile.get("alpha", 1.0),
                            shift_x=shift_x,
                            shift_y=shift_y,
                        )
                    )
            self._lens_sharing = sharing
            z_cursor += ml_layer_thickness
        else:
            self._lens_sharing = 1
            self._ml_base_thickness = 0.0

        # 6. Air layer (top)
        air_cfg = layers_cfg.get("air", {})
        air_thickness = air_cfg.get("thickness", 1.0)
        self.layers.append(
            Layer(
                name="air",
                z_start=z_cursor,
                z_end=z_cursor + air_thickness,
                thickness=air_thickness,
                base_material="air",
            )
        )

    def _compute_snell_shift(
        self,
        cra_deg: float,
        layers_cfg: dict,
        ref_wavelength: float = 0.55,
    ) -> float:
        """Compute microlens CRA shift using Snell's law ray tracing.

        Traces the chief ray through all layers below the microlens
        (planarization, color filter, BARL, silicon to PD center),
        accumulating lateral displacement via Snell's law refraction
        at each interface.

        Based on: J.-H. Hwang and Y. Kim, "A Numerical Method of Aligning
        the Optical Stacks for All Pixels," Sensors, vol. 23, no. 2, 702, 2023.

        Args:
            cra_deg: Chief ray angle in degrees (in air).
            layers_cfg: Layer configuration dictionary.
            ref_wavelength: Reference wavelength in um for refractive index lookup.

        Returns:
            Total lateral shift in um.
        """
        if cra_deg == 0.0:
            return 0.0

        cra_rad = deg_to_rad(cra_deg)
        sin_cra = np.sin(cra_rad)
        n_air = 1.0

        # Collect layers below microlens (top to bottom): planarization, CF, BARL, Si to PD
        layer_entries: list[tuple[float, float]] = []  # (thickness, n_real)

        # Planarization
        plan_cfg = layers_cfg.get("planarization", {})
        plan_t = plan_cfg.get("thickness", 0.3)
        plan_mat = plan_cfg.get("material", "sio2")
        n_plan, _ = self.material_db.get_nk(plan_mat, ref_wavelength)
        layer_entries.append((plan_t, n_plan))

        # Color filter (use cf_green as reference)
        cf_cfg = layers_cfg.get("color_filter", {})
        cf_t = self._color_filter_stack_thickness(cf_cfg)
        cf_ref_material = self._color_filter_spec(cf_cfg, "G")["material"]
        n_cf, _ = self.material_db.get_nk(cf_ref_material, ref_wavelength)
        layer_entries.append((cf_t, n_cf))

        # BARL sub-layers
        barl_cfg = layers_cfg.get("barl", {})
        for bl in barl_cfg.get("layers", []):
            bl_t = bl.get("thickness", 0.01)
            bl_mat = bl.get("material", "sio2")
            n_bl, _ = self.material_db.get_nk(bl_mat, ref_wavelength)
            layer_entries.append((bl_t, n_bl))

        # Silicon down to photodiode center
        si_cfg = layers_cfg.get("silicon", {})
        pd_z = si_cfg.get("photodiode", {}).get("position", [0.0, 0.0, 0.5])
        if isinstance(pd_z, (list, tuple)):
            pd_depth = pd_z[2]
        else:
            pd_depth = 0.5
        si_mat = si_cfg.get("material", "silicon")
        n_si, _ = self.material_db.get_nk(si_mat, ref_wavelength)
        si_thickness = si_cfg.get("thickness", 3.0)
        # PD center is at pd_depth from bottom, so distance from Si top = thickness - pd_depth
        si_to_pd = si_thickness - pd_depth
        layer_entries.append((si_to_pd, n_si))

        # Accumulate lateral displacement: Snell's law at each layer
        total_shift = 0.0
        for thickness, n_layer in layer_entries:
            # Snell's law: n_air * sin(CRA) = n_layer * sin(theta_layer)
            sin_theta = n_air * sin_cra / n_layer
            # Clamp for total internal reflection (shouldn't happen for typical materials)
            sin_theta = min(sin_theta, 1.0)
            cos_theta = np.sqrt(1.0 - sin_theta**2)
            # Lateral displacement: h * tan(theta) = h * sin(theta) / cos(theta)
            if cos_theta > 0:
                total_shift += thickness * sin_theta / cos_theta

        return float(total_shift)

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
                ml_slices = self._microlens_staircase(layer, wavelength, nx, ny, n_lens_slices)
                slices.extend(ml_slices)

            elif layer.name == "color_filter":
                # Patterned layer with Bayer color filter + optional metal grid
                slices.extend(self._build_color_filter_slices(layer, wavelength, nx, ny, cf_cfg))

            elif layer.name == "silicon":
                # Silicon with optional DTI
                slices.extend(self._build_silicon_slices(layer, wavelength, nx, ny, si_cfg))

            else:
                # Uniform layer
                eps = self.material_db.get_epsilon(layer.base_material, wavelength)
                eps_grid = np.full((ny, nx), eps, dtype=complex)
                slices.append(
                    LayerSlice(
                        z_start=layer.z_start,
                        z_end=layer.z_end,
                        thickness=layer.thickness,
                        eps_grid=eps_grid,
                        name=layer.name,
                        material=layer.base_material,
                    )
                )

        return slices

    def _get_meshgrid(self, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
        """Get cached meshgrid for given resolution.

        The meshgrid only depends on domain size and resolution, not wavelength,
        so it can be reused across all wavelength sweeps.
        """
        cache_key = (nx, ny)
        if cache_key not in self._meshgrid_cache:
            lx, ly = self.domain_size
            # Cell-centered sampling (see GeometryBuilder): keeps masks and
            # the microlens height map mirror-symmetric about pixel centers.
            x = (np.arange(nx) + 0.5) * (lx / nx)
            y = (np.arange(ny) + 0.5) * (ly / ny)
            self._meshgrid_cache[cache_key] = np.meshgrid(x, y, indexing="xy")
        return self._meshgrid_cache[cache_key]

    def _get_height_map(self, nx: int, ny: int) -> np.ndarray:
        """Get cached microlens height map for given resolution.

        The height map is purely geometric (no wavelength dependence),
        so it can be computed once and reused for all wavelengths.
        """
        cache_key = (nx, ny)
        if cache_key not in self._height_map_cache:
            xx, yy = self._get_meshgrid(nx, ny)
            height_map = np.zeros((ny, nx))
            sharing = getattr(self, "_lens_sharing", 1)
            n_groups_x = max(1, self.unit_cell[1] // sharing)
            for idx, ml in enumerate(self.microlenses):
                gr = idx // n_groups_x
                gc = idx % n_groups_x
                # Center each shared lens over its sharing x sharing pixel group.
                cx = (gc + 0.5) * sharing * self.pitch
                cy = (gr + 0.5) * sharing * self.pitch

                h = GeometryBuilder.superellipse_lens(
                    xx,
                    yy,
                    center_x=cx,
                    center_y=cy,
                    rx=ml.radius_x,
                    ry=ml.radius_y,
                    height=ml.height,
                    n=ml.n_param,
                    alpha=ml.alpha_param,
                    shift_x=ml.shift_x,
                    shift_y=ml.shift_y,
                )
                height_map = np.maximum(height_map, h)
            self._height_map_cache[cache_key] = height_map
        return self._height_map_cache[cache_key]

    def _microlens_staircase(
        self,
        layer: Layer,
        wavelength: float,
        nx: int,
        ny: int,
        n_slices: int,
    ) -> list[LayerSlice]:
        """Generate staircase approximation of microlens layer."""
        # Use cached height map (geometry doesn't change with wavelength)
        height_map = self._get_height_map(nx, ny)

        # Only the permittivity values depend on wavelength
        eps_lens = self.material_db.get_epsilon(layer.base_material, wavelength)
        eps_air = self.material_db.get_epsilon("air", wavelength)

        slices = []
        slice_thickness = layer.thickness / n_slices
        # A flat residual slab of lens polymer sits under the curved cap.
        base_t = getattr(self, "_ml_base_thickness", 0.0)

        for i in range(n_slices):
            z_lo = layer.z_start + i * slice_thickness
            z_hi = z_lo + slice_thickness
            z_mid = (z_lo + z_hi) / 2.0
            rel_z = z_mid - layer.z_start  # height within lens layer

            # Lens material in the residual base, plus the superellipse cap on
            # top of it: lens where (base + cap_height_map) > rel_z.
            eps_grid = np.where(
                (base_t + height_map) > rel_z,
                eps_lens,
                eps_air,
            )

            slices.append(
                LayerSlice(
                    z_start=z_lo,
                    z_end=z_hi,
                    thickness=slice_thickness,
                    eps_grid=eps_grid,
                    name=f"microlens_slice_{i}",
                    material=layer.base_material,
                )
            )

        return slices

    def _cf_channel_config(self, cf_cfg: dict, color: str) -> dict:
        """Return per-channel color filter config for a Bayer color code."""
        color_code = str(color).upper()
        channel_name = _CF_CHANNEL_NAMES.get(color_code, str(color).lower())
        merged: dict = {}
        for key in (channel_name, color_code):
            raw = cf_cfg.get(key)
            if raw is not None and hasattr(raw, "model_dump"):
                raw = raw.model_dump(exclude_none=True)
            if isinstance(raw, dict):
                merged.update({k: v for k, v in raw.items() if v is not None})
        return merged

    def _has_cf_channel_overrides(self, cf_cfg: dict) -> bool:
        """Whether any color filter channel uses the new per-color schema."""
        return any(self._cf_channel_config(cf_cfg, color) for color in _CF_CHANNEL_NAMES)

    def _color_filter_spec(self, cf_cfg: dict, color: str) -> dict:
        """Resolve material, height, and contact angle for one CF channel."""
        color_code = str(color).upper()
        channel_name = _CF_CHANNEL_NAMES.get(color_code, str(color).lower())

        cf_materials = dict(_DEFAULT_CF_MATERIALS)
        cf_materials.update(cf_cfg.get("materials", {}) or {})

        default_thickness = float(cf_cfg.get("thickness", 0.6))
        default_contact_angle = float(cf_cfg.get("contact_angle", 90.0))
        channel_cfg = self._cf_channel_config(cf_cfg, color_code)

        return {
            "name": channel_name,
            "material": channel_cfg.get(
                "material",
                cf_materials.get(color_code, f"cf_{channel_name}"),
            ),
            "thickness": max(
                0.0,
                float(channel_cfg.get("thickness", default_thickness)),
            ),
            "contact_angle": float(channel_cfg.get("contact_angle", default_contact_angle)),
        }

    def _color_filter_specs(self, cf_cfg: dict) -> dict[str, dict]:
        """Resolve the standard RGB color filter channel specifications."""
        return {color: self._color_filter_spec(cf_cfg, color) for color in _CF_CHANNEL_NAMES}

    def _grid_thickness(self, cf_cfg: dict) -> float:
        """Resolve metal grid thickness, accepting legacy grid.height."""
        grid_cfg = cf_cfg.get("grid", {}) or {}
        if not grid_cfg.get("enabled", False):
            return 0.0

        if grid_cfg.get("thickness") is not None:
            return max(0.0, float(grid_cfg.get("thickness", 0.0)))
        if grid_cfg.get("height") is not None:
            return max(0.0, float(grid_cfg.get("height", 0.0)))
        return max(0.0, float(cf_cfg.get("thickness", 0.6)))

    def _color_filter_stack_thickness(self, cf_cfg: dict) -> float:
        """Return the z-span needed by color filters and their metal grid."""
        grid_t = self._grid_thickness(cf_cfg)
        if self._has_cf_channel_overrides(cf_cfg):
            cf_t = max(spec["thickness"] for spec in self._color_filter_specs(cf_cfg).values())
        else:
            cf_t = max(0.0, float(cf_cfg.get("thickness", 0.6)))
        return float(max(cf_t, grid_t))

    def _uses_color_filter_relief(self, cf_cfg: dict, layer: Layer) -> bool:
        """Whether the CF stack needs z-aware slices instead of one flat slab."""
        specs = self._color_filter_specs(cf_cfg)
        thicknesses = [spec["thickness"] for spec in specs.values()]
        grid_enabled = (cf_cfg.get("grid", {}) or {}).get("enabled", False)
        grid_t = min(self._grid_thickness(cf_cfg), layer.thickness)

        if any(not np.isclose(t, thicknesses[0]) for t in thicknesses[1:]):
            return True
        if not np.isclose(thicknesses[0], layer.thickness):
            return True
        if grid_enabled and not np.isclose(grid_t, layer.thickness):
            return True

        for spec in specs.values():
            contact_angle = float(spec["contact_angle"])
            if contact_angle < 89.999 and spec["thickness"] > grid_t:
                return True

        return False

    def _color_filter_slice_breaks(self, layer: Layer, cf_cfg: dict) -> list[float]:
        """Build local z breakpoints for the color filter relief profile."""
        specs = self._color_filter_specs(cf_cfg)
        grid_t = min(self._grid_thickness(cf_cfg), layer.thickness)
        has_sloped_surface = any(
            float(spec["contact_angle"]) < 89.999 and spec["thickness"] > grid_t
            for spec in specs.values()
        )

        points = {0.0, layer.thickness}
        if 0.0 < grid_t < layer.thickness:
            points.add(grid_t)
        for spec in specs.values():
            t = min(float(spec["thickness"]), layer.thickness)
            if 0.0 < t < layer.thickness:
                points.add(t)

        if has_sloped_surface:
            n_slices = max(1, int(cf_cfg.get("n_slices", 8)))
            points.update(np.linspace(0.0, layer.thickness, n_slices + 1))

        return sorted(points)

    def _build_color_filter_slices(
        self,
        layer: Layer,
        wavelength: float,
        nx: int,
        ny: int,
        cf_cfg: dict,
    ) -> list[LayerSlice]:
        """Build flat or z-aware color filter slices."""
        if not self._uses_color_filter_relief(cf_cfg, layer):
            eps_grid = self._build_cf_layer(wavelength, nx, ny, cf_cfg)
            return [
                LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name="color_filter",
                    material="bayer_pattern",
                )
            ]

        slices = []
        breaks = self._color_filter_slice_breaks(layer, cf_cfg)
        for i, (z0_rel, z1_rel) in enumerate(itertools.pairwise(breaks)):
            thickness = z1_rel - z0_rel
            if thickness <= 0.0:
                continue
            z_mid_rel = 0.5 * (z0_rel + z1_rel)
            eps_grid = self._build_cf_layer_at_z(wavelength, nx, ny, cf_cfg, z_mid_rel)
            slices.append(
                LayerSlice(
                    z_start=layer.z_start + z0_rel,
                    z_end=layer.z_start + z1_rel,
                    thickness=thickness,
                    eps_grid=eps_grid,
                    name=f"color_filter_slice_{i}",
                    material="bayer_pattern",
                )
            )
        return slices

    def _build_cf_layer(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        cf_cfg: dict,
    ) -> np.ndarray:
        """Build color filter layer with Bayer pattern and optional metal grid."""
        return self._build_cf_layer_at_z(wavelength, nx, ny, cf_cfg, z_rel=0.0, flat=True)

    def _build_cf_layer_at_z(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        cf_cfg: dict,
        z_rel: float,
        flat: bool = False,
    ) -> np.ndarray:
        """Build the CF permittivity grid at a relative z height."""
        eps_air = self.material_db.get_epsilon("air", wavelength)
        eps_grid = np.full((ny, nx), eps_air, dtype=np.complex128)

        grid_cfg = cf_cfg.get("grid", {}) or {}
        grid_enabled = grid_cfg.get("enabled", False)
        grid_width = float(grid_cfg.get("width", 0.05)) if grid_enabled else 0.0
        corner_radius = float(grid_cfg.get("corner_radius", 0.0)) if grid_enabled else 0.0
        grid_t = self._grid_thickness(cf_cfg)

        for r in range(self.unit_cell[0]):
            for c in range(self.unit_cell[1]):
                color = self.bayer_map[r][c]
                spec = self._color_filter_spec(cf_cfg, color)
                if not flat and z_rel >= spec["thickness"]:
                    continue
                mat_name = spec["material"]
                eps = self.material_db.get_epsilon(mat_name, wavelength)

                inset = (
                    0.0 if flat else self._cf_lateral_inset(z_rel, grid_t, spec["contact_angle"])
                )
                mask_2d = self._color_filter_pixel_mask(
                    nx,
                    ny,
                    r,
                    c,
                    grid_width,
                    corner_radius,
                    inset,
                )
                eps_grid[mask_2d] = eps

        # Metal grid overlay (mask is geometry-only, cache it)
        if grid_enabled and (flat or z_rel < grid_t):
            grid_material = grid_cfg.get("material", "tungsten")
            eps_grid_metal = self.material_db.get_epsilon(grid_material, wavelength)
            cache_key = (nx, ny, grid_width, corner_radius)
            if cache_key not in self._metal_grid_cache:
                self._metal_grid_cache[cache_key] = GeometryBuilder.metal_grid(
                    nx,
                    ny,
                    self.pitch,
                    self.unit_cell,
                    grid_width,
                    corner_radius=corner_radius,
                )
            metal_mask = self._metal_grid_cache[cache_key]
            eps_grid[metal_mask > 0.5] = eps_grid_metal

        return eps_grid

    def _cf_lateral_inset(
        self,
        z_rel: float,
        grid_thickness: float,
        contact_angle: float,
    ) -> float:
        """Inset the CF sidewall above the grid using contact angle in degrees."""
        protrusion = max(0.0, z_rel - grid_thickness)
        if protrusion <= 0.0 or contact_angle >= 89.999:
            return 0.0

        theta = np.deg2rad(np.clip(contact_angle, 1.0, 89.999))
        return float(protrusion / np.tan(theta))

    def _color_filter_pixel_mask(
        self,
        nx: int,
        ny: int,
        row: int,
        col: int,
        grid_width: float,
        corner_radius: float,
        inset: float,
    ) -> np.ndarray:
        """Mask one color-filter cell as a rectangular/trapezoidal z slice."""
        inner_half = (self.pitch - grid_width) / 2.0 - inset
        if inner_half <= 0.0:
            return np.zeros((ny, nx), dtype=bool)

        xx, yy = self._get_meshgrid(nx, ny)
        cx = (col + 0.5) * self.pitch
        cy = (row + 0.5) * self.pitch
        dx = np.abs(xx - cx)
        dy = np.abs(yy - cy)

        if corner_radius <= 0.0:
            return (dx <= inner_half) & (dy <= inner_half)

        radius = min(corner_radius, inner_half)
        ex = np.maximum(dx - (inner_half - radius), 0.0)
        ey = np.maximum(dy - (inner_half - radius), 0.0)
        return np.asarray(
            (dx <= inner_half) & (dy <= inner_half) & (ex * ex + ey * ey <= radius * radius)
        )

    def _build_silicon_slices(
        self,
        layer: Layer,
        wavelength: float,
        nx: int,
        ny: int,
        si_cfg: dict,
    ) -> list[LayerSlice]:
        """Build z-aware silicon slices.

        Dispatches to a fast single/two-slice path for plain vertical-wall DTI
        and switches to a z-resolved staircase when any realistic feature is
        enabled: a conformal DTI liner, a tapered DTI sidewall, or a backside
        inverted-pyramid surface texture.
        """
        dti_cfg = si_cfg.get("dti", {}) or {}
        tex_cfg = si_cfg.get("surface_texture", {}) or {}
        liner_cfg = dti_cfg.get("liner", {}) or {}

        dti_enabled = bool(dti_cfg.get("enabled", False))
        tex_enabled = (
            bool(tex_cfg.get("enabled", False)) and float(tex_cfg.get("height", 0.0)) > 0.0
        )
        liner_enabled = (
            dti_enabled
            and bool(liner_cfg.get("enabled", False))
            and float(liner_cfg.get("thickness", 0.0)) > 0.0
        )
        tapered = dti_enabled and float(dti_cfg.get("taper_angle", 90.0)) < 89.999

        if not (tex_enabled or liner_enabled or tapered):
            return self._legacy_silicon_slices(layer, wavelength, nx, ny, si_cfg)

        material = si_cfg.get("material", "silicon")
        breaks = self._silicon_slice_breaks(layer, si_cfg)
        slices: list[LayerSlice] = []
        for i, (z0_rel, z1_rel) in enumerate(itertools.pairwise(breaks)):
            thickness = z1_rel - z0_rel
            if thickness <= 0.0:
                continue
            z_mid_rel = 0.5 * (z0_rel + z1_rel)
            depth_from_top = layer.thickness - z_mid_rel
            eps_grid = self._build_si_grid_at_depth(wavelength, nx, ny, si_cfg, depth_from_top)
            slices.append(
                LayerSlice(
                    z_start=layer.z_start + z0_rel,
                    z_end=layer.z_start + z1_rel,
                    thickness=thickness,
                    eps_grid=eps_grid,
                    name=f"silicon_slice_{i}",
                    material=material,
                )
            )
        return slices

    def _silicon_slice_breaks(self, layer: Layer, si_cfg: dict) -> list[float]:
        """Local z breakpoints (from Si bottom) for DTI taper and texture."""
        thickness = layer.thickness
        points = {0.0, thickness}

        dti_cfg = si_cfg.get("dti", {}) or {}
        if dti_cfg.get("enabled", False):
            depth = float(np.clip(dti_cfg.get("depth", thickness), 0.0, thickness))
            dti_start = thickness - depth
            if 0.0 < dti_start < thickness:
                points.add(dti_start)
            if depth > 0.0 and float(dti_cfg.get("taper_angle", 90.0)) < 89.999:
                n = max(1, int(dti_cfg.get("n_slices", 6)))
                points.update(np.linspace(dti_start, thickness, n + 1))

        tex_cfg = si_cfg.get("surface_texture", {}) or {}
        if tex_cfg.get("enabled", False):
            th = float(np.clip(tex_cfg.get("height", 0.0), 0.0, thickness))
            if th > 0.0:
                tex_start = thickness - th
                if 0.0 < tex_start < thickness:
                    points.add(tex_start)
                n = max(1, int(tex_cfg.get("n_slices", 8)))
                points.update(np.linspace(tex_start, thickness, n + 1))

        return sorted(float(p) for p in points)

    @staticmethod
    def _taper_inset(depth_from_top: float, taper_angle: float) -> float:
        """Lateral inset of a tapered sidewall at a depth below the opening."""
        if depth_from_top <= 0.0 or taper_angle >= 89.999:
            return 0.0
        theta = np.deg2rad(np.clip(taper_angle, 1.0, 89.999))
        return float(depth_from_top / np.tan(theta))

    def _build_si_grid_at_depth(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        si_cfg: dict,
        depth_from_top: float,
    ) -> np.ndarray:
        """Build the silicon permittivity grid at a depth below the Si top.

        Composes bulk silicon, a (tapered, lined) DTI trench, and a backside
        inverted-pyramid texture, in that order.
        """
        eps_si = self.material_db.get_epsilon(si_cfg.get("material", "silicon"), wavelength)
        eps_grid = np.full((ny, nx), eps_si, dtype=complex)

        dti_cfg = si_cfg.get("dti", {}) or {}
        if dti_cfg.get("enabled", False):
            si_thickness = si_cfg.get("thickness", 3.0)
            depth = float(np.clip(dti_cfg.get("depth", si_thickness), 0.0, si_thickness))
            if 0.0 <= depth_from_top <= depth:
                self._apply_dti_at_depth(eps_grid, wavelength, nx, ny, dti_cfg, depth_from_top)

        tex_cfg = si_cfg.get("surface_texture", {}) or {}
        if tex_cfg.get("enabled", False):
            self._apply_texture_at_depth(eps_grid, wavelength, nx, ny, tex_cfg, depth_from_top)

        return eps_grid

    def _apply_dti_at_depth(
        self,
        eps_grid: np.ndarray,
        wavelength: float,
        nx: int,
        ny: int,
        dti_cfg: dict,
        depth_from_top: float,
    ) -> None:
        """Stamp a tapered, optionally lined DTI trench into ``eps_grid``."""
        width = float(dti_cfg.get("width", 0.1))
        taper = float(dti_cfg.get("taper_angle", 90.0))
        outer_half = width / 2.0 - self._taper_inset(depth_from_top, taper)
        if outer_half <= 0.0:
            return

        liner_cfg = dti_cfg.get("liner", {}) or {}
        liner_t = float(liner_cfg.get("thickness", 0.0)) if liner_cfg.get("enabled", False) else 0.0

        eps_core = self.material_db.get_epsilon(dti_cfg.get("material", "sio2"), wavelength)
        outer_mask = GeometryBuilder.trench_grid(nx, ny, self.pitch, self.unit_cell, outer_half)
        if liner_t > 0.0:
            eps_liner = self.material_db.get_epsilon(liner_cfg.get("material", "al2o3"), wavelength)
            eps_grid[outer_mask > 0.5] = eps_liner
            core_half = max(0.0, outer_half - liner_t)
            core_mask = GeometryBuilder.trench_grid(nx, ny, self.pitch, self.unit_cell, core_half)
            eps_grid[core_mask > 0.5] = eps_core
        else:
            eps_grid[outer_mask > 0.5] = eps_core

    def _apply_texture_at_depth(
        self,
        eps_grid: np.ndarray,
        wavelength: float,
        nx: int,
        ny: int,
        tex_cfg: dict,
        depth_from_top: float,
    ) -> None:
        """Carve inverted-pyramid pits into ``eps_grid`` at a given depth."""
        tex_h = float(tex_cfg.get("height", 0.0))
        if tex_h <= 0.0 or depth_from_top < 0.0 or depth_from_top > tex_h:
            return
        period = tex_cfg.get("period") or self.pitch
        half = (period / 2.0) * (1.0 - depth_from_top / tex_h)
        lx, ly = self.domain_size
        mask = GeometryBuilder.inverted_pyramid_mask(nx, ny, lx, ly, period, half)
        eps_fill = self.material_db.get_epsilon(tex_cfg.get("fill_material", "sio2"), wavelength)
        eps_grid[mask > 0.5] = eps_fill

    def _legacy_silicon_slices(
        self,
        layer: Layer,
        wavelength: float,
        nx: int,
        ny: int,
        si_cfg: dict,
    ) -> list[LayerSlice]:
        """Build z-aware silicon slices for plain FDTI and BDTI layouts."""

        material = si_cfg.get("material", "silicon")
        dti_cfg = si_cfg.get("dti", {})
        if not dti_cfg.get("enabled", False):
            eps_grid = self._build_si_layer(wavelength, nx, ny, si_cfg, include_dti=False)
            return [
                LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name="silicon",
                    material=material,
                )
            ]

        mode = str(dti_cfg.get("mode", "fdti")).lower()
        if mode not in {"fdti", "bdti"}:
            raise ValueError(f"Unsupported DTI mode '{mode}'. Expected 'fdti' or 'bdti'.")

        eps_dti_grid = self._build_si_layer(wavelength, nx, ny, si_cfg, include_dti=True)
        if mode == "fdti":
            return [
                LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_dti_grid,
                    name="silicon",
                    material=material,
                )
            ]

        depth = float(dti_cfg.get("depth", layer.thickness))
        depth = float(np.clip(depth, 0.0, layer.thickness))
        if depth <= 0.0:
            eps_grid = self._build_si_layer(wavelength, nx, ny, si_cfg, include_dti=False)
            return [
                LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_grid,
                    name="silicon",
                    material=material,
                )
            ]
        if depth >= layer.thickness:
            return [
                LayerSlice(
                    z_start=layer.z_start,
                    z_end=layer.z_end,
                    thickness=layer.thickness,
                    eps_grid=eps_dti_grid,
                    name="silicon",
                    material=material,
                )
            ]

        eps_bulk_grid = self._build_si_layer(wavelength, nx, ny, si_cfg, include_dti=False)
        dti_start = layer.z_end - depth
        return [
            LayerSlice(
                z_start=layer.z_start,
                z_end=dti_start,
                thickness=dti_start - layer.z_start,
                eps_grid=eps_bulk_grid,
                name="silicon_bulk",
                material=material,
            ),
            LayerSlice(
                z_start=dti_start,
                z_end=layer.z_end,
                thickness=layer.z_end - dti_start,
                eps_grid=eps_dti_grid,
                name="silicon_bdti",
                material=material,
            ),
        ]

    def _build_si_layer(
        self,
        wavelength: float,
        nx: int,
        ny: int,
        si_cfg: dict,
        include_dti: bool = True,
    ) -> np.ndarray:
        """Build silicon layer with optional DTI."""
        eps_si = self.material_db.get_epsilon(si_cfg.get("material", "silicon"), wavelength)
        eps_grid = np.full((ny, nx), eps_si, dtype=complex)

        dti_cfg = si_cfg.get("dti", {})
        if include_dti and dti_cfg.get("enabled", False):
            dti_width = dti_cfg.get("width", 0.1)
            dti_material = dti_cfg.get("material", "sio2")
            eps_dti = self.material_db.get_epsilon(dti_material, wavelength)
            # DTI mask is geometry-only (no wavelength dependence), cache it
            cache_key = (nx, ny)
            if cache_key not in self._dti_mask_cache:
                self._dti_mask_cache[cache_key] = GeometryBuilder.dti_grid(
                    nx, ny, self.pitch, self.unit_cell, dti_width
                )
            dti_mask = self._dti_mask_cache[cache_key]
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
            nx,
            ny,
            nz,
            self.pitch,
            self.unit_cell,
            tuple(pd_cfg.get("position", [0.0, 0.0, 0.5])),
            tuple(pd_cfg.get("size", [0.7, 0.7, 2.0])),
            si_layer.z_start,
            si_layer.z_end,
            self.bayer_map,
        )
