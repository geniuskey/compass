"""Meep solver adapter for COMPASS.

Meep is a free, open-source FDTD simulation software package developed at MIT.
It supports dispersive materials (Lorentz-Drude), Bloch-periodic boundary
conditions, DFT flux monitors, and sub-pixel smoothing.

Meep uses its own geometry primitives (mp.Block, mp.Cylinder, etc.) and its
own material system. This adapter translates the COMPASS PixelStack into
Meep's native representation.

Reference: https://meep.readthedocs.io/
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from compass.core.types import FieldData, SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

_MEEP_AVAILABLE = False
try:
    import meep as mp

    _MEEP_AVAILABLE = all(
        hasattr(mp, attr) for attr in ("Simulation", "Medium", "Vector3", "Block", "PML")
    )
except ImportError:
    pass


class MeepSolver(SolverBase):
    """Meep 3D FDTD solver adapter.

    Converts PixelStack geometry to Meep's native geometry objects and runs
    frequency-domain or time-domain simulations with DFT flux monitors for
    R/T extraction and volume DFT monitors for per-pixel QE.

    Meep uses its own unit system where the unit of length is arbitrary.
    We use um as the length unit so that frequencies are in units of c/um.

    Config params (under config["params"]):
        resolution: Grid points per um (default: 50, i.e. 20nm spacing).
        pml_thickness: PML thickness in um (default: 0.5).
        runtime_periods: Fixed post-source optical periods (default: 100).
        decay_threshold: Legacy setting; fixed-duration runs do not auto-stop.
        use_dispersive: Allow Drude fits for negative epsilon; dielectric loss is always retained.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        if not _MEEP_AVAILABLE:
            logger.warning(
                "MIT photonics Meep Python API is not installed. "
                "Install with conda/mamba from conda-forge as pymeep; "
                "the PyPI package named 'meep' is a different task runner."
            )
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_sim: Any = None  # Store last mp.Simulation for field extraction
        self._last_fields: dict[str, FieldData] | None = None
        self._last_dft_fields: object = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to Meep geometry representation.

        This stores the PixelStack for later conversion to mp.Block objects
        during run(). The actual Meep geometry is built per-wavelength since
        material properties can be dispersive.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.
        """
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(
            f"meep: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um, "
            f"domain={pixel_stack.domain_size}, z_range={pixel_stack.z_range}"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source for Meep.

        Args:
            source_config: Source configuration dictionary.
        """
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(
            f"meep: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg}deg, pol={self._source.polarization}"
        )

    def run(self) -> SimulationResult:
        """Execute Meep FDTD simulation for all wavelengths.

        For each wavelength and polarization:
        1. Build Meep geometry from PixelStack (mp.Block objects with materials).
        2. Add Bloch-periodic boundaries on x/y and PML on z.
        3. Add a planewave source (CW at target frequency).
        4. Add DFT flux monitors above and below the structure.
        5. Run until fields decay or max time is reached.
        6. Extract R, T from flux monitors and compute per-pixel QE.

        Returns:
            SimulationResult with R, T, A spectra and per-pixel QE.
        """
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        if not _MEEP_AVAILABLE:
            raise ImportError(
                "MIT photonics Meep Python API is required. "
                "Install with conda/mamba from conda-forge as pymeep; "
                "the PyPI package named 'meep' is a different task runner."
            )

        import meep as mp

        params = self.config.get("params", {})
        resolution = params.get("resolution", 50)  # grid points per um
        pml_thickness = params.get("pml_thickness", 0.5)  # um
        runtime_periods = params.get("runtime_periods", 100)
        decay_threshold = params.get("decay_threshold", 1e-5)  # legacy; fixed runtime below
        use_dispersive = params.get("use_dispersive", True)

        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        total_z = z_max - z_min

        if self.device != "cpu":
            raise ValueError("Meep supports CPU execution only")
        padding = float(params.get("air_padding", 0.8))
        if padding <= 0:
            raise ValueError("air_padding must be positive")
        # Keep the source and flux planes in homogeneous air outside PML.
        sz = total_z + 2 * (padding + pml_thickness)
        cell_size = mp.Vector3(lx, ly, sz)

        # Source and monitor positions (z coordinates, centered)
        z_center = (z_min + z_max) / 2.0
        src_z = z_max - z_center + padding * 0.7
        refl_z = z_max - z_center + padding * 0.35
        trans_z = z_min - z_center - padding * 0.35

        logger.info(
            f"meep: cell_size=({lx:.2f}, {ly:.2f}, {sz:.2f})um, "
            f"resolution={resolution}, PML={pml_thickness}um"
        )

        pol_runs = self._source.get_polarization_runs()
        self._failed_runs = []
        all_qe: dict[str, list[float]] = {}
        all_R: list[float] = []
        all_T: list[float] = []
        all_A: list[float] = []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"meep: wavelength {wavelength:.4f} um ({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            fcen = 1.0 / wavelength  # meep frequency in units of c/um

            R_pol: list[float] = []
            T_pol: list[float] = []
            A_pol: list[float] = []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    R, T, A, field_data = self._run_single_wavelength(
                        mp,
                        wavelength,
                        fcen,
                        cell_size,
                        lx,
                        ly,
                        sz,
                        z_center,
                        pml_thickness,
                        resolution,
                        runtime_periods,
                        decay_threshold,
                        src_z,
                        refl_z,
                        trans_z,
                        pol,
                        use_dispersive,
                    )
                except Exception as e:
                    logger.error(
                        f"meep: failed at lambda={wavelength:.4f}um, pol={pol}: {e}",
                        exc_info=True,
                    )
                    self._record_failed_run(wavelength, pol, e)
                    R = T = A = float("nan")
                    field_data = None

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                # Per-pixel QE
                if np.isnan(A):
                    qe_this_pol = self._nan_pixel_qe()
                else:
                    qe_this_pol = self._compute_pixel_qe(
                        mp,
                        field_data,
                        wavelength,
                        resolution,
                        lx,
                        ly,
                        z_center,
                    )
                for k, v in qe_this_pol.items():
                    qe_pol_accum.setdefault(k, []).append(v)

            # Average over polarizations
            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                import warnings

                warnings.warn(f"meep: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "meep",
                "qe_method": "incident_power_field_integration",
                "resolution": resolution,
                "pml_thickness": pml_thickness,
                "runtime_periods": runtime_periods,
                "device": "cpu",
                "air_padding": padding,
                "termination": "fixed post-source duration: runtime_periods * wavelength",
                **self._failure_metadata(),
            },
        )

    def _run_single_wavelength(
        self,
        mp,
        wavelength: float,
        fcen: float,
        cell_size,
        lx: float,
        ly: float,
        sz: float,
        z_center: float,
        pml_thickness: float,
        resolution: int,
        runtime_periods: int,
        decay_threshold: float,
        src_z: float,
        refl_z: float,
        trans_z: float,
        polarization: str,
        use_dispersive: bool,
    ) -> tuple[float, float, float, object | None]:
        """Run a single-wavelength, single-polarization Meep simulation.

        This method:
        1. Builds geometry as mp.Block objects with correct materials.
        2. Normalizes flux with an empty (no structure) reference run.
        3. Runs the structure simulation.
        4. Extracts R, T from DFT flux monitors.

        Args:
            mp: The meep module.
            wavelength: Wavelength in um.
            fcen: Center frequency (1/wavelength in meep units).
            cell_size: mp.Vector3 cell size.
            lx, ly, sz: Cell dimensions in um.
            z_center: Z-center of the structure.
            pml_thickness: PML thickness in um.
            resolution: Grid points per um.
            runtime_periods: Max optical periods.
            decay_threshold: Auto-stop field decay threshold.
            src_z: Source z-position (relative to cell center).
            refl_z: Reflection monitor z-position.
            trans_z: Transmission monitor z-position.
            polarization: "TE" or "TM".
            use_dispersive: Whether to use Lorentz-Drude materials.

        Returns:
            Tuple of (R, T, A, simulation_object or None).
        """
        if self._last_sim is not None:
            self._last_sim.reset_meep()
            self._last_sim = None

        # Frequency width for the CW source (narrow Gaussian pulse)
        fwidth = 0.1 * fcen

        # Bloch periodic boundaries on x/y, PML on z
        k_point = mp.Vector3(0, 0, 0)
        if self._source is None:
            raise RuntimeError("source is not set; call setup_source() first")
        if self._source.theta_deg != 0.0:
            # Bloch periodic with oblique incidence
            theta_rad = self._source.theta_rad
            phi_rad = self._source.phi_rad
            n_inc = 1.0  # incident medium refractive index (air)
            kx = n_inc * fcen * np.sin(theta_rad) * np.cos(phi_rad)
            ky = n_inc * fcen * np.sin(theta_rad) * np.sin(phi_rad)
            k_point = mp.Vector3(kx, ky, 0)

        pml_layers = [mp.PML(thickness=pml_thickness, direction=mp.Z)]

        # Source polarization: TE (s) is perpendicular to the plane of
        # incidence, TM (p) lies in it; both rotate with phi. The current
        # sheet carries the Bloch phase exp(i 2 pi k.r) so it launches a
        # single oblique plane wave rather than a mix of diffraction orders.
        phi_rad = self._source.phi_rad
        if polarization == "TE":
            comps = [(mp.Ex, -np.sin(phi_rad)), (mp.Ey, np.cos(phi_rad))]
        else:
            comps = [(mp.Ex, np.cos(phi_rad)), (mp.Ey, np.sin(phi_rad))]
        kvec = mp.Vector3(k_point.x, k_point.y, 0)

        def bloch_amp(p, _k=kvec):
            return complex(np.exp(2j * np.pi * (_k.x * p.x + _k.y * p.y)))

        sources = [
            mp.Source(
                src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                component=comp,
                center=mp.Vector3(0, 0, src_z),
                size=mp.Vector3(lx, ly, 0),
                amplitude=float(amp),
                amp_func=bloch_amp,
            )
            for comp, amp in comps
            if abs(amp) > 1e-12
        ]

        # --- Reference run (no structure, for flux normalization) ---
        sim_ref = mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            boundary_layers=pml_layers,
            sources=sources,
            k_point=k_point,
            dimensions=3,
        )

        # Add flux monitors for reference run
        refl_fr_ref = sim_ref.add_flux(
            fcen,
            fwidth,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        trans_fr_ref = sim_ref.add_flux(
            fcen,
            fwidth,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, trans_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        # Run reference until fields decay
        max_time = runtime_periods / fcen
        sim_ref.run(until_after_sources=max_time)

        # Save reference reflection flux data (for subtracting later)
        refl_flux_data = sim_ref.get_flux_data(refl_fr_ref)
        input_flux = mp.get_fluxes(trans_fr_ref)  # incident flux

        sim_ref.reset_meep()

        # --- Structure run ---
        geometry = self._build_meep_geometry(mp, wavelength, z_center, use_dispersive)

        sim = mp.Simulation(
            cell_size=cell_size,
            resolution=resolution,
            boundary_layers=pml_layers,
            sources=sources,
            geometry=geometry,
            k_point=k_point,
            dimensions=3,
        )

        # Add flux monitors
        refl_fr = sim.add_flux(
            fcen,
            fwidth,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        trans_fr = sim.add_flux(
            fcen,
            fwidth,
            1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, trans_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        # Subtract reference flux from reflection monitor
        # (so that only reflected power is measured)
        sim.load_minus_flux_data(refl_fr, refl_flux_data)

        # Add DFT volume monitor in silicon region for QE computation
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        si_layer = None
        for layer in self._pixel_stack.layers:
            if layer.name == "silicon":
                si_layer = layer
                break

        dft_vol = None
        if si_layer is not None:
            si_z_center = (si_layer.z_start + si_layer.z_end) / 2.0 - z_center
            si_z_size = si_layer.thickness
            dft_vol = sim.add_dft_fields(
                [mp.Ex, mp.Ey, mp.Ez],
                fcen,
                0,
                1,
                center=mp.Vector3(0, 0, si_z_center),
                size=mp.Vector3(lx, ly, si_z_size),
            )

        # Run structure simulation until decay
        sim.run(until_after_sources=max_time)

        # Extract flux values
        refl_flux = mp.get_fluxes(refl_fr)
        trans_flux = mp.get_fluxes(trans_fr)

        # Compute R, T
        if len(input_flux) > 0 and abs(input_flux[0]) > 1e-30:
            R = float(-refl_flux[0] / input_flux[0])
            T = float(trans_flux[0] / input_flux[0])
        else:
            raise RuntimeError("Meep reference incident flux is zero")

        A = float(1.0 - R - T)

        logger.debug(
            f"meep: lambda={wavelength:.4f}um, pol={polarization}, R={R:.4f}, T={T:.4f}, A={A:.4f}"
        )

        # Store simulation for field extraction and QE
        self._last_sim = sim
        self._last_dft_fields = dft_vol

        self._incident_power = abs(float(input_flux[0]))
        # Keep fields alive for QE and subsequent field queries. The next run
        # releases the preceding simulation after its fields have been read.
        return R, T, A, dft_vol

    def _build_meep_geometry(
        self,
        mp,
        wavelength: float,
        z_center: float,
        use_dispersive: bool,
    ) -> list:
        """Build Meep geometry objects from PixelStack.

        Converts each layer in the PixelStack to mp.Block objects with
        appropriate materials. For patterned layers (color filter, silicon
        with DTI, microlens), creates sub-blocks for each distinct region.

        Args:
            mp: The meep module.
            wavelength: Wavelength in um (for material properties).
            z_center: Z-center offset for Meep coordinate system.
            use_dispersive: Whether to use Lorentz-Drude models.

        Returns:
            List of mp.GeometryObject instances.
        """
        geometry = []
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        lx, ly = self._pixel_stack.domain_size

        for layer in self._pixel_stack.layers:
            layer_z_center = (layer.z_start + layer.z_end) / 2.0 - z_center

            if layer.name == "color_filter":
                # Create per-pixel color filter blocks
                cf_cfg = self._pixel_stack._layer_configs.get("color_filter", {})

                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = self._pixel_stack.bayer_map[r][c]
                        cf_spec = self._pixel_stack._color_filter_spec(cf_cfg, color)
                        mat_name = cf_spec["material"]
                        cf_thickness = min(float(cf_spec["thickness"]), layer.thickness)
                        if cf_thickness <= 0.0:
                            continue
                        meep_mat = self._get_meep_material(mp, mat_name, wavelength, use_dispersive)

                        # Pixel center in meep coordinates
                        px = (c + 0.5) * self._pixel_stack.pitch - lx / 2.0
                        py = (r + 0.5) * self._pixel_stack.pitch - ly / 2.0
                        pz = layer.z_start + cf_thickness / 2.0 - z_center

                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(px, py, pz),
                                size=mp.Vector3(
                                    self._pixel_stack.pitch,
                                    self._pixel_stack.pitch,
                                    cf_thickness,
                                ),
                                material=meep_mat,
                            )
                        )

                # Add metal grid if enabled
                grid_cfg = cf_cfg.get("grid", {})
                if grid_cfg.get("enabled", False):
                    grid_width = grid_cfg.get("width", 0.05)
                    grid_thickness = min(
                        self._pixel_stack._grid_thickness(cf_cfg),
                        layer.thickness,
                    )
                    if grid_thickness <= 0.0:
                        continue
                    grid_z_center = layer.z_start + grid_thickness / 2.0 - z_center
                    grid_mat_name = grid_cfg.get("material", "tungsten")
                    grid_meep_mat = self._get_meep_material(
                        mp, grid_mat_name, wavelength, use_dispersive
                    )

                    # Vertical grid lines
                    for c in range(self._pixel_stack.unit_cell[1] + 1):
                        gx = c * self._pixel_stack.pitch - lx / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(gx, 0, grid_z_center),
                                size=mp.Vector3(grid_width, ly, grid_thickness),
                                material=grid_meep_mat,
                            )
                        )

                    # Horizontal grid lines
                    for r in range(self._pixel_stack.unit_cell[0] + 1):
                        gy = r * self._pixel_stack.pitch - ly / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(0, gy, grid_z_center),
                                size=mp.Vector3(lx, grid_width, grid_thickness),
                                material=grid_meep_mat,
                            )
                        )

            elif layer.name == "silicon":
                # Silicon bulk block
                si_cfg = self._pixel_stack._layer_configs.get("silicon", {})
                si_mat_name = si_cfg.get("material", layer.base_material)
                si_meep_mat = self._get_meep_material(mp, si_mat_name, wavelength, use_dispersive)

                geometry.append(
                    mp.Block(
                        center=mp.Vector3(0, 0, layer_z_center),
                        size=mp.Vector3(lx, ly, layer.thickness),
                        material=si_meep_mat,
                    )
                )

                # Add DTI if enabled
                dti_cfg = si_cfg.get("dti", {})
                if dti_cfg.get("enabled", False):
                    # Same trench extent as PixelStack (FDTI full depth, BDTI
                    # `depth` below the Si top).
                    z0_rel, z1_rel = self._pixel_stack._dti_span(dti_cfg, layer.thickness)
                    dti_h = z1_rel - z0_rel
                    dti_zc = layer.z_start + 0.5 * (z0_rel + z1_rel) - z_center
                    dti_width = dti_cfg.get("width", 0.1)
                    dti_mat_name = dti_cfg.get("material", "sio2")
                    dti_meep_mat = self._get_meep_material(
                        mp, dti_mat_name, wavelength, use_dispersive
                    )

                    # Vertical DTI trenches
                    for c in range(self._pixel_stack.unit_cell[1] + 1):
                        dx = c * self._pixel_stack.pitch - lx / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(dx, 0, dti_zc),
                                size=mp.Vector3(dti_width, ly, dti_h),
                                material=dti_meep_mat,
                            )
                        )

                    # Horizontal DTI trenches
                    for r in range(self._pixel_stack.unit_cell[0] + 1):
                        dy = r * self._pixel_stack.pitch - ly / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(0, dy, dti_zc),
                                size=mp.Vector3(lx, dti_width, dti_h),
                                material=dti_meep_mat,
                            )
                        )

            elif layer.name == "microlens":
                ml_meep_mat = self._get_meep_material(
                    mp, layer.base_material, wavelength, use_dispersive
                )
                geometry.extend(
                    self._microlens_geometry(mp, layer, ml_meep_mat, lx, ly, z_center)
                )

            elif layer.name == "air":
                # Air is the default medium, no geometry needed
                pass

            else:
                # Uniform dielectric layers (BARL, planarization, etc.)
                meep_mat = self._get_meep_material(
                    mp, layer.base_material, wavelength, use_dispersive
                )
                geometry.append(
                    mp.Block(
                        center=mp.Vector3(0, 0, layer_z_center),
                        size=mp.Vector3(lx, ly, layer.thickness),
                        material=meep_mat,
                    )
                )

        logger.debug(f"meep: built {len(geometry)} geometry objects")
        return geometry

    def _microlens_geometry(self, mp, layer, material, lx, ly, z_center) -> list:
        """Microlens as a staircase of superellipse prisms (plus residual base).

        Mirrors :meth:`PixelStack._microlens_staircase`: lens material where
        ``base + h(x, y) > z``, with the superellipse cross-section
        ``|x/rx|^n + |y/ry|^n <= r(z)^n`` of each slice. Prisms (not
        ellipsoids, whose z-extent tapers within each slice) fill every
        slice fully. Lenses shifted across the cell edge are replicated at
        their periodic images, and shared lenses are centred on their
        N x N pixel group like in PixelStack.
        """
        stack = self._pixel_stack
        if stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        n_slices = 20
        base_t = float(getattr(stack, "_ml_base_thickness", 0.0))
        geometry = []
        if base_t > 0:
            geometry.append(
                mp.Block(
                    center=mp.Vector3(0, 0, layer.z_start + base_t / 2 - z_center),
                    size=mp.Vector3(lx, ly, base_t),
                    material=material,
                )
            )
        sharing = int(getattr(stack, "_lens_sharing", 1))
        n_groups_x = max(1, stack.unit_cell[1] // sharing)
        t = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        for idx, ml in enumerate(stack.microlenses):
            gr, gc = divmod(idx, n_groups_x)
            cx = (gc + 0.5) * sharing * stack.pitch - lx / 2.0 + ml.shift_x
            cy = (gr + 0.5) * sharing * stack.pitch - ly / 2.0 + ml.shift_y
            cap_t = ml.height / n_slices
            for i in range(n_slices):
                frac = (i + 0.5) / n_slices  # height fraction of this slice
                r_frac = np.sqrt(max(0.0, 1.0 - frac ** (2.0 * ml.alpha_param)))
                rx, ry = ml.radius_x * r_frac, ml.radius_y * r_frac
                if rx < 1e-3 or ry < 1e-3:
                    continue
                e = 2.0 / ml.n_param
                px = rx * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e
                py = ry * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e
                z_lo = layer.z_start + base_t + i * cap_t - z_center
                for ox in (-lx, 0.0, lx):
                    for oy in (-ly, 0.0, ly):
                        x0, y0 = cx + ox, cy + oy
                        if abs(x0) - rx >= lx / 2 or abs(y0) - ry >= ly / 2:
                            continue  # image does not reach into the cell
                        verts = [
                            mp.Vector3(x0 + a, y0 + b, z_lo) for a, b in zip(px, py, strict=True)
                        ]
                        geometry.append(mp.Prism(verts, height=cap_t, material=material))
        return geometry

    def _get_meep_material(
        self,
        mp,
        mat_name: str,
        wavelength: float,
        use_dispersive: bool,
    ):
        """Convert a COMPASS material to a Meep Medium.

        For simple (non-dispersive) materials, creates mp.Medium with
        epsilon at the target wavelength. For metals and absorbing materials,
        creates a Lorentz-Drude dispersive model when use_dispersive is True.

        Args:
            mp: The meep module.
            mat_name: Material name in the COMPASS MaterialDB.
            wavelength: Wavelength in um.
            use_dispersive: Whether to attempt dispersive model creation.

        Returns:
            mp.Medium instance.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        mat_db = self._pixel_stack.material_db

        if not mat_db.has_material(mat_name):
            raise ValueError(f"Unknown Meep material: {mat_name}")

        n, k = mat_db.get_nk(mat_name, wavelength)

        eps = complex(n, k) ** 2
        fcen = 1.0 / wavelength
        if eps.real > 0:
            # Single-frequency match, including arbitrarily weak absorption.
            return mp.Medium(epsilon=eps.real,
                             D_conductivity=2 * np.pi * fcen * eps.imag / eps.real)
        if not use_dispersive:
            raise ValueError("Negative permittivity requires a dispersive material")
        # Stable passive Drude medium matching this frequency, without clamping eps'.
        eps_inf = 1.0
        gamma = fcen * eps.imag / (eps_inf - eps.real)
        sigma = (eps_inf - eps.real) * (fcen**2 + gamma**2) / fcen**2
        return mp.Medium(epsilon=eps_inf, E_susceptibilities=[
            mp.DrudeSusceptibility(frequency=fcen, gamma=gamma, sigma=sigma),
        ])

    def _compute_pixel_qe(
        self,
        mp,
        dft_vol,
        wavelength: float,
        resolution: int,
        lx: float,
        ly: float,
        z_center: float,
    ) -> dict[str, float]:
        """Compute per-pixel quantum efficiency from DFT volume fields.

        QE is computed by integrating |E|^2 * Im(eps) over each photodiode
        region and normalizing by incident power.

        Args:
            mp: The meep module.
            dft_vol: DFT volume monitor object (or None).
            wavelength: Wavelength in um.
            resolution: Grid resolution (points per um).
            lx, ly: Domain size in um.
            z_center: Z-center offset.

        Returns:
            Dict mapping pixel key to QE value.
        """
        qe_per_pixel: dict[str, float] = {}
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        bayer = self._pixel_stack.bayer_map
        _n_pixels = self._pixel_stack.unit_cell[0] * self._pixel_stack.unit_cell[1]

        if not self._pixel_stack.photodiodes:
            return {}
        if dft_vol is None or self._last_sim is None:
            # Regional absorption is unavailable without DFT fields.
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    qe_per_pixel[f"{color}_{r}_{c}"] = float("nan")
            return qe_per_pixel

        try:
            sim = self._last_sim

            # Get DFT field arrays from the volume monitor
            ex_dft = sim.get_dft_array(dft_vol, mp.Ex, 0)  # type: ignore[attr-defined]
            ey_dft = sim.get_dft_array(dft_vol, mp.Ey, 0)  # type: ignore[attr-defined]
            ez_dft = sim.get_dft_array(dft_vol, mp.Ez, 0)  # type: ignore[attr-defined]

            E_sq = np.abs(ex_dft) ** 2 + np.abs(ey_dft) ** 2 + np.abs(ez_dft) ** 2

            # Get permittivity (epsilon) in the volume for absorption calc
            # Use the PixelStack to get Im(eps) in the silicon region
            si_layer = None
            for layer in self._pixel_stack.layers:
                if layer.name == "silicon":
                    si_layer = layer
                    break

            if si_layer is None:
                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = bayer[r][c]
                        qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
                return qe_per_pixel

            # Get the shape of the DFT field array
            field_shape = E_sq.shape

            if len(field_shape) < 3 or any(s == 0 for s in field_shape):
                logger.warning("meep: DFT field array has unexpected shape, using fallback QE")
                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = bayer[r][c]
                        qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
                return qe_per_pixel

            # Get silicon permittivity (imaginary part for absorption)
            eps_si = self._pixel_stack.material_db.get_epsilon(si_layer.base_material, wavelength)
            eps_imag_si = np.imag(eps_si)

            # Volume absorption: Im(eps) * |E|^2
            abs_density = eps_imag_si * E_sq
            x_arr, y_arr, z_arr, weights = sim.get_array_metadata(dft_cell=dft_vol)
            # DFT fields and fluxes share the same Fourier normalization.
            abs_density = (2 * np.pi / wavelength) * abs_density
            total_absorbed = np.sum(abs_density * weights)

            if total_absorbed < 1e-30:
                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = bayer[r][c]
                        qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
                return qe_per_pixel

            # Build pixel masks on the DFT grid
            xx, yy, zz = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

            for pd in self._pixel_stack.photodiodes:
                r_idx, c_idx = pd.pixel_index
                color = pd.color
                key = f"{color}_{r_idx}_{c_idx}"

                # Photodiode center in meep coordinates
                pd_cx = (c_idx + 0.5) * self._pixel_stack.pitch - lx / 2.0 + pd.position[0]
                pd_cy = (r_idx + 0.5) * self._pixel_stack.pitch - ly / 2.0 + pd.position[1]
                pd_cz = si_layer.z_end - pd.position[2] - z_center

                # Rectangular photodiode mask
                mask = (
                    (np.abs(xx - pd_cx) <= pd.size[0] / 2.0 + 1e-12)
                    & (np.abs(yy - pd_cy) <= pd.size[1] / 2.0 + 1e-12)
                    & (np.abs(zz - pd_cz) <= pd.size[2] / 2.0 + 1e-12)
                )

                pixel_abs = np.sum(abs_density * mask * weights)
                # Incident power per pixel area, consistent with COMPASS QE.
                qe_per_pixel[key] = float(pixel_abs * _n_pixels / self._incident_power)

        except Exception as e:
            logger.warning(f"meep: QE computation failed: {e}; returning NaN")
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    qe_per_pixel[f"{color}_{r}_{c}"] = float("nan")

        # Fill in any missing pixels
        for r in range(self._pixel_stack.unit_cell[0]):
            for c in range(self._pixel_stack.unit_cell[1]):
                color = bayer[r][c]
                key = f"{color}_{r}_{c}"
                if key not in qe_per_pixel:
                    qe_per_pixel[key] = 0.0

        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract 2D field slice from the last Meep simulation.

        Uses Meep's get_array method to extract field data from the
        simulation object if still available, otherwise returns zeros.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        if self._last_sim is None or not _MEEP_AVAILABLE:
            logger.warning("meep: no simulation data available, returning zeros")
            return np.zeros((64, 64))

        import meep as mp

        sim = self._last_sim
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        z_center = (z_min + z_max) / 2.0
        total_z = z_max - z_min

        try:
            # Map component to meep field components
            if component == "Ex":
                meep_comp = mp.Ex
            elif component == "Ey":
                meep_comp = mp.Ey
            elif component == "Ez":
                meep_comp = mp.Ez
            elif component == "|E|2":
                # Need to combine components
                meep_comp = None
            else:
                logger.warning(f"meep: unknown component '{component}', using |E|2")
                meep_comp = None

            # Define the slice volume
            if plane == "xy":
                vol = mp.Volume(
                    center=mp.Vector3(0, 0, position - z_center),
                    size=mp.Vector3(lx, ly, 0),
                )
            elif plane == "xz":
                vol = mp.Volume(
                    center=mp.Vector3(0, position - ly / 2.0, 0),
                    size=mp.Vector3(lx, 0, total_z),
                )
            elif plane == "yz":
                vol = mp.Volume(
                    center=mp.Vector3(position - lx / 2.0, 0, 0),
                    size=mp.Vector3(0, ly, total_z),
                )
            else:
                logger.warning(f"meep: unknown plane '{plane}', using xy")
                vol = mp.Volume(
                    center=mp.Vector3(0, 0, 0),
                    size=mp.Vector3(lx, ly, 0),
                )

            if meep_comp is not None:
                field_slice = np.abs(sim.get_array(vol=vol, component=meep_comp)) ** 2  # type: ignore[attr-defined]
            else:
                # |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2
                ex = sim.get_array(vol=vol, component=mp.Ex)  # type: ignore[attr-defined]
                ey = sim.get_array(vol=vol, component=mp.Ey)  # type: ignore[attr-defined]
                ez = sim.get_array(vol=vol, component=mp.Ez)  # type: ignore[attr-defined]
                field_slice = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2

            return np.asarray(field_slice)

        except Exception as e:
            logger.warning(f"meep: field extraction failed: {e}, returning zeros")
            return np.zeros((64, 64))


SolverFactory.register("meep", MeepSolver)
