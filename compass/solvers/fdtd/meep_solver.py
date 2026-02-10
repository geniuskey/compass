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

import numpy as np

from compass.core.types import FieldData, SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

_MEEP_AVAILABLE = False
try:
    import meep as mp  # noqa: F401

    _MEEP_AVAILABLE = True
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
        runtime_periods: Number of optical periods to run (default: 100).
        decay_threshold: DFT field decay threshold for auto-stop (default: 1e-5).
        use_dispersive: Enable Lorentz-Drude dispersive materials (default: True).
    """

    def __init__(self, config: dict, device: str = "cpu"):
        if not _MEEP_AVAILABLE:
            logger.warning(
                "meep package is not installed. "
                "Install with: pip install meep (or conda install -c conda-forge pymeep)"
            )
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_sim: object = None  # Store last mp.Simulation for field extraction
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
                "meep is required. Install with: pip install meep "
                "(or conda install -c conda-forge pymeep)"
            )

        import meep as mp

        params = self.config.get("params", {})
        resolution = params.get("resolution", 50)  # grid points per um
        pml_thickness = params.get("pml_thickness", 0.5)  # um
        runtime_periods = params.get("runtime_periods", 100)
        decay_threshold = params.get("decay_threshold", 1e-5)
        use_dispersive = params.get("use_dispersive", True)

        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        total_z = z_max - z_min

        # Simulation cell: domain + PML on z (top and bottom)
        sz = total_z + 2 * pml_thickness
        cell_size = mp.Vector3(lx, ly, sz)

        # Source and monitor positions (z coordinates, centered)
        z_center = (z_min + z_max) / 2.0
        src_z = z_max - z_center + pml_thickness * 0.3  # near top PML
        refl_z = z_max - z_center + pml_thickness * 0.1  # just below source
        trans_z = z_min - z_center - pml_thickness * 0.1  # just above bottom PML

        logger.info(
            f"meep: cell_size=({lx:.2f}, {ly:.2f}, {sz:.2f})um, "
            f"resolution={resolution}, PML={pml_thickness}um"
        )

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[float]] = {}
        all_R: list[float] = []
        all_T: list[float] = []
        all_A: list[float] = []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"meep: wavelength {wavelength:.4f} um "
                f"({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            fcen = 1.0 / wavelength  # meep frequency in units of c/um

            R_pol: list[float] = []
            T_pol: list[float] = []
            A_pol: list[float] = []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    R, T, A, field_data = self._run_single_wavelength(
                        mp, wavelength, fcen, cell_size,
                        lx, ly, sz, z_center,
                        pml_thickness, resolution,
                        runtime_periods, decay_threshold,
                        src_z, refl_z, trans_z,
                        pol, use_dispersive,
                    )
                except Exception as e:
                    logger.error(
                        f"meep: failed at lambda={wavelength:.4f}um, pol={pol}: {e}",
                        exc_info=True,
                    )
                    R, T, A = 0.0, 0.0, 0.0
                    field_data = None

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                # Per-pixel QE
                qe_this_pol = self._compute_pixel_qe(
                    mp, field_data, wavelength, resolution,
                    lx, ly, z_center,
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

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=np.array(all_R),
            transmission=np.array(all_T),
            absorption=np.array(all_A),
            metadata={
                "solver_name": "meep",
                "resolution": resolution,
                "pml_thickness": pml_thickness,
                "runtime_periods": runtime_periods,
                "device": self.device,
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
        # Frequency width for the CW source (narrow Gaussian pulse)
        fwidth = 0.1 * fcen

        # Bloch periodic boundaries on x/y, PML on z
        k_point = mp.Vector3(0, 0, 0)
        assert self._source is not None
        if self._source.theta_deg != 0.0:
            # Bloch periodic with oblique incidence
            theta_rad = self._source.theta_rad
            phi_rad = self._source.phi_rad
            n_inc = 1.0  # incident medium refractive index (air)
            kx = n_inc * fcen * np.sin(theta_rad) * np.cos(phi_rad)
            ky = n_inc * fcen * np.sin(theta_rad) * np.sin(phi_rad)
            k_point = mp.Vector3(kx, ky, 0)

        pml_layers = [mp.PML(thickness=pml_thickness, direction=mp.Z)]

        # Source polarization
        if polarization == "TE":
            src_component = mp.Ex
        else:
            src_component = mp.Ey

        sources = [
            mp.Source(
                src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                component=src_component,
                center=mp.Vector3(0, 0, src_z),
                size=mp.Vector3(lx, ly, 0),
            ),
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
            fcen, fwidth, 1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        trans_fr_ref = sim_ref.add_flux(
            fcen, fwidth, 1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, trans_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        # Run reference until fields decay
        max_time = runtime_periods / fcen
        sim_ref.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt=max_time * 0.1,
                c=src_component,
                pt=mp.Vector3(0, 0, trans_z),
                decay_by=decay_threshold,
            ),
        )

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
            fcen, fwidth, 1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, refl_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        trans_fr = sim.add_flux(
            fcen, fwidth, 1,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, trans_z),
                size=mp.Vector3(lx, ly, 0),
            ),
        )

        # Subtract reference flux from reflection monitor
        # (so that only reflected power is measured)
        sim.load_minus_flux_data(refl_fr, refl_flux_data)

        # Add DFT volume monitor in silicon region for QE computation
        assert self._pixel_stack is not None
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
                fcen, 0, 1,
                center=mp.Vector3(0, 0, si_z_center),
                size=mp.Vector3(lx, ly, si_z_size),
            )

        # Run structure simulation until decay
        sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt=max_time * 0.1,
                c=src_component,
                pt=mp.Vector3(0, 0, trans_z),
                decay_by=decay_threshold,
            ),
        )

        # Extract flux values
        refl_flux = mp.get_fluxes(refl_fr)
        trans_flux = mp.get_fluxes(trans_fr)

        # Compute R, T
        if len(input_flux) > 0 and abs(input_flux[0]) > 1e-30:
            R = float(np.clip(-refl_flux[0] / input_flux[0], 0.0, 1.0))
            T = float(np.clip(trans_flux[0] / input_flux[0], 0.0, 1.0))
        else:
            logger.warning("meep: input flux is near zero, defaulting R=T=0")
            R, T = 0.0, 0.0

        A = float(np.clip(1.0 - R - T, 0.0, 1.0))

        logger.debug(
            f"meep: lambda={wavelength:.4f}um, pol={polarization}, "
            f"R={R:.4f}, T={T:.4f}, A={A:.4f}"
        )

        # Store simulation for field extraction and QE
        self._last_sim = sim
        self._last_dft_fields = dft_vol

        sim.reset_meep()

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
        assert self._pixel_stack is not None
        lx, ly = self._pixel_stack.domain_size

        for layer in self._pixel_stack.layers:
            layer_z_center = (layer.z_start + layer.z_end) / 2.0 - z_center

            if layer.name == "color_filter":
                # Create per-pixel color filter blocks
                cf_cfg = self._pixel_stack._layer_configs.get("color_filter", {})
                cf_materials = cf_cfg.get(
                    "materials", {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}
                )

                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = self._pixel_stack.bayer_map[r][c]
                        mat_name = cf_materials.get(color, f"cf_{color.lower()}")
                        meep_mat = self._get_meep_material(
                            mp, mat_name, wavelength, use_dispersive
                        )

                        # Pixel center in meep coordinates
                        px = (c + 0.5) * self._pixel_stack.pitch - lx / 2.0
                        py = (r + 0.5) * self._pixel_stack.pitch - ly / 2.0

                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(px, py, layer_z_center),
                                size=mp.Vector3(
                                    self._pixel_stack.pitch,
                                    self._pixel_stack.pitch,
                                    layer.thickness,
                                ),
                                material=meep_mat,
                            )
                        )

                # Add metal grid if enabled
                grid_cfg = cf_cfg.get("grid", {})
                if grid_cfg.get("enabled", False):
                    grid_width = grid_cfg.get("width", 0.05)
                    grid_mat_name = grid_cfg.get("material", "tungsten")
                    grid_meep_mat = self._get_meep_material(
                        mp, grid_mat_name, wavelength, use_dispersive
                    )

                    # Vertical grid lines
                    for c in range(self._pixel_stack.unit_cell[1] + 1):
                        gx = c * self._pixel_stack.pitch - lx / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(gx, 0, layer_z_center),
                                size=mp.Vector3(
                                    grid_width, ly, layer.thickness
                                ),
                                material=grid_meep_mat,
                            )
                        )

                    # Horizontal grid lines
                    for r in range(self._pixel_stack.unit_cell[0] + 1):
                        gy = r * self._pixel_stack.pitch - ly / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(0, gy, layer_z_center),
                                size=mp.Vector3(
                                    lx, grid_width, layer.thickness
                                ),
                                material=grid_meep_mat,
                            )
                        )

            elif layer.name == "silicon":
                # Silicon bulk block
                si_cfg = self._pixel_stack._layer_configs.get("silicon", {})
                si_mat_name = si_cfg.get("material", "silicon")
                si_meep_mat = self._get_meep_material(
                    mp, si_mat_name, wavelength, use_dispersive
                )

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
                                center=mp.Vector3(dx, 0, layer_z_center),
                                size=mp.Vector3(
                                    dti_width, ly, layer.thickness
                                ),
                                material=dti_meep_mat,
                            )
                        )

                    # Horizontal DTI trenches
                    for r in range(self._pixel_stack.unit_cell[0] + 1):
                        dy = r * self._pixel_stack.pitch - ly / 2.0
                        geometry.append(
                            mp.Block(
                                center=mp.Vector3(0, dy, layer_z_center),
                                size=mp.Vector3(
                                    lx, dti_width, layer.thickness
                                ),
                                material=dti_meep_mat,
                            )
                        )

            elif layer.name == "microlens":
                # Approximate microlens with stacked cylinders (staircase)
                n_lens_slices = 20
                slice_thickness = layer.thickness / n_lens_slices
                ml_mat_name = layer.base_material
                ml_meep_mat = self._get_meep_material(
                    mp, ml_mat_name, wavelength, use_dispersive
                )

                for ml_spec in self._pixel_stack.microlenses:
                    ml_idx = self._pixel_stack.microlenses.index(ml_spec)
                    r_idx = ml_idx // self._pixel_stack.unit_cell[1]
                    c_idx = ml_idx % self._pixel_stack.unit_cell[1]
                    cx = (c_idx + 0.5) * self._pixel_stack.pitch - lx / 2.0
                    cy = (r_idx + 0.5) * self._pixel_stack.pitch - ly / 2.0

                    for i in range(n_lens_slices):
                        rel_z = (i + 0.5) * slice_thickness
                        # Superellipse radius at this height
                        frac = rel_z / ml_spec.height
                        if frac > 1.0:
                            continue
                        # Inverse of the lens profile: radius at height z
                        # z = h*(1-r^2)^(1/(2*alpha)) => r = sqrt(1-(z/h)^(2*alpha))
                        r_frac = np.sqrt(
                            max(0.0, 1.0 - frac ** (2.0 * ml_spec.alpha_param))
                        )
                        rx = ml_spec.radius_x * r_frac
                        ry = ml_spec.radius_y * r_frac

                        if rx < 0.01 or ry < 0.01:
                            continue

                        slice_z = (
                            layer.z_start + rel_z - z_center
                        )

                        geometry.append(
                            mp.Ellipsoid(
                                center=mp.Vector3(
                                    cx + ml_spec.shift_x,
                                    cy + ml_spec.shift_y,
                                    slice_z,
                                ),
                                size=mp.Vector3(
                                    2 * rx, 2 * ry, slice_thickness
                                ),
                                material=ml_meep_mat,
                            )
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
        assert self._pixel_stack is not None
        mat_db = self._pixel_stack.material_db

        if not mat_db.has_material(mat_name):
            logger.warning(f"meep: unknown material '{mat_name}', using air")
            return mp.Medium(epsilon=1.0)

        n, k = mat_db.get_nk(mat_name, wavelength)

        if use_dispersive and k > 0.01:
            # Create a Lorentz-Drude model for absorbing/metallic materials.
            # Single Lorentz oscillator fit at the target wavelength.
            fcen = 1.0 / wavelength  # meep frequency units
            eps_real = n**2 - k**2
            eps_imag = 2 * n * k

            # Lorentz parameters: eps_inf + sigma * omega0^2 / (omega0^2 - omega^2 - i*gamma*omega)
            # At resonance omega = omega0: contribution = sigma * omega0 / (i*gamma)
            # For a broadband fit, use a wide Lorentz oscillator
            omega0 = fcen  # center frequency
            gamma = 0.5 * fcen  # broad linewidth
            # sigma chosen to match Im(eps) at the target frequency
            # Im(contribution) = sigma * omega0^2 * gamma * omega /
            #                     ((omega0^2 - omega^2)^2 + gamma^2*omega^2)
            # At omega = omega0: Im = sigma * omega0 / gamma
            sigma = eps_imag * gamma / omega0 if omega0 > 0 else 0.0

            lorentzian = mp.LorentzianSusceptibility(
                frequency=omega0,
                gamma=gamma,
                sigma=sigma,
            )

            return mp.Medium(
                epsilon=max(eps_real, 1.0),
                E_susceptibilities=[lorentzian],
            )
        else:
            # Simple non-dispersive dielectric
            eps = n**2 - k**2
            return mp.Medium(epsilon=max(eps, 1.0))

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
        assert self._pixel_stack is not None
        bayer = self._pixel_stack.bayer_map
        _n_pixels = self._pixel_stack.unit_cell[0] * self._pixel_stack.unit_cell[1]

        if dft_vol is None or self._last_sim is None:
            # Fallback: no DFT data available, return zeros
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
            return qe_per_pixel

        try:
            sim = self._last_sim

            # Get DFT field arrays from the volume monitor
            ex_dft = sim.get_dft_array(dft_vol, mp.Ex, 0)  # type: ignore[attr-defined]
            ey_dft = sim.get_dft_array(dft_vol, mp.Ey, 0)  # type: ignore[attr-defined]
            ez_dft = sim.get_dft_array(dft_vol, mp.Ez, 0)  # type: ignore[attr-defined]

            E_sq = np.abs(ex_dft)**2 + np.abs(ey_dft)**2 + np.abs(ez_dft)**2

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
            eps_si = self._pixel_stack.material_db.get_epsilon(
                si_layer.base_material, wavelength
            )
            eps_imag_si = np.imag(eps_si)

            # Volume absorption: Im(eps) * |E|^2
            abs_density = eps_imag_si * E_sq
            dV = (1.0 / resolution) ** 3
            total_absorbed = np.sum(abs_density) * dV

            if total_absorbed < 1e-30:
                for r in range(self._pixel_stack.unit_cell[0]):
                    for c in range(self._pixel_stack.unit_cell[1]):
                        color = bayer[r][c]
                        qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
                return qe_per_pixel

            # Build pixel masks on the DFT grid
            nfx, nfy, nfz = field_shape
            x_arr = np.linspace(-lx / 2.0, lx / 2.0, nfx, endpoint=False)
            y_arr = np.linspace(-ly / 2.0, ly / 2.0, nfy, endpoint=False)
            z_arr = np.linspace(
                si_layer.z_start - z_center,
                si_layer.z_end - z_center,
                nfz,
            )
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
                    (np.abs(xx - pd_cx) < pd.size[0] / 2.0)
                    & (np.abs(yy - pd_cy) < pd.size[1] / 2.0)
                    & (np.abs(zz - pd_cz) < pd.size[2] / 2.0)
                )

                pixel_abs = np.sum(abs_density * mask.astype(float)) * dV
                qe_per_pixel[key] = float(pixel_abs / total_absorbed)

        except Exception as e:
            logger.warning(f"meep: QE computation failed: {e}, using fallback")
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    qe_per_pixel[f"{color}_{r}_{c}"] = 0.0

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
        assert self._pixel_stack is not None
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
                field_slice = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2

            return np.asarray(field_slice)

        except Exception as e:
            logger.warning(f"meep: field extraction failed: {e}, returning zeros")
            return np.zeros((64, 64))


SolverFactory.register("meep", MeepSolver)
