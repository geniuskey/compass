"""torcwa RCWA solver adapter for COMPASS.

Wraps the torcwa library to implement the SolverBase interface.
torcwa is a PyTorch-based RCWA solver with GPU support.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


class TorcwaSolver(SolverBase):
    """torcwa RCWA solver adapter.

    Converts PixelStack geometry to torcwa layer structure and executes
    RCWA simulation with S-matrix algorithm.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._layer_slices = None
        self._sim = None
        self._last_sim = None
        self._last_layer_info: list | None = None
        self._last_wavelength: float | None = None

        # Configure precision
        self._setup_precision()

    def _setup_precision(self) -> None:
        """Configure numerical precision settings."""
        try:
            import torch

            stability = self.config.get("stability", {})
            torch.backends.cuda.matmul.allow_tf32 = stability.get("allow_tf32", False)
            torch.backends.cudnn.allow_tf32 = stability.get("allow_tf32", False)
        except ImportError:
            pass

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to torcwa layer structure."""
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(
            f"torcwa: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source from config."""
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(
            f"torcwa: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg}deg, pol={self._source.polarization}"
        )

    def run(self) -> SimulationResult:
        """Execute RCWA simulation for all wavelengths."""
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import torch
            import torcwa
        except ImportError as err:
            raise ImportError("torcwa is required. Install with: pip install torcwa") from err

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        dtype_str = params.get("dtype", "complex64")
        dtype = getattr(torch, dtype_str)
        n_lens_slices = params.get("n_lens_slices", 30)
        grid_multiplier = params.get("grid_multiplier", 3)

        stability = self.config.get("stability", {})
        precision_strategy = stability.get("precision_strategy", "mixed")

        lx, ly = self._pixel_stack.domain_size
        L = [lx, ly]  # Period in um

        nx = max(64, (2 * fourier_order[0] + 1) * grid_multiplier)
        ny = max(64, (2 * fourier_order[1] + 1) * grid_multiplier)

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[np.ndarray]] = {}
        all_R, all_T, all_A = [], [], []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"torcwa: wavelength {wavelength:.4f} um ({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength,
                nx,
                ny,
                n_lens_slices=n_lens_slices,
            )

            qe_pol_accum: dict[str, list] = {}
            R_pol, T_pol, A_pol = [], [], []

            for pol in pol_runs:
                try:
                    result = self._run_single(
                        torcwa,
                        torch,
                        wavelength,
                        L,
                        fourier_order,
                        layer_slices,
                        pol,
                        dtype,
                        precision_strategy,
                    )
                    R_pol.append(result["R"])
                    T_pol.append(result["T"])
                    A_pol.append(result["A"])

                    for k, v in result["qe_per_pixel"].items():
                        qe_pol_accum.setdefault(k, []).append(v)

                except Exception as e:
                    logger.error(f"torcwa: failed at λ={wavelength:.4f}um, pol={pol}: {e}")
                    R_pol.append(0.0)
                    T_pol.append(0.0)
                    A_pol.append(0.0)

            # Average over polarizations
            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        # Assemble result
        qe_per_pixel = {k: np.array(v) for k, v in all_qe.items()}

        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                import warnings

                warnings.warn(f"torcwa: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel=qe_per_pixel,
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "torcwa",
                "fourier_order": fourier_order,
                "device": self.device,
            },
        )

    def _run_single(
        self,
        torcwa,
        torch,
        wavelength: float,
        L: list,
        fourier_order: list,
        layer_slices,
        polarization: str,
        dtype,
        precision_strategy: str,
    ) -> dict:
        """Run single wavelength, single polarization RCWA calculation."""
        freq = 1.0 / wavelength  # torcwa uses normalized frequency

        sim = torcwa.rcwa(
            freq=freq,
            order=fourier_order,
            L=L,
            dtype=dtype,
            device=self.device,
        )

        # Input layer (air, above structure)
        sim.add_input_layer(eps=1.0)

        # Set incidence angle BEFORE adding layers (torcwa needs Kx_norm for eigendecomp)
        if self._source is None:
            raise RuntimeError("source is not set; call setup_source() first")
        sim.set_incident_angle(
            inc_ang=self._source.theta_rad,
            azi_ang=self._source.phi_rad,
        )

        # Track layers for per-pixel QE and field extraction
        layer_info = []
        for s in reversed(layer_slices):
            eps_tensor = torch.tensor(s.eps_grid, dtype=dtype, device=self.device)
            sim.add_layer(thickness=s.thickness, eps=eps_tensor)
            layer_info.append(s)

        # Solve (no add_output_layer — defaults to free space)
        sim.solve_global_smatrix()

        # Map COMPASS polarization runs onto torcwa's ps notation:
        # TE = s-polarized, TM = p-polarized. amplitude is [Ep, Es].
        if polarization in ("TM", "p", "P"):
            in_tag, source_amp = "p", [1.0, 0.0]
        else:
            in_tag, source_amp = "s", [0.0, 1.0]

        # Extract R, T via S_parameters method
        if callable(sim.S_parameters):
            # Total reflectance/transmittance: sum diffraction efficiency over
            # ALL Fourier orders and BOTH output polarizations. The zeroth
            # order alone underestimates R/T (and overstates A) as soon as
            # higher orders propagate, which is the norm for multi-um Bayer
            # unit cells in the visible.
            all_orders = [
                [m, n]
                for m in range(-fourier_order[0], fourier_order[0] + 1)
                for n in range(-fourier_order[1], fourier_order[1] + 1)
            ]
            R = 0.0
            T = 0.0
            for out_tag in ("s", "p"):
                pol_tag = out_tag + in_tag
                S_R = sim.S_parameters(
                    orders=all_orders,
                    direction="forward",
                    port="reflection",
                    polarization=pol_tag,
                    power_norm=True,
                )
                S_T = sim.S_parameters(
                    orders=all_orders,
                    direction="forward",
                    port="transmission",
                    polarization=pol_tag,
                    power_norm=True,
                )
                R += float(torch.sum(torch.abs(S_R) ** 2))
                T += float(torch.sum(torch.abs(S_T) ** 2))
        else:
            # Legacy: S_parameters is an object with R, T attributes
            R = float(sim.S_parameters.R.real)
            T = float(sim.S_parameters.T.real)
        A = max(0.0, 1.0 - R - T)

        # Store sim reference for field extraction
        self._last_sim = sim
        self._last_layer_info = layer_info
        self._last_wavelength = wavelength

        # Per-pixel QE: integrate the absorbed power density eps'' |E|^2 in
        # the silicon under each photodiode footprint. Falls back to the
        # geometric eps''-weighting approximation if field reconstruction
        # is unavailable.
        try:
            sim.source_planewave(
                amplitude=source_amp,
                direction="forward",
                notation="ps",
            )
            qe_per_pixel = self._compute_per_pixel_qe_fields(sim, torch, layer_info, wavelength, A)
        except Exception as e:
            logger.warning(
                f"torcwa: field-based QE failed at λ={wavelength:.4f}um ({e}); "
                "falling back to eps''-weight approximation"
            )
            qe_per_pixel = self._compute_per_pixel_qe(sim, torch, layer_slices, wavelength, A)

        return {"R": R, "T": T, "A": A, "qe_per_pixel": qe_per_pixel}

    def _compute_per_pixel_qe_fields(
        self,
        sim,
        torch,
        layer_info,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Per-pixel QE from field reconstruction.

        The absorbed power density is p(r) = (omega eps0 / 2) eps''(r) |E(r)|^2.
        Normalized by the incident power (n_in cos(theta) A_cell / 2) sqrt(eps0/mu0)
        |E0|^2 this gives, in COMPASS units (lengths in um, |E0| = 1):

            QE_pd = k0 * sum_PD eps'' |E|^2 dV / (n_in cos(theta) A_cell)

        QE counts absorption in silicon slices only (parasitic absorption in
        color filters / metal grid is excluded), integrated over the photodiode
        xy footprint and the full silicon depth. Per-pixel QE follows the
        standard definition: absorbed power normalized by the power incident
        on THAT pixel's area (pitch^2), not on the whole unit cell.

        Args:
            sim: torcwa rcwa object (solved, with a planewave source set).
            torch: torch module.
            layer_info: LayerSlice list in torcwa layer order (input side first).
            wavelength: Wavelength in um.
            total_absorption: A = 1 - R - T for the energy cross-check.

        Returns:
            Dictionary mapping pixel name to QE value.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        assert self._source is not None

        lx, ly = self._pixel_stack.domain_size
        pitch = self._pixel_stack.pitch
        k0 = 2.0 * np.pi / wavelength
        cos_theta = float(np.cos(self._source.theta_rad))
        norm = k0 / (cos_theta * lx * ly)  # n_in = 1 (air)

        # Integrate eps'' |E|^2 per absorbing slice; keep silicon separately.
        si_density = None  # accumulated eps''|E|^2 dz on the xy grid
        total_field_abs = 0.0
        grid_shape = None

        for li, s in enumerate(layer_info):
            eps_imag = np.imag(s.eps_grid)
            if float(eps_imag.max()) < 1e-9:
                continue
            n0, n1 = s.eps_grid.shape
            if grid_shape is None:
                grid_shape = (n0, n1)
            # Sample fields at the torcwa positions of the eps grid samples
            # (eps_grid[a, b] sits at (a*lx/n0, b*ly/n1) in torcwa's lattice).
            x_axis = torch.arange(n0, device=self.device, dtype=torch.float32) * (lx / n0)
            y_axis = torch.arange(n1, device=self.device, dtype=torch.float32) * (ly / n1)
            nz = int(np.clip(round(s.thickness / 0.05), 8, 64))
            dz = s.thickness / nz
            e2_sum = torch.zeros((n0, n1), device=self.device, dtype=torch.float32)
            for kz in range(nz):
                z_prop = (kz + 0.5) * dz
                E, _H = sim.field_xy(li, x_axis, y_axis, z_prop=z_prop)
                e2_sum += sum(torch.abs(c) ** 2 for c in E)
            density = e2_sum.cpu().numpy() * eps_imag * dz  # eps''|E|^2 integrated in z
            cell_area = (lx / n0) * (ly / n1)
            total_field_abs += norm * float(density.sum()) * cell_area
            if s.name.startswith("silicon"):
                si_density = density if si_density is None else si_density + density

        if si_density is None or grid_shape is None:
            raise RuntimeError("no absorbing silicon slice found for QE integration")

        if total_absorption > 0.02 and abs(total_field_abs - total_absorption) > 0.1:
            logger.warning(
                f"torcwa: field-integrated absorption {total_field_abs:.3f} deviates "
                f"from 1-R-T = {total_absorption:.3f} at λ={wavelength:.4f}um "
                "(increase fourier_order / grid resolution)"
            )

        # Photodiode xy footprints. PhotodiodeSpec.position is the offset from
        # the pixel center; convert to absolute domain coordinates per pixel.
        n0, n1 = grid_shape
        cell_area = (lx / n0) * (ly / n1)
        n_rows, n_cols = self._pixel_stack.unit_cell
        pixel_area = (lx / n_cols) * (ly / n_rows)
        norm_pixel = k0 / (cos_theta * pixel_area)
        qe_per_pixel: dict[str, float] = {}
        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            x_c = (c + 0.5) * pitch + pd.position[0]
            y_c = (r + 0.5) * pitch + pd.position[1]
            # eps grids are indexed [row=y, col=x] on cell centers
            iy0 = max(0, int(np.floor((y_c - pd.size[1] / 2) / ly * n0)))
            iy1 = min(n0, int(np.ceil((y_c + pd.size[1] / 2) / ly * n0)))
            ix0 = max(0, int(np.floor((x_c - pd.size[0] / 2) / lx * n1)))
            ix1 = min(n1, int(np.ceil((x_c + pd.size[0] / 2) / lx * n1)))
            key = f"{pd.color}_{r}_{c}"
            if ix1 <= ix0 or iy1 <= iy0:
                qe_per_pixel[key] = 0.0
                continue
            qe = norm_pixel * float(si_density[iy0:iy1, ix0:ix1].sum()) * cell_area
            qe_per_pixel[key] = float(np.clip(qe, 0.0, 1.0))

        return qe_per_pixel

    def _compute_per_pixel_qe(
        self,
        sim,
        torch,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Compute per-pixel QE from layer absorption profiles.

        Uses the imaginary part of permittivity in each photodiode region
        to weight the absorption distribution. When field reconstruction
        is available, uses Poynting vector differences; otherwise falls
        back to eps_imag weighting.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
        if n_pixels == 0:
            return {}
        pitch = self._pixel_stack.pitch

        # Build absorption weight per pixel from eps_imag in PD regions
        pixel_weights = {}
        total_weight = 0.0

        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            color = pd.color
            key = f"{color}_{r}_{c}"

            # PD bounding box in absolute coordinates. PhotodiodeSpec.position
            # is the offset from the pixel center, so shift by the pixel origin.
            pd_cx = (c + 0.5) * pitch + pd.position[0]
            pd_cy = (r + 0.5) * pitch + pd.position[1]
            pd_x_min = pd_cx - pd.size[0] / 2
            pd_x_max = pd_cx + pd.size[0] / 2
            pd_y_min = pd_cy - pd.size[1] / 2
            pd_y_max = pd_cy + pd.size[1] / 2
            pd_z_min = pd.position[2] - pd.size[2] / 2
            pd_z_max = pd.position[2] + pd.size[2] / 2

            weight = 0.0
            for s in layer_slices:
                # Check z overlap with photodiode
                z_overlap_min = max(s.z_start, pd_z_min)
                z_overlap_max = min(s.z_end, pd_z_max)
                if z_overlap_max <= z_overlap_min:
                    continue

                dz = z_overlap_max - z_overlap_min
                eps = s.eps_grid
                nx_s, ny_s = eps.shape

                # Map PD xy (domain coordinates in [0, lx) x [0, ly)) to grid indices
                lx, ly = self._pixel_stack.domain_size
                ix_min = max(0, int((pd_x_min / lx) * nx_s))
                ix_max = min(nx_s, int(np.ceil((pd_x_max / lx) * nx_s)))
                iy_min = max(0, int((pd_y_min / ly) * ny_s))
                iy_max = min(ny_s, int(np.ceil((pd_y_max / ly) * ny_s)))

                if ix_max <= ix_min or iy_max <= iy_min:
                    continue

                # Absorption weight = integral of eps_imag over PD volume
                eps_region = eps[ix_min:ix_max, iy_min:iy_max]
                eps_imag_mean = float(np.mean(np.imag(eps_region)))
                weight += eps_imag_mean * dz * (ix_max - ix_min) * (iy_max - iy_min) / (nx_s * ny_s)

            pixel_weights[key] = max(weight, 0.0)
            total_weight += max(weight, 0.0)

        # Distribute total absorption proportionally; normalize per pixel area
        # (QE convention: absorbed power / power incident on that pixel).
        qe_per_pixel = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = min(1.0, total_absorption * (w / total_weight) * n_pixels)
        else:
            # Fallback: uniform stack — every pixel sees the cell-average absorption
            for r in range(n_rows):
                for c in range(n_cols):
                    color = bayer[r][c]
                    key = f"{color}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption

        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract 2D field slice from last simulation.

        Uses torcwa's field reconstruction when available. Otherwise
        builds an approximate field map from layer permittivity and
        absorption profile.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2", "Sz").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        if self._last_sim is None or self._last_layer_info is None:
            logger.warning("torcwa: no simulation data, returning zeros")
            return np.zeros((64, 64))

        sim = self._last_sim
        layer_info = self._last_layer_info

        # Try using torcwa's built-in field reconstruction
        try:
            return self._extract_field_from_sim(sim, layer_info, component, plane, position)
        except Exception as e:
            logger.debug(f"torcwa field reconstruction failed: {e}")

        # Fallback: build approximate field from permittivity
        return self._approximate_field(layer_info, component, plane, position)

    def _extract_field_from_sim(
        self,
        sim,
        layer_info,
        component,
        plane,
        position,
    ) -> np.ndarray:
        """Extract field using torcwa's internal field reconstruction."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")

        nx_field, ny_field = 64, 64
        _lx, _ly = self._pixel_stack.domain_size

        if plane == "xy":
            # Find the layer at the given z position
            z_accum = 0.0
            target_layer_idx = 0
            target_z_in_layer = 0.0
            for idx, s in enumerate(layer_info):
                if z_accum + s.thickness >= position or idx == len(layer_info) - 1:
                    target_layer_idx = idx + 1  # +1 for input layer
                    target_z_in_layer = position - z_accum
                    break
                z_accum += s.thickness

            # Use torcwa field_cell if available
            if hasattr(sim, "field_cell"):
                E, _H = sim.field_cell(
                    layer_idx=target_layer_idx,
                    nx=nx_field,
                    ny=ny_field,
                    z_pos=target_z_in_layer,
                )
                E_np = E.detach().cpu().numpy()
                return self._component_from_field(E_np, component)

        raise RuntimeError("field_cell not available")

    def _approximate_field(
        self,
        layer_info,
        component,
        plane,
        position,
    ) -> np.ndarray:
        """Build approximate field distribution from permittivity profile.

        Models field intensity as roughly proportional to exp(-alpha*z)
        where alpha depends on the imaginary part of the permittivity.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        nz = len(layer_info)
        nx_out, ny_out = 64, 64
        lx, ly = self._pixel_stack.domain_size

        if plane == "xy":
            # Single z-slice: find the layer
            z_accum = 0.0
            for s in layer_info:
                if z_accum + s.thickness >= position:
                    eps = s.eps_grid
                    from scipy.ndimage import zoom

                    target_shape = (nx_out, ny_out)
                    if eps.shape != target_shape:
                        zx = target_shape[0] / eps.shape[0]
                        zy = target_shape[1] / eps.shape[1]
                        if component == "|E|2":
                            return np.asarray(np.abs(zoom(np.real(eps), (zx, zy), order=1)))
                        else:
                            return np.asarray(zoom(np.real(eps), (zx, zy), order=1))
                    if component == "|E|2":
                        return np.asarray(np.abs(np.real(eps)))
                    return np.asarray(np.real(eps))
                z_accum += s.thickness
            return np.zeros((nx_out, ny_out))

        elif plane == "xz":
            # Build xz cross-section at y=position
            field_2d = np.zeros((nx_out, nz))
            for zi, s in enumerate(layer_info):
                eps = s.eps_grid
                ny_s = eps.shape[1]
                y_idx = min(int(((position + ly / 2) / ly) * ny_s), ny_s - 1)
                y_idx = max(0, y_idx)
                col = eps[:, y_idx]
                # Resample to nx_out
                x_orig = np.linspace(0, 1, len(col))
                x_new = np.linspace(0, 1, nx_out)
                field_2d[:, zi] = np.interp(x_new, x_orig, np.abs(np.imag(col)) + 1e-10)

            if component == "|E|2":
                # Approximate |E|^2 decay through absorbing media
                if self._last_wavelength is None:
                    raise RuntimeError("no wavelength data; run a simulation first")
                k0 = 2 * np.pi / self._last_wavelength
                for xi in range(nx_out):
                    intensity = 1.0
                    for zi in range(nz):
                        alpha = 2 * k0 * field_2d[xi, zi]
                        dz = layer_info[zi].thickness
                        intensity *= np.exp(-alpha * dz)
                        field_2d[xi, zi] = intensity
            return field_2d

        elif plane == "yz":
            field_2d = np.zeros((ny_out, nz))
            for zi, s in enumerate(layer_info):
                eps = s.eps_grid
                nx_s = eps.shape[0]
                x_idx = min(int(((position + lx / 2) / lx) * nx_s), nx_s - 1)
                x_idx = max(0, x_idx)
                col = eps[x_idx, :]
                y_orig = np.linspace(0, 1, len(col))
                y_new = np.linspace(0, 1, ny_out)
                field_2d[:, zi] = np.interp(y_new, y_orig, np.abs(np.imag(col)) + 1e-10)

            if component == "|E|2":
                if self._last_wavelength is None:
                    raise RuntimeError("no wavelength data; run a simulation first")
                k0 = 2 * np.pi / self._last_wavelength
                for yi in range(ny_out):
                    intensity = 1.0
                    for zi in range(nz):
                        alpha = 2 * k0 * field_2d[yi, zi]
                        dz = layer_info[zi].thickness
                        intensity *= np.exp(-alpha * dz)
                        field_2d[yi, zi] = intensity
            return field_2d

        return np.zeros((64, 64))

    @staticmethod
    def _component_from_field(E: np.ndarray, component: str) -> np.ndarray:
        """Extract a specific component from a 3-component field array."""
        if component == "Ex":
            return np.asarray(np.abs(E[..., 0]) ** 2)
        elif component == "Ey":
            return np.asarray(np.abs(E[..., 1]) ** 2)
        elif component == "Ez":
            return np.asarray(np.abs(E[..., 2]) ** 2)
        elif component == "|E|2":
            return np.asarray(np.sum(np.abs(E) ** 2, axis=-1))
        elif component == "Sz":
            # Approximate Poynting z: Re(Ex*Hy - Ey*Hx) ~ |E|^2 for planewave
            return np.asarray(np.sum(np.abs(E) ** 2, axis=-1))
        return np.asarray(np.abs(E[..., 0]) ** 2)


# Register with factory
SolverFactory.register("torcwa", TorcwaSolver)
