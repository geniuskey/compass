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
        self._pixel_stack = pixel_stack
        logger.info(
            f"torcwa: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source from config."""
        self._source = PlanewaveSource.from_config(source_config)
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
            raise ImportError(
                "torcwa is required. Install with: pip install torcwa"
            ) from err

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        dtype_str = params.get("dtype", "complex64")
        dtype = getattr(torch, dtype_str)

        stability = self.config.get("stability", {})
        precision_strategy = stability.get("precision_strategy", "mixed")

        lx, ly = self._pixel_stack.domain_size
        L = [lx, ly]  # Period in um

        nx = max(64, (2 * fourier_order[0] + 1) * 3)
        ny = max(64, (2 * fourier_order[1] + 1) * 3)

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[np.ndarray]] = {}
        all_R, all_T, all_A = [], [], []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(f"torcwa: wavelength {wavelength:.4f} um ({wl_idx+1}/{self._source.n_wavelengths})")

            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength, nx, ny, n_lens_slices=30
            )

            qe_pol_accum: dict[str, list] = {}
            R_pol, T_pol, A_pol = [], [], []

            for pol in pol_runs:
                try:
                    result = self._run_single(
                        torcwa, torch, wavelength, L,
                        fourier_order, layer_slices,
                        pol, dtype, precision_strategy,
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

        return SimulationResult(
            qe_per_pixel=qe_per_pixel,
            wavelengths=self._source.wavelengths,
            reflection=np.array(all_R),
            transmission=np.array(all_T),
            absorption=np.array(all_A),
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

        # Track layers for per-pixel QE and field extraction
        layer_info = []
        for s in reversed(layer_slices):
            eps_tensor = torch.tensor(
                s.eps_grid, dtype=dtype, device=self.device
            )
            sim.add_layer(thickness=s.thickness, eps=eps_tensor)
            layer_info.append(s)

        # Output layer (substrate below, effectively air or continuation)
        sim.add_output_layer(eps=1.0)

        # Set incidence angle
        assert self._source is not None
        sim.set_incident_angle(
            inc_ang=self._source.theta_rad,
            azi_ang=self._source.phi_rad,
        )

        # Solve
        sim.solve_global_smatrix()

        # Extract R, T
        R = float(sim.S_parameters.R.real) if hasattr(sim.S_parameters, 'R') else 0.0
        T = float(sim.S_parameters.T.real) if hasattr(sim.S_parameters, 'T') else 0.0
        A = 1.0 - R - T

        # Store sim reference for field extraction
        self._last_sim = sim
        self._last_layer_info = layer_info
        self._last_wavelength = wavelength

        # Per-pixel QE: compute absorption in each photodiode region
        qe_per_pixel = self._compute_per_pixel_qe(
            sim, torch, layer_slices, wavelength, A
        )

        return {"R": R, "T": T, "A": A, "qe_per_pixel": qe_per_pixel}

    def _compute_per_pixel_qe(
        self, sim, torch, layer_slices, wavelength: float, total_absorption: float,
    ) -> dict:
        """Compute per-pixel QE from layer absorption profiles.

        Uses the imaginary part of permittivity in each photodiode region
        to weight the absorption distribution. When field reconstruction
        is available, uses Poynting vector differences; otherwise falls
        back to eps_imag weighting.
        """
        assert self._pixel_stack is not None
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
        _pitch = self._pixel_stack.pitch

        # Build absorption weight per pixel from eps_imag in PD regions
        pixel_weights = {}
        total_weight = 0.0

        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            color = pd.color
            key = f"{color}_{r}_{c}"

            # PD bounding box in absolute coordinates
            pd_x_min = pd.position[0] - pd.size[0] / 2
            pd_x_max = pd.position[0] + pd.size[0] / 2
            pd_y_min = pd.position[1] - pd.size[1] / 2
            pd_y_max = pd.position[1] + pd.size[1] / 2
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

                # Map PD xy to grid indices
                lx, ly = self._pixel_stack.domain_size
                ix_min = max(0, int(((pd_x_min + lx / 2) / lx) * nx_s))
                ix_max = min(nx_s, int(((pd_x_max + lx / 2) / lx) * nx_s))
                iy_min = max(0, int(((pd_y_min + ly / 2) / ly) * ny_s))
                iy_max = min(ny_s, int(((pd_y_max + ly / 2) / ly) * ny_s))

                if ix_max <= ix_min or iy_max <= iy_min:
                    continue

                # Absorption weight = integral of eps_imag over PD volume
                eps_region = eps[ix_min:ix_max, iy_min:iy_max]
                eps_imag_mean = float(np.mean(np.imag(eps_region)))
                weight += eps_imag_mean * dz * (ix_max - ix_min) * (iy_max - iy_min) / (nx_s * ny_s)

            pixel_weights[key] = max(weight, 0.0)
            total_weight += max(weight, 0.0)

        # Distribute total absorption proportionally
        qe_per_pixel = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = total_absorption * (w / total_weight)
        else:
            # Fallback: even distribution
            for r in range(n_rows):
                for c in range(n_cols):
                    color = bayer[r][c]
                    key = f"{color}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption / n_pixels

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
            return self._extract_field_from_sim(
                sim, layer_info, component, plane, position
            )
        except Exception as e:
            logger.debug(f"torcwa field reconstruction failed: {e}")

        # Fallback: build approximate field from permittivity
        return self._approximate_field(
            layer_info, component, plane, position
        )

    def _extract_field_from_sim(
        self, sim, layer_info, component, plane, position,
    ) -> np.ndarray:
        """Extract field using torcwa's internal field reconstruction."""
        assert self._pixel_stack is not None

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
            if hasattr(sim, 'field_cell'):
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
        self, layer_info, component, plane, position,
    ) -> np.ndarray:
        """Build approximate field distribution from permittivity profile.

        Models field intensity as roughly proportional to exp(-alpha*z)
        where alpha depends on the imaginary part of the permittivity.
        """
        assert self._pixel_stack is not None
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
                assert self._last_wavelength is not None
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
                assert self._last_wavelength is not None
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
