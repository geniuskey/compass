"""meent RCWA solver adapter for COMPASS.

Wraps the meent library (multi-backend RCWA: NumPy, JAX, PyTorch).
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


def _dfs_work_grid_shape(shape_yx, fourier_order, enhanced_dfs):
    """Installed meent raster-DFS repeat rule, before the three Fourier transforms."""
    result = []
    for size, order in zip(shape_yx, reversed(fourier_order), strict=True):
        minimum = (4 * order + 1) * size if enhanced_dfs else 4 * order + 1
        result.append(size * (minimum // size + 1) if size < minimum else size)
    return result


class MeentSolver(SolverBase):
    """meent RCWA solver adapter.

    Supports NumPy, JAX, and PyTorch backends via config.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._backend = config.get("params", {}).get("backend", "numpy")
        self._last_layer_slices: list | None = None
        self._last_wavelength: float | None = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(f"meent: geometry setup, backend={self._backend}")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(f"meent: source setup - {self._source.n_wavelengths} wavelengths")

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import meent
        except ImportError as err:
            raise ImportError("meent is required. Install with: pip install meent") from err

        if self._backend == "numpy" and self.device != "cpu":
            raise ValueError("meent/numpy supports CPU execution only")
        if self._backend == "jax":
            import jax

            target = "gpu" if self.device.startswith("cuda") else "cpu"
            jax.config.update("jax_default_device", jax.devices(target)[0])

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        fourier_type = params.get("fourier_type", 0)
        if fourier_type not in (0, 1):
            raise ValueError("meent fourier_type must be 0 (DFS) or 1 (CFS)")
        enhanced_dfs = params.get("enhanced_dfs", True)
        if not isinstance(enhanced_dfs, bool):
            raise ValueError("meent enhanced_dfs must be a boolean")

        lx, ly = self._pixel_stack.domain_size
        # Zero disables automatic oversampling for independent grid/order studies.
        grid_multiplier = params.get("grid_multiplier", 3)
        nx = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[0] + 1) * grid_multiplier)
        ny = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[1] + 1) * grid_multiplier)

        pol_runs = self._source.get_polarization_runs()
        self._failed_runs = []
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength, nx, ny, n_lens_slices=params.get("n_lens_slices", 30),
            )

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    # meent backend selection
                    backend_id = {"numpy": 0, "jax": 1, "torch": 2}.get(self._backend, 0)
                    device_kwargs = {}
                    if backend_id == 1:
                        device_kwargs["device"] = 1 if self.device.startswith("cuda") else 0
                    if backend_id == 2:
                        import torch

                        device_kwargs["device"] = torch.device(self.device)

                    mee = meent.call_mee(
                        backend=backend_id,
                        **device_kwargs,
                        pol=0 if pol == "TE" else 1,
                        n_top=1.0,
                        n_bot=1.0,
                        theta=self._source.theta_rad,
                        phi=self._source.phi_rad,
                        fto=fourier_order,
                        fourier_type=fourier_type,
                        enhanced_dfs=enhanced_dfs,
                        period=[lx * 1000, ly * 1000],  # meent uses nm
                        wavelength=wavelength * 1000,  # meent uses nm
                        thickness=[s.thickness * 1000 for s in reversed(layer_slices)],
                        type_complex=np.complex128,
                    )

                    # Set permittivity for each layer
                    ucell_list = []
                    for s in reversed(layer_slices):
                        ucell_list.append(s.eps_grid[np.newaxis, :, :])

                    # meent uses n-ik for passive media; COMPASS uses n+ik.
                    # Convert both the optical quantity and time convention.
                    mee.ucell = np.conj(np.sqrt(np.concatenate(ucell_list, axis=0)))
                    result = mee.conv_solve()
                    de_ri = result.de_ri
                    de_ti = result.de_ti
                    if backend_id == 2:
                        # NumPy reductions cannot operate directly on torch tensors,
                        # including tensors resident on a CUDA device.
                        de_ri = de_ri.detach().cpu().numpy()
                        de_ti = de_ti.detach().cpu().numpy()

                    R = float(np.sum(de_ri))
                    T = float(np.sum(de_ti))
                    if R + T > 1.0 + 0.01:
                        logger.warning(
                            f"meent: R+T={R + T:.4f} > 1 at λ={wavelength:.4f}um "
                            f"(numerical instability for multi-layer 2D stack)"
                        )
                    A = 1.0 - R - T

                except Exception as e:
                    logger.error(f"meent failed at λ={wavelength:.4f}um: {e}")
                    self._record_failed_run(wavelength, pol, e)
                    R = T = A = float("nan")

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_wavelength = wavelength

                if np.isnan(A):
                    pixel_qe = self._nan_pixel_qe()
                else:
                    pixel_qe = self._compute_per_pixel_qe(layer_slices, wavelength, A)
                for key, val in pixel_qe.items():
                    qe_pol_accum.setdefault(key, []).append(val)

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

                warnings.warn(f"meent: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "meent",
                "backend": self._backend,
                "fourier_order": fourier_order,
                "epsilon_grid_requested": int(params.get("epsilon_grid", 64)),
                "epsilon_grid_shape_yx": [ny, nx],
                "grid_multiplier": grid_multiplier,
                "fourier_type": fourier_type,
                "fourier_algorithm": "raster_dfs" if fourier_type == 0 else "raster_cfs",
                "enhanced_dfs": enhanced_dfs,
                "fourier_work_grid_shape_yx": (
                    _dfs_work_grid_shape([ny, nx], fourier_order, enhanced_dfs)
                    if fourier_type == 0 else None
                ),
                "fourier_work_grid_shape_source": (
                    "derived from meent raster DFS repeat rule; CFS has no FFT work grid"
                ),
                "qe_method": "eps_imag_weight",
                **self._failure_metadata(),
            },
        )

    def _compute_per_pixel_qe(
        self,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Compute per-pixel QE using eps_imag weighting in PD regions."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
        if n_pixels == 0:
            return {}
        lx, ly = self._pixel_stack.domain_size

        pixel_weights = {}
        total_weight = 0.0

        pitch = self._pixel_stack.pitch
        si_z_end = max(
            (layer.z_end for layer in self._pixel_stack.layers if layer.name == "silicon"),
            default=0.0,
        )
        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            key = f"{pd.color}_{r}_{c}"
            # position[2] is the PD-center depth below the silicon top surface
            pd_cz = si_z_end - pd.position[2]
            pd_z_min = pd_cz - pd.size[2] / 2
            pd_z_max = pd_cz + pd.size[2] / 2

            # PhotodiodeSpec.position is the offset from the pixel center;
            # convert to absolute domain coordinates ([0, lx) x [0, ly)).
            pd_cx = (c + 0.5) * pitch + pd.position[0]
            pd_cy = (r + 0.5) * pitch + pd.position[1]

            weight = 0.0
            for s in layer_slices:
                z_overlap_min = max(s.z_start, pd_z_min)
                z_overlap_max = min(s.z_end, pd_z_max)
                if z_overlap_max <= z_overlap_min:
                    continue
                dz = z_overlap_max - z_overlap_min
                # eps grids are indexed [row=y, col=x] (shape (ny, nx))
                ny_s, nx_s = s.eps_grid.shape
                ix_min = max(0, int(((pd_cx - pd.size[0] / 2) / lx) * nx_s))
                ix_max = min(nx_s, int(np.ceil(((pd_cx + pd.size[0] / 2) / lx) * nx_s)))
                iy_min = max(0, int(((pd_cy - pd.size[1] / 2) / ly) * ny_s))
                iy_max = min(ny_s, int(np.ceil(((pd_cy + pd.size[1] / 2) / ly) * ny_s)))
                if ix_max <= ix_min or iy_max <= iy_min:
                    continue
                eps_region = s.eps_grid[iy_min:iy_max, ix_min:ix_max]
                weight += float(np.mean(np.imag(eps_region))) * dz

            pixel_weights[key] = max(weight, 0.0)
            total_weight += max(weight, 0.0)

        # Per-pixel QE convention: absorbed power normalized by the power
        # incident on that pixel's own area (shared with the torcwa solver).
        qe_per_pixel = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = min(1.0, total_absorption * (w / total_weight) * n_pixels)
        else:
            # Uniform stack: every pixel sees the cell-average absorption
            for r in range(n_rows):
                for c in range(n_cols):
                    key = f"{bayer[r][c]}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption
        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract approximate 2D field from permittivity profile."""
        if self._last_layer_slices is None:
            logger.warning("meent: no simulation data, returning zeros")
            return np.zeros((64, 64))

        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        layer_info = self._last_layer_slices
        _lx, ly = self._pixel_stack.domain_size
        nz = len(layer_info)
        nx_out, ny_out = 64, 64
        wl = self._last_wavelength or 0.55
        k0 = 2 * np.pi / wl

        if plane == "xz":
            field_2d = np.zeros((nx_out, nz))
            for zi, s in enumerate(layer_info):
                eps = s.eps_grid
                ny_s = eps.shape[0]
                y_idx = max(0, min(ny_s - 1, int(((position + ly / 2) / ly) * ny_s)))
                col = eps[y_idx, :]
                x_orig = np.linspace(0, 1, len(col))
                x_new = np.linspace(0, 1, nx_out)
                field_2d[:, zi] = np.interp(x_new, x_orig, np.abs(np.imag(col)) + 1e-10)
            if component == "|E|2":
                for xi in range(nx_out):
                    intensity = 1.0
                    for zi in range(nz):
                        alpha = 2 * k0 * field_2d[xi, zi]
                        dz = layer_info[zi].thickness
                        intensity *= np.exp(-alpha * dz)
                        field_2d[xi, zi] = intensity
            return field_2d

        elif plane == "xy":
            z_accum = 0.0
            for s in layer_info:
                if z_accum + s.thickness >= position:
                    from scipy.ndimage import zoom

                    eps = s.eps_grid
                    zx = nx_out / eps.shape[0]
                    zy = ny_out / eps.shape[1]
                    return np.asarray(np.abs(zoom(np.real(eps), (zx, zy), order=1)))
                z_accum += s.thickness

        return np.zeros((64, 64))


SolverFactory.register("meent", MeentSolver)
