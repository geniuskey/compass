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

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])

        lx, ly = self._pixel_stack.domain_size
        nx = max(64, (2 * fourier_order[0] + 1) * 3)
        ny = max(64, (2 * fourier_order[1] + 1) * 3)

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            layer_slices = self._pixel_stack.get_layer_slices(wavelength, nx, ny)

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    # meent backend selection
                    backend_id = {"numpy": 0, "jax": 1, "torch": 2}.get(self._backend, 0)

                    mee = meent.call_mee(
                        backend=backend_id,
                        pol=0 if pol == "TE" else 1,
                        n_top=1.0,
                        n_bot=1.0,
                        theta=self._source.theta_deg,
                        phi=self._source.phi_deg,
                        fto=fourier_order,
                        period=[lx * 1000, ly * 1000],  # meent uses nm
                        wavelength=wavelength * 1000,  # meent uses nm
                        thickness=[s.thickness * 1000 for s in reversed(layer_slices)],
                        type_complex=np.complex128,
                    )

                    # Set permittivity for each layer
                    ucell_list = []
                    for s in reversed(layer_slices):
                        ucell_list.append(s.eps_grid[np.newaxis, :, :])

                    # meent ucell expects refractive index (n), not permittivity (eps)
                    mee.ucell = np.sqrt(np.concatenate(ucell_list, axis=0))
                    result = mee.conv_solve()
                    de_ri = result.de_ri
                    de_ti = result.de_ti

                    R = float(np.sum(de_ri))
                    T = float(np.sum(de_ti))
                    # meent 0.12.0 has numerical instability for multi-layer
                    # 2D stacks where R+T > 1. Clamp to physical range.
                    if R + T > 1.0 + 0.01:
                        logger.warning(
                            f"meent: R+T={R+T:.4f} > 1 at λ={wavelength:.4f}um "
                            f"(numerical instability for multi-layer 2D stack)"
                        )
                    A = max(0.0, 1.0 - R - T)

                except Exception as e:
                    logger.error(f"meent failed at λ={wavelength:.4f}um: {e}")
                    R, T, A = 0.0, 0.0, 0.0

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_wavelength = wavelength

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
            **result_arrays,
            metadata={"solver_name": "meent", "backend": self._backend, "fourier_order": fourier_order},
        )

    def _compute_per_pixel_qe(
        self, layer_slices, wavelength: float, total_absorption: float,
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

        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            key = f"{pd.color}_{r}_{c}"
            pd_z_min = pd.position[2] - pd.size[2] / 2
            pd_z_max = pd.position[2] + pd.size[2] / 2

            weight = 0.0
            for s in layer_slices:
                z_overlap_min = max(s.z_start, pd_z_min)
                z_overlap_max = min(s.z_end, pd_z_max)
                if z_overlap_max <= z_overlap_min:
                    continue
                dz = z_overlap_max - z_overlap_min
                nx_s, ny_s = s.eps_grid.shape
                ix_min = max(0, int(((pd.position[0] - pd.size[0] / 2 + lx / 2) / lx) * nx_s))
                ix_max = min(nx_s, int(((pd.position[0] + pd.size[0] / 2 + lx / 2) / lx) * nx_s))
                iy_min = max(0, int(((pd.position[1] - pd.size[1] / 2 + ly / 2) / ly) * ny_s))
                iy_max = min(ny_s, int(((pd.position[1] + pd.size[1] / 2 + ly / 2) / ly) * ny_s))
                if ix_max <= ix_min or iy_max <= iy_min:
                    continue
                eps_region = s.eps_grid[ix_min:ix_max, iy_min:iy_max]
                weight += float(np.mean(np.imag(eps_region))) * dz

            pixel_weights[key] = max(weight, 0.0)
            total_weight += max(weight, 0.0)

        qe_per_pixel = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = total_absorption * (w / total_weight)
        else:
            for r in range(n_rows):
                for c in range(n_cols):
                    key = f"{bayer[r][c]}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption / n_pixels
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
                ny_s = eps.shape[1]
                y_idx = max(0, min(ny_s - 1, int(((position + ly / 2) / ly) * ny_s)))
                col = eps[:, y_idx]
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
