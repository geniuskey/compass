"""grcwa RCWA solver adapter for COMPASS.

Wraps the grcwa library (GPU-accelerated RCWA with autograd backend).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


class GrcwaSolver(SolverBase):
    """grcwa RCWA solver adapter.

    grcwa uses normalized units and autograd-compatible numpy backend.
    This adapter handles unit conversion and interface mapping.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: Optional[PlanewaveSource] = None
        self._last_layer_slices = None
        self._last_wavelength = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        self._pixel_stack = pixel_stack
        logger.info(f"grcwa: geometry setup for {pixel_stack.unit_cell} unit cell")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        self._source_config = source_config
        logger.info(f"grcwa: source setup - {self._source.n_wavelengths} wavelengths")

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import grcwa
        except ImportError:
            raise ImportError("grcwa is required. Install with: pip install grcwa")

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        nG = fourier_order[0]  # grcwa uses single order

        lx, ly = self._pixel_stack.domain_size
        nx = max(64, (2 * nG + 1) * 3)
        ny = max(64, (2 * nG + 1) * 3)

        pol_runs = self._source.get_polarization_runs()
        all_qe: Dict[str, List[float]] = {}
        all_R, all_T, all_A = [], [], []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            freq = 1.0 / wavelength  # normalized frequency

            layer_slices = self._pixel_stack.get_layer_slices(wavelength, nx, ny)

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: Dict[str, List[float]] = {}

            for pol in pol_runs:
                # Set polarization
                if pol == "TE":
                    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
                else:
                    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}

                try:
                    obj = grcwa.obj(nG, lx, ly, freq, self._source.theta_deg, self._source.phi_deg, verbose=0)
                    obj.Add_LayerUniform(1.0, 0.0)  # input: air

                    for s in reversed(layer_slices):
                        eps_real = np.real(s.eps_grid)
                        eps_imag = np.imag(s.eps_grid)
                        obj.Add_LayerGrid(s.thickness, nx, ny)
                        obj.Grid_SetGrid(eps_real, eps_imag)

                    obj.Add_LayerUniform(1.0, 0.0)  # output: air

                    obj.Init_Setup(Gmethod=0)
                    obj.MakeExcitationPlanewave(
                        planewave["p_amp"], planewave["s_amp"],
                        planewave["p_phase"], planewave["s_phase"],
                        order=0,
                    )
                    R, T = obj.RT_Solve(normalize=1)
                    R = float(np.real(R))
                    T = float(np.real(T))
                    A = max(0.0, 1.0 - R - T)

                except Exception as e:
                    logger.error(f"grcwa failed at λ={wavelength:.4f}um: {e}")
                    R, T, A = 0.0, 0.0, 0.0

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_wavelength = wavelength

                # Per-pixel QE via eps_imag weighting in PD regions
                pixel_qe = self._compute_per_pixel_qe(layer_slices, wavelength, A)
                for key, val in pixel_qe.items():
                    qe_pol_accum.setdefault(key, []).append(val)

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
            metadata={"solver_name": "grcwa", "fourier_order": fourier_order, "device": self.device},
        )

    def _compute_per_pixel_qe(
        self, layer_slices, wavelength: float, total_absorption: float,
    ) -> dict:
        """Compute per-pixel QE using eps_imag weighting in PD regions."""
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
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
            logger.warning("grcwa: no simulation data, returning zeros")
            return np.zeros((64, 64))

        layer_info = self._last_layer_slices
        lx, ly = self._pixel_stack.domain_size
        nz = len(layer_info)
        nx_out, ny_out = 64, 64

        if plane == "xz":
            field_2d = np.zeros((nx_out, nz))
            wl = self._last_wavelength or 0.55
            k0 = 2 * np.pi / wl
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

        elif plane == "yz":
            field_2d = np.zeros((ny_out, nz))
            wl = self._last_wavelength or 0.55
            k0 = 2 * np.pi / wl
            for zi, s in enumerate(layer_info):
                eps = s.eps_grid
                nx_s = eps.shape[0]
                x_idx = max(0, min(nx_s - 1, int(((position + lx / 2) / lx) * nx_s)))
                col = eps[x_idx, :]
                y_orig = np.linspace(0, 1, len(col))
                y_new = np.linspace(0, 1, ny_out)
                field_2d[:, zi] = np.interp(y_new, y_orig, np.abs(np.imag(col)) + 1e-10)
            if component == "|E|2":
                for yi in range(ny_out):
                    intensity = 1.0
                    for zi in range(nz):
                        alpha = 2 * k0 * field_2d[yi, zi]
                        dz = layer_info[zi].thickness
                        intensity *= np.exp(-alpha * dz)
                        field_2d[yi, zi] = intensity
            return field_2d

        elif plane == "xy":
            z_accum = 0.0
            for s in layer_info:
                if z_accum + s.thickness >= position:
                    from scipy.ndimage import zoom
                    eps = s.eps_grid
                    zx = nx_out / eps.shape[0]
                    zy = ny_out / eps.shape[1]
                    return np.abs(zoom(np.real(eps), (zx, zy), order=1))
                z_accum += s.thickness
            return np.zeros((nx_out, ny_out))

        return np.zeros((64, 64))


SolverFactory.register("grcwa", GrcwaSolver)
