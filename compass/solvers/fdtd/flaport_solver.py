"""flaport/fdtd solver adapter for COMPASS.

Wraps the flaport fdtd library (Python-native 3D FDTD with a PyTorch
backend). The simulation is a periodic unit cell excited by a CW plane
wave at normal incidence; steady-state phasors are extracted by a running
DFT over the last ``dft_periods`` optical periods.

Grid layout along z (index increasing upward, light travelling -z)::

    [PML | gap (T monitor) | pixel stack (Si bottom ... top air) |
     R monitor, soft source | PML]

* The source is *soft* (added to E), so reflected light passes through it
  into the upper PML instead of being trapped by a hard (assigned) source.
* R and T come from field phasors with the vacuum reference run's
  incident field subtracted, so they contain no source/reflection
  interference terms.
* QE integrates the absorbed power density ``k0 eps'' |E|^2`` over each
  photodiode footprint and the full silicon depth, normalized by the
  incident flux, i.e. the same definition as the RCWA adapters
  (:func:`compass.solvers.rcwa.fields.integrate_photodiode_qe`).

Fields use the library's units, where H is scaled so a plane wave has
``|H| = |E|``; flux ``Re(E x H*)`` and ``k0 eps'' |E|^2`` share those units.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.solvers.rcwa.fields import approximate_field, validate_field_request
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

_C = 299_792_458.0
_EPS0 = 8.8541878128e-12
#: Vacuum cells between the stack and each PML (monitors and source live here).
_GAP_BOTTOM = 6
_GAP_TOP = 8


class FlaportFdtdSolver(SolverBase):
    """flaport/fdtd 3D FDTD solver adapter (normal incidence).

    Config params:
        grid_spacing: Target cell size in um (adjusted so the lateral period
            is an integer number of cells). Default 0.02.
        runtime: Simulated time in fs. Default 200.
        pml_layers: PML thickness in cells. Default 15.
        courant_number: Courant number (None = library default 0.99/sqrt(3)).
        dtype: "float32" or "float64". Default "float64".
        dft_periods: Optical periods in the steady-state DFT window. Default 8.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_E: np.ndarray | None = None  # (nx, ny, nz_stack, 3) phasor
        self._last_grid_spacing: float | None = None
        self._last_wavelength: float | None = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(f"fdtd_flaport: geometry setup for {pixel_stack.unit_cell} unit cell")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.theta_deg != 0.0:
            raise NotImplementedError("fdtd_flaport supports normal incidence only")
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(f"fdtd_flaport: source setup - {self._source.n_wavelengths} wavelengths")

    # ------------------------------------------------------------------ grid

    def _grid_dims(self) -> tuple[float, int, int, int]:
        """(spacing um, x cells per period, y cells per period, z cells in stack)."""
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        params = self.config.get("params", {})
        target = float(params.get("grid_spacing", 0.02))
        if target <= 0:
            raise ValueError(f"grid_spacing must be positive, got {target}")
        lx, ly = self._pixel_stack.domain_size
        n_x = max(2, int(round(lx / target)))
        gs = lx / n_x
        n_y = max(2, int(round(ly / gs)))
        if abs(n_y * gs - ly) > 1e-6 * ly:
            raise ValueError(
                f"domain {lx} x {ly} um has no common cell size near {target} um; "
                "fdtd_flaport needs ly / lx to be a ratio of integers at this spacing"
            )
        z_min, z_max = self._pixel_stack.z_range
        n_z = max(1, int(round((z_max - z_min) / gs)))
        return gs, n_x, n_y, n_z

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import fdtd
        except ImportError as err:
            raise ImportError("fdtd is required. Install with: pip install fdtd") from err

        params = self.config.get("params", {})
        dtype = str(params.get("dtype", "float64"))
        if dtype not in ("float32", "float64"):
            raise ValueError(f"fdtd_flaport dtype must be float32 or float64, got {dtype!r}")
        backend = "torch.cuda" if self.device.startswith("cuda") else "torch"
        fdtd.set_backend(f"{backend}.{dtype}")

        gs, n_x, n_y, n_z = self._grid_dims()
        pml = int(params.get("pml_layers", 15))
        runtime_fs = float(params.get("runtime", 200))
        courant = params.get("courant_number")
        dft_periods = int(params.get("dft_periods", 8))

        pol_runs = self._source.get_polarization_runs()
        self._failed_runs = []
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"fdtd_flaport: wavelength {wavelength:.4f} um "
                f"({wl_idx + 1}/{self._source.n_wavelengths})"
            )
            R_pol, T_pol, A_pol = [], [], []
            qe_pol: dict[str, list[float]] = {}
            eps_3d = self._pixel_stack.get_permittivity_grid(wavelength, n_x, n_y, n_z)
            for pol in pol_runs:
                try:
                    common = dict(
                        fdtd=fdtd, gs=gs, n_x=n_x, n_y=n_y, n_z=n_z, pml=pml,
                        wavelength=float(wavelength), pol=pol, runtime_fs=runtime_fs,
                        courant=courant, dft_periods=dft_periods,
                    )
                    ref = self._simulate(eps_3d=None, **common)
                    tot = self._simulate(eps_3d=eps_3d, **common)
                    R, T, A, qe = self._reduce(ref, tot, eps_3d, gs, float(wavelength))
                    self._last_E = tot["E_stack"]
                    self._last_grid_spacing = gs
                    self._last_wavelength = float(wavelength)
                except Exception as e:
                    logger.error(f"fdtd_flaport failed at λ={wavelength:.4f}um: {e}")
                    self._record_failed_run(wavelength, pol, e)
                    R = T = A = float("nan")
                    qe = self._nan_pixel_qe()
                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)
                for key, val in qe.items():
                    qe_pol.setdefault(key, []).append(val)

            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)
            for k, vals in qe_pol.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(~np.isfinite(arr)):
                import warnings

                warnings.warn(f"fdtd_flaport: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "fdtd_flaport",
                "grid_spacing": gs,
                "grid_cells_xyz": [n_x, n_y, n_z],
                "qe_method": "field_integration",
                "device": self.device,
                **self._failure_metadata(),
            },
        )

    # ------------------------------------------------------------ simulation

    def _simulate(
        self,
        fdtd,
        gs: float,
        n_x: int,
        n_y: int,
        n_z: int,
        pml: int,
        wavelength: float,
        pol: str,
        runtime_fs: float,
        courant,
        dft_periods: int,
        eps_3d: np.ndarray | None,
    ) -> dict:
        """Run one CW simulation and return steady-state phasors.

        Returns:
            dict with "E_R"/"H_R" and "E_T"/"H_T" (monitor planes, (nx, ny, 3)
            colocated phasors) and "E_stack" ((nx, ny, n_z, 3) E phasor over
            the pixel stack, or None for the reference run).
        """
        import torch

        z0 = pml + _GAP_BOTTOM  # first stack cell
        k_T = pml + _GAP_BOTTOM // 2
        k_R = z0 + n_z + 2
        k_src = z0 + n_z + 5
        nz = z0 + n_z + _GAP_TOP + pml
        # fdtd's PeriodicBoundary identifies cell 0 with cell n-1, so a period
        # of n_x cells needs n_x + 1 grid points.
        kwargs = {"courant_number": float(courant)} if courant is not None else {}
        grid = fdtd.Grid(
            shape=(n_x + 1, n_y + 1, nz), grid_spacing=gs * 1e-6, permittivity=1.0, **kwargs
        )
        grid[0, :, :] = fdtd.PeriodicBoundary(name="x_boundary")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="y_boundary")
        grid[:, :, 0:pml] = fdtd.PML(name="z_pml_low")
        grid[:, :, -pml:] = fdtd.PML(name="z_pml_high")

        E = grid.E
        dev, rdtype = E.device, E.dtype
        cdtype = torch.complex128 if rdtype == torch.float64 else torch.complex64
        omega = 2 * np.pi * _C / (wavelength * 1e-6)
        dt = float(grid.time_step)

        damping = None
        if eps_3d is not None:
            eps_xyz = np.transpose(eps_3d, (1, 0, 2))  # [y, x, z] -> [x, y, z]
            eps_xyz = np.concatenate([eps_xyz, eps_xyz[:1]], axis=0)  # wrap point
            eps_xyz = np.concatenate([eps_xyz, eps_xyz[:, :1]], axis=1)
            eps_r = np.real(eps_xyz)
            if np.any(eps_r < 1.0 - 1e-9):
                raise ValueError(
                    "fdtd_flaport models loss as a conductivity on Re(eps) and cannot "
                    f"represent Re(eps) < 1 (min {eps_r.min():.3g}, e.g. metals); "
                    "use meep or an RCWA solver"
                )
            inv = torch.tensor(1.0 / eps_r, dtype=rdtype, device=dev).unsqueeze(-1)
            grid.inverse_permittivity[:, :, z0 : z0 + n_z] = inv
            sigma = omega * _EPS0 * np.maximum(np.imag(eps_xyz), 0.0)
            if np.any(sigma > 0):
                alpha = sigma * dt / (2.0 * _EPS0 * eps_r)
                damping = torch.tensor(
                    ((1.0 - alpha) / (1.0 + alpha))[..., None], dtype=rdtype, device=dev
                )

        comp = 1 if pol == "TE" else 0  # TE: E along y (s-pol for phi=0)
        n_steps = int(round(runtime_fs * 1e-15 / dt))
        period_steps = 2 * np.pi / omega / dt
        n_dft = int(round(dft_periods * period_steps))
        if n_steps < n_dft + 4 * period_steps:
            raise ValueError(
                f"runtime {runtime_fs} fs is too short for a {dft_periods}-period DFT window"
            )
        dft_start = n_steps - n_dft
        t_ramp = 3.0 * 2 * np.pi / omega

        acc: dict[str, torch.Tensor] = {}

        def add(name, field, phase):
            val = field.to(cdtype) * phase
            acc[name] = acc[name] + val if name in acc else val.clone()

        for n in range(n_steps):
            grid.update_E()
            t_e = (n + 1) * dt
            ramp = 1.0 if t_e >= t_ramp else 0.5 * (1.0 - np.cos(np.pi * t_e / t_ramp))
            grid.E[:, :, k_src, comp] += ramp * np.sin(omega * t_e)
            if damping is not None:
                grid.E[:, :, z0 : z0 + n_z] *= damping
            grid.update_H()
            grid.time_steps_passed += 1
            if n >= dft_start:
                pe = complex(np.exp(1j * omega * t_e))
                ph = complex(np.exp(1j * omega * (t_e + 0.5 * dt)))
                for name, k in (("R", k_R), ("T", k_T)):
                    add(f"E_{name}", grid.E[:-1, :-1, k], pe)
                    # Hx, Hy sit half a cell above E; average to colocate.
                    add(f"H_{name}", 0.5 * (grid.H[:-1, :-1, k] + grid.H[:-1, :-1, k - 1]), ph)
                if eps_3d is not None:
                    add("E_stack", grid.E[:-1, :-1, z0 : z0 + n_z], pe)

        scale = 2.0 / n_dft

        def out(name):
            return None if name not in acc else (acc[name] * scale).cpu().numpy()

        return {
            "E_R": out("E_R"), "H_R": out("H_R"), "E_T": out("E_T"), "H_T": out("H_T"),
            "E_stack": out("E_stack"),
        }

    @staticmethod
    def _flux_up(E: np.ndarray, H: np.ndarray) -> float:
        """Plane-averaged Re(E x H*)_z (positive upward)."""
        sz = np.real(E[..., 0] * np.conj(H[..., 1]) - E[..., 1] * np.conj(H[..., 0]))
        return float(np.mean(sz))

    def _reduce(self, ref: dict, tot: dict, eps_3d: np.ndarray, gs: float, wavelength: float):
        """R, T, A and per-pixel QE from the reference and structure runs."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        p_inc = -self._flux_up(ref["E_R"], ref["H_R"])  # downward incident flux
        if not np.isfinite(p_inc) or p_inc <= 0:
            raise RuntimeError(f"non-positive incident flux {p_inc}")
        R = self._flux_up(tot["E_R"] - ref["E_R"], tot["H_R"] - ref["H_R"]) / p_inc
        T = -self._flux_up(tot["E_T"], tot["H_T"]) / p_inc
        A = 1.0 - R - T
        qe = self._field_qe(tot["E_stack"], eps_3d, gs, wavelength, p_inc, A)
        return float(R), float(T), float(A), qe

    def _field_qe(
        self,
        E_stack: np.ndarray,
        eps_3d: np.ndarray,
        gs: float,
        wavelength: float,
        p_inc: float,
        total_absorption: float,
    ) -> dict[str, float]:
        """QE_pd = k0 * sum_{PD footprint, Si} eps''|E|^2 dV / (P_inc * A_pixel)."""
        stack = self._pixel_stack
        if stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        if not stack.photodiodes:
            return {}
        k0 = 2 * np.pi / wavelength
        density = np.sum(np.abs(E_stack) ** 2, axis=-1) * np.maximum(
            np.imag(np.transpose(eps_3d, (1, 0, 2))), 0.0
        )  # [x, y, z]
        n_x, n_y, n_z = density.shape
        z_min = stack.z_range[0]
        z_c = z_min + (np.arange(n_z) + 0.5) * gs
        si_mask = np.zeros(n_z, dtype=bool)
        for layer in stack.layers:
            if layer.name == "silicon":
                si_mask |= (z_c >= layer.z_start) & (z_c < layer.z_end)
        dv = gs**3
        total_field = k0 * float(density.sum()) * dv / (p_inc * n_x * n_y * gs * gs)
        if total_absorption > 0.02 and abs(total_field - total_absorption) > 0.1:
            logger.warning(
                f"fdtd_flaport: field-integrated absorption {total_field:.3f} deviates from "
                f"1-R-T = {total_absorption:.3f} at λ={wavelength:.4f}um "
                "(refine grid_spacing or lengthen runtime)"
            )
        si_density = density[:, :, si_mask].sum(axis=2)  # [x, y]
        n_rows, n_cols = stack.unit_cell
        lx, ly = stack.domain_size
        a_pixel = (lx / n_cols) * (ly / n_rows)
        pitch = stack.pitch
        qe: dict[str, float] = {}
        for pd in stack.photodiodes:
            r, c = pd.pixel_index
            x_c = (c + 0.5) * pitch + pd.position[0]
            y_c = (r + 0.5) * pitch + pd.position[1]
            ix0 = max(0, int(np.floor((x_c - pd.size[0] / 2) / gs)))
            ix1 = min(n_x, int(np.ceil((x_c + pd.size[0] / 2) / gs)))
            iy0 = max(0, int(np.floor((y_c - pd.size[1] / 2) / gs)))
            iy1 = min(n_y, int(np.ceil((y_c + pd.size[1] / 2) / gs)))
            val = k0 * float(si_density[ix0:ix1, iy0:iy1].sum()) * dv / (p_inc * a_pixel)
            qe[f"{pd.color}_{r}_{c}"] = float(np.clip(val, 0.0, 1.0))
        return qe

    # ---------------------------------------------------------------- fields

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Time-averaged field slice from the last (wavelength, polarization) run.

        Coordinates follow the other adapters: x, y in ``[0, lx) x [0, ly)``,
        z up from the stack bottom. Values are ``|E_c|^2`` / ``|E|^2`` of the
        steady-state phasor (incident amplitude set by the soft source, not
        normalized to 1).

        Returns:
            ``(nx, ny)`` for "xy", ``(nx, nz)`` for "xz", ``(ny, nz)`` for "yz",
            at the FDTD grid resolution.

        Raises:
            ValueError: Unknown component/plane, ``Sz`` (not recorded), or a
                position outside the stack.
        """
        validate_field_request(component, plane)
        if component == "Sz":
            raise ValueError("fdtd_flaport records E phasors only; Sz is not available")
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        if self._last_E is None or self._last_grid_spacing is None:
            logger.warning(
                "fdtd_flaport: no solved field available; returning a "
                "permittivity-based approximation, not a solved field"
            )
            wl = self._last_wavelength or 0.55
            return approximate_field(
                self._pixel_stack,
                self._pixel_stack.get_layer_slices(wl, 64, 64),
                wl, component, plane, position,
            )

        E = self._last_E
        gs = self._last_grid_spacing
        if component == "|E|2":
            field = np.sum(np.abs(E) ** 2, axis=-1)
        else:
            field = np.abs(E[..., "xyz".index(component[1])]) ** 2
        n_x, n_y, n_z = field.shape
        z_min, z_max = self._pixel_stack.z_range
        if plane == "xy":
            if not z_min <= position <= z_max:
                raise ValueError(f"xy-plane position z={position} um is outside the stack")
            k = min(n_z - 1, int((position - z_min) / gs))
            return np.asarray(field[:, :, k])
        n_axis = n_y if plane == "xz" else n_x
        idx = int(np.floor(position / gs)) % n_axis
        return np.asarray(field[:, idx, :] if plane == "xz" else field[idx, :, :])


SolverFactory.register("fdtd_flaport", FlaportFdtdSolver)
