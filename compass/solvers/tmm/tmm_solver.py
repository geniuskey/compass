"""Transfer Matrix Method (TMM) solver adapter for COMPASS.

Wraps the TMM core algorithm to conform to the SolverBase interface.
TMM treats the pixel as a 1D planar thin-film stack (no lateral patterning),
making it extremely fast (~1000x faster than RCWA) for initial stack design
and thin-film optimization.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import FieldData, SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.solvers.rcwa.fields import qe_z_window
from compass.solvers.tmm.tmm_core import (
    layer_absorption_1d,
    tmm_field_profile,
    transfer_matrix_1d,
)
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


class TMMSolver(SolverBase):
    """Transfer Matrix Method solver for 1D planar thin-film stacks.

    TMM computes reflection, transmission, and absorption analytically
    using 2x2 transfer matrices. It ignores lateral patterning (microlens
    shapes, color filter Bayer patterns, DTI trenches), treating each layer
    as a uniform slab with average optical properties.

    This makes TMM ideal for:
    - Rapid initial stack design and optimization
    - Anti-reflection coating design
    - BARL thickness sweeps
    - Baseline comparison for full RCWA simulations

    Limitations:
    - No lateral resolution: all pixels see the same 1D stack
    - Microlens focusing is not modeled
    - DTI and metal grids are averaged out
    - Per-pixel QE is the 1D silicon absorption (identical for every pixel)
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """Initialize TMM solver.

        Args:
            config: Solver configuration dictionary.
            device: Compute device (TMM always runs on CPU, parameter is
                accepted for API compatibility).
        """
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._layer_names: list[str] = []
        self._layer_materials: list[str] = []
        self._layer_thicknesses: np.ndarray = np.array([])
        self._field_resolution: int = config.get("params", {}).get("field_resolution", 200)
        # Polarization averaging follows the source: unpolarized sources
        # always average TE and TM. The param is kept for config
        # compatibility; False cannot turn averaging off.
        self._requested_polarization_average: bool = bool(
            config.get("params", {}).get("polarization_average", True)
        )
        self._polarization_average: bool = self._requested_polarization_average
        # "air": finite silicon over an air half-space, the same stack the
        # RCWA / FDTD adapters simulate. "silicon": silicon as a semi-infinite
        # substrate (all transmitted light is absorbed; QE is an upper bound).
        self._substrate: str = config.get("params", {}).get("substrate", "air")
        if self._substrate not in ("air", "silicon"):
            raise ValueError(f"tmm substrate must be 'air' or 'silicon', got {self._substrate!r}")
        if self._substrate == "silicon" and self.qe_depth == "photodiode":
            raise ValueError(
                "tmm: qe_depth='photodiode' needs a finite silicon layer; "
                "use params.substrate='air'"
            )
        # Cache for last run (for field extraction)
        self._last_field_data: FieldData | None = None
        self._last_n_layers: np.ndarray | None = None
        self._last_d_layers: np.ndarray | None = None
        self._last_wavelength: float | None = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Extract 1D layer stack from PixelStack.

        Lateral variations (Bayer pattern, DTI, microlens shape) are ignored.
        Each layer is represented by its base material and thickness.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.
        """
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")

        self._pixel_stack = pixel_stack

        # Extract 1D layer stack: base material and thickness per layer.
        # TMM ignores patterning, uses base material for each layer.
        self._layer_names = []
        self._layer_materials = []
        thicknesses = []

        for layer in pixel_stack.layers:
            self._layer_names.append(layer.name)
            self._layer_materials.append(layer.base_material)
            thicknesses.append(layer.thickness)

        self._layer_thicknesses = np.array(thicknesses)

        logger.info(
            f"TMM: geometry setup - {len(self._layer_names)} layers, "
            f"total height = {sum(thicknesses):.3f} um"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure excitation source from config dictionary.

        Args:
            source_config: Source configuration with wavelength, angle, polarization.
        """
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config

        # Re-derived on every call: a solver reused across sources (cone
        # runner) must not keep the flag of a previous polarized source.
        self._polarization_average = self._source.is_unpolarized
        if self._source.is_unpolarized and not self._requested_polarization_average:
            logger.warning(
                "TMM: params.polarization_average=false is ignored for an unpolarized "
                "source (TE and TM are averaged); set source.polarization to TE or TM "
                "for a single polarization"
            )

        logger.info(
            f"TMM: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg:.1f} deg"
        )

    def _tmm_stack(self, wavelength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TMM arrays (incident air first) from the PixelStack's layer slices.

        Each slice is replaced by its laterally averaged permittivity, so the
        1D stack uses the same materials as the RCWA/FDTD solvers (e.g. the
        configured color-filter materials and the microlens fill factor)
        instead of each layer's nominal base material.

        Returns:
            (n, d, si_mask) in TMM order: index 0 is the incident air
            half-space, the last entry the substrate; ``si_mask`` flags
            finite silicon slices.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        params = self.config.get("params", {})
        slices = self._pixel_stack.get_layer_slices(
            wavelength, 64, 64, n_lens_slices=int(params.get("n_lens_slices", 20))
        )
        top_first = sorted(slices, key=lambda sl: -sl.z_end)
        n_film = [complex(np.sqrt(np.mean(sl.eps_grid))) for sl in top_first]
        d_film = [sl.thickness for sl in top_first]
        is_si = [sl.name.startswith("silicon") for sl in top_first]
        if self.qe_depth == "photodiode":
            n_film, d_film, is_si = self._split_at_photodiode(top_first, n_film, d_film, is_si)
        if self._substrate == "silicon":
            if not any(is_si):
                raise ValueError("substrate='silicon' needs a silicon layer")
            n_sub = n_film[max(i for i, v in enumerate(is_si) if v)]  # bottom Si slice
            keep = [i for i, v in enumerate(is_si) if not v]
            n_film = [n_film[i] for i in keep]
            d_film = [d_film[i] for i in keep]
            is_si = [False] * len(keep)
        else:
            n_sub = 1.0 + 0j
        n = np.array([1.0 + 0j, *n_film, n_sub], dtype=complex)
        d = np.array([np.inf, *d_film, np.inf])
        si_mask = np.array([False, *is_si, False])
        return n, d, si_mask

    def _split_at_photodiode(
        self,
        top_first: list,
        n_film: list[complex],
        d_film: list[float],
        is_si: list[bool],
    ) -> tuple[list[complex], list[float], list[bool]]:
        """Split silicon slices at the photodiode depth window.

        The pieces keep the slice's index, so R and T are unchanged; only
        the piece inside the window stays flagged for the QE sum. TMM is
        laterally uniform, so one window (the first photodiode's) is used.
        """
        if self._pixel_stack is None or not self._pixel_stack.photodiodes:
            return n_film, d_film, is_si
        pds = self._pixel_stack.photodiodes
        z_lo, z_hi = qe_z_window(self._pixel_stack, pds[0], "photodiode")
        if any(qe_z_window(self._pixel_stack, pd, "photodiode") != (z_lo, z_hi) for pd in pds):
            logger.warning("tmm: photodiodes differ in depth; using the first one's window")
        n_out: list[complex] = []
        d_out: list[float] = []
        si_out: list[bool] = []
        for sl, n_i, d_i, si in zip(top_first, n_film, d_film, is_si, strict=True):
            if not si:
                n_out.append(n_i)
                d_out.append(d_i)
                si_out.append(False)
                continue
            # Top to bottom: above the PD, inside it, below it (COMPASS z up).
            pieces = (
                (max(z_hi, sl.z_start), sl.z_end, False),
                (max(z_lo, sl.z_start), min(z_hi, sl.z_end), True),
                (sl.z_start, min(z_lo, sl.z_end), False),
            )
            for z_a, z_b, inside in pieces:
                if z_b - z_a > 1e-12:
                    n_out.append(n_i)
                    d_out.append(z_b - z_a)
                    si_out.append(inside)
        return n_out, d_out, si_out

    def run(self) -> SimulationResult:
        """Execute TMM simulation over all wavelengths.

        Returns:
            SimulationResult with R, T, A spectra and approximate per-pixel QE.

        Raises:
            RuntimeError: If setup_geometry() or setup_source() not called.
        """
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        # Determine polarization runs
        pol_runs = self._source.get_polarization_runs()
        theta_rad = self._source.theta_rad

        all_R = np.zeros(self._source.n_wavelengths)
        all_T = np.zeros(self._source.n_wavelengths)
        all_A = np.zeros(self._source.n_wavelengths)
        all_si = np.zeros(self._source.n_wavelengths)

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            n_layers_tmm, d_layers_tmm, si_mask = self._tmm_stack(float(wavelength))

            R_pol, T_pol, A_pol, si_pol = [], [], [], []

            for pol in pol_runs:
                R, T, A = transfer_matrix_1d(
                    n_layers_tmm,
                    d_layers_tmm,
                    wavelength,
                    theta_rad,
                    pol,
                )
                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)
                layer_abs = layer_absorption_1d(
                    n_layers_tmm, d_layers_tmm, wavelength, theta_rad, pol
                )
                if self._substrate == "air":
                    si_pol.append(float(layer_abs[si_mask].sum()))
                else:
                    si_pol.append(T)  # everything entering the substrate is absorbed

            n_pol = len(pol_runs)
            all_R[wl_idx] = sum(R_pol) / n_pol
            all_T[wl_idx] = sum(T_pol) / n_pol
            all_A[wl_idx] = sum(A_pol) / n_pol
            all_si[wl_idx] = sum(si_pol) / n_pol

            # Cache for field extraction (use last wavelength)
            self._last_n_layers = n_layers_tmm
            self._last_d_layers = d_layers_tmm
            self._last_wavelength = wavelength

        # Compute approximate per-pixel QE
        qe_per_pixel = self._compute_approximate_qe(all_si)

        return SimulationResult(
            qe_per_pixel=qe_per_pixel,
            wavelengths=self._source.wavelengths.copy(),
            reflection=all_R,
            transmission=all_T,
            absorption=all_A,
            metadata={
                "solver_name": "tmm",
                "solver_type": "tmm",
                "qe_method": "tmm_1d_analytic",
                "n_layers": len(self._layer_names),
                "layer_names": self._layer_names.copy(),
                "polarization_average": self._polarization_average,
                "substrate": self._substrate,
            },
        )

    def _compute_approximate_qe(self, total_absorption: np.ndarray) -> dict[str, np.ndarray]:
        """Per-pixel QE from the silicon absorption of the 1D stack.

        TMM has no lateral resolution: every pixel sees the same uniform 1D
        stack, so under the per-pixel QE convention shared with the RCWA
        solvers each pixel's QE equals the stack's silicon absorption. (It
        used to be the total film absorption A, which with a semi-infinite
        silicon substrate is exactly the parasitic color-filter/BARL loss and
        excludes the light absorbed in silicon.)

        Args:
            total_absorption: Silicon absorption spectrum.

        Returns:
            Dictionary mapping pixel name to QE spectrum.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        if self._source is None:
            raise RuntimeError("source is not set; call setup_source() first")

        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols

        if n_pixels == 0:
            return {}

        per_pixel_qe = total_absorption

        qe_per_pixel: dict[str, np.ndarray] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                color = bayer[r % len(bayer)][c % len(bayer[0])]
                key = f"{color}_{r}_{c}"
                qe_per_pixel[key] = per_pixel_qe.copy()

        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract the |E|^2 profile of the last simulated wavelength.

        For TMM, only the z-profile (through the stack) is physically meaningful.
        The xz and yz planes show the 1D field profile replicated laterally.
        The xy plane returns a uniform array (no lateral variation in TMM).

        Coordinates follow the COMPASS frame used by the other solvers: z is
        measured from the silicon bottom (z=0) upward, so ``position`` of an
        "xy" plane is a COMPASS height and xz/yz columns run bottom to top.
        The incident plane wave has |E0| = 1; an unpolarized source averages
        the TE and TM intensities.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
                For TMM, only "|E|2" is computed; others return zeros.
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um (a COMPASS
                height for "xy"; ignored for "xz"/"yz").

        Returns:
            ``(64, 64)`` for "xy"; ``(64, field_resolution)`` for "xz"/"yz"
            with z increasing upward.

        Raises:
            ValueError: Unknown plane, or an "xy" position outside the stack.
        """
        if plane not in ("xy", "xz", "yz"):
            raise ValueError(f"Unknown plane {plane!r}; expected 'xy', 'xz' or 'yz'")
        nz = self._field_resolution
        nx_out = 64
        empty = np.zeros((nx_out, nx_out)) if plane == "xy" else np.zeros((nx_out, nz))

        if (
            self._last_n_layers is None
            or self._last_d_layers is None
            or self._source is None
            or self._pixel_stack is None
        ):
            logger.warning("TMM: no simulation data, returning zeros")
            return empty

        if component != "|E|2":
            # TMM only computes |E|^2 profile
            logger.info(f"TMM: component '{component}' not available, returning zeros")
            return empty

        z_min, z_max = self._pixel_stack.z_range
        if plane == "xy":
            if not z_min <= position <= z_max:
                raise ValueError(
                    f"xy-plane position z={position} um is outside the stack [{z_min}, {z_max}]"
                )
            z_compass = np.array([float(position)])
        else:
            z_compass = np.linspace(z_min, z_max, nz)
        # TMM depth is 0 at the top of the stack (incident side) and grows
        # toward the silicon, i.e. opposite to COMPASS z.
        depth = z_max - z_compass

        wl = self._last_wavelength or 0.55
        theta_rad = self._source.theta_rad
        pol_runs = self._source.get_polarization_runs()
        intensity_1d = np.mean(
            [
                tmm_field_profile(
                    self._last_n_layers, self._last_d_layers, wl, theta_rad, pol, depth
                )
                for pol in pol_runs
            ],
            axis=0,
        )

        if plane == "xy":
            # Uniform field in xy plane (no lateral variation)
            return np.full((nx_out, nx_out), float(intensity_1d[0]))
        # Replicate 1D z-profile across lateral dimension
        return np.tile(intensity_1d, (nx_out, 1))


# Register with SolverFactory
SolverFactory.register("tmm", TMMSolver)
