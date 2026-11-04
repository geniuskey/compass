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
        self._polarization_average: bool = config.get("params", {}).get(
            "polarization_average", True
        )
        # "air": finite silicon over an air half-space, the same stack the
        # RCWA / FDTD adapters simulate. "silicon": silicon as a semi-infinite
        # substrate (all transmitted light is absorbed; QE is an upper bound).
        self._substrate: str = config.get("params", {}).get("substrate", "air")
        if self._substrate not in ("air", "silicon"):
            raise ValueError(f"tmm substrate must be 'air' or 'silicon', got {self._substrate!r}")
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

        # Override polarization averaging from config if specified
        if self._polarization_average and self._source.polarization != "unpolarized":
            self._polarization_average = False

        logger.info(
            f"TMM: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg:.1f} deg"
        )

    def _get_n_layers_at_wavelength(self, wavelength: float) -> np.ndarray:
        """Get complex refractive indices for all layers at a given wavelength.

        Args:
            wavelength: Wavelength in um.

        Returns:
            Array of complex n + ik for each layer.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        material_db = self._pixel_stack.material_db

        n_complex = np.zeros(len(self._layer_materials), dtype=complex)
        for i, mat_name in enumerate(self._layer_materials):
            n, k = material_db.get_nk(mat_name, wavelength)
            n_complex[i] = n + 1j * k

        return n_complex

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

    def _build_d_layers(self) -> np.ndarray:
        """Build the thickness array for TMM.

        The first and last layers are treated as semi-infinite.
        Their thicknesses are set to np.inf.

        Returns:
            Thickness array with inf at boundaries.
        """
        d = self._layer_thicknesses.copy()
        # First layer (bottom = silicon in BSI) and last layer (top = air)
        # are semi-infinite in TMM. However, COMPASS convention has light
        # entering from air (top/last layer) through the stack to silicon
        # (bottom/first layer).
        #
        # For TMM: incident medium = air (top), substrate = silicon (bottom).
        # We need to reverse the layer order for TMM convention where
        # layer[0] = incident medium, layer[-1] = substrate.
        d_tmm = d[::-1].copy()
        d_tmm[0] = np.inf  # incident medium (air)
        if self._substrate == "air":
            d_tmm = np.append(d_tmm, np.inf)  # finite silicon, air half-space below
        else:
            d_tmm[-1] = np.inf  # semi-infinite silicon substrate
        return d_tmm

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
        """Extract field profile from the TMM solution.

        For TMM, only the z-profile (through the stack) is physically meaningful.
        The xz and yz planes show the 1D field profile replicated laterally.
        The xy plane returns a uniform array (no lateral variation in TMM).

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
                For TMM, only "|E|2" is computed; others return zeros.
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        if self._last_n_layers is None or self._last_d_layers is None:
            logger.warning("TMM: no simulation data, returning zeros")
            return np.zeros((64, 64))

        if self._source is None:
            return np.zeros((64, 64))

        nz = self._field_resolution
        nx_out = 64

        if component != "|E|2":
            # TMM only computes |E|^2 profile
            logger.info(f"TMM: component '{component}' not available, returning zeros")
            return np.zeros((nx_out, nz))

        # Compute z_points spanning the internal layers (skip semi-infinite boundaries)
        # Sum of internal layer thicknesses
        internal_d = self._last_d_layers[1:-1]
        total_internal = float(np.sum(internal_d))

        if total_internal <= 0:
            return np.zeros((nx_out, nz))

        z_points = np.linspace(0, total_internal, nz)

        wl = self._last_wavelength or 0.55
        theta_rad = self._source.theta_rad
        pol = self._source.polarization if self._source.polarization != "unpolarized" else "TE"

        intensity_1d = tmm_field_profile(
            self._last_n_layers,
            self._last_d_layers,
            wl,
            theta_rad,
            pol,
            z_points,
        )

        if plane == "xy":
            # Uniform field in xy plane (no lateral variation)
            val = np.interp(
                position, z_points, intensity_1d, left=intensity_1d[0], right=intensity_1d[-1]
            )
            return np.full((nx_out, nx_out), val)
        elif plane in ("xz", "yz"):
            # Replicate 1D z-profile across lateral dimension
            field_2d = np.tile(intensity_1d, (nx_out, 1))
            return field_2d
        else:
            return np.zeros((nx_out, nz))


# Register with SolverFactory
SolverFactory.register("tmm", TMMSolver)
