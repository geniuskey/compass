"""Cone illumination runner — angular averaging over exit pupil."""

from __future__ import annotations

import copy
import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory
from compass.sources.cone_illumination import ConeIllumination

logger = logging.getLogger(__name__)


class ConeIlluminationRunner:
    """Run solver at multiple angles from ConeIllumination and accumulate weighted results.

    Integrates ConeIllumination angular sampling with any registered solver.
    For each sampled angle, modifies the source config and runs the solver,
    then accumulates R, T, A and per-pixel QE with cosine-weighted averaging.
    """

    @staticmethod
    def run(config: dict) -> SimulationResult:
        """Execute cone-averaged simulation.

        Args:
            config: Full simulation config dict with pixel, solver, source, compute keys.
                The source config must contain a 'cone' section.

        Returns:
            SimulationResult with weighted-average R, T, A and QE.
        """
        solver_config = config.get("solver", {})
        source_config = config.get("source", {})
        compute_config = config.get("compute", {})
        cone_config = source_config.get("cone", {})

        # Build ConeIllumination from config
        sampling_cfg = cone_config.get("sampling", {})
        cone = ConeIllumination(
            cra_deg=cone_config.get("cra_deg", 0.0),
            f_number=cone_config.get("f_number", 2.0),
            n_points=sampling_cfg.get("n_points", 37),
            sampling=sampling_cfg.get("type", "fibonacci"),
            weighting=cone_config.get("weighting", "cosine"),
        )

        # Determine device
        device = compute_config.get("backend", "cpu")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Build pixel stack
        material_db = MaterialDB()
        pixel_stack = PixelStack(config, material_db)

        # Create solver
        solver_name = solver_config.get("name", "torcwa")
        solver = SolverFactory.create(solver_name, solver_config, device)
        solver.setup_geometry(pixel_stack)

        # Get angular sampling points
        points = cone.get_sampling_points()
        logger.info(
            f"ConeIlluminationRunner: {len(points)} angular points, "
            f"F/{cone.f_number}, CRA={cone.cra_deg}°"
        )

        # Accumulators
        weighted_R: np.ndarray | None = None
        weighted_T: np.ndarray | None = None
        weighted_A: np.ndarray | None = None
        weighted_qe: dict[str, np.ndarray] = {}
        wavelengths: np.ndarray | None = None

        for i, (theta_deg, phi_deg, weight) in enumerate(points):
            logger.debug(
                f"  angle {i + 1}/{len(points)}: θ={theta_deg:.2f}°, "
                f"φ={phi_deg:.2f}°, w={weight:.4f}"
            )

            # Build source config for this angle
            angle_source_config = copy.deepcopy(source_config)
            angle_source_config.setdefault("angle", {})
            angle_source_config["angle"]["theta_deg"] = theta_deg
            angle_source_config["angle"]["phi_deg"] = phi_deg

            solver.setup_source(angle_source_config)
            result = solver.run()

            if wavelengths is None:
                n_wl = len(result.wavelengths)
                wavelengths = result.wavelengths
                weighted_R = np.zeros(n_wl)
                weighted_T = np.zeros(n_wl)
                weighted_A = np.zeros(n_wl)

            if result.reflection is not None:
                weighted_R += weight * result.reflection
            if result.transmission is not None:
                weighted_T += weight * result.transmission
            if result.absorption is not None:
                weighted_A += weight * result.absorption

            for key, qe_arr in result.qe_per_pixel.items():
                if key not in weighted_qe:
                    weighted_qe[key] = np.zeros_like(qe_arr)
                weighted_qe[key] += weight * qe_arr

        if wavelengths is None:
            raise RuntimeError("No angular sampling points produced results")

        return SimulationResult(
            qe_per_pixel=weighted_qe,
            wavelengths=wavelengths,
            reflection=weighted_R,
            transmission=weighted_T,
            absorption=weighted_A,
            metadata={
                "solver_name": solver_name,
                "device": device,
                "runner": "ConeIlluminationRunner",
                "cone_f_number": cone.f_number,
                "cone_cra_deg": cone.cra_deg,
                "cone_n_points": len(points),
                "cone_sampling": cone.sampling,
                "cone_weighting": cone.weighting,
            },
        )
