"""Cone illumination model — exit pupil illumination."""
from __future__ import annotations

import logging

import numpy as np

from compass.core.units import deg_to_rad

logger = logging.getLogger(__name__)

class ConeIllumination:
    """Model cone illumination from lens exit pupil.

    CRA (Chief Ray Angle) and F-number define the illumination cone.
    Angular sampling generates planewave directions for weighted summation.
    """

    def __init__(self, cra_deg: float = 0.0, f_number: float = 2.0, n_points: int = 37, sampling: str = "fibonacci", weighting: str = "cosine"):
        self.cra_deg = cra_deg
        self.f_number = f_number
        self.n_points = n_points
        self.sampling = sampling
        self.weighting = weighting
        self.half_cone_rad = np.arcsin(1.0 / (2.0 * f_number))

    def get_sampling_points(self) -> list[tuple[float, float, float]]:
        """Generate angular sampling points with weights.
        Returns: list of (theta_deg, phi_deg, weight)
        """
        if self.sampling == "fibonacci":
            return self._fibonacci_sampling()
        elif self.sampling == "grid":
            return self._grid_sampling()
        elif self.sampling == "gauss":
            return self._gaussian_quadrature_sampling()
        else:
            return self._fibonacci_sampling()

    def _fibonacci_sampling(self) -> list[tuple[float, float, float]]:
        """Fibonacci spiral sampling on hemisphere cone."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        cra_rad = deg_to_rad(self.cra_deg)
        for i in range(self.n_points):
            theta_frac = np.sqrt((i + 0.5) / self.n_points)
            theta_local = theta_frac * self.half_cone_rad
            phi = 2 * np.pi * i / golden_ratio
            theta = np.sqrt((cra_rad * np.cos(phi) + theta_local * np.cos(phi))**2 + (cra_rad * np.sin(phi) + theta_local * np.sin(phi))**2)
            theta = min(theta_local + abs(cra_rad), np.pi/2)
            weight = self._compute_weight(theta_local)
            points.append((np.degrees(theta), np.degrees(phi), weight))
        total_w = sum(p[2] for p in points)
        return [(t, p, w / total_w) for t, p, w in points]

    def _grid_sampling(self) -> list[tuple[float, float, float]]:
        """Uniform grid sampling."""
        n_theta = max(int(np.sqrt(self.n_points)), 3)
        n_phi = max(self.n_points // n_theta, 4)
        cra_rad = deg_to_rad(self.cra_deg)
        points = []
        for i in range(n_theta):
            theta_local = self.half_cone_rad * (i + 0.5) / n_theta
            for j in range(n_phi):
                phi = 2 * np.pi * j / n_phi
                theta = theta_local + cra_rad
                weight = self._compute_weight(theta_local) * np.sin(theta_local + 1e-10)
                points.append((np.degrees(theta), np.degrees(phi), weight))
        total_w = sum(p[2] for p in points)
        return [(t, p, w / total_w) for t, p, w in points]

    def _gaussian_quadrature_sampling(self) -> list[tuple[float, float, float]]:
        """Gauss-Legendre quadrature in theta, uniform in phi.

        Uses numpy's Gauss-Legendre nodes and weights mapped to the
        interval [0, half_cone_rad] for the polar angle, and a uniform
        grid in the azimuthal direction.
        """
        n_theta = max(int(np.sqrt(self.n_points)), 3)
        n_phi = max(self.n_points // n_theta, 4)
        cra_rad = deg_to_rad(self.cra_deg)

        # Gauss-Legendre nodes on [-1, 1], mapped to [0, half_cone_rad]
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_theta)
        # Map from [-1, 1] to [0, half_cone_rad]
        theta_nodes = 0.5 * self.half_cone_rad * (gl_nodes + 1.0)
        theta_weights = 0.5 * self.half_cone_rad * gl_weights

        phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        phi_weight = 1.0 / n_phi

        points: list[tuple[float, float, float]] = []
        for _ti, (theta_local, tw) in enumerate(zip(theta_nodes, theta_weights)):
            for phi in phi_vals:
                theta = theta_local + cra_rad
                weight = tw * phi_weight * self._compute_weight(theta_local)
                weight *= np.sin(theta_local + 1e-10)  # Jacobian factor
                points.append((np.degrees(theta), np.degrees(phi), weight))

        total_w = sum(p[2] for p in points)
        if total_w > 0:
            points = [(t, p, w / total_w) for t, p, w in points]
        return points

    def _compute_weight(self, theta: float) -> float:
        if self.weighting == "uniform":
            return 1.0
        elif self.weighting == "cosine":
            return float(np.cos(theta))
        elif self.weighting == "cos4":
            return float(np.cos(theta)**4)
        elif self.weighting == "gaussian":
            sigma = self.half_cone_rad / 2
            return float(np.exp(-theta**2 / (2 * sigma**2)))
        elif self.weighting == "custom":
            # Default to uniform when no callable is provided via string
            return 1.0
        elif callable(self.weighting):
            return float(self.weighting(theta))
        return 1.0
