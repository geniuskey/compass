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

    def __init__(
        self,
        cra_deg: float = 0.0,
        f_number: float = 2.0,
        n_points: int = 37,
        sampling: str = "fibonacci",
        weighting: str = "cosine",
    ):
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
        sampling = str(self.sampling).lower().replace("-", "_")
        if sampling in {"fibonacci", "sunflower", "golden_angle"}:
            return self._fibonacci_sampling()
        elif sampling == "grid":
            return self._grid_sampling()
        elif sampling in {"gauss", "gaussian_quadrature", "legendre"}:
            return self._gaussian_quadrature_sampling()
        elif sampling in {"rings", "concentric", "concentric_rings"}:
            return self._ring_sampling()
        elif sampling in {"halton", "low_discrepancy"}:
            return self._halton_sampling()
        elif sampling in {"hammersley", "hammersley_point_set"}:
            return self._hammersley_sampling()
        else:
            return self._fibonacci_sampling()

    def _fibonacci_sampling(self) -> list[tuple[float, float, float]]:
        """Fibonacci spiral sampling on the cone cap."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        n_points = max(1, int(self.n_points))
        for i in range(n_points):
            # Equal-area samples on a spherical cap: u is uniform over cap area.
            u = (i + 0.5) / n_points
            theta_local = np.arccos(1.0 - u * (1.0 - np.cos(self.half_cone_rad)))
            phi = 2 * np.pi * i / golden_ratio
            weight = self._compute_weight(theta_local)
            points.append((*self._global_angles_deg(theta_local, phi), weight))
        return self._normalize(points)

    def _grid_sampling(self) -> list[tuple[float, float, float]]:
        """Uniform grid sampling."""
        n_theta = max(int(np.sqrt(self.n_points)), 3)
        n_phi = max(self.n_points // n_theta, 4)
        points = []
        for i in range(n_theta):
            theta_local = self.half_cone_rad * (i + 0.5) / n_theta
            for j in range(n_phi):
                phi = 2 * np.pi * j / n_phi
                weight = self._compute_weight(theta_local) * np.sin(theta_local + 1e-10)
                points.append((*self._global_angles_deg(theta_local, phi), weight))
        return self._normalize(points)

    def _gaussian_quadrature_sampling(self) -> list[tuple[float, float, float]]:
        """Gauss-Legendre quadrature in theta, uniform in phi.

        Uses numpy's Gauss-Legendre nodes and weights mapped to the
        interval [0, half_cone_rad] for the polar angle, and a uniform
        grid in the azimuthal direction.
        """
        n_theta = max(int(np.sqrt(self.n_points)), 3)
        n_phi = max(self.n_points // n_theta, 4)

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
                weight = tw * phi_weight * self._compute_weight(theta_local)
                weight *= np.sin(theta_local + 1e-10)  # Jacobian factor
                points.append((*self._global_angles_deg(theta_local, phi), weight))

        return self._normalize(points)

    def _ring_sampling(self) -> list[tuple[float, float, float]]:
        """Equal-area concentric-ring sampling on the cone cap.

        Compared with a rectangular polar grid, rings allocate more azimuth
        samples near the edge where circumference is larger. This gives a
        deterministic pattern that is easier to inspect than low-discrepancy
        points while avoiding the strong center bias of a naive theta/phi grid.
        """
        n_points = max(1, int(self.n_points))
        n_rings = max(1, round(np.sqrt(n_points)))
        counts = self._ring_counts(n_points, n_rings)
        cap = 1.0 - np.cos(self.half_cone_rad)

        points: list[tuple[float, float, float]] = []
        for ring_idx, count in enumerate(counts):
            u_inner = ring_idx / n_rings
            u_outer = (ring_idx + 1) / n_rings
            u_mid = 0.5 * (u_inner + u_outer)
            theta_inner = np.arccos(1.0 - u_inner * cap)
            theta_outer = np.arccos(1.0 - u_outer * cap)
            theta_local = np.arccos(1.0 - u_mid * cap)
            ring_solid_angle = max(np.cos(theta_inner) - np.cos(theta_outer), 1e-12)
            sample_weight = ring_solid_angle * self._compute_weight(theta_local) / count
            offset = 0.5 * (ring_idx % 2)

            for j in range(count):
                phi = 2 * np.pi * ((j + offset) / count)
                points.append((*self._global_angles_deg(theta_local, phi), sample_weight))

        return self._normalize(points)

    def _halton_sampling(self) -> list[tuple[float, float, float]]:
        """Low-discrepancy Halton sampling on the cone cap."""
        points: list[tuple[float, float, float]] = []
        n_points = max(1, int(self.n_points))
        cap = 1.0 - np.cos(self.half_cone_rad)
        for i in range(n_points):
            u = self._radical_inverse(i + 1, 2)
            v = self._radical_inverse(i + 1, 3)
            theta_local = np.arccos(1.0 - u * cap)
            phi = 2 * np.pi * v
            weight = self._compute_weight(theta_local)
            points.append((*self._global_angles_deg(theta_local, phi), weight))
        return self._normalize(points)

    def _hammersley_sampling(self) -> list[tuple[float, float, float]]:
        """Fixed-budget Hammersley point-set sampling on the cone cap."""
        points: list[tuple[float, float, float]] = []
        n_points = max(1, int(self.n_points))
        cap = 1.0 - np.cos(self.half_cone_rad)
        for i in range(n_points):
            u = (i + 0.5) / n_points
            v = self._radical_inverse(i + 1, 2)
            theta_local = np.arccos(1.0 - u * cap)
            phi = 2 * np.pi * v
            weight = self._compute_weight(theta_local)
            points.append((*self._global_angles_deg(theta_local, phi), weight))
        return self._normalize(points)

    def _global_angles_deg(self, theta_local: float, phi_local: float) -> tuple[float, float]:
        """Rotate a local cone sample around the chief ray into global angles."""
        cra_rad = deg_to_rad(self.cra_deg)
        chief = np.array([np.sin(cra_rad), 0.0, np.cos(cra_rad)])
        basis_x = np.array([np.cos(cra_rad), 0.0, -np.sin(cra_rad)])
        basis_y = np.array([0.0, 1.0, 0.0])
        direction = np.cos(theta_local) * chief + np.sin(theta_local) * (
            np.cos(phi_local) * basis_x + np.sin(phi_local) * basis_y
        )
        direction = direction / np.linalg.norm(direction)
        theta = np.arccos(np.clip(direction[2], -1.0, 1.0))
        phi = np.mod(np.arctan2(direction[1], direction[0]), 2 * np.pi)
        return float(np.degrees(theta)), float(np.degrees(phi))

    @staticmethod
    def _normalize(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        total_w = sum(max(0.0, p[2]) for p in points)
        if total_w <= 0.0:
            n_points = max(1, len(points))
            return [(t, p, 1.0 / n_points) for t, p, _ in points]
        return [(t, p, max(0.0, w) / total_w) for t, p, w in points]

    @staticmethod
    def _ring_counts(n_points: int, n_rings: int) -> list[int]:
        weights = np.arange(1, n_rings + 1, dtype=float)
        raw = n_points * weights / weights.sum()
        counts = [max(1, round(v)) for v in raw]
        while sum(counts) < n_points:
            counts[-1] += 1
        while sum(counts) > n_points:
            for idx in range(len(counts) - 1, -1, -1):
                if counts[idx] > 1:
                    counts[idx] -= 1
                    break
        return counts

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        result = 0.0
        inv_base = 1.0 / base
        fraction = inv_base
        while index > 0:
            result += fraction * (index % base)
            index //= base
            fraction *= inv_base
        return result

    def _compute_weight(self, theta: float) -> float:
        if self.weighting == "uniform":
            return 1.0
        elif self.weighting == "cosine":
            return float(np.cos(theta))
        elif self.weighting == "cos4":
            return float(np.cos(theta) ** 4)
        elif self.weighting == "gaussian":
            sigma = self.half_cone_rad / 2
            return float(np.exp(-(theta**2) / (2 * sigma**2)))
        elif self.weighting == "custom":
            # Default to uniform when no callable is provided via string
            return 1.0
        elif callable(self.weighting):
            return float(self.weighting(theta))
        return 1.0
