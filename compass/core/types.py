"""Core data types for COMPASS simulation results and structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LayerSlice:
    """A single z-slice of the pixel stack for RCWA solvers.

    Attributes:
        z_start: Bottom z-coordinate in um.
        z_end: Top z-coordinate in um.
        thickness: Layer thickness in um.
        eps_grid: 2D complex permittivity array of shape (nx, ny).
        name: Layer name for identification.
        material: Base material name.
    """

    z_start: float
    z_end: float
    thickness: float
    eps_grid: np.ndarray
    name: str = ""
    material: str = ""


@dataclass
class FieldData:
    """Electromagnetic field data container.

    Attributes:
        Ex: x-component of electric field (3D complex array).
        Ey: y-component of electric field.
        Ez: z-component of electric field.
        x: x-coordinate array.
        y: y-coordinate array.
        z: z-coordinate array.
    """

    Ex: Optional[np.ndarray] = None
    Ey: Optional[np.ndarray] = None
    Ez: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None

    @property
    def E_intensity(self) -> Optional[np.ndarray]:
        """Compute |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2."""
        components = [c for c in [self.Ex, self.Ey, self.Ez] if c is not None]
        if not components:
            return None
        return sum(np.abs(c) ** 2 for c in components)


@dataclass
class SimulationResult:
    """Solver-agnostic simulation result container.

    Attributes:
        qe_per_pixel: Mapping of pixel name (e.g. "R_0_0") to QE spectrum array.
        wavelengths: Wavelength array in um.
        fields: Optional field data per wavelength.
        reflection: Reflection spectrum R(lambda).
        transmission: Transmission spectrum T(lambda).
        absorption: Absorption spectrum A(lambda).
        metadata: Solver info, timing, convergence data.
    """

    qe_per_pixel: Dict[str, np.ndarray]
    wavelengths: np.ndarray
    fields: Optional[Dict[str, FieldData]] = None
    poynting: Optional[Dict[str, np.ndarray]] = None
    reflection: Optional[np.ndarray] = None
    transmission: Optional[np.ndarray] = None
    absorption: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Layer:
    """Generic layer in the pixel stack.

    Attributes:
        name: Layer identifier.
        z_start: Bottom z-coordinate in um.
        z_end: Top z-coordinate in um.
        thickness: Layer thickness in um.
        base_material: Material name.
        is_patterned: Whether the layer has lateral patterns.
    """

    name: str
    z_start: float
    z_end: float
    thickness: float
    base_material: str
    is_patterned: bool = False


@dataclass
class MicrolensSpec:
    """Microlens geometry specification.

    Attributes:
        height: Lens height in um.
        radius_x: Semi-axis in x direction in um.
        radius_y: Semi-axis in y direction in um.
        material: Lens material name.
        profile_type: Profile type ("superellipse", "spherical").
        n_param: Superellipse squareness parameter.
        alpha_param: Curvature parameter.
        shift_x: CRA-dependent x offset in um.
        shift_y: CRA-dependent y offset in um.
    """

    height: float
    radius_x: float
    radius_y: float
    material: str
    profile_type: str = "superellipse"
    n_param: float = 2.5
    alpha_param: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0


@dataclass
class PhotodiodeSpec:
    """Photodiode region definition.

    Attributes:
        position: (x, y, z) relative to pixel center in um.
        size: (dx, dy, dz) in um.
        pixel_index: (row, col) in unit cell.
        color: Color channel (R, G, B).
    """

    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    pixel_index: Tuple[int, int]
    color: str = ""
