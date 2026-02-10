"""Core data types for COMPASS simulation results and structures."""

from __future__ import annotations

from dataclasses import dataclass, field

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

    Ex: np.ndarray | None = None
    Ey: np.ndarray | None = None
    Ez: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None

    @property
    def E_intensity(self) -> np.ndarray | None:
        """Compute |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2."""
        components = [c for c in [self.Ex, self.Ey, self.Ez] if c is not None]
        if not components:
            return None
        return np.asarray(sum(np.abs(c) ** 2 for c in components))


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

    qe_per_pixel: dict[str, np.ndarray]
    wavelengths: np.ndarray
    fields: dict[str, FieldData] | None = None
    poynting: dict[str, np.ndarray] | None = None
    reflection: np.ndarray | None = None
    transmission: np.ndarray | None = None
    absorption: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


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

    position: tuple[float, float, float]
    size: tuple[float, float, float]
    pixel_index: tuple[int, int]
    color: str = ""
