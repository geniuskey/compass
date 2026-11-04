"""Solver-independent helpers for RCWA field slices and field-based QE.

torcwa and grcwa share the same conventions once their fields are
expressed on a common footing:

* Incident plane wave of unit amplitude (|E0| = 1) from the top (air).
* The backend z axis points *down* from the top interface of the stack,
  while COMPASS z points up from the silicon bottom.
* With lengths in um, ``k0 = 2 pi / wavelength`` and the time-averaged
  Poynting z-component is ``Re(Ex Hy* - Ey Hx*)`` (incident flux through a
  z-plane equals ``cos(theta)``).

Backends only need to supply the depth-integrated ``|E|^2`` on each
absorbing slice's permittivity grid; QE integration over photodiode
footprints is shared here so every solver reports the same QE definition.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from compass.core.types import LayerSlice

logger = logging.getLogger(__name__)

#: Supported field components and slice planes.
FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "|E|2", "Sz")
FIELD_PLANES = ("xy", "xz", "yz")
#: Samples per in-plane axis and along z for solved-field slices.
FIELD_N_XY = 64
FIELD_N_Z = 128


def validate_field_request(component: str, plane: str) -> None:
    """Raise ValueError for unsupported component/plane names."""
    if component not in FIELD_COMPONENTS:
        raise ValueError(
            f"Unknown field component {component!r}; expected one of {FIELD_COMPONENTS}"
        )
    if plane not in FIELD_PLANES:
        raise ValueError(f"Unknown plane {plane!r}; expected one of {FIELD_PLANES}")


def stack_z_range(layer_info: Sequence[LayerSlice]) -> tuple[float, float]:
    """Return (z_bottom, z_top) of the simulated slices in COMPASS z."""
    return min(s.z_start for s in layer_info), max(s.z_end for s in layer_info)


def check_xy_position(layer_info: Sequence[LayerSlice], position: float) -> None:
    """Raise ValueError if an xy-plane height lies outside the stack."""
    z_lo, z_hi = stack_z_range(layer_info)
    if not z_lo <= position <= z_hi:
        raise ValueError(
            f"xy-plane position z={position} um is outside the stack [{z_lo}, {z_hi}]"
        )


def locate_slice(layer_info: Sequence[LayerSlice], z: float) -> tuple[int, float]:
    """Find the slice containing COMPASS height ``z``.

    Args:
        layer_info: Slices in backend order (top of the stack first).
        z: Height in um (COMPASS frame).

    Returns:
        (index into ``layer_info``, depth below that slice's top in um).
    """
    for li, s in enumerate(layer_info):
        if s.z_start <= z <= s.z_end:
            return li, float(s.z_end - z)
    raise ValueError(f"z={z} um is outside the simulated stack")


def column_heights(layer_info: Sequence[LayerSlice], n_z: int = FIELD_N_Z) -> np.ndarray:
    """COMPASS heights sampled for xz/yz slices, bottom to top."""
    z_bot, z_top = stack_z_range(layer_info)
    return np.linspace(z_bot, z_top, n_z)


def slice_depth_samples(thickness: float) -> tuple[np.ndarray, float]:
    """Midpoint depths (from the slice top) for integrating over a slice."""
    nz = int(np.clip(round(thickness / 0.05), 8, 64))
    dz = thickness / nz
    return (np.arange(nz) + 0.5) * dz, dz


def component_from_field(E: Sequence, H: Sequence, component: str) -> np.ndarray:
    """Reduce [Ex, Ey, Ez] / [Hx, Hy, Hz] (backend frame, z down) to one quantity.

    ``Sz`` is returned in the COMPASS frame (z up), so downward power flow
    is negative.
    """
    if component in ("Ex", "Ey", "Ez"):
        return np.asarray(np.abs(E["xyz".index(component[1])]) ** 2)
    if component == "|E|2":
        return np.asarray(sum(np.abs(c) ** 2 for c in E))
    if component == "Sz":
        s_down = np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
        return np.asarray(-s_down)
    raise ValueError(f"Unknown field component {component!r}")


def integrate_photodiode_qe(
    pixel_stack,
    layer_info: Sequence[LayerSlice],
    wavelength: float,
    theta_rad: float,
    e2_dz: Callable[[int, LayerSlice], np.ndarray],
    total_absorption: float,
    solver_name: str,
) -> dict[str, float]:
    """Per-pixel QE from the absorbed power density eps'' |E|^2.

    The absorbed power density is p(r) = (omega eps0 / 2) eps''(r) |E(r)|^2.
    Normalized by the incident power on one pixel (cos(theta) * pitch^2 for
    |E0| = 1, n_in = 1), in COMPASS units this gives

        QE_pd = k0 * sum_PD eps'' |E|^2 dV / (cos(theta) * A_pixel)

    QE counts absorption in silicon slices only (parasitic absorption in
    color filters / metal grid is excluded), integrated over the photodiode
    xy footprint and the full silicon depth.

    Args:
        pixel_stack: PixelStack providing domain size, pitch, unit cell and
            photodiodes.
        layer_info: Slices in backend order.
        wavelength: Wavelength in um.
        theta_rad: Polar incidence angle in radians.
        e2_dz: Callback ``(index, slice) -> array`` returning the
            depth-integrated ``|E|^2`` (um) on the slice's eps grid,
            indexed [row=y, col=x], with sample (a, b) at
            (b * lx / n1, a * ly / n0).
        total_absorption: A = 1 - R - T, for an energy cross-check.
        solver_name: Name used in log messages.

    Returns:
        Mapping of pixel name (e.g. "R_0_0") to QE.

    Raises:
        RuntimeError: If no absorbing silicon slice exists.
    """
    lx, ly = pixel_stack.domain_size
    pitch = pixel_stack.pitch
    k0 = 2.0 * np.pi / wavelength
    cos_theta = float(np.cos(theta_rad))
    norm_cell = k0 / (cos_theta * lx * ly)

    si_density = None
    total_field_abs = 0.0
    grid_shape = None
    for li, s in enumerate(layer_info):
        eps_imag = np.imag(s.eps_grid)
        if float(eps_imag.max()) < 1e-9:
            continue
        n0, n1 = s.eps_grid.shape
        density = np.asarray(e2_dz(li, s)) * eps_imag
        cell_area = (lx / n1) * (ly / n0)
        total_field_abs += norm_cell * float(density.sum()) * cell_area
        if s.name.startswith("silicon"):
            if grid_shape is None:
                grid_shape = (n0, n1)
            si_density = density if si_density is None else si_density + density

    if si_density is None or grid_shape is None:
        raise RuntimeError("no absorbing silicon slice found for QE integration")

    if total_absorption > 0.02 and abs(total_field_abs - total_absorption) > 0.1:
        logger.warning(
            f"{solver_name}: field-integrated absorption {total_field_abs:.3f} deviates "
            f"from 1-R-T = {total_absorption:.3f} at λ={wavelength:.4f}um "
            "(increase Fourier truncation / grid resolution)"
        )

    n0, n1 = grid_shape
    cell_area = (lx / n1) * (ly / n0)
    n_rows, n_cols = pixel_stack.unit_cell
    pixel_area = (lx / n_cols) * (ly / n_rows)
    norm_pixel = k0 / (cos_theta * pixel_area)
    qe_per_pixel: dict[str, float] = {}
    for pd in pixel_stack.photodiodes:
        r, c = pd.pixel_index
        # PhotodiodeSpec.position is the offset from the pixel center.
        x_c = (c + 0.5) * pitch + pd.position[0]
        y_c = (r + 0.5) * pitch + pd.position[1]
        iy0 = max(0, int(np.floor((y_c - pd.size[1] / 2) / ly * n0)))
        iy1 = min(n0, int(np.ceil((y_c + pd.size[1] / 2) / ly * n0)))
        ix0 = max(0, int(np.floor((x_c - pd.size[0] / 2) / lx * n1)))
        ix1 = min(n1, int(np.ceil((x_c + pd.size[0] / 2) / lx * n1)))
        key = f"{pd.color}_{r}_{c}"
        if ix1 <= ix0 or iy1 <= iy0:
            qe_per_pixel[key] = 0.0
            continue
        qe = norm_pixel * float(si_density[iy0:iy1, ix0:ix1].sum()) * cell_area
        qe_per_pixel[key] = float(np.clip(qe, 0.0, 1.0))
    return qe_per_pixel
