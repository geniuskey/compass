"""Solver-independent helpers for RCWA field slices and field-based QE.

The RCWA adapters (torcwa, grcwa, meent, fmmax) share the same conventions
once their fields are expressed on a common footing:

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


def backend_grid_shift(pixel_stack, layer_info: Sequence[LayerSlice]) -> tuple[float, float]:
    """(dx, dy) in um from COMPASS to an FFT backend's eps-sample frame.

    PixelStack samples eps at cell centres, ``(b + 0.5) * lx / n``, while
    torcwa and grcwa take sample ``b`` of the grid to sit at ``b * lx / n``.
    A COMPASS position ``x`` therefore maps to backend ``x - dx``.
    """
    lx, ly = pixel_stack.domain_size
    n0, n1 = layer_info[0].eps_grid.shape
    return 0.5 * lx / n1, 0.5 * ly / n0


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
    slice_absorption: Callable[[int, LayerSlice], float] | None = None,
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
        slice_absorption: Optional callback ``(index, slice) -> fraction``
            giving the exact fraction of the incident power absorbed in a
            slice (from the Fourier-space Poynting flux at its faces). When
            given, each slice's density is rescaled to that total, so only
            the in-plane distribution comes from the real-space field. Use
            it for backends whose permittivity factorization (Li's inverse
            rule) makes the truncated real-space ``|E|^2`` inexact.

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
        raw = norm_cell * float(density.sum()) * cell_area
        if slice_absorption is not None:
            exact = max(float(slice_absorption(li, s)), 0.0)
            density = density * (exact / raw) if raw > 1e-15 else density * 0.0
            logger.debug(f"{solver_name}: slice {s.name}: field {raw:.4f}, flux {exact:.4f}")
            raw = exact
        total_field_abs += raw
        if s.name.startswith("silicon"):
            if grid_shape is None:
                grid_shape = (n0, n1)
            si_density = density if si_density is None else si_density + density

    if si_density is None or grid_shape is None:
        raise RuntimeError("no absorbing silicon slice found for QE integration")

    logger.debug(
        f"{solver_name}: field-integrated absorption {total_field_abs:.4f}, "
        f"1-R-T = {total_absorption:.4f} at λ={wavelength:.4f}um"
    )
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


def eps_imag_weight_qe(
    pixel_stack,
    layer_slices: Sequence[LayerSlice],
    total_absorption: float,
) -> dict[str, float]:
    """Approximate per-pixel QE by distributing ``1 - R - T`` with eps'' weights.

    Fallback only, used when field reconstruction fails: it ignores where
    the light actually goes (no color filtering, no lensing), so it is not
    comparable to :func:`integrate_photodiode_qe`. Absorption is split
    between photodiodes in proportion to the integral of eps'' over each
    photodiode box, and normalized per pixel area.

    Args:
        pixel_stack: PixelStack providing domain size, pitch and photodiodes.
        layer_slices: Slices in any order (COMPASS z coordinates).
        total_absorption: A = 1 - R - T.

    Returns:
        Mapping of pixel name (e.g. "R_0_0") to QE.
    """
    n_rows, n_cols = pixel_stack.unit_cell
    n_pixels = n_rows * n_cols
    if n_pixels == 0:
        return {}
    lx, ly = pixel_stack.domain_size
    pitch = pixel_stack.pitch
    si_z_end = max(
        (layer.z_end for layer in pixel_stack.layers if layer.name == "silicon"),
        default=0.0,
    )

    pixel_weights: dict[str, float] = {}
    for pd in pixel_stack.photodiodes:
        r, c = pd.pixel_index
        # position[2] is the PD-center depth below the silicon top surface
        pd_cz = si_z_end - pd.position[2]
        pd_z_min = pd_cz - pd.size[2] / 2
        pd_z_max = pd_cz + pd.size[2] / 2
        # PhotodiodeSpec.position is the offset from the pixel center.
        pd_cx = (c + 0.5) * pitch + pd.position[0]
        pd_cy = (r + 0.5) * pitch + pd.position[1]

        weight = 0.0
        for s in layer_slices:
            dz = min(s.z_end, pd_z_max) - max(s.z_start, pd_z_min)
            if dz <= 0:
                continue
            # eps grids are indexed [row=y, col=x] (shape (ny, nx))
            ny_s, nx_s = s.eps_grid.shape
            ix_min = max(0, int(((pd_cx - pd.size[0] / 2) / lx) * nx_s))
            ix_max = min(nx_s, int(np.ceil(((pd_cx + pd.size[0] / 2) / lx) * nx_s)))
            iy_min = max(0, int(((pd_cy - pd.size[1] / 2) / ly) * ny_s))
            iy_max = min(ny_s, int(np.ceil(((pd_cy + pd.size[1] / 2) / ly) * ny_s)))
            if ix_max <= ix_min or iy_max <= iy_min:
                continue
            eps_region = s.eps_grid[iy_min:iy_max, ix_min:ix_max]
            area_frac = (ix_max - ix_min) * (iy_max - iy_min) / (nx_s * ny_s)
            weight += float(np.mean(np.imag(eps_region))) * dz * area_frac
        pixel_weights[f"{pd.color}_{r}_{c}"] = max(weight, 0.0)

    total_weight = sum(pixel_weights.values())
    if total_weight > 0:
        return {
            key: min(1.0, total_absorption * (w / total_weight) * n_pixels)
            for key, w in pixel_weights.items()
        }
    # Uniform stack: every pixel sees the cell-average absorption
    bayer = pixel_stack.bayer_map
    return {
        f"{bayer[r][c]}_{r}_{c}": total_absorption
        for r in range(n_rows)
        for c in range(n_cols)
    }


def approximate_field(
    pixel_stack,
    layer_info: Sequence[LayerSlice],
    wavelength: float | None,
    component: str,
    plane: str,
    position: float,
) -> np.ndarray:
    """Permittivity-based stand-in for a field slice (NOT a solved field).

    Used only when field reconstruction fails. "xy" returns Re(eps) of the
    slice at height ``position``; "xz"/"yz" return, per slice, either the
    absorption index k (components other than "|E|2") or a Beer-Lambert
    decay ``exp(-2 k0 sum k dz)`` from the top of the stack ("|E|2").
    Coordinates follow the solved-field convention: x, y in
    ``[0, lx) x [0, ly)``, columns ordered bottom to top.

    Returns:
        ``(64, 64)`` for "xy" (indexed [x, y]); ``(64, n_slices)`` otherwise.
    """
    validate_field_request(component, plane)
    n_out = FIELD_N_XY
    lx, ly = pixel_stack.domain_size
    top_first = sorted(layer_info, key=lambda s: -s.z_end)
    u = (np.arange(n_out) + 0.5) / n_out

    if plane == "xy":
        li, _ = locate_slice(top_first, position)
        eps = top_first[li].eps_grid  # [y, x]
        n0, n1 = eps.shape
        ix = np.minimum((u * n1).astype(int), n1 - 1)
        iy = np.minimum((u * n0).astype(int), n0 - 1)
        return np.asarray(np.abs(np.real(eps[np.ix_(iy, ix)].T)))

    cols = []
    for s in top_first:
        eps = s.eps_grid
        n0, n1 = eps.shape
        if plane == "xz":
            row = min(max(int(position / ly * n0), 0), n0 - 1)
            line = eps[row, np.minimum((u * n1).astype(int), n1 - 1)]
        else:
            col = min(max(int(position / lx * n1), 0), n1 - 1)
            line = eps[np.minimum((u * n0).astype(int), n0 - 1), col]
        cols.append(np.abs(np.imag(np.sqrt(line))))
    kappa = np.stack(cols, axis=1)  # (n_out, n_slices), top first

    if component == "|E|2":
        k0 = 2 * np.pi / (wavelength or 0.55)
        dz = np.array([s.thickness for s in top_first])
        kappa = np.exp(-2 * k0 * np.cumsum(kappa * dz, axis=1))
    return np.asarray(kappa[:, ::-1])
