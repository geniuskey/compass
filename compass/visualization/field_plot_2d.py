"""2D electromagnetic field visualization for COMPASS simulation results.

Provides matplotlib-based plotting of field components (Ex, Ey, Ez, |E|^2, Sz)
on XZ, YZ, or XY cross-section planes, with optional structure overlay and
support for multi-wavelength comparison.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from compass.core.types import FieldData, SimulationResult

logger = logging.getLogger(__name__)

# Valid field component identifiers.
_VALID_COMPONENTS = ("Ex", "Ey", "Ez", "|E|2", "Sz")


def _extract_field_component(
    field_data: FieldData,
    component: str,
    poynting: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract a specific field component from FieldData.

    Args:
        field_data: FieldData instance containing Ex, Ey, Ez arrays.
        component: One of 'Ex', 'Ey', 'Ez', '|E|2', or 'Sz'.
        poynting: Optional Poynting vector Sz array (used when component='Sz').

    Returns:
        2D or 3D numpy array of the requested component, or None if data
        is not available.

    Raises:
        ValueError: If component is not recognized.
    """
    if component not in _VALID_COMPONENTS:
        raise ValueError(
            f"Unknown field component '{component}'. "
            f"Valid choices: {_VALID_COMPONENTS}"
        )

    if component == "Ex":
        if field_data.Ex is None:
            return None
        return np.asarray(np.abs(field_data.Ex))
    elif component == "Ey":
        if field_data.Ey is None:
            return None
        return np.asarray(np.abs(field_data.Ey))
    elif component == "Ez":
        if field_data.Ez is None:
            return None
        return np.asarray(np.abs(field_data.Ez))
    elif component == "|E|2":
        return field_data.E_intensity
    elif component == "Sz":
        if poynting is not None:
            return poynting
        # Approximate Sz from E fields if Poynting not available directly
        logger.warning(
            "Poynting vector Sz not directly available; "
            "falling back to |E|^2 as proxy."
        )
        return field_data.E_intensity

    return None


def _take_slice(
    data_3d: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    plane: str,
    position: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2D slice from a 3D field array.

    The 3D array is assumed to have shape (ny, nx, nz) with coordinate arrays
    x (nx,), y (ny,), z (nz,).

    Args:
        data_3d: 3D field data array of shape (ny, nx, nz).
        coords: Tuple of (x, y, z) coordinate arrays.
        plane: Slice orientation: 'xz', 'yz', or 'xy'.
        position: Position along the perpendicular axis in um.

    Returns:
        Tuple of (slice_2d, axis1_coords, axis2_coords) suitable for
        pcolormesh. For 'xz': (data[y_idx, :, :], x, z); for 'yz':
        (data[:, x_idx, :], y, z); for 'xy': (data[:, :, z_idx], x, y).
    """
    x, y, z = coords

    if plane == "xz":
        idx = int(np.argmin(np.abs(y - position)))
        return data_3d[idx, :, :].T, x, z
    elif plane == "yz":
        idx = int(np.argmin(np.abs(x - position)))
        return data_3d[:, idx, :].T, y, z
    elif plane == "xy":
        idx = int(np.argmin(np.abs(z - position)))
        return data_3d[:, :, idx], x, y
    else:
        raise ValueError(f"Unsupported plane '{plane}'. Must be 'xz', 'yz', or 'xy'.")


def plot_field_2d(
    result: SimulationResult,
    component: str = "|E|2",
    plane: str = "xz",
    position: float = 0.0,
    wavelength_key: str | None = None,
    overlay_structure: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
    log_scale: bool = False,
    cmap: str = "hot",
) -> plt.Axes:
    """Plot a 2D cross-section of a field component from simulation results.

    Extracts the specified field component from the SimulationResult, slices
    it along the requested plane, and renders it as a colormapped image.
    Optionally overlays a simplified structure outline derived from the
    permittivity data.

    Args:
        result: SimulationResult containing field data and optional
            Poynting vector.
        component: Field component to plot. One of 'Ex', 'Ey', 'Ez',
            '|E|2', or 'Sz'.
        plane: Cross-section plane: 'xz', 'yz', or 'xy'.
        position: Slice position in um along the axis perpendicular
            to the plane.
        wavelength_key: Key into result.fields dict. If None, uses the
            first available wavelength.
        overlay_structure: If True, overlays contour lines showing material
            boundaries from the permittivity distribution.
        ax: Optional matplotlib Axes. If None, a new figure is created.
        figsize: Figure size in inches, used only when ax is None.
        log_scale: If True, uses logarithmic color normalization.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib Axes containing the plot.
    """
    plane = plane.lower()
    if plane not in ("xz", "yz", "xy"):
        raise ValueError(f"Unsupported plane '{plane}'. Must be 'xz', 'yz', or 'xy'.")

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Resolve field data
    if result.fields is None or len(result.fields) == 0:
        logger.warning("No field data available in SimulationResult.")
        ax.set_title("No field data available")
        return ax

    if wavelength_key is None:
        wavelength_key = next(iter(result.fields))
        logger.info("Using wavelength key '%s' (first available).", wavelength_key)

    if wavelength_key not in result.fields:
        available = list(result.fields.keys())
        logger.warning(
            "Wavelength key '%s' not found. Available: %s", wavelength_key, available
        )
        ax.set_title(f"Wavelength key '{wavelength_key}' not found")
        return ax

    field_data = result.fields[wavelength_key]

    # Get coordinate arrays
    if field_data.x is None or field_data.y is None or field_data.z is None:
        logger.warning("Field coordinate arrays (x, y, z) are not set.")
        ax.set_title("Field coordinates not available")
        return ax

    # Resolve Poynting data
    poynting = None
    if component == "Sz" and result.poynting is not None:
        poynting = result.poynting.get(wavelength_key)

    # Extract field component
    field_3d = _extract_field_component(field_data, component, poynting)
    if field_3d is None:
        logger.warning("Field component '%s' is not available.", component)
        ax.set_title(f"Component '{component}' not available")
        return ax

    coords = (field_data.x, field_data.y, field_data.z)

    try:
        slice_2d, ax1, ax2 = _take_slice(field_3d, coords, plane, position)
    except Exception as exc:
        logger.error("Failed to extract %s slice: %s", plane, exc)
        ax.set_title(f"Slice error: {exc}")
        return ax

    # Determine normalization
    vmin = float(np.nanmin(slice_2d)) if not log_scale else max(float(np.nanmin(slice_2d)), 1e-10)
    vmax = float(np.nanmax(slice_2d))

    norm: LogNorm | Normalize
    if log_scale:
        # Clamp small values for log scale
        slice_2d_plot = np.where(slice_2d > 0, slice_2d, vmin)
        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=max(vmax, 1e-9))
    else:
        slice_2d_plot = slice_2d
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(
        ax1, ax2, slice_2d_plot,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    # Colorbar
    cbar_label = component
    if component == "|E|2":
        cbar_label = "|E|$^2$ (V$^2$/m$^2$)"
    elif component == "Sz":
        cbar_label = "S$_z$ (W/m$^2$)"
    plt.colorbar(im, ax=ax, label=cbar_label)

    # Structure overlay
    if overlay_structure:
        _overlay_structure_outline(
            result, wavelength_key, plane, position, ax, coords
        )

    # Axis labels
    if plane == "xz":
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        perp_label = f"y = {position:.2f} um"
    elif plane == "yz":
        ax.set_xlabel("y (um)")
        ax.set_ylabel("z (um)")
        perp_label = f"x = {position:.2f} um"
    else:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        perp_label = f"z = {position:.2f} um"

    scale_label = " [log]" if log_scale else ""
    ax.set_title(f"{component}{scale_label} -- {plane.upper()} plane, {perp_label}")

    return ax


def _overlay_structure_outline(
    result: SimulationResult,
    wavelength_key: str,
    plane: str,
    position: float,
    ax: plt.Axes,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Overlay material boundary contours on the field plot.

    Uses the electric field intensity as a proxy to identify the permittivity
    structure boundaries. If direct permittivity data is not embedded in the
    result, this function attempts to derive boundaries from the field
    discontinuities.

    Args:
        result: SimulationResult (used for metadata inspection).
        wavelength_key: Active wavelength key.
        plane: Slice plane.
        position: Slice position.
        ax: Axes to draw on.
        coords: (x, y, z) coordinate arrays.
    """
    if result.fields is None:
        return
    field_data = result.fields.get(wavelength_key)
    if field_data is None:
        return

    # Build a rough permittivity-like map from |E|^2 for boundary detection.
    # If the simulation stores permittivity in metadata, prefer that.
    eps_3d = result.metadata.get("eps_3d")
    if eps_3d is not None:
        try:
            eps_slice, ax1, ax2 = _take_slice(
                np.real(eps_3d), coords, plane, position
            )
            ax.contour(
                ax1, ax2, eps_slice,
                levels=6,
                colors="white",
                linewidths=0.6,
                alpha=0.7,
            )
        except Exception as exc:
            logger.debug("Structure overlay failed: %s", exc)
    else:
        logger.debug(
            "No permittivity data in result.metadata['eps_3d']; "
            "skipping structure overlay."
        )


def plot_field_multi_wavelength(
    result: SimulationResult,
    wavelengths: Sequence[str],
    component: str = "|E|2",
    plane: str = "xz",
    position: float = 0.0,
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (5, 4),
    log_scale: bool = False,
    cmap: str = "hot",
) -> plt.Figure:
    """Plot field cross-sections for multiple wavelengths side by side.

    Creates a grid of subplots, one per wavelength key, showing the same
    field component and slice plane for easy comparison.

    Args:
        result: SimulationResult with field data at multiple wavelengths.
        wavelengths: Sequence of wavelength keys to plot (must be present
            in result.fields).
        component: Field component to plot.
        plane: Cross-section plane.
        position: Slice position in um.
        ncols: Number of columns in the subplot grid.
        figsize_per_plot: (width, height) per individual subplot in inches.
        log_scale: If True, uses logarithmic color scale.
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib Figure containing all subplots.
    """
    if result.fields is None or len(result.fields) == 0:
        logger.warning("No field data available for multi-wavelength plot.")
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("No field data available")
        return fig

    # Filter to available wavelengths
    available_keys = set(result.fields.keys())
    valid_wls = [wl for wl in wavelengths if wl in available_keys]
    if not valid_wls:
        logger.warning(
            "None of the requested wavelength keys %s found in result. "
            "Available: %s",
            wavelengths, list(available_keys),
        )
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("No matching wavelength keys")
        return fig

    n_plots = len(valid_wls)
    nrows = max(1, int(np.ceil(n_plots / ncols)))
    actual_ncols = min(ncols, n_plots)

    fig_w = figsize_per_plot[0] * actual_ncols
    fig_h = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, actual_ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, wl_key in enumerate(valid_wls):
        row = idx // actual_ncols
        col = idx % actual_ncols
        plot_field_2d(
            result,
            component=component,
            plane=plane,
            position=position,
            wavelength_key=wl_key,
            overlay_structure=False,
            ax=axes[row][col],
            log_scale=log_scale,
            cmap=cmap,
        )
        axes[row][col].set_title(f"{component} -- wl={wl_key}")

    # Hide unused subplots
    for idx in range(n_plots, nrows * actual_ncols):
        row = idx // actual_ncols
        col = idx % actual_ncols
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig
