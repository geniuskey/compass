"""2D cross-section visualization of CMOS pixel stack structures.

Provides matplotlib-based plotting of XZ, YZ, and XY cross-sections
of the pixel stack, with material color-coding and layer annotations.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches

from compass.geometry.pixel_stack import PixelStack

logger = logging.getLogger(__name__)

# Material-to-color mapping for structure visualization.
# Keys are matched as substrings against material/layer names.
MATERIAL_COLORS: dict[str, str] = {
    "silicon": "gray",
    "si": "gray",
    "sio2": "lightblue",
    "cf_red": "red",
    "cf_r": "red",
    "cf_green": "green",
    "cf_g": "green",
    "cf_blue": "blue",
    "cf_b": "blue",
    "tungsten": "yellow",
    "w": "yellow",
    "polymer": "plum",
    "air": "white",
    "hfo2": "lightyellow",
    "si3n4": "lightsalmon",
    "tio2": "wheat",
}


def _resolve_color(material_name: str, layer_name: str = "") -> str:
    """Resolve a display color from a material or layer name.

    Checks the material name first, then the layer name, using substring
    matching against the MATERIAL_COLORS mapping.

    Args:
        material_name: Material identifier (e.g. "silicon", "cf_red").
        layer_name: Layer identifier (e.g. "color_filter", "air").

    Returns:
        Matplotlib color string.
    """
    name_lower = material_name.lower()
    for key, color in MATERIAL_COLORS.items():
        if key in name_lower:
            return color

    layer_lower = layer_name.lower()
    for key, color in MATERIAL_COLORS.items():
        if key in layer_lower:
            return color

    return "lightgray"


def plot_pixel_cross_section(
    pixel_stack: PixelStack,
    plane: str = "xz",
    position: float = 0.0,
    wavelength: float = 0.55,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Axes:
    """Plot a 2D cross-section of the pixel stack.

    For XZ and YZ planes, renders a vertical cross-section showing all layers
    from bottom (silicon) to top (air) with material color-coding, layer
    boundary lines, and name annotations.

    For XY planes, renders the real part of the permittivity distribution at a
    specified z-position as a colormapped image.

    Args:
        pixel_stack: A fully constructed PixelStack instance.
        plane: Cross-section orientation, one of 'xz', 'yz', or 'xy'.
        position: Slice position in um along the axis perpendicular to the
            cross-section plane. For 'xz': y-position; for 'yz': x-position;
            for 'xy': z-position.
        wavelength: Wavelength in um for computing permittivity (used in XY
            mode and for layer slice generation).
        ax: Optional matplotlib Axes to draw on. If None, a new figure is
            created with the given figsize.
        figsize: Figure size as (width, height) in inches, used only when
            ax is None.

    Returns:
        The matplotlib Axes containing the plot.

    Raises:
        ValueError: If plane is not one of 'xz', 'yz', or 'xy'.
    """
    plane = plane.lower()
    if plane not in ("xz", "yz", "xy"):
        raise ValueError(f"Unsupported plane '{plane}'. Must be 'xz', 'yz', or 'xy'.")

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=figsize)

    if plane in ("xz", "yz"):
        _plot_vertical_cross_section(pixel_stack, plane, position, wavelength, ax)
    else:
        _plot_xy_cross_section(pixel_stack, position, wavelength, ax)

    return ax


def _plot_vertical_cross_section(
    pixel_stack: PixelStack,
    plane: str,
    position: float,
    wavelength: float,
    ax: plt.Axes,
) -> None:
    """Render a vertical (XZ or YZ) cross-section of the pixel stack.

    Each layer is drawn as a colored rectangle spanning the full lateral domain.
    Color filter layers are subdivided per pixel to show the Bayer pattern.
    Layer boundaries are drawn as solid black lines and layer names are annotated
    on the right margin.

    Args:
        pixel_stack: The pixel stack to visualize.
        plane: 'xz' or 'yz'.
        position: Slice position along the perpendicular axis (um).
        wavelength: Wavelength in um (for future permittivity overlay).
        ax: Matplotlib Axes to draw on.
    """
    lx, ly = pixel_stack.domain_size
    layers = pixel_stack.layers
    lateral_extent = lx if plane == "xz" else ly

    if not layers:
        logger.warning("PixelStack has no layers; nothing to plot.")
        ax.set_title("Empty pixel stack")
        return

    rect_patches = []
    rect_colors = []

    for layer in layers:
        z_start = layer.z_start
        thickness = layer.thickness

        if layer.name == "color_filter" and pixel_stack.bayer_map:
            # Draw per-pixel color filter regions
            n_pixels = (pixel_stack.unit_cell[1] if plane == "xz"
                        else pixel_stack.unit_cell[0])
            pitch = pixel_stack.pitch

            for idx in range(n_pixels):
                if plane == "xz":
                    # Determine which row the slice position falls in
                    row_idx = min(
                        int(position / pitch),
                        pixel_stack.unit_cell[0] - 1,
                    )
                    row_idx = max(row_idx, 0)
                    col_idx = idx
                else:
                    col_idx = min(
                        int(position / pitch),
                        pixel_stack.unit_cell[1] - 1,
                    )
                    col_idx = max(col_idx, 0)
                    row_idx = idx

                color_char = pixel_stack.bayer_map[
                    row_idx % len(pixel_stack.bayer_map)
                ][
                    col_idx % len(pixel_stack.bayer_map[0])
                ]
                cf_material = f"cf_{color_char.lower()}"
                face_color = _resolve_color(cf_material)

                x0 = idx * pitch
                rect = mpatches.Rectangle(
                    (x0, z_start), pitch, thickness,
                )
                rect_patches.append(rect)
                rect_colors.append(face_color)
        else:
            face_color = _resolve_color(layer.base_material, layer.name)
            rect = mpatches.Rectangle(
                (0.0, z_start), lateral_extent, thickness,
            )
            rect_patches.append(rect)
            rect_colors.append(face_color)

    # Draw patches
    for rect, color in zip(rect_patches, rect_colors):
        rect.set_facecolor(color)
        rect.set_edgecolor("black")
        rect.set_linewidth(0.5)
        ax.add_patch(rect)

    # Draw layer boundary lines and annotations
    z_boundaries = set()
    for layer in layers:
        z_boundaries.add(layer.z_start)
        z_boundaries.add(layer.z_end)

    for z_val in sorted(z_boundaries):
        ax.axhline(y=z_val, color="black", linewidth=0.8, linestyle="-")

    # Annotate layer names on the right side
    annotation_x = lateral_extent * 1.02
    for layer in layers:
        z_mid = (layer.z_start + layer.z_end) / 2.0
        ax.annotate(
            layer.name,
            xy=(lateral_extent, z_mid),
            xytext=(annotation_x, z_mid),
            fontsize=8,
            va="center",
            ha="left",
            color="dimgray",
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.5),
        )

    # Display grid spacing info
    z_min = layers[0].z_start
    z_max = layers[-1].z_end
    pitch_text = f"pitch = {pixel_stack.pitch:.3f} um"
    ax.text(
        0.02, 0.98, pitch_text,
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Set axis limits and labels
    lateral_label = "x" if plane == "xz" else "y"
    ax.set_xlim(-0.05 * lateral_extent, lateral_extent * 1.25)
    ax.set_ylim(z_min - 0.05 * (z_max - z_min), z_max + 0.05 * (z_max - z_min))
    ax.set_xlabel(f"{lateral_label} (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(
        f"Pixel Stack Cross-Section ({plane.upper()} plane, "
        f"{'y' if plane == 'xz' else 'x'} = {position:.2f} um)"
    )
    ax.set_aspect("auto")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")


def _plot_xy_cross_section(
    pixel_stack: PixelStack,
    z_position: float,
    wavelength: float,
    ax: plt.Axes,
) -> None:
    """Render an XY cross-section showing permittivity at a given z-height.

    Generates layer slices at the specified wavelength, finds the slice
    containing the requested z-position, and plots the real part of the
    complex permittivity as a colormapped image.

    Args:
        pixel_stack: The pixel stack to visualize.
        z_position: z-coordinate (um) for the cross-section.
        wavelength: Wavelength in um for permittivity computation.
        ax: Matplotlib Axes to draw on.
    """
    nx, ny = 256, 256
    lx, ly = pixel_stack.domain_size

    try:
        slices = pixel_stack.get_layer_slices(wavelength, nx=nx, ny=ny)
    except Exception as exc:
        logger.error("Failed to generate layer slices: %s", exc)
        ax.set_title(f"Error generating slices: {exc}")
        return

    if not slices:
        logger.warning("No layer slices available.")
        ax.set_title("No data available for XY cross-section")
        return

    # Find the slice containing z_position
    target_slice = None
    for s in slices:
        if s.z_start <= z_position < s.z_end:
            target_slice = s
            break

    if target_slice is None:
        # Fall back to nearest slice
        distances = [abs((s.z_start + s.z_end) / 2.0 - z_position) for s in slices]
        target_slice = slices[int(np.argmin(distances))]
        logger.warning(
            "z = %.3f um is outside stack; using nearest slice '%s' "
            "(z = %.3f - %.3f um).",
            z_position, target_slice.name, target_slice.z_start, target_slice.z_end,
        )

    eps_real = np.real(target_slice.eps_grid)

    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)

    im = ax.pcolormesh(
        x, y, eps_real,
        shading="auto",
        cmap="viridis",
    )
    _cbar = plt.colorbar(im, ax=ax, label="Re(epsilon)")

    # Add pixel boundary lines
    pitch = pixel_stack.pitch
    for c in range(pixel_stack.unit_cell[1] + 1):
        ax.axvline(x=c * pitch, color="white", linewidth=0.5, linestyle="--")
    for r in range(pixel_stack.unit_cell[0] + 1):
        ax.axhline(y=r * pitch, color="white", linewidth=0.5, linestyle="--")

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        f"XY Cross-Section at z = {z_position:.3f} um "
        f"(layer: {target_slice.name}, wl = {wavelength:.3f} um)"
    )
    ax.set_aspect("equal")
