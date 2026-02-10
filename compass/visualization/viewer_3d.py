"""3D visualization of CMOS pixel stack structures.

Provides interactive 3D rendering of pixel stacks using plotly for layer-by-layer
visualization with transparency, material color coding, microlens surfaces,
DTI trenches, and photodiode regions.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from compass.core.types import Layer
from compass.geometry.pixel_stack import PixelStack

logger = logging.getLogger(__name__)

# Material-to-RGBA color mapping for 3D rendering.
# Values are (R, G, B) in 0-255 range; opacity is set per layer type.
MATERIAL_COLORS_3D: dict[str, tuple[int, int, int]] = {
    "silicon": (160, 160, 160),
    "si": (160, 160, 160),
    "sio2": (173, 216, 230),
    "cf_red": (220, 50, 50),
    "cf_r": (220, 50, 50),
    "cf_green": (50, 180, 50),
    "cf_g": (50, 180, 50),
    "cf_blue": (50, 80, 220),
    "cf_b": (50, 80, 220),
    "tungsten": (220, 200, 50),
    "w": (220, 200, 50),
    "polymer": (221, 160, 221),
    "air": (240, 248, 255),
    "hfo2": (255, 255, 224),
    "si3n4": (250, 180, 150),
    "tio2": (245, 222, 179),
}


def _resolve_color_3d(
    material_name: str,
    layer_name: str = "",
) -> tuple[int, int, int]:
    """Resolve RGB color tuple for a material in 3D rendering.

    Args:
        material_name: Material identifier.
        layer_name: Layer identifier (fallback for matching).

    Returns:
        (R, G, B) color tuple with values in 0-255.
    """
    name_lower = material_name.lower()
    for key, rgb in MATERIAL_COLORS_3D.items():
        if key in name_lower:
            return rgb

    layer_lower = layer_name.lower()
    for key, rgb in MATERIAL_COLORS_3D.items():
        if key in layer_lower:
            return rgb

    return (200, 200, 200)


def _make_box_mesh(
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate vertex arrays for a rectangular box as a triangulated mesh.

    Returns the (x, y, z, i, j, k) arrays suitable for plotly Mesh3d.
    The box has 8 vertices and 12 triangular faces.

    Args:
        x0, x1: X range.
        y0, y1: Y range.
        z0, z1: Z range.

    Returns:
        Tuple of (x_verts, y_verts, z_verts, tri_i, tri_j, tri_k) arrays.
    """
    # 8 vertices of the box
    vx = np.array([x0, x1, x1, x0, x0, x1, x1, x0])
    vy = np.array([y0, y0, y1, y1, y0, y0, y1, y1])
    vz = np.array([z0, z0, z0, z0, z1, z1, z1, z1])

    # 12 triangles (2 per face, 6 faces)
    ti = np.array([0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 2, 2])
    tj = np.array([1, 2, 5, 6, 1, 4, 2, 6, 3, 4, 3, 3])
    tk = np.array([2, 3, 6, 7, 5, 5, 6, 5, 7, 7, 7, 6])

    return vx, vy, vz, ti, tj, tk


def _make_lens_surface(
    center_x: float,
    center_y: float,
    rx: float,
    ry: float,
    height: float,
    z_base: float,
    n_param: float = 2.5,
    alpha_param: float = 1.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    resolution: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth microlens surface mesh.

    Uses the superellipse profile to compute the lens height map on a polar
    grid, yielding a smooth 3D surface suitable for plotly Surface traces.

    Args:
        center_x, center_y: Lens center position in um.
        rx, ry: Semi-axes in um.
        height: Maximum lens height in um.
        z_base: Z-coordinate of the lens base.
        n_param: Superellipse squareness parameter.
        alpha_param: Curvature parameter.
        shift_x, shift_y: CRA offsets in um.
        resolution: Number of grid points per axis for the surface mesh.

    Returns:
        Tuple of (X, Y, Z) 2D arrays for the lens surface.
    """
    x = np.linspace(center_x - rx * 1.05, center_x + rx * 1.05, resolution)
    y = np.linspace(center_y - ry * 1.05, center_y + ry * 1.05, resolution)
    xx, yy = np.meshgrid(x, y)

    dx = xx - (center_x + shift_x)
    dy = yy - (center_y + shift_y)

    rx_safe = max(rx, 1e-10)
    ry_safe = max(ry, 1e-10)

    r = (np.abs(dx / rx_safe) ** n_param + np.abs(dy / ry_safe) ** n_param) ** (
        1.0 / n_param
    )
    r = np.clip(r, 0, 1.0 + 1e-10)

    zz = np.full_like(r, z_base)
    mask = r <= 1.0
    zz[mask] = z_base + height * (1.0 - r[mask] ** 2) ** (1.0 / (2.0 * alpha_param))

    return xx, yy, zz


def view_pixel_3d(
    pixel_stack: PixelStack,
    backend: str = "plotly",
    wavelength: float = 0.55,
) -> Any:
    """Create an interactive 3D visualization of the pixel stack.

    Renders each layer as a semi-transparent box with material-appropriate
    colors. Microlenses are shown as smooth curved surfaces. DTI trenches
    and photodiode regions are highlighted with distinct colors and opacity.

    Currently supports the plotly backend, returning a plotly.graph_objects.Figure
    that can be displayed in Jupyter notebooks or exported to HTML.

    Args:
        pixel_stack: A fully constructed PixelStack instance.
        backend: Rendering backend. Currently only 'plotly' is supported.
        wavelength: Wavelength in um for any permittivity-dependent
            visualization (e.g., color filter pattern).

    Returns:
        A plotly Figure object (when backend='plotly').

    Raises:
        ImportError: If the plotly package is not installed.
        ValueError: If an unsupported backend is specified.
    """
    backend = backend.lower()
    if backend != "plotly":
        raise ValueError(
            f"Unsupported 3D backend '{backend}'. Currently only 'plotly' "
            f"is supported."
        )

    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "The 'plotly' package is required for 3D visualization. "
            "Install it with: pip install plotly"
        ) from err

    lx, ly = pixel_stack.domain_size
    layers = pixel_stack.layers

    if not layers:
        logger.warning("PixelStack has no layers; returning empty figure.")
        fig = go.Figure()
        fig.update_layout(title="Empty Pixel Stack")
        return fig

    traces: list[Any] = []

    # Opacity mapping per layer type
    opacity_map = {
        "silicon": 0.6,
        "air": 0.05,
        "microlens": 0.4,
        "color_filter": 0.7,
        "planarization": 0.3,
    }

    for layer in layers:
        if layer.name == "microlens":
            # Render microlenses as smooth surfaces instead of boxes
            _add_microlens_traces(pixel_stack, layer, traces, go)
            continue

        if layer.name == "color_filter" and pixel_stack.bayer_map:
            # Render per-pixel color filters
            _add_color_filter_traces(pixel_stack, layer, traces, go)
            continue

        # Default: render as semi-transparent box
        rgb = _resolve_color_3d(layer.base_material, layer.name)
        opacity = opacity_map.get(layer.name, 0.4)

        # Skip air layer if very transparent
        if layer.name == "air":
            opacity = 0.05

        vx, vy, vz, ti, tj, tk = _make_box_mesh(
            0, lx, 0, ly, layer.z_start, layer.z_end
        )

        color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        traces.append(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                color=color_str,
                opacity=opacity,
                name=layer.name,
                showlegend=True,
                hoverinfo="name",
            )
        )

    # Add photodiode regions
    _add_photodiode_traces(pixel_stack, traces, go)

    # Add DTI trenches if silicon is patterned
    _add_dti_traces(pixel_stack, traces, go)

    # Assemble figure
    fig = go.Figure(data=traces)

    fig.update_layout(
        title="COMPASS Pixel Stack 3D View",
        scene=dict(
            xaxis_title="x (um)",
            yaxis_title="y (um)",
            zaxis_title="z (um)",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def _add_microlens_traces(
    pixel_stack: PixelStack,
    layer: Layer,
    traces: list,
    go: Any,
) -> None:
    """Add microlens surface traces to the plotly figure.

    Each microlens in the unit cell is rendered as a smooth curved surface
    using the superellipse profile with configurable resolution.

    Args:
        pixel_stack: The pixel stack instance.
        layer: The microlens Layer object.
        traces: List to append plotly traces to.
        go: plotly.graph_objects module reference.
    """
    rgb = _resolve_color_3d(layer.base_material, "microlens")
    color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    for idx, ml in enumerate(pixel_stack.microlenses):
        r = idx // pixel_stack.unit_cell[1]
        c = idx % pixel_stack.unit_cell[1]
        cx = (c + 0.5) * pixel_stack.pitch
        cy = (r + 0.5) * pixel_stack.pitch

        xx, yy, zz = _make_lens_surface(
            center_x=cx,
            center_y=cy,
            rx=ml.radius_x,
            ry=ml.radius_y,
            height=ml.height,
            z_base=layer.z_start,
            n_param=ml.n_param,
            alpha_param=ml.alpha_param,
            shift_x=ml.shift_x,
            shift_y=ml.shift_y,
            resolution=40,
        )

        # Clip to pixel region
        x_lo = c * pixel_stack.pitch
        x_hi = (c + 1) * pixel_stack.pitch
        y_lo = r * pixel_stack.pitch
        y_hi = (r + 1) * pixel_stack.pitch

        mask = (
            (xx >= x_lo) & (xx <= x_hi) &
            (yy >= y_lo) & (yy <= y_hi)
        )
        zz_clipped = np.where(mask, zz, layer.z_start)

        colorscale = [[0, color_str], [1, color_str]]

        traces.append(
            go.Surface(
                x=xx, y=yy, z=zz_clipped,
                colorscale=colorscale,
                showscale=False,
                opacity=0.5,
                name=f"microlens_{r}_{c}",
                showlegend=(idx == 0),
                hoverinfo="name",
            )
        )

    # Add a flat base for the microlens layer
    vx, vy, vz, ti, tj, tk = _make_box_mesh(
        0, pixel_stack.domain_size[0],
        0, pixel_stack.domain_size[1],
        layer.z_start, layer.z_start + 0.01,
    )
    traces.append(
        go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=ti, j=tj, k=tk,
            color=color_str,
            opacity=0.2,
            name="microlens_base",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _add_color_filter_traces(
    pixel_stack: PixelStack,
    layer: Layer,
    traces: list,
    go: Any,
) -> None:
    """Add per-pixel color filter boxes to the plotly figure.

    Each pixel in the Bayer pattern is rendered as a separate colored box
    for the color filter layer.

    Args:
        pixel_stack: The pixel stack instance.
        layer: The color filter Layer object.
        traces: List to append plotly traces to.
        go: plotly.graph_objects module reference.
    """
    pitch = pixel_stack.pitch
    legend_shown = set()

    for r in range(pixel_stack.unit_cell[0]):
        for c in range(pixel_stack.unit_cell[1]):
            color_char = pixel_stack.bayer_map[r][c]
            cf_material = f"cf_{color_char.lower()}"
            rgb = _resolve_color_3d(cf_material)

            x0 = c * pitch
            x1 = (c + 1) * pitch
            y0 = r * pitch
            y1 = (r + 1) * pitch

            vx, vy, vz, ti, tj, tk = _make_box_mesh(
                x0, x1, y0, y1, layer.z_start, layer.z_end
            )

            color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            show_legend = color_char not in legend_shown
            legend_shown.add(color_char)

            traces.append(
                go.Mesh3d(
                    x=vx, y=vy, z=vz,
                    i=ti, j=tj, k=tk,
                    color=color_str,
                    opacity=0.7,
                    name=f"CF_{color_char}",
                    showlegend=show_legend,
                    hoverinfo="name",
                )
            )


def _add_photodiode_traces(
    pixel_stack: PixelStack,
    traces: list,
    go: Any,
) -> None:
    """Add photodiode region boxes to the plotly figure.

    Photodiodes are rendered as highlighted semi-transparent boxes within the
    silicon layer, colored according to their Bayer channel assignment.

    Args:
        pixel_stack: The pixel stack instance.
        traces: List to append plotly traces to.
        go: plotly.graph_objects module reference.
    """
    if not pixel_stack.photodiodes:
        return

    # Find silicon layer for z reference
    si_layer = None
    for layer in pixel_stack.layers:
        if layer.name == "silicon":
            si_layer = layer
            break

    if si_layer is None:
        logger.warning("No silicon layer found; cannot render photodiodes.")
        return

    pitch = pixel_stack.pitch
    pd_colors = {
        "R": (255, 100, 100),
        "G": (100, 220, 100),
        "B": (100, 100, 255),
    }
    legend_shown = set()

    for pd in pixel_stack.photodiodes:
        r, c = pd.pixel_index
        px, py, pz = pd.position
        dx, dy, dz = pd.size

        cx = (c + 0.5) * pitch + px
        cy = (r + 0.5) * pitch + py
        cz = si_layer.z_end - pz

        x0 = cx - dx / 2.0
        x1 = cx + dx / 2.0
        y0 = cy - dy / 2.0
        y1 = cy + dy / 2.0
        z0 = max(cz - dz / 2.0, si_layer.z_start)
        z1 = min(cz + dz / 2.0, si_layer.z_end)

        vx, vy, vz, ti, tj, tk = _make_box_mesh(x0, x1, y0, y1, z0, z1)

        rgb = pd_colors.get(pd.color, (200, 200, 200))
        color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        label = f"PD_{pd.color}"
        show_legend = label not in legend_shown
        legend_shown.add(label)

        traces.append(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                color=color_str,
                opacity=0.4,
                name=label,
                showlegend=show_legend,
                hoverinfo="name",
            )
        )


def _add_dti_traces(
    pixel_stack: PixelStack,
    traces: list,
    go: Any,
) -> None:
    """Add DTI (Deep Trench Isolation) traces to the plotly figure.

    DTI trenches at pixel boundaries in the silicon layer are rendered as
    thin semi-transparent boxes.

    Args:
        pixel_stack: The pixel stack instance.
        traces: List to append plotly traces to.
        go: plotly.graph_objects module reference.
    """
    # Check if silicon layer is patterned (DTI enabled)
    si_layer = None
    for layer in pixel_stack.layers:
        if layer.name == "silicon" and layer.is_patterned:
            si_layer = layer
            break

    if si_layer is None:
        return

    # Retrieve DTI width from config
    si_cfg = pixel_stack._layer_configs.get("silicon", {})
    dti_cfg = si_cfg.get("dti", {})
    if not dti_cfg.get("enabled", False):
        return

    dti_width = dti_cfg.get("width", 0.1)
    half_w = dti_width / 2.0
    pitch = pixel_stack.pitch
    lx, ly = pixel_stack.domain_size
    rows, cols = pixel_stack.unit_cell

    rgb = (220, 220, 240)
    color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    first_trace = True

    # Vertical DTI lines (along y direction)
    for c in range(cols + 1):
        x_pos = c * pitch
        x0 = max(x_pos - half_w, 0)
        x1 = min(x_pos + half_w, lx)
        if x1 <= x0:
            continue

        vx, vy, vz, ti, tj, tk = _make_box_mesh(
            x0, x1, 0, ly, si_layer.z_start, si_layer.z_end
        )
        traces.append(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                color=color_str,
                opacity=0.5,
                name="DTI",
                showlegend=first_trace,
                hoverinfo="name",
            )
        )
        first_trace = False

    # Horizontal DTI lines (along x direction)
    for r in range(rows + 1):
        y_pos = r * pitch
        y0 = max(y_pos - half_w, 0)
        y1 = min(y_pos + half_w, ly)
        if y1 <= y0:
            continue

        vx, vy, vz, ti, tj, tk = _make_box_mesh(
            0, lx, y0, y1, si_layer.z_start, si_layer.z_end
        )
        traces.append(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=ti, j=tj, k=tk,
                color=color_str,
                opacity=0.5,
                name="DTI",
                showlegend=False,
                hoverinfo="name",
            )
        )
