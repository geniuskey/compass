"""Recent-generation vendor pixel structures and structure-parameter heuristics.

This module captures publicly-known structural features of recent (2022-2024)
flagship CMOS image sensors from Samsung, Sony and OmniVision, and exposes a
parameter-derivation utility that turns a small number of vendor-disclosed
inputs (pitch, color-filter pattern, OCL sharing, etc.) into a complete pixel
config dict that COMPASS can simulate.

# Why a heuristic?

Vendors publish *headline* parameters in datasheets / press releases:

* Pixel pitch (e.g. Samsung HP9 = 0.56 µm, Sony LYT-900 = 1.6 µm).
* Color-filter binning pattern (Bayer / Quad Bayer / Nona / Tetra²).
* On-chip-lens (OCL) sharing (per-pixel / 2x2 OCL / 4x4 OCL).
* Major architectural innovations (2-Layer Transistor, LOFIC, F-DTI fill, HRI
  microlens material).

But the *internal* structural numbers needed for an EM simulation -- microlens
sag, color-filter thickness, BARL stack, DTI width, photodiode depth, etc. --
are almost never disclosed.  We close that gap with the empirical scaling rules
embedded below, derived from public ISSCC/IEDM/SPIE papers and TechInsights
reverse-engineering reports for sub-µm to 2-µm BSI pixels.

# Three ways to determine missing parameters

1. **Scaling rules (this module)**: Default starting point. `derive_parameters`
   returns physically reasonable values from pitch alone.
2. **Calibration to measured QE**: If you have measured QE(λ) per color, fit
   thickness/microlens parameters with `compass.optimization` (e.g. the
   ``MaximizeQE`` / custom L2-distance-to-target objectives). See the
   ``Optimization`` section in CLAUDE.md.
3. **TechInsights-style cross-section**: When a die-shot SEM is available,
   measure layer thicknesses directly and override the heuristic defaults.

The functions below cover (1) and produce a config dict that PixelStack
accepts directly; (2) and (3) are workflow recommendations rather than code.
"""

from __future__ import annotations

from typing import Any, Literal

VendorKey = Literal[
    "samsung_hp9",
    "samsung_gnj",
    "sony_lyt900",
    "sony_2x2ocl_quad",
    "omnivision_ov50k40",
    "generic_bsi",
]


# Public, vendor-disclosed headline parameters for recent (2022-2024) sensors.
# Sources: vendor press releases, product pages, TechInsights summaries.
VENDOR_HEADLINES: dict[str, dict[str, Any]] = {
    "samsung_hp9": {
        "pitch": 0.56,
        "format": "1/1.4\"",
        "megapixels": 200,
        "cf_pattern": "tetra2cell",     # 4x4 same-color (Hexadeca / Tetra^2)
        "ocl_sharing": 1,
        "microlens_material": "polymer_hri_n1p70",  # HRI "new material"
        "dti_fill": "sio2",                          # F-DTI oxide fill
        "year": 2024,
    },
    "samsung_gnj": {
        "pitch": 1.0,
        "format": "1/1.57\"",
        "megapixels": 50,
        "cf_pattern": "tetracell",
        "ocl_sharing": 1,
        "microlens_material": "polymer_n1p56",
        "dti_fill": "sio2",
        "year": 2024,
    },
    "sony_lyt900": {
        "pitch": 1.6,
        "format": "1\"",
        "megapixels": 50,
        "cf_pattern": "bayer_rggb",
        "ocl_sharing": 1,
        "microlens_material": "polymer_n1p56",
        "dti_fill": "sio2",
        "two_layer_transistor": True,    # PD volume enlarged
        "year": 2024,
    },
    "sony_2x2ocl_quad": {
        "pitch": 1.22,
        "format": "1/1.3\"",
        "megapixels": 48,
        "cf_pattern": "tetracell",
        "ocl_sharing": 2,                # 2x2 OCL
        "microlens_material": "polymer_n1p56",
        "dti_fill": "sio2",
        "year": 2023,
    },
    "omnivision_ov50k40": {
        "pitch": 1.2,
        "format": "1/1.3\"",
        "megapixels": 50,
        "cf_pattern": "tetracell",
        "ocl_sharing": 2,                # Quad PD = 2x2 OCL
        "microlens_material": "polymer_n1p56",
        "dti_fill": "sio2",
        "lofic": True,                   # TheiaCel: PD lateral footprint shrinks
        "year": 2024,
    },
    "generic_bsi": {
        "pitch": 1.0,
        "cf_pattern": "bayer_rggb",
        "ocl_sharing": 1,
        "microlens_material": "polymer_n1p56",
        "dti_fill": "sio2",
    },
}


# --- Scaling rules ----------------------------------------------------------
#
# All values below come from public ISSCC/IEDM/SPIE pixel-architecture papers
# (2018-2024) and TechInsights cross-section reports.  They are *typical*
# numbers; individual products will deviate by 10-30%.
#
# - Microlens sag-to-diameter ratio for sub-2 µm BSI pixels: 0.30-0.45.
# - Color-filter thickness shrinks roughly linearly with pitch for sub-µm
#   pixels (process limits cap CF aspect ratio).
# - DTI trench width is process-limited: 60 nm at 0.5 µm pitch up to ~100 nm
#   at 2 µm pitch.
# - Silicon epi thickness for visible BSI: 2-4 µm; thicker for larger pitch
#   to capture longer wavelengths in the photodiode.
# - BARL (bottom anti-reflective layers) is largely pitch-independent — the
#   stack is tuned for visible λ, not for pixel geometry.

def _ml_height(pitch: float) -> float:
    """Empirical microlens sag for typical BSI pixels."""
    return round(min(0.95, 0.42 * pitch + 0.20), 2)


def _ml_radius(pitch: float, gap: float, sharing: int) -> float:
    """Microlens semi-axis. Lens diameter ~ sharing*pitch - 2*gap."""
    return round(max(0.05, (sharing * pitch - 2.0 * gap) / 2.0), 3)


def _cf_thickness(pitch: float) -> float:
    """Empirical color-filter thickness."""
    if pitch <= 0.7:
        return round(0.35 + 0.25 * (pitch / 0.7), 2)  # 0.35 - 0.60
    return round(min(0.90, 0.45 * pitch + 0.15), 2)


def _planarization(pitch: float) -> float:
    return round(0.20 + 0.10 * min(pitch, 2.0) / 2.0, 2)


def _dti_width(pitch: float) -> float:
    """Process-limited DTI width."""
    return round(0.05 + 0.025 * min(pitch, 2.0), 3)


def _si_thickness(pitch: float) -> float:
    """Si epi thickness; deeper Si captures longer wavelengths."""
    return round(min(4.5, 1.4 + 1.5 * pitch), 2)


def _ml_gap(pitch: float) -> float:
    return round(0.02 + 0.02 * min(pitch, 2.0), 3)


def derive_parameters(
    vendor: VendorKey | str = "generic_bsi",
    *,
    pitch: float | None = None,
    cf_pattern: str | None = None,
    ocl_sharing: int | None = None,
    microlens_material: str | None = None,
    dti_fill: str | None = None,
    cra_deg: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full pixel config dict for a given vendor / pitch.

    Disclosed (headline) parameters are taken from VENDOR_HEADLINES; missing
    structural numbers are filled in by the scaling rules above.  Any
    keyword argument explicitly passed in overrides the vendor headline, and
    the optional ``overrides`` dict (keyed by ``layers.<name>.<field>``) wins
    over everything else.

    Args:
        vendor: Vendor key. ``"generic_bsi"`` is the catch-all.
        pitch: Pixel pitch in µm. Required if ``vendor == "generic_bsi"``.
        cf_pattern: CF binning pattern (bayer_rggb / tetracell / nonacell /
            tetra2cell). Defaults to vendor headline.
        ocl_sharing: Microlens sharing (1, 2, 3, 4). Defaults to vendor.
        microlens_material: Material name registered in MaterialDB.
        dti_fill: Material name for DTI trench fill.
        cra_deg: Chief ray angle in degrees for auto microlens shift.
        overrides: Dotted-key overrides, e.g.
            ``{"layers.silicon.thickness": 2.5}``.

    Returns:
        Config dict ready to pass to PixelStack(config=...).
    """
    headline = dict(VENDOR_HEADLINES.get(vendor, {}))
    if pitch is not None:
        headline["pitch"] = pitch
    if cf_pattern is not None:
        headline["cf_pattern"] = cf_pattern
    if ocl_sharing is not None:
        headline["ocl_sharing"] = ocl_sharing
    if microlens_material is not None:
        headline["microlens_material"] = microlens_material
    if dti_fill is not None:
        headline["dti_fill"] = dti_fill

    p = headline.get("pitch")
    if p is None:
        raise ValueError(f"pitch is required for vendor={vendor!r}")
    pattern = headline.get("cf_pattern", "bayer_rggb")
    sharing = int(headline.get("ocl_sharing", 1))
    ml_mat = headline.get("microlens_material", "polymer_n1p56")
    dti_mat = headline.get("dti_fill", "sio2")

    # Unit-cell size derived from the binning pattern.
    group_size = {
        "bayer_rggb": 1, "bayer_grbg": 1, "bayer_gbrg": 1, "bayer_bggr": 1,
        "tetracell": 2, "quad_bayer": 2,
        "nonacell": 3,
        "tetra2cell": 4, "hexadeca": 4,
    }.get(pattern, 1)
    unit_dim = 2 * group_size
    unit_cell = [unit_dim, unit_dim]

    gap = _ml_gap(p)
    ml_h = _ml_height(p)
    ml_r = _ml_radius(p, gap, sharing)
    cf_t = _cf_thickness(p)
    plan_t = _planarization(p)
    dti_w = _dti_width(p)
    si_t = _si_thickness(p)

    # 2-Layer-Transistor models a larger PD footprint and depth (Sony LYT-900).
    if headline.get("two_layer_transistor"):
        pd_xy = round(p * 0.88, 3)
        pd_z = round(si_t * 0.85, 3)
        pd_z_pos = round(si_t * 0.075, 3)
    elif headline.get("lofic"):
        # LOFIC capacitor consumes part of the in-pixel Si: PD smaller (~ 65%
        # area).
        pd_xy = round(p * 0.65, 3)
        pd_z = round(si_t * 0.67, 3)
        pd_z_pos = round(si_t * 0.17, 3)
    else:
        pd_xy = round(p * 0.70, 3)
        pd_z = round(si_t * 0.67, 3)
        pd_z_pos = round(si_t * 0.17, 3)

    # Bayer map: build group-binned RGGB super-tile.
    tile = [["R", "G"], ["G", "B"]]
    bayer_map = []
    for r in range(unit_dim):
        row = []
        for c in range(unit_dim):
            br, bc = r // group_size, c // group_size
            row.append(tile[br][bc])
        bayer_map.append(row)

    cfg: dict[str, Any] = {
        "pitch": float(p),
        "unit_cell": unit_cell,
        "bayer_map": bayer_map,
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True,
                "height": ml_h,
                "sharing": sharing,
                "radius_x": ml_r,
                "radius_y": ml_r,
                "material": ml_mat,
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                "shift": {"mode": "auto_cra", "cra_deg": float(cra_deg)},
                "gap": gap,
            },
            "planarization": {"thickness": plan_t, "material": "sio2"},
            "color_filter": {
                "thickness": cf_t,
                "pattern": pattern,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {
                    "enabled": True,
                    "width": round(min(0.10, 0.05 + 0.03 * p), 3),
                    "height": cf_t,
                    "material": "tungsten",
                },
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                    {"thickness": 0.015, "material": "sio2"},
                    {"thickness": 0.030, "material": "si3n4"},
                ]
            },
            "silicon": {
                "thickness": si_t,
                "material": "silicon",
                "photodiode": {
                    "position": [0.0, 0.0, pd_z_pos],
                    "size": [pd_xy, pd_xy, pd_z],
                },
                "dti": {
                    "enabled": True,
                    "width": dti_w,
                    "depth": si_t,
                    "material": dti_mat,
                },
            },
        },
    }

    if overrides:
        for dotted, value in overrides.items():
            _set_dotted(cfg, dotted, value)

    return cfg


def _set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` from key ``"a.b.c"``."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
