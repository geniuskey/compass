# PixelStack

`compass.geometry.pixel_stack.PixelStack` is the solver-agnostic representation of a BSI pixel structure. It constructs the full 3D pixel from a YAML configuration and provides output in both RCWA (layer slices) and FDTD (voxel grid) formats.

## Constructor

```python
class PixelStack:
    def __init__(self, config: dict, material_db: Optional[MaterialDB] = None):
```

**Parameters:**
- `config` -- Configuration dictionary. Must contain a `"pixel"` key (or be the pixel config directly).
- `material_db` -- `MaterialDB` instance. Created automatically if `None`.

**Example:**

```python
from compass.materials.database import MaterialDB
from compass.geometry.pixel_stack import PixelStack

db = MaterialDB()
config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "silicon": {"thickness": 3.0, "material": "silicon"},
            "color_filter": {"thickness": 0.6},
        },
    }
}
stack = PixelStack(config, db)
```

## Properties

### `domain_size -> Tuple[float, float]`

Returns `(Lx, Ly)` simulation domain size in um. Equal to `(pitch * unit_cell[1], pitch * unit_cell[0])`.

### `total_height -> float`

Total stack height from bottom of silicon to top of air layer, in um.

### `z_range -> Tuple[float, float]`

`(z_min, z_max)` of the stack. `z_min = 0.0` at bottom of silicon.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pitch` | `float` | Pixel pitch in um |
| `unit_cell` | `Tuple[int, int]` | (rows, cols) in unit cell |
| `material_db` | `MaterialDB` | Material database instance |
| `layers` | `List[Layer]` | Ordered list of layers (bottom to top) |
| `microlenses` | `List[MicrolensSpec]` | Microlens specs for each pixel |
| `photodiodes` | `List[PhotodiodeSpec]` | Photodiode specs for each pixel |
| `bayer_map` | `List[List[str]]` | Color assignment matrix |

## Methods

### `get_layer_slices`

```python
def get_layer_slices(
    self,
    wavelength: float,
    nx: int = 128,
    ny: int = 128,
    n_lens_slices: int = 30,
) -> List[LayerSlice]:
```

Generates the z-wise layer decomposition for RCWA solvers. Each `LayerSlice` contains a 2D complex permittivity grid `eps_grid` of shape `(ny, nx)`.

**Parameters:**
- `wavelength` -- Wavelength in um for computing permittivity values.
- `nx`, `ny` -- Spatial grid resolution in x and y.
- `n_lens_slices` -- Number of staircase slices for approximating the curved microlens.

**Returns:** List of `LayerSlice` objects ordered from bottom (z_min) to top (z_max).

**How it works:**
- Uniform layers (air, planarization, BARL) produce a single slice with constant permittivity.
- The color filter layer produces a patterned slice with Bayer-arranged materials and optional metal grid.
- The silicon layer includes optional DTI trenches.
- The microlens layer is discretized into `n_lens_slices` staircase slices.

### `get_permittivity_grid`

```python
def get_permittivity_grid(
    self,
    wavelength: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 128,
) -> np.ndarray:
```

Generates a 3D complex permittivity array for FDTD solvers.

**Parameters:**
- `wavelength` -- Wavelength in um.
- `nx`, `ny`, `nz` -- 3D grid resolution.

**Returns:** Complex array of shape `(ny, nx, nz)`.

### `get_photodiode_mask`

```python
def get_photodiode_mask(
    self,
    nx: int = 64,
    ny: int = 64,
    nz: int = 128,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
```

Generates 3D boolean masks for photodiode regions.

**Returns:** Tuple of `(full_mask, per_pixel_masks)` where `per_pixel_masks` maps pixel names to individual 3D masks.

## Layer build order

The `PixelStack` constructs layers from bottom to top, with `z=0` at the bottom of the silicon:

1. **Silicon** (z=0 to z=thickness_si)
2. **BARL layers** (thin dielectric stack)
3. **Color filter** (patterned with Bayer pattern + metal grid)
4. **Planarization** (uniform dielectric)
5. **Microlens** (curved, approximated by staircase)
6. **Air** (top layer)

## Data types

### Layer

```python
@dataclass
class Layer:
    name: str               # "silicon", "color_filter", "microlens", etc.
    z_start: float          # Bottom z-coordinate in um
    z_end: float            # Top z-coordinate in um
    thickness: float        # Layer thickness in um
    base_material: str      # Material name
    is_patterned: bool      # Whether layer has lateral patterns
```

### LayerSlice

```python
@dataclass
class LayerSlice:
    z_start: float          # Bottom z-coordinate
    z_end: float            # Top z-coordinate
    thickness: float        # Slice thickness
    eps_grid: np.ndarray    # 2D complex permittivity (ny, nx)
    name: str               # Slice identifier
    material: str           # Base material name
```

### MicrolensSpec

```python
@dataclass
class MicrolensSpec:
    height: float           # Lens height in um
    radius_x: float         # Semi-axis in x (um)
    radius_y: float         # Semi-axis in y (um)
    material: str           # Lens material name
    profile_type: str       # "superellipse" or "spherical"
    n_param: float          # Squareness parameter
    alpha_param: float      # Curvature parameter
    shift_x: float          # CRA x-offset (um)
    shift_y: float          # CRA y-offset (um)
```

### PhotodiodeSpec

```python
@dataclass
class PhotodiodeSpec:
    position: Tuple[float, float, float]  # (x, y, z) offset in um
    size: Tuple[float, float, float]      # (dx, dy, dz) in um
    pixel_index: Tuple[int, int]          # (row, col) in unit cell
    color: str                             # "R", "G", or "B"
```
