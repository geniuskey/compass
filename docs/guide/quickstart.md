# Quick Start

This 5-minute tutorial walks through the core COMPASS workflow: define a pixel, build the geometry, visualize it, and set up a solver.

## 1. Create a configuration

COMPASS uses Pydantic-validated configuration objects. You can create one from a Python dictionary or load from YAML.

```python
from compass.core.config_schema import CompassConfig

config = CompassConfig(**{
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True,
                "height": 0.6,
                "radius_x": 0.48,
                "radius_y": 0.48,
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "pattern": "bayer_rggb",
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "silicon": {
                "thickness": 3.0,
                "material": "silicon",
                "dti": {"enabled": True, "width": 0.1, "depth": 3.0},
            },
        },
        "bayer_map": [["R", "G"], ["G", "B"]],
    },
    "solver": {"name": "torcwa", "type": "rcwa"},
    "source": {
        "type": "planewave",
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
})

print(f"Pixel pitch: {config.pixel.pitch} um")
print(f"Unit cell: {config.pixel.unit_cell}")
print(f"Solver: {config.solver.name}")
```

Try building a pixel stack interactively by adjusting layer parameters:

<PixelStackBuilder />

Experiment with different wavelength values using the slider below:

<WavelengthSlider />

## 2. Initialize the material database

The `MaterialDB` loads built-in materials automatically and supports custom CSV files.

```python
from compass.materials.database import MaterialDB

mat_db = MaterialDB()

# Query silicon at 550 nm
n, k = mat_db.get_nk("silicon", 0.55)
print(f"Silicon at 550nm: n={n:.3f}, k={k:.4f}")

# List all available materials
print(mat_db.list_materials())
# ['air', 'cf_blue', 'cf_green', 'cf_red', 'hfo2', 'polymer_n1p56',
#  'si3n4', 'silicon', 'sio2', 'tio2', 'tungsten']
```

## 3. Build the PixelStack

The `GeometryBuilder` converts a config into a solver-agnostic `PixelStack`.

```python
from compass.geometry.builder import GeometryBuilder

builder = GeometryBuilder(config.pixel, mat_db)
pixel_stack = builder.build()

print(f"Stack z-range: [{pixel_stack.z_min:.3f}, {pixel_stack.z_max:.3f}] um")
print(f"Total thickness: {pixel_stack.total_thickness:.3f} um")
print(f"Domain size: {pixel_stack.domain_size} um")

# Inspect layers
for layer in pixel_stack.layers:
    print(f"  {layer.name}: z=[{layer.z_start:.3f}, {layer.z_end:.3f}], "
          f"material={layer.base_material}")
```

## 4. Generate layer slices

RCWA solvers consume `LayerSlice` objects -- 2D permittivity grids at each z-height.

```python
slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=128, ny=128)

for s in slices:
    print(f"  {s.name}: z=[{s.z_start:.3f}, {s.z_end:.3f}], "
          f"eps shape={s.eps_grid.shape}")
```

## 5. Visualize the structure

Plot a vertical cross-section to confirm the pixel geometry.

```python
from compass.visualization.structure_plot_2d import plot_pixel_cross_section

ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xz",
    position=0.5,  # y = 0.5 um (center of first row)
    wavelength=0.55,
)
```

This produces a matplotlib figure showing all layers from silicon (bottom) to air (top), with color-coded materials and layer annotations.

For an XY slice at a specific z-height:

```python
ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xy",
    position=4.5,  # z position in the color filter region
    wavelength=0.55,
)
```

## 6. Set up a solver

Create a solver instance and run a simulation (requires solver packages installed):

```python
from compass.solvers.base import SolverFactory

solver = SolverFactory.create("torcwa", config.solver)
solver.setup_geometry(pixel_stack)
solver.setup_source({
    "wavelength": 0.55,
    "theta": 0.0,
    "phi": 0.0,
    "polarization": "unpolarized",
})
result = solver.run()

print(f"R = {result.reflection}")
print(f"T = {result.transmission}")
print(f"A = {result.absorption}")
```

## Next steps

- [First Simulation](./first-simulation.md) -- full end-to-end walkthrough with wavelength sweep and QE plots
- [Pixel Stack Config](./pixel-stack-config.md) -- detailed YAML configuration reference
- [Choosing a Solver](./choosing-solver.md) -- which solver to use and when
