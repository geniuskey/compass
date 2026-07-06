# Config Reference

Complete reference for the COMPASS configuration schema. The composed Hydra config is validated by the Pydantic models in `compass.core.config_schema` before a simulation runs: unknown keys in any nested section (a typo like `thicknes:`) fail fast with a validation error. Only the top level accepts extra sections, so experiment overlays can add `experiment`, `optimization`, or sweep tables consumed by dedicated runners.

## Top-level: CompassConfig

```yaml
pixel: ...          # PixelConfig
solver: ...         # SolverConfig
source: ...         # SourceConfig
compute: ...        # ComputeConfig
experiment_name: "default"
output_dir: "./outputs"
seed: 42
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name` | str | `"default"` | Experiment identifier for output directory |
| `output_dir` | str | `"./outputs"` | Base output directory |
| `seed` | int | `42` | Random seed for reproducibility |

## pixel: PixelConfig

```yaml
pixel:
  pitch: 1.0
  unit_cell: [2, 2]
  bayer_map: [["R", "G"], ["G", "B"]]
  layers: ...       # LayersConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pitch` | float | `1.0` | Pixel pitch in um |
| `unit_cell` | [int, int] | `[2, 2]` | Unit cell size [rows, cols] |
| `bayer_map` | list[list[str]] | `[["R","G"],["G","B"]]` | Color channel map |

### pixel.layers: LayersConfig

```yaml
layers:
  air: {thickness: 1.0, material: "air"}
  microlens: ...
  planarization: {thickness: 0.3, material: "sio2"}
  color_filter: ...
  barl: ...
  silicon: ...
```

<PixelStackBuilder />

### pixel.layers.microlens: MicrolensConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable microlens |
| `height` | float | `0.6` | Lens sag height (um) |
| `radius_x` | float | `0.48` | Semi-axis x (um) |
| `radius_y` | float | `0.48` | Semi-axis y (um) |
| `material` | str | `"polymer_n1p56"` | Lens material |
| `profile.type` | str | `"superellipse"` | Profile model |
| `profile.n` | float | `2.5` | Squareness parameter |
| `profile.alpha` | float | `1.0` | Curvature parameter |
| `shift.mode` | str | `"auto_cra"` | Shift mode: `"none"`, `"manual"`, `"auto_cra"` |
| `shift.cra_deg` | float | `0.0` | CRA for auto shift (degrees) |
| `shift.shift_x` | float | `0.0` | Manual x-shift (um) |
| `shift.shift_y` | float | `0.0` | Manual y-shift (um) |
| `gap` | float | `0.0` | Inter-lens gap (um) |

### pixel.layers.color_filter: ColorFilterConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `thickness` | float | `0.6` | Legacy flat CFA thickness (um), used when per-channel thickness is absent |
| `pattern` | str | `"bayer_rggb"` | Filter pattern |
| `materials` | dict | `{"R":"cf_red","G":"cf_green","B":"cf_blue"}` | Legacy color-to-material mapping |
| `red.material`, `green.material`, `blue.material` | str | `cf_*` | Per-channel material name |
| `red.thickness`, `green.thickness`, `blue.thickness` | float | `thickness` | Per-channel CFA height (um) |
| `red.contact_angle`, `green.contact_angle`, `blue.contact_angle` | float | `90.0` | Sidewall angle in degrees for the trapezoid above the grid |
| `grid.enabled` | bool | `true` | Enable metal grid |
| `grid.width` | float | `0.05` | Grid line width (um) |
| `grid.thickness` | float | `thickness` | Metal grid height (um) |
| `grid.height` | float | `0.6` | Legacy alias for `grid.thickness` |
| `grid.material` | str | `"tungsten"` | Grid material |
| `grid.corner_radius` | float | `0.0` | Rounded-rectangle corner radius `r` (um) for each CF cell, identical at all four corners. `0` = sharp; `> 0` models the CF as a rounded rectangle and the grid as its complement. Clamped to `(pitch - grid.width) / 2`. |
| `n_slices` | int | `8` for tapered surfaces | Number of z-slices used to staircase the tapered color-filter relief |

### pixel.layers.barl: BarlConfig

```yaml
barl:
  layers:
    - {thickness: 0.010, material: "sio2"}
    - {thickness: 0.025, material: "hfo2"}
```

List of `{thickness, material}` pairs, ordered top to bottom.

### pixel.layers.silicon: SiliconConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `thickness` | float | `3.0` | Silicon thickness (um) |
| `material` | str | `"silicon"` | Substrate material |
| `photodiode.position` | [float, float, float] | `[0, 0, 0.5]` | PD offset (x, y, z) um |
| `photodiode.size` | [float, float, float] | `[0.7, 0.7, 2.0]` | PD extent (dx, dy, dz) um |
| `dti.enabled` | bool | `true` | Enable DTI |
| `dti.mode` | str | `"fdti"` | `"fdti"` (full) or `"bdti"` (backside partial) |
| `dti.width` | float | `0.1` | Trench width at the opening (um) |
| `dti.depth` | float | `3.0` | Trench depth (um) |
| `dti.material` | str | `"sio2"` | Core fill material |
| `dti.liner.enabled` | bool | `false` | Conformal high-k liner on trench sidewalls |
| `dti.liner.material` | str | `"al2o3"` | Liner material |
| `dti.liner.thickness` | float | `0.0` | Liner thickness (um) |
| `dti.taper_angle` | float | `90.0` | Sidewall angle from substrate plane (90 = vertical) |
| `dti.n_slices` | int | `6` | Staircase z-slices for tapered trenches |
| `surface_texture.enabled` | bool | `false` | Backside inverted-pyramid array for NIR light trapping |
| `surface_texture.height` | float | `0.3` | Pyramid height (um) |
| `surface_texture.period` | float or null | `null` | Pyramid period (um); defaults to pixel pitch |
| `surface_texture.fill_material` | str | `"sio2"` | Pit back-fill material |
| `surface_texture.n_slices` | int | `8` | Staircase z-slices for the pyramids |

## solver: SolverConfig

```yaml
solver:
  name: torcwa
  type: rcwa
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  stability: ...
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | `"torcwa"` | Solver backend name |
| `type` | str | `"rcwa"` | `"rcwa"`, `"fdtd"`, or `"tmm"` |
| `params` | dict | `{"fourier_order": [9,9]}` | Solver-specific parameters (see below) |

### solver.params semantics

`params` is passed through to the solver adapter, so the meaning of each key is
solver-specific. The most important difference:

- **torcwa / meent / fmmax** use a per-axis Fourier order
  `fourier_order: [m, m]` → `(2m+1)²` total plane waves.
- **grcwa** truncates by TOTAL plane-wave count: set `nG` (e.g. `nG: 49`).
  `fourier_order[0]` is accepted as a legacy fallback with a warning, but it is
  **not** equivalent to the same number in the other RCWA solvers.
- **FDTD solvers** use `grid_spacing` (um) or `resolution` (pixels/um, meep).

Every result records `metadata["qe_method"]` (`field_integration`,
`eps_imag_weight`, `tmm_1d_analytic`) so cross-solver QE differences can be
attributed to post-processing methodology rather than solver accuracy.

### solver.stability: StabilityConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `precision_strategy` | str | `"mixed"` | `"float32"`, `"float64"`, `"mixed"`, `"adaptive"` — consumed by the diagnostics pre-simulation checks |
| `allow_tf32` | bool | `false` | Allow TF32 on Ampere+ GPUs (keep `false` for RCWA) |
| `fourier_factorization` | str | `"li_inverse"` | `"naive"`, `"li_inverse"`, `"normal_vector"` |
| `energy_check.enabled` | bool | `true` | Validate R+T+A ≈ 1 after the run |
| `energy_check.tolerance` | float | `0.02` | Max allowed \|R+T+A-1\| |
| `energy_check.auto_retry_float64` | bool | `true` | On violation, rerun once with the dtype promoted (complex64→complex128, float32→float64); the retry is tagged `metadata["energy_retry_dtype"]` |

## source: SourceConfig

```yaml
source:
  type: planewave
  wavelength:
    mode: single
    value: 0.55
  angle:
    theta_deg: 0.0
    phi_deg: 0.0
  polarization: unpolarized
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | str | `"planewave"` | `"planewave"` or `"cone_illumination"` |
| `wavelength.mode` | str | `"single"` | `"single"`, `"sweep"`, or `"list"` |
| `wavelength.value` | float | `0.55` | Single wavelength (um) |
| `wavelength.sweep.start` | float | `0.38` | Sweep start (um) |
| `wavelength.sweep.stop` | float | `0.78` | Sweep stop (um) |
| `wavelength.sweep.step` | float | `0.01` | Sweep step (um) |
| `wavelength.values` | list[float] | null | Explicit wavelength list |
| `angle.theta_deg` | float | `0.0` | Polar angle (degrees) |
| `angle.phi_deg` | float | `0.0` | Azimuthal angle (degrees) |
| `polarization` | str | `"unpolarized"` | `"TE"`, `"TM"`, or `"unpolarized"` |

## compute: ComputeConfig

```yaml
compute:
  backend: auto
  gpu_id: 0
  num_workers: 4
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | str | `"auto"` | `"auto"`, `"cuda"`, `"cpu"`, `"mps"` |
| `gpu_id` | int | `0` | GPU device index |
| `num_workers` | int | `4` | Worker threads for parallel tasks |

## Hydra config structure

COMPASS uses Hydra for modular configuration:

```
configs/
  config.yaml           # Main config with defaults
  pixel/
    default_bsi_1um.yaml
    default_bsi_0p8um.yaml
  solver/
    torcwa.yaml
    grcwa.yaml            # + grcwa_fast.yaml, grcwa_converged.yaml presets
    meent.yaml
    fmmax.yaml
    fdtd_flaport.yaml
    fdtdz.yaml
    fdtdx.yaml
    meep.yaml
    tmm.yaml
  source/
    planewave.yaml
    wavelength_sweep.yaml
    cone_illumination.yaml
  compute/
    cuda.yaml
    cpu.yaml
    mps.yaml
  experiment/
    solver_comparison.yaml
    qe_benchmark.yaml
    roi_sweep.yaml
    optimize_microlens.yaml
```

Override any parameter from the command line:

```bash
python scripts/run_simulation.py \
    pixel.pitch=0.8 \
    solver.params.fourier_order=[11,11] \
    source.wavelength.mode=sweep
```
