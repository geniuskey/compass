# Config Reference

Complete reference for the COMPASS configuration schema. All configuration is validated by Pydantic models defined in `compass.core.config_schema`.

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
| `thickness` | float | `0.6` | CFA thickness (um) |
| `pattern` | str | `"bayer_rggb"` | Filter pattern |
| `materials` | dict | `{"R":"cf_red","G":"cf_green","B":"cf_blue"}` | Color-to-material mapping |
| `grid.enabled` | bool | `true` | Enable metal grid |
| `grid.width` | float | `0.05` | Grid line width (um) |
| `grid.height` | float | `0.6` | Grid height (um) |
| `grid.material` | str | `"tungsten"` | Grid material |

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
| `dti.width` | float | `0.1` | Trench width (um) |
| `dti.depth` | float | `3.0` | Trench depth (um) |
| `dti.material` | str | `"sio2"` | Fill material |

## solver: SolverConfig

```yaml
solver:
  name: torcwa
  type: rcwa
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  convergence: ...
  stability: ...
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | `"torcwa"` | Solver backend name |
| `type` | str | `"rcwa"` | `"rcwa"` or `"fdtd"` |
| `params` | dict | `{"fourier_order": [9,9]}` | Solver-specific parameters |

### solver.convergence: ConvergenceConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_converge` | bool | `false` | Auto Fourier order sweep |
| `order_range` | [int, int] | `[5, 25]` | Min/max Fourier order |
| `qe_tolerance` | float | `0.01` | Convergence threshold |
| `spacing_range` | [float, float] or null | `null` | Grid spacing range for FDTD |

### solver.stability: StabilityConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `precision_strategy` | str | `"mixed"` | `"float32"`, `"float64"`, `"mixed"`, `"adaptive"` |
| `allow_tf32` | bool | `false` | Allow TF32 on Ampere+ GPUs |
| `eigendecomp_device` | str | `"cpu"` | `"cpu"` or `"gpu"` |
| `fourier_factorization` | str | `"li_inverse"` | `"naive"`, `"li_inverse"`, `"normal_vector"` |
| `energy_check.enabled` | bool | `true` | Enable energy balance check |
| `energy_check.tolerance` | float | `0.02` | Max allowed |R+T+A-1| |
| `energy_check.auto_retry_float64` | bool | `true` | Auto retry in float64 on failure |
| `eigenvalue_broadening` | float | `1e-10` | Degeneracy detection threshold |
| `condition_number_warning` | float | `1e12` | Warn on ill-conditioned matrices |

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
    grcwa.yaml
    meent.yaml
    fdtd_flaport.yaml
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
```

Override any parameter from the command line:

```bash
python scripts/run_simulation.py \
    pixel.pitch=0.8 \
    solver.params.fourier_order=[11,11] \
    source.wavelength.mode=sweep
```
