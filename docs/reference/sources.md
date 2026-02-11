# Sources

COMPASS supports multiple illumination source types for simulating different optical scenarios.

## PlanewaveSource

`compass.sources.planewave.PlanewaveSource` models a single plane wave or a set of plane waves at different wavelengths.

### Constructor

```python
@dataclass
class PlanewaveSource:
    wavelengths: np.ndarray          # Array of wavelengths in um
    theta_deg: float = 0.0           # Polar incidence angle (degrees)
    phi_deg: float = 0.0             # Azimuthal angle (degrees)
    polarization: str = "unpolarized"  # "TE", "TM", or "unpolarized"
```

### Factory method

```python
@classmethod
def from_config(cls, source_config: dict) -> PlanewaveSource:
```

Creates a `PlanewaveSource` from a source configuration dictionary. Supports three wavelength modes:

**Single wavelength:**
```yaml
source:
  wavelength:
    mode: single
    value: 0.55     # 550 nm
```

**Wavelength sweep:**
```yaml
source:
  wavelength:
    mode: sweep
    sweep:
      start: 0.38   # 380 nm
      stop: 0.78    # 780 nm
      step: 0.01    # 10 nm steps
```

**Wavelength list:**
```yaml
source:
  wavelength:
    mode: list
    values: [0.45, 0.55, 0.65]
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `theta_rad` | `float` | Polar angle in radians |
| `phi_rad` | `float` | Azimuthal angle in radians |
| `n_wavelengths` | `int` | Number of wavelengths |
| `is_unpolarized` | `bool` | Whether TE+TM averaging is needed |

### Methods

#### `get_polarization_runs`

```python
def get_polarization_runs(self) -> List[str]:
```

Returns `["TE", "TM"]` for unpolarized light, or `["TE"]` / `["TM"]` for polarized.

<PolarizationViewer />

#### `to_solver_params`

```python
def to_solver_params(self) -> dict:
```

Converts to a solver-compatible parameter dictionary with keys: `wavelengths`, `theta_rad`, `phi_rad`, `polarization`, `polarization_runs`.

## ConeIllumination

`compass.sources.cone_illumination.ConeIllumination` models the exit-pupil illumination from a camera lens, specified by the Chief Ray Angle (CRA) and f-number.

### Constructor

```python
class ConeIllumination:
    def __init__(
        self,
        cra_deg: float = 0.0,
        f_number: float = 2.0,
        n_points: int = 37,
        sampling: str = "fibonacci",
        weighting: str = "cosine",
    ):
```

**Parameters:**
- `cra_deg` -- Chief Ray Angle in degrees. The central direction of the illumination cone.
- `f_number` -- Lens f-number. Determines the cone half-angle: $\theta_{half} = \arcsin(1 / 2F)$.
- `n_points` -- Number of angular sampling points within the cone.
- `sampling` -- Sampling method: `"fibonacci"` (default) or `"grid"`.
- `weighting` -- Angular weighting: `"uniform"`, `"cosine"` (default), `"cos4"`, or `"gaussian"`.

<ConeIlluminationViewer />

### Configuration

```yaml
source:
  type: cone_illumination
  cone:
    cra_deg: 15.0
    f_number: 2.0
    pupil_shape: circular
    sampling:
      type: fibonacci
      n_points: 37
    weighting: cosine
```

### Methods

#### `get_sampling_points`

```python
def get_sampling_points(self) -> List[Tuple[float, float, float]]:
```

Returns a list of `(theta_deg, phi_deg, weight)` tuples representing the angular sampling of the illumination cone. Weights are normalized to sum to 1.

### Sampling methods

**Fibonacci spiral**: Generates quasi-uniform sampling on the cone cap using the golden ratio. Provides good angular coverage with fewer points than a regular grid.

**Grid**: Regular $N_\theta \times N_\phi$ grid in polar coordinates. Uniform in angle but over-samples near the cone center.

### Weighting options

| Weight | Formula | Use case |
|--------|---------|----------|
| `uniform` | $w = 1$ | Equal weight to all angles |
| `cosine` | $w = \cos\theta$ | Lambertian illumination (default) |
| `cos4` | $w = \cos^4\theta$ | More realistic camera lens roll-off |
| `gaussian` | $w = \exp(-\theta^2/2\sigma^2)$ | Apodized illumination |

### Usage with COMPASS

Cone illumination is implemented as a weighted sum of plane-wave simulations:

$$\text{QE}_\text{cone} = \sum_i w_i \cdot \text{QE}(\theta_i, \phi_i)$$

The runner performs one RCWA or FDTD simulation per sampling point and averages the results.

## RayFileReader

`compass.sources.ray_file_reader` imports ray data from optical design tools (e.g., Zemax).

### Configuration

```yaml
source:
  ray_file:
    enabled: true
    path: "data/zemax_rays.json"
    format: zemax_json    # or "csv"
```

This allows importing realistic illumination conditions directly from lens design simulations.

## Source configuration reference

```yaml
source:
  type: "planewave"              # "planewave" or "cone_illumination"
  wavelength:
    mode: "single"               # "single", "sweep", or "list"
    value: 0.55                  # For single mode
    sweep:                       # For sweep mode
      start: 0.38
      stop: 0.78
      step: 0.01
    values: [0.45, 0.55, 0.65]  # For list mode
  angle:
    theta_deg: 0.0               # Polar angle (degrees)
    phi_deg: 0.0                 # Azimuthal angle (degrees)
  polarization: "unpolarized"    # "TE", "TM", or "unpolarized"
  cone:                          # For cone_illumination type
    cra_deg: 0.0
    f_number: 2.0
    pupil_shape: "circular"
    sampling:
      type: "fibonacci"
      n_points: 37
    weighting: "cosine"
  ray_file:                      # External ray import
    enabled: false
    path: ""
    format: "zemax_json"
```
