# SolverBase

`compass.solvers.base.SolverBase` is the abstract base class that all EM solvers in COMPASS must implement. It defines the uniform interface used by runners and analysis modules.

## Abstract interface

```python
class SolverBase(ABC):
    def __init__(self, config: dict, device: str = "cpu"):
```

**Parameters:**
- `config` -- Solver configuration dictionary (from Hydra/YAML).
- `device` -- Compute device: `"cpu"`, `"cuda"`, or `"mps"`.

## Properties

### `name -> str`

Solver name, derived from `config["name"]` or the class name.

### `solver_type -> str`

Solver type: `"rcwa"` or `"fdtd"`, from `config["type"]`.

## Abstract methods

Every solver backend must implement these four methods:

### `setup_geometry`

```python
@abstractmethod
def setup_geometry(self, pixel_stack: PixelStack) -> None:
```

Convert the solver-agnostic `PixelStack` to the solver's internal geometry representation.

- RCWA solvers call `pixel_stack.get_layer_slices()` to get 2D permittivity grids.
- FDTD solvers call `pixel_stack.get_permittivity_grid()` to get a 3D voxel grid.

### `setup_source`

```python
@abstractmethod
def setup_source(self, source_config: dict) -> None:
```

Configure the excitation source (wavelength, angle, polarization).

### `run`

```python
@abstractmethod
def run(self) -> SimulationResult:
```

Execute the simulation and return a standardized `SimulationResult`.

### `get_field_distribution`

```python
@abstractmethod
def get_field_distribution(
    self,
    component: str,   # "Ex", "Ey", "Ez", "|E|2"
    plane: str,        # "xy", "xz", "yz"
    position: float,   # Position along normal axis in um
) -> np.ndarray:
```

Extract a 2D field slice from the simulation.

## Concrete methods

### `validate_energy_balance`

```python
def validate_energy_balance(
    self,
    result: SimulationResult,
    tolerance: float = 0.01,
) -> bool:
```

Checks that $R + T + A \approx 1$ for the simulation result. Returns `True` if energy is conserved within the tolerance. Logs a warning on violation.

### `run_timed`

```python
def run_timed(self) -> SimulationResult:
```

Wraps `run()` with timing instrumentation. Adds to `result.metadata`:
- `runtime_seconds` -- wall-clock time
- `solver_name` -- solver name string
- `solver_type` -- `"rcwa"` or `"fdtd"`
- `device` -- compute device used

## SolverFactory

`compass.solvers.base.SolverFactory` creates solver instances by name using a registry pattern.

### `SolverFactory.create`

```python
@classmethod
def create(cls, name: str, config: dict, device: str = "cpu") -> SolverBase:
```

Create a solver instance. Performs lazy import if the solver module is not yet loaded.

**Parameters:**
- `name` -- Solver name: `"torcwa"`, `"grcwa"`, `"meent"`, `"fdtd_flaport"`, etc.
- `config` -- Solver config dictionary.
- `device` -- Compute device.

**Raises:** `ValueError` if the solver name is unknown and its module cannot be imported.

### `SolverFactory.register`

```python
@classmethod
def register(cls, name: str, solver_class: type) -> None:
```

Register a solver class. Called automatically by solver modules on import.

### `SolverFactory.list_solvers`

```python
@classmethod
def list_solvers(cls) -> list:
```

Returns names of all registered solvers.

## Available solver backends

| Name | Module | Type |
|------|--------|------|
| `torcwa` | `compass.solvers.rcwa.torcwa_solver` | RCWA |
| `grcwa` | `compass.solvers.rcwa.grcwa_solver` | RCWA |
| `meent` | `compass.solvers.rcwa.meent_solver` | RCWA |
| `fdtd_flaport` | `compass.solvers.fdtd.flaport_solver` | FDTD |

## SimulationResult

The standardized output from all solvers:

```python
@dataclass
class SimulationResult:
    qe_per_pixel: Dict[str, np.ndarray]    # Pixel name -> QE spectrum
    wavelengths: np.ndarray                 # Wavelength array in um
    fields: Optional[Dict[str, FieldData]]  # Field data per wavelength
    poynting: Optional[Dict[str, np.ndarray]]
    reflection: Optional[np.ndarray]        # R(lambda)
    transmission: Optional[np.ndarray]      # T(lambda)
    absorption: Optional[np.ndarray]        # A(lambda)
    metadata: Dict                          # Timing, solver info
```

### FieldData

```python
@dataclass
class FieldData:
    Ex: Optional[np.ndarray]   # x-component of E-field (3D complex)
    Ey: Optional[np.ndarray]
    Ez: Optional[np.ndarray]
    x: Optional[np.ndarray]    # Coordinate arrays
    y: Optional[np.ndarray]
    z: Optional[np.ndarray]

    @property
    def E_intensity(self) -> Optional[np.ndarray]:
        """Compute |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2."""
```

## Implementing a custom solver

To add a new solver backend:

```python
from compass.solvers.base import SolverBase, SolverFactory

class MySolver(SolverBase):
    def setup_geometry(self, pixel_stack):
        self._pixel_stack = pixel_stack
        # Convert to internal representation...

    def setup_source(self, source_config):
        self._source_config = source_config

    def run(self):
        # Run simulation, return SimulationResult
        ...

    def get_field_distribution(self, component, plane, position):
        # Extract 2D field slice
        ...

# Register
SolverFactory.register("my_solver", MySolver)
```
