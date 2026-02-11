# MaterialDB

`compass.materials.database.MaterialDB` is the central registry for optical material properties. It provides wavelength-dependent refractive index $(n, k)$ and complex permittivity $\varepsilon$ for all materials used in simulations.

<MaterialBrowser />

## Constructor

```python
class MaterialDB:
    def __init__(self, db_path: Optional[str] = None):
```

**Parameters:**
- `db_path` -- Path to directory containing CSV material files. Defaults to the `materials/` directory at the project root.

On construction, `MaterialDB` automatically:
1. Registers built-in constant and analytic materials (air, polymers, dielectrics)
2. Loads tabulated CSV data from the materials directory (silicon, tungsten, color filters)
3. Falls back to approximate built-in data if CSV files are missing

## Querying methods

### `get_nk`

```python
def get_nk(self, name: str, wavelength: float) -> Tuple[float, float]:
```

Returns `(n, k)` -- real and imaginary parts of the refractive index -- at the given wavelength (in um).

**Raises:** `KeyError` if the material is not registered.

### `get_epsilon`

```python
def get_epsilon(self, name: str, wavelength: float) -> complex:
```

Returns complex permittivity $\varepsilon = (n + ik)^2$ at the given wavelength.

### `get_epsilon_spectrum`

```python
def get_epsilon_spectrum(self, name: str, wavelengths: np.ndarray) -> np.ndarray:
```

Returns an array of complex permittivity values over a wavelength array.

### `list_materials`

```python
def list_materials(self) -> List[str]:
```

Returns a sorted list of all registered material names.

### `has_material`

```python
def has_material(self, name: str) -> bool:
```

Returns `True` if the material exists in the database.

## Registration methods

### `register_constant`

```python
def register_constant(self, name: str, n: float, k: float = 0.0) -> None:
```

Register a material with fixed refractive index, independent of wavelength.

```python
db.register_constant("my_glass", n=1.52, k=0.0)
```

### `register_cauchy`

```python
def register_cauchy(self, name: str, A: float, B: float = 0.0, C: float = 0.0) -> None:
```

Register a material with Cauchy dispersion model:

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$

where $\lambda$ is in um. The extinction coefficient $k$ is zero (non-absorbing).

```python
db.register_cauchy("polymer", A=1.56, B=0.004, C=0.0)
```

### `register_sellmeier`

```python
def register_sellmeier(self, name: str, B: List[float], C: List[float]) -> None:
```

Register a material with Sellmeier dispersion model:

$$n^2(\lambda) = 1 + \sum_i \frac{B_i \lambda^2}{\lambda^2 - C_i}$$

```python
db.register_sellmeier(
    "sio2",
    B=[0.6961663, 0.4079426, 0.8974794],
    C=[0.0684043**2, 0.1162414**2, 9.896161**2],
)
```

### `load_csv`

```python
def load_csv(self, name: str, filepath: str, interpolation: str = "cubic_spline") -> None:
```

Load tabulated material data from a CSV file. The CSV format is:

```
# wavelength(um), n, k
0.400, 5.381, 0.340
0.410, 5.253, 0.296
...
```

Lines starting with `#` are treated as comments. Data is automatically sorted by wavelength. Interpolation uses cubic splines (requires >= 4 data points) or linear interpolation as fallback.

**Parameters:**
- `name` -- Material name for registration.
- `filepath` -- Path to CSV file.
- `interpolation` -- `"cubic_spline"` (default) or `"linear"`.

## Built-in materials

### Core materials

| Name | Type | Model | Typical n | Typical k |
|------|------|-------|-----------|-----------|
| `air` | Constant | n=1.0, k=0.0 | 1.0 | 0.0 |
| `polymer_n1p56` | Cauchy | A=1.56, B=0.004 | 1.56 | 0.0 |
| `sio2` | Sellmeier | 3-term | 1.46 | 0.0 |
| `si3n4` | Sellmeier | 2-term | 2.0 | 0.0 |
| `hfo2` | Cauchy | A=1.90, B=0.02 | 1.90 | 0.0 |
| `tio2` | Cauchy | A=2.27, B=0.05 | 2.27 | 0.0 |
| `silicon` | Tabulated | Green 2008 | 3.5-5.6 | 0.003-3.0 |
| `tungsten` | Tabulated | Approximate | 3.4-3.7 | 2.7-3.4 |
| `cf_red` | Tabulated | Lorentzian | 1.55 | 0-0.15 |
| `cf_green` | Tabulated | Lorentzian | 1.55 | 0-0.12 |
| `cf_blue` | Tabulated | Lorentzian | 1.55 | 0-0.18 |

### Metals

Tabulated optical constants for metals commonly used in CIS interconnects, metal grids, and plasmonic structures. Wavelength range: 300--1100 nm.

| Name | Formula | Source | Typical n (550nm) | Typical k (550nm) |
|------|---------|--------|--------------------|--------------------|
| `aluminum` | Al | Rakic 1998 | 0.58 | 5.52 |
| `gold` | Au | Johnson & Christy 1972 | 0.17 | 3.29 |
| `silver` | Ag | Johnson & Christy 1972 | 0.07 | 4.39 |
| `copper` | Cu | Johnson & Christy 1972 | 0.53 | 1.74 |
| `titanium` | Ti | Johnson & Christy 1974 | 2.37 | 3.30 |
| `titanium_nitride` | TiN | Patsalas 2003 | 1.31 | 2.31 |

### Dielectrics

Tabulated optical constants for dielectric films used in antireflection coatings (ARC), passivation, and transparent conductors.

| Name | Formula | Source | Typical n (550nm) | Typical k (550nm) |
|------|---------|--------|--------------------|--------------------|
| `silicon_nitride` | Si3N4 | Philipp 1973 | 2.02 | 0.0 |
| `aluminum_oxide` | Al2O3 | Malitson 1962 | 1.77 | 0.0 |
| `tantalum_pentoxide` | Ta2O5 | Bright 2012 | 2.11 | 0.0 |
| `magnesium_fluoride` | MgF2 | Dodge 1984 | 1.38 | 0.0 |
| `zinc_oxide` | ZnO | Bond 1965 | 2.07 | 0.0 |
| `indium_tin_oxide` | ITO | Konig 2014 | 1.95 | 0.0 |
| `silicon_oxynitride` | SiON | Approximate | 1.70 | 0.0 |

### Polymers

Tabulated optical constants for polymer materials used in microlenses, planarization layers, and photoresists.

| Name | Formula | Source | Typical n (550nm) | Typical k (550nm) |
|------|---------|--------|--------------------|--------------------|
| `pmma` | PMMA | Sultanova 2009 | 1.49 | 0.0 |
| `polycarbonate` | PC | Sultanova 2009 | 1.58 | 0.0 |
| `polyimide` | PI | Birkholz 2000 | 1.70 | 0.0 |
| `benzocyclobutene` | BCB | Dow Chemical | 1.54 | 0.0 |
| `su8` | SU-8 | MicroChem | 1.59 | 0.0 |

### Semiconductors

Tabulated optical constants for semiconductor materials beyond silicon, for NIR and III-V sensor modeling.

| Name | Formula | Source | Typical n (550nm) | Typical k (550nm) |
|------|---------|--------|--------------------|--------------------|
| `germanium` | Ge | Aspnes 1983 | 4.04 | 0.36 |
| `gallium_arsenide` | GaAs | Aspnes 1983 | 3.98 | 0.17 |
| `indium_phosphide` | InP | Aspnes 1983 | 3.70 | 0.17 |

## MaterialData

The internal data container for each material:

```python
@dataclass
class MaterialData:
    name: str
    mat_type: Literal["constant", "tabulated", "cauchy", "sellmeier"]
    n_const: float = 1.0
    k_const: float = 0.0
    wavelengths: Optional[np.ndarray] = None
    n_data: Optional[np.ndarray] = None
    k_data: Optional[np.ndarray] = None
    interpolation: str = "cubic_spline"
    cauchy_A: float = 1.0
    cauchy_B: float = 0.0
    cauchy_C: float = 0.0
    sellmeier_B: Optional[List[float]] = None
    sellmeier_C: Optional[List[float]] = None
```

### `MaterialData.get_nk(wavelength) -> Tuple[float, float]`

Returns `(n, k)` using the appropriate dispersion model.

### `MaterialData.get_epsilon(wavelength) -> complex`

Returns $\varepsilon = (n + ik)^2$.

## CSV auto-loading

On construction, `MaterialDB` looks for CSV files in the materials directory using these filename mappings:

**Core materials** (top-level directory):

| Material name | Expected filenames |
|--------------|-------------------|
| `silicon` | `silicon_green2008.csv`, `silicon_palik.csv` |
| `tungsten` | `tungsten.csv` |
| `cf_red` | `color_filter_red.csv` |
| `cf_green` | `color_filter_green.csv` |
| `cf_blue` | `color_filter_blue.csv` |

If a CSV file is found, it is loaded automatically. Otherwise, a built-in approximation is used.

**Extended materials** (categorized subdirectories):

| Subdirectory | Materials |
|-------------|-----------|
| `metals/` | `aluminum.csv`, `gold.csv`, `silver.csv`, `copper.csv`, `titanium.csv`, `titanium_nitride.csv` |
| `dielectrics/` | `silicon_nitride.csv`, `aluminum_oxide.csv`, `tantalum_pentoxide.csv`, `magnesium_fluoride.csv`, `zinc_oxide.csv`, `indium_tin_oxide.csv`, `silicon_oxynitride.csv` |
| `polymers/` | `pmma.csv`, `polycarbonate.csv`, `polyimide.csv`, `benzocyclobutene.csv`, `su8.csv` |
| `semiconductors/` | `germanium.csv`, `gallium_arsenide.csv`, `indium_phosphide.csv` |

Extended material CSV files are optional. If a file is missing, the corresponding material is simply not registered (no fallback).
