# Analysis

COMPASS provides analysis modules for computing quantum efficiency, checking energy balance, and comparing solver results.

## QECalculator

`compass.analysis.qe_calculator.QECalculator` computes per-pixel QE from simulation results.

### `from_absorption`

```python
@staticmethod
def from_absorption(
    absorption_per_pixel: Dict[str, np.ndarray],
    incident_power: np.ndarray,
) -> Dict[str, np.ndarray]:
```

Computes QE from the absorbed power in each photodiode:

$$\text{QE}_i(\lambda) = \frac{P_{\text{absorbed},i}(\lambda)}{P_{\text{incident}}(\lambda)}$$

Results are clipped to the range [0, 1].

**Parameters:**
- `absorption_per_pixel` -- Dict mapping pixel name (e.g., `"R_0_0"`) to absorbed power spectrum.
- `incident_power` -- Total incident power spectrum.

**Returns:** Dict mapping pixel names to QE arrays.

### `from_poynting_flux`

```python
@staticmethod
def from_poynting_flux(
    flux_top: np.ndarray,
    flux_bottom: np.ndarray,
    incident_power: np.ndarray,
) -> np.ndarray:
```

Computes QE from the Poynting vector flux difference at the top and bottom of the photodiode region:

$$\text{QE} = \frac{S_{z,\text{top}} - S_{z,\text{bottom}}}{P_{\text{incident}}}$$

### `compute_crosstalk`

```python
@staticmethod
def compute_crosstalk(
    qe_per_pixel: Dict[str, np.ndarray],
    bayer_map: list,
) -> np.ndarray:
```

Computes the crosstalk matrix. Entry $(i,j)$ represents the fraction of light intended for pixel $i$ that is detected by pixel $j$.

**Returns:** 3D array of shape `(n_pixels, n_pixels, n_wavelengths)`.

<CrosstalkHeatmap />

### `spectral_response`

```python
@staticmethod
def spectral_response(
    qe_per_pixel: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
```

Groups QE by color channel and averages same-color pixels. Useful for getting per-channel (R, G, B) spectral response.

**Returns:** Dict mapping color name (e.g., `"R"`) to `(wavelengths, mean_qe)` tuples.

**Example:**

```python
channel_qe = QECalculator.spectral_response(result.qe_per_pixel, result.wavelengths)
for color, (wl, qe) in channel_qe.items():
    print(f"{color}: peak QE = {qe.max():.2%} at {wl[qe.argmax()]*1000:.0f} nm")
```

<QESpectrumChart />

## EnergyBalance

`compass.analysis.energy_balance.EnergyBalance` verifies energy conservation.

### `check`

```python
@staticmethod
def check(
    result: SimulationResult,
    tolerance: float = 0.01,
) -> dict:
```

Checks that $R + T + A \approx 1$ at all wavelengths.

**Returns:** Dictionary with keys:
- `valid` (bool) -- `True` if max error < tolerance
- `max_error` (float) -- Maximum $|R + T + A - 1|$
- `mean_error` (float) -- Mean error across wavelengths
- `per_wavelength` (np.ndarray) -- Error at each wavelength
- `R`, `T`, `A` (np.ndarray) -- Individual spectra

**Example:**

```python
from compass.analysis.energy_balance import EnergyBalance

check = EnergyBalance.check(result, tolerance=0.02)
if not check["valid"]:
    print(f"Energy violation: max error = {check['max_error']:.4f}")
```

<EnergyBalanceDiagram />

## SolverComparison

`compass.analysis.solver_comparison.SolverComparison` compares results from multiple solvers.

### Constructor

```python
class SolverComparison:
    def __init__(
        self,
        results: List[SimulationResult],
        labels: List[str],
        reference_idx: int = 0,
    ):
```

**Parameters:**
- `results` -- List of `SimulationResult` objects from different solvers.
- `labels` -- Display labels for each solver.
- `reference_idx` -- Index of the reference result (default: first).

### `qe_difference`

```python
def qe_difference(self) -> Dict[str, np.ndarray]:
```

Computes absolute QE difference vs. the reference for each pixel. Returns a dict with keys like `"grcwa_vs_torcwa_R_0_0"`.

### `qe_relative_error`

```python
def qe_relative_error(self) -> Dict[str, np.ndarray]:
```

Computes relative QE error (%) vs. the reference:

$$\text{Error}_\% = 100 \times \frac{|\text{QE}_\text{solver} - \text{QE}_\text{ref}|}{|\text{QE}_\text{ref}|}$$

### `runtime_comparison`

```python
def runtime_comparison(self) -> Dict[str, float]:
```

Returns a dict mapping solver label to runtime in seconds (from `result.metadata["runtime_seconds"]`).

### `summary`

```python
def summary(self) -> dict:
```

Returns a comprehensive comparison summary:

```python
{
    "max_qe_diff": {"grcwa_vs_torcwa_R_0_0": 0.012, ...},
    "mean_qe_diff": {"grcwa_vs_torcwa_R_0_0": 0.004, ...},
    "max_qe_relative_error_pct": {"grcwa_vs_torcwa_R_0_0": 2.1, ...},
    "runtimes_seconds": {"torcwa": 0.3, "grcwa": 0.5},
}
```

**Example:**

```python
from compass.analysis.solver_comparison import SolverComparison

comp = SolverComparison(
    results=[rcwa_result, fdtd_result],
    labels=["torcwa", "fdtd_flaport"],
)

summary = comp.summary()
print(f"Max QE difference: {max(summary['max_qe_diff'].values()):.4f}")
print(f"Runtimes: {summary['runtimes_seconds']}")
```
