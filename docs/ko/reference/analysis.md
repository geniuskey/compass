# 분석(Analysis)

COMPASS는 양자 효율(quantum efficiency) 계산, 에너지 밸런스(energy balance) 확인 및 솔버 결과 비교를 위한 분석 모듈을 제공합니다.

## QECalculator

`compass.analysis.qe_calculator.QECalculator`는 시뮬레이션 결과로부터 픽셀별 QE를 계산합니다.

### `from_absorption`

```python
@staticmethod
def from_absorption(
    absorption_per_pixel: Dict[str, np.ndarray],
    incident_power: np.ndarray,
) -> Dict[str, np.ndarray]:
```

각 포토다이오드(photodiode)의 흡수 전력으로부터 QE를 계산합니다:

$$\text{QE}_i(\lambda) = \frac{P_{\text{absorbed},i}(\lambda)}{P_{\text{incident}}(\lambda)}$$

결과는 [0, 1] 범위로 클리핑됩니다.

**매개변수:**
- `absorption_per_pixel` -- 픽셀 이름(예: `"R_0_0"`)을 흡수 전력 스펙트럼에 매핑하는 딕셔너리.
- `incident_power` -- 전체 입사 전력 스펙트럼.

**반환값:** 픽셀 이름을 QE 배열에 매핑하는 딕셔너리.

### `from_poynting_flux`

```python
@staticmethod
def from_poynting_flux(
    flux_top: np.ndarray,
    flux_bottom: np.ndarray,
    incident_power: np.ndarray,
) -> np.ndarray:
```

포토다이오드 영역의 상단과 하단에서의 포인팅 벡터(Poynting vector) 플럭스 차이로부터 QE를 계산합니다:

$$\text{QE} = \frac{S_{z,\text{top}} - S_{z,\text{bottom}}}{P_{\text{incident}}}$$

### `compute_crosstalk`

```python
@staticmethod
def compute_crosstalk(
    qe_per_pixel: Dict[str, np.ndarray],
    bayer_map: list,
) -> np.ndarray:
```

크로스토크 매트릭스(crosstalk matrix)를 계산합니다. 항목 $(i,j)$는 픽셀 $i$에 의도된 광량 중 픽셀 $j$에서 검출되는 비율을 나타냅니다.

**반환값:** `(n_pixels, n_pixels, n_wavelengths)` 형상의 3D 배열.

### `spectral_response`

```python
@staticmethod
def spectral_response(
    qe_per_pixel: Dict[str, np.ndarray],
    wavelengths: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
```

QE를 색상 채널별로 그룹화하고 동일 색상 픽셀을 평균합니다. 채널별(R, G, B) 분광 응답(spectral response)을 얻는 데 유용합니다.

**반환값:** 색상 이름(예: `"R"`)을 `(wavelengths, mean_qe)` 튜플에 매핑하는 딕셔너리.

**예제:**

```python
channel_qe = QECalculator.spectral_response(result.qe_per_pixel, result.wavelengths)
for color, (wl, qe) in channel_qe.items():
    print(f"{color}: peak QE = {qe.max():.2%} at {wl[qe.argmax()]*1000:.0f} nm")
```

## EnergyBalance

`compass.analysis.energy_balance.EnergyBalance`는 에너지 보존(energy conservation)을 검증합니다.

### `check`

```python
@staticmethod
def check(
    result: SimulationResult,
    tolerance: float = 0.01,
) -> dict:
```

모든 파장에서 $R + T + A \approx 1$을 검사합니다.

**반환값:** 다음 키를 포함하는 딕셔너리:
- `valid` (bool) -- 최대 오차가 허용 오차 미만이면 `True`
- `max_error` (float) -- 최대 $|R + T + A - 1|$
- `mean_error` (float) -- 전체 파장에 대한 평균 오차
- `per_wavelength` (np.ndarray) -- 각 파장별 오차
- `R`, `T`, `A` (np.ndarray) -- 개별 스펙트럼

**예제:**

```python
from compass.analysis.energy_balance import EnergyBalance

check = EnergyBalance.check(result, tolerance=0.02)
if not check["valid"]:
    print(f"Energy violation: max error = {check['max_error']:.4f}")
```

## SolverComparison

`compass.analysis.solver_comparison.SolverComparison`은 여러 솔버의 결과를 비교합니다.

### 생성자

```python
class SolverComparison:
    def __init__(
        self,
        results: List[SimulationResult],
        labels: List[str],
        reference_idx: int = 0,
    ):
```

**매개변수:**
- `results` -- 서로 다른 솔버의 `SimulationResult` 객체 리스트.
- `labels` -- 각 솔버의 표시 라벨.
- `reference_idx` -- 기준 결과의 인덱스 (기본값: 첫 번째).

### `qe_difference`

```python
def qe_difference(self) -> Dict[str, np.ndarray]:
```

각 픽셀에 대해 기준 대비 절대 QE 차이를 계산합니다. `"grcwa_vs_torcwa_R_0_0"`과 같은 키를 가진 딕셔너리를 반환합니다.

### `qe_relative_error`

```python
def qe_relative_error(self) -> Dict[str, np.ndarray]:
```

기준 대비 상대 QE 오차(%)를 계산합니다:

$$\text{Error}_\% = 100 \times \frac{|\text{QE}_\text{solver} - \text{QE}_\text{ref}|}{|\text{QE}_\text{ref}|}$$

### `runtime_comparison`

```python
def runtime_comparison(self) -> Dict[str, float]:
```

솔버 라벨을 실행 시간(초)에 매핑하는 딕셔너리를 반환합니다 (`result.metadata["runtime_seconds"]`에서 가져옴).

### `summary`

```python
def summary(self) -> dict:
```

포괄적인 비교 요약을 반환합니다:

```python
{
    "max_qe_diff": {"grcwa_vs_torcwa_R_0_0": 0.012, ...},
    "mean_qe_diff": {"grcwa_vs_torcwa_R_0_0": 0.004, ...},
    "max_qe_relative_error_pct": {"grcwa_vs_torcwa_R_0_0": 2.1, ...},
    "runtimes_seconds": {"torcwa": 0.3, "grcwa": 0.5},
}
```

**예제:**

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
