# 광원(Sources)

COMPASS는 다양한 광학 시나리오를 시뮬레이션하기 위해 여러 조명 광원(illumination source) 유형을 지원합니다.

## PlanewaveSource

`compass.sources.planewave.PlanewaveSource`는 단일 평면파(plane wave) 또는 여러 파장에서의 평면파 세트를 모델링합니다.

### 생성자

```python
@dataclass
class PlanewaveSource:
    wavelengths: np.ndarray          # 파장 배열 (um)
    theta_deg: float = 0.0           # 극 입사각 (도)
    phi_deg: float = 0.0             # 방위각 (도)
    polarization: str = "unpolarized"  # "TE", "TM", 또는 "unpolarized"
```

### 팩토리 메서드

```python
@classmethod
def from_config(cls, source_config: dict) -> PlanewaveSource:
```

광원 설정 딕셔너리에서 `PlanewaveSource`를 생성합니다. 세 가지 파장 모드를 지원합니다:

**단일 파장:**
```yaml
source:
  wavelength:
    mode: single
    value: 0.55     # 550 nm
```

**파장 스윕(sweep):**
```yaml
source:
  wavelength:
    mode: sweep
    sweep:
      start: 0.38   # 380 nm
      stop: 0.78    # 780 nm
      step: 0.01    # 10 nm 간격
```

**파장 목록:**
```yaml
source:
  wavelength:
    mode: list
    values: [0.45, 0.55, 0.65]
```

### 속성(Properties)

| 속성 | 타입 | 설명 |
|------|------|------|
| `theta_rad` | `float` | 극각 (라디안) |
| `phi_rad` | `float` | 방위각 (라디안) |
| `n_wavelengths` | `int` | 파장 수 |
| `is_unpolarized` | `bool` | TE+TM 평균이 필요한지 여부 |

### 메서드

#### `get_polarization_runs`

```python
def get_polarization_runs(self) -> List[str]:
```

비편광(unpolarized)인 경우 `["TE", "TM"]`을 반환하고, 편광(polarized)인 경우 `["TE"]` 또는 `["TM"]`을 반환합니다.

#### `to_solver_params`

```python
def to_solver_params(self) -> dict:
```

솔버 호환 매개변수 딕셔너리로 변환합니다. 키: `wavelengths`, `theta_rad`, `phi_rad`, `polarization`, `polarization_runs`.

## ConeIllumination

`compass.sources.cone_illumination.ConeIllumination`은 주광선각(Chief Ray Angle, CRA)과 f-넘버(f-number)로 지정된 카메라 렌즈의 사출 동공 조명(exit-pupil illumination)을 모델링합니다.

### 생성자

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

**매개변수:**
- `cra_deg` -- 주광선각(Chief Ray Angle) (도 단위). 조명 콘의 중심 방향입니다.
- `f_number` -- 렌즈 f-넘버. 콘 반각을 결정합니다: $\theta_{half} = \arcsin(1 / 2F)$.
- `n_points` -- 콘 내부의 각도 샘플링 포인트 수.
- `sampling` -- 샘플링 방법: `"fibonacci"` (기본값) 또는 `"grid"`.
- `weighting` -- 각도 가중치: `"uniform"`, `"cosine"` (기본값), `"cos4"`, 또는 `"gaussian"`.

### 설정

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

### 메서드

#### `get_sampling_points`

```python
def get_sampling_points(self) -> List[Tuple[float, float, float]]:
```

조명 콘의 각도 샘플링을 나타내는 `(theta_deg, phi_deg, weight)` 튜플 리스트를 반환합니다. 가중치의 합은 1로 정규화됩니다.

### 샘플링 방법

**피보나치 나선(Fibonacci spiral)**: 황금비(golden ratio)를 사용하여 콘 캡 위에 준균일 샘플링을 생성합니다. 정규 그리드보다 적은 포인트로 우수한 각도 커버리지를 제공합니다.

**그리드(Grid)**: 극좌표에서의 정규 $N_\theta \times N_\phi$ 그리드. 각도 기준으로는 균일하지만 콘 중심 부근에서 과샘플링(over-sampling)됩니다.

### 가중치 옵션

| 가중치 | 수식 | 사용 사례 |
|--------|------|----------|
| `uniform` | $w = 1$ | 모든 각도에 동일한 가중치 |
| `cosine` | $w = \cos\theta$ | 램버시안 조명(Lambertian illumination) (기본값) |
| `cos4` | $w = \cos^4\theta$ | 보다 현실적인 카메라 렌즈 감쇠 |
| `gaussian` | $w = \exp(-\theta^2/2\sigma^2)$ | 아포다이즈된 조명(apodized illumination) |

### COMPASS에서의 사용

콘형 조명(cone illumination)은 평면파 시뮬레이션의 가중 합으로 구현됩니다:

$$\text{QE}_\text{cone} = \sum_i w_i \cdot \text{QE}(\theta_i, \phi_i)$$

러너(runner)는 샘플링 포인트마다 하나의 RCWA 또는 FDTD 시뮬레이션을 수행하고 결과를 평균합니다.

## RayFileReader

`compass.sources.ray_file_reader`는 광학 설계 도구(예: Zemax)의 광선 데이터를 가져옵니다.

### 설정

```yaml
source:
  ray_file:
    enabled: true
    path: "data/zemax_rays.json"
    format: zemax_json    # 또는 "csv"
```

이를 통해 렌즈 설계 시뮬레이션에서 현실적인 조명 조건을 직접 가져올 수 있습니다.

## 광원 설정 레퍼런스

```yaml
source:
  type: "planewave"              # "planewave" 또는 "cone_illumination"
  wavelength:
    mode: "single"               # "single", "sweep", 또는 "list"
    value: 0.55                  # 단일 모드용
    sweep:                       # 스윕 모드용
      start: 0.38
      stop: 0.78
      step: 0.01
    values: [0.45, 0.55, 0.65]  # 목록 모드용
  angle:
    theta_deg: 0.0               # 극각 (도)
    phi_deg: 0.0                 # 방위각 (도)
  polarization: "unpolarized"    # "TE", "TM", 또는 "unpolarized"
  cone:                          # cone_illumination 유형용
    cra_deg: 0.0
    f_number: 2.0
    pupil_shape: "circular"
    sampling:
      type: "fibonacci"
      n_points: 37
    weighting: "cosine"
  ray_file:                      # 외부 광선 가져오기
    enabled: false
    path: ""
    format: "zemax_json"
```
