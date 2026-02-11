# SolverBase

`compass.solvers.base.SolverBase`는 COMPASS의 모든 전자기(EM) 솔버가 구현해야 하는 추상 기본 클래스(abstract base class)입니다. 러너(runner) 및 분석 모듈에서 사용하는 통일된 인터페이스를 정의합니다.

## 추상 인터페이스

```python
class SolverBase(ABC):
    def __init__(self, config: dict, device: str = "cpu"):
```

**매개변수:**
- `config` -- 솔버 설정 딕셔너리 (Hydra/YAML에서 제공).
- `device` -- 계산 장치: `"cpu"`, `"cuda"`, 또는 `"mps"`.

## 속성(Properties)

### `name -> str`

솔버 이름. `config["name"]` 또는 클래스 이름에서 파생됩니다.

### `solver_type -> str`

솔버 유형: `config["type"]`에서 가져온 `"rcwa"` 또는 `"fdtd"`.

## 추상 메서드

모든 솔버 백엔드는 다음 네 가지 메서드를 구현해야 합니다:

<SolverPipelineDiagram />

### `setup_geometry`

```python
@abstractmethod
def setup_geometry(self, pixel_stack: PixelStack) -> None:
```

솔버 비의존적인 `PixelStack`을 솔버의 내부 지오메트리 표현으로 변환합니다.

- RCWA 솔버는 `pixel_stack.get_layer_slices()`를 호출하여 2D 유전율 그리드를 가져옵니다.
- FDTD 솔버는 `pixel_stack.get_permittivity_grid()`를 호출하여 3D 복셀 그리드를 가져옵니다.

### `setup_source`

```python
@abstractmethod
def setup_source(self, source_config: dict) -> None:
```

여기 광원(excitation source)을 설정합니다 (파장, 각도, 편광).

### `run`

```python
@abstractmethod
def run(self) -> SimulationResult:
```

시뮬레이션을 실행하고 표준화된 `SimulationResult`를 반환합니다.

### `get_field_distribution`

```python
@abstractmethod
def get_field_distribution(
    self,
    component: str,   # "Ex", "Ey", "Ez", "|E|2"
    plane: str,        # "xy", "xz", "yz"
    position: float,   # 법선 축 방향 위치 (um)
) -> np.ndarray:
```

시뮬레이션에서 2D 전계 슬라이스(field slice)를 추출합니다.

## 구상 메서드

### `validate_energy_balance`

```python
def validate_energy_balance(
    self,
    result: SimulationResult,
    tolerance: float = 0.01,
) -> bool:
```

시뮬레이션 결과에 대해 $R + T + A \approx 1$을 검사합니다. 허용 오차(tolerance) 이내에서 에너지가 보존되면 `True`를 반환합니다. 위반 시 경고를 로깅합니다.

### `run_timed`

```python
def run_timed(self) -> SimulationResult:
```

`run()`에 타이밍 계측을 추가하여 래핑합니다. `result.metadata`에 다음 항목을 추가합니다:
- `runtime_seconds` -- 벽시계 시간(wall-clock time)
- `solver_name` -- 솔버 이름 문자열
- `solver_type` -- `"rcwa"` 또는 `"fdtd"`
- `device` -- 사용된 계산 장치

## SolverFactory

`compass.solvers.base.SolverFactory`는 레지스트리 패턴을 사용하여 이름으로 솔버 인스턴스를 생성합니다.

### `SolverFactory.create`

```python
@classmethod
def create(cls, name: str, config: dict, device: str = "cpu") -> SolverBase:
```

솔버 인스턴스를 생성합니다. 솔버 모듈이 아직 로드되지 않은 경우 지연 임포트(lazy import)를 수행합니다.

**매개변수:**
- `name` -- 솔버 이름: `"torcwa"`, `"grcwa"`, `"meent"`, `"fdtd_flaport"` 등.
- `config` -- 솔버 설정 딕셔너리.
- `device` -- 계산 장치.

**예외 발생:** 솔버 이름이 알 수 없고 해당 모듈을 임포트할 수 없는 경우 `ValueError`가 발생합니다.

### `SolverFactory.register`

```python
@classmethod
def register(cls, name: str, solver_class: type) -> None:
```

솔버 클래스를 등록합니다. 솔버 모듈이 임포트될 때 자동으로 호출됩니다.

### `SolverFactory.list_solvers`

```python
@classmethod
def list_solvers(cls) -> list:
```

등록된 모든 솔버의 이름을 반환합니다.

## 사용 가능한 솔버 백엔드

| 이름 | 모듈 | 유형 | 비고 |
|------|------|------|------|
| `torcwa` | `compass.solvers.rcwa.torcwa_solver` | RCWA | |
| `grcwa` | `compass.solvers.rcwa.grcwa_solver` | RCWA | |
| `meent` | `compass.solvers.rcwa.meent_solver` | RCWA | |
| `fmmax` | `compass.solvers.rcwa.fmmax_solver` | RCWA | |
| `fdtd_flaport` | `compass.solvers.fdtd.flaport_solver` | FDTD | |
| `fdtdx` | `compass.solvers.fdtd.fdtdx_solver` | FDTD | JAX-based 3D FDTD, multi-GPU, fully differentiable, MIT license |
| `tmm` | `compass.solvers.tmm.tmm_solver` | TMM | 1D 평면 스택 전용, RCWA 대비 ~1000배 빠름 |

<SolverComparisonChart />

## SimulationResult

모든 솔버에서 반환하는 표준화된 출력입니다:

```python
@dataclass
class SimulationResult:
    qe_per_pixel: Dict[str, np.ndarray]    # 픽셀 이름 -> QE 스펙트럼
    wavelengths: np.ndarray                 # 파장 배열 (um)
    fields: Optional[Dict[str, FieldData]]  # 파장별 전계 데이터
    poynting: Optional[Dict[str, np.ndarray]]
    reflection: Optional[np.ndarray]        # R(lambda)
    transmission: Optional[np.ndarray]      # T(lambda)
    absorption: Optional[np.ndarray]        # A(lambda)
    metadata: Dict                          # 타이밍, 솔버 정보
```

### FieldData

```python
@dataclass
class FieldData:
    Ex: Optional[np.ndarray]   # 전계 x 성분 (3D 복소수)
    Ey: Optional[np.ndarray]
    Ez: Optional[np.ndarray]
    x: Optional[np.ndarray]    # 좌표 배열
    y: Optional[np.ndarray]
    z: Optional[np.ndarray]

    @property
    def E_intensity(self) -> Optional[np.ndarray]:
        """Compute |E|^2 = |Ex|^2 + |Ey|^2 + |Ez|^2."""
```

## 커스텀 솔버 구현

새로운 솔버 백엔드를 추가하려면 다음과 같이 구현합니다:

```python
from compass.solvers.base import SolverBase, SolverFactory

class MySolver(SolverBase):
    def setup_geometry(self, pixel_stack):
        self._pixel_stack = pixel_stack
        # 내부 표현으로 변환...

    def setup_source(self, source_config):
        self._source_config = source_config

    def run(self):
        # 시뮬레이션 실행, SimulationResult 반환
        ...

    def get_field_distribution(self, component, plane, position):
        # 2D 전계 슬라이스 추출
        ...

# 등록
SolverFactory.register("my_solver", MySolver)
```
