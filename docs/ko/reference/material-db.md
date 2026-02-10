# MaterialDB

`compass.materials.database.MaterialDB`는 광학 재료 물성(optical material properties)의 중앙 레지스트리입니다. 시뮬레이션에 사용되는 모든 재료에 대해 파장 의존적 굴절률(refractive index) $(n, k)$ 및 복소 유전율(complex permittivity) $\varepsilon$을 제공합니다.

## 생성자

```python
class MaterialDB:
    def __init__(self, db_path: Optional[str] = None):
```

**매개변수:**
- `db_path` -- CSV 재료 파일이 포함된 디렉터리 경로. 기본값은 프로젝트 루트의 `materials/` 디렉터리입니다.

생성 시 `MaterialDB`는 자동으로 다음을 수행합니다:
1. 내장 상수 및 해석적(analytic) 재료를 등록합니다 (에어, 폴리머, 유전체)
2. 재료 디렉터리에서 테이블화된 CSV 데이터를 로드합니다 (실리콘, 텅스텐, 컬러 필터)
3. CSV 파일이 없는 경우 근사 내장 데이터로 대체합니다

## 조회 메서드

### `get_nk`

```python
def get_nk(self, name: str, wavelength: float) -> Tuple[float, float]:
```

주어진 파장(um 단위)에서 굴절률의 실수부와 허수부인 `(n, k)`를 반환합니다.

**예외 발생:** 재료가 등록되어 있지 않은 경우 `KeyError`가 발생합니다.

### `get_epsilon`

```python
def get_epsilon(self, name: str, wavelength: float) -> complex:
```

주어진 파장에서 복소 유전율 $\varepsilon = (n + ik)^2$을 반환합니다.

### `get_epsilon_spectrum`

```python
def get_epsilon_spectrum(self, name: str, wavelengths: np.ndarray) -> np.ndarray:
```

파장 배열에 대한 복소 유전율 값의 배열을 반환합니다.

### `list_materials`

```python
def list_materials(self) -> List[str]:
```

등록된 모든 재료 이름의 정렬된 목록을 반환합니다.

### `has_material`

```python
def has_material(self, name: str) -> bool:
```

해당 재료가 데이터베이스에 존재하면 `True`를 반환합니다.

## 등록 메서드

### `register_constant`

```python
def register_constant(self, name: str, n: float, k: float = 0.0) -> None:
```

파장에 무관한 고정 굴절률을 가진 재료를 등록합니다.

```python
db.register_constant("my_glass", n=1.52, k=0.0)
```

### `register_cauchy`

```python
def register_cauchy(self, name: str, A: float, B: float = 0.0, C: float = 0.0) -> None:
```

코시 분산 모델(Cauchy dispersion model)을 사용하는 재료를 등록합니다:

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$

여기서 $\lambda$의 단위는 um입니다. 소광 계수(extinction coefficient) $k$는 0입니다 (비흡수성).

```python
db.register_cauchy("polymer", A=1.56, B=0.004, C=0.0)
```

### `register_sellmeier`

```python
def register_sellmeier(self, name: str, B: List[float], C: List[float]) -> None:
```

셀마이어 분산 모델(Sellmeier dispersion model)을 사용하는 재료를 등록합니다:

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

CSV 파일에서 테이블화된 재료 데이터를 로드합니다. CSV 형식은 다음과 같습니다:

```
# wavelength(um), n, k
0.400, 5.381, 0.340
0.410, 5.253, 0.296
...
```

`#`으로 시작하는 줄은 주석으로 처리됩니다. 데이터는 파장 기준으로 자동 정렬됩니다. 보간(interpolation)은 3차 스플라인(cubic spline)을 사용하며 (4개 이상의 데이터 포인트 필요), 대체 방법으로 선형 보간(linear interpolation)을 사용합니다.

**매개변수:**
- `name` -- 등록할 재료 이름.
- `filepath` -- CSV 파일 경로.
- `interpolation` -- `"cubic_spline"` (기본값) 또는 `"linear"`.

## 내장 재료

| 이름 | 유형 | 모델 | 일반적인 n | 일반적인 k |
|------|------|------|-----------|-----------|
| `air` | 상수 | n=1.0, k=0.0 | 1.0 | 0.0 |
| `polymer_n1p56` | 코시 | A=1.56, B=0.004 | 1.56 | 0.0 |
| `sio2` | 셀마이어 | 3항 | 1.46 | 0.0 |
| `si3n4` | 셀마이어 | 2항 | 2.0 | 0.0 |
| `hfo2` | 코시 | A=1.90, B=0.02 | 1.90 | 0.0 |
| `tio2` | 코시 | A=2.27, B=0.05 | 2.27 | 0.0 |
| `silicon` | 테이블화 | Green 2008 | 3.5-5.6 | 0.003-3.0 |
| `tungsten` | 테이블화 | 근사값 | 3.4-3.7 | 2.7-3.4 |
| `cf_red` | 테이블화 | 로렌츠 | 1.55 | 0-0.15 |
| `cf_green` | 테이블화 | 로렌츠 | 1.55 | 0-0.12 |
| `cf_blue` | 테이블화 | 로렌츠 | 1.55 | 0-0.18 |

## MaterialData

각 재료의 내부 데이터 컨테이너입니다:

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

적절한 분산 모델(dispersion model)을 사용하여 `(n, k)`를 반환합니다.

### `MaterialData.get_epsilon(wavelength) -> complex`

$\varepsilon = (n + ik)^2$을 반환합니다.

## CSV 자동 로딩

생성 시 `MaterialDB`는 다음 파일명 매핑을 사용하여 재료 디렉터리에서 CSV 파일을 검색합니다:

| 재료 이름 | 예상 파일명 |
|-----------|------------|
| `silicon` | `silicon_green2008.csv`, `silicon_palik.csv` |
| `tungsten` | `tungsten.csv` |
| `cf_red` | `color_filter_red.csv` |
| `cf_green` | `color_filter_green.csv` |
| `cf_blue` | `color_filter_blue.csv` |

CSV 파일이 발견되면 자동으로 로드됩니다. 그렇지 않으면 내장 근사값이 사용됩니다.
