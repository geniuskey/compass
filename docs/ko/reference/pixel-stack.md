# PixelStack

`compass.geometry.pixel_stack.PixelStack`은 BSI 픽셀 구조(BSI pixel structure)의 솔버 비의존적(solver-agnostic) 표현입니다. YAML 설정으로부터 전체 3D 픽셀을 구성하며, RCWA(레이어 슬라이스)와 FDTD(복셀 그리드) 두 가지 형식으로 출력을 제공합니다.

## 생성자

```python
class PixelStack:
    def __init__(self, config: dict, material_db: Optional[MaterialDB] = None):
```

**매개변수:**
- `config` -- 설정 딕셔너리. `"pixel"` 키를 포함하거나, 픽셀 설정 자체여야 합니다.
- `material_db` -- `MaterialDB` 인스턴스. `None`인 경우 자동으로 생성됩니다.

**예제:**

```python
from compass.materials.database import MaterialDB
from compass.geometry.pixel_stack import PixelStack

db = MaterialDB()
config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "silicon": {"thickness": 3.0, "material": "silicon"},
            "color_filter": {"thickness": 0.6},
        },
    }
}
stack = PixelStack(config, db)
```

## 속성(Properties)

### `domain_size -> Tuple[float, float]`

시뮬레이션 도메인 크기 `(Lx, Ly)`를 um 단위로 반환합니다. `(pitch * unit_cell[1], pitch * unit_cell[0])`과 동일합니다.

### `total_height -> float`

실리콘 하단부터 에어 레이어 상단까지의 전체 스택 높이를 um 단위로 반환합니다.

### `z_range -> Tuple[float, float]`

스택의 `(z_min, z_max)`. 실리콘 하단에서 `z_min = 0.0`입니다.

## 어트리뷰트

| 어트리뷰트 | 타입 | 설명 |
|------------|------|------|
| `pitch` | `float` | 픽셀 피치 (um) |
| `unit_cell` | `Tuple[int, int]` | 유닛 셀 내 (행, 열) |
| `material_db` | `MaterialDB` | 재료 데이터베이스 인스턴스 |
| `layers` | `List[Layer]` | 정렬된 레이어 목록 (하단에서 상단 순) |
| `microlenses` | `List[MicrolensSpec]` | 각 픽셀의 마이크로렌즈 사양 |
| `photodiodes` | `List[PhotodiodeSpec]` | 각 픽셀의 포토다이오드 사양 |
| `bayer_map` | `List[List[str]]` | 컬러 할당 매트릭스 |

## 메서드

### `get_layer_slices`

```python
def get_layer_slices(
    self,
    wavelength: float,
    nx: int = 128,
    ny: int = 128,
    n_lens_slices: int = 30,
) -> List[LayerSlice]:
```

RCWA 솔버를 위한 z축 방향 레이어 분해(layer decomposition)를 생성합니다. 각 `LayerSlice`는 `(ny, nx)` 형상의 2D 복소 유전율 그리드(permittivity grid) `eps_grid`를 포함합니다.

**매개변수:**
- `wavelength` -- 유전율 계산을 위한 파장 (um 단위).
- `nx`, `ny` -- x, y 방향 공간 그리드 해상도.
- `n_lens_slices` -- 곡면 마이크로렌즈 근사를 위한 계단식 슬라이스 수.

**반환값:** 하단(z_min)에서 상단(z_max)까지 정렬된 `LayerSlice` 객체 리스트.

**동작 원리:**
- 균일 레이어(에어, 평탄화층, BARL)는 일정한 유전율을 가진 단일 슬라이스를 생성합니다.
- 컬러 필터 레이어는 베이어 패턴(Bayer pattern)으로 배열된 재료와 선택적 금속 그리드를 포함하는 패턴 슬라이스를 생성합니다.
- 실리콘 레이어는 선택적 DTI 트렌치(trench)를 포함합니다.
- 마이크로렌즈 레이어는 `n_lens_slices`개의 계단식 슬라이스로 이산화됩니다.

### `get_permittivity_grid`

```python
def get_permittivity_grid(
    self,
    wavelength: float,
    nx: int = 64,
    ny: int = 64,
    nz: int = 128,
) -> np.ndarray:
```

FDTD 솔버를 위한 3D 복소 유전율 배열을 생성합니다.

**매개변수:**
- `wavelength` -- 파장 (um 단위).
- `nx`, `ny`, `nz` -- 3D 그리드 해상도.

**반환값:** `(ny, nx, nz)` 형상의 복소 배열.

### `get_photodiode_mask`

```python
def get_photodiode_mask(
    self,
    nx: int = 64,
    ny: int = 64,
    nz: int = 128,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
```

포토다이오드(photodiode) 영역에 대한 3D 부울 마스크를 생성합니다.

**반환값:** `(full_mask, per_pixel_masks)` 튜플. `per_pixel_masks`는 픽셀 이름을 개별 3D 마스크에 매핑합니다.

## 레이어 구축 순서

`PixelStack`은 하단에서 상단으로 레이어를 구축하며, 실리콘 하단에서 `z=0`입니다:

1. **실리콘(Silicon)** (z=0 ~ z=thickness_si)
2. **BARL 레이어** (얇은 유전체 스택)
3. **컬러 필터(Color filter)** (베이어 패턴 + 금속 그리드로 패터닝)
4. **평탄화층(Planarization)** (균일 유전체)
5. **마이크로렌즈(Microlens)** (곡면, 계단식으로 근사)
6. **에어(Air)** (최상위 레이어)

## 데이터 타입

### Layer

```python
@dataclass
class Layer:
    name: str               # "silicon", "color_filter", "microlens" 등
    z_start: float          # 하단 z 좌표 (um)
    z_end: float            # 상단 z 좌표 (um)
    thickness: float        # 레이어 두께 (um)
    base_material: str      # 재료 이름
    is_patterned: bool      # 레이어에 수평 패턴이 있는지 여부
```

### LayerSlice

```python
@dataclass
class LayerSlice:
    z_start: float          # 하단 z 좌표
    z_end: float            # 상단 z 좌표
    thickness: float        # 슬라이스 두께
    eps_grid: np.ndarray    # 2D 복소 유전율 (ny, nx)
    name: str               # 슬라이스 식별자
    material: str           # 기본 재료 이름
```

### MicrolensSpec

```python
@dataclass
class MicrolensSpec:
    height: float           # 렌즈 높이 (um)
    radius_x: float         # x 방향 반축 (um)
    radius_y: float         # y 방향 반축 (um)
    material: str           # 렌즈 재료 이름
    profile_type: str       # "superellipse" 또는 "spherical"
    n_param: float          # 사각도 매개변수
    alpha_param: float      # 곡률 매개변수
    shift_x: float          # CRA x 오프셋 (um)
    shift_y: float          # CRA y 오프셋 (um)
```

### PhotodiodeSpec

```python
@dataclass
class PhotodiodeSpec:
    position: Tuple[float, float, float]  # (x, y, z) 오프셋 (um)
    size: Tuple[float, float, float]      # (dx, dy, dz) (um)
    pixel_index: Tuple[int, int]          # 유닛 셀 내 (행, 열)
    color: str                             # "R", "G", 또는 "B"
```
