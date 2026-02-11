# 빠른 시작

이 5분 튜토리얼은 COMPASS의 핵심 워크플로우를 안내합니다: 픽셀 정의, 지오메트리(Geometry) 구축, 시각화, 솔버(Solver) 설정.

## 1. 설정 생성

COMPASS는 Pydantic으로 검증된 설정 객체를 사용합니다. Python 딕셔너리에서 생성하거나 YAML에서 로드할 수 있습니다.

```python
from compass.core.config_schema import CompassConfig

config = CompassConfig(**{
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True,
                "height": 0.6,
                "radius_x": 0.48,
                "radius_y": 0.48,
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "pattern": "bayer_rggb",
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "silicon": {
                "thickness": 3.0,
                "material": "silicon",
                "dti": {"enabled": True, "width": 0.1, "depth": 3.0},
            },
        },
        "bayer_map": [["R", "G"], ["G", "B"]],
    },
    "solver": {"name": "torcwa", "type": "rcwa"},
    "source": {
        "type": "planewave",
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
})

print(f"Pixel pitch: {config.pixel.pitch} um")
print(f"Unit cell: {config.pixel.unit_cell}")
print(f"Solver: {config.solver.name}")
```

레이어 파라미터를 조정하여 인터랙티브하게 픽셀 스택을 구성해 보십시오:

<PixelStackBuilder />

아래 슬라이더를 사용하여 다양한 파장 값을 실험해 보십시오:

<WavelengthSlider />

## 2. 재료 데이터베이스(Material Database) 초기화

`MaterialDB`는 내장 재료를 자동으로 로드하며 사용자 정의 CSV 파일을 지원합니다.

```python
from compass.materials.database import MaterialDB

mat_db = MaterialDB()

# Query silicon at 550 nm
n, k = mat_db.get_nk("silicon", 0.55)
print(f"Silicon at 550nm: n={n:.3f}, k={k:.4f}")

# List all available materials
print(mat_db.list_materials())
# ['air', 'cf_blue', 'cf_green', 'cf_red', 'hfo2', 'polymer_n1p56',
#  'si3n4', 'silicon', 'sio2', 'tio2', 'tungsten']
```

## 3. 픽셀 스택(Pixel Stack) 구축

`GeometryBuilder`는 설정을 솔버 비의존적인 `PixelStack`으로 변환합니다.

```python
from compass.geometry.builder import GeometryBuilder

builder = GeometryBuilder(config.pixel, mat_db)
pixel_stack = builder.build()

print(f"Stack z-range: [{pixel_stack.z_min:.3f}, {pixel_stack.z_max:.3f}] um")
print(f"Total thickness: {pixel_stack.total_thickness:.3f} um")
print(f"Domain size: {pixel_stack.domain_size} um")

# Inspect layers
for layer in pixel_stack.layers:
    print(f"  {layer.name}: z=[{layer.z_start:.3f}, {layer.z_end:.3f}], "
          f"material={layer.base_material}")
```

## 4. 레이어 슬라이스(Layer Slice) 생성

RCWA 솔버는 `LayerSlice` 객체, 즉 각 z 높이에서의 2D 유전율(Permittivity) 그리드를 사용합니다.

```python
slices = pixel_stack.get_layer_slices(wavelength=0.55, nx=128, ny=128)

for s in slices:
    print(f"  {s.name}: z=[{s.z_start:.3f}, {s.z_end:.3f}], "
          f"eps shape={s.eps_grid.shape}")
```

## 5. 구조 시각화

수직 단면(Cross-section)을 플롯하여 픽셀 지오메트리를 확인합니다.

```python
from compass.visualization.structure_plot_2d import plot_pixel_cross_section

ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xz",
    position=0.5,  # y = 0.5 um (center of first row)
    wavelength=0.55,
)
```

이 함수는 실리콘(하단)에서 공기(상단)까지의 모든 레이어를 색상으로 구분하고 레이어 주석이 포함된 matplotlib 그림을 생성합니다.

특정 z 높이에서의 XY 슬라이스:

```python
ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xy",
    position=4.5,  # z position in the color filter region
    wavelength=0.55,
)
```

## 6. 솔버 설정

솔버 인스턴스를 생성하고 시뮬레이션(Simulation)을 실행합니다(솔버 패키지 설치 필요):

```python
from compass.solvers.base import SolverFactory

solver = SolverFactory.create("torcwa", config.solver)
solver.setup_geometry(pixel_stack)
solver.setup_source({
    "wavelength": 0.55,
    "theta": 0.0,
    "phi": 0.0,
    "polarization": "unpolarized",
})
result = solver.run()

print(f"R = {result.reflection}")
print(f"T = {result.transmission}")
print(f"A = {result.absorption}")
```

## 다음 단계

- [첫 번째 시뮬레이션](./first-simulation.md) -- 파장 스위프(Wavelength Sweep) 및 QE 플롯을 포함한 전체 워크스루
- [픽셀 스택 설정](./pixel-stack-config.md) -- 상세 YAML 설정 참조
- [솔버 선택](./choosing-solver.md) -- 어떤 솔버를 언제 사용할 것인지
