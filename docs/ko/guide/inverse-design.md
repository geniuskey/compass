# 역설계 (Inverse Design)

COMPASS는 픽셀 구조의 역설계를 위한 내장 최적화 모듈을 제공합니다. 기본 픽셀 구성에서 시작하여, 최적화기가 자동으로 기하학적 매개변수(마이크로렌즈 형상, BARL 두께, 컬러 필터 두께 등)를 탐색하여 양자 효율을 최대화하거나, 크로스토크를 최소화하거나, 기타 사용자 정의 목표를 달성합니다.

## 개요

최적화 워크플로는 세 부분으로 구성됩니다:

1. **매개변수 공간** -- 최적화할 물리적 치수를 선택하고 범위를 설정합니다.
2. **목적 함수** -- "더 나은 것"의 의미를 정의합니다: 높은 QE, 낮은 크로스토크, 또는 가중 조합.
3. **최적화기** -- 알고리즘(Nelder-Mead, L-BFGS-B, 차등 진화 등)을 선택하여 COMPASS 시뮬레이션을 반복적으로 평가하여 최적값을 찾습니다.

내부적으로, 각 최적화 반복은 업데이트된 매개변수로 전체 COMPASS 시뮬레이션을 실행하고, 목적 함수를 평가하고, 결과를 최적화 이력에 기록합니다.

## 빠른 예제

```python
import copy
from compass.optimization import (
    PixelOptimizer,
    ParameterSpace,
    MicrolensHeight,
    MicrolensSquareness,
    MaximizeQE,
    MinimizeCrosstalk,
    CompositeObjective,
)

# 기본 설정 딕셔너리에서 시작
base_config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "microlens": {"height": 0.6, "profile": {"n": 2.5}},
            "color_filter": {"thickness": 0.6},
        },
    },
    "solver": {"name": "meent"},
    "source": {"wavelength": {"mode": "sweep", "sweep": {"start": 0.4, "stop": 0.7, "step": 0.02}}},
    "compute": {"backend": "cpu"},
}

# 1. 매개변수 공간 정의
params = ParameterSpace([
    MicrolensHeight(base_config, min_val=0.2, max_val=1.2),
    MicrolensSquareness(base_config, min_val=1.5, max_val=5.0),
])

# 2. 목적 함수 정의
objective = CompositeObjective([
    (1.0, MaximizeQE(wavelength_range=(0.4, 0.7))),
    (0.5, MinimizeCrosstalk()),
])

# 3. 최적화 실행
optimizer = PixelOptimizer(
    base_config=base_config,
    parameter_space=params,
    objective=objective,
    solver_name="meent",
    method="nelder-mead",
    max_iterations=50,
)
result = optimizer.optimize()

print(f"최적 QE 목적값: {result.best_objective:.4f}")
print(f"최적 매개변수: {result.best_params}")
print(f"평가 횟수: {result.n_evaluations}")
print(f"수렴 여부: {result.converged}")

# 최적화 이력 저장
result.history.save("outputs/optimization_history.json")
```

## YAML 설정 사용

Hydra YAML 설정으로도 최적화를 실행할 수 있습니다:

```yaml
# configs/experiment/optimize_microlens.yaml
optimization:
  solver: meent
  method: nelder-mead
  max_iterations: 50
  parameters:
    - type: microlens_height
      bounds: [0.2, 1.2]
    - type: microlens_squareness
      bounds: [1.5, 5.0]
  objective:
    type: composite
    components:
      - weight: 1.0
        type: maximize_qe
        wavelength_range: [0.4, 0.7]
      - weight: 0.5
        type: minimize_crosstalk
```

## 사용 가능한 매개변수

| 매개변수 | 클래스 | 크기 | 설명 |
|----------|--------|------|------|
| `microlens_height` | `MicrolensHeight` | 1 | 마이크로렌즈 새그 높이 (um) |
| `microlens_squareness` | `MicrolensSquareness` | 1 | 초타원 지수 n |
| `microlens_radii` | `MicrolensRadii` | 2 | 렌즈 반축 (radius_x, radius_y) |
| `barl_thicknesses` | `BARLThicknesses` | N | BARL 레이어당 하나의 값 |
| `color_filter_thickness` | `ColorFilterThickness` | 1 | 컬러 필터 레이어 두께 (um) |

각 매개변수 클래스는 경계 제약을 위한 `min_val` 및 `max_val` 인수를 받습니다.

## 목적 함수

모든 목적 함수는 **최소화**할 스칼라를 반환합니다. 최대화 목적(QE, 피크 QE)은 내부적으로 부호가 반전됩니다.

| 목적 함수 | 설명 |
|-----------|------|
| `MaximizeQE` | 픽셀과 파장에 걸친 평균 QE의 음수. `target_pixels`, `wavelength_range`, `weights` 지원. |
| `MinimizeCrosstalk` | 평균 비대각 QE 비율 (크로스토크). `target_wavelength_range` 지원. |
| `MaximizePeakQE` | 특정 색상 채널(`"R"`, `"G"`, `"B"`)의 피크 QE 음수. |
| `EnergyBalanceRegularizer` | R+T+A가 허용 오차를 넘어 1에서 벗어날 때 이차 페널티. |
| `CompositeObjective` | 여러 목적 함수의 가중 합. |

### CompositeObjective 예제

```python
objective = CompositeObjective([
    (1.0, MaximizeQE()),
    (0.5, MinimizeCrosstalk()),
    (0.1, EnergyBalanceRegularizer(tolerance=0.02, penalty_weight=10.0)),
])
```

<EnergyBalanceDiagram />

## 최적화 방법

| 방법 | 유형 | 적합한 용도 |
|------|------|------------|
| `nelder-mead` | 무경사 심플렉스 | 견고한 기본값, 1--5 매개변수 |
| `powell` | 무경사 방향 탐색 | 매끄러운 지형, 경계 있음 |
| `l-bfgs-b` | 경계 있는 준뉴턴법 | 매끄럽고 미분 가능한 솔버 |
| `differential-evolution` | 전역 확률적 | 다중 모달 지형, 지역 최소 회피 |

## 팁

- **Nelder-Mead로 시작하세요** - 약 5개 이하의 매개변수에 적합합니다. 경사를 필요로 하지 않으며 견고합니다.
- **`differential-evolution`을 사용하세요** - 여러 지역 최소값이 의심되거나 5개 이상의 매개변수가 있을 때.
- **`EnergyBalanceRegularizer`를 추가하세요** - 비물리적 구성에서 최적화기를 멀리하는 소프트 제약으로 사용합니다.
- **이력을 저장하세요** - 각 실행 후 `result.history.save(path)`로 저장하여 나중에 수렴을 시각화할 수 있습니다.
- **최적화 중에 거친 파장 스윕을 사용하세요** (예: step=0.02 um) - 각 평가 속도를 높이고, 마지막에 세밀한 스윕으로 검증합니다.
- **솔버로 `meent`를 선택하세요** - 빠르고 미분 가능한 RCWA 백엔드입니다.
- 각 평가는 전체 COMPASS 시뮬레이션을 실행하므로, 벽시계 시간은 평가 횟수에 비례합니다.
