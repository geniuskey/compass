---
title: RCWA 솔버 실행
description: COMPASS에서 torcwa, grcwa, meent RCWA 솔버를 설정하고 실행하는 방법. 솔버별 설정, 안정성 제어, 수렴 테스트를 포함합니다.
---

# RCWA 솔버(Solver) 실행

COMPASS는 세 가지 RCWA 솔버 백엔드(torcwa, grcwa, meent)를 제공하며, 모두 동일한 `SolverFactory` 인터페이스를 통해 접근할 수 있습니다. 이 가이드에서는 솔버별 설정, 안정성 설정, 수렴(Convergence) 검증을 다룹니다.

## RCWA 솔버 생성

모든 RCWA 솔버는 `SolverFactory.create()`를 통해 생성됩니다:

```python
from compass.solvers.base import SolverFactory

# torcwa (recommended default)
solver = SolverFactory.create("torcwa", solver_config, device="cuda")

# grcwa (NumPy/JAX backend, good for cross-validation)
solver = SolverFactory.create("grcwa", solver_config, device="cpu")

# meent (multi-backend: numpy, jax, torch)
solver = SolverFactory.create("meent", solver_config, device="cpu")
```

## 솔버별 설정

### torcwa

torcwa는 주력 RCWA 솔버입니다. GPU 가속 행렬 연산에 PyTorch를 사용합니다.

```yaml
solver:
  name: "torcwa"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  stability:
    precision_strategy: "mixed"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
    eigenvalue_broadening: 1.0e-10
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
```

torcwa의 주요 고려사항:

- 항상 `allow_tf32: false`로 설정하십시오. TF32는 Ampere 이상 GPU에서 가수부 정밀도를 23비트에서 10비트로 줄여 S-행렬(S-matrix) 정확도를 저하시킵니다.
- `"mixed"` 정밀도 전략은 대부분의 계산을 float32로 실행하되 고유값 분해(Eigendecomposition)를 float64로 승격하여 CPU에서 실행합니다. 이것이 최적의 속도/안정성 트레이드오프입니다.
- Li의 역규칙(`fourier_factorization: "li_inverse"`)은 텅스텐 금속 격자와 같은 높은 대비 경계가 있는 구조에서 매우 중요합니다.

### grcwa

grcwa는 선택적 JAX 가속이 가능한 NumPy를 사용합니다. 기본적으로 float64를 사용하여 더 느리지만 수치적으로 더 안정적입니다.

```yaml
solver:
  name: "grcwa"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex128"
```

grcwa는 torcwa에 대한 교차 검증(Cross-Validation)에 가장 적합합니다. 동일한 RCWA 알고리즘의 다른 구현을 사용하므로, 두 솔버 간의 일치는 정확성을 확인해 줍니다.

### meent

meent는 런타임에 선택 가능한 세 가지 백엔드를 지원합니다:

```yaml
solver:
  name: "meent"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
    backend: "torch"   # "numpy" | "jax" | "torch"
```

COMPASS 어댑터가 단위 변환을 자동으로 처리합니다. meent는 내부적으로 나노미터를 사용하고 COMPASS는 마이크로미터를 사용합니다.

## 안정성 설정

### PrecisionManager

RCWA 솔버를 실행하기 전에 정밀도 설정을 구성합니다:

```python
from compass.solvers.rcwa.stability import PrecisionManager

PrecisionManager.configure(solver_config)
```

이 함수는 TF32를 비활성화하고 올바른 정밀도 컨텍스트를 설정합니다. `SingleRunner`는 이를 자동으로 호출하지만, 직접 솔버를 사용할 때는 수동으로 호출해야 합니다.

### 혼합 정밀도 고유값 분해

고유값 문제(Eigenvalue Problem)는 RCWA에서 수치적으로 가장 민감한 단계입니다. 혼합 정밀도를 사용하여 속도를 유지하면서 불안정성을 방지합니다:

```python
from compass.solvers.rcwa.stability import PrecisionManager
import numpy as np

# NumPy path (grcwa, meent-numpy)
matrix = np.random.randn(722, 722) + 1j * np.random.randn(722, 722)
eigenvalues, eigenvectors = PrecisionManager.mixed_precision_eigen(matrix)

# PyTorch path (torcwa, meent-torch)
eigenvalues, eigenvectors = PrecisionManager.mixed_precision_eigen_torch(matrix_tensor)
```

두 함수 모두 입력을 float64로 승격한 후 고유값 분해를 수행하고, 원래 정밀도로 다시 캐스팅합니다.

### 적응형 폴백(Adaptive Fallback)

일부 파장에서 실패할 수 있는 프로덕션 스위프의 경우, `AdaptivePrecisionRunner`를 사용합니다:

```python
from compass.solvers.rcwa.stability import AdaptivePrecisionRunner

runner = AdaptivePrecisionRunner(tolerance=0.02)
result = runner.run_with_fallback(solver, wavelength=0.45, config=solver_config)
```

폴백 체인은 다음과 같습니다: GPU float32 -> GPU float64 -> CPU float64. 세 가지 모두 실패하면 러너는 푸리에 차수를 줄일 것을 제안하는 `RuntimeError`를 발생시킵니다.

## 수렴 테스트

결과를 신뢰하기 전에 항상 수렴 여부를 검증하십시오.

### 푸리에 차수 스위프

푸리에 차수(Fourier Order)는 고조파의 수를 결정합니다. 차수 `[N, N]`의 경우 행렬 크기는 $(2N+1)^2$입니다. `N`을 스위프하고 QE가 안정화되는지 확인합니다:

```python
import numpy as np
from compass.solvers.base import SolverFactory

orders = range(5, 22, 2)
green_qe_values = []

for N in orders:
    solver_config["params"]["fourier_order"] = [N, N]
    solver = SolverFactory.create("torcwa", solver_config)
    solver.setup_geometry(pixel_stack)
    solver.setup_source({"wavelength": 0.55, "theta": 0.0,
                         "phi": 0.0, "polarization": "unpolarized"})
    result = solver.run()

    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    green_qe_values.append(float(green_qe))
    print(f"Order {N:2d}: Green QE = {green_qe:.4f}")

# Check convergence: relative change between successive orders
for i in range(1, len(green_qe_values)):
    delta = abs(green_qe_values[i] - green_qe_values[i-1])
    converged = "CONVERGED" if delta < 0.005 else ""
    print(f"  Order {list(orders)[i]}: delta = {delta:.5f} {converged}")
```

일반적 수렴: 대부분의 1 um 피치 픽셀에서 차수 [9, 9]이면 충분합니다. 미세한 금속 격자나 높은 대비 레이어가 있는 구조는 [13, 13] 이상이 필요할 수 있습니다.

### 실행시간 vs 정확도 트레이드오프

| 차수    | 행렬 크기 | 일반적 시간 (GPU) | 사용 사례            |
|---------|-------------|---------------------|---------------------|
| [5, 5]  | 121         | 0.1 s               | 빠른 스크리닝        |
| [9, 9]  | 361         | 0.3 s               | 표준 프로덕션        |
| [13,13] | 729         | 1.5 s               | 높은 정확도          |
| [17,17] | 1225        | 5.0 s               | 논문 수준            |

## 전체 시뮬레이션 실행

```python
from compass.runners.single_run import SingleRunner

config = {
    "pixel": { ... },  # pixel stack definition
    "solver": {
        "name": "torcwa",
        "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
print(f"Completed: {len(result.wavelengths)} wavelengths")
```

## 다음 단계

- [FDTD 실행](./running-fdtd.md) -- flaport FDTD 솔버 설정
- [교차 검증](./cross-validation.md) -- RCWA 솔버 간 비교
- [문제 해결](./troubleshooting.md) -- 안정성 및 수렴 문제
