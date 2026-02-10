---
title: FDTD 솔버 실행
description: COMPASS에서 flaport FDTD 솔버를 설정하고 실행하는 방법. 격자 간격, 실행 시간, PML 설정, 광대역 대 단일 파장 운용을 포함합니다.
---

# FDTD 솔버(Solver) 실행

COMPASS는 flaport fdtd 백엔드를 통해 FDTD(유한차분 시간영역법, Finite-Difference Time-Domain) 시뮬레이션(Simulation)을 지원합니다. FDTD는 RCWA와 상호 보완적입니다. 단일 실행으로 광대역 결과를 제공하고, 임의의 지오메트리를 처리하며, 전체 시간 영역 장(Field) 진화를 제공하지만, 더 느리고 메모리 집약적입니다.

## FDTD 사용 시점

다음이 필요한 경우 FDTD를 사용합니다:

- 단일 시뮬레이션 실행에서의 광대역 스펙트럼 응답
- 비주기적 또는 유한 크기 구조
- 픽셀 스택(Pixel Stack) 내부의 근접장(Near-field) 분포 세부 정보
- RCWA 결과에 대한 교차 검증(Cross-Validation)
- 장 전파의 시간 영역 분석

표준 주기적 베이어 QE 스위프의 경우, RCWA(torcwa)가 더 빠릅니다. 상세한 비교는 [솔버 선택](./choosing-solver.md)을 참조하십시오.

## FDTD 솔버 생성

```python
from compass.solvers.base import SolverFactory

solver_config = {
    "name": "fdtd_flaport",
    "type": "fdtd",
    "params": {
        "grid_spacing": 0.02,   # 20 nm grid cells
        "runtime": 200,          # 200 femtoseconds
        "pml_layers": 15,        # 15-cell PML absorber
        "dtype": "float64",
    },
}

solver = SolverFactory.create("fdtd_flaport", solver_config, device="cuda")
```

## 격자 간격(Grid Spacing)

격자 간격(`grid_spacing`, 마이크로미터 단위)은 가장 중요한 FDTD 파라미터입니다. 가장 작은 기하학적 특징과 가장 높은 굴절률 재료 내부의 가장 짧은 파장을 모두 분해해야 합니다.

가시광 범위 실리콘 시뮬레이션의 경험 법칙:

$$\Delta x \leq \frac{\lambda_{\min}}{n_{\max} \times \text{PPW}}$$

여기서 PPW(파장당 포인트 수, Points Per Wavelength)는 최소 10이어야 합니다. 400 nm에서 실리콘($n \approx 4.0$)의 경우:

$$\Delta x \leq \frac{0.400}{4.0 \times 10} = 0.010 \text{ um (10 nm)}$$

실제로 20 nm이면 대부분의 가시광 범위 시뮬레이션에 적합하며, 정확도와 리소스 사용 간의 좋은 트레이드오프를 제공합니다.

### 격자 간격 권장사항

| 격자 간격  | 400nm/Si에서 PPW | 메모리 (2x2, 1um) | 실행시간  | 정확도    |
|-------------|-----------------|---------------------|---------|-----------|
| 40 nm       | 2.5             | ~250 MB             | ~15 s   | 낮음      |
| 20 nm       | 5.0             | ~2 GB               | ~45 s   | 적절      |
| 10 nm       | 10.0            | ~8 GB               | ~180 s  | 높음      |
| 5 nm        | 20.0            | ~64 GB              | ~900 s  | 매우 높음  |

메모리는 3D에서 $O(1/\Delta x^3)$으로 스케일링되므로, 격자 간격을 절반으로 줄이면 메모리가 8배 증가합니다.

## 실행 시간(Runtime) 설정

`runtime` 파라미터(펨토초 단위)는 FDTD 시뮬레이션의 실행 시간을 제어합니다. 시뮬레이션은 다음을 위해 충분히 오래 실행되어야 합니다:

1. 소스 펄스가 전체 구조를 통과할 때까지
2. 모든 내부 반사가 충분히 감쇠할 때까지
3. 모니터의 장이 정상 상태에 도달할 때까지

```yaml
solver:
  name: "fdtd_flaport"
  type: "fdtd"
  params:
    grid_spacing: 0.02
    runtime: 200          # femtoseconds
    pml_layers: 15
    dtype: "float64"
```

200 fs의 실행 시간은 3 um 실리콘을 가진 대부분의 BSI 픽셀 구조에 충분합니다. 더 두꺼운 실리콘(>4 um)이나 높은 Q 공진을 가진 구조의 경우 300-500 fs로 증가하십시오.

서로 다른 실행 시간에서 결과를 비교하여 수렴을 검증할 수 있습니다:

```python
import copy
from compass.runners.single_run import SingleRunner

runtimes = [100, 150, 200, 300]

for rt in runtimes:
    cfg = copy.deepcopy(config)
    cfg["solver"]["params"]["runtime"] = rt
    result = SingleRunner.run(cfg)
    avg_qe = sum(qe.mean() for qe in result.qe_per_pixel.values()) / len(result.qe_per_pixel)
    print(f"Runtime {rt} fs: avg QE = {avg_qe:.4f}")
```

## PML 설정

PML(완전 정합층, Perfectly Matched Layer)은 시뮬레이션 경계에서 나가는 복사를 흡수합니다. `pml_layers` 파라미터는 격자 셀 수로 PML 영역의 두께를 설정합니다.

```python
solver_config = {
    "name": "fdtd_flaport",
    "type": "fdtd",
    "params": {
        "grid_spacing": 0.02,
        "runtime": 200,
        "pml_layers": 15,    # 15 cells = 0.3 um at 20 nm grid
        "dtype": "float64",
    },
}
```

가이드라인:

- **15셀**이 기본값이며 대부분의 시뮬레이션에서 잘 작동합니다
- 경계에서 허위 반사가 관측되면 **20-25셀**로 증가하십시오
- 주기적 구조의 경우, PML은 z 방향(상/하)에만 적용됩니다. 측면 경계는 주기적 경계 조건을 사용합니다
- PML은 시뮬레이션 도메인 크기에 추가되므로, 두꺼운 PML은 메모리 사용량을 증가시킵니다

## 광대역 vs 단일 파장

### 광대역 운용

FDTD는 소스가 시간 영역 펄스이기 때문에 단일 시뮬레이션에서 자연스럽게 광대역 결과를 생성합니다. 시뮬레이션 후, 기록된 장의 푸리에 변환(Fourier Transform)으로 모든 주파수에서의 스펙트럼 응답을 동시에 얻을 수 있습니다.

```python
config = {
    "pixel": { ... },
    "solver": {
        "name": "fdtd_flaport",
        "type": "fdtd",
        "params": {"grid_spacing": 0.02, "runtime": 200, "pml_layers": 15},
    },
    "source": {
        "wavelength": {
            "mode": "sweep",
            "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01},
        },
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
# All 31 wavelength points come from a single FDTD run
print(f"Wavelengths: {len(result.wavelengths)} points")
```

이것이 밀집 스펙트럼 샘플링에서 FDTD의 핵심 장점입니다. 100점 파장 스위프는 10점 스위프와 거의 동일한 시간이 소요되지만, RCWA에서는 각 파장이 별도의 계산입니다.

### 단일 파장 운용

단일 파장의 경우, FDTD는 CW(연속파, Continuous Wave) 소스를 사용합니다. 단일 점에서는 RCWA보다 느리지만 전체 시간 영역 장 진화를 제공합니다:

```python
config["source"] = {
    "wavelength": {"mode": "single", "value": 0.55},
    "polarization": "unpolarized",
}

result = SingleRunner.run(config)
```

## 격자 간격 수렴 테스트

항상 여러 격자 간격에서 실행하여 수렴(Convergence)을 검증하십시오:

```python
import copy
import numpy as np
from compass.runners.single_run import SingleRunner

spacings = [0.04, 0.02, 0.01]
qe_results = []

for dx in spacings:
    cfg = copy.deepcopy(config)
    cfg["solver"]["params"]["grid_spacing"] = dx
    result = SingleRunner.run(cfg)
    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    qe_results.append(green_qe)
    print(f"Grid {dx*1000:.0f} nm: Green QE = {green_qe:.4f}")

# Check convergence
for i in range(1, len(qe_results)):
    delta = abs(qe_results[i] - qe_results[i-1])
    print(f"  {spacings[i]*1000:.0f} nm vs {spacings[i-1]*1000:.0f} nm: "
          f"delta = {delta:.5f}")
```

## GPU 및 메모리 고려사항

flaport FDTD 솔버는 PyTorch 텐서를 사용하며 CUDA GPU 가속을 지원합니다. 주요 사항:

- **GPU 메모리**: 20 nm 격자 간격의 2x2 단위 셀은 약 2 GB의 VRAM이 필요합니다. 10 nm에서는 약 8 GB로 증가합니다.
- **CPU 폴백**: GPU 메모리가 부족하면 `backend: "cpu"`로 설정하십시오. CPU 실행은 5-10배 느리지만 시스템 RAM 이외의 메모리 한계가 없습니다.
- **dtype**: 정확도를 위해 `float64`를 사용하십시오. `float32`는 메모리를 절약하지만 긴 시간 트레이스의 푸리에 변환에서 아티팩트가 발생할 수 있습니다.

```yaml
compute:
  backend: "cuda"    # or "cpu" for memory-constrained systems
  gpu_id: 0
```

## 에너지 균형 검증(Energy Balance Validation)

모든 FDTD 실행 후 에너지 보존을 확인하십시오:

```python
from compass.analysis.energy_balance import EnergyBalance

check = EnergyBalance.check(result, tolerance=0.02)
print(f"Valid: {check['valid']}, max error: {check['max_error']:.4f}")
```

FDTD 에너지 균형 오차는 공간 이산화로 인해 일반적으로 RCWA보다 큽니다(1-3% vs <1%). 오차가 3%를 초과하면 격자 간격을 줄이십시오.

## 다음 단계

- [RCWA 실행](./running-rcwa.md) -- RCWA 솔버 설정
- [교차 검증](./cross-validation.md) -- FDTD 결과를 RCWA와 비교
- [솔버 선택](./choosing-solver.md) -- FDTD vs RCWA 사용 시점
