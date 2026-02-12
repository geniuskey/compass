# 솔버 비교 & 선택 가이드

<SolverComparisonChart />

## 개요

COMPASS는 픽셀 시뮬레이션을 위해 여러 전자기 솔버를 지원합니다: 빠른 1D 평면 분석을 위한 **TMM** (Transfer Matrix Method), 2D 주기 구조를 위한 4개의 **RCWA** (Rigorous Coupled-Wave Analysis) 구현체(torcwa, grcwa, meent, fmmax), 그리고 임의 기하 구조를 위한 **FDTD** (Finite-Difference Time-Domain) 솔버입니다. 이 가이드는 벤치마크 결과, 교차 솔버 검증 데이터, 실용적인 선택 지침을 통합하여 올바른 솔버를 선택하는 데 도움을 줍니다.

## Part 1: RCWA 솔버 벤치마크

이 섹션에서는 동일한 픽셀 구조에서 RCWA 솔버(torcwa, grcwa, meent)를 비교하여 정확도와 성능을 측정합니다.

### 목표

동일한 BSI 픽셀 시뮬레이션을 사용 가능한 모든 RCWA 솔버로 실행한 후, QE 결과와 실행 시간을 비교합니다.

### 설정

```python
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner
from compass.analysis.solver_comparison import SolverComparison
from compass.visualization.qe_plot import plot_qe_comparison

pixel_config = {
    "pitch": 1.0,
    "unit_cell": [2, 2],
    "bayer_map": [["R", "G"], ["G", "B"]],
    "layers": {
        "air": {"thickness": 1.0, "material": "air"},
        "microlens": {
            "enabled": True, "height": 0.6,
            "radius_x": 0.48, "radius_y": 0.48,
            "material": "polymer_n1p56",
            "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
            "shift": {"mode": "none"},
        },
        "planarization": {"thickness": 0.3, "material": "sio2"},
        "color_filter": {
            "thickness": 0.6,
            "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
            "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
        },
        "barl": {"layers": [
            {"thickness": 0.010, "material": "sio2"},
            {"thickness": 0.025, "material": "hfo2"},
            {"thickness": 0.015, "material": "sio2"},
            {"thickness": 0.030, "material": "si3n4"},
        ]},
        "silicon": {
            "thickness": 3.0, "material": "silicon",
            "photodiode": {"position": [0, 0, 0.5], "size": [0.7, 0.7, 2.0]},
            "dti": {"enabled": True, "width": 0.1, "material": "sio2"},
        },
    },
}

source_config = {
    "wavelength": {"mode": "sweep", "sweep": {"start": 0.42, "stop": 0.68, "step": 0.02}},
    "polarization": "unpolarized",
}
```

### 모든 RCWA 솔버 실행

```python
solvers = [
    {"name": "torcwa", "type": "rcwa", "params": {"fourier_order": [9, 9]}},
    {"name": "grcwa",  "type": "rcwa", "params": {"fourier_order": [9, 9]}},
    {"name": "meent",  "type": "rcwa", "params": {"fourier_order": [9, 9]}},
]

results = []
labels = []

for solver_cfg in solvers:
    config = {
        "pixel": pixel_config,
        "solver": {
            **solver_cfg,
            "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
        },
        "source": source_config,
        "compute": {"backend": "auto"},
    }

    try:
        result = SingleRunner.run(config)
        results.append(result)
        labels.append(solver_cfg["name"])
        runtime = result.metadata.get("runtime_seconds", 0)
        print(f"{solver_cfg['name']}: {runtime:.2f}s")
    except Exception as e:
        print(f"{solver_cfg['name']}: FAILED - {e}")
```

### QE 스펙트럼 비교

```python
if len(results) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_qe_comparison(results, labels, ax=ax)
    ax.set_title("RCWA Solver Comparison")
    plt.tight_layout()
    plt.savefig("solver_comparison_qe.png", dpi=150)
    plt.show()
```

### 정량적 비교

```python
if len(results) >= 2:
    comparison = SolverComparison(results, labels)
    summary = comparison.summary()

    print("\n=== Comparison Summary ===")
    print("\nMax QE difference (absolute):")
    for key, val in summary["max_qe_diff"].items():
        print(f"  {key}: {val:.5f}")

    print("\nMax QE relative error (%):")
    for key, val in summary["max_qe_relative_error_pct"].items():
        print(f"  {key}: {val:.2f}%")

    print("\nRuntimes:")
    for solver, runtime in summary["runtimes_seconds"].items():
        print(f"  {solver}: {runtime:.2f}s")
```

<RCWAConvergenceDemo />

<FourierOrderDemo />

### 푸리에 차수 수렴 비교

각 솔버가 푸리에 차수(Fourier order)에 따라 어떻게 수렴하는지 비교합니다:

```python
orders = [5, 7, 9, 11, 13, 15]
convergence_data = {s["name"]: [] for s in solvers}

for order in orders:
    for solver_cfg in solvers:
        config = {
            "pixel": pixel_config,
            "solver": {
                **solver_cfg,
                "params": {"fourier_order": [order, order]},
                "stability": {"precision_strategy": "mixed"},
            },
            "source": {
                "wavelength": {"mode": "single", "value": 0.55},
                "polarization": "unpolarized",
            },
            "compute": {"backend": "auto"},
        }

        try:
            result = SingleRunner.run(config)
            avg_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
            convergence_data[solver_cfg["name"]].append(avg_qe)
        except:
            convergence_data[solver_cfg["name"]].append(np.nan)

    print(f"Order {order}: done")

# Plot convergence
plt.figure(figsize=(8, 5))
for name, qe_values in convergence_data.items():
    plt.plot(orders, qe_values, "o-", label=name, linewidth=2)

plt.xlabel("Fourier Order")
plt.ylabel("Average QE at 550 nm")
plt.title("Convergence: QE vs Fourier Order")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence_comparison.png", dpi=150)
```

### 실행 시간 스케일링

푸리에 차수에 따른 실행 시간의 변화를 측정합니다:

```python
orders = [5, 7, 9, 11, 13, 15, 17]
timing_data = {s["name"]: [] for s in solvers}

for order in orders:
    for solver_cfg in solvers:
        config = {
            "pixel": pixel_config,
            "solver": {
                **solver_cfg,
                "params": {"fourier_order": [order, order]},
                "stability": {"precision_strategy": "mixed"},
            },
            "source": {
                "wavelength": {"mode": "single", "value": 0.55},
                "polarization": "TE",  # Single pol for timing
            },
            "compute": {"backend": "auto"},
        }

        try:
            result = SingleRunner.run(config)
            t = result.metadata.get("runtime_seconds", 0)
            timing_data[solver_cfg["name"]].append(t)
        except:
            timing_data[solver_cfg["name"]].append(np.nan)

# Plot
plt.figure(figsize=(8, 5))
for name, times in timing_data.items():
    plt.plot(orders, times, "o-", label=name, linewidth=2)

plt.xlabel("Fourier Order")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Scaling vs Fourier Order")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.tight_layout()
plt.savefig("runtime_scaling.png", dpi=150)
```

### 예상 결과

- **QE 일치도**: 모든 RCWA 솔버는 동일한 푸리에 차수에서 0.5-1% QE 이내로 일치해야 합니다.
- **수렴성**: 모든 솔버는 차수가 증가함에 따라 동일한 QE로 수렴합니다. 수렴 속도는 약간 다를 수 있습니다.
- **실행 시간**: torcwa는 일반적으로 GPU에서 가장 빠릅니다. grcwa와 meent도 경쟁력 있는 성능을 보입니다.
- **FDTD 대 RCWA**: 1-2% QE 이내로 일치해야 합니다. FDTD는 단일 파장 실행에서 상당히 느립니다.

## Part 2: TMM vs RCWA

TMM(1D)과 RCWA(2D/3D) 솔버 간 교차 검증은 서로 다른 수치 방법이 물리적으로 일관된 결과를 생성하는지 확인하고, 1D 근사가 2D 전자기 해석 대비 어디서 한계를 보이는지 식별하는 데 도움을 줍니다.

### 개요

1.0 um BSI 픽셀 스택에서 세 가지 솔버 비교:

- **TMM** (Transfer Matrix Method): 1D 해석적 솔버, 각 층을 무한 균일 슬래브로 처리합니다. 횡방향 패터닝(마이크로렌즈, Bayer 패턴, DTI)을 무시하므로 빠르지만 2D/3D 효과를 반영하지 못합니다.
- **torcwa**: PyTorch 기반 2D RCWA 솔버, S-matrix 알고리즘을 사용합니다. GPU 가속이 가능하며 미분 가능한 시뮬레이션을 지원합니다.
- **grcwa**: NumPy 기반 2D RCWA 솔버, autograd를 지원합니다. 역설계 최적화에 적합하며 CPU 환경에서 안정적으로 동작합니다.

::: info 시뮬레이션 파라미터
- **픽셀**: 1.0 um 피치 BSI, 2x2 RGGB Bayer
- **스택**: 9개 층 (air / microlens / planarization / CF / 4층 BARL / silicon)
- **RCWA**: Fourier order [3,3], 마이크로렌즈 5 슬라이스, 64x64 그리드
- **광원**: 수직 입사, 무편광, 380-780 nm (20 nm 스텝)
- **플랫폼**: macOS (Apple Silicon), CPU 전용
:::

### 인터랙티브 비교

<CrossSolverValidation />

### 파장 스윕 결과

#### 전체 데이터 테이블

수직 입사, 무편광 조건에서 380-780 nm 가시광 스펙트럼에 대한 세 솔버의 반사율(R), 투과율(T), 흡수율(A) 비교입니다.

| λ (nm) | TMM R | TMM T | TMM A | torcwa R | torcwa T | torcwa A | grcwa R | grcwa T | grcwa A |
|-------:|------:|------:|------:|---------:|---------:|---------:|--------:|--------:|--------:|
| 380 | 0.0401 | 0.0517 | 0.9081 | 0.0147 | 0.0000 | 0.9853 | 0.0195 | 0.0000 | 0.9805 |
| 400 | 0.0529 | 0.0722 | 0.8750 | 0.0239 | 0.0000 | 0.9761 | 0.0139 | 0.0000 | 0.9861 |
| 420 | 0.0961 | 0.0830 | 0.8210 | 0.0021 | 0.0000 | 0.9979 | 0.0131 | 0.0000 | 0.9869 |
| 440 | 0.0348 | 0.1085 | 0.8566 | 0.0069 | 0.0000 | 0.9931 | 0.0125 | 0.0000 | 0.9875 |
| 460 | 0.0633 | 0.1443 | 0.7923 | 0.0018 | 0.0000 | 0.9982 | 0.0102 | 0.0000 | 0.9898 |
| 480 | 0.0324 | 0.2549 | 0.7127 | 0.0138 | 0.0000 | 0.9862 | 0.0119 | 0.0000 | 0.9881 |
| 500 | 0.1461 | 0.4549 | 0.3990 | 0.0186 | 0.0000 | 0.9814 | 0.0148 | 0.0000 | 0.9852 |
| 520 | 0.0088 | 0.9135 | 0.0777 | 0.0258 | 0.0000 | 0.9742 | 0.0144 | 0.0000 | 0.9856 |
| 540 | 0.2405 | 0.7023 | 0.0572 | 0.0131 | 0.0001 | 0.9868 | 0.0121 | 0.0000 | 0.9879 |
| 560 | 0.1697 | 0.4775 | 0.3528 | 0.0013 | 0.0003 | 0.9984 | 0.0114 | 0.0000 | 0.9886 |
| 580 | 0.0280 | 0.3331 | 0.6389 | 0.0021 | 0.0008 | 0.9971 | 0.0139 | 0.0000 | 0.9861 |
| 600 | 0.0387 | 0.2423 | 0.7190 | 0.0077 | 0.0026 | 0.9897 | 0.0176 | 0.0000 | 0.9824 |
| 620 | 0.0477 | 0.2175 | 0.7349 | 0.0137 | 0.0061 | 0.9802 | 0.0202 | 0.0000 | 0.9797 |
| 640 | 0.0180 | 0.2258 | 0.7563 | 0.0223 | 0.0093 | 0.9684 | 0.0206 | 0.0001 | 0.9793 |
| 660 | 0.0209 | 0.2336 | 0.7455 | 0.0126 | 0.0113 | 0.9761 | 0.0193 | 0.0001 | 0.9807 |
| 680 | 0.0781 | 0.2284 | 0.6935 | 0.0116 | 0.0126 | 0.9757 | 0.0173 | 0.0001 | 0.9826 |
| 700 | 0.1199 | 0.2252 | 0.6549 | 0.0093 | 0.0130 | 0.9777 | 0.0159 | 0.0001 | 0.9840 |
| 720 | 0.1091 | 0.2345 | 0.6564 | 0.0055 | 0.0140 | 0.9805 | 0.0161 | 0.0002 | 0.9837 |
| 740 | 0.0664 | 0.2522 | 0.6815 | 0.0032 | 0.0147 | 0.9821 | 0.0178 | 0.0003 | 0.9819 |
| 760 | 0.0372 | 0.2667 | 0.6961 | 0.0045 | 0.0142 | 0.9813 | 0.0205 | 0.0004 | 0.9792 |
| 780 | 0.0430 | 0.2721 | 0.6849 | 0.0116 | 0.0145 | 0.9739 | 0.0231 | 0.0004 | 0.9765 |

### 주요 관찰 사항

#### 1D vs 2D 차이

##### 1. 흡수율 차이: TMM vs RCWA

TMM은 가시광 전 영역에서 RCWA보다 현저히 낮은 흡수율을 보여줍니다. TMM의 흡수율은 40-90% 범위에서 크게 변동하는 반면, RCWA(torcwa, grcwa)는 97-99% 이상의 높은 흡수율을 일관되게 유지합니다.

이 차이의 주요 원인:
- **TMM은 1D 균일 슬래브 모델**: 마이크로렌즈의 집광 효과를 반영하지 못합니다. 실제 마이크로렌즈는 입사광을 실리콘 포토다이오드 영역으로 집중시켜 흡수율을 크게 높입니다.
- **RCWA는 2D 패터닝 효과 반영**: 마이크로렌즈 형상, 컬러 필터 패턴, DTI 구조 등 횡방향 구조를 모두 고려하므로 실제 픽셀에 더 가까운 결과를 제공합니다.
- **박막 간섭 패턴 차이**: TMM에서는 다층 박막 간섭이 큰 진동을 만들지만(특히 520 nm 부근 투과 창), RCWA에서는 2D 구조의 회절 효과가 이러한 간섭 패턴을 평탄화합니다.

##### 2. 투과율 차이: TMM T > 0 vs RCWA T ≈ 0

TMM은 모든 파장에서 상당한 투과율(5-91%)을 보이는 반면, RCWA는 투과율이 거의 0에 가깝습니다.

- **TMM**: 실리콘을 3 um 두께의 균일 슬래브로 처리합니다. 장파장에서 실리콘의 소광 계수(k)가 작아 상당량의 빛이 투과합니다.
- **RCWA**: S-matrix 알고리즘에서 실리콘 기판 층의 두꺼운 흡수를 정확하게 계산합니다. 3 um 두께 실리콘에서 2D 회절 모드의 실효 광 경로가 길어져 대부분의 빛이 흡수됩니다. 또한 DTI 구조가 횡방향 빛의 탈출을 방지하여 투과율을 더욱 낮춥니다.

##### 3. 반사율 차이: TMM 간섭 진동 vs RCWA 평탄 저반사

TMM의 반사율 스펙트럼은 다층 박막 간섭으로 인해 큰 진동(1-24%)을 보입니다. 반면 RCWA는 전 파장에서 1-2%의 매우 낮고 평탄한 반사율을 보여줍니다.

- **TMM**: 모든 계면에서의 Fresnel 반사가 간섭하여 파장에 따른 큰 진동을 만듭니다. 이는 1D 평면 슬래브 모델의 특성입니다.
- **RCWA**: 마이크로렌즈의 곡면 형상이 반사광을 분산시키고, 2D 패터닝이 간섭 패턴을 깨뜨려 전체적으로 낮고 부드러운 반사 스펙트럼을 만듭니다. 이는 실제 센서에서 마이크로렌즈가 반사 방지 역할도 수행함을 보여줍니다.

##### 4. torcwa vs grcwa 일치도

두 RCWA 솔버는 전 파장에서 0.5-2% 이내로 일치합니다. 이는 두 독립적인 구현체가 동일한 물리를 올바르게 계산하고 있음을 검증합니다.

- 반사율: 두 솔버 모두 1-2% 범위의 낮은 반사율
- 투과율: 두 솔버 모두 거의 0 (grcwa가 torcwa보다 약간 더 낮음)
- 흡수율: 두 솔버 모두 97-99% 범위

미세한 차이는 수치 구현의 차이(PyTorch vs NumPy, 부동소수점 연산 순서, 고유값 분해 알고리즘)에서 기인하며, 물리적으로 유의미한 차이가 아닙니다.

### TMM vs RCWA 사용 시점

| 용도 | 추천 솔버 | 이유 |
|------|----------|------|
| 스택 설계 / BARL 최적화 | TMM | ~3 ms로 수천 가지 파라미터 조합을 빠르게 탐색 가능 |
| 빠른 파라미터 스윕 | TMM | 전체 파장 스윕이 수 밀리초 이내 |
| 2D 효과 분석 (마이크로렌즈, DTI) | RCWA (torcwa/grcwa) | 횡방향 구조 효과를 정확하게 반영 |
| QE 절대값 예측 | RCWA (torcwa/grcwa) | 실제 픽셀에 가까운 흡수율 계산 |
| 크로스토크 분석 | RCWA (torcwa/grcwa) | 인접 픽셀 간 광 결합을 계산 가능 |
| 역설계 최적화 | grcwa | autograd 지원으로 기울기 기반 최적화 가능 |

### 실행 시간 비교

| 솔버 | 파장 수 | 실행 시간 | 속도비 |
|------|:-------:|--------:|------:|
| TMM | 21 | 2.9 ms | 5400x |
| grcwa | 21 | 0.1 s | 157x |
| torcwa | 21 | 15.7 s | 1x |

- **TMM**은 해석적 행렬 곱셈만 수행하므로 밀리초 단위로 동작합니다. BARL 최적화나 빠른 파라미터 스윕에 이상적입니다.
- **grcwa**는 NumPy 기반이지만 효율적인 구현 덕분에 torcwa 대비 약 157배 빠릅니다. CPU 전용 환경에서 특히 유리합니다.
- **torcwa**는 PyTorch 기반으로 GPU 환경에서 더 빠를 수 있지만, CPU 전용 실행에서는 오버헤드가 있습니다.

### 솔버 호환성 참고

::: warning meent 수치 안정성
meent 0.12.0은 다층 2D 구조에서 수치 불안정성 문제가 있습니다: 패턴 층이 2개 이상일 때 R+T > 1이 발생합니다. 단층 시뮬레이션은 정상 동작합니다. 현재 원인을 조사 중입니다.
:::

::: tip FDTD 물질 흡수 처리
flaport fdtd 솔버는 복소 유전율의 허수부를 모델링하기 위해 복셀별 전도도 기반 감쇠를 사용합니다. 2패스 참조 정규화와 결합하여 가시광 영역 전체에서 RCWA와 3% 이내의 흡수율 일치를 달성합니다.
:::

### 에너지 보존

각 솔버의 에너지 보존 정확도 (|1 - (R+T+A)| 최대값):

| 솔버 | 최대 \|1-(R+T+A)\| | 비고 |
|------|:------------------:|------|
| TMM | 1.11 x 10⁻¹⁶ | 기계 정밀도 (해석적 방법) |
| torcwa | 0.0000 | S-matrix 알고리즘 보장 |
| grcwa | 0.0000 | S-matrix 알고리즘 보장 |

TMM은 이산화 오차가 없는 해석적 방법이므로 기계 정밀도 수준의 에너지 보존을 달성합니다. RCWA 솔버들도 S-matrix 알고리즘의 고유한 에너지 보존 특성 덕분에 정확한 에너지 보존을 보여줍니다.

### 실행 환경

```
플랫폼     : macOS (Darwin 25.2.0, Apple Silicon)
Python    : 3.11
PyTorch   : 2.5.0 (torcwa 백엔드)
NumPy     : (grcwa 백엔드)
RCWA Order: [3, 3] (49 harmonics)
그리드      : 64 × 64
렌즈 슬라이스 : 5
```

## Part 3: RCWA vs FDTD

이 섹션에서는 BSI CMOS 픽셀에 대해 RCWA (grcwa) 결과를 FDTD (flaport)와 교차 검증하고, 직광(직접 조명)과 원뿔 조명을 비교하는 방법을 보여줍니다.

### 인터랙티브 차트

<RcwaFdtdValidation />

### 교차 솔버 검증이 필요한 이유

RCWA와 FDTD는 근본적으로 다른 접근 방식으로 맥스웰 방정식을 풉니다:

| | RCWA | FDTD |
|---|---|---|
| **영역** | 주파수 영역 | 시간 영역 |
| **주기성** | 본질적으로 주기적 | PML 경계 필요 |
| **장점** | 박막 스택에 빠름 | 임의 기하 처리 가능 |
| **수렴** | 푸리에 차수 | 격자 간격 + 실행 시간 |

두 방법이 흡수/반사/투과 스펙트럼에서 일치하면, 시뮬레이션의 물리적 타당성에 대한 강한 확신을 제공합니다.

### 검증 실행

```bash
PYTHONPATH=. python3.11 scripts/validate_rcwa_vs_fdtd.py
```

이 스크립트는 세 가지 실험을 수행합니다:

#### 실험 1: 수직 입사 스윕

grcwa (fourier_order=[5,5]) vs fdtd_flaport (dx=0.015um, 500fs, pml=20)을 400-700nm 범위에서 비교합니다. FDTD 솔버는 복셀별 흡수 감쇠와 2패스 참조 정규화를 사용하여 정확한 R/T/A를 추출합니다.

**허용 기준:** 최대 |A_grcwa - A_fdtd| < 5%

#### 실험 2: 원뿔 조명

grcwa를 사용하여 세 가지 조명 조건을 비교합니다:
- **직광**: 수직 입사 (θ=0°)
- **원뿔 F/2.0 CRA=0°**: 피보나치 19포인트 샘플링, 코사인 가중치
- **원뿔 F/2.0 CRA=15°**: 동일 원뿔에 15° 주광선 각도(CRA)

#### 실험 3: RCWA 교차 검증

동일한 원뿔 조명에서 grcwa vs torcwa를 검증하여 RCWA 솔버 일관성을 확인합니다.

**허용 기준:** 최대 |A_grcwa - A_torcwa| < 5%

### ConeIlluminationRunner 사용법

```python
from compass.runners.cone_runner import ConeIlluminationRunner

config = {
    "pixel": { ... },  # 픽셀 스택 설정
    "solver": {"name": "grcwa", "type": "rcwa", "params": {"fourier_order": [5, 5]}},
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
        "cone": {
            "cra_deg": 0.0,       # 주광선 각도
            "f_number": 2.0,       # 렌즈 F-넘버
            "sampling": {"type": "fibonacci", "n_points": 19},
            "weighting": "cosine",
        },
    },
    "compute": {"backend": "cpu"},
}

result = ConeIlluminationRunner.run(config)
```

러너 동작:
1. `ConeIllumination`에서 각도 샘플링 포인트 생성
2. 각 각도마다: (θ, φ)로 소스를 설정하고 솔버 실행
3. 가중 R, T, A 및 픽셀별 QE 누적
4. 가중 평균된 단일 `SimulationResult` 반환

### 수렴 팁

RCWA와 FDTD 결과가 벌어질 경우:

- **FDTD**: `runtime` 증가 (500→1000fs), `grid_spacing` 감소 (0.015→0.01um), `pml_layers` 증가 (20→30)
- **RCWA**: `fourier_order` 증가 ([5,5]→[9,9]→[13,13])

FDTD 정확도의 핵심 요소:

1. **복셀별 흡수 감쇠**: flaport 라이브러리는 그리드에 실수 유전율만 지원합니다. COMPASS는 허수 유전율에서 등가 전도도를 계산하고 매 타임스텝 후 복셀별 E-필드 감쇠를 적용합니다: `damping = (1 - α) / (1 + α)`, 여기서 `α = σ·dt / (2·ε₀·εᵣ)`.
2. **2패스 참조 정규화**: 진공 참조 시뮬레이션으로 입사 전력(P_inc)을 확립합니다. 구조 시뮬레이션의 반사율은 참조 대비 초과 상향 플럭스로 계산되어 소프트 소스 아티팩트를 제거합니다.
3. **충분한 실행 시간**: CW 소스가 3 um 실리콘 구조에서 정상 상태에 도달하려면 최소 500 fs가 권장됩니다.

## Part 4: 솔버 선택 가이드

시뮬레이션에 적합한 솔버를 선택하기 위한 의사결정 트리입니다:

### 1. 균일(패턴 없는) 층만 있습니까?

**예 → TMM 사용**
- RCWA보다 ~1000배, FDTD보다 ~5000배 빠름
- BARL 두께 최적화, 스택 설계, 빠른 파라미터 스윕에 이상적
- 실행 시간: 전체 파장 스윕(21 파장) ~3 ms
- 제한: 마이크로렌즈 집광, Bayer 패턴 회절, DTI 제한 효과를 모델링할 수 없음

### 2. 주기적 2D 패턴 층(마이크로렌즈, 컬러 필터 그리드, DTI)이 있습니까?

**예 → RCWA 사용**

필요에 따라 특정 RCWA 솔버를 선택하세요:

| 솔버 | 백엔드 | 주요 장점 | 최적 용도 |
|------|--------|----------|----------|
| **torcwa** | PyTorch | GPU 가속, S-matrix 안정성, 정밀도를 위한 TF32 비활성화 | 프로덕션 GPU 실행, 큰 푸리에 차수 |
| **grcwa** | NumPy | autograd 지원, CPU에서 빠름, 프로젝트 핵심 솔버 | 역설계, CPU 전용 환경, 교차 검증 |
| **meent** | PyTorch/JAX | 멀티 백엔드 유연성 | 실험적 사용, 단층 구조만 (안정성 참고) |
| **fmmax** | JAX | 4가지 FMM 벡터 공식 | 연구, 고급 푸리에 분해 연구 |

### 3. 비주기적이거나 복잡한 3D 기하 구조입니까?

**예 → FDTD 사용**

| 솔버 | 백엔드 | 주요 장점 | 최적 용도 |
|------|--------|----------|----------|
| **fdtd_flaport** | PyTorch | 간단한 API, 미분 가능 | 빠른 FDTD 프로토타이핑 |
| **fdtdz** | JAX | 효율적인 z-전파 | JAX 기반 워크플로우 |
| **fdtdx** | JAX | 멀티 GPU, 미분 가능 | 대규모 역설계 |
| **meep** | C++/Python | 성숙하고 기능 풍부 | 복잡한 기하, 참조 솔루션 |

### 빠른 참조

| 시나리오 | 추천 솔버 | 대략적 실행 시간 |
|---------|----------|---------------|
| BARL 두께 스윕 (1000 설정) | TMM | 총 ~3초 |
| 단일 파장 QE (2D 픽셀) | grcwa 또는 torcwa | 0.5-15초 |
| 전체 파장 스윕 (2D 픽셀) | grcwa (CPU) / torcwa (GPU) | 0.1-16초 |
| 역설계 최적화 | grcwa (autograd) | 반복당 수 분 |
| 교차 검증 참조 | TMM + RCWA 솔버 2개 실행 | 결과 비교 |
| 비주기 구조 | meep 또는 fdtdx | 수 분~수 시간 |
