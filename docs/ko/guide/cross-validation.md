---
title: 교차 검증
description: ComparisonRunner와 SolverComparison을 사용하여 COMPASS에서 동일한 픽셀 구조에 대해 둘 이상의 EM 솔버를 비교하는 워크플로우.
---

# 교차 검증(Cross-Validation)

교차 검증은 동일한 픽셀 구조를 두 개 이상의 독립적인 EM 솔버(Solver)로 실행하고 결과를 비교하는 실천 방법입니다. 이는 COMPASS의 핵심 동기 중 하나입니다. 서로 다른 수치 방법 간의 일치를 검증하여 시뮬레이션(Simulation) 예측이 올바르다는 신뢰를 확립하는 것입니다.

## 교차 검증의 이유

단일 솔버가 모든 구조에 대해 정확하다고 보장할 수 없습니다. 교차 검증은 다음에 도움이 됩니다:

- 특정 솔버의 구현 버그 또는 설정 오류 감지
- RCWA 푸리에 차수(Fourier Order) 또는 FDTD 격자 간격(Grid Spacing)이 수렴되었는지 검증
- QE 예측에 대한 오차 범위 설정
- 서로 다른 방법이 불일치하는 구조 식별(및 원인 조사)

잘 수렴된 솔버 간에 표준 BSI 픽셀 구조에서 절대 QE 1-2% 이내의 일치가 예상됩니다.

## ComparisonRunner를 사용한 빠른 시작

`ComparisonRunner`는 동일한 시뮬레이션 설정을 여러 솔버로 실행하고 비교 요약을 생성합니다:

```python
from compass.runners.comparison_runner import ComparisonRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {"enabled": True, "height": 0.6,
                          "radius_x": 0.48, "radius_y": 0.48},
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
            },
            "barl": {"layers": [
                {"thickness": 0.010, "material": "sio2"},
                {"thickness": 0.025, "material": "hfo2"},
            ]},
            "silicon": {
                "thickness": 3.0, "material": "silicon",
                "dti": {"enabled": True, "width": 0.1},
            },
        },
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

solver_configs = [
    {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    {
        "name": "grcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9], "dtype": "complex128"},
    },
    {
        "name": "meent", "type": "rcwa",
        "params": {"fourier_order": [9, 9], "backend": "numpy"},
    },
]

comparison = ComparisonRunner.run(config, solver_configs)
```

## 비교 출력 이해

`ComparisonRunner.run()`은 세 개의 키를 가진 딕셔너리를 반환합니다:

```python
# Individual SimulationResult objects
results = comparison["results"]       # list of SimulationResult
labels = comparison["labels"]         # ["torcwa", "grcwa", "meent"]
summary = comparison["summary"]       # dict with comparison metrics

print("Max QE difference per pixel:")
for key, val in summary["max_qe_diff"].items():
    print(f"  {key}: {val:.4f}")

print("\nMean QE difference per pixel:")
for key, val in summary["mean_qe_diff"].items():
    print(f"  {key}: {val:.4f}")

print("\nMax relative error (%):")
for key, val in summary["max_qe_relative_error_pct"].items():
    print(f"  {key}: {val:.2f}%")

print("\nRuntime comparison:")
for key, val in summary["runtimes_seconds"].items():
    print(f"  {key}: {val:.2f} s")
```

## SolverComparison 직접 사용

더 상세한 분석을 위해 `SolverComparison` 클래스를 직접 사용합니다:

```python
from compass.analysis.solver_comparison import SolverComparison

comp = SolverComparison(
    results=comparison["results"],
    labels=comparison["labels"],
    reference_idx=0,              # torcwa as reference
)

# Absolute QE difference vs reference for each pixel at each wavelength
qe_diff = comp.qe_difference()
for key, arr in qe_diff.items():
    print(f"{key}: max |dQE| = {arr.max():.5f}, mean = {arr.mean():.5f}")

# Relative error (%)
qe_rel = comp.qe_relative_error()
for key, arr in qe_rel.items():
    print(f"{key}: max relative error = {arr.max():.2f}%")

# Runtime comparison
runtimes = comp.runtime_comparison()
for solver, time_s in runtimes.items():
    print(f"{solver}: {time_s:.2f} s")
```

## 비교 결과 플롯

### QE 스펙트럼 오버레이

```python
import numpy as np
import matplotlib.pyplot as plt
from compass.visualization.qe_plot import plot_qe_comparison

fig, (ax_main, ax_diff) = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
)

# Plot QE spectra on top panel
colors = {"torcwa": "tab:blue", "grcwa": "tab:orange", "meent": "tab:green"}
ref_result = comparison["results"][0]
wl_nm = ref_result.wavelengths * 1000

for result, label in zip(comparison["results"], comparison["labels"]):
    for pixel_name, qe in result.qe_per_pixel.items():
        if pixel_name.startswith("G"):
            ax_main.plot(wl_nm, qe, color=colors[label], label=f"{label} ({pixel_name})")
            break

ax_main.set_ylabel("Quantum Efficiency")
ax_main.set_title("Cross-Validation: Green QE Spectrum")
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# Plot pairwise difference on bottom panel
for key, diff in comp.qe_difference().items():
    if "G_0_1" in key or "G_1_0" in key:
        ax_diff.plot(wl_nm, diff, label=key)

ax_diff.set_xlabel("Wavelength (nm)")
ax_diff.set_ylabel("|QE difference|")
ax_diff.legend()
ax_diff.grid(True, alpha=0.3)

plt.tight_layout()
```

### 일치도 히트맵

```python
fig, ax = plt.subplots(figsize=(8, 6))

# Build matrix of max QE differences between all solver pairs
n = len(comparison["labels"])
diff_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        pair_comp = SolverComparison(
            results=[comparison["results"][i], comparison["results"][j]],
            labels=[comparison["labels"][i], comparison["labels"][j]],
        )
        all_diffs = pair_comp.qe_difference()
        max_diff = max(arr.max() for arr in all_diffs.values())
        diff_matrix[i, j] = max_diff
        diff_matrix[j, i] = max_diff

im = ax.imshow(diff_matrix, cmap="YlOrRd", vmin=0, vmax=0.05)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(comparison["labels"])
ax.set_yticklabels(comparison["labels"])
plt.colorbar(im, label="Max |QE diff|")
ax.set_title("Pairwise Max QE Difference")
```

<SolverComparisonChart />

## RCWA vs FDTD 교차 검증

RCWA와 FDTD를 비교하면 근본적으로 다른 수치 방법을 사용하므로 가장 강력한 검증을 제공합니다:

```python
solver_configs = [
    {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [13, 13]},
        "stability": {"precision_strategy": "mixed"},
    },
    {
        "name": "fdtd_flaport", "type": "fdtd",
        "params": {"grid_spacing": 0.01, "runtime": 300, "pml_layers": 20},
    },
]

comparison = ComparisonRunner.run(config, solver_configs)
```

RCWA와 FDTD를 비교할 때, 불일치가 불충분한 수렴이 아닌 방법 간의 진정한 차이를 반영하도록 양쪽 모두 고정밀 설정(RCWA 푸리에 차수 >= 13, FDTD 격자 간격 <= 10 nm)을 사용하십시오.

## 불일치 해석

| 최대 |dQE|   | 해석                                             |
|-------------|--------------------------------------------------|
| < 0.01      | 우수한 일치. 결과를 신뢰할 수 있습니다.               |
| 0.01 - 0.03 | 양호한 일치. 서로 다른 솔버에서 정상적입니다.          |
| 0.03 - 0.05 | 허용 가능. 양쪽 솔버의 수렴을 확인하십시오.           |
| > 0.05      | 조사 필요. 수렴 또는 설정 문제 가능성이 있습니다.      |

불일치의 일반적 원인:

1. **불충분한 푸리에 차수** -- 차수를 증가시키고 다시 실행
2. **불충분한 FDTD 격자 해상도** -- 격자 간격 감소
3. **상이한 푸리에 인수분해 규칙** -- 양쪽 모두 Li의 역규칙을 사용하는지 확인
4. **정밀도 불일치** -- 한 솔버는 float32, 다른 솔버는 float64
5. **경계 조건 차이** -- 주기적(RCWA) vs PML(FDTD)

## 명령줄 비교

배치 처리를 위해 비교 스크립트를 사용합니다:

```bash
python scripts/compare_solvers.py experiment=solver_comparison

# Override solvers and pixel
python scripts/compare_solvers.py \
    experiment=solver_comparison \
    pixel=default_bsi_1um \
    solvers="[torcwa,grcwa]"
```

## 다음 단계

- [RCWA 실행](./running-rcwa.md) -- RCWA 솔버 세부사항
- [FDTD 실행](./running-fdtd.md) -- FDTD 솔버 세부사항
- [솔버 벤치마크 쿡북](../cookbook/solver-benchmark.md) -- 전체 벤치마크 레시피
