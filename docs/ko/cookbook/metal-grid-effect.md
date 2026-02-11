# 메탈 그리드 효과

이 레시피는 컬러 필터(color filter) 서브 픽셀 사이의 텅스텐(tungsten) 메탈 그리드(metal grid)가 QE 및 광학적 크로스토크(optical crosstalk)에 미치는 영향을 보여줍니다.

## 배경

메탈 그리드는 인접한 컬러 필터 요소 사이에 위치합니다. 컬러 필터 층을 통해 인접 픽셀로 빛이 유입되는 것을 차단하여 광학적 격리를 제공합니다. 그러나 다음과 같은 영향도 있습니다:

- 유효 개구(effective aperture)가 감소합니다 (각 픽셀에 도달하는 빛이 줄어듦)
- 그리드 가장자리에서 회절(diffraction) 효과가 발생할 수 있습니다
- 일부 빛을 흡수합니다 (텅스텐은 손실성 물질)

이 레시피는 메탈 그리드가 있는 경우와 없는 경우의 두 가지 시뮬레이션을 실행하고 결과를 비교합니다.

## 설정

```python
import copy
from compass.runners.single_run import SingleRunner
from compass.analysis.solver_comparison import SolverComparison
from compass.visualization.qe_plot import plot_qe_comparison, plot_crosstalk_heatmap
import matplotlib.pyplot as plt

base_config = {
    "pixel": {
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
                "pattern": "bayer_rggb",
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                ]
            },
            "silicon": {
                "thickness": 3.0, "material": "silicon",
                "photodiode": {"position": [0, 0, 0.5], "size": [0.7, 0.7, 2.0]},
                "dti": {"enabled": True, "width": 0.1, "material": "sio2"},
            },
        },
    },
    "solver": {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [11, 11]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

## 실행: 메탈 그리드 포함

```python
config_with_grid = copy.deepcopy(base_config)
config_with_grid["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = True

result_with = SingleRunner.run(config_with_grid)
print("With grid: done")
```

## 실행: 메탈 그리드 미포함

```python
config_no_grid = copy.deepcopy(base_config)
config_no_grid["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = False

result_without = SingleRunner.run(config_no_grid)
print("Without grid: done")
```

## QE 스펙트럼 비교

```python
fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=[result_with, result_without],
    labels=["With grid", "No grid"],
    ax=ax,
)
ax.set_title("Metal Grid Effect on QE")
plt.tight_layout()
plt.savefig("metal_grid_qe_comparison.png", dpi=150)
plt.show()
```

## 차이 정량화

```python
comparison = SolverComparison(
    results=[result_with, result_without],
    labels=["with_grid", "no_grid"],
    reference_idx=0,
)
summary = comparison.summary()

for key, val in summary["max_qe_diff"].items():
    print(f"{key}: max |dQE| = {val:.4f}")
```

## 크로스토크 비교

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_crosstalk_heatmap(result_with, ax=ax1)
ax1.set_title("Crosstalk: With Grid")

plot_crosstalk_heatmap(result_without, ax=ax2)
ax2.set_title("Crosstalk: No Grid")

plt.tight_layout()
plt.savefig("metal_grid_crosstalk.png", dpi=150)
plt.show()
```

<CrosstalkHeatmap />

## 예상 관측 결과

1. **그리드에 의한 QE 감소**: 메탈 그리드는 일부 빛을 차단하고 에너지를 흡수하므로 피크 QE를 약간 감소시킵니다 (일반적으로 2-5%).
2. **크로스토크 개선**: 그리드는 인접 픽셀 간의 광학적 크로스토크를 크게 줄여줍니다. 특히 비축(off-axis) 조명에서 효과가 두드러집니다.
3. **파장 의존성**: 그리드 폭 대비 회절 효과가 더 뚜렷한 짧은 파장에서 그리드 효과가 더 강합니다.

## 그리드 폭 스윕

그리드 폭이 QE/크로스토크 트레이드오프에 미치는 영향을 연구합니다:

```python
import numpy as np

grid_widths = [0.0, 0.03, 0.05, 0.08, 0.10]
results_vs_width = []

for width in grid_widths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = width > 0
    cfg["pixel"]["layers"]["color_filter"]["grid"]["width"] = width
    r = SingleRunner.run(cfg)
    results_vs_width.append(r)
    print(f"Grid width {width*1000:.0f} nm: done")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=results_vs_width,
    labels=[f"w={w*1000:.0f}nm" for w in grid_widths],
    ax=ax,
)
ax.set_title("QE vs Metal Grid Width")
plt.tight_layout()
plt.savefig("grid_width_sweep.png", dpi=150)
```
