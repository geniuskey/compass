# 솔버 벤치마크

이 레시피는 동일한 픽셀 구조에서 RCWA 솔버(solver)(torcwa, grcwa, meent)를 비교하여 정확도와 성능을 측정합니다.

## 목표

동일한 BSI 픽셀 시뮬레이션을 사용 가능한 모든 RCWA 솔버와 하나의 FDTD 솔버로 실행한 후, QE 결과와 실행 시간을 비교합니다.

## 설정

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

## 모든 RCWA 솔버 실행

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

## QE 스펙트럼 비교

```python
if len(results) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_qe_comparison(results, labels, ax=ax)
    ax.set_title("RCWA Solver Comparison")
    plt.tight_layout()
    plt.savefig("solver_comparison_qe.png", dpi=150)
    plt.show()
```

## 정량적 비교

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

<SolverComparisonChart />

## 푸리에 차수 수렴 비교

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

## 실행 시간 스케일링

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

## 예상 결과

- **QE 일치도**: 모든 RCWA 솔버는 동일한 푸리에 차수에서 0.5-1% QE 이내로 일치해야 합니다.
- **수렴성**: 모든 솔버는 차수가 증가함에 따라 동일한 QE로 수렴합니다. 수렴 속도는 약간 다를 수 있습니다.
- **실행 시간**: torcwa는 일반적으로 GPU에서 가장 빠릅니다. grcwa와 meent도 경쟁력 있는 성능을 보입니다.
- **FDTD 대 RCWA**: 1-2% QE 이내로 일치해야 합니다. FDTD는 단일 파장 실행에서 상당히 느립니다.
