# 마이크로렌즈 최적화

이 레시피는 마이크로렌즈(microlens) 파라미터(높이, 반경, 직각도)를 스윕하여 QE를 최대화하고 CRA(Chief Ray Angle, 주광선 각도)의 영향을 연구하는 방법을 보여줍니다.

## 배경

마이크로렌즈는 BSI 픽셀에서 광학적으로 가장 중요한 요소입니다. 렌즈의 형상이 빛을 포토다이오드(photodiode)에 얼마나 잘 집속시키는지를 결정합니다. 주요 파라미터는 다음과 같습니다:

- **높이** ($H$): 집속력을 제어합니다. 너무 높으면 과집속(over-focusing)이, 너무 낮으면 부족 집속(under-focusing)이 발생합니다.
- **반경** ($r_x, r_y$): 렌즈 개구를 결정합니다. 피치의 절반보다 약간 작아야 합니다.
- **직각도** ($n$): 초타원(superellipse) 파라미터입니다. $n=2$는 타원이며, 값이 높을수록 더 상자형 렌즈가 되어 충진율이 향상됩니다.
- **CRA 시프트**: 센서 가장자리에 위치한 픽셀의 경우, 비스듬한 각도로 입사하는 빛을 받아들이기 위해 마이크로렌즈를 이동시켜야 합니다.

## 설정

```python
import numpy as np
import copy
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner
from compass.analysis.qe_calculator import QECalculator

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
    },
    "solver": {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

## 스윕 1: 마이크로렌즈 높이

```python
heights = np.arange(0.2, 1.01, 0.1)
avg_qe_vs_height = []

for h in heights:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["microlens"]["height"] = float(h)
    result = SingleRunner.run(cfg)

    # Average QE across all pixels
    all_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
    avg_qe_vs_height.append(all_qe)
    print(f"  height={h:.1f} um -> avg QE = {all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(heights, avg_qe_vs_height, "o-", linewidth=2)
plt.xlabel("Microlens Height (um)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs Microlens Height")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_height_sweep.png", dpi=150)
```

## 스윕 2: 직각도 파라미터

```python
n_values = [2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
avg_qe_vs_n = []

for n in n_values:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["microlens"]["profile"]["n"] = n
    result = SingleRunner.run(cfg)

    all_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
    avg_qe_vs_n.append(all_qe)
    print(f"  n={n:.1f} -> avg QE = {all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(n_values, avg_qe_vs_n, "s-", linewidth=2)
plt.xlabel("Superellipse Squareness (n)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs Microlens Squareness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_squareness_sweep.png", dpi=150)
```

## 스윕 3: CRA 응답

주광선 각도가 증가함에 따라 QE가 어떻게 저하되는지 연구합니다:

```python
cra_angles = [0, 5, 10, 15, 20, 25, 30]
results_no_shift = []
results_with_shift = []

for cra in cra_angles:
    # Without microlens shift
    cfg = copy.deepcopy(base_config)
    cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "none"
    r = SingleRunner.run(cfg)
    results_no_shift.append(r)

    # With auto CRA shift
    cfg2 = copy.deepcopy(cfg)
    cfg2["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg2["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
    r2 = SingleRunner.run(cfg2)
    results_with_shift.append(r2)
    print(f"  CRA={cra}deg: done")

# Plot
qe_no_shift = [np.mean([qe[0] for qe in r.qe_per_pixel.values()])
                for r in results_no_shift]
qe_with_shift = [np.mean([qe[0] for qe in r.qe_per_pixel.values()])
                  for r in results_with_shift]

plt.figure(figsize=(8, 5))
plt.plot(cra_angles, qe_no_shift, "o-", label="No ML shift", linewidth=2)
plt.plot(cra_angles, qe_with_shift, "s-", label="Auto CRA shift", linewidth=2)
plt.xlabel("Chief Ray Angle (degrees)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs CRA: Effect of Microlens Shift")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_cra_sweep.png", dpi=150)
```

## 예상 결과

- **높이**: QE는 최적 높이(1 um 피치에서 일반적으로 0.5-0.7 um)에서 최대값을 나타냅니다. 너무 높으면 과집속이, 너무 낮으면 약한 집속이 발생합니다.
- **직각도**: 높은 $n$ 값(더 사각형 렌즈)은 일반적으로 충진율 향상으로 QE를 개선하지만, 매우 높은 값은 가장자리 회절을 유발할 수 있습니다.
- **CRA**: 마이크로렌즈 시프트 없이 높은 CRA에서 QE가 감소합니다. 자동 CRA 시프트를 적용하면 렌즈가 보상하여 15-20도까지 QE를 유지한 후 감소합니다.

## 2D 최적화

더 철저한 연구를 위해 높이와 반경을 동시에 스윕합니다:

```python
heights = np.arange(0.3, 0.9, 0.1)
radii = np.arange(0.35, 0.50, 0.02)
qe_map = np.zeros((len(heights), len(radii)))

for i, h in enumerate(heights):
    for j, r in enumerate(radii):
        cfg = copy.deepcopy(base_config)
        cfg["pixel"]["layers"]["microlens"]["height"] = float(h)
        cfg["pixel"]["layers"]["microlens"]["radius_x"] = float(r)
        cfg["pixel"]["layers"]["microlens"]["radius_y"] = float(r)
        result = SingleRunner.run(cfg)
        qe_map[i, j] = np.mean([qe[0] for qe in result.qe_per_pixel.values()])

plt.figure(figsize=(8, 6))
plt.pcolormesh(radii, heights, qe_map, shading="auto", cmap="viridis")
plt.colorbar(label="Average QE")
plt.xlabel("Microlens Radius (um)")
plt.ylabel("Microlens Height (um)")
plt.title("QE vs Height and Radius")
plt.tight_layout()
plt.savefig("ml_2d_optimization.png", dpi=150)
```
