# BARL 설계

이 레시피는 다층 하부 반사 방지층(BARL, Bottom Anti-Reflective Layer) 최적화를 보여주며, 4분의 1 파장 조건, 두께 스윕, BSI 픽셀용 광대역 ARC(Anti-Reflective Coating) 설계를 포함합니다.

## 배경

BARL은 컬러 필터와 실리콘 포토다이오드 사이에 위치합니다. 반사 방지 코팅이 없으면 공기/SiO2/실리콘 계면에서 입사광의 15--30%가 반사되어 QE를 크게 감소시킵니다. 잘 설계된 BARL 스택은 가시광선 전 영역에서 이 반사를 5% 이하로 줄여줍니다.

단층 ARC의 4분의 1 파장 조건은 다음과 같습니다:

$$n_{\text{ARC}} = \sqrt{n_1 \cdot n_2}, \quad t_{\text{ARC}} = \frac{\lambda_0}{4 \cdot n_{\text{ARC}}}$$

다층 스택의 경우, COMPASS를 이용한 수치 최적화가 해석적 설계 규칙을 대체합니다.

## 설정

```python
import numpy as np
import copy
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner

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
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                    {"thickness": 0.015, "material": "sio2"},
                    {"thickness": 0.030, "material": "si3n4"},
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
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

## 기준선: BARL 없음 vs BARL 포함

반사 방지층 유무에 따른 QE를 비교합니다:

```python
# With BARL (baseline config)
result_with = SingleRunner.run(base_config)

# Without BARL (remove BARL layers)
config_no_barl = copy.deepcopy(base_config)
config_no_barl["pixel"]["layers"]["barl"]["layers"] = []
result_without = SingleRunner.run(config_no_barl)

from compass.visualization.qe_plot import plot_qe_comparison

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=[result_with, result_without],
    labels=["With BARL", "No BARL"],
    ax=ax,
)
ax.set_title("BARL Effect on QE")
plt.tight_layout()
plt.savefig("barl_vs_no_barl.png", dpi=150)
```

<ThinFilmReflectance />

## 4분의 1 파장 단층 설계

550 nm(녹색 피크)에 최적화된 단층 ARC를 설계합니다:

```python
# Ideal ARC refractive index for SiO2 (n~1.46) to Si (n~4.08) interface
n_sio2 = 1.46
n_si = 4.08
n_ideal = np.sqrt(n_sio2 * n_si)
print(f"Ideal ARC index: {n_ideal:.2f}")  # ~2.44

# Quarter-wave thickness at 550 nm
wl_design = 0.55  # um
t_qw = wl_design / (4 * n_ideal)
print(f"Quarter-wave thickness: {t_qw*1000:.1f} nm")  # ~56 nm

# Si3N4 (n~2.0) is the closest standard material
t_si3n4_qw = wl_design / (4 * 2.0)
print(f"Si3N4 quarter-wave: {t_si3n4_qw*1000:.1f} nm")  # ~69 nm
```

이 단층 설계를 테스트합니다:

```python
config_single = copy.deepcopy(base_config)
config_single["pixel"]["layers"]["barl"]["layers"] = [
    {"thickness": 0.069, "material": "si3n4"},
]
result_single = SingleRunner.run(config_single)
```

## 단층 BARL 두께 스윕

Si3N4 두께를 스윕하여 최적값을 찾습니다:

```python
thicknesses = np.arange(0.020, 0.151, 0.005)  # 20 to 150 nm
avg_green_qe = []

for t in thicknesses:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = [
        {"thickness": float(t), "material": "si3n4"},
    ]
    result = SingleRunner.run(cfg)
    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items() if name.startswith("G")
    ], axis=0)
    # Average across wavelengths
    avg_green_qe.append(float(np.mean(green_qe)))

plt.figure(figsize=(8, 5))
plt.plot(thicknesses * 1000, avg_green_qe, "o-", linewidth=2)
plt.axvline(69, color="red", linestyle="--", alpha=0.5, label="Quarter-wave (69 nm)")
plt.xlabel("Si3N4 Thickness (nm)")
plt.ylabel("Average Green QE (400-700 nm)")
plt.title("Single-Layer BARL Thickness Sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_thickness_sweep.png", dpi=150)
```

## 다층 광대역 BARL 설계

단층은 하나의 파장에 최적화됩니다. 광대역 성능을 위해서는 고굴절률과 저굴절률 층을 교대로 사용합니다:

```python
# Two-layer: HfO2 (high-n) + SiO2 (low-n)
designs = {
    "2-layer HfO2/SiO2": [
        {"thickness": 0.025, "material": "hfo2"},
        {"thickness": 0.035, "material": "sio2"},
    ],
    "3-layer SiO2/HfO2/SiO2": [
        {"thickness": 0.010, "material": "sio2"},
        {"thickness": 0.030, "material": "hfo2"},
        {"thickness": 0.020, "material": "sio2"},
    ],
    "4-layer (baseline)": [
        {"thickness": 0.010, "material": "sio2"},
        {"thickness": 0.025, "material": "hfo2"},
        {"thickness": 0.015, "material": "sio2"},
        {"thickness": 0.030, "material": "si3n4"},
    ],
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, layers in designs.items():
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = layers
    result = SingleRunner.run(cfg)

    green_qe = np.mean([
        qe for pname, qe in result.qe_per_pixel.items() if pname.startswith("G")
    ], axis=0)

    total_t = sum(l["thickness"] for l in layers) * 1000
    ax.plot(result.wavelengths * 1000, green_qe,
            label=f"{name} ({total_t:.0f} nm total)", linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Green QE")
ax.set_title("Multi-Layer BARL Design Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_multilayer_comparison.png", dpi=150)
```

## 2층 스택에서의 HfO2 두께 스윕

SiO2를 15 nm로 고정하고 HfO2 두께를 스윕합니다:

```python
hfo2_thicknesses = np.arange(0.010, 0.061, 0.005)
sio2_thickness = 0.015

broadband_qe = []

for t_hfo2 in hfo2_thicknesses:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = [
        {"thickness": float(sio2_thickness), "material": "sio2"},
        {"thickness": float(t_hfo2), "material": "hfo2"},
    ]
    result = SingleRunner.run(cfg)

    # Broadband average QE across all channels
    all_qe = np.mean([qe for qe in result.qe_per_pixel.values()], axis=0)
    broadband_qe.append(float(np.mean(all_qe)))

plt.figure(figsize=(8, 5))
plt.plot(hfo2_thicknesses * 1000, broadband_qe, "s-", linewidth=2, color="tab:orange")
plt.xlabel("HfO2 Thickness (nm)")
plt.ylabel("Broadband Average QE")
plt.title("HfO2 Thickness Sweep (SiO2 = 15 nm)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_hfo2_sweep.png", dpi=150)
```

## 반사율 분석

BARL의 효과를 이해하기 위해 반사율 스펙트럼을 확인합니다:

```python
fig, ax = plt.subplots(figsize=(10, 5))

for name, layers in designs.items():
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = layers
    result = SingleRunner.run(cfg)
    if result.reflection is not None:
        ax.plot(result.wavelengths * 1000, result.reflection,
                label=name, linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("BARL Stack Reflectance Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_reflectance.png", dpi=150)
```

<EnergyBalanceDiagram />

## 설계 가이드라인

| 파라미터              | 가이드라인                                             |
|----------------------|-------------------------------------------------------|
| 층 수                | 가시광선 광대역 ARC에 2--4층                            |
| 물질 선택            | SiO2 (저굴절률), Si3N4 (중굴절률), HfO2/TiO2 (고굴절률) |
| 총 BARL 두께         | 일반적으로 40--120 nm                                   |
| 목표 반사율           | 400--700 nm 전 영역에서 5% 미만                         |
| 최적화 지표           | 모든 채널에 걸친 광대역 평균 QE                          |

양산 설계에서는 BARL 최적화를 마이크로렌즈 및 컬러 필터 최적화와 결합하여 통합 파라미터 스윕을 수행하십시오.
