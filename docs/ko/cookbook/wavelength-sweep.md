# 파장 스윕

이 레시피는 가시광선 전 영역 파장 스윕(wavelength sweep)과 스펙트럼 데이터를 분석하는 방법을 보여줍니다.

## 기본 스윕

가장 간단한 파장 스윕은 소스(source) 설정에서 `"sweep"` 모드를 사용합니다:

```python
from compass.runners.single_run import SingleRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {"enabled": True, "height": 0.6, "radius_x": 0.48, "radius_y": 0.48},
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
        "wavelength": {
            "mode": "sweep",
            "sweep": {"start": 0.38, "stop": 0.78, "step": 0.01},
        },
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
print(f"Sweep: {len(result.wavelengths)} wavelengths from "
      f"{result.wavelengths[0]*1000:.0f} to {result.wavelengths[-1]*1000:.0f} nm")
```

## 전체 가시광선 스펙트럼 시각화

```python
import matplotlib.pyplot as plt
import numpy as np
from compass.visualization.qe_plot import plot_qe_spectrum

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_spectrum(result, ax=ax)

# Add wavelength color bar on x-axis for reference
wl_nm = result.wavelengths * 1000
for i in range(len(wl_nm) - 1):
    color = wavelength_to_rgb(wl_nm[i])  # You'd need a helper for this
    ax.axvspan(wl_nm[i], wl_nm[i+1], alpha=0.03, color=color)

ax.set_title("Full Visible Spectrum QE")
plt.tight_layout()
plt.savefig("full_spectrum_qe.png", dpi=200)
```

## 반사율, 투과율, 흡수율

에너지 수지(energy balance) 구성 요소를 QE와 함께 시각화합니다:

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# QE
plot_qe_spectrum(result, ax=ax1)
ax1.set_xlabel("")

# R, T, A
wl_nm = result.wavelengths * 1000
if result.reflection is not None:
    ax2.plot(wl_nm, result.reflection, label="Reflection", color="tab:blue")
if result.transmission is not None:
    ax2.plot(wl_nm, result.transmission, label="Transmission", color="tab:orange")
if result.absorption is not None:
    ax2.plot(wl_nm, result.absorption, label="Absorption", color="tab:red")

ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Fraction")
ax2.set_title("Energy Balance Components")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qe_and_energy.png", dpi=150)
```

## 고해상도 스윕

박막 간섭 줄무늬(thin-film interference fringes)를 분해하려면 더 작은 스텝을 사용합니다:

```python
config_fine = config.copy()
config_fine["source"] = {
    "wavelength": {
        "mode": "sweep",
        "sweep": {"start": 0.50, "stop": 0.60, "step": 0.002},  # 2 nm step
    },
    "polarization": "unpolarized",
}

result_fine = SingleRunner.run(config_fine)
```

2 nm 스텝은 실리콘 캐비티의 패브리-페로(Fabry-Perot) 줄무늬를 분해합니다 (550 nm에서 3 um Si의 줄무늬 간격 ~12 nm).

## 파장 목록 모드

특정 파장(예: 레이저 라인 또는 LED 피크)에 대해:

```python
config_list = config.copy()
config_list["source"] = {
    "wavelength": {
        "mode": "list",
        "values": [0.405, 0.450, 0.525, 0.590, 0.625, 0.680, 0.780],
    },
    "polarization": "unpolarized",
}

result_list = SingleRunner.run(config_list)
```

## 서로 다른 실리콘 두께 비교

다양한 실리콘 두께에 대해 스윕을 실행하고 오버레이합니다:

```python
import copy

thicknesses = [2.0, 3.0, 4.0]
results_by_thickness = []

for t in thicknesses:
    cfg = copy.deepcopy(config)
    cfg["pixel"]["layers"]["silicon"]["thickness"] = t
    cfg["pixel"]["layers"]["silicon"]["dti"]["depth"] = t
    r = SingleRunner.run(cfg)
    results_by_thickness.append(r)

from compass.visualization.qe_plot import plot_qe_comparison

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=results_by_thickness,
    labels=[f"Si {t}um" for t in thicknesses],
    ax=ax,
)
ax.set_title("QE vs Silicon Thickness")
plt.tight_layout()
plt.savefig("si_thickness_comparison.png", dpi=150)
```

## 주요 관측 사항

| 파장 범위 | 관측 결과 |
|-----------|----------|
| 380-420 nm | 표면 재결합(surface recombination)과 얕은 흡수로 인한 낮은 QE |
| 420-500 nm | 청색 채널 피크. 짧은 흡수 깊이로 BARL 설계에 민감 |
| 500-580 nm | 녹색 채널 피크. 중간 흡수 깊이 |
| 580-650 nm | 적색 채널 피크. 긴 흡수 깊이로 더 두꺼운 Si에서 이점 |
| 650-780 nm | 흡수 깊이가 실리콘 두께를 초과하면서 QE 감소 |

::: tip
근적외선(NIR, Near-Infrared) 응용에서는 실리콘 두께를 5-6 um으로 늘리고 스윕을 1000 nm까지 확장하십시오. 780 nm 이상에서의 실리콘 물질 데이터를 추가해야 할 수 있습니다.
:::
