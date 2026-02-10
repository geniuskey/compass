# DTI 크로스토크 분석

이 레시피는 심층 트렌치 격리(DTI, Deep Trench Isolation)가 인접 픽셀 간 광학적 크로스토크에 미치는 영향을 연구하는 방법을 보여주며, DTI 깊이 및 폭 스윕을 포함합니다.

## 배경

DTI는 픽셀 사이에 에칭된 좁은 트렌치로, 저굴절률 물질(일반적으로 SiO2)로 채워집니다. 전반사(total internal reflection)를 통한 광학 도파관 장벽 역할을 하여, 한 픽셀에서 생성된 광자가 인접 픽셀로 확산되는 것을 방지합니다. DTI는 소형 피치 픽셀(1.2 um 이하)에서 매우 중요하며, 이 없으면 광학적 및 전기적 크로스토크가 색 정확도를 저하시킵니다.

주요 DTI 파라미터:

- **폭**: 일반적으로 80--120 nm. 넓은 DTI는 격리 성능이 우수하지만 포토다이오드 면적이 줄어듭니다.
- **깊이**: 부분 깊이(1--2 um)부터 전체 깊이(실리콘 두께와 동일)까지 범위가 있습니다. 전체 깊이 DTI가 최상의 격리를 제공합니다.
- **물질**: 보통 SiO2 (n ~ 1.46). 실리콘(n ~ 4.0)과의 굴절률 대비가 강한 광학적 구속을 제공합니다.

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
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
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

참고: 좁은 DTI 트렌치를 정확하게 분해하기 위해 더 많은 고조파가 필요하므로 푸리에 차수를 [9, 9] 대신 [11, 11]로 사용합니다.

## DTI 포함 vs DTI 미포함

```python
# With DTI (baseline)
result_with = SingleRunner.run(base_config)

# Without DTI
config_no_dti = copy.deepcopy(base_config)
config_no_dti["pixel"]["layers"]["silicon"]["dti"]["enabled"] = False
result_without = SingleRunner.run(config_no_dti)
```

## 크로스토크 계산

크로스토크는 한 픽셀에 의도된 빛이 인접 픽셀에 흡수되는 비율입니다. 2x2 베이어 단위 셀에서 녹색 채널 빛이 적색 및 청색 픽셀로 누설되는 정도를 측정합니다:

```python
def compute_crosstalk(result):
    """Compute crosstalk matrix from QE per pixel."""
    pixel_names = sorted(result.qe_per_pixel.keys())
    n_pixels = len(pixel_names)
    wavelengths = result.wavelengths

    # Average QE per pixel across all wavelengths
    qe_avg = {}
    for name in pixel_names:
        qe_avg[name] = float(np.mean(result.qe_per_pixel[name]))

    # Total QE across all pixels
    total = sum(qe_avg.values())

    # Crosstalk: fraction of total QE absorbed by each pixel
    crosstalk = {name: qe_avg[name] / total for name in pixel_names}
    return crosstalk, qe_avg

xt_with, qe_with = compute_crosstalk(result_with)
xt_without, qe_without = compute_crosstalk(result_without)

print("With DTI:")
for name, xt in xt_with.items():
    print(f"  {name}: QE={qe_with[name]:.3f}, fraction={xt:.3f}")

print("\nWithout DTI:")
for name, xt in xt_without.items():
    print(f"  {name}: QE={qe_without[name]:.3f}, fraction={xt:.3f}")
```

## DTI 폭 스윕

DTI 트렌치 폭을 스윕하여 격리와 충진율 사이의 최적 균형을 찾습니다:

```python
widths = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
avg_qe_per_width = []
green_qe_per_width = []

for w in widths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["silicon"]["dti"]["enabled"] = w > 0
    cfg["pixel"]["layers"]["silicon"]["dti"]["width"] = w
    result = SingleRunner.run(cfg)

    # Average QE across all pixels
    all_qe = np.mean([np.mean(qe) for qe in result.qe_per_pixel.values()])
    avg_qe_per_width.append(float(all_qe))

    # Green channel only
    green_qe = np.mean([
        np.mean(qe) for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    green_qe_per_width.append(float(green_qe))

    print(f"DTI width={w*1000:.0f} nm: avg QE={all_qe:.3f}, green QE={green_qe:.3f}")
```

결과를 시각화합니다:

```python
fig, ax = plt.subplots(figsize=(8, 5))

widths_nm = [w * 1000 for w in widths]
ax.plot(widths_nm, avg_qe_per_width, "o-", label="All channels avg", linewidth=2)
ax.plot(widths_nm, green_qe_per_width, "s-", label="Green channel avg", linewidth=2)
ax.set_xlabel("DTI Width (nm)")
ax.set_ylabel("Average QE")
ax.set_title("QE vs DTI Width")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_width_sweep.png", dpi=150)
```

## DTI 깊이 스윕

부분 DTI의 경우, 깊이를 0(DTI 없음)부터 전체 실리콘 두께까지 스윕합니다:

```python
si_thickness = 3.0
depths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
qe_vs_depth = []

for d in depths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["silicon"]["dti"]["enabled"] = d > 0
    cfg["pixel"]["layers"]["silicon"]["dti"]["depth"] = d
    cfg["pixel"]["layers"]["silicon"]["dti"]["width"] = 0.10
    result = SingleRunner.run(cfg)

    all_qe = np.mean([np.mean(qe) for qe in result.qe_per_pixel.values()])
    qe_vs_depth.append(float(all_qe))
    print(f"DTI depth={d:.1f} um: avg QE={all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(depths, qe_vs_depth, "o-", linewidth=2, color="tab:purple")
plt.xlabel("DTI Depth (um)")
plt.ylabel("Average QE")
plt.title(f"QE vs DTI Depth (Si thickness = {si_thickness} um)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_depth_sweep.png", dpi=150)
```

## 스펙트럼 크로스토크 비교

DTI 유무에 따른 적색 픽셀의 파장별 QE를 시각화하여 크로스토크가 가장 심한 곳을 확인합니다:

```python
fig, ax = plt.subplots(figsize=(10, 6))

wl_nm = result_with.wavelengths * 1000

# Red pixel QE with and without DTI
for name, qe in result_with.qe_per_pixel.items():
    if name.startswith("R"):
        ax.plot(wl_nm, qe, "r-", linewidth=2, label=f"{name} with DTI")
        break

for name, qe in result_without.qe_per_pixel.items():
    if name.startswith("R"):
        ax.plot(wl_nm, qe, "r--", linewidth=2, alpha=0.6, label=f"{name} no DTI")
        break

# Blue pixel QE with and without DTI
for name, qe in result_with.qe_per_pixel.items():
    if name.startswith("B"):
        ax.plot(wl_nm, qe, "b-", linewidth=2, label=f"{name} with DTI")
        break

for name, qe in result_without.qe_per_pixel.items():
    if name.startswith("B"):
        ax.plot(wl_nm, qe, "b--", linewidth=2, alpha=0.6, label=f"{name} no DTI")
        break

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("QE")
ax.set_title("Spectral QE: DTI Effect on Red and Blue Channels")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_spectral_crosstalk.png", dpi=150)
```

## 크로스토크 히트맵

픽셀 간 에너지 분포를 시각화합니다:

```python
from compass.visualization.qe_plot import plot_crosstalk_heatmap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_crosstalk_heatmap(result_with, ax=ax1)
ax1.set_title("With DTI (100 nm)")

plot_crosstalk_heatmap(result_without, ax=ax2)
ax2.set_title("Without DTI")

plt.suptitle("Optical Crosstalk Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("dti_crosstalk_heatmap.png", dpi=150)
```

## 예상 관측 결과

1. **DTI 없음**: 특히 긴 파장(>600 nm)에서 상당한 광학적 크로스토크가 발생합니다. 광자가 실리콘 깊숙이 침투하여 흡수되기 전에 인접 픽셀로 측면 확산될 수 있기 때문입니다.
2. **전체 깊이 DTI 포함**: 크로스토크가 5--10배 감소합니다. SiO2 트렌치가 전반사를 통해 광자를 의도된 픽셀에 가둡니다.
3. **부분 DTI**: 중간 수준의 크로스토크 감소. DTI 깊이 아래에서 흡수된 광자는 여전히 넘어갈 수 있습니다.
4. **DTI 폭 트레이드오프**: 넓은 DTI는 격리를 개선하지만 픽셀당 활성 실리콘 면적을 줄여 피크 QE를 약간 낮춥니다. 1 um 피치에서의 최적점은 일반적으로 80--100 nm입니다.
5. **파장 의존성**: 짧은 파장(청색, 400--500 nm)은 표면 근처에서 흡수되므로 DTI의 영향을 덜 받습니다. 긴 파장(적색/NIR, 600--780 nm)은 더 깊이 침투하여 측면 확산 기회가 많으므로 DTI의 이점이 가장 큽니다.
