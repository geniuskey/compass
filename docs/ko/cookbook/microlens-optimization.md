# 마이크로렌즈 & CRA 최적화

이 레시피는 마이크로렌즈(microlens) 파라미터(높이, 반경, 직각도)를 스윕하여 QE를 최대화하고 CRA(Chief Ray Angle, 주광선 각도) 시프트 보상이 픽셀 성능에 미치는 영향을 연구하는 방법을 보여줍니다.

## 배경

마이크로렌즈는 BSI 픽셀에서 광학적으로 가장 중요한 요소입니다. 렌즈의 형상이 빛을 포토다이오드(photodiode)에 얼마나 잘 집속시키는지를 결정합니다. 주요 파라미터는 다음과 같습니다:

- **높이** ($H$): 집속력을 제어합니다. 너무 높으면 과집속(over-focusing)이, 너무 낮으면 부족 집속(under-focusing)이 발생합니다.
- **반경** ($r_x, r_y$): 렌즈 개구를 결정합니다. 피치의 절반보다 약간 작아야 합니다.
- **직각도** ($n$): 초타원(superellipse) 파라미터입니다. $n=2$는 타원이며, 값이 높을수록 더 상자형 렌즈가 되어 충진율이 향상됩니다.
- **CRA 시프트**: 센서 가장자리에 위치한 픽셀의 경우, 비스듬한 각도로 입사하는 빛을 받아들이기 위해 마이크로렌즈를 이동시켜야 합니다. 마이크로렌즈 시프트가 없으면 집속된 스폿(spot)이 포토다이오드를 벗어나 QE 손실과 크로스토크를 유발합니다. 최신 이미지 센서는 이를 보상하기 위해 마이크로렌즈를 광축 방향으로 오프셋합니다.

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

<StaircaseMicrolensViewer />

<PixelStackBuilder />

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

## 스윕 3: CRA 시프트 최적화

CRA(Chief Ray Angle, 주광선 각도)는 센서 가장자리의 픽셀에 입사하는 빛이 광축에서 얼마나 벗어나는지를 나타냅니다. CRA가 증가하면 집속된 스폿이 포토다이오드 중심에서 벗어나 QE가 감소하고 광학적 크로스토크가 증가합니다. 마이크로렌즈를 광축 방향으로 시프트하여 보상하는 것은 실제 센서 설계에서 가장 효과적인 최적화 중 하나입니다.

<ConeIlluminationViewer />

### 기본 CRA 스윕: 시프트 없음 vs 자동 시프트

각 CRA 각도에 대해 마이크로렌즈 시프트 없이 한 번, 자동 CRA 기반 시프트로 한 번, 총 두 번 실행합니다:

```python
cra_angles = np.arange(0, 31, 5)  # 0, 5, 10, ..., 30 degrees
qe_no_shift = []
qe_with_shift = []

for cra in cra_angles:
    # Without microlens shift
    cfg = copy.deepcopy(base_config)
    cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "none"
    result = SingleRunner.run(cfg)
    avg_qe = np.mean([float(np.mean(qe)) for qe in result.qe_per_pixel.values()])
    qe_no_shift.append(avg_qe)

    # With auto CRA shift
    cfg2 = copy.deepcopy(cfg)
    cfg2["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg2["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
    result2 = SingleRunner.run(cfg2)
    avg_qe2 = np.mean([float(np.mean(qe)) for qe in result2.qe_per_pixel.values()])
    qe_with_shift.append(avg_qe2)

    print(f"CRA={cra:2d} deg: no shift QE={avg_qe:.3f}, with shift QE={avg_qe2:.3f}")
```

### CRA 응답 시각화

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

ax1.plot(cra_angles, qe_no_shift, "o-", label="No ML shift", linewidth=2, color="tab:red")
ax1.plot(cra_angles, qe_with_shift, "s-", label="Auto CRA shift", linewidth=2, color="tab:blue")
ax1.set_ylabel("Average QE at 550 nm")
ax1.set_title("CRA Response: Microlens Shift Compensation")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# QE improvement from shift
improvement = np.array(qe_with_shift) - np.array(qe_no_shift)
ax2.bar(cra_angles, improvement, width=3, color="tab:green", alpha=0.7)
ax2.set_xlabel("Chief Ray Angle (degrees)")
ax2.set_ylabel("QE improvement")
ax2.set_title("QE Gain from Microlens Shift")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cra_shift_analysis.png", dpi=150)
```

### 채널별 CRA 분석

각 색상 채널이 어떻게 다르게 영향을 받는지 확인합니다:

```python
channels = {"R": "red", "G": "green", "B": "blue"}
channel_qe = {ch: {"no_shift": [], "with_shift": []} for ch in channels}

for cra in cra_angles:
    for mode, shift_mode in [("no_shift", "none"), ("with_shift", "auto_cra")]:
        cfg = copy.deepcopy(base_config)
        cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
        cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = shift_mode
        if shift_mode == "auto_cra":
            cfg["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
        result = SingleRunner.run(cfg)

        for ch in channels:
            ch_qe = [float(np.mean(qe)) for name, qe in result.qe_per_pixel.items()
                      if name.startswith(ch)]
            channel_qe[ch][mode].append(np.mean(ch_qe) if ch_qe else 0.0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (ch, color) in zip(axes, channels.items()):
    ax.plot(cra_angles, channel_qe[ch]["no_shift"], "o--",
            label="No shift", color=color, alpha=0.6)
    ax.plot(cra_angles, channel_qe[ch]["with_shift"], "s-",
            label="With shift", color=color)
    ax.set_xlabel("CRA (degrees)")
    ax.set_ylabel("QE at 550 nm")
    ax.set_title(f"{ch} Channel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.suptitle("Per-Channel CRA Response", fontsize=14)
plt.tight_layout()
plt.savefig("cra_per_channel.png", dpi=150)
```

### 스펙트럼 CRA 분석

완전한 그림을 위해 파장과 CRA를 동시에 스윕합니다:

```python
wavelengths = np.arange(0.40, 0.701, 0.02)  # 20 nm step for speed
cra_list = [0, 10, 20, 30]

fig, ax = plt.subplots(figsize=(10, 6))

for cra in cra_list:
    cfg = copy.deepcopy(base_config)
    cfg["source"] = {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": float(cra), "phi_deg": 0.0},
        "polarization": "unpolarized",
    }
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)

    result = SingleRunner.run(cfg)

    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items() if name.startswith("G")
    ], axis=0)
    ax.plot(result.wavelengths * 1000, green_qe, label=f"CRA={cra} deg", linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Green QE")
ax.set_title("Green QE Spectrum at Various CRA (with ML shift)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cra_spectral.png", dpi=150)
```

### 시프트 테이블에 대한 실용적 참고사항

실제 센서 설계에서는 렌즈-센서 공동 설계 과정에서 보정된, 이미지 높이에 대한 마이크로렌즈 시프트의 다항식 또는 조회 테이블(lookup table)을 사용합니다. `auto_cra` 모드는 Hwang & Kim, "A Numerical Method of Aligning the Optical Stacks for All Pixels," *Sensors*, vol. 23, no. 2, 702, 2023 (DOI: [10.3390/s23020702](https://doi.org/10.3390/s23020702))에서 제안한 방법에 따라, 모든 중간 레이어(평탄화층, 컬러 필터, BARL, 실리콘)를 통해 스넬 법칙 광선 추적으로 시프트를 계산합니다. 양산 설계에서는 `shift.table` 설정 옵션을 통해 사용자 정의 시프트 테이블을 제공할 수 있습니다.

## 스윕 4: 2D 최적화

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

## 요약 & 권장사항

- **높이**: QE는 최적 높이(1 um 피치에서 일반적으로 0.5-0.7 um)에서 최대값을 나타냅니다. 너무 높으면 과집속이, 너무 낮으면 약한 집속이 발생합니다.
- **직각도**: 높은 $n$ 값(더 사각형 렌즈)은 일반적으로 충진율 향상으로 QE를 개선하지만, 매우 높은 값은 가장자리 회절을 유발할 수 있습니다.
- **CRA 0--10도**: ML 시프트를 적용하면 QE가 수직 입사 대비 5% 이내로 유지됩니다.
- **CRA 10--20도**: QE가 감소하기 시작합니다. ML 시프트는 시프트 미적용 대비 5--10%의 절대 QE를 회복합니다.
- **CRA 20--30도**: 시프트를 적용하더라도 상당한 QE 손실이 발생합니다. 짧은 파장(청색)이 가장 많이 영향을 받는데, 이는 작은 흡수 깊이로 인해 초점 오프셋에 민감하기 때문입니다.
- **청색 채널**은 CRA에 가장 민감합니다. 표면 근처에서 흡수가 이루어지므로 마이크로렌즈 집속 정확도가 가장 중요하기 때문입니다.
- **적색 채널**은 가장 덜 민감합니다. 광자가 집속 품질과 관계없이 실리콘 깊숙이 침투하기 때문입니다.
- **CRA 시프트**는 센서 가장자리 픽셀에 대한 가장 효과적인 단일 보상 방법입니다. 자동 CRA 시프트를 적용하면 렌즈가 보상하여 15-20도까지 QE를 유지한 후 감소합니다.
