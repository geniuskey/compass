# CRA 시프트 분석

이 레시피는 주광선 각도(CRA, Chief Ray Angle)를 0도에서 30도까지 스윕하고, 마이크로렌즈 시프트 보상 유무에 따른 QE를 비교하며, 시프트가 픽셀 성능에 미치는 영향을 정량화합니다.

## 배경

센서 가장자리에서는 주광선이 비스듬한 각도로 도달합니다. 마이크로렌즈 시프트가 없으면 집속된 스폿(spot)이 포토다이오드를 벗어나 QE 손실과 크로스토크를 유발합니다. 최신 이미지 센서는 이를 보상하기 위해 마이크로렌즈를 광축 방향으로 오프셋합니다. 이 레시피는 그 효과를 정량화합니다.

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

## CRA 스윕: 시프트 없음 vs 자동 시프트

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

## CRA 응답 시각화

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

## 채널별 CRA 응답

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

## 스펙트럼 CRA 분석

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

## 예상 결과

- **0--10도 CRA**: ML 시프트를 적용하면 QE가 수직 입사 대비 5% 이내로 유지됩니다.
- **10--20도 CRA**: QE가 감소하기 시작합니다. ML 시프트는 시프트 미적용 대비 5--10%의 절대 QE를 회복합니다.
- **20--30도 CRA**: 시프트를 적용하더라도 상당한 QE 손실이 발생합니다. 짧은 파장(청색)이 가장 많이 영향을 받는데, 이는 작은 흡수 깊이로 인해 초점 오프셋에 민감하기 때문입니다.
- **청색 채널**은 CRA에 가장 민감합니다. 표면 근처에서 흡수가 이루어지므로 마이크로렌즈 집속 정확도가 가장 중요하기 때문입니다.
- **적색 채널**은 가장 덜 민감합니다. 광자가 집속 품질과 관계없이 실리콘 깊숙이 침투하기 때문입니다.

## 시프트 테이블에 대한 실용적 참고사항

실제 센서 설계에서는 렌즈-센서 공동 설계 과정에서 보정된, 이미지 높이에 대한 마이크로렌즈 시프트의 다항식 또는 조회 테이블(lookup table)을 사용합니다. `auto_cra` 모드는 1차 근사(first-order approximation)를 제공합니다. 양산 설계에서는 `shift.table` 설정 옵션을 통해 사용자 정의 시프트 테이블을 제공할 수 있습니다.
