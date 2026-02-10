---
title: ROI 스위프
description: COMPASS에서 센서 이미지 평면 전체에 걸친 관심 영역 스위프 실행. CRA vs 이미지 높이 곡선, 마이크로렌즈 시프트 맵, ROISweepRunner를 사용합니다.
---

# ROI 스위프(ROI Sweep)

실제 카메라에서는 센서 표면 전체에 걸쳐 광학 조건이 변합니다. 센서 중심의 픽셀은 거의 수직 입사(CRA가 0에 가까움)를 받고, 가장자리의 픽셀은 가파른 각도(CRA가 30도 이상)로 빛을 받습니다. `ROISweepRunner`는 여러 센서 위치에서의 시뮬레이션(Simulation)을 자동화하여 공간적으로 변하는 QE와 상대 조도(Relative Illumination)를 예측합니다.

## ROI 스위프 작동 방식

각 센서 위치(이미지 높이로 정의)에서 러너는 다음을 수행합니다:

1. 사용자가 제공한 CRA vs 이미지 높이 테이블에서 주광선 각도(Chief Ray Angle, CRA)를 보간합니다
2. 비스듬한 입사를 보상하기 위해 해당 마이크로렌즈(Microlens) 시프트를 적용합니다
3. 로컬 CRA에 맞게 소스 각도를 설정합니다
4. 해당 위치에서 전체 시뮬레이션을 실행합니다
5. 모든 위치의 QE 결과를 통합된 출력으로 수집합니다

## 기본 사용법

```python
from compass.runners.roi_sweep_runner import ROISweepRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {
                "enabled": True, "height": 0.6,
                "radius_x": 0.48, "radius_y": 0.48,
                "material": "polymer_n1p56",
                "shift": {"mode": "none"},
            },
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
    "solver": {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed"},
    },
    "source": {
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

roi_config = {
    "image_heights": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "cra_table": [
        {"image_height": 0.0, "cra_deg": 0.0},
        {"image_height": 0.2, "cra_deg": 5.0},
        {"image_height": 0.4, "cra_deg": 10.0},
        {"image_height": 0.6, "cra_deg": 17.0},
        {"image_height": 0.8, "cra_deg": 24.0},
        {"image_height": 1.0, "cra_deg": 30.0},
    ],
}

results = ROISweepRunner.run(config, roi_config)
```

## 출력 이해

출력은 위치 레이블을 `SimulationResult` 객체에 매핑하는 딕셔너리입니다:

```python
for key, result in results.items():
    avg_qe = sum(
        float(qe.mean()) for qe in result.qe_per_pixel.values()
    ) / len(result.qe_per_pixel)
    print(f"{key}: avg QE = {avg_qe:.3f}")

# Output:
# ih_0.00: avg QE = 0.712
# ih_0.20: avg QE = 0.698
# ih_0.40: avg QE = 0.671
# ih_0.60: avg QE = 0.623
# ih_0.80: avg QE = 0.558
# ih_1.00: avg QE = 0.481
```

## CRA 테이블 정의

CRA 테이블은 정규화된 이미지 높이(0.0 = 중심, 1.0 = 코너)를 주광선 각도(도)에 매핑합니다. 이 곡선은 카메라 렌즈 설계에 따라 달라집니다.

### 렌즈 설계 데이터로부터

Zemax, Code V 또는 기타 렌즈 설계 도구의 CRA 데이터가 있는 경우:

```python
roi_config = {
    "image_heights": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "cra_table": [
        {"image_height": 0.0, "cra_deg": 0.0},
        {"image_height": 0.1, "cra_deg": 2.5},
        {"image_height": 0.2, "cra_deg": 5.1},
        {"image_height": 0.3, "cra_deg": 7.8},
        {"image_height": 0.4, "cra_deg": 10.8},
        {"image_height": 0.5, "cra_deg": 14.2},
        {"image_height": 0.6, "cra_deg": 17.5},
        {"image_height": 0.7, "cra_deg": 20.8},
        {"image_height": 0.8, "cra_deg": 24.0},
        {"image_height": 0.9, "cra_deg": 27.1},
        {"image_height": 1.0, "cra_deg": 30.0},
    ],
}
```

### 이상화된 선형 CRA 곡선

렌즈 데이터 없이 빠른 연구를 위해 선형 근사가 일반적입니다:

```python
import numpy as np

max_cra_deg = 30.0
image_heights = np.linspace(0.0, 1.0, 11)
cra_table = [
    {"image_height": float(ih), "cra_deg": float(ih * max_cra_deg)}
    for ih in image_heights
]
```

### CRA 보간

러너는 선형 보간에 `numpy.interp`를 사용하므로, CRA 테이블이 `image_heights` 스위프 포인트와 정확히 일치할 필요가 없습니다. 밀집 CRA 테이블을 제공하고 더 성긴 위치에서 스위프할 수 있습니다:

```python
roi_config = {
    "image_heights": [0.0, 0.5, 1.0],   # Only 3 positions
    "cra_table": cra_table,               # Dense 11-point table
}
```

## 마이크로렌즈 시프트 맵

각 관심 영역(Region of Interest, ROI) 위치에서 러너는 자동으로 마이크로렌즈 시프트 모드를 `auto_cra`로 설정하고 보간된 CRA를 제공합니다. 마이크로렌즈는 집광된 빛을 포토다이오드에 다시 중심 정렬하기 위해 횡방향으로 시프트됩니다:

시프트 방향과 크기는 CRA에 따라 달라집니다. 이미지 높이 0.6, CRA = 17도인 픽셀의 경우, 마이크로렌즈는 대략 다음만큼 광축 방향으로 시프트됩니다:

$$\Delta x \approx d \times \tan(\text{CRA})$$

여기서 $d$는 마이크로렌즈에서 포토다이오드까지의 거리입니다.

이는 `shift.mode = "auto_cra"`일 때 COMPASS 지오메트리 빌더에 의해 자동으로 처리됩니다.

## QE vs 이미지 높이 플롯

```python
import numpy as np
import matplotlib.pyplot as plt

image_heights = roi_config["image_heights"]
channel_colors = {"R": "red", "G": "green", "B": "blue"}

fig, ax = plt.subplots(figsize=(10, 6))

# Extract average QE per channel at each position
for channel in ["R", "G", "B"]:
    qe_vs_ih = []
    for ih in image_heights:
        key = f"ih_{ih:.2f}"
        result = results[key]
        channel_qe = [
            float(np.mean(qe))
            for name, qe in result.qe_per_pixel.items()
            if name.startswith(channel)
        ]
        qe_vs_ih.append(np.mean(channel_qe) if channel_qe else 0.0)
    ax.plot(image_heights, qe_vs_ih, "o-", color=channel_colors[channel],
            label=f"{channel} channel", linewidth=2)

ax.set_xlabel("Image Height (normalized)")
ax.set_ylabel("Average QE at 550 nm")
ax.set_title("QE vs Image Height (ROI Sweep)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roi_qe_vs_image_height.png", dpi=150)
```

## 상대 조도 맵

상대 조도(Relative Illumination, RI)는 각 위치의 QE와 중심 QE의 비율입니다:

```python
center_key = "ih_0.00"
center_qe = np.mean([
    float(np.mean(qe)) for qe in results[center_key].qe_per_pixel.values()
])

ri_values = []
for ih in image_heights:
    key = f"ih_{ih:.2f}"
    pos_qe = np.mean([
        float(np.mean(qe)) for qe in results[key].qe_per_pixel.values()
    ])
    ri_values.append(pos_qe / center_qe)

plt.figure(figsize=(8, 5))
plt.plot(image_heights, ri_values, "o-", linewidth=2, color="navy")
plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Image Height (normalized)")
plt.ylabel("Relative Illumination")
plt.title("Relative Illumination vs Image Height")
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("relative_illumination.png", dpi=150)
```

## ROI 스위프와 파장 스위프 결합

전체 스펙트럼 ROI 분석을 위해 소스를 스위프 모드로 설정합니다:

```python
config["source"] = {
    "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
    "polarization": "unpolarized",
}

results = ROISweepRunner.run(config, roi_config)

# Now each result contains a full wavelength sweep
for key, result in results.items():
    print(f"{key}: {len(result.wavelengths)} wavelengths")
```

## 성능 고려사항

각 ROI 위치는 완전한 시뮬레이션을 실행하므로, 총 실행 시간은 이미지 높이 수에 선형적으로 비례합니다. 31점 파장 스위프와 11개 ROI 위치의 경우:

- RCWA (torcwa, 차수 [9,9]): ~110 s 총 소요 (11 x 10 s/스위프)
- FDTD (flaport, 20 nm): ~500 s 총 소요 (11 x 45 s/광대역 실행)

실행 시간을 줄이려면, 초기 탐색에는 적은 이미지 높이 점을 사용하고 관심 영역 주변에서만 밀도를 높이십시오.

## 다음 단계

- [원뿔 조명](./cone-illumination.md) -- 현실적 조명을 위한 각도 샘플링
- [CRA 시프트 분석 쿡북](../cookbook/cra-shift-analysis.md) -- 상세 CRA vs QE 연구
- [시각화](./visualization.md) -- 스위프 결과를 위한 플롯 도구
