---
title: 원뿔 조명
description: COMPASS에서 렌즈 사출 동공으로부터의 원뿔 조명 설정. CRA, F 넘버, 각도 샘플링 방법, 가중 함수, 평면파 솔버와의 통합을 포함합니다.
---

# 원뿔 조명(Cone Illumination)

실제 카메라 시스템에서 각 픽셀에 도달하는 빛은 단일 방향이 아닌 렌즈 사출 동공(Exit Pupil)의 전체 영역에서 옵니다. `ConeIllumination` 클래스는 조명 원뿔을 가중 평면파(Planewave)로 분해하고 결과를 적분하여 이를 모델링합니다.

## 물리적 배경

<ConeIlluminationViewer />

조명 원뿔은 다음으로 특성화됩니다:

- **CRA(주광선 각도, Chief Ray Angle)**: 광축과 사출 동공에서 픽셀까지의 중심 광선 사이의 각도. CRA는 센서 중심에서 0이고 가장자리로 갈수록 증가합니다.
- **F 넘버(F-number)**: $\theta_{\text{half}} = \arcsin(1 / 2F)$를 통해 조명의 반원뿔 각도를 결정합니다. F/2.0 렌즈는 약 14.5도의 반원뿔을 제공합니다.
- **가중(Weighting)**: 원뿔 전체에 걸친 강도 분포. 원뿔 가장자리 근처의 빛은 일반적으로 중심보다 약합니다(예: 코사인 또는 cos^4 가중).

원뿔 조명 결과는 원뿔 내 샘플링된 각도에서 여러 평면파 시뮬레이션(Simulation)을 실행한 다음 QE의 가중 평균을 계산하여 얻습니다.

## ConeIllumination 인스턴스 생성

```python
from compass.sources.cone_illumination import ConeIllumination

cone = ConeIllumination(
    cra_deg=15.0,          # Chief Ray Angle in degrees
    f_number=2.0,          # F-number of the lens
    n_points=37,           # Number of angular sample points
    sampling="fibonacci",  # "fibonacci" or "grid"
    weighting="cosine",    # "uniform", "cosine", "cos4", or "gaussian"
)

print(f"Half-cone angle: {cone.half_cone_rad * 180 / 3.14159:.1f} degrees")
```

## 샘플링 방법

`get_sampling_points()` 메서드는 `(theta_deg, phi_deg, weight)` 튜플의 리스트를 반환합니다. 각 튜플은 평면파 방향과 관련 적분 가중치를 나타냅니다.

### 피보나치 샘플링(Fibonacci Sampling)

피보나치(황금각 나선) 샘플링은 원뿔 영역에 점을 준균일하게 분포시킵니다. 비교적 적은 점으로 좋은 커버리지를 제공합니다.

```python
cone = ConeIllumination(
    cra_deg=10.0, f_number=2.8, n_points=37, sampling="fibonacci"
)

points = cone.get_sampling_points()
print(f"Number of sample points: {len(points)}")
for i, (theta, phi, w) in enumerate(points[:5]):
    print(f"  Point {i}: theta={theta:.2f} deg, phi={phi:.2f} deg, weight={w:.4f}")
```

피보나치 샘플링은 대부분의 경우에 권장됩니다. 빠른 추정에는 19-37점, 프로덕션 결과에는 61-91점을 사용하십시오.

### 격자 샘플링(Grid Sampling)

격자 샘플링은 균일한 $(\theta, \phi)$ 격자를 사용합니다. 간단하지만 동일한 점 수에서 피보나치보다 효율이 떨어집니다.

```python
cone = ConeIllumination(
    cra_deg=10.0, f_number=2.8, n_points=36, sampling="grid"
)

points = cone.get_sampling_points()
print(f"Grid sampling: {len(points)} points")
```

격자 샘플링은 `n_theta x n_phi`개의 점을 생성하며, `n_theta = sqrt(n_points)`, `n_phi = n_points / n_theta`입니다.

## 가중 함수

가중 함수는 동공 전체에 걸친 강도 분포를 모델링합니다:

| 가중       | 수식                     | 물리적 모델                       |
|------------|------------------------|--------------------------------------|
| `uniform`  | $w = 1$               | 균일 동공 조명                        |
| `cosine`   | $w = \cos\theta$       | 람베르트 소스 / 아플라나틱 렌즈        |
| `cos4`     | $w = \cos^4\theta$     | 이미지 평면에서의 cos-fourth 감쇠     |
| `gaussian`  | $w = e^{-\theta^2/2\sigma^2}$ | 아포다이즈된 동공             |

기본값은 `cosine`이며, 대부분의 카메라 렌즈 시스템에 적합합니다.

```python
# Compare different weighting functions
for wf in ["uniform", "cosine", "cos4", "gaussian"]:
    cone = ConeIllumination(
        cra_deg=0.0, f_number=2.0, n_points=37, weighting=wf
    )
    points = cone.get_sampling_points()
    weights = [p[2] for p in points]
    print(f"{wf:10s}: max_w={max(weights):.4f}, min_w={min(weights):.4f}")
```

## 평면파 솔버와의 통합

원뿔 조명 QE를 계산하려면, 각 샘플링된 각도에서 평면파 시뮬레이션을 실행하고 가중합을 계산합니다:

```python
import numpy as np
from compass.sources.cone_illumination import ConeIllumination
from compass.solvers.base import SolverFactory

# Set up cone
cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=37, weighting="cosine")
points = cone.get_sampling_points()

# Create solver
solver = SolverFactory.create("torcwa", solver_config, device="cuda")
solver.setup_geometry(pixel_stack)

# Run planewave at each sample point
wavelength = 0.55
weighted_qe = {}

for theta_deg, phi_deg, weight in points:
    solver.setup_source({
        "wavelength": wavelength,
        "theta": float(theta_deg),
        "phi": float(phi_deg),
        "polarization": "unpolarized",
    })
    result = solver.run()

    for pixel_name, qe in result.qe_per_pixel.items():
        if pixel_name not in weighted_qe:
            weighted_qe[pixel_name] = 0.0
        weighted_qe[pixel_name] += weight * float(np.mean(qe))

print("Cone-illuminated QE at 550 nm:")
for pixel_name, qe in weighted_qe.items():
    print(f"  {pixel_name}: QE = {qe:.3f}")
```

## 원뿔 조명과 파장 스위프 결합

원뿔 조명에서의 전체 스펙트럼 스위프를 위해, 파장과 각도 샘플을 반복합니다:

```python
wavelengths = np.arange(0.40, 0.701, 0.01)
cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=37)
points = cone.get_sampling_points()

# Initialize QE storage
pixel_names = None
cone_qe = {}

for wl in wavelengths:
    wl_qe = {}

    for theta_deg, phi_deg, weight in points:
        solver.setup_source({
            "wavelength": float(wl),
            "theta": float(theta_deg),
            "phi": float(phi_deg),
            "polarization": "unpolarized",
        })
        result = solver.run()

        if pixel_names is None:
            pixel_names = list(result.qe_per_pixel.keys())
            for pn in pixel_names:
                cone_qe[pn] = []

        for pn in pixel_names:
            if pn not in wl_qe:
                wl_qe[pn] = 0.0
            wl_qe[pn] += weight * float(np.mean(result.qe_per_pixel[pn]))

    for pn in pixel_names:
        cone_qe[pn].append(wl_qe[pn])

# Plot results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
wl_nm = wavelengths * 1000
for pn in pixel_names:
    ax.plot(wl_nm, cone_qe[pn], label=pn)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("QE (cone illumination)")
ax.set_title(f"Cone Illumination QE (CRA={cone.cra_deg} deg, F/{cone.f_number})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

## 샘플링 수렴

`n_points`를 증가시키며 결과를 비교하여 샘플 점 수가 충분한지 확인합니다:

```python
for n in [7, 19, 37, 61, 91]:
    cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=n)
    points = cone.get_sampling_points()
    # ... run weighted sum, record QE
    print(f"n_points={n}: avg QE = ...")
```

일반적으로 F/2.0 이하에서 37점이면 완전히 수렴된 적분 대비 1% 이내의 결과를 제공합니다.

## YAML을 통한 설정

원뿔 조명은 소스 설정에서 지정할 수 있습니다:

```yaml
source:
  type: "cone"
  cone:
    cra_deg: 15.0
    f_number: 2.0
    n_points: 37
    sampling: "fibonacci"
    weighting: "cosine"
  wavelength:
    mode: "sweep"
    sweep: {start: 0.40, stop: 0.70, step: 0.01}
  polarization: "unpolarized"
```

## 다음 단계

- [ROI 스위프](./roi-sweep.md) -- 센서 전체에 걸친 원뿔 조명 스위프
- [CRA 시프트 분석 쿡북](../cookbook/cra-shift-analysis.md) -- CRA vs QE 연구
- [첫 번째 시뮬레이션](./first-simulation.md) -- 기본 평면파 설정
