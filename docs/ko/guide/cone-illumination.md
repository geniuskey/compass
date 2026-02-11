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

## 상면도: 픽셀 어레이 위의 풋프린트

<ConeIlluminationTopView />

위의 측면도는 원뿔 기하학을 단면으로 보여줍니다. **상면도(Top View)**는 보완적인 관점을 제공합니다: 픽셀 어레이를 위에서 내려다보면, 조명 원뿔이 2x2 Bayer 패턴에 어떻게 투영되는지 확인할 수 있습니다.

상면도에서의 핵심 관찰 사항:

- **풋프린트 직경**: 초점면에서의 원뿔 풋프린트 직경은 $d = 2 h \tan(\theta_{\text{half}})$이며, 여기서 $h$는 픽셀 스택 높이입니다. F 넘버가 낮을수록 더 넓은 풋프린트가 생성됩니다.
- **CRA 시프트**: CRA가 0이 아닌 경우 풋프린트 중심이 픽셀 중심에서 벗어납니다. 일반적인 5 um 스택에서 CRA = 20°일 때, 시프트는 1.5 um을 초과할 수 있으며 이는 픽셀 피치와 비슷한 수준입니다.
- **샘플링 커버리지**: 위의 인터랙티브 뷰어에서 피보나치와 격자 샘플링 포인트가 풋프린트에 어떻게 분포하는지 확인할 수 있습니다. 피보나치 샘플링이 더 균일한 각도 커버리지를 제공합니다.
- **렌즈 면적**: 풋프린트 면적 $A = \pi r^2$ (여기서 $r = h \tan(\theta_{\text{half}})$)는 인접 픽셀이 원뿔로부터 얼마나 많은 빛을 받는지를 결정하며, 이는 크로스토크에 직접적으로 영향을 미칩니다.

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

## 렌즈 면적 스위프: F 넘버 vs QE 및 크로스토크

조명 원뿔의 풋프린트 면적은 F 넘버에 따라 변합니다. F 넘버를 스위프하면 렌즈 속도가 QE와 광학적 크로스토크에 미치는 영향을 파악할 수 있으며, 이는 CIS 설계에서 핵심적인 트레이드오프입니다.

### 풋프린트 면적 vs F 넘버

풋프린트 반경 $r = h \tan(\theta_{\text{half}})$이고 $\theta_{\text{half}} = \arcsin(1/2F)$입니다. 따라서 풋프린트 면적 $A = \pi r^2$는 F 넘버가 감소할수록 급격히 증가합니다:

| F 넘버 | $\theta_{\text{half}}$ (도) | 풋프린트 반경 (um) | 면적 (um²) |
|--------|------|----------|--------|
| F/1.4  | 20.9 | 1.91     | 11.5   |
| F/2.0  | 14.5 | 1.29     | 5.3    |
| F/2.8  | 10.3 | 0.91     | 2.6    |
| F/4.0  | 7.2  | 0.63     | 1.2    |
| F/5.6  | 5.1  | 0.45     | 0.63   |

*(스택 높이 h = 5.0 um 기준)*

### F 넘버 스위프 실행

```python
import numpy as np
from compass.sources.cone_illumination import ConeIllumination
from compass.solvers.base import SolverFactory

f_numbers = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
wavelength = 0.55
cra_deg = 15.0

solver = SolverFactory.create("torcwa", solver_config, device="cuda")
solver.setup_geometry(pixel_stack)

results = {}
for fn in f_numbers:
    cone = ConeIllumination(cra_deg=cra_deg, f_number=fn, n_points=37)
    points = cone.get_sampling_points()

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

    results[fn] = weighted_qe
    print(f"F/{fn}: {weighted_qe}")
```

### 트레이드오프 분석

빠른 렌즈(낮은 F 넘버)는 더 많은 빛을 수집하여 신호를 개선합니다. 그러나 넓어진 원뿔은 인접 픽셀 간의 각도 확산과 크로스토크도 증가시킵니다:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 녹색 픽셀의 QE vs F 넘버
green_qe = [results[fn].get("green_tl", 0) for fn in f_numbers]
ax1.plot(f_numbers, green_qe, "go-", linewidth=2, markersize=8)
ax1.set_xlabel("F-number")
ax1.set_ylabel("QE (green pixel)")
ax1.set_title("QE vs F-number (550 nm)")
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# 크로스토크: 비대상 픽셀 QE / 대상 픽셀 QE 비율
crosstalk = []
for fn in f_numbers:
    green = results[fn].get("green_tl", 1e-9)
    red = results[fn].get("red_tr", 0)
    xtalk = red / green * 100  # 백분율
    crosstalk.append(xtalk)

ax2.plot(f_numbers, crosstalk, "rs-", linewidth=2, markersize=8)
ax2.set_xlabel("F-number")
ax2.set_ylabel("Crosstalk (%)")
ax2.set_title("Green→Red crosstalk vs F-number")
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
```

### CRA + F 넘버 결합 스위프

완전한 렌즈 면적 감도 분석을 위해, CRA와 F 넘버를 모두 스위프하여 2D 맵을 구축합니다:

```python
cra_values = [0, 5, 10, 15, 20, 25, 30]
f_numbers = [1.4, 2.0, 2.8, 4.0]
wavelength = 0.55

qe_map = np.zeros((len(cra_values), len(f_numbers)))

for i, cra in enumerate(cra_values):
    for j, fn in enumerate(f_numbers):
        cone = ConeIllumination(cra_deg=cra, f_number=fn, n_points=37)
        points = cone.get_sampling_points()

        weighted_qe = 0.0
        for theta_deg, phi_deg, weight in points:
            solver.setup_source({
                "wavelength": wavelength,
                "theta": float(theta_deg),
                "phi": float(phi_deg),
                "polarization": "unpolarized",
            })
            result = solver.run()
            weighted_qe += weight * float(
                np.mean(result.qe_per_pixel.get("green_tl", [0]))
            )

        qe_map[i, j] = weighted_qe

# 히트맵으로 시각화
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(qe_map, aspect="auto", origin="lower",
               extent=[f_numbers[0], f_numbers[-1],
                       cra_values[0], cra_values[-1]])
ax.set_xlabel("F-number")
ax.set_ylabel("CRA (degrees)")
ax.set_title("Green pixel QE: CRA vs F-number")
plt.colorbar(im, ax=ax, label="QE")
plt.tight_layout()
```

이 2D 스위프는 목표 QE 임계값에 대한 CRA 및 F 넘버 한계를 식별하는 데 도움이 되며, 이는 마이크로렌즈 설계 최적화에 필수적인 정보입니다.

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
