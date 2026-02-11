# 첫 번째 시뮬레이션

이 가이드는 COMPASS 시뮬레이션(Simulation)의 전체 과정을 처음부터 끝까지 안내합니다: 픽셀 설정 로드, torcwa RCWA 솔버(Solver)를 사용한 파장 스위프(Wavelength Sweep) 실행, QE 계산, 결과 저장.

## 개요

1.0 um 피치(Pitch)의 2x2 BSI(후면 조사, Back-Side Illuminated) 픽셀 단위 셀(Unit Cell)을 시뮬레이션하고, 수직 입사에서 400~700 nm 파장을 스위프한 후, 색상 채널별 QE 스펙트럼을 플롯합니다.

## 1단계: 설정 로드

```python
from pathlib import Path
from compass.core.config_schema import CompassConfig
from omegaconf import OmegaConf

# Load from YAML via Hydra/OmegaConf
yaml_path = Path("configs/pixel/default_bsi_1um.yaml")
raw = OmegaConf.load(yaml_path)

config = CompassConfig(**{
    "pixel": OmegaConf.to_container(raw["pixel"], resolve=True),
    "solver": {
        "name": "torcwa",
        "type": "rcwa",
        "params": {"fourier_order": [9, 9], "dtype": "complex64"},
    },
    "source": {
        "type": "planewave",
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
})
```

## 2단계: 지오메트리(Geometry) 구축

```python
from compass.materials.database import MaterialDB
from compass.geometry.builder import GeometryBuilder

mat_db = MaterialDB()
builder = GeometryBuilder(config.pixel, mat_db)
pixel_stack = builder.build()

print(f"Pixel: {config.pixel.pitch} um pitch, "
      f"{config.pixel.unit_cell[0]}x{config.pixel.unit_cell[1]} unit cell")
print(f"Stack: {pixel_stack.total_thickness:.2f} um total, "
      f"{len(pixel_stack.layers)} layers")
```

## 3단계: 구조 검증

솔버를 실행하기 전에 항상 지오메트리를 시각적으로 확인하십시오.

```python
from compass.visualization.structure_plot_2d import plot_pixel_cross_section

# XZ cross-section through pixel center
ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xz",
    position=config.pixel.pitch / 2,
    wavelength=0.55,
    figsize=(12, 6),
)
```

다음 사항을 확인합니다:
- 마이크로렌즈(Microlens)가 평탄화층(Planarization Layer) 위에 있는지
- 컬러 필터(Color Filter)가 예상된 베이어 패턴(Bayer Pattern) 색상을 보여주는지
- DTI(Deep Trench Isolation) 트렌치가 실리콘 레이어에서 보이는지
- 레이어 두께가 합리적인지

픽셀 스택 구조를 인터랙티브하게 탐색해 보십시오:

<StackVisualizer />

## 4단계: 솔버 생성

```python
from compass.solvers.base import SolverFactory

solver = SolverFactory.create("torcwa", config.solver)
print(f"Solver: {solver.name}, device: {solver.device}")
```

## 5단계: 파장 스위프 실행

```python
import numpy as np
from compass.sources.planewave import PlaneWave

wavelengths = np.arange(0.40, 0.701, 0.01)  # 400-700 nm in um
results = []

for wl in wavelengths:
    source_cfg = {
        "wavelength": wl,
        "theta": 0.0,
        "phi": 0.0,
        "polarization": "unpolarized",
    }

    solver.setup_geometry(pixel_stack)
    solver.setup_source(source_cfg)
    result = solver.run_timed()
    results.append(result)

    # Check energy balance
    is_valid = solver.validate_energy_balance(result, tolerance=0.01)
    if not is_valid:
        print(f"  WARNING: energy violation at {wl*1000:.0f} nm")

print(f"Completed {len(results)} wavelength points")
```

## 6단계: 픽셀별 QE 추출

```python
qe_data = {}  # pixel_name -> qe_array

for pixel_name in results[0].qe_per_pixel.keys():
    qe_spectrum = np.array([r.qe_per_pixel[pixel_name] for r in results])
    qe_data[pixel_name] = qe_spectrum.flatten()

print("Pixels found:", list(qe_data.keys()))
# Example: ['R_0_0', 'G_0_1', 'G_1_0', 'B_1_1']
```

## 7단계: QE 스펙트럼 플롯

```python
import matplotlib.pyplot as plt

wavelengths_nm = wavelengths * 1000  # Convert um to nm

fig, ax = plt.subplots(figsize=(10, 6))

channel_colors = {"R": "red", "G": "green", "B": "blue"}
channel_qe = {"R": [], "G": [], "B": []}

for pixel_name, qe in qe_data.items():
    channel = pixel_name[0]  # First character: R, G, or B
    channel_qe[channel].append(qe)
    ax.plot(wavelengths_nm, qe, color=channel_colors[channel],
            alpha=0.4, linewidth=0.8)

# Plot channel averages
for channel, qe_list in channel_qe.items():
    if qe_list:
        mean_qe = np.mean(qe_list, axis=0)
        ax.plot(wavelengths_nm, mean_qe, color=channel_colors[channel],
                linewidth=2.5, label=f"{channel} (mean)")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Quantum Efficiency")
ax.set_title("BSI 1.0um Pixel QE Spectrum (torcwa, normal incidence)")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("qe_spectrum.png", dpi=150)
```

아래에서 QE 스펙트럼을 인터랙티브하게 확인할 수 있습니다. 곡선 위에 마우스를 올려 각 파장에서의 채널별 QE를 확인하십시오:

<QESpectrumChart />

## 8단계: 에너지 균형(Energy Balance) 확인

```python
R_values = np.array([r.reflection.mean() if r.reflection is not None else 0
                      for r in results])
T_values = np.array([r.transmission.mean() if r.transmission is not None else 0
                      for r in results])
A_values = np.array([r.absorption.mean() if r.absorption is not None else 0
                      for r in results])

energy_sum = R_values + T_values + A_values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

ax1.plot(wavelengths_nm, R_values, label="R (reflectance)", color="tab:blue")
ax1.plot(wavelengths_nm, T_values, label="T (transmittance)", color="tab:orange")
ax1.plot(wavelengths_nm, A_values, label="A (absorption)", color="tab:red")
ax1.set_ylabel("Fraction")
ax1.set_title("Energy Balance: R + T + A")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(wavelengths_nm, energy_sum - 1.0, color="black")
ax2.axhline(0, color="gray", linestyle="--")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("R+T+A - 1")
ax2.set_title("Energy Conservation Error")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
```

오차는 모든 파장에서 1% (0.01) 미만이어야 합니다. 이를 초과하는 경우 [문제 해결](./troubleshooting.md)을 참조하십시오.

아래 인터랙티브 다이어그램은 반사, 투과, 흡수가 에너지 보존과 어떻게 관련되는지를 보여줍니다:

<EnergyBalanceDiagram />

## 9단계: 결과 저장

```python
from compass.io.hdf5_handler import save_results_hdf5

save_results_hdf5(
    results,
    wavelengths=wavelengths,
    pixel_stack=pixel_stack,
    filepath="outputs/bsi_1um_torcwa_qe.h5",
    metadata={
        "solver": "torcwa",
        "fourier_order": [9, 9],
        "polarization": "unpolarized",
        "angle_theta": 0.0,
    },
)
print("Results saved to outputs/bsi_1um_torcwa_qe.h5")
```

## 스크립트 러너 사용

프로덕션 실행에는 설정 로드, 스위프, 출력을 자동으로 처리하는 명령줄 러너를 사용합니다:

```bash
# Default simulation
python scripts/run_simulation.py

# Override solver and pixel config
python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um

# Wavelength sweep
python scripts/run_simulation.py \
    source.wavelength.mode=sweep \
    source.wavelength.sweep.start=0.40 \
    source.wavelength.sweep.stop=0.70 \
    source.wavelength.sweep.step=0.01
```

## 다음 단계

- [픽셀 스택 설정](./pixel-stack-config.md) -- 다양한 픽셀 구조 설정
- [재료 데이터베이스](./material-database.md) -- 사용자 정의 재료 추가
- [솔버 선택](./choosing-solver.md) -- 솔버 옵션 비교
