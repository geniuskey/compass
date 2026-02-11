# BSI 2x2 기본 시뮬레이션

표준 2x2 베이어(Bayer) BSI 픽셀 단위 셀(unit cell)을 시뮬레이션하고 양자 효율(QE, Quantum Efficiency) 스펙트럼을 계산하는 완전한 레시피입니다.

<StackVisualizer />

## 학습 내용

- BSI 픽셀 구성을 처음부터 설정하는 방법
- RCWA를 사용한 파장 스윕 실행
- 채널별 QE 스펙트럼 시각화
- 에너지 수지 검증

## 사전 요구사항

```bash
pip install -e ".[rcwa]"
```

## 구성 설정

설정 딕셔너리를 생성합니다 (또는 YAML에서 로드):

```python
config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True,
                "height": 0.6,
                "radius_x": 0.48,
                "radius_y": 0.48,
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                "shift": {"mode": "none"},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "pattern": "bayer_rggb",
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
                "thickness": 3.0,
                "material": "silicon",
                "photodiode": {"position": [0, 0, 0.5], "size": [0.7, 0.7, 2.0]},
                "dti": {"enabled": True, "width": 0.1, "material": "sio2"},
            },
        },
    },
    "solver": {
        "name": "torcwa",
        "type": "rcwa",
        "params": {"fourier_order": [9, 9], "dtype": "complex64"},
        "stability": {
            "precision_strategy": "mixed",
            "allow_tf32": False,
            "fourier_factorization": "li_inverse",
        },
    },
    "source": {
        "type": "planewave",
        "wavelength": {
            "mode": "sweep",
            "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01},
        },
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

## 시뮬레이션 실행

```python
from compass.runners.single_run import SingleRunner

result = SingleRunner.run(config)

print(f"Wavelengths: {result.wavelengths.shape[0]} points "
      f"({result.wavelengths[0]*1000:.0f}-{result.wavelengths[-1]*1000:.0f} nm)")
print(f"Pixels: {list(result.qe_per_pixel.keys())}")
```

## QE 스펙트럼 시각화

```python
from compass.visualization.qe_plot import plot_qe_spectrum
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
plot_qe_spectrum(result, ax=ax)
ax.set_title("BSI 2x2 Bayer (1 um pitch) - QE Spectrum")
plt.tight_layout()
plt.savefig("bsi_2x2_qe.png", dpi=150)
plt.show()
```

## 에너지 수지 확인

```python
from compass.analysis.energy_balance import EnergyBalance

check = EnergyBalance.check(result, tolerance=0.02)
print(f"Valid: {check['valid']}, max error: {check['max_error']:.4f}")
```

## 채널별 QE 추출

```python
from compass.analysis.qe_calculator import QECalculator

channel_qe = QECalculator.spectral_response(result.qe_per_pixel, result.wavelengths)

for color, (wl, qe) in channel_qe.items():
    peak_idx = qe.argmax()
    print(f"{color}: peak QE = {qe[peak_idx]:.1%} at {wl[peak_idx]*1000:.0f} nm")
```

예상 출력 (근사값):

```
B: peak QE = 62.3% at 460 nm
G: peak QE = 71.5% at 540 nm
R: peak QE = 59.8% at 620 nm
```

<QESpectrumChart />

## 변형 실험

### 더 작은 피치 (0.8 um)

```python
config["pixel"]["pitch"] = 0.8
config["pixel"]["layers"]["microlens"]["radius_x"] = 0.38
config["pixel"]["layers"]["microlens"]["radius_y"] = 0.38
config["pixel"]["layers"]["silicon"]["photodiode"]["size"] = [0.55, 0.55, 1.6]
```

### 더 두꺼운 실리콘 (4 um)

```python
config["pixel"]["layers"]["silicon"]["thickness"] = 4.0
config["pixel"]["layers"]["silicon"]["dti"]["depth"] = 4.0
config["pixel"]["layers"]["silicon"]["photodiode"]["size"] = [0.7, 0.7, 3.0]
```

### 마이크로렌즈 없음

```python
config["pixel"]["layers"]["microlens"]["enabled"] = False
```

마이크로렌즈 유무에 따른 QE를 비교하여 마이크로렌즈의 기여도를 정량화할 수 있습니다.
