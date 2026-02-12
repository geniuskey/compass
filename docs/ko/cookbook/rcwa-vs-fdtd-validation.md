# RCWA vs FDTD 교차 솔버 검증

이 레시피에서는 BSI CMOS 픽셀에 대해 RCWA (grcwa) 결과를 FDTD (flaport)와 교차 검증하고, 직광(직접 조명)과 원뿔 조명을 비교하는 방법을 보여줍니다.

## 인터랙티브 차트

<RcwaFdtdValidation />

## 교차 솔버 검증이 필요한 이유

RCWA와 FDTD는 근본적으로 다른 접근 방식으로 맥스웰 방정식을 풉니다:

| | RCWA | FDTD |
|---|---|---|
| **영역** | 주파수 영역 | 시간 영역 |
| **주기성** | 본질적으로 주기적 | PML 경계 필요 |
| **장점** | 박막 스택에 빠름 | 임의 기하 처리 가능 |
| **수렴** | 푸리에 차수 | 격자 간격 + 실행 시간 |

두 방법이 흡수/반사/투과 스펙트럼에서 일치하면, 시뮬레이션의 물리적 타당성에 대한 강한 확신을 제공합니다.

## 검증 실행

```bash
PYTHONPATH=. python3.11 scripts/validate_rcwa_vs_fdtd.py
```

이 스크립트는 세 가지 실험을 수행합니다:

### 실험 1: 수직 입사 스윕

grcwa (fourier_order=[3,3]) vs fdtd_flaport (dx=0.02um, 300fs)을 400-700nm 범위에서 비교합니다.

**허용 기준:** 최대 |A_grcwa - A_fdtd| < 10%

### 실험 2: 원뿔 조명

grcwa를 사용하여 세 가지 조명 조건을 비교합니다:
- **직광**: 수직 입사 (θ=0°)
- **원뿔 F/2.0 CRA=0°**: 피보나치 19포인트 샘플링, 코사인 가중치
- **원뿔 F/2.0 CRA=15°**: 동일 원뿔에 15° 주광선 각도(CRA)

### 실험 3: RCWA 교차 검증

동일한 원뿔 조명에서 grcwa vs torcwa를 검증하여 RCWA 솔버 일관성을 확인합니다.

**허용 기준:** 최대 |A_grcwa - A_torcwa| < 5%

## ConeIlluminationRunner 사용법

```python
from compass.runners.cone_runner import ConeIlluminationRunner

config = {
    "pixel": { ... },  # 픽셀 스택 설정
    "solver": {"name": "grcwa", "type": "rcwa", "params": {"fourier_order": [5, 5]}},
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
        "cone": {
            "cra_deg": 0.0,       # 주광선 각도
            "f_number": 2.0,       # 렌즈 F-넘버
            "sampling": {"type": "fibonacci", "n_points": 19},
            "weighting": "cosine",
        },
    },
    "compute": {"backend": "cpu"},
}

result = ConeIlluminationRunner.run(config)
```

러너 동작:
1. `ConeIllumination`에서 각도 샘플링 포인트 생성
2. 각 각도마다: (θ, φ)로 소스를 설정하고 솔버 실행
3. 가중 R, T, A 및 픽셀별 QE 누적
4. 가중 평균된 단일 `SimulationResult` 반환

## 수렴 팁

RCWA와 FDTD 결과가 벌어질 경우:

- **FDTD**: `runtime` 증가 (300→500fs), `grid_spacing` 감소 (0.02→0.01um), `pml_layers` 증가 (15→25)
- **RCWA**: `fourier_order` 증가 ([3,3]→[5,5]→[9,9])

FDTD의 2패스 참조 정규화는 정확한 R/T 추출에 필수적입니다 — 반사 검출기에서 전체 전기장에서 입사 전기장을 빼는 방식입니다.
