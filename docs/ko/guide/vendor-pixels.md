---
title: 벤더 픽셀 구조
description: 최근 3년(2022–2024) 삼성, 소니, 옴니비전의 플래그십 CMOS 이미지 센서 픽셀 구조와, 벤더가 공개하지 않는 내부 레이어 파라미터를 결정하기 위한 방법론을 정리합니다.
---

# 벤더 픽셀 구조

이 문서는 최근 3년간 주요 CIS 벤더가 공개한 픽셀 아키텍처를 정리하고, COMPASS 에서 바로 시뮬레이션할 수 있는 YAML 설정과, 비공개 파라미터를 채워 넣기 위한 결정 방법론을 함께 제공합니다.

## 왜 단순히 "벤더 설정"을 만드는 게 어려운가

벤더가 공개하는 정보는 *헤드라인* 수준입니다 — 픽셀 피치, 색필터 비닝, OCL 공유, 그리고 구조 혁신의 마케팅 명칭(2-Layer Transistor, TheiaCel, Tetra²pixel) 정도입니다. 정작 EM 시뮬레이션에 필요한 내부 수치는 거의 공개되지 않습니다:

- 마이크로렌즈 sag / 반경 / 굴절률
- 색필터, 평탄화, BARL 스택 두께
- DTI 트렌치 폭, 깊이, 채움 재료
- 포토다이오드 수평 footprint 와 z-extent

COMPASS 는 이 부분을 경험식 기반 스케일링 룰로 채워 넣습니다 (아래 [파라미터 결정 방법](#parameter-determination)).

## 지원하는 플래그십 센서

| 벤더 / 모델                              | 설정 파일                                | 피치     | CFA 패턴            | OCL 공유 | 주요 특징                                          |
|------------------------------------------|------------------------------------------|----------|---------------------|----------|----------------------------------------------------|
| Samsung **ISOCELL HP9** (2024)           | `samsung_hp9_0p56um.yaml`                | 0.56 µm | Tetra²pixel (4×4)   | 1        | 고굴절률 마이크로렌즈, F-DTI SiO₂                 |
| Samsung **GNJ-class 50 MP** (2024)       | `samsung_gnj_50mp_1p0um.yaml`            | 1.0 µm  | Quad Bayer (2×2)    | 1        | F-DTI SiO₂ 채움                                    |
| Sony **LYTIA LYT-900** (2024)            | `sony_lyt900_1p6um.yaml`                 | 1.6 µm  | 표준 Bayer          | 1        | 2-Layer Transistor (PD 부피 약 2배)               |
| Sony **2×2 OCL Quad** (IMX803-급)         | `sony_2x2ocl_quad_1p22um.yaml`           | 1.22 µm | Quad Bayer          | 2        | 같은 색 2×2 그룹마다 큰 렌즈 한 개                |
| OmniVision **OV50K40** (2024)            | `omnivision_ov50k40_1p2um.yaml`          | 1.2 µm  | Quad Bayer          | 2        | TheiaCel (LOFIC) + Quad PD 2×2 OCL                 |

Hydra 로 실행:

```bash
python scripts/run_simulation.py pixel=samsung_hp9_0p56um source=wavelength_sweep
python scripts/run_simulation.py pixel=sony_lyt900_1p6um solver=torcwa
python scripts/run_simulation.py pixel=omnivision_ov50k40_1p2um source=cone_illumination
```

### Samsung ISOCELL HP9 — 200 MP, 0.56 µm, Tetra²pixel

스마트폰 망원용 최초의 200 MP 센서. 모델링한 광학적 핵심 요소:

- **피치 0.56 µm**, 1/1.4″ 포맷.
- **Tetra²pixel** 색필터 (Hexadeca / 16-cell binning 으로도 발표): 4×4 같은 색 그룹, 8×8 슈퍼픽셀.
- **고굴절률 마이크로렌즈** ("새로운 재료" 발표). `MaterialDB` 의 `polymer_hri_n1p70` (TiO₂-doped polymer, n ≈ 1.70) 항목으로 모델링.
- **F-DTI SiO₂ 채움** (GNJ 세대에서 폴리실리콘 → 산화막으로 업그레이드되어 크로스토크 감소).

### Samsung GNJ-class 50 MP — 1.0 µm

일반적인 50 MP 메인 카메라용 픽셀 (GN3/GNJ 세대). Quad Bayer + 픽셀당 OCL + F-DTI SiO₂.

### Sony LYTIA LYT-900 — 1.6 µm, 2-Layer Transistor

소니의 첫 1-인치 50 MP 스마트폰 센서로, IEDM 2021 에서 발표된 **2-Layer Transistor (2LT)** 적층 CMOS 기술 적용. 포토다이오드와 픽셀 트랜지스터를 다른 기판층에 분리해 PD 면적과 깊이를 키웠고, 결과적으로 포화 신호 수준이 약 2배.

광학 모델: PD 가 픽셀 footprint 의 거의 전체를 차지하고 더 깊게 형성되도록 `silicon` 의 `photodiode.size` 를 키워 표현했습니다.

### Sony 2×2 OCL Quad Bayer (IMX803-급)

폰 메인카메라 센서로, **2×2 on-chip lens** 기술 적용: 같은 색 2×2 그룹을 하나의 큰 렌즈가 덮어 모든 픽셀에서 PDAF 가능. `microlens.sharing: 2` 로 표현해 4픽셀 클러스터마다 큰 렌즈 하나가 배치되도록 모델링.

### OmniVision OV50K40 — TheiaCel HDR

50 MP, 1.2 µm. **TheiaCel** 은 옴니비전의 LOFIC (Lateral Overflow Integration Capacitor) 구현 명칭입니다. 광학적으로는 LOFIC 캐패시터가 픽셀 내 실리콘 면적의 일부를 차지하므로 PD footprint 가 약간 작아집니다. Quad PD 가 추가로 2×2 OCL 을 얹습니다.

## 파라미터 결정 방법 {#parameter-determination}

벤더가 비공개로 두는 내부 파라미터를 결정하기 위한 세 가지 보완적 접근법을 제공합니다.

### 1. 경험식 스케일링 룰 (기본 동작)

`compass.geometry.derive_parameters` 는 공개된 ISSCC, IEDM, SPIE 픽셀 아키텍처 논문 (2018–2024) 과 sub-µm ~ 2 µm BSI 픽셀에 대한 TechInsights 단면 보고서에서 도출한 피치 기반 스케일링 룰을 사용합니다:

```python
from compass.geometry import derive_parameters, PixelStack

cfg = derive_parameters(vendor="samsung_hp9", cra_deg=20.0)
stack = PixelStack({"pixel": cfg})
```

현재 사용하는 룰:

| 항목                          | 디폴트 룰                                              |
|-------------------------------|--------------------------------------------------------|
| 마이크로렌즈 sag              | `min(0.95, 0.42·pitch + 0.20)` µm                      |
| 마이크로렌즈 반경 (semi-axis) | `(sharing·pitch − 2·gap) / 2` µm                       |
| 마이크로렌즈 갭               | `0.02 + 0.02·min(pitch, 2)` µm                         |
| 색필터 두께                   | sub-µm: `0.35 + 0.25·(pitch/0.7)` / 그 외: `min(0.90, 0.45·pitch + 0.15)` |
| 평탄화 두께                   | `0.20 + 0.10·min(pitch, 2)/2` µm                       |
| DTI 폭                        | `0.05 + 0.025·min(pitch, 2)` µm (공정 한계)            |
| Si epi 두께                   | `min(4.5, 1.4 + 1.5·pitch)` µm                         |
| PD footprint                  | `0.70·pitch` (2LT 는 `0.88·pitch`, LOFIC 는 `0.65·pitch`) |
| PD z-extent                   | `0.67·t_Si` (2LT 는 `0.85·t_Si`)                       |

`two_layer_transistor`, `lofic` 같은 아키텍처 플래그가 PD 형상을 조정합니다. 이 디폴트로 공개 TechInsights 단면을 ~30% 이내로 재현합니다.

### 2. 측정 QE / 크로스토크 캘리브레이션

색별 측정 QE(λ) 또는 측정 공간 크로스토크가 있으면 최적화 프레임워크로 미지수 파라미터를 fit 하세요:

```python
from compass.geometry import derive_parameters
from compass.optimization import (
    ParameterSpace, MicrolensHeight, BARLThicknesses, Optimizer
)

base = derive_parameters(vendor="samsung_gnj", cra_deg=0.0)

space = ParameterSpace([
    MicrolensHeight(min=0.4, max=0.8),
    BARLThicknesses(min=0.005, max=0.05, n_layers=4),
])

# 사용자 정의 목적함수: 측정 QE 와의 L2 거리
optimizer = Optimizer(method="L-BFGS-B", parameter_space=space)
best = optimizer.minimize(
    objective=lambda params: l2_to_measured(params, measured_qe),
    base_config=base,
)
```

실제 제품에 맞춰 파라미터를 결정해야 할 때 권장되는 워크플로우입니다.

### 3. 직접 측정 (TechInsights / SEM 단면)

다이샷 SEM 이 있다면 레이어 두께를 직접 측정한 뒤 `overrides` 로 휴리스틱을 덮어쓰세요:

```python
cfg = derive_parameters(
    vendor="samsung_hp9",
    overrides={
        "layers.silicon.thickness": 1.85,       # 측정 Si 두께
        "layers.color_filter.thickness": 0.42,  # 측정 CF 두께
        "layers.silicon.dti.width": 0.075,
    },
)
```

`overrides` 는 어느 깊이의 dotted key 든 받으며, 휴리스틱보다 우선합니다.

## 벤더 헤드라인 레퍼런스

벤더가 공개한 헤드라인 값은 `compass.geometry.VENDOR_HEADLINES` 에 저장되어 있고, `derive_parameters` 의 진입점으로 사용할 수 있습니다:

```python
>>> from compass.geometry import VENDOR_HEADLINES
>>> VENDOR_HEADLINES["samsung_hp9"]
{'pitch': 0.56, 'format': '1/1.4"', 'megapixels': 200,
 'cf_pattern': 'tetra2cell', 'ocl_sharing': 1,
 'microlens_material': 'polymer_hri_n1p70', 'dti_fill': 'sio2',
 'year': 2024}
```

## 출처

- Samsung Newsroom (2024.06), *Samsung Unveils Versatile Image Sensors for Superior Smartphone Photography* — HP9 발표.
- Samsung Semiconductor 제품 페이지 — ISOCELL HP9, GN 시리즈.
- Samsung Newsroom (2020.02), *108MP Nonacell Image Sensor*.
- Sony Semiconductor Solutions, *2-Layer Transistor Pixel* 기술 페이지.
- Sony Semiconductor Solutions, *All-pixel AF / 2×2 OCL* 기술 페이지.
- OmniVision 보도자료 (2024.03), *OV50K40 with TheiaCel*.
- 최근 플래그십 CIS 단면에 대한 공개 TechInsights 요약.
- Hwang & Kim, *Sensors* 23, 702 (2023) — `auto_cra` 가 사용하는 Snell 법칙 CRA shift.
