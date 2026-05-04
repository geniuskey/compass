---
title: 샘플 픽셀 구조
description: 최근 세대 CMOS 이미지 센서를 대표할 만한 픽셀 구조 샘플과, 공개되지 않는 내부 레이어 파라미터를 결정하기 위한 방법론을 정리합니다.
---

# 샘플 픽셀 구조

이 문서는 0.56–1.6 µm 피치 범위를 대표할 수 있는 CIS 픽셀 아키텍처 샘플들을 정리하고, COMPASS 에서 바로 시뮬레이션할 수 있는 YAML 설정과 헤드라인 입력으로부터 전체 설정을 만들어 주는 Python 헬퍼(`compass.geometry.derive_parameters`) 사용법을 함께 제공합니다.

## 왜 단순히 픽셀 설정을 만드는 게 어려운가

현세대 플래그십 픽셀에 대해 공개되는 정보는 보통 *헤드라인* 수준입니다 — 픽셀 피치, 색필터 비닝, OCL 공유, 그리고 구조 혁신의 마케팅 명칭(예: 분리형 트랜지스터 기판, LOFIC, 4×4 슈퍼셀 비닝) 정도입니다. 정작 EM 시뮬레이션에 필요한 내부 수치는 거의 공개되지 않습니다:

- 마이크로렌즈 sag / 반경 / 굴절률
- 색필터, 평탄화, BARL 스택 두께
- DTI 트렌치 폭, 깊이, 채움 재료
- 포토다이오드 수평 footprint 와 z-extent

COMPASS 는 이 부분을 경험식 기반 스케일링 룰로 채워 넣습니다 (아래 [파라미터 결정 방법](#parameter-determination)).

## 샘플 픽셀 구조 목록

| 키                          | 설정 파일                            | 피치     | CFA 패턴            | OCL 공유 | 주요 특징                                                |
|-----------------------------|--------------------------------------|----------|---------------------|----------|----------------------------------------------------------|
| `sample_p0p56um_4x4ocl`    | `sample_p0p56um_4x4ocl.yaml`        | 0.56 µm | 4×4 슈퍼셀          | 1        | sub-µm 픽셀, 고굴절률 마이크로렌즈 (n ≈ 1.70), F-DTI SiO₂ |
| `sample_p1p0um_quadbayer`  | `sample_p1p0um_quadbayer.yaml`      | 1.0 µm  | Quad Bayer (2×2)    | 1        | 일반적인 50 MP 급 메인 카메라 픽셀, F-DTI SiO₂           |
| `sample_p1p6um_split_pd`   | `sample_p1p6um_split_pd.yaml`       | 1.6 µm  | 표준 Bayer          | 1        | 분리형 트랜지스터 기판 → PD 부피 약 2배 확대             |
| `sample_p1p22um_2x2ocl`    | `sample_p1p22um_2x2ocl.yaml`        | 1.22 µm | Quad Bayer          | 2        | 같은 색 2×2 그룹마다 큰 마이크로렌즈 1개 (전 픽셀 PDAF)  |
| `sample_p1p2um_lofic`      | `sample_p1p2um_lofic.yaml`          | 1.2 µm  | Quad Bayer          | 2        | LOFIC HDR — 캐패시터가 PD footprint 잠식, 2×2 OCL         |

Hydra 로 실행:

```bash
python scripts/run_simulation.py pixel=sample_p0p56um_4x4ocl source=wavelength_sweep
python scripts/run_simulation.py pixel=sample_p1p6um_split_pd solver=torcwa
python scripts/run_simulation.py pixel=sample_p1p2um_lofic source=cone_illumination
```

### `sample_p0p56um_4x4ocl` — 4×4 비닝의 sub-µm 픽셀

가장 작은 피치 샘플. 모델링한 광학적 핵심 요소:

- **피치 0.56 µm**.
- **4×4 같은 색 슈퍼셀** 색필터 (16-cell 비닝); 8×8 unit cell.
- **고굴절률 마이크로렌즈** — `MaterialDB` 의 `polymer_hri_n1p70` (TiO₂-doped polymer, n ≈ 1.70) 항목으로 모델링.
- **F-DTI SiO₂ 채움** (크로스토크 감소를 위한 산화막 채움 DTI).

### `sample_p1p0um_quadbayer` — 1.0 µm Quad Bayer

일반적인 50 MP 급 메인 카메라용 픽셀. Quad Bayer + 픽셀당 OCL + F-DTI SiO₂.

### `sample_p1p6um_split_pd` — 1.6 µm 분리형 트랜지스터 픽셀

피치가 큰 샘플로, 포토다이오드와 픽셀 트랜지스터를 다른 기판층에 분리해 PD 면적과 깊이를 키운 구조. 결과적으로 포화 신호 수준이 약 2배.

광학 모델: PD 가 픽셀 footprint 의 거의 전체를 차지하고 더 깊게 형성되도록 `silicon` 의 `photodiode.size` 를 키워 표현했습니다.

### `sample_p1p22um_2x2ocl` — 2×2 OCL Quad Bayer

폰 메인카메라 급 픽셀로, 같은 색 2×2 그룹을 하나의 큰 렌즈가 덮어 모든 픽셀에서 PDAF 가 가능합니다. `microlens.sharing: 2` 로 표현해 4픽셀 클러스터마다 큰 렌즈 하나가 배치되도록 모델링.

### `sample_p1p2um_lofic` — LOFIC HDR

50 MP 급, 1.2 µm. **LOFIC** (Lateral Overflow Integration Capacitor) 캐패시터가 픽셀 내 실리콘 면적의 일부를 차지하므로 PD footprint 가 약간 작아집니다. 그 위로 Quad PD 가 2×2 OCL 을 추가합니다.

## 파라미터 결정 방법 {#parameter-determination}

공개되지 않은 내부 파라미터를 결정하기 위한 세 가지 보완적 접근법을 제공합니다.

### 1. 경험식 스케일링 룰 (기본 동작)

`compass.geometry.derive_parameters` 는 공개된 ISSCC, IEDM, SPIE 픽셀 아키텍처 논문 (2018–2024) 과 sub-µm ~ 2 µm BSI 픽셀에 대한 단면 보고서에서 도출한 피치 기반 스케일링 룰을 사용합니다:

```python
from compass.geometry import derive_parameters, PixelStack

cfg = derive_parameters(sample="sample_p0p56um_4x4ocl", cra_deg=20.0)
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
| PD footprint                  | `0.70·pitch` (split-PD 는 `0.88·pitch`, LOFIC 는 `0.65·pitch`) |
| PD z-extent                   | `0.67·t_Si` (split-PD 는 `0.85·t_Si`)                  |

`split_pd`, `lofic` 같은 아키텍처 플래그가 PD 형상을 조정합니다. 이 디폴트로 공개된 단면을 ~30% 이내로 재현합니다.

### 2. 측정 QE / 크로스토크 캘리브레이션

색별 측정 QE(λ) 또는 측정 공간 크로스토크가 있으면 최적화 프레임워크로 미지수 파라미터를 fit 하세요:

```python
from compass.geometry import derive_parameters
from compass.optimization import (
    ParameterSpace, MicrolensHeight, BARLThicknesses, Optimizer
)

base = derive_parameters(sample="sample_p1p0um_quadbayer", cra_deg=0.0)

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

### 3. 직접 측정 (SEM / 다이샷 단면)

다이샷 SEM 이 있다면 레이어 두께를 직접 측정한 뒤 `overrides` 로 휴리스틱을 덮어쓰세요:

```python
cfg = derive_parameters(
    sample="sample_p0p56um_4x4ocl",
    overrides={
        "layers.silicon.thickness": 1.85,       # 측정 Si 두께
        "layers.color_filter.thickness": 0.42,  # 측정 CF 두께
        "layers.silicon.dti.width": 0.075,
    },
)
```

`overrides` 는 어느 깊이의 dotted key 든 받으며, 휴리스틱보다 우선합니다.

## 샘플 헤드라인 레퍼런스

샘플별 헤드라인 값은 `compass.geometry.SAMPLE_HEADLINES` 에 저장되어 있고, `derive_parameters` 의 진입점으로 사용할 수 있습니다:

```python
>>> from compass.geometry import SAMPLE_HEADLINES
>>> SAMPLE_HEADLINES["sample_p0p56um_4x4ocl"]
{'pitch': 0.56, 'megapixels': 200, 'cf_pattern': 'tetra2cell',
 'ocl_sharing': 1, 'microlens_material': 'polymer_hri_n1p70',
 'dti_fill': 'sio2', 'year': 2024}
```

## 참고 자료 (References)

위 샘플 구조들은 공개된 최근 상용 CIS 제품 정보를 참고해 일반화한 형태입니다. 설정값은 특정 벤더의 공정 수치가 아니라 본 가이드의 경험식 스케일링 룰에서 도출한 값이며, 각 샘플이 모티브로 삼은 기술에 대한 추가 자료 링크는 다음과 같습니다:

- 4×4 슈퍼셀 비닝 + 고굴절률 마이크로렌즈를 갖는 sub-µm 픽셀 — [Samsung ISOCELL HP9 발표 (2024.06)](https://news.samsung.com/global/samsung-unveils-versatile-image-sensors-for-superior-smartphone-photography) 및 [Samsung Image Sensor 제품 페이지](https://semiconductor.samsung.com/image-sensor/) 참조.
- 1.0 µm Quad Bayer + F-DTI 산화막 채움 — [Samsung ISOCELL GN 시리즈 제품 페이지](https://semiconductor.samsung.com/image-sensor/mobile-image-sensor/isocell-gn/) 및 Samsung Newsroom (2020.02), *108 MP Nonacell Image Sensor* 발표.
- 1.6 µm 분리형 트랜지스터 픽셀 — [Sony Semiconductor Solutions — 2-Layer Transistor 픽셀 기술 페이지](https://www.sony-semicon.com/en/technology/lsi/2layer-transistor-pixel.html) (IEDM 2021).
- 1.22 µm 2×2 OCL Quad Bayer — [Sony Semiconductor Solutions — All-pixel AF / 2×2 OCL 기술 페이지](https://www.sony-semicon.com/en/technology/mobile/index.html).
- 1.2 µm LOFIC HDR — [OmniVision OV50K40 보도자료 (2024.03)](https://www.ovt.com/news-events/product-releases/) 및 OmniVision TheiaCel 기술 브리프.
- 최근 플래그십 CIS 단면에 대한 공개 TechInsights 요약 (단면 두께 / DTI 폭 참고치).
- Hwang & Kim, *Sensors* 23, 702 (2023) — `auto_cra` 가 사용하는 Snell 법칙 CRA shift.
