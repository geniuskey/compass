---
outline: deep
---

# 픽셀 구조 현실성 리포트

_생성일: 2026-06-06. `compass.geometry.sample_pixels`와 `PixelStack`에서 생성._

이 리포트는 최신 후면 조사형(BSI) CMOS 이미지 센서 픽셀에 대해 COMPASS가 모델링하는 구조적 현실성 요소를 문서화하고, 생성된 solver 입력에서 직접 시각화한다. 단면은 RCWA solver가 적분하는 실제 유전율 `Re(eps)(x, z)`를 샘플링하므로, 광학 성능 주장이 아니라 geometry evidence다.

## 왜 중요한가

- **후면 역피라미드 텍스처(IPA).** 후면의 그라데이션 실리콘 충전율은 moth-eye 반사방지/광 트래핑 층으로 작동한다. 공개된 시뮬레이션·공정 보고서는 IPA를 깊은 DTI와 결합할 때 근적외선 QE가 크게(850 nm에서 최대 ~3배, 940 nm에서 ~5배) 증가함을 보인다.
- **DTI 컨포멀 high-k 라이너.** 실제 후면 DTI 트렌치는 식각된 실리콘을 패시베이션하고 음의 고정 전하를 갖는 얇은 high-k 막(Al2O3 / HfO2 / Ta2O5, 약 30-100 nm)으로 라이닝된다. 광학적으로는 실리콘과 저굴절 충전재 사이의 얇은 고굴절 링이다.
- **테이퍼 DTI 측벽.** 식각된 트렌치는 깊이에 따라 좁아진다. 수직 측벽 이상화는 기판 깊은 곳의 격리 산화막을 과대 계산한다.
- **마이크로렌즈 잔류층.** Reflow / etch-back은 곡면 캡 아래에 평탄한 폴리머 슬랩을 남긴다. 렌즈는 가장자리에서 두께가 0이 아니다.

## 기준 픽셀 vs NIR 강화 픽셀

![Baseline vs realism-enhanced pixel](/reports/geometry/structure-realism/baseline_vs_realism.png)

## 실리콘 후면 상세

역피라미드 텍스처의 그라데이션 `Re(eps)`는 트렌치 충전재에서 벌크 실리콘으로 매끄러운 유효 굴절률 전이를 만들고, 픽셀 경계에서는 테이퍼 DTI 트렌치와 high-k 라이너가 보인다.

![Silicon backside detail](/reports/geometry/structure-realism/silicon_backside_zoom.png)

- 텍스처 높이: **0.350 um**
- 텍스처 전체의 면적 평균 유효 굴절률: 표면 **1.46** → 정점 방향 **3.67** (단조 증가하는 graded-index 반사방지).

## 기능 커버리지 매트릭스

| 실제 픽셀 요소 | COMPASS 상태 | config 키 |
| --- | --- | --- |
| Microlens superellipse profile + CRA shift | modelled | `microlens.profile, microlens.shift` |
| Multi-pixel shared lens (2x2 / 4x4 OCL) | modelled | `microlens.sharing` |
| Microlens residual base layer | modelled (new) | `microlens.base_thickness` |
| Per-color CF thickness + contact-angle relief | modelled | `color_filter.<color>.thickness/contact_angle` |
| Metal grid (W) with rounded corners | modelled | `color_filter.grid` |
| BARL anti-reflection multilayer | modelled | `barl.layers` |
| FDTI / BDTI deep trench isolation | modelled | `silicon.dti.mode/depth/width` |
| DTI conformal high-k passivation liner | modelled (new) | `silicon.dti.liner` |
| Tapered DTI sidewall (etch profile) | modelled (new) | `silicon.dti.taper_angle` |
| Backside inverted-pyramid light-trapping texture | modelled (new) | `silicon.surface_texture` |
| Photodiode collection window | modelled | `silicon.photodiode` |
| Composite / air-gap metal grid liner | roadmap | `-` |
| In-pixel light pipe / inner lens | roadmap | `-` |

## 이 픽셀 재현

```bash
python scripts/run_simulation.py pixel=sample_p1p12um_nir
```

또는 Python에서 config를 유도한다:

```python
from compass.geometry.sample_pixels import derive_parameters
from compass.geometry.pixel_stack import PixelStack

cfg = derive_parameters("sample_p1p12um_nir")
stack = PixelStack({"pixel": cfg})
```

## 재생성

```bash
python scripts/generate_realism_report.py
```

생성 metric은 `docs/public/reports/geometry/structure-realism/structure_realism_metrics.json`에 저장된다.
