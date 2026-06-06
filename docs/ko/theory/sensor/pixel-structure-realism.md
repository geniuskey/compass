---
title: 픽셀 구조 현실성
description: 이상적인 레이어 스택을 넘어 실제 제조된 BSI CMOS 이미지 센서 픽셀이 포함하는 구조 요소 — 후면 역피라미드 광 트래핑 텍스처, 컨포멀 high-k DTI 라이너, 테이퍼 트렌치 프로파일, 마이크로렌즈 잔류층 — 와 COMPASS가 각각을 모델링하는 방법.
---

# 픽셀 구조 현실성

::: tip 사전 지식
[픽셀 해부학](../basics/pixel-anatomy.md)이 레이어 스택을 소개하고,
[픽셀 광학 효과](./pixel-optical-effects.md)가 각 구성 요소가 QE와 crosstalk를
어떻게 움직이는지 설명한다. 이 페이지는 교과서적 스택과 실제 제조된 픽셀을
구분하는 *구조적* 세부사항을 다룬다.
:::

1차 근사 픽셀 모델 — 평탄한 실리콘, 사각형 산화막 트렌치, 깔끔한 마이크로렌즈
캡 — 만으로도 지배적인 QE 추세는 재현할 수 있다. 하지만 마지막 몇 %의 QE,
근적외선(NIR) 응답, crosstalk 하한은 모두 평탄 모델이 놓치는 구조적 세부사항이
결정한다. 이 페이지는 중요한 현실성 요소를 모으고 각각을 COMPASS config 키에
대응시킨다.

## 현실성 격차 지도

| 실제 픽셀 요소 | 광학적으로 왜 중요한가 | COMPASS config |
|---|---|---|
| 후면 역피라미드 텍스처 | Graded-index moth-eye AR + 광 트래핑; 큰 NIR QE 증가 | `silicon.surface_texture` |
| DTI 컨포멀 high-k 라이너 | 얇은 고굴절 링; 패시베이션 + 음의 고정 전하 | `silicon.dti.liner` |
| 테이퍼 DTI 측벽 | 식각 트렌치는 깊이에 따라 좁아짐; 깊은 격리 충전 변경 | `silicon.dti.taper_angle` |
| 마이크로렌즈 잔류층 | Reflow/etch-back이 캡 아래 평탄 폴리머 슬랩을 남김 | `microlens.base_thickness` |
| 색별 CF relief + contact angle | 각 resist는 고유 높이와 측벽 기울기를 가짐 | `color_filter.<color>` |
| 모서리 둥근 금속 그리드 | 실제 그리드는 완전한 사각형이 아님 | `color_filter.grid.corner_radius` |

## 후면 역피라미드 텍스처 (광 트래핑)

최신 NIR 강화 BSI 센서는 입사광을 향하는 실리콘 후면에 **역피라미드
배열(IPA)** 을 식각한다. 두 가지 메커니즘이 QE를 높인다:

1. **Graded-index 반사방지.** 트렌치 충전재에서 벌크 실리콘으로 들어갈수록
   면적 평균 실리콘 비율이 표면의 ~0에서 피라미드 정점의 1로 매끄럽게
   증가한다. 따라서 유효 굴절률이 평탄한 Si 계면에서처럼 급격히 계단지는 대신
   서서히 변하며, moth-eye 코팅과 같은 방식으로 전면 반사를 억제한다.
2. **광 트래핑.** 다면 표면은 장파장 빛을 비스듬한 각도로 굴절시켜 (약하게
   흡수하는) 실리콘 내 경로를 늘리고, 빠져나가기 전에 흡수될 확률을 높인다.

공개된 소자 시뮬레이션과 공정 보고서는 IPA를 깊은 DTI와 결합하면 근적외선
QE가 평탄 후면 대비 상당히 — 850 nm에서 약 3배, 940 nm에서 약 5배 — 증가함을
보인다. 해당 대역에서 실리콘 흡수 길이가 수십 µm로 커지기 때문이다.

COMPASS는 IPA를 실리콘 레이어 상단에서 깎아낸 피라미드형 피트의 staircase로
모델링하고 `fill_material`로 채운다. 피트 반폭은 표면의 `period/2`에서 정점의
0까지 선형으로 줄어들어 그라데이션 실리콘 충전율을 만든다:

```yaml
silicon:
  surface_texture:
    enabled: true
    type: inverted_pyramid
    height: 0.35          # 실리콘 내부로의 텍스처 깊이 (um)
    period: null          # 기본값은 픽셀 pitch
    fill_material: sio2
    n_slices: 8           # RCWA용 staircase 해상도
```

## DTI: 라이너, 충전재, 테이퍼

실제 후면 deep trench는 깔끔한 산화막 사각형이 아니다:

- **컨포멀 high-k 라이너.** 식각된 실리콘 측벽은 얇은 high-k 막(Al2O3, HfO2,
  Ta2O5, 보통 수십 nm)으로 라이닝되어 댕글링 본드를 패시베이션하고, **음의
  고정 전하**로 손상된 표면에서 전자를 밀어내 암전류를 낮춘다. 광학적으로는
  실리콘과 저굴절 코어 충전재 사이의 얇은 고굴절 링이므로 트렌치의 반사와
  crosstalk 기여를 바꾼다.
- **코어 충전재.** 라이너 안쪽은 산화막(가장 일반적), 또는 일부 설계에서는
  폴리실리콘, 텅스텐(metal DTI), 심지어 air gap으로 채워진다.
- **테이퍼 프로파일.** 플라즈마 식각 트렌치는 깊이에 따라 좁아진다. 수직 측벽
  이상화는 기판 깊은 곳의 격리 물질을 과대 계산하고, 광선·필드 계산에서
  트렌치/실리콘 경계를 잘못 위치시킨다.

```yaml
silicon:
  dti:
    enabled: true
    mode: fdti
    width: 0.12           # 후면 개구부에서의 트렌치 폭 (um)
    depth: 4.0
    material: sio2        # 코어 충전재
    taper_angle: 82.0     # 기판면 기준 측벽 각도 (90 = 수직)
    n_slices: 6           # 테이퍼 staircase 해상도
    liner:
      enabled: true
      material: al2o3     # high-k 패시베이션 라이너
      thickness: 0.015
```

라이너가 비활성화되고 테이퍼 각도가 90°이며 텍스처가 없으면 COMPASS는 기존의
빠른 단일 슬라이스(FDTI) 또는 2-슬라이스(BDTI) 경로를 사용하므로 기존 config는
변하지 않는다. 이 기능 중 하나라도 켜면 실리콘 레이어가 z-resolved staircase로
전환되며 RCWA·FDTD 백엔드가 투명하게 소비한다.

## 마이크로렌즈 잔류층

폴리머 마이크로렌즈는 패터닝된 resist를 reflow하거나 etch-back하여 형성된다.
공정은 항상 곡면 캡 아래에 동일 폴리머의 **평탄한 잔류 슬랩**을 남긴다 — 렌즈는
가장자리에서 두께가 0이 아니다. 이를 무시하면 렌즈 물질 내 광경로와 컬러 필터
위 air gap 높이를 약간 과소평가한다.

```yaml
microlens:
  height: 0.67            # 곡면 캡 sag
  base_thickness: 0.10    # 캡 아래 평탄 잔류 폴리머 슬랩
```

## 종합

`sample_p1p12um_nir` preset은 이 모두를 한 번에 켜며,
[픽셀 구조 현실성 리포트](/ko/reports/pixel-structure-realism)의 주제다. 이
리포트는 실제 solver 유전율 `Re(eps)(x, z)`를 렌더링하여 solver가 적분하는
geometry에서 그라데이션 텍스처, 라이닝된 테이퍼 트렌치, 잔류 렌즈 베이스를 직접
볼 수 있게 한다.

```bash
python scripts/run_simulation.py pixel=sample_p1p12um_nir
```

::: warning 범위
이 기능들은 *geometry*를 더 충실하게 만든다. 이것이 당신의 스택에서 QE를
바꾸는지는 광학적 질문이다 — delta를 인용하기 전에 RCWA order 수렴 sweep(그리고
텍스처의 경우 NIR로의 파장 sweep)을 실행하라.
:::
