---
title: CMOS 이미지 센서 광학 이론
description: CMOS 이미지 센서 광학, 파동 광학, 양자 효율, 신호 체인, RCWA, FDTD, 수치 안정성을 배우기 위한 이론 학습 경로.
---

# 이론

이 섹션은 모든 COMPASS 시뮬레이션의 기반이 되는 물리, 센서 구조, 수치 해석 방법을 다룹니다. 위에서부터 차례로 읽어도 되고, 필요한 주제로 바로 이동해도 됩니다.

이 섹션은 개념과 개념적 모델을 다룹니다. 실행 절차는 [가이드](/ko/guide/), 생성된 검증 근거는 [리포트](/ko/reports/), 정확한 클래스와 설정 필드는 [API 레퍼런스](/ko/reference/)에서 다룹니다.

## 챕터 구성

| 챕터 | 범위 | 이런 분께 추천 |
|---|---|---|
| [입문](./basics/) | CIS 픽셀, 광학, QE를 수식 없이 잡는 개관 | 깊은 이론 페이지로 가기 전에 직관을 만들고 싶을 때 |
| [광학](./optics/) | 파동 광학의 기초: 전자기장, 박막, 회절 | 서브파장 픽셀에서 *왜* 파동 광학 솔버가 필요한지 이해하고 싶을 때 |
| [이미지 센서](./sensor/) | BSI 픽셀 구조, QE, 라디오메트릭 신호 체인 | *무엇*을 시뮬레이션하고 결과를 어떻게 해석하는지 알고 싶을 때 |
| [시뮬레이션](./simulation/) | RCWA, FDTD, TMM과 그 수치적 거동 | 솔버가 *어떻게* 답을 계산하는지 이해하고 싶을 때 |

## 경계 규칙

| 필요한 것 | 읽을 곳 | 여기서 기대하지 말 것 |
|---|---|---|
| 수식을 최소화한 첫 직관 | [입문](./basics/) | 솔버 알고리즘이나 정확한 공식 |
| 광학 법칙과 식 | [광학](./optics/) | 전체 픽셀 스택이나 API 호출 |
| 픽셀 구성 요소와 성능 지표 | [이미지 센서](./sensor/) | 수치 해석 방법의 유도 |
| 이산화, 수렴, 안정성 | [시뮬레이션](./simulation/) | 초보자용 CIS 배경 설명 |

## 이 섹션에서 다루지 않는 것

- 설치와 명령줄 실행 절차는 [가이드](/ko/guide/)에서 다룹니다.
- 생성된 벤치마크 근거는 [리포트](/ko/reports/)에서 다룹니다.
- API 시그니처와 YAML 스키마 세부사항은 [API 레퍼런스](/ko/reference/)에서 다룹니다.

## 추천 학습 경로

**이미지 센서 광학을 처음 접하는 경우**
1. [입문 개요](./basics/) -> [CMOS 이미지 센서란?](./basics/what-is-cmos-sensor.md) -> [초보자를 위한 광학 입문](./basics/optics-primer.md) -> [픽셀 해부학](./basics/pixel-anatomy.md)
2. [빛의 기초](./optics/light-basics.md) -> [전자기파](./optics/electromagnetic-waves.md) -> [박막 광학](./optics/thin-film-optics.md) -> [회절](./optics/diffraction.md)
3. [이미지 센서 광학](./sensor/image-sensor-optics.md) -> [양자 효율](./sensor/quantum-efficiency.md)
4. [시뮬레이션 개요](./simulation/) -> 관심 있는 솔버 페이지 선택

**프로젝트에 COMPASS 도입을 검토하는 엔지니어**
1. [이미지 센서 광학](./sensor/image-sensor-optics.md) -> [양자 효율](./sensor/quantum-efficiency.md) -> [신호 체인](./sensor/signal-chain.md)
2. [시뮬레이션 개요](./simulation/) -> [RCWA vs FDTD](./simulation/rcwa-vs-fdtd.md) -> [수치 안정성](./simulation/numerical-stability.md)

**새로운 솔버나 분석 모듈을 구현하는 연구자**
1. [광학 개요](./optics/) -> [전자기파](./optics/electromagnetic-waves.md)
2. [RCWA 설명](./simulation/rcwa-explained.md), [FDTD 설명](./simulation/fdtd-explained.md)
3. [수치 안정성](./simulation/numerical-stability.md)

::: tip
COMPASS를 처음 사용하시는 경우 [입문 -> CMOS 이미지 센서란?](/ko/theory/basics/what-is-cmos-sensor)에서 비수식 개관을 먼저 보신 후 이곳으로 돌아오시는 것을 권장합니다.
:::
