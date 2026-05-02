# 이론

이 섹션은 모든 COMPASS 시뮬레이션의 기반이 되는 물리, 센서 구조, 수치 해석 방법을 다룹니다. 위에서부터 차례로 읽어도 되고, 필요한 주제로 바로 이동해도 됩니다.

## 챕터 구성

| 챕터 | 범위 | 이런 분께 추천 |
|---|---|---|
| [광학](./optics-intro.md) | 파동 광학의 기초 — 전자기장, 박막, 회절 | 서브파장 픽셀에서 *왜* 파동 광학 솔버가 필요한지 이해하고 싶을 때 |
| [이미지 센서](./sensor-intro.md) | BSI 픽셀 구조, QE, 라디오메트릭 신호 체인 | *무엇*을 시뮬레이션하고 결과를 어떻게 해석하는지 알고 싶을 때 |
| [시뮬레이션](./simulation-intro.md) | RCWA, FDTD, TMM과 그 수치적 거동 | 솔버가 *어떻게* 답을 계산하는지 이해하고 싶을 때 |

## 추천 학습 경로

**이미지 센서 광학을 처음 접하는 경우**
1. [빛의 기초](./light-basics.md) → [전자기파](./electromagnetic-waves.md) → [박막 광학](./thin-film-optics.md) → [회절](./diffraction.md)
2. [이미지 센서 광학](./image-sensor-optics.md) → [양자 효율](./quantum-efficiency.md)
3. [시뮬레이션 개요](./simulation-intro.md) → 관심 있는 솔버 페이지 선택

**프로젝트에 COMPASS 도입을 검토하는 엔지니어**
1. [이미지 센서 광학](./image-sensor-optics.md) → [양자 효율](./quantum-efficiency.md) → [신호 체인](./signal-chain.md)
2. [시뮬레이션 개요](./simulation-intro.md) → [RCWA vs FDTD](./rcwa-vs-fdtd.md) → [수치 안정성](./numerical-stability.md)

**새로운 솔버나 분석 모듈을 구현하는 연구자**
1. [광학 개요](./optics-intro.md) → [전자기파](./electromagnetic-waves.md)
2. [RCWA 설명](./rcwa-explained.md), [FDTD 설명](./fdtd-explained.md)
3. [수치 안정성](./numerical-stability.md)

::: tip
COMPASS를 처음 사용하시는 경우 [소개 → CMOS 이미지 센서란?](/ko/introduction/what-is-cmos-sensor)에서 비수식 개관을 먼저 보신 후 이곳으로 돌아오시는 것을 권장합니다.
:::
