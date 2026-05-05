---
title: COMPASS 가이드
description: COMPASS 설치, CMOS 이미지 센서 픽셀 설정, 솔버 선택, 검증된 시뮬레이션 실행을 위한 실무 가이드.
---

# 가이드

이 섹션은 물리 이론을 읽는 곳이 아니라 COMPASS를 실제로 실행하는 곳입니다. 설치, 픽셀 스택 설정, 솔버 실행, 검증, 레시피형 작업 절차를 하나의 흐름으로 연결합니다.

## 어디서 시작할까

| 목표 | 먼저 볼 문서 | 이어서 볼 문서 |
|---|---|---|
| COMPASS 설치 | [설치](./installation.md) | [빠른 시작](./quickstart.md) |
| 첫 픽셀 시뮬레이션 실행 | [첫 번째 시뮬레이션](./first-simulation.md) | [시각화](./visualization.md) |
| BSI 픽셀 설정 | [픽셀 스택 구성](./pixel-stack-config.md) | [샘플 픽셀 구조](./sample-pixels.md) |
| 솔버 선택 | [솔버 선택](./choosing-solver.md) | [교차 검증](./cross-validation.md) |
| 수치 설정 튜닝 | [수렴 연구](/ko/cookbook/convergence-study) | [솔버 비교 가이드](/ko/cookbook/solver-benchmark) |
| 시스템 지표로 확장 | [신호 시뮬레이션](./signal-simulation.md) | [신호 체인 색 정확도](/ko/cookbook/signal-chain-color-accuracy) |

## 구성 방식

1. **시작하기**는 설치와 최소 시뮬레이션 실행을 다룹니다.
2. **설정**은 YAML, 재료 데이터, 픽셀 스택, 솔버 선택을 설명합니다.
3. **솔버 실행**은 RCWA, FDTD, 교차 검증 워크플로를 다룹니다.
4. **고급**은 원뿔 조명, 신호 시뮬레이션, ROI 스윕, 역설계, 시각화, 문제 해결을 다룹니다.
5. **레시피**는 자주 쓰는 설계 및 검증 작업을 바로 따라 할 수 있는 절차로 정리합니다.

::: tip
가이드에서 나오는 물리 용어가 낯설다면 [이론](/ko/theory/)에서 배경을 확인한 뒤 다시 이곳으로 돌아오면 됩니다.
:::
