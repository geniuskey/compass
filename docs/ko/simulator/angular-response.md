---
title: 각도 응답
---

# 각도 응답 시뮬레이터

실제 카메라에서 빛은 센서 위의 픽셀 위치와 렌즈 설계에 따라 다양한 각도로 픽셀에 도달합니다. 주광선각(CRA)은 센서 가장자리에서 20–30°에 달할 수 있어 QE에 상당한 영향을 미칩니다.

<AngularResponseSimulator />

::: info 모델 범위
이 브라우저 도구는 직관 형성, 상대 경향 비교, 설계 공간 탐색에 사용하세요. 로컬에서 실행되는 간이 모델이며 RCWA/FDTD sign-off, 실리콘 보정, 벤더 공정 데이터를 대체하지 않습니다.
:::

::: tip 더 알아보기
[픽셀 광학 효과](/ko/theory/sensor/pixel-optical-effects) · [원뿔 조명 가이드](/ko/guide/cone-illumination)
:::

<SimulatorTheory slug="angular-response" />
