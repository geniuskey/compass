---
title: 이론 커버리지 지도
description: COMPASS 이론 문서의 커버리지 감사 문서로, CMOS 이미지 센서 광학의 각 주제가 어디에서 설명되는지와 남은 확장 지점을 정리합니다.
---

# 이론 커버리지 지도

이 페이지는 공개 이론 문서의 커버리지 감사 문서입니다. 개별 theory 페이지를 계획 문서로 만들지 않으면서, 유지보수자와 심화 독자가 각 주제가 어디에 속하는지 확인할 수 있게 합니다.

## 현재 커버리지

| 영역 | 다루는 문서 | 상태 |
|---|---|---|
| CMOS 이미지 센서 개관 | [입문](../theory/basics/) | 안정적인 진입점 |
| 쉬운 광학 설명 | [광학 기초 입문](../theory/basics/optics-primer.md) | 안정적인 진입점 |
| 파동 광학 공식 | [광학](../theory/optics/) | 핵심 커버리지 있음 |
| 픽셀 스택 해부 | [이미지 센서 광학](../theory/sensor/image-sensor-optics.md) | 핵심 커버리지 있음 |
| 픽셀 수준 광학 트레이드오프 | [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md) | 핵심 커버리지 있음 |
| QE와 크로스토크 지표 | [양자 효율](../theory/sensor/quantum-efficiency.md) | 핵심 커버리지 있음 |
| 복사 측정 신호 체인 | [신호 체인](../theory/sensor/signal-chain.md) | 핵심 커버리지 있음 |
| 솔버 기초 | [광학 시뮬레이션](../theory/simulation/) | 핵심 커버리지 있음 |
| RCWA/FDTD 비교 | [RCWA vs FDTD](../theory/simulation/rcwa-vs-fdtd.md) | 핵심 커버리지 있음 |
| 검증 근거 | [리포트](../reports/) | 생성된 근거를 별도 게시 |

## 주제별 소유권

| 주제 | 담당 문서 | 경계 |
|---|---|---|
| 마이크로렌즈 형상과 CRA | [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md) | Cookbook 문서는 실행 가능한 스윕을 보여줌 |
| CFA 분광 거동 | [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md) | Signal Chain은 광원과 색 지표를 담당 |
| BARL과 ARC | [박막 광학](../theory/optics/thin-film-optics.md), [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md) | Optics는 박막 원리, Sensor는 설계 영향을 설명 |
| DTI와 크로스토크 | [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md), [양자 효율](../theory/sensor/quantum-efficiency.md) | QE 문서는 행렬 정의를 담당 |
| 각도/편광 응답 | [픽셀 광학 효과](../theory/sensor/pixel-optical-effects.md), [빛의 기초](../theory/optics/light-basics.md) | Simulation 문서는 솔버 처리를 설명 |
| 솔버 수렴 | [수치 안정성](../theory/simulation/numerical-stability.md), [리포트](../reports/convergence-analysis.md) | 리포트는 생성된 수치를 포함 |

## 확장 지점

이론 섹션에 더 깊이가 필요할 때 우선순위가 높은 다음 주제들입니다:

| 확장 주제 | 적합한 위치 | 이유 |
|---|---|---|
| 전기적 수집과 carrier diffusion | 새 sensor 페이지 | 전하 수집 모델이 없으면 optical QE는 상한으로 해석해야 함 |
| 공정 편차와 tolerance analysis | Guide 또는 research note | 실행 절차와 통계적 해석을 함께 제시하는 편이 좋음 |
| Lens shading과 module-level CRA map | Sensor 또는 simulator 페이지 | 픽셀 광학과 카메라 모듈 가정을 연결 |
| MTF와 optical crosstalk 관계 | Sensor 또는 signal-chain 페이지 | 픽셀 누설을 이미지 수준 선명도와 연결 |
| 측정 QE와의 calibration | Reports 또는 validation guide | 데이터 출처와 측정 조건에 묶어야 함 |

## 유지보수 규칙

새 내용을 추가할 때는 페이지 역할을 분리합니다. 입문은 직관, 광학은 물리 법칙, 이미지 센서는 픽셀 구성 요소와의 대응, 시뮬레이션은 수치 방법, 가이드는 명령과 절차, 리포트는 생성된 검증 근거를 담당합니다.
