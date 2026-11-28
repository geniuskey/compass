---
title: 시뮬레이터
---

# 인터랙티브 시뮬레이터

브라우저 기반 시뮬레이터를 통해 CMOS 이미지 센서 픽셀 광학을 탐색하세요. 계산은 브라우저에서 로컬로 실행되며 서버가 필요하지 않습니다.

::: tip 시뮬레이터의 위치
이 시뮬레이터들은 **가벼운 학습용 도구**입니다 — TMM, 기하광학, 해석적 카메라 방정식, 시각화 위주입니다. 직관 형성·교육·설계 공간 탐색에 적합하며, 물리적으로 정확한 픽셀별 QE/크로스토크/필드 맵이 필요하면 [Python 솔버 파이프라인](/ko/guide/quickstart)(RCWA/FDTD/Meep)을 사용하세요. 도구별 적용 범위는 페이지 하단 표를 참고하세요.
:::

## 사용 가능한 시뮬레이터

카테고리를 고르거나 키워드로 찾아보세요.

<SimulatorGallery />

## 빠른 검증 예제

시뮬레이터 trend를 믿기 전에 아래 한계 사례를 먼저 확인하세요. 복잡한 sweep을 해석하기 전에 단순 검증을 통과해야 합니다.

| 시뮬레이터 | sanity check | 기대 동작 |
|------------|--------------|-----------|
| [TMM QE 계산기](./tmm-qe) | coating layer를 제거하고 normal-incidence air/Si 반사를 $R=|(1-n_{Si})/(1+n_{Si})|^2$와 비교합니다. | 반사율은 Fresnel limit에 가까워야 하고 에너지는 $R+T+\sum A_j\approx1$을 만족해야 합니다. |
| [마이크로렌즈 공정 형상](./microlens-process-shape) | reflow 조건을 고정하고 etch time을 늘립니다. | Final gap은 줄고 height retention은 낮아져야 하며, 과도한 etch에서는 height-loss 또는 over-etch 경고가 떠야 합니다. |
| [컬러 필터 설계](./color-filter) | RGB filter FWHM을 줄이고 peak는 유지합니다. | Channel crosstalk은 감소하지만 총 신호와 low-light SNR margin도 줄어야 합니다. |
| [SNR 계산기](./snr-calculator) / [Photon Transfer Curve](./photon-transfer-curve) | read noise를 고정하고 illumination을 sweep합니다. | 낮은 신호에서는 read-noise limited, 높은 신호에서는 $\sigma\approx\sqrt{S}$ shot-noise behavior가 나타나야 합니다. |
| [Si 흡수 깊이](./si-absorption) | red/NIR 파장에서 silicon thickness를 늘립니다. | 장파장은 더 깊이 침투하므로 blue보다 red/NIR absorption 개선이 크게 나타나야 합니다. |

## 모델 범위

| 도구군 | 모델 유형 | 적합한 용도 |
|--------|-----------|-------------|
| TMM / 박막 도구 | 1D 평면 다층막 광학 | BARL 경향, 반사율, 실리콘 흡수, 1차 QE 검토 |
| 광선 / 형상 도구 | 근축 또는 Snell 법칙 기반 기하광학 | 마이크로렌즈 직관, CRA 방향성, 레이아웃 민감도 |
| 센서 지표 도구 | 해석적 카메라 방정식 | SNR, 다이나믹 레인지, MTF, EMVA식 비교 |
| 시각화 도구 | 결정론적 설명 모델 | 구조 설명, 상대 경향, 실패 모드 이해 |

::: warning
이 도구들은 학습과 설계 공간 탐색을 위한 것입니다. RCWA/FDTD sign-off, 실리콘 보정, 벤더 공정 데이터를 대체하지 않습니다. 솔버 워크플로는 [가이드](/ko/guide/)와 [리포트](/ko/reports/)를 참고하세요.
:::
