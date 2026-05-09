---
outline: deep
---

# 시뮬레이션 리포트

Python benchmark 산출물과 geometry 감사 스크립트에서 생성한 publication-style 리포트다. Reports는 검증 근거를 위한 섹션이다: 생성된 그림, metric table, 정확한 재생성 명령을 담는다.

## 리포트 목록

- [RCWA/FDTD 수렴 분석](./convergence-analysis.md) (생성일 2026-05-07)
- [픽셀 스택 Geometry 감사](./pixel-stack-geometry-audit.md) (생성일 2026-05-08)
- [컬러 필터 Relief 민감도](./color-filter-relief-sensitivity.md) (생성일 2026-05-08)

## 리포트 대기열

| 우선순위 | 리포트 | 필요한 근거 |
| --- | --- | --- |
| 1 | RCWA backend parity | torcwa/grcwa/meent/fmmax QE, R/T/A, runtime table |
| 2 | Angular response characterization | 구조적인 $(\theta,\phi,\lambda)$ QE/EQE grid, ray-file cone averaging, CRA/F-number/corner sampling map |
| 3 | BARL optimization benchmark | 로컬 output에서 승격한 thickness/material sweep |
| 4 | DTI crosstalk benchmark | FDTI/BDTI width/depth/material sweep과 crosstalk matrix |
| 5 | Performance benchmark | CPU/GPU runtime, memory, wavelength-sweep cost |

## Characterization report template

다음 angular-response report는 근거를 세 층으로 분리하는 것이 좋습니다:

| Layer | 필요한 output | 중요한 이유 |
| --- | --- | --- |
| Optical angular grid | pixel별 $\text{QE}(\lambda,\theta,\phi)$ 또는 $\text{OE}(\lambda,\theta,\phi)$ | 여러 lens position에서 재사용 가능한 lookup table |
| Cone/ray averaging | 각 ray bundle의 `intensity * weight`를 사용한 weighted average | angular response를 sensor-position response로 변환 |
| Electrical collection, 사용 가능한 경우 | $W_i(\mathbf{r})$ map 또는 문서화된 approximation | optical absorption과 collected charge를 구분 |

Angular grid와 weighting convention 없이 cone-averaged curve 하나만 공개하지 마세요. 그렇지 않으면 solver, lens file, sensor position 간 비교가 어려워집니다.

외부 workflow 참고: Ansys Optics의 [CMOS Sensor Camera - Sensor Characterization](https://optics.ansys.com/hc/en-us/articles/360062131614-CMOS-Sensor-Camera-Sensor-Characterization)는 angular optical simulation, electrical weighting, ray 기반 cone averaging을 분리해서 다루는 좋은 예입니다. COMPASS는 특정 상용 tool chain에 의존하지 않고 이 workflow 개념만 재사용하는 방향이 맞습니다.

## 이 섹션에 들어갈 내용

- GitHub Pages에서 바로 확인할 수 있어야 하는 cross-solver 검증 결과.
- solver 입력 stack이 의도한 config와 일치함을 보이는 geometry 감사.
- 로컬 `outputs/` 산출물에서 `docs/public/reports/`로 승격한 그림과 표.
- 공개된 그림을 어떤 스크립트로 다시 만들 수 있는지 설명하는 재현성 노트.

개념은 [이론](/ko/theory/), 실행 절차는 [가이드](/ko/guide/), 레시피는 [쿡북](/ko/cookbook/bsi-2x2-basic), 생성 근거는 Reports에 둔다.
