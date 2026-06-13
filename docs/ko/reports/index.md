---
outline: deep
---

# 시뮬레이션 리포트

Python benchmark 산출물과 geometry 감사 스크립트에서 생성한 publication-style 리포트다. Reports는 검증 근거를 위한 섹션이다: 생성된 그림, metric table, 정확한 재생성 명령을 담는다.

## 리포트 목록

- [TMM vs RCWA 평면 스택 검증](./tmm-rcwa-planar-validation.md) (생성일 2026-06-11)
- [RCWA Backend Parity](./rcwa-backend-parity.md) (생성일 2026-06-11)
- [CRA Cone Illumination Sweep](./cra-cone-illumination-sweep.md) (생성일 2026-06-11)
- [BARL Optimization Benchmark](./barl-optimization-benchmark.md) (생성일 2026-06-11)
- [DTI Crosstalk Benchmark](./dti-crosstalk-benchmark.md) (생성일 2026-06-11)
- [Performance Benchmark](./performance-benchmark.md) (생성일 2026-06-11)
- [RCWA/FDTD 수렴 분석](./convergence-analysis.md) (생성일 2026-05-07)
- [픽셀 스택 Geometry 감사](./pixel-stack-geometry-audit.md) (생성일 2026-06-06)
- [컬러 필터 Relief 민감도](./color-filter-relief-sensitivity.md) (생성일 2026-06-06)
- [픽셀 구조 현실성](./pixel-structure-realism.md) (생성일 2026-06-06)

## 리포트 대기열

_현재 대기 중인 리포트는 없다._

## 이 섹션에 들어갈 내용

- GitHub Pages에서 바로 확인할 수 있어야 하는 cross-solver 검증 결과.
- solver 입력 stack이 의도한 config와 일치함을 보이는 geometry 감사.
- 로컬 `outputs/` 산출물에서 `docs/public/reports/`로 승격한 그림과 표.
- 공개된 그림을 어떤 스크립트로 다시 만들 수 있는지 설명하는 재현성 노트.

개념은 [이론](/ko/theory/), 실행 절차는 [가이드](/ko/guide/), 레시피는 [쿡북](/ko/cookbook/bsi-2x2-basic), 생성 근거는 Reports에 둔다.
