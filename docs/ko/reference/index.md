---
title: COMPASS API 레퍼런스
description: COMPASS 설정, geometry, material database, source, solver, analysis, result 객체를 찾기 위한 API 레퍼런스 시작점.
---

# API 레퍼런스

정확한 클래스 이름, 모듈 경계, 설정 필드, 결과 객체가 필요할 때 이 섹션을 사용합니다. 레퍼런스는 일부러 간결하게 유지합니다. 실행 절차는 [가이드](/ko/guide/), 개념 배경은 [이론](/ko/theory/)에서 다룹니다.

## API 지도

| 영역 | 먼저 볼 문서 | 용도 |
|---|---|---|
| 패키지 개요 | [API 지도](./api-overview.md) | 주요 모듈과 workflow 파악 |
| Geometry | [PixelStack](./pixel-stack.md) | 솔버 독립적인 CMOS 픽셀 스택 구성 |
| Materials | [MaterialDB](./material-db.md) | 파장 의존 광학 상수 조회 |
| Solvers | [SolverBase](./solver-base.md) | RCWA/FDTD 백엔드 구현 또는 호출 |
| Sources | [Sources](./sources.md) | plane wave, cone illumination, ray input 설정 |
| Analysis | [Analysis](./analysis.md) | QE, energy balance, solver comparison 계산 |
| Configuration | [Config Reference](./config-reference.md) | YAML schema 필드와 기본값 확인 |
| Terms | [Glossary](./glossary.md) | 문서 전반의 용어 확인 |

## 언제 사용하는가

- COMPASS 클래스에 직접 맞춰 코드를 작성할 때.
- 특정 YAML key나 기본값을 확인해야 할 때.
- 새 solver adapter나 analysis module을 구현할 때.
- config에서 geometry, solver result로 데이터가 흐르는 과정을 디버깅할 때.
