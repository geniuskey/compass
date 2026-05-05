---
title: COMPASS 리서치 노트
description: 전자기 솔버, CMOS 이미지 센서 기술 동향, 시뮬레이션 방법론, 핵심 논문, 검증 전략에 대한 COMPASS 리서치 노트.
---

# 리서치

이 섹션은 COMPASS 설계 판단의 배경이 되는 조사 자료를 모아 둔 곳입니다. 단계별 실행 가이드는 아니며, 특정 솔버, 벤치마크, 센서 모델링 가정을 왜 선택했는지 이해할 때 사용합니다.

## 리서치 지도

| 주제 | 먼저 볼 문서 | 중요한 이유 |
|---|---|---|
| 솔버 생태계 | [EM 솔버 서베이](./open-source-em-solvers-survey.md) | open-source RCWA/FDTD 옵션과 라이선스 비교 |
| 센서 방향 | [CIS 기술 동향](./cis-technology-trends.md) | pixel scaling, BSI 구조, DTI, optical stack 동향 파악 |
| 방법 선택 | [시뮬레이션 방법론](./simulation-methods-comparison.md) | TMM, RCWA, FDTD, hybrid validation 전략 비교 |
| 문헌 | [핵심 논문](./key-papers.md) | 구현 선택을 공개 논문과 연결 |
| 검증 | [벤치마크 & 검증](./benchmarks-and-validation.md) | COMPASS가 solver correctness를 입증하는 방식 정의 |

## 사용 방법

- 새 solver backend를 추가하기 전에 읽습니다.
- 외부 근거가 필요한 설계 결정을 issue나 PR에서 설명할 때 링크합니다.
- 생성된 benchmark 증거는 [리포트](/ko/reports/)에서 보고, 이 섹션은 더 넓은 기술 맥락을 확인할 때 사용합니다.
