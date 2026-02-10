---
title: 변경 이력
description: COMPASS 시뮬레이션 플랫폼의 버전 이력입니다.
---

# 변경 이력

COMPASS 프로젝트의 모든 주요 변경 사항은 이 문서에 기록됩니다. 이 프로젝트는 [유의적 버전 관리(Semantic Versioning)](https://semver.org/)를 따릅니다.

## v0.1.0 -- 최초 릴리스

**릴리스 날짜**: 2025-02-10

멀티 솔버 지원, 물질 데이터베이스, 시각화 도구를 갖춘 핵심 시뮬레이션 프레임워크를 확립한 COMPASS의 첫 번째 공개 릴리스입니다.

### 핵심 프레임워크

- 완전한 유효성 검증을 갖춘 Pydantic 기반 설정 스키마 (`CompassConfig`)
- YAML 설정 로딩 및 CLI 오버라이드를 위한 Hydra/OmegaConf 통합
- 내부 좌표계: 모든 길이에 마이크로미터, 외부 각도에 도(degree), 내부 각도에 라디안(radian) 사용
- 빛 전파 규약: -z 방향 (z_max에 공기, z_min에 실리콘)

### 기하학 엔진

- 솔버 비의존적 픽셀 구조 정의를 위한 `PixelStack` 및 `GeometryBuilder`
- 높이, 반경, 직각도, CRA 시프트를 설정할 수 있는 초타원 마이크로렌즈 프로파일
- 다층 BARL 지원
- 베이어 패턴(RGGB)과 설정 가능한 메탈 그리드를 갖춘 컬러 필터
- 내장 포토다이오드 및 DTI를 갖춘 실리콘 층
- 평탄화(오버코트) 층
- RCWA용 층 슬라이스 생성 (각 z 높이에서의 2D 유전율 그리드)

### 솔버

- 모든 EM 솔버를 위한 `SolverBase` 추상 인터페이스
- 지연 임포트 및 플러그인 등록을 갖춘 `SolverFactory`
- **torcwa** RCWA 솔버 어댑터 (PyTorch, CUDA GPU)
- **grcwa** RCWA 솔버 어댑터 (NumPy/JAX)
- **meent** RCWA 솔버 어댑터 (NumPy/JAX/PyTorch 멀티 백엔드)
- **fdtd_flaport** FDTD 솔버 어댑터 (PyTorch, CUDA GPU)
- 모든 솔버에 걸친 통합 `SimulationResult` 출력 형식

### 수치 안정성 (5중 방어)

- `PrecisionManager`: TF32 비활성화, 혼합 정밀도 고유값 분해
- `StableSMatrixAlgorithm`: Redheffer 별곱(T-행렬 발산 없음)
- `LiFactorization`: 고대비 경계를 위한 Li 역규칙
- `EigenvalueStabilizer`: 축퇴 고유값 처리, 분기 선택
- `AdaptivePrecisionRunner`: 자동 float32 -> float64 -> CPU 폴백

### 소스

- `PlanewaveSource`: 단일 파장, 스윕, 목록 모드
- TE, TM, 비편광(TE+TM 평균) 편광 지원
- `ConeIllumination`: CRA, F-수(F-number), 각도 샘플링(피보나치, 그리드), 가중 함수(균일, 코사인, cos4, 가우시안)를 갖춘 사출 동공 조명 모델

### 분석

- `QECalculator`: 픽셀별 양자 효율 추출
- `EnergyBalance`: R + T + A = 1 검증
- `SolverComparison`: QE 차이 쌍별 비교, 상대 오차, 실행 시간 비교

### 러너

- `SingleRunner`: 설정에서 단일 시뮬레이션 실행
- `SweepRunner`: 파장 스윕 오케스트레이션
- `ComparisonRunner`: 멀티 솔버 비교 워크플로
- `ROISweepRunner`: CRA 보간 및 마이크로렌즈 시프트를 갖춘 센서 위치 스윕

### 시각화

- `plot_pixel_cross_section`: 2D 단면도 (XZ, YZ, XY 평면)
- `plot_qe_spectrum`: 색상 채널별 QE 대 파장
- `plot_qe_comparison`: 다중 결과 QE 오버레이 및 차이 플롯
- `plot_crosstalk_heatmap`: 픽셀 간 에너지 분포
- `field_plot_2d`: EM 필드 분포 시각화
- 3D 픽셀 스택 뷰어 (PyVista/Plotly)

### 입출력

- 내장 메타데이터 및 설정을 포함한 HDF5 결과 저장
- 후처리를 위한 CSV/JSON 내보내기

### 물질 데이터베이스

- 내장 물질: air, silicon (Green 2008), SiO2, Si3N4, HfO2, TiO2, tungsten, polymer (n=1.56), 컬러 필터 염료 (R/G/B)
- CSV를 통한 사용자 정의 물질 (파장, n, k)
- 해석적 분산 모델: 셀마이어(Sellmeier), 코시(Cauchy)

### 인프라

- 전체에 타입 힌트를 적용한 Python 3.10+
- 단위 및 통합 테스트를 포함한 pytest 테스트 스위트
- ruff 린팅 (E, F, W, I 규칙)
- mypy 정적 타입 검사
- Google 스타일 독스트링
