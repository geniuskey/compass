---
layout: home

hero:
  name: "COMPASS"
  text: "고급 센서 시뮬레이션을 위한 크로스 솔버 광학 모델링 플랫폼"
  tagline: RCWA 및 FDTD 솔버를 사용하여 BSI CMOS 이미지 센서 픽셀의 양자 효율을 시뮬레이션합니다
  actions:
    - theme: brand
      text: 시작하기
      link: /ko/guide/installation
    - theme: alt
      text: 이론 배우기
      link: /ko/theory/light-basics
    - theme: alt
      text: GitHub에서 보기
      link: https://github.com/compass-sim/compass

features:
  - title: 멀티 솔버 지원
    details: 동일한 픽셀 구조를 RCWA(torcwa, grcwa, meent) 및 FDTD(fdtd, meep) 솔버로 실행하고 결과를 직접 비교할 수 있습니다.
  - title: 파라메트릭 픽셀 모델링
    details: 마이크로렌즈, 컬러 필터, BARL, 실리콘, DTI 등 완전한 BSI 픽셀 스택을 하나의 YAML 설정으로 정의하고 모든 파라미터를 스윕할 수 있습니다.
  - title: 내장 시각화
    details: QE 스펙트럼, 필드 분포, 크로스토크 행렬, 3D 구조 뷰를 matplotlib과 PyVista로 바로 시각화할 수 있습니다.
  - title: 수치 안정성
    details: 혼합 정밀도 고유값 분해, S-행렬 재귀, Li 분해, 적응형 폴백을 포함한 5중 안정성 방어 체계를 갖추고 있습니다.
---

## COMPASS란 무엇인가?

COMPASS는 후면 조사(BSI, Backside-Illuminated) CMOS 이미지 센서 픽셀의 광학 성능을 시뮬레이션하기 위한 Python 프레임워크입니다. 전자기(EM) 이론과 실용적 센서 설계 사이의 간극을 해소하기 위해 여러 솔버 백엔드에 대한 통합 인터페이스를 제공합니다.

픽셀 스택 정의(마이크로렌즈 기하학, 컬러 필터 패턴, 반사 방지 코팅, 실리콘 포토다이오드)가 주어지면, COMPASS는 파장, 각도, 편광에 걸쳐 **양자 효율(QE)** -- 입사 광자 중 각 포토다이오드에서 전자-정공 쌍을 생성하는 비율 -- 을 계산합니다.

### 일반적인 워크플로

```
YAML config  -->  PixelStack  -->  Solver (RCWA / FDTD)  -->  QE spectrum
                                                           -->  Field maps
                                                           -->  Energy balance
```

## 빠른 예제

```python
from compass.runners.single_run import SingleRunner

result = SingleRunner.run({
    "pixel": {"pitch": 1.0, "unit_cell": [2, 2]},
    "solver": {"name": "torcwa", "type": "rcwa"},
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.4, "stop": 0.7, "step": 0.01}},
        "polarization": "unpolarized",
    },
})

# result.qe_per_pixel is a dict mapping pixel names to QE arrays
for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```
