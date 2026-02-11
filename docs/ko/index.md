---
layout: home

hero:
  name: "COMPASS"
  text: "고급 센서 시뮬레이션을 위한 크로스 솔버 광학 모델링 플랫폼"
  tagline: 하나의 YAML 설정으로 여러 EM 솔버를 사용하여 CMOS 이미지 센서 픽셀을 시뮬레이션합니다
  image:
    src: /logo.svg
    alt: COMPASS
  actions:
    - theme: brand
      text: 입문 — 이미지 센서 기초
      link: /ko/introduction/what-is-cmos-sensor
    - theme: alt
      text: 시작하기
      link: /ko/guide/installation
    - theme: alt
      text: GitHub
      link: https://github.com/geniuskey/compass
features:
  - title: "\U0001F4D6 입문자 친화적"
    details: 처음부터 시작하세요 — 시뮬레이션에 들어가기 전에 이미지 센서 광학 기초를 배우세요
    link: /ko/introduction/what-is-cmos-sensor
---

<HeroAnimation />

## COMPASS를 선택하는 이유

COMPASS는 전자기 이론과 실용적인 CMOS 이미지 센서 설계 사이의 간극을 해소합니다. 픽셀 스택을 한 번 정의하고, 모든 솔버로 실행하고, 결과를 비교하세요 -- 모두 Python에서 가능합니다.

<FeatureShowcase />

## 아키텍처

깔끔한 5단계 파이프라인이 YAML 설정에서 출판 가능한 결과까지 안내합니다. 각 단계를 클릭하면 자세히 알아볼 수 있습니다.

<ArchitectureOverview />

## 솔버 백엔드

COMPASS는 세 가지 전자기 방법에 걸쳐 **8개 솔버 백엔드**에 대한 통합 인터페이스를 제공합니다. 솔버를 클릭하면 세부 정보를 확인할 수 있습니다.

<SolverShowcase />

## 빠른 예제

하나의 YAML 설정으로 시뮬레이션을 정의하고 세 줄의 Python으로 실행하세요:

```yaml
# config.yaml
pixel:
  pitch: 1.0          # um
  unit_cell: [2, 2]   # 2x2 베이어 패턴

solver:
  name: torcwa
  type: rcwa
  fourier_order: 9

source:
  wavelength:
    mode: sweep
    sweep: { start: 0.4, stop: 0.7, step: 0.01 }
  polarization: unpolarized
```

```python
from compass.runners.single_run import SingleRunner

result = SingleRunner.run("config.yaml")

for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```

<div class="landing-cta-section">

## 시작하기

<div class="cta-grid">
<a href="/ko/introduction/what-is-cmos-sensor" class="cta-card">
  <strong>이미지 센서 기초</strong>
  <span>이미지 센서가 처음이라면 여기서 시작하세요</span>
</a>
<a href="/ko/guide/installation" class="cta-card">
  <strong>설치 가이드</strong>
  <span>COMPASS와 솔버 백엔드를 설정하세요</span>
</a>
<a href="/ko/guide/quickstart" class="cta-card">
  <strong>빠른 시작</strong>
  <span>몇 분 안에 첫 번째 시뮬레이션을 실행하세요</span>
</a>
<a href="/ko/theory/light-basics" class="cta-card">
  <strong>이론 배경</strong>
  <span>시뮬레이션의 물리학을 이해하세요</span>
</a>
<a href="/ko/cookbook/bsi-2x2-basic" class="cta-card">
  <strong>쿡북</strong>
  <span>일반적인 작업을 위한 실용적인 레시피</span>
</a>
</div>

</div>
