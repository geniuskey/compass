---
layout: home
title: COMPASS — 오픈소스 CMOS 이미지 센서 광학 플랫폼
titleTemplate: false

hero:
  name: "COMPASS"
  text: "픽셀 설정 하나, 9개의 EM 솔버, 교차 검증."
  tagline: 오픈소스 CMOS 이미지 센서 광학 플랫폼 — 픽셀 스택을 한 번 정의하면 9개의 RCWA/FDTD/TMM 백엔드로 QE·크로스토크·필드 맵을 교차 검증합니다.
  image:
    src: /logo.svg
    alt: COMPASS
  actions:
    - theme: brand
      text: 시작하기
      link: /ko/guide/quickstart
    - theme: alt
      text: 이미지 센서가 처음이라면
      link: /ko/theory/basics/what-is-cmos-sensor
    - theme: alt
      text: GitHub
      link: https://github.com/geniuskey/compass

features:
  - icon: 📖
    title: 물리부터 배우기
    details: 픽셀 구조, 박막, 회절, QE, RCWA와 FDTD까지 — 이미지 센서 광학을 기초부터 설명합니다.
    link: /ko/theory/
    linkText: 학습 시작
  - icon: 🐍
    title: Python 파이프라인 실행
    details: YAML 설정 하나로 9개 솔버를 실행하고, 픽셀별 QE·크로스토크·필드 맵을 비교합니다.
    link: /ko/guide/
    linkText: 가이드 보기
  - icon: 🧪
    title: 브라우저에서 바로 실험
    details: TMM QE, SNR, MTF, 마이크로렌즈 광선추적 등 26개 시뮬레이터 — 설치가 필요 없습니다.
    link: /ko/simulator/
    linkText: 시뮬레이터 열기
  - icon: ✅
    title: 검증 근거 확인
    details: 솔버 간 일치도, 수렴 연구, 벤치마크를 생성 스크립트와 함께 공개합니다.
    link: /ko/reports/
    linkText: 리포트 보기
---

<div class="home-stats">
  <div class="home-stat"><strong>9</strong><span>EM 솔버 백엔드</span></div>
  <div class="home-stat"><strong>3</strong><span>방법: RCWA · FDTD · TMM</span></div>
  <div class="home-stat"><strong>26</strong><span>브라우저 시뮬레이터</span></div>
  <div class="home-stat"><strong>EN · KO</strong><span>이중 언어 문서</span></div>
</div>

<div class="home-eyebrow">문제 정의</div>

## 광자에서 전자까지

빛은 마이크로렌즈, 컬러 필터, 반사 방지막을 지나 실리콘에서 흡수됩니다. 서브마이크론 피치에서는 모든 층이 파동광학 문제가 되고, 솔버마다 답이 달라질 수 있습니다. COMPASS는 스택을 한 번만 모델링하고 여러 독립적인 방법으로 결과를 교차 확인하게 해줍니다.

<HeroAnimation />

<div class="home-eyebrow">왜 COMPASS인가</div>

## 교차 검증을 위한 픽셀 광학 플랫폼

<FeatureShowcase />

<div class="home-eyebrow">아키텍처</div>

## 5단계 파이프라인

YAML 설정에서 출판 가능한 결과까지. 각 단계를 클릭하면 자세히 볼 수 있습니다.

<ArchitectureOverview />

<div class="home-eyebrow">백엔드</div>

## 하나의 인터페이스, 9개의 솔버

세 가지 전자기 방법을 하나의 `SolverBase` API로 다룹니다. 솔버를 클릭하면 세부 정보를 확인할 수 있습니다.

<SolverShowcase />

<div class="home-eyebrow">빠른 예제</div>

## 한 번 정의하고 어디서나 실행

픽셀을 YAML로 기술하고 몇 줄의 Python으로 실행합니다. `solver.name`만 바꾸면 같은 구조를 다른 백엔드로 교차 확인할 수 있습니다.

::: code-group

```yaml [config.yaml]
pixel:
  pitch: 1.0          # um
  unit_cell: [2, 2]   # 2x2 베이어 패턴

solver:
  name: torcwa        # 또는 grcwa, meent, fmmax, fdtd_flaport, meep, tmm ...
  type: rcwa
  params:
    fourier_order: [9, 9]

source:
  wavelength:
    mode: sweep
    sweep: { start: 0.40, stop: 0.70, step: 0.01 }
  polarization: unpolarized
```

```python [run.py]
import yaml
from compass.runners.single_run import SingleRunner

with open("config.yaml") as f:
    config = yaml.safe_load(f)

result = SingleRunner.run(config)

for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```

```bash [CLI (Hydra)]
python scripts/run_simulation.py solver=torcwa source=wavelength_sweep
python scripts/run_simulation.py solver=grcwa  source=wavelength_sweep  # 교차 확인
```

:::

<div class="landing-cta-section">

## 시작하기

원하는 지점에서 시작하세요 — 모든 페이지는 언어 메뉴에서 영어로 전환할 수 있습니다.

<HomeCta />

</div>
