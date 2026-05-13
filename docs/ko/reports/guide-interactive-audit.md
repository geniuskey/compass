---
outline: deep
---

# 가이드 인터랙티브 컴포넌트 감사

생성일: 2026-05-13

이 리포트는 영어/한국어 가이드 페이지에 삽입된 인터랙티브 Vue 컴포넌트를 추적한다. 물리 solver benchmark가 아니라 문서 품질 검증 리포트다. 두 언어의 가이드가 같은 학습 도구를 제공하는지, 사용된 컴포넌트가 전역 등록되어 있는지, SVG 마크업의 명백한 결함이 배포 전에 잡히는지를 확인한다.

## 현재 결과

현재 자동 검사는 통과한다:

```bash
cd docs
npm run docs:guide-check
```

결과:

```text
Guide interactive check passed for 17 guide page(s) and 21 component(s).
```

문서 배포 workflow는 이제 VitePress build 이후 이 검사를 실행한다. pixel-stack visual smoke check도 함께 실행된다.

## 검사 항목

| 검사 | 중요한 이유 |
| --- | --- |
| EN/KO component parity | 영어 가이드에 있는 인터랙티브 뷰어가 한국어 가이드에서 조용히 누락되면 안 된다. |
| Theme registration | VitePress markdown 컴포넌트는 `docs/.vitepress/theme/index.ts`에 등록되어 있어야 한다. 그렇지 않으면 페이지가 unresolved custom tag로 degraded될 수 있다. |
| Component file existence | 등록만 남고 파일이 rename/delete된 경우 GitHub Pages 배포 전에 실패해야 한다. |
| Empty SVG coordinate/size attributes | `y1=""` 같은 잘못된 속성은 build를 통과할 수 있지만 브라우저 렌더링을 취약하게 만든다. |

## 적용 범위

현재 감사는 `docs/guide/**/*.md`와 `docs/ko/guide/**/*.md`에서 사용되는 컴포넌트를 대상으로 한다:

| 페이지 영역 | 컴포넌트 |
| --- | --- |
| 시작하기 | `PixelStackBuilder`, `WavelengthSlider`, `StackVisualizer`, `QESpectrumChart`, `EnergyBalanceDiagram` |
| 설정 | `CoordinateSystemMini`, `PixelParameterDiagram`, `PixelSectionTopView`, `MaterialBrowser` |
| 솔버 선택과 검증 | `SolverComparisonChart`, `SolverPipelineDiagram`, `PrecisionComparison`, `RCWAConvergenceDemo`, `FourierOrderDemo`, `YeeCellViewer` |
| 원뿔 조명 | `ConeIlluminationViewer`, `ConeIlluminationTopView`, `FabryPerotConeSimulator` |
| 신호와 시스템 workflow | `SignalChainDiagram`, `BlackbodySpectrum`, `ModuleArchitectureDiagram` |

## 이번 감사에서 고친 것

- `docs/scripts/check-guide-interactives.mjs` 추가.
- `npm run docs:guide-check` 추가.
- `.github/workflows/docs-check.yml`와 `.github/workflows/docs.yml`에 guide check 연결.
- `MaterialBrowser.vue`의 SVG tick mark에서 `y1=""`와 `:y1="plotBottom"`이 동시에 있던 잘못된 마크업 수정.

## 범위와 다음 단계

이 검사는 의존성 없는 가벼운 구조 검사다. 아직 렌더링 스크린샷, canvas pixel, hover 상태, 모바일 레이아웃 비교까지 보장하지는 않는다. 그런 검사는 Playwright 같은 브라우저 runner가 필요하다.

다음 업그레이드는 무거운 guide interactive에 대한 browser smoke test가 적합하다:

1. `PixelParameterDiagram` - XZ/XY tab과 hover highlight.
2. `ConeIlluminationViewer` - CRA shift가 물리 pixel view 안에 유지되는지.
3. `ConeIlluminationTopView` - sampling method 변경 시 grid가 아닌 서로 다른 point distribution이 나타나는지.
4. `FabryPerotConeSimulator` - chart path와 top-view sample point가 비어 있지 않은지.
5. `MaterialBrowser` - material별 hover readout과 n/k curve 렌더링.
