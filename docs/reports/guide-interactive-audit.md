---
outline: deep
---

# Guide Interactive Component Audit

Generated: 2026-05-13

This report tracks the interactive Vue components embedded in the English and Korean guide pages. It is not a physics benchmark. It is a documentation-quality validation report: every guide page should expose the same interactive learning aids in both locales, every used component should be globally registered, and obvious SVG markup defects should be caught before deployment.

## Current result

The current automated check passes:

```bash
cd docs
npm run docs:guide-check
```

Result:

```text
Guide interactive check passed for 17 guide page(s) and 21 component(s).
```

The docs deployment workflow now runs this check after the VitePress build, together with the pixel-stack visual smoke check.

## What is checked

| Check | Why it matters |
| --- | --- |
| EN/KO component parity | A Korean guide page should not silently lose an interactive viewer that exists in the English page. |
| Theme registration | VitePress markdown components must be registered in `docs/.vitepress/theme/index.ts`; otherwise the page can degrade to an unresolved custom tag. |
| Component file existence | A stale registration or renamed file should fail CI before GitHub Pages deployment. |
| Empty SVG coordinate/size attributes | Invalid attributes such as `y1=""` can survive markdown build but create brittle browser rendering. |

## Components covered

The audit currently covers the components used from `docs/guide/**/*.md` and `docs/ko/guide/**/*.md`:

| Page area | Components |
| --- | --- |
| Getting started | `PixelStackBuilder`, `WavelengthSlider`, `StackVisualizer`, `QESpectrumChart`, `EnergyBalanceDiagram` |
| Configuration | `CoordinateSystemMini`, `PixelParameterDiagram`, `PixelSectionTopView`, `MaterialBrowser` |
| Solver selection and validation | `SolverComparisonChart`, `SolverPipelineDiagram`, `PrecisionComparison`, `RCWAConvergenceDemo`, `FourierOrderDemo`, `YeeCellViewer` |
| Cone illumination | `ConeIlluminationViewer`, `ConeIlluminationTopView`, `FabryPerotConeSimulator` |
| Signal and system workflow | `SignalChainDiagram`, `BlackbodySpectrum`, `ModuleArchitectureDiagram` |

## Fixes made in this audit

- Added `docs/scripts/check-guide-interactives.mjs`.
- Added `npm run docs:guide-check`.
- Wired the check into `.github/workflows/docs-check.yml` and `.github/workflows/docs.yml`.
- Fixed an invalid SVG tick mark in `MaterialBrowser.vue` where both `y1=""` and `:y1="plotBottom"` were present.

## Scope and next step

This check is intentionally lightweight and dependency-free. It does not yet compare rendered screenshots, canvas pixels, hover states, or mobile layout. Those require a browser runner such as Playwright.

The next upgrade should add a browser smoke test for the heaviest guide interactives:

1. `PixelParameterDiagram` - XZ/XY tabs and hover highlighting.
2. `ConeIlluminationViewer` - CRA shift stays inside the physical pixel view.
3. `ConeIlluminationTopView` - sampling method changes produce distinct non-grid point distributions.
4. `FabryPerotConeSimulator` - chart paths and top-view sample points remain nonempty.
5. `MaterialBrowser` - hover readout and n/k curves render for each material.
