---
outline: deep
---

# Simulation Reports

Publication-style reports generated from Python benchmark artifacts. These pages are written for both readers who want the current engineering conclusion and maintainers who need the validation trail behind that conclusion.

## Available reports

- [RCWA/FDTD Convergence Analysis](./convergence-analysis.md) (generated 2026-05-05)

## What belongs here

- Cross-solver validation results that should be inspectable from GitHub Pages.
- Plots and tables promoted from local `outputs/` artifacts into `docs/public/reports/`.
- Reproducibility notes that explain which scripts regenerate the published figures.

The report assets are served from `docs/public/reports/`, so they are included in the VitePress build and the GitHub Pages deployment.
