---
title: COMPASS Research Notes
description: Research notes on electromagnetic solvers, CMOS image sensor technology trends, simulation methods, key papers, and validation strategy.
---

# Research

This section collects background research that informs COMPASS design decisions. It is not a step-by-step guide; use it to understand why a solver, benchmark, or sensor-modeling assumption was chosen.

## Research Map

| Topic | Start here | Why it matters |
|---|---|---|
| Theory coverage | [Theory Coverage Map](./theory-coverage-map.md) | Keep theory pages scoped and avoid duplicate explanations |
| Solver ecosystem | [EM Solver Survey](./open-source-em-solvers-survey.md) | Compare open-source RCWA/FDTD options and licensing |
| Sensor direction | [CIS Technology Trends](./cis-technology-trends.md) | Track pixel scaling, BSI structures, DTI, and optical stack trends |
| Method choice | [Simulation Methods](./simulation-methods-comparison.md) | Compare TMM, RCWA, FDTD, and hybrid validation strategies |
| Literature | [Key Papers](./key-papers.md) | Anchor implementation choices in published work |
| Validation | [Benchmarks & Validation](./benchmarks-and-validation.md) | Define how COMPASS proves solver correctness |

## How to Use These Notes

- Read them before adding a new solver backend.
- Link them from issues or PRs when a design decision depends on external evidence.
- Use [Reports](/reports/) for generated benchmark evidence and this section for broader technical context.
