---
title: COMPASS API Reference
description: API reference entry point for COMPASS configuration, geometry, material database, sources, solvers, analysis, and result objects.
---

# API Reference

Use this section when you need exact class names, module boundaries, configuration fields, or result objects. The reference is intentionally terse: it complements the runnable [Guide](/guide/) and the conceptual [Theory](/theory/).

## API Map

| Area | Start here | Use it for |
|---|---|---|
| Package overview | [API Map](./api-overview.md) | Understand the major modules and workflow |
| Geometry | [PixelStack](./pixel-stack.md) | Build solver-agnostic CMOS pixel stacks |
| Materials | [MaterialDB](./material-db.md) | Query wavelength-dependent optical constants |
| Solvers | [SolverBase](./solver-base.md) | Implement or call RCWA/FDTD backends |
| Sources | [Sources](./sources.md) | Configure plane waves, cone illumination, and ray inputs |
| Analysis | [Analysis](./analysis.md) | Compute QE, energy balance, and solver comparisons |
| Configuration | [Config Reference](./config-reference.md) | Check YAML schema fields and defaults |
| Terms | [Glossary](./glossary.md) | Resolve names used across theory and guides |

## When to Use This Section

- You are writing code against COMPASS classes.
- You need the exact YAML key or default value for a simulation.
- You are implementing a new solver adapter or analysis module.
- You are debugging how data flows from config to geometry to solver result.
