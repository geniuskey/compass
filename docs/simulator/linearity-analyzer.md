---
title: "Linearity Analyzer"
---

# Linearity Analyzer

Analyze sensor transfer curve linearity with adjustable non-linearity and knee point. View transfer function and residual plots side by side.

<LinearityAnalyzer />

## Linearity Model

The ideal transfer function is linear:

**DN_ideal = (Signal / FWC) × DN_max**

Real sensors deviate from linearity due to:
- Charge-to-voltage conversion non-linearity (source follower)
- Capacitance variation with voltage (junction capacitance)
- ADC integral non-linearity (INL)

### Non-Linearity Metric

**NL(%) = max|DN_actual - DN_ideal| / DN_max × 100**

### Knee Point

The knee point defines where non-linearity onset becomes significant. Below the knee point, response is nearly linear; above it, compression or expansion effects increase.

### Key Metrics
- **Max Non-Linearity %** — worst-case deviation from ideal
- **Linear Range %** — percentage of full scale within ±1% NL
- **RMS Error %** — root-mean-square deviation across full range

::: warning
High non-linearity (>2%) degrades color accuracy, HDR merging quality, and photometric measurements. Most machine vision applications require NL < 1%.
:::
