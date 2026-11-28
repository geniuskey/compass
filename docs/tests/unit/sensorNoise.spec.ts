import { describe, expect, it } from 'vitest'
import {
  clippedGaussianMoments,
  emvaThresholdE,
  erf,
  sensorIlluminance,
  totalNoiseE,
} from '../../.vitepress/theme/composables/sensorNoise'

describe('erf', () => {
  it('matches reference values', () => {
    expect(erf(0)).toBeCloseTo(0, 6)
    expect(erf(1)).toBeCloseTo(0.8427008, 6)
    expect(erf(-0.5)).toBeCloseTo(-0.5204999, 6)
  })
})

describe('clippedGaussianMoments', () => {
  it('is unclipped far below saturation', () => {
    const r = clippedGaussianMoments(1000, 31.6, 10000)
    expect(r.mean).toBeCloseTo(1000, 6)
    expect(r.sigma).toBeCloseTo(31.6, 6)
  })

  it('collapses the variance above saturation (PTC roll-off)', () => {
    const r = clippedGaussianMoments(12000, 110, 10000)
    expect(r.mean).toBeCloseTo(10000, 3)
    expect(r.sigma).toBeLessThan(1e-3)
  })

  it('halves the variance at mu = sat', () => {
    // Var(min(X, mu)) = σ²(1/2 − 1/(2π)) for X ~ N(mu, σ²)
    const r = clippedGaussianMoments(100, 10, 100)
    expect(r.sigma ** 2).toBeCloseTo(100 * (0.5 - 1 / (2 * Math.PI)), 3)
    expect(r.mean).toBeCloseTo(100 - 10 / Math.sqrt(2 * Math.PI), 4)
  })
})

describe('noise budget', () => {
  it('adds terms in quadrature', () => {
    expect(totalNoiseE(0, { readNoise: 3 })).toBeCloseTo(3)
    expect(totalNoiseE(16, { readNoise: 3 })).toBeCloseTo(5)
    expect(totalNoiseE(10000, { readNoise: 0, prnu: 0.01 })).toBeCloseTo(Math.sqrt(10000 + 10000))
  })

  it('uses the EMVA 1288 absolute sensitivity threshold', () => {
    expect(emvaThresholdE(0)).toBeCloseTo(1)
    expect(emvaThresholdE(3)).toBeCloseTo(0.5 + Math.sqrt(9.25))
  })

  it('converts scene to sensor-plane illuminance', () => {
    // f/2, 18 % grey, 90 % lens → 1000 lux scene ≈ 10.1 lux at the sensor
    expect(sensorIlluminance(1000, 2)).toBeCloseTo(10.125, 3)
  })
})
