import { describe, expect, it } from 'vitest'
import {
  airyEncircledEnergy,
  airyEnsquaredEnergy,
  airyIntensity,
  besselJ0,
  besselJ1,
} from '../../.vitepress/theme/composables/specialFunctions'

describe('Bessel functions', () => {
  it('matches tabulated values', () => {
    expect(besselJ0(0)).toBeCloseTo(1, 8)
    expect(besselJ1(0)).toBeCloseTo(0, 8)
    expect(besselJ0(1)).toBeCloseTo(0.7651976866, 7)
    expect(besselJ1(1)).toBeCloseTo(0.4400505857, 7)
    expect(besselJ0(5)).toBeCloseTo(-0.1775967713, 7)
    expect(besselJ1(10)).toBeCloseTo(0.04347274617, 7)
    expect(besselJ0(20)).toBeCloseTo(0.16702466, 7)
    expect(besselJ1(20)).toBeCloseTo(0.06683312, 7)
    expect(besselJ0(50)).toBeCloseTo(0.05581232767, 7)
  })

  it('has correct zeros', () => {
    expect(Math.abs(besselJ0(2.404825558))).toBeLessThan(1e-7)
    expect(Math.abs(besselJ1(3.831705970))).toBeLessThan(1e-7)
    expect(Math.abs(besselJ1(7.015586670))).toBeLessThan(1e-7)
  })

  it('does not diverge for large arguments (power-series failure region)', () => {
    for (let x = 0; x <= 200; x += 0.37) {
      expect(Math.abs(besselJ0(x))).toBeLessThanOrEqual(1 + 1e-8)
      expect(Math.abs(besselJ1(x))).toBeLessThanOrEqual(0.6)
    }
  })

  it('J1 is odd, J0 is even', () => {
    expect(besselJ1(-3.3)).toBeCloseTo(-besselJ1(3.3), 10)
    expect(besselJ0(-3.3)).toBeCloseTo(besselJ0(3.3), 10)
  })
})

describe('Airy helpers', () => {
  const wl = 0.55
  const N = 2.8
  const airyR = 1.22 * wl * N

  it('intensity is 1 at centre and ~0 at first dark ring', () => {
    expect(airyIntensity(0, wl, N)).toBe(1)
    expect(airyIntensity(airyR, wl, N)).toBeLessThan(1e-5)
  })

  it('encircled energy stays in [0,1] and is 83.8% at the Airy radius', () => {
    expect(airyEncircledEnergy(airyR, wl, N)).toBeCloseTo(0.838, 2)
    for (let r = 0; r < 40; r += 0.13) {
      const ee = airyEncircledEnergy(r, wl, N)
      expect(ee).toBeGreaterThanOrEqual(0)
      expect(ee).toBeLessThanOrEqual(1)
    }
  })

  it('ensquared energy lies between inscribed and circumscribed circle EE', () => {
    for (const p of [0.5, 1.0, 2.0, 5.0]) {
      const sq = airyEnsquaredEnergy(p, wl, N)
      expect(sq).toBeGreaterThanOrEqual(airyEncircledEnergy(p / 2, wl, N) - 2e-3)
      expect(sq).toBeLessThanOrEqual(airyEncircledEnergy(p / Math.SQRT2, wl, N) + 2e-3)
    }
  })

  it('ensquared energy tends to 1 for large squares', () => {
    expect(airyEnsquaredEnergy(60, wl, N, 400)).toBeGreaterThan(0.99)
  })
})
