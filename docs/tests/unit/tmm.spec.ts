import { describe, expect, it } from 'vitest'
import {
  MATERIALS,
  SI_LAYER_IDX,
  defaultBsiStack,
  getN,
  tmmCalc,
  wlRange,
} from '../../.vitepress/theme/composables/tmm'

describe('silicon optical constants', () => {
  // Literature anchors: penetration depth 1/α = λ/(4πk) for c-Si at 300K
  // (Green 2008): ≈0.10µm @400nm, ≈1.4µm @550nm, ≈2.4µm @600nm, ≈12µm @800nm
  const cases: Array<[number, number, number]> = [
    [0.4, 0.08, 0.13],
    [0.55, 1.1, 1.8],
    [0.6, 1.9, 3.0],
    [0.8, 9, 15],
    [1.0, 100, 220],
  ]
  it.each(cases)('penetration depth at %f µm is in literature range', (wl, lo, hi) => {
    const [, k] = getN(MATERIALS.silicon, wl)
    const depth = wl / (4 * Math.PI * k)
    expect(depth).toBeGreaterThan(lo)
    expect(depth).toBeLessThan(hi)
  })

  it('k decreases monotonically from 400nm to 1000nm (no interband spikes)', () => {
    let prev = Infinity
    for (const wl of wlRange(0.4, 1.0, 0.01)) {
      const [, k] = getN(MATERIALS.silicon, wl)
      expect(k).toBeLessThanOrEqual(prev + 1e-12)
      prev = k
    }
  })

  it('n matches HeNe reference n(633nm) ≈ 3.88', () => {
    const [n] = getN(MATERIALS.silicon, 0.633)
    expect(n).toBeGreaterThan(3.83)
    expect(n).toBeLessThan(3.93)
  })
})

describe('tmmCalc energy balance', () => {
  const stacks: Array<['red' | 'green' | 'blue', number]> = [
    ['red', 3.0],
    ['green', 3.0],
    ['blue', 3.0],
  ]
  it.each(stacks)('R + T + A = 1 for %s pixel stack across spectrum and angle', (cf) => {
    for (const wl of wlRange(0.4, 0.7, 0.05)) {
      for (const ang of [0, 15, 30]) {
        const r = tmmCalc(defaultBsiStack(cf), 'air', 'sio2', wl, ang, 'avg')
        expect(r.R + r.T + r.A).toBeCloseTo(1, 2)
        const layerSum = r.layerA.reduce((s, a) => s + a, 0)
        expect(Math.abs(layerSum - r.A)).toBeLessThan(0.02)
      }
    }
  })

  it('QE (silicon absorption) stays within [0, 1] and peaks in the filter passband', () => {
    const qeAt = (cf: 'red' | 'green' | 'blue', wl: number) =>
      tmmCalc(defaultBsiStack(cf), 'air', 'sio2', wl, 0, 'avg').layerA[SI_LAYER_IDX]
    for (const cf of ['red', 'green', 'blue'] as const) {
      for (const wl of wlRange(0.4, 0.7, 0.02)) {
        const qe = qeAt(cf, wl)
        expect(qe).toBeGreaterThanOrEqual(0)
        expect(qe).toBeLessThanOrEqual(1)
      }
    }
    expect(qeAt('green', 0.53)).toBeGreaterThan(qeAt('green', 0.45))
    expect(qeAt('green', 0.53)).toBeGreaterThan(qeAt('green', 0.65))
    expect(qeAt('blue', 0.45)).toBeGreaterThan(qeAt('blue', 0.6))
    expect(qeAt('red', 0.62)).toBeGreaterThan(qeAt('red', 0.5))
  })

  it('3µm silicon transmits most light at 1000nm (weak NIR absorption)', () => {
    const r = tmmCalc([{ material: 'silicon', thickness: 3.0 }], 'air', 'sio2', 1.0, 0, 'avg')
    // 1/α ≈ 156µm at 1000nm → single-pass absorption of 3µm Si is only a few percent
    expect(r.layerA[0]).toBeLessThan(0.15)
  })
})
