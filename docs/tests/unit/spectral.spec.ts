import { describe, expect, it } from 'vitest'
import {
  cmfOn,
  daylightSpd,
  fitReflectanceToXyz,
  illuminantA,
  reflectanceXyz,
} from '../../.vitepress/theme/composables/spectral'

const WL = Array.from({ length: 31 }, (_, i) => 400 + i * 10)

describe('CIE illuminants', () => {
  it('reproduces tabulated D65 values', () => {
    // CIE D65 at 400, 450, 500, 550, 600, 650, 700 nm
    const ref = [82.75, 117.01, 109.35, 104.05, 90.01, 80.21, 71.61]
    const d65 = daylightSpd(6504, [400, 450, 500, 550, 600, 650, 700])
    d65.forEach((v, i) => expect(Math.abs(v - ref[i]) / ref[i]).toBeLessThan(0.01))
  })

  it('puts the D65 and D50 white points at their CIE chromaticities', () => {
    const cmf = cmfOn(WL)
    const white = WL.map(() => 1)
    for (const [cct, x0, y0] of [[6504, 0.3127, 0.329], [5003, 0.3457, 0.3585]]) {
      const [X, Y, Z] = reflectanceXyz(white, daylightSpd(cct, WL), cmf)
      expect(X / (X + Y + Z)).toBeCloseTo(x0, 2)
      expect(Y / (X + Y + Z)).toBeCloseTo(y0, 2)
    }
  })

  it('matches illuminant A at reference wavelengths', () => {
    const a = illuminantA([400, 560, 700])
    expect(a[0]).toBeCloseTo(14.71, 1)
    expect(a[1]).toBeCloseTo(100, 6)
    expect(a[2]).toBeCloseTo(198.26, 0)
  })
})

describe('fitReflectanceToXyz', () => {
  it('hits the target XYZ while staying a valid reflectance', () => {
    const cmf = cmfOn(WL)
    const d65 = daylightSpd(6504, WL)
    // A deliberately wrong starting shape (peaks in the green) fitted to a sky-blue target
    const r0 = WL.map(wl => 0.1 + 0.1 * Math.exp(-0.5 * ((wl - 550) / 40) ** 2))
    const target = [0.1690, 0.1898, 0.3316] // linear sRGB (98,122,157) → XYZ
    const r = fitReflectanceToXyz(r0, WL, target, d65, cmf)
    const xyz = reflectanceXyz(r, d65, cmf)
    xyz.forEach((v, k) => expect(v).toBeCloseTo(target[k], 4))
    r.forEach(v => { expect(v).toBeGreaterThan(0); expect(v).toBeLessThanOrEqual(1) })
    // Blue end must now exceed the red end
    expect(r[5]).toBeGreaterThan(r[25])
  })
})
