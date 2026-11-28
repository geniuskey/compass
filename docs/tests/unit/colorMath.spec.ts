import { describe, expect, it } from 'vitest'
import {
  cond2,
  fitWhitePreservingCcm,
  interp1,
  mat3Vec,
  sym3Eigenvalues,
} from '../../.vitepress/theme/composables/colorMath'

describe('interp1', () => {
  it('interpolates linearly and clamps at the ends', () => {
    const xs = [400, 450, 500]
    const ys = [0, 1, 3]
    expect(interp1(xs, ys, 425)).toBeCloseTo(0.5)
    expect(interp1(xs, ys, 480)).toBeCloseTo(2.2)
    expect(interp1(xs, ys, 300)).toBe(0)
    expect(interp1(xs, ys, 600)).toBe(3)
    expect(interp1(xs, ys, 450)).toBeCloseTo(1)
  })
})

describe('sym3Eigenvalues / cond2', () => {
  it('matches a diagonal matrix', () => {
    expect(sym3Eigenvalues([[3, 0, 0], [0, 1, 0], [0, 0, 2]])).toEqual([1, 2, 3])
    expect(cond2([[4, 0, 0], [0, 1, 0], [0, 0, 2]])).toBeCloseTo(4)
  })

  it('finds eigenvalues of a dense symmetric matrix', () => {
    // eigenvalues of [[2,1,0],[1,2,1],[0,1,2]] are 2-sqrt2, 2, 2+sqrt2
    const ev = sym3Eigenvalues([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    expect(ev[0]).toBeCloseTo(2 - Math.SQRT2, 10)
    expect(ev[1]).toBeCloseTo(2, 10)
    expect(ev[2]).toBeCloseTo(2 + Math.SQRT2, 10)
  })

  it('is invariant to orthogonal rotation and is the 2-norm condition number', () => {
    const c = Math.cos(0.3), s = Math.sin(0.3)
    // R * diag(5, 2, 1)
    const M = [[5 * c, -2 * s, 0], [5 * s, 2 * c, 0], [0, 0, 1]]
    expect(cond2(M)).toBeCloseTo(5, 8)
  })

  it('returns Infinity for a singular matrix', () => {
    expect(cond2([[1, 1, 0], [1, 1, 0], [0, 0, 1]])).toBe(Infinity)
  })
})

describe('fitWhitePreservingCcm', () => {
  const trueM = [[1.6, -0.4, -0.2], [-0.3, 1.5, -0.2], [0.05, -0.55, 1.5]]
  const S = [
    [0.2, 0.3, 0.1], [0.5, 0.4, 0.2], [0.1, 0.2, 0.6], [0.7, 0.6, 0.5],
    [0.3, 0.1, 0.05], [0.05, 0.4, 0.3], [0.9, 0.9, 0.9], [0.4, 0.4, 0.4],
  ]

  it('recovers the exact matrix when data are consistent', () => {
    const T = S.map(s => mat3Vec(trueM, s))
    const w = [0.9, 0.9, 0.9]
    const M = fitWhitePreservingCcm(S, T, w, mat3Vec(trueM, w))!
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) expect(M[i][j]).toBeCloseTo(trueM[i][j], 8)
  })

  it('maps the reference white exactly even with noisy targets', () => {
    const T = S.map((s, k) => mat3Vec(trueM, s).map((v, i) => v + 0.03 * Math.sin(k * 3 + i)))
    const w = [0.8, 0.85, 0.7]
    const tw = [0.9, 0.9, 0.88]
    const M = fitWhitePreservingCcm(S, T, w, tw)!
    const out = mat3Vec(M, w)
    for (let i = 0; i < 3; i++) expect(out[i]).toBeCloseTo(tw[i], 10)
  })

  it('returns null for rank-deficient data', () => {
    const Sd = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    expect(fitWhitePreservingCcm(Sd, Sd, [1, 1, 1], [1, 1, 1])).toBeNull()
  })
})
