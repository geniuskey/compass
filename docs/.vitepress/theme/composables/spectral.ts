/**
 * Spectral helpers for the colour simulators: CIE standard illuminants computed
 * from their definitions, and a colorimetric fit that adjusts a reflectance
 * spectrum so it reproduces a target CIE XYZ.
 *
 * Wavelengths are in nanometres. Pure functions, unit-tested.
 */

import { CIE_WL, CIE_X, CIE_Y, CIE_Z } from './tmm'

/** CIE daylight basis functions S0, S1, S2 (CIE 15:2004), 400–700 nm @ 10 nm. */
const DAYLIGHT_WL = Array.from({ length: 31 }, (_, i) => 400 + i * 10)
const S0 = [94.8, 104.8, 105.9, 96.8, 113.9, 125.6, 125.5, 121.3, 121.3, 113.5, 113.1, 110.8, 106.5, 108.8, 105.3, 104.4, 100.0, 96.0, 95.1, 89.1, 90.5, 90.3, 88.4, 84.0, 85.1, 81.9, 82.6, 84.9, 81.3, 71.9, 74.3]
const S1 = [43.4, 46.3, 43.9, 37.1, 36.7, 35.9, 32.6, 27.9, 24.3, 20.1, 16.2, 13.2, 8.6, 6.1, 4.2, 1.9, 0.0, -1.6, -3.5, -3.5, -5.8, -7.2, -8.6, -9.5, -10.9, -10.7, -12.0, -14.0, -13.6, -12.0, -13.3]
const S2 = [-1.1, -0.5, -0.7, -1.2, -2.6, -2.9, -2.8, -2.6, -2.6, -1.8, -1.5, -1.3, -1.2, -1.0, -0.5, -0.3, 0.0, 0.2, 0.5, 2.1, 3.2, 4.1, 4.7, 5.1, 6.7, 7.3, 8.6, 9.8, 10.2, 8.3, 9.6]

function lerp(xs: number[], ys: number[], x: number): number {
  if (x <= xs[0]) return ys[0]
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1]
  let i = 1
  while (xs[i] < x) i++
  const f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
  return ys[i - 1] * (1 - f) + ys[i] * f
}

/**
 * CIE D-series daylight SPD (relative, 100 at 560 nm) for a correlated colour
 * temperature `cct` in kelvin. D65 uses cct = 6504 K, D50 uses 5003 K.
 */
export function daylightSpd(cct: number, wlNm: number[]): number[] {
  const t = cct
  const xD = t <= 7000
    ? -4.607e9 / t ** 3 + 2.9678e6 / t ** 2 + 0.09911e3 / t + 0.244063
    : -2.0064e9 / t ** 3 + 1.9018e6 / t ** 2 + 0.24748e3 / t + 0.23704
  const yD = -3 * xD * xD + 2.87 * xD - 0.275
  const M = 0.0241 + 0.2562 * xD - 0.7341 * yD
  const M1 = (-1.3515 - 1.7703 * xD + 5.9114 * yD) / M
  const M2 = (0.03 - 31.4424 * xD + 30.0717 * yD) / M
  return wlNm.map(wl => lerp(DAYLIGHT_WL, S0, wl) + M1 * lerp(DAYLIGHT_WL, S1, wl) + M2 * lerp(DAYLIGHT_WL, S2, wl))
}

/** CIE standard illuminant A (relative, 100 at 560 nm), from its defining Planck formula. */
export function illuminantA(wlNm: number[]): number[] {
  const c2 = 1.435e7 // nm·K, as fixed in the CIE definition
  const T = 2848
  return wlNm.map(wl => 100 * (560 / wl) ** 5 * (Math.exp(c2 / (T * 560)) - 1) / (Math.exp(c2 / (T * wl)) - 1))
}

/** CIE 1931 2° colour-matching functions resampled onto `wlNm`. */
export function cmfOn(wlNm: number[]): { x: number[]; y: number[]; z: number[] } {
  const wl = CIE_WL.map(w => w * 1000)
  return {
    x: wlNm.map(w => lerp(wl, CIE_X, w)),
    y: wlNm.map(w => lerp(wl, CIE_Y, w)),
    z: wlNm.map(w => lerp(wl, CIE_Z, w)),
  }
}

/** XYZ of a reflectance under `spd`, normalised so a perfect white diffuser has Y = 1. */
export function reflectanceXyz(refl: number[], spd: number[], cmf: { x: number[]; y: number[]; z: number[] }): number[] {
  let X = 0, Y = 0, Z = 0, N = 0
  for (let i = 0; i < refl.length; i++) {
    const w = spd[i]
    X += refl[i] * w * cmf.x[i]
    Y += refl[i] * w * cmf.y[i]
    Z += refl[i] * w * cmf.z[i]
    N += w * cmf.y[i]
  }
  return [X / N, Y / N, Z / N]
}

/**
 * Smoothly adjust a reflectance spectrum so that, under `spd`, it reproduces
 * `targetXyz` exactly (to ~1e-6), while keeping the original spectral shape as
 * much as possible:
 *
 *   r(λ) = clamp( r0(λ) · exp( Σ_k c_k B_k(λ) ), 1e-4, 1 )
 *
 * with three broad Gaussian bases B_k centred at 450 / 540 / 610 nm. The three
 * coefficients are solved by Gauss–Newton (3 equations, 3 unknowns).
 */
export function fitReflectanceToXyz(
  r0: number[],
  wlNm: number[],
  targetXyz: number[],
  spd: number[],
  cmf: { x: number[]; y: number[]; z: number[] },
): number[] {
  const centres = [450, 540, 610]
  const basis = centres.map(c => wlNm.map(wl => Math.exp(-0.5 * ((wl - c) / 45) ** 2)))
  const base = r0.map(v => Math.max(1e-4, v))
  const build = (c: number[]) =>
    base.map((v, i) => Math.min(1, Math.max(1e-4, v * Math.exp(c[0] * basis[0][i] + c[1] * basis[1][i] + c[2] * basis[2][i]))))

  let c = [0, 0, 0]
  for (let iter = 0; iter < 40; iter++) {
    const r = build(c)
    const f = reflectanceXyz(r, spd, cmf).map((v, k) => v - targetXyz[k])
    if (Math.max(...f.map(Math.abs)) < 1e-7) break
    // Numerical Jacobian J[k][j] = ∂XYZ_k / ∂c_j
    const J = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    const h = 1e-4
    for (let j = 0; j < 3; j++) {
      const cp = c.slice(); cp[j] += h
      const fp = reflectanceXyz(build(cp), spd, cmf)
      for (let k = 0; k < 3; k++) J[k][j] = (fp[k] - (f[k] + targetXyz[k])) / h
    }
    const step = solve3(J, f.map(v => -v))
    if (!step) break
    // Damp large steps to keep the exponent well behaved.
    const maxStep = Math.max(...step.map(Math.abs))
    const s = maxStep > 1 ? 1 / maxStep : 1
    c = c.map((v, j) => v + s * step[j])
  }
  return build(c)
}

function solve3(A: number[][], b: number[]): number[] | null {
  const [a, bb, c] = A[0], [d, e, f] = A[1], [g, h, i] = A[2]
  const det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g)
  if (!Number.isFinite(det) || Math.abs(det) < 1e-18) return null
  const inv = [
    [(e * i - f * h) / det, (c * h - bb * i) / det, (bb * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (bb * g - a * h) / det, (a * e - bb * d) / det],
  ]
  return inv.map(row => row[0] * b[0] + row[1] * b[1] + row[2] * b[2])
}
