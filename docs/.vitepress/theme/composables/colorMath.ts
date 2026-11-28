/**
 * Small, dependency-free linear-algebra helpers for the colour simulators
 * (ColorAccuracyAnalyzer, ColorFilterDesigner). Pure functions, unit-tested.
 */

export type Mat3 = number[][]

/** Linear interpolation of (xs, ys) at x; clamps to the end values outside the range. */
export function interp1(xs: number[], ys: number[], x: number): number {
  const n = xs.length
  if (n === 0) return 0
  if (x <= xs[0]) return ys[0]
  if (x >= xs[n - 1]) return ys[n - 1]
  let i = 1
  while (i < n - 1 && xs[i] < x) i++
  const x0 = xs[i - 1], x1 = xs[i]
  const f = (x - x0) / (x1 - x0)
  return ys[i - 1] * (1 - f) + ys[i] * f
}

export function inv3(m: Mat3): Mat3 | null {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2]
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  const scale = Math.max(...m.flat().map(Math.abs))
  if (!Number.isFinite(det) || scale === 0 || Math.abs(det) < 1e-12 * scale ** 3) return null
  const k = 1 / det
  return [
    [(e * i - f * h) * k, (c * h - b * i) * k, (b * f - c * e) * k],
    [(f * g - d * i) * k, (a * i - c * g) * k, (c * d - a * f) * k],
    [(d * h - e * g) * k, (b * g - a * h) * k, (a * e - b * d) * k],
  ]
}

export function mat3Vec(m: Mat3, v: number[]): number[] {
  return [0, 1, 2].map(r => m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/**
 * Fit a 3x3 colour-correction matrix M (rows = output channels) so that
 * M * s_k ~= t_k in the least-squares sense over all patches k, subject to the
 * exact white-preservation constraint M * w = tw.
 *
 * Each output row m_i solves  min ||S m_i - t_i||^2  s.t.  w . m_i = tw_i,
 * whose Lagrange solution is
 *   m_i = A^-1 b_i - A^-1 w * (w . A^-1 b_i - tw_i) / (w . A^-1 w),
 * with A = S^T S and b_i = S^T t_i.
 *
 * @param S  sensor RGB per patch (N x 3)
 * @param T  target linear RGB per patch (N x 3)
 * @param w  sensor RGB of the reference white (3)
 * @param tw target linear RGB of the reference white (3)
 * @returns M (3x3, rows = outputs) or null if S^T S is singular
 */
export function fitWhitePreservingCcm(S: number[][], T: number[][], w: number[], tw: number[]): Mat3 | null {
  const A: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  const B: Mat3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] // B[i] = S^T t_i  (i = output channel)
  for (let k = 0; k < S.length; k++) {
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) A[a][b] += S[k][a] * S[k][b]
      for (let i = 0; i < 3; i++) B[i][a] += S[k][a] * T[k][i]
    }
  }
  const Ai = inv3(A)
  if (!Ai) return null
  const Aiw = mat3Vec(Ai, w)
  const wAiw = w[0] * Aiw[0] + w[1] * Aiw[1] + w[2] * Aiw[2]
  if (!(Math.abs(wAiw) > 0)) return null
  return [0, 1, 2].map(i => {
    const m0 = mat3Vec(Ai, B[i])
    const lam = (w[0] * m0[0] + w[1] * m0[1] + w[2] * m0[2] - tw[i]) / wAiw
    return [m0[0] - lam * Aiw[0], m0[1] - lam * Aiw[1], m0[2] - lam * Aiw[2]]
  })
}

/** Eigenvalues of a real symmetric 3x3 matrix (closed form, ascending order). */
export function sym3Eigenvalues(m: Mat3): [number, number, number] {
  const p1 = m[0][1] ** 2 + m[0][2] ** 2 + m[1][2] ** 2
  if (p1 === 0) {
    const d = [m[0][0], m[1][1], m[2][2]].sort((a, b) => a - b)
    return [d[0], d[1], d[2]]
  }
  const q = (m[0][0] + m[1][1] + m[2][2]) / 3
  const p2 = (m[0][0] - q) ** 2 + (m[1][1] - q) ** 2 + (m[2][2] - q) ** 2 + 2 * p1
  const p = Math.sqrt(p2 / 6)
  const Bm = [0, 1, 2].map(i => [0, 1, 2].map(j => (m[i][j] - (i === j ? q : 0)) / p))
  const detB =
    Bm[0][0] * (Bm[1][1] * Bm[2][2] - Bm[1][2] * Bm[2][1]) -
    Bm[0][1] * (Bm[1][0] * Bm[2][2] - Bm[1][2] * Bm[2][0]) +
    Bm[0][2] * (Bm[1][0] * Bm[2][1] - Bm[1][1] * Bm[2][0])
  const r = Math.max(-1, Math.min(1, detB / 2))
  const phi = Math.acos(r) / 3
  const e1 = q + 2 * p * Math.cos(phi)
  const e3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3)
  const e2 = 3 * q - e1 - e3
  return [e3, e2, e1]
}

/** Spectral (2-norm) condition number sigma_max / sigma_min of a 3x3 matrix. */
export function cond2(m: Mat3): number {
  const MtM: Mat3 = [0, 1, 2].map(i => [0, 1, 2].map(j => m[0][i] * m[0][j] + m[1][i] * m[1][j] + m[2][i] * m[2][j]))
  const ev = sym3Eigenvalues(MtM)
  const lo = Math.max(0, ev[0]), hi = Math.max(0, ev[2])
  if (hi === 0) return Infinity
  if (lo === 0) return Infinity
  return Math.sqrt(hi / lo)
}
