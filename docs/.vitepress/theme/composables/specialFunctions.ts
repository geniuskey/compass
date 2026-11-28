// Special functions and diffraction (Airy) helpers shared by the optics simulators.
//
// Bessel J0/J1 use the Numerical Recipes (bessj0/bessj1) rational approximations
// for |x| < 8 and the Hankel asymptotic forms for |x| >= 8. Absolute accuracy is
// ~1e-8 over the whole real line, unlike a truncated power series which diverges
// for x > ~18.

/** Bessel function of the first kind, order 0. */
export function besselJ0(x: number): number {
  const ax = Math.abs(x)
  if (ax < 8.0) {
    const y = x * x
    const ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
      + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
    const ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
      + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
    return ans1 / ans2
  }
  const z = 8.0 / ax
  const y = z * z
  const xx = ax - 0.785398164
  const ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
    + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
  const ans2 = -0.1562499995e-1 + y * (0.1430488765e-3
    + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)))
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2)
}

/** Bessel function of the first kind, order 1. */
export function besselJ1(x: number): number {
  const ax = Math.abs(x)
  if (ax < 8.0) {
    const y = x * x
    const ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
      + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
    const ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
      + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
    return ans1 / ans2
  }
  const z = 8.0 / ax
  const y = z * z
  const xx = ax - 2.356194491
  const ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
    + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
  const ans2 = 0.04687499995 + y * (-0.2002690873e-3
    + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
  const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2)
  return x < 0 ? -ans : ans
}

/** Normalised Airy intensity I(r)/I(0) = (2 J1(x)/x)^2, x = π r / (λ N). */
export function airyIntensity(r: number, wlUm: number, fNum: number): number {
  const x = (Math.PI * r) / (wlUm * fNum)
  if (Math.abs(x) < 1e-8) return 1.0
  const v = (2 * besselJ1(x)) / x
  return v * v
}

/** Encircled energy fraction within radius r: 1 − J0(x)² − J1(x)², clamped to [0, 1]. */
export function airyEncircledEnergy(r: number, wlUm: number, fNum: number): number {
  if (r <= 0) return 0
  const x = (Math.PI * r) / (wlUm * fNum)
  const j0 = besselJ0(x)
  const j1 = besselJ1(x)
  return Math.min(1, Math.max(0, 1 - j0 * j0 - j1 * j1))
}

/**
 * Ensquared energy: fraction of the total Airy energy falling inside a square
 * of side `side` centred on the PSF. Integrated numerically (midpoint rule over
 * one quadrant) and normalised by the analytic total ∫I dA = 4(λN)²/π.
 */
export function airyEnsquaredEnergy(side: number, wlUm: number, fNum: number, n = 160): number {
  if (side <= 0) return 0
  const h = side / 2
  const d = h / n
  let sum = 0
  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) * d
    for (let j = 0; j < n; j++) {
      const y = (j + 0.5) * d
      sum += airyIntensity(Math.hypot(x, y), wlUm, fNum)
    }
  }
  const lN = wlUm * fNum
  const total = (4 * lN * lN) / Math.PI
  return Math.min(1, Math.max(0, (4 * sum * d * d) / total))
}
