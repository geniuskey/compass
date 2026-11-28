/**
 * Shared sensor-noise helpers for the browser simulators.
 *
 * All quantities are in electrons unless noted. Conventions follow EMVA 1288:
 * SNR and dynamic range are reported as 20·log10 of an amplitude ratio.
 */

/** Photon flux of 1 lux at 555 nm: 4.09e11 photons / s / cm² = 4090 photons / s / µm². */
export const PHOTONS_PER_LUX_S_UM2 = 4090

/** Abramowitz & Stegun 7.1.26 erf approximation (|error| < 1.5e-7). */
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1
  const ax = Math.abs(x)
  const tt = 1 / (1 + 0.3275911 * ax)
  const y =
    1 -
    ((((1.061405429 * tt - 1.453152027) * tt + 1.421413741) * tt - 0.284496736) * tt + 0.254829592) *
      tt *
      Math.exp(-ax * ax)
  return sign * y
}

export function normCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2))
}

export function normPdf(z: number): number {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
}

/**
 * Mean and standard deviation of min(X, sat) for X ~ N(mu, sigma²).
 *
 * This is what a real photon transfer curve measures near saturation: the
 * output clips, so the variance collapses toward zero while the mean
 * approaches `sat`.
 */
export function clippedGaussianMoments(
  mu: number,
  sigma: number,
  sat: number,
): { mean: number; sigma: number } {
  if (sigma <= 0) {
    const m = Math.min(mu, sat)
    return { mean: m, sigma: 0 }
  }
  const a = (sat - mu) / sigma
  const Phi = normCdf(a)
  const phi = normPdf(a)
  const mean = mu * Phi - sigma * phi + sat * (1 - Phi)
  const second = (mu * mu + sigma * sigma) * Phi - sigma * (mu + sat) * phi + sat * sat * (1 - Phi)
  const variance = Math.max(0, second - mean * mean)
  return { mean, sigma: Math.sqrt(variance) }
}

/**
 * Total temporal + spatial noise at signal `n` (electrons):
 * σ² = σ_read² + σ_dark² + n + (prnu·n)².
 */
export function totalNoiseE(
  n: number,
  opts: { readNoise: number; darkElectrons?: number; prnu?: number; dsnu?: number },
): number {
  const dark = opts.darkElectrons ?? 0
  const prnu = opts.prnu ?? 0
  const dsnu = opts.dsnu ?? 0
  return Math.sqrt(opts.readNoise ** 2 + dark + dsnu ** 2 + Math.max(0, n) + (prnu * n) ** 2)
}

/** SNR in dB (20·log10) at signal n with the given noise; returns -Infinity for n <= 0. */
export function snrDb(n: number, sigma: number): number {
  if (n <= 0 || sigma <= 0) return -Infinity
  return 20 * Math.log10(n / sigma)
}

/**
 * EMVA 1288 absolute sensitivity threshold in electrons:
 * μ_e,min = ½ + √(σ_d² + ¼), where σ_d is the temporal dark noise in electrons
 * (read, dark shot and quantisation noise combined).
 */
export function emvaThresholdE(sigmaDarkE: number): number {
  return 0.5 + Math.sqrt(sigmaDarkE ** 2 + 0.25)
}

/**
 * Sensor-plane illuminance from scene illuminance for a Lambertian scene:
 * E_sensor = E_scene · ρ · T_lens / (4 N² ) (on-axis, object at infinity).
 */
export function sensorIlluminance(
  sceneLux: number,
  fNumber: number,
  reflectance = 0.18,
  lensTransmission = 0.9,
): number {
  return (sceneLux * reflectance * lensTransmission) / (4 * fNumber * fNumber)
}
