<template>
  <div class="ca-container">
    <h4>{{ t('Color Accuracy Analyzer', '색 정확도 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Compute color reproduction accuracy (deltaE) for ColorChecker patches using TMM-based QE spectra and a least-squares Color Correction Matrix.',
        'TMM 기반 QE 스펙트럼과 최소제곱 색 보정 행렬(CCM)을 사용하여 ColorChecker 패치의 색 재현 정확도(deltaE)를 계산합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThickness.toFixed(1) }} &mu;m</strong>
        </label>
        <input type="range" min="1" max="5" step="0.1" v-model.number="siThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('CF Bandwidth (FWHM)', 'CF 대역폭 (FWHM)') }}: <strong>{{ cfBandwidth }} nm</strong>
        </label>
        <input type="range" min="50" max="150" step="5" v-model.number="cfBandwidth" class="ctrl-range" />
      </div>
    </div>

    <!-- Summary bar -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Average deltaE', '평균 deltaE') }}</div>
        <div class="result-value highlight">{{ avgDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Max deltaE', '최대 deltaE') }}</div>
        <div class="result-value">{{ maxDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Excellent (deltaE < 3)', '우수 (deltaE < 3)') }}</div>
        <div class="result-value highlight">{{ excellentPct.toFixed(0) }}%</div>
      </div>
    </div>

    <!-- Patch grid -->
    <div class="patch-section">
      <h5>{{ t('ColorChecker Patches', 'ColorChecker 패치') }}</h5>
      <div class="patch-grid">
        <div v-for="(patch, idx) in patchResults" :key="idx" class="patch-cell">
          <div class="patch-swatch">
            <div class="patch-ref" :style="{ background: rgbStr(patch.refSrgb) }"></div>
            <div class="patch-sensor" :style="{ background: rgbStr(patch.corrSrgb) }"></div>
          </div>
          <div class="patch-name">{{ patch.name }}</div>
          <div
            class="patch-de"
            :style="{ color: deColor(patch.deltaE) }"
          >{{ patch.deltaE.toFixed(1) }}</div>
        </div>
      </div>
    </div>

    <!-- DeltaE bar chart -->
    <div class="chart-section">
      <h5>{{ t('deltaE per Patch', '패치별 deltaE') }}</h5>
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${chartW} ${chartH}`"
          class="de-svg"
          @mousemove="onChartMouseMove"
          @mouseleave="chartHover = null"
        >
          <!-- Grid -->
          <line
            v-for="tick in yTicks" :key="'yg'+tick"
            :x1="pad.left" :y1="yScale(tick)"
            :x2="pad.left + plotW" :y2="yScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />

          <!-- Threshold lines -->
          <line
            :x1="pad.left" :y1="yScale(3)"
            :x2="pad.left + plotW" :y2="yScale(3)"
            stroke="#27ae60" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(3) + 3" class="ref-label" fill="#27ae60">3</text>
          <line
            :x1="pad.left" :y1="yScale(6)"
            :x2="pad.left + plotW" :y2="yScale(6)"
            stroke="#e67e22" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(6) + 3" class="ref-label" fill="#e67e22">6</text>

          <!-- Axes -->
          <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

          <!-- Y tick labels -->
          <text
            v-for="tick in yTicks" :key="'yl'+tick"
            :x="pad.left - 6" :y="yScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick }}</text>

          <!-- Bars -->
          <rect
            v-for="(patch, idx) in patchResults" :key="'bar'+idx"
            :x="barX(idx)"
            :y="yScale(patch.deltaE)"
            :width="barWidth"
            :height="Math.max(0, pad.top + plotH - yScale(patch.deltaE))"
            :fill="deColor(patch.deltaE)"
            opacity="0.8"
            rx="2"
          />

          <!-- X axis labels (rotated) -->
          <text
            v-for="(patch, idx) in patchResults" :key="'xl'+idx"
            :x="barX(idx) + barWidth / 2"
            :y="pad.top + plotH + 10"
            text-anchor="end"
            class="tick-label"
            :transform="`rotate(-45, ${barX(idx) + barWidth / 2}, ${pad.top + plotH + 10})`"
          >{{ patch.name.substring(0, 8) }}</text>

          <!-- Axis title -->
          <text :x="10" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`">&Delta;E*ab</text>

          <!-- Hover tooltip -->
          <template v-if="chartHover">
            <rect
              :x="chartHover.tx" :y="pad.top + 4"
              width="140" height="34" rx="4"
              fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95"
            />
            <text :x="chartHover.tx + 6" :y="pad.top + 18" class="tooltip-text">{{ chartHover.name }}</text>
            <text :x="chartHover.tx + 6" :y="pad.top + 30" class="tooltip-text">&Delta;E = {{ chartHover.de.toFixed(2) }}</text>
          </template>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'

const { t } = useLocale()

// ---- Controls ----
const siThickness = ref(3.0)
const cfBandwidth = ref(100)

// ---- ColorChecker patch data (8 wavelength samples 400-700nm @ 50nm) ----
const PATCHES = [
  { name: 'Dark Skin', srgb: [115,82,68], refl: [0.05,0.06,0.07,0.10,0.13,0.14,0.15] },
  { name: 'Blue Sky', srgb: [194,150,130], refl: [0.23,0.20,0.18,0.17,0.19,0.20,0.18] },
  { name: 'Foliage', srgb: [87,108,67], refl: [0.04,0.05,0.09,0.13,0.10,0.07,0.06] },
  { name: 'Blue Flower', srgb: [130,128,176], refl: [0.14,0.10,0.07,0.06,0.12,0.19,0.23] },
  { name: 'Moderate Red', srgb: [157,122,98], refl: [0.06,0.06,0.06,0.10,0.22,0.24,0.16] },
  { name: 'Purple', srgb: [122,91,165], refl: [0.12,0.07,0.04,0.04,0.06,0.16,0.24] },
  { name: 'Orange Yellow', srgb: [222,118,32], refl: [0.03,0.04,0.05,0.28,0.50,0.40,0.15] },
  { name: 'Purplish Blue', srgb: [72,91,165], refl: [0.15,0.15,0.08,0.04,0.03,0.05,0.16] },
  { name: 'Cyan', srgb: [0,135,166], refl: [0.06,0.14,0.20,0.16,0.08,0.05,0.06] },
  { name: 'Magenta', srgb: [200,82,97], refl: [0.08,0.05,0.04,0.06,0.15,0.25,0.20] },
  { name: 'Yellow', srgb: [227,198,52], refl: [0.03,0.04,0.10,0.35,0.55,0.48,0.25] },
  { name: 'Yellow Green', srgb: [162,163,55], refl: [0.03,0.05,0.14,0.30,0.25,0.15,0.08] },
  { name: 'Orange', srgb: [232,153,44], refl: [0.04,0.04,0.06,0.18,0.45,0.48,0.22] },
  { name: 'Green', srgb: [67,109,62], refl: [0.03,0.06,0.14,0.18,0.10,0.06,0.04] },
  { name: 'Red', srgb: [174,48,39], refl: [0.04,0.04,0.04,0.05,0.20,0.35,0.22] },
  { name: 'White', srgb: [243,243,242], refl: [0.85,0.86,0.87,0.88,0.88,0.87,0.86] },
  { name: 'Neutral 6.5', srgb: [161,157,154], refl: [0.35,0.36,0.36,0.36,0.36,0.35,0.35] },
  { name: 'Neutral 3.5', srgb: [52,52,52], refl: [0.03,0.03,0.03,0.03,0.03,0.03,0.03] },
]

// D65 illuminant (normalized, 7 wavelengths 400-700nm @ 50nm)
const D65 = [82.75, 109.35, 117.01, 114.86, 100.0, 90.01, 71.61]

// Wavelengths in um for TMM
const WL_UM = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

// ---- Color filter model with adjustable bandwidth ----
const LN2 = Math.log(2)

function cfTransmittance(wlUm: number, centerUm: number, fwhmNm: number): number {
  const fwhmUm = fwhmNm / 1000
  return Math.exp(-4 * LN2 * ((wlUm - centerUm) / fwhmUm) ** 2)
}

// CF center wavelengths in um
const CF_CENTERS = { red: 0.620, green: 0.530, blue: 0.450 }

// ---- TMM-based QE computation ----
function computeQE(color: 'red' | 'green' | 'blue', wlUm: number): number {
  // Use default BSI stack but adjust Si thickness
  const stack = defaultBsiStack(color, siThickness.value)
  const result = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
  // QE = silicon absorption = layerA[SI_LAYER_IDX]
  return result.layerA[SI_LAYER_IDX]
}

// Compute QE with bandwidth-adjusted CF transmittance overlay
function sensorResponse(color: 'red' | 'green' | 'blue', wlUm: number): number {
  const qe = computeQE(color, wlUm)
  const cfCenter = CF_CENTERS[color]
  const cfT = cfTransmittance(wlUm, cfCenter, cfBandwidth.value)
  return qe * cfT
}

// ---- sRGB / linear color math ----
function srgbToLinear(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
}

function linearToSrgb(c: number): number {
  const v = Math.max(0, Math.min(1, c))
  const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055
  return Math.round(Math.max(0, Math.min(255, s * 255)))
}

// sRGB to linear RGB matrix to XYZ (D65)
// Standard sRGB to XYZ matrix
const SRGB_TO_XYZ = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
]

function srgbToXYZ(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0])
  const g = srgbToLinear(srgb[1])
  const b = srgbToLinear(srgb[2])
  return [
    SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b,
    SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b,
    SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b,
  ]
}

// XYZ to sRGB (inverse of above)
const XYZ_TO_SRGB = [
  [ 3.2404542, -1.5371385, -0.4985314],
  [-0.9692660,  1.8760108,  0.0415560],
  [ 0.0556434, -0.2040259,  1.0572252],
]

function xyzToLinearRgb(xyz: number[]): number[] {
  return [
    XYZ_TO_SRGB[0][0] * xyz[0] + XYZ_TO_SRGB[0][1] * xyz[1] + XYZ_TO_SRGB[0][2] * xyz[2],
    XYZ_TO_SRGB[1][0] * xyz[0] + XYZ_TO_SRGB[1][1] * xyz[1] + XYZ_TO_SRGB[1][2] * xyz[2],
    XYZ_TO_SRGB[2][0] * xyz[0] + XYZ_TO_SRGB[2][1] * xyz[1] + XYZ_TO_SRGB[2][2] * xyz[2],
  ]
}

// ---- XYZ to Lab ----
const D65_WP = { Xn: 0.9505, Yn: 1.0, Zn: 1.089 }

function labF(t: number): number {
  return t > 0.008856 ? Math.pow(t, 1 / 3) : 7.787 * t + 16 / 116
}

function xyzToLab(xyz: number[]): number[] {
  const fx = labF(xyz[0] / D65_WP.Xn)
  const fy = labF(xyz[1] / D65_WP.Yn)
  const fz = labF(xyz[2] / D65_WP.Zn)
  return [
    116 * fy - 16,
    500 * (fx - fy),
    200 * (fy - fz),
  ]
}

function deltaE(lab1: number[], lab2: number[]): number {
  return Math.sqrt(
    (lab1[0] - lab2[0]) ** 2 +
    (lab1[1] - lab2[1]) ** 2 +
    (lab1[2] - lab2[2]) ** 2
  )
}

// ---- 3x3 matrix operations for CCM ----
function mat3x3Inverse(m: number[][]): number[][] | null {
  const [a, b, c] = [m[0][0], m[0][1], m[0][2]]
  const [d, e, f] = [m[1][0], m[1][1], m[1][2]]
  const [g, h, i] = [m[2][0], m[2][1], m[2][2]]
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  if (Math.abs(det) < 1e-12) return null
  const invDet = 1 / det
  return [
    [(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet],
    [(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet],
    [(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet],
  ]
}

function matMul3x3(a: number[][], b: number[][]): number[][] {
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) {
        result[i][j] += a[i][k] * b[k][j]
      }
    }
  }
  return result
}

function matTranspose(m: number[][]): number[][] {
  const rows = m.length
  const cols = m[0].length
  const result: number[][] = []
  for (let j = 0; j < cols; j++) {
    result[j] = []
    for (let i = 0; i < rows; i++) {
      result[j][i] = m[i][j]
    }
  }
  return result
}

// Multiply Nx3 transposed (3xN) by Nx3 → 3x3
function matTransposeMulNx3(s: number[][]): number[][] {
  const n = s.length
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] += s[k][i] * s[k][j]
      }
    }
  }
  return result
}

// Multiply S^T (3xN) by T (Nx3) → 3x3
function matSTmulT(s: number[][], tgt: number[][]): number[][] {
  const n = s.length
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] += s[k][i] * tgt[k][j]
      }
    }
  }
  return result
}

// Apply 3x3 matrix to a 3-vector
function mat3Vec(m: number[][], v: number[]): number[] {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ]
}

// ---- Main computed pipeline ----
interface PatchResult {
  name: string
  refSrgb: number[]
  corrSrgb: number[]
  deltaE: number
}

const patchResults = computed<PatchResult[]>(() => {
  // Step 1: Compute sensor R/G/B response for each wavelength
  const sensorR: number[] = WL_UM.map(wl => sensorResponse('red', wl))
  const sensorG: number[] = WL_UM.map(wl => sensorResponse('green', wl))
  const sensorB: number[] = WL_UM.map(wl => sensorResponse('blue', wl))

  // Step 2: For each patch, integrate reflectance * D65 * sensor
  const sensorRGB: number[][] = PATCHES.map(patch => {
    let rSum = 0, gSum = 0, bSum = 0
    for (let i = 0; i < 7; i++) {
      const w = patch.refl[i] * D65[i]
      rSum += w * sensorR[i]
      gSum += w * sensorG[i]
      bSum += w * sensorB[i]
    }
    return [rSum, gSum, bSum]
  })

  // Step 3: Target linear RGB from sRGB values
  const targetLinear: number[][] = PATCHES.map(patch => [
    srgbToLinear(patch.srgb[0]),
    srgbToLinear(patch.srgb[1]),
    srgbToLinear(patch.srgb[2]),
  ])

  // Step 4: CCM = (S^T S)^(-1) S^T T
  const STS = matTransposeMulNx3(sensorRGB)
  const STSinv = mat3x3Inverse(STS)
  if (!STSinv) {
    // Fallback: identity CCM if singular
    return PATCHES.map(patch => ({
      name: patch.name,
      refSrgb: patch.srgb,
      corrSrgb: [128, 128, 128],
      deltaE: 99,
    }))
  }
  const STT = matSTmulT(sensorRGB, targetLinear)
  const CCM = matMul3x3(STSinv, STT)

  // Step 5: Apply CCM, convert to sRGB, compute deltaE
  return PATCHES.map((patch, idx) => {
    const corrLinear = mat3Vec(CCM, sensorRGB[idx])
    const corrSrgb = [
      linearToSrgb(corrLinear[0]),
      linearToSrgb(corrLinear[1]),
      linearToSrgb(corrLinear[2]),
    ]

    // Reference Lab
    const refXYZ = srgbToXYZ(patch.srgb)
    const refLab = xyzToLab(refXYZ)

    // Corrected: linear → XYZ → Lab
    const corrClamp = corrLinear.map(v => Math.max(0, Math.min(1, v)))
    const corrXYZ = [
      SRGB_TO_XYZ[0][0] * corrClamp[0] + SRGB_TO_XYZ[0][1] * corrClamp[1] + SRGB_TO_XYZ[0][2] * corrClamp[2],
      SRGB_TO_XYZ[1][0] * corrClamp[0] + SRGB_TO_XYZ[1][1] * corrClamp[1] + SRGB_TO_XYZ[1][2] * corrClamp[2],
      SRGB_TO_XYZ[2][0] * corrClamp[0] + SRGB_TO_XYZ[2][1] * corrClamp[1] + SRGB_TO_XYZ[2][2] * corrClamp[2],
    ]
    const corrLab = xyzToLab(corrXYZ)

    return {
      name: patch.name,
      refSrgb: patch.srgb,
      corrSrgb,
      deltaE: deltaE(refLab, corrLab),
    }
  })
})

const avgDeltaE = computed(() => {
  const results = patchResults.value
  return results.reduce((sum, p) => sum + p.deltaE, 0) / results.length
})

const maxDeltaE = computed(() => {
  return Math.max(...patchResults.value.map(p => p.deltaE))
})

const excellentPct = computed(() => {
  const results = patchResults.value
  const count = results.filter(p => p.deltaE < 3).length
  return (count / results.length) * 100
})

// ---- Helpers ----
function rgbStr(rgb: number[]): string {
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
}

function deColor(de: number): string {
  if (de < 3) return '#27ae60'
  if (de < 6) return '#e67e22'
  return '#e74c3c'
}

// ---- DeltaE bar chart ----
const chartW = 600
const chartH = 200
const pad = { top: 16, right: 30, bottom: 60, left: 40 }
const plotW = chartW - pad.left - pad.right
const plotH = chartH - pad.top - pad.bottom

const yMax = computed(() => {
  const m = maxDeltaE.value
  return Math.max(8, Math.ceil(m + 2))
})

const yTicks = computed(() => {
  const ticks: number[] = []
  const step = yMax.value <= 12 ? 2 : 5
  for (let v = 0; v <= yMax.value; v += step) ticks.push(v)
  return ticks
})

function yScale(v: number): number {
  return pad.top + plotH - (v / yMax.value) * plotH
}

const barGap = 2
const barWidth = computed(() => {
  return Math.max(4, (plotW - barGap * (PATCHES.length - 1)) / PATCHES.length)
})

function barX(idx: number): number {
  return pad.left + idx * (barWidth.value + barGap)
}

// Chart hover
const chartHover = ref<{ tx: number; name: string; de: number } | null>(null)

function onChartMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = chartW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const idx = Math.floor((mouseX - pad.left) / (barWidth.value + barGap))
  if (idx >= 0 && idx < patchResults.value.length) {
    const patch = patchResults.value[idx]
    const bx = barX(idx) + barWidth.value / 2
    const tx = bx + 150 > chartW - pad.right ? bx - 150 : bx + 10
    chartHover.value = { tx, name: patch.name, de: patch.deltaE }
  } else {
    chartHover.value = null
  }
}
</script>

<style scoped>
.ca-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.ca-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.ca-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 200px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.0em;
  font-family: var(--vp-font-family-mono);
}
.result-value.highlight {
  color: var(--vp-c-brand-1);
}

/* Patch grid */
.patch-section {
  margin-bottom: 20px;
}
.patch-grid {
  display: grid;
  grid-template-columns: repeat(9, 1fr);
  gap: 6px;
}
@media (max-width: 640px) {
  .patch-grid {
    grid-template-columns: repeat(6, 1fr);
  }
}
.patch-cell {
  text-align: center;
}
.patch-swatch {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  display: flex;
  flex-direction: column;
}
.patch-ref {
  flex: 1;
}
.patch-sensor {
  flex: 1;
}
.patch-name {
  font-size: 0.6em;
  color: var(--vp-c-text-3);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.patch-de {
  font-size: 0.75em;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}

/* Chart */
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.de-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.ref-label {
  font-size: 8px;
  font-weight: 600;
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
</style>
