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
      <div class="chart-toggle">
        <button
          :class="['toggle-btn', { active: chartType === 'classic' }]"
          @click="chartType = 'classic'"
        >Classic 24</button>
        <button
          :class="['toggle-btn', { active: chartType === 'sg' }]"
          @click="chartType = 'sg'"
        >SG 140</button>
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
      <h5>{{ t('ColorChecker Patches', 'ColorChecker 패치') }}
        <span class="patch-count">({{ activePatches.length }})</span>
      </h5>
      <div :class="['patch-grid', chartType === 'sg' ? 'patch-grid-sg' : 'patch-grid-classic']">
        <div v-for="(patch, idx) in patchResults" :key="idx" class="patch-cell">
          <div class="patch-swatch">
            <div class="patch-ref" :style="{ background: rgbStr(patch.refSrgb) }"></div>
            <div class="patch-sensor" :style="{ background: rgbStr(patch.corrSrgb) }"></div>
          </div>
          <div v-if="chartType === 'classic'" class="patch-name">{{ patch.name }}</div>
          <div
            class="patch-de"
            :class="{ 'patch-de-sg': chartType === 'sg' }"
            :style="{ color: deColor(patch.deltaE) }"
          >{{ patch.deltaE.toFixed(chartType === 'sg' ? 0 : 1) }}</div>
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
            rx="1"
          />

          <!-- X axis labels (Classic only) -->
          <template v-if="chartType === 'classic'">
            <text
              v-for="(patch, idx) in patchResults" :key="'xl'+idx"
              :x="barX(idx) + barWidth / 2"
              :y="pad.top + plotH + 10"
              text-anchor="end"
              class="tick-label"
              :transform="`rotate(-45, ${barX(idx) + barWidth / 2}, ${pad.top + plotH + 10})`"
            >{{ patch.name.substring(0, 8) }}</text>
          </template>

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
const chartType = ref<'classic' | 'sg'>('classic')

// ---- ColorChecker Classic 24 data (standard 24 patches, 7 wavelengths 400-700nm @ 50nm) ----
const CLASSIC_PATCHES: { name: string; srgb: number[]; refl: number[] }[] = [
  // Row 1: Dark Skin → Bluish Green
  { name: 'Dark Skin',     srgb: [115,82,68],   refl: [0.055,0.058,0.069,0.099,0.132,0.143,0.146] },
  { name: 'Light Skin',    srgb: [194,150,130],  refl: [0.092,0.107,0.152,0.191,0.260,0.286,0.275] },
  { name: 'Blue Sky',      srgb: [98,122,157],   refl: [0.117,0.143,0.178,0.193,0.149,0.115,0.108] },
  { name: 'Foliage',       srgb: [87,108,67],    refl: [0.042,0.051,0.085,0.131,0.099,0.068,0.060] },
  { name: 'Blue Flower',   srgb: [133,128,177],  refl: [0.137,0.132,0.128,0.120,0.133,0.154,0.228] },
  { name: 'Bluish Green',  srgb: [103,189,170],  refl: [0.131,0.226,0.318,0.341,0.301,0.232,0.218] },

  // Row 2: Orange → Orange Yellow
  { name: 'Orange',        srgb: [214,126,44],   refl: [0.050,0.054,0.063,0.170,0.395,0.413,0.266] },
  { name: 'Purplish Blue', srgb: [80,91,166],    refl: [0.153,0.128,0.091,0.058,0.042,0.060,0.157] },
  { name: 'Moderate Red',  srgb: [193,90,99],    refl: [0.065,0.053,0.058,0.078,0.200,0.305,0.231] },
  { name: 'Purple',        srgb: [94,60,108],    refl: [0.065,0.052,0.040,0.038,0.043,0.065,0.100] },
  { name: 'Yellow Green',  srgb: [157,188,64],   refl: [0.047,0.065,0.172,0.331,0.350,0.237,0.120] },
  { name: 'Orange Yellow', srgb: [224,163,46],   refl: [0.050,0.058,0.093,0.284,0.481,0.465,0.287] },

  // Row 3: Blue → Cyan
  { name: 'Blue',          srgb: [56,61,150],    refl: [0.142,0.103,0.056,0.033,0.026,0.032,0.088] },
  { name: 'Green',         srgb: [70,148,73],    refl: [0.035,0.060,0.140,0.175,0.099,0.058,0.044] },
  { name: 'Red',           srgb: [175,54,60],    refl: [0.043,0.036,0.035,0.047,0.153,0.331,0.271] },
  { name: 'Yellow',        srgb: [231,199,31],   refl: [0.042,0.054,0.124,0.397,0.559,0.504,0.316] },
  { name: 'Magenta',       srgb: [187,86,149],   refl: [0.094,0.065,0.067,0.073,0.127,0.219,0.276] },
  { name: 'Cyan',          srgb: [8,133,161],    refl: [0.073,0.139,0.210,0.219,0.137,0.083,0.076] },

  // Row 4: White → Black (neutrals)
  { name: 'White',         srgb: [243,243,242],  refl: [0.875,0.886,0.892,0.894,0.892,0.882,0.870] },
  { name: 'Neutral 8',     srgb: [200,200,200],  refl: [0.570,0.578,0.584,0.586,0.585,0.578,0.572] },
  { name: 'Neutral 6.5',   srgb: [160,160,160],  refl: [0.354,0.362,0.366,0.368,0.366,0.361,0.356] },
  { name: 'Neutral 5',     srgb: [122,122,121],  refl: [0.195,0.200,0.204,0.206,0.205,0.201,0.197] },
  { name: 'Neutral 3.5',   srgb: [85,85,85],     refl: [0.091,0.094,0.096,0.097,0.096,0.094,0.092] },
  { name: 'Black',         srgb: [52,52,52],      refl: [0.032,0.033,0.034,0.034,0.034,0.033,0.032] },
]

// ---- ColorChecker SG 140 data ----
// L*a*b* (D50/2°) values for 140 patches, 14 rows (A-N) × 10 cols (1-10)
// Compact: [L, a, b] per patch, row-major order
const SG_LAB: number[][] = [
  // Row A (1-10)
  [96.04,-0.12,0.31],[53.35,-36.85,15.22],[70.48,-32.43,0.55],[48.52,-28.76,-8.49],[39.43,-16.58,-26.34],
  [55.28,9.35,-34.09],[53.34,14.25,-13.53],[80.57,3.73,-7.71],[50.72,51.66,-14.77],[96.04,-0.12,0.31],
  // Row B (1-10)
  [81.02,-0.17,0.23],[46.47,52.54,19.94],[50.76,50.96,27.64],[66.47,35.85,60.44],[62.27,33.28,57.71],
  [72.70,-0.97,69.36],[53.61,10.59,53.90],[44.19,-14.25,38.96],[35.17,-13.17,22.74],[81.02,-0.17,0.23],
  // Row C (1-10)
  [65.73,-0.09,0.22],[35.75,60.42,34.16],[40.06,48.08,27.44],[30.87,23.44,22.47],[25.53,13.84,15.33],
  [52.91,-1.37,55.42],[39.77,17.41,47.95],[27.38,-0.63,30.60],[20.17,-0.67,13.69],[65.73,-0.09,0.22],
  // Row D (1-10)
  [50.87,-0.06,0.11],[42.35,12.63,-44.77],[52.17,2.07,-30.04],[51.20,49.36,-16.78],[60.21,25.51,2.46],
  [53.13,12.31,17.49],[71.82,-23.83,57.10],[60.94,-29.79,41.53],[49.48,-29.99,22.66],[50.87,-0.06,0.11],
  // Row E (1-10)
  [96.04,-0.12,0.31],[38.63,12.21,-45.58],[62.86,36.15,57.33],[71.93,-23.53,57.21],[55.76,-38.34,31.73],
  [40.02,10.05,-44.52],[30.36,22.89,-20.64],[72.39,-27.57,1.35],[49.06,30.18,-4.58],[96.04,-0.12,0.31],
  // Row F (1-10)
  [81.02,-0.17,0.23],[42.53,52.21,28.87],[63.85,18.23,18.15],[71.15,11.26,17.58],[78.36,0.41,0.10],
  [64.16,-18.45,-17.62],[60.06,26.25,-19.86],[61.77,0.63,0.51],[50.50,-31.76,-27.86],[81.02,-0.17,0.23],
  // Row G (1-10)
  [65.73,-0.09,0.22],[83.68,3.73,79.79],[55.09,-38.56,32.12],[31.77,1.29,-23.25],[81.68,1.02,79.96],
  [52.00,48.63,-14.84],[32.79,18.35,21.36],[72.91,0.69,0.10],[79.66,-1.02,75.21],[65.73,-0.09,0.22],
  // Row H (1-10)
  [50.87,-0.06,0.11],[62.67,37.32,68.03],[39.46,49.57,31.76],[72.67,-23.21,58.12],[51.69,-28.81,49.32],
  [51.51,54.10,25.56],[81.22,2.51,80.51],[40.02,50.40,27.55],[30.25,26.53,-22.62],[50.87,-0.06,0.11],
  // Row I (1-10)
  [96.04,-0.12,0.31],[43.25,15.29,24.73],[61.50,10.04,17.26],[68.80,12.52,15.81],[78.36,0.41,0.10],
  [57.44,-8.73,-10.74],[50.62,-28.32,-1.05],[48.97,-1.02,-0.31],[23.53,2.10,-2.48],[96.04,-0.12,0.31],
  // Row J (1-10)
  [81.02,-0.17,0.23],[53.72,7.22,-25.45],[59.75,-28.34,-26.83],[49.30,-1.96,44.83],[34.95,14.40,24.37],
  [38.93,55.97,29.59],[67.52,-30.41,-0.67],[92.71,0.06,1.45],[32.65,-23.41,0.65],[81.02,-0.17,0.23],
  // Row K (1-10)
  [65.73,-0.09,0.22],[44.44,23.12,31.85],[38.52,11.43,-45.87],[32.97,16.78,-30.95],[53.53,-41.67,34.15],
  [42.68,15.63,-44.01],[31.50,24.37,19.33],[68.48,0.37,0.37],[60.58,37.49,67.63],[65.73,-0.09,0.22],
  // Row L (1-10)
  [50.87,-0.06,0.11],[66.84,17.81,-21.56],[42.15,53.06,28.98],[43.26,48.54,-5.29],[42.83,54.65,26.66],
  [62.32,-5.55,63.37],[50.45,50.81,-14.56],[34.63,10.14,27.26],[26.47,15.39,12.83],[50.87,-0.06,0.11],
  // Row M (1-10)
  [96.04,-0.12,0.31],[49.81,-3.25,49.37],[38.73,-16.30,30.58],[28.61,-10.42,20.02],[21.17,-7.13,10.92],
  [52.72,47.41,56.15],[40.22,42.73,28.43],[30.49,28.63,16.00],[24.54,15.14,6.43],[96.04,-0.12,0.31],
  // Row N (1-10)
  [96.04,-0.12,0.31],[81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],[96.04,-0.12,0.31],
]

const SG_NAMES: string[] = (() => {
  const rows = 'ABCDEFGHIJKLMN'
  const names: string[] = []
  for (let r = 0; r < 14; r++) {
    for (let c = 1; c <= 10; c++) {
      names.push(`${rows[r]}${c}`)
    }
  }
  return names
})()

// ---- D65 illuminant (normalized, 7 wavelengths 400-700nm @ 50nm) ----
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
  const stack = defaultBsiStack(color, siThickness.value)
  const result = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
  return result.layerA[SI_LAYER_IDX]
}

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

// ---- XYZ to Lab (D65) ----
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

// ---- Lab (D50) → sRGB (D65) via Bradford chromatic adaptation ----
const D50_WP = { Xn: 0.9642, Yn: 1.0, Zn: 0.8251 }

// Bradford M matrix and its inverse
const BRAD_M = [
  [ 0.8951,  0.2664, -0.1614],
  [-0.7502,  1.7135,  0.0367],
  [ 0.0389, -0.0685,  1.0296],
]
const BRAD_MI = [
  [ 0.9870, -0.1471,  0.1600],
  [ 0.4323,  0.5184,  0.0493],
  [-0.0085,  0.0400,  0.9685],
]

// D50 white point XYZ
const D50_XYZ = [0.9642, 1.0, 0.8251]
// D65 white point XYZ
const D65_XYZ = [0.9505, 1.0, 1.089]

// Precompute Bradford adaptation matrix D50→D65
const bradAdapt: number[][] = (() => {
  // cone responses for source (D50) and destination (D65)
  const coneS = [
    BRAD_M[0][0] * D50_XYZ[0] + BRAD_M[0][1] * D50_XYZ[1] + BRAD_M[0][2] * D50_XYZ[2],
    BRAD_M[1][0] * D50_XYZ[0] + BRAD_M[1][1] * D50_XYZ[1] + BRAD_M[1][2] * D50_XYZ[2],
    BRAD_M[2][0] * D50_XYZ[0] + BRAD_M[2][1] * D50_XYZ[1] + BRAD_M[2][2] * D50_XYZ[2],
  ]
  const coneD = [
    BRAD_M[0][0] * D65_XYZ[0] + BRAD_M[0][1] * D65_XYZ[1] + BRAD_M[0][2] * D65_XYZ[2],
    BRAD_M[1][0] * D65_XYZ[0] + BRAD_M[1][1] * D65_XYZ[1] + BRAD_M[1][2] * D65_XYZ[2],
    BRAD_M[2][0] * D65_XYZ[0] + BRAD_M[2][1] * D65_XYZ[1] + BRAD_M[2][2] * D65_XYZ[2],
  ]
  // diagonal scale
  const scale = [[coneD[0]/coneS[0],0,0],[0,coneD[1]/coneS[1],0],[0,0,coneD[2]/coneS[2]]]
  // M^-1 * scale * M
  const tmp: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        tmp[i][j] += scale[i][k] * BRAD_M[k][j]
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        result[i][j] += BRAD_MI[i][k] * tmp[k][j]
  return result
})()

function labFInv(t: number): number {
  return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787
}

function labToSrgb(lab: number[]): number[] {
  // Lab → XYZ (D50)
  const fy = (lab[0] + 16) / 116
  const fx = lab[1] / 500 + fy
  const fz = fy - lab[2] / 200
  const xyzD50 = [
    D50_WP.Xn * labFInv(fx),
    D50_WP.Yn * labFInv(fy),
    D50_WP.Zn * labFInv(fz),
  ]
  // Bradford D50→D65
  const xyzD65 = [
    bradAdapt[0][0] * xyzD50[0] + bradAdapt[0][1] * xyzD50[1] + bradAdapt[0][2] * xyzD50[2],
    bradAdapt[1][0] * xyzD50[0] + bradAdapt[1][1] * xyzD50[1] + bradAdapt[1][2] * xyzD50[2],
    bradAdapt[2][0] * xyzD50[0] + bradAdapt[2][1] * xyzD50[1] + bradAdapt[2][2] * xyzD50[2],
  ]
  // XYZ(D65) → linear sRGB → gamma sRGB
  const lin = xyzToLinearRgb(xyzD65)
  return [linearToSrgb(lin[0]), linearToSrgb(lin[1]), linearToSrgb(lin[2])]
}

// ---- Gaussian basis spectral reconstruction from sRGB ----
function reconstructRefl(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0])
  const g = srgbToLinear(srgb[1])
  const b = srgbToLinear(srgb[2])
  // Basis functions: R 610nm/σ55nm, G 535nm/σ50nm, B 445nm/σ35nm
  const wlNm = [400, 450, 500, 550, 600, 650, 700]
  const refl: number[] = []
  for (const wl of wlNm) {
    const bR = Math.exp(-0.5 * ((wl - 610) / 55) ** 2)
    const bG = Math.exp(-0.5 * ((wl - 535) / 50) ** 2)
    const bB = Math.exp(-0.5 * ((wl - 445) / 35) ** 2)
    refl.push(Math.max(0.01, Math.min(1, r * bR + g * bG + b * bB)))
  }
  return refl
}

// ---- Active patches computed ----
interface PatchData {
  name: string
  srgb: number[]
  refl: number[]
}

const activePatches = computed<PatchData[]>(() => {
  if (chartType.value === 'classic') {
    return CLASSIC_PATCHES
  }
  // SG 140: convert Lab→sRGB, reconstruct reflectance
  return SG_LAB.map((lab, idx) => {
    const srgb = labToSrgb(lab)
    const refl = reconstructRefl(srgb)
    return { name: SG_NAMES[idx], srgb, refl }
  })
})

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
  const patches = activePatches.value

  // Step 1: Compute sensor R/G/B response for each wavelength
  const sensorR: number[] = WL_UM.map(wl => sensorResponse('red', wl))
  const sensorG: number[] = WL_UM.map(wl => sensorResponse('green', wl))
  const sensorB: number[] = WL_UM.map(wl => sensorResponse('blue', wl))

  // Step 2: For each patch, integrate reflectance * D65 * sensor
  const sensorRGB: number[][] = patches.map(patch => {
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
  const targetLinear: number[][] = patches.map(patch => [
    srgbToLinear(patch.srgb[0]),
    srgbToLinear(patch.srgb[1]),
    srgbToLinear(patch.srgb[2]),
  ])

  // Step 4: CCM = (S^T S)^(-1) S^T T
  const STS = matTransposeMulNx3(sensorRGB)
  const STSinv = mat3x3Inverse(STS)
  if (!STSinv) {
    return patches.map(patch => ({
      name: patch.name,
      refSrgb: patch.srgb,
      corrSrgb: [128, 128, 128],
      deltaE: 99,
    }))
  }
  const STT = matSTmulT(sensorRGB, targetLinear)
  const CCM = matMul3x3(STSinv, STT)

  // Step 5: Apply CCM, convert to sRGB, compute deltaE
  return patches.map((patch, idx) => {
    const corrLinear = mat3Vec(CCM, sensorRGB[idx])
    const corrSrgb = [
      linearToSrgb(corrLinear[0]),
      linearToSrgb(corrLinear[1]),
      linearToSrgb(corrLinear[2]),
    ]

    const refXYZ = srgbToXYZ(patch.srgb)
    const refLab = xyzToLab(refXYZ)

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

const barGap = computed(() => chartType.value === 'sg' ? 0.5 : 2)

const barWidth = computed(() => {
  const n = activePatches.value.length
  return Math.max(2, (plotW - barGap.value * (n - 1)) / n)
})

function barX(idx: number): number {
  return pad.left + idx * (barWidth.value + barGap.value)
}

// Chart hover
const chartHover = ref<{ tx: number; name: string; de: number } | null>(null)

function onChartMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = chartW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const idx = Math.floor((mouseX - pad.left) / (barWidth.value + barGap.value))
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
.patch-count {
  font-weight: 400;
  color: var(--vp-c-text-3);
  font-size: 0.85em;
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
  align-items: flex-end;
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

/* Chart type toggle */
.chart-toggle {
  display: flex;
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.toggle-btn {
  padding: 6px 14px;
  font-size: 0.82em;
  font-weight: 500;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}
.toggle-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.toggle-btn:not(.active):hover {
  background: var(--vp-c-bg-soft);
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
.patch-grid-classic {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
}
.patch-grid-sg {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
}
@media (max-width: 640px) {
  .patch-grid-classic {
    grid-template-columns: repeat(4, 1fr);
  }
  .patch-grid-sg {
    grid-template-columns: repeat(10, 1fr);
    gap: 2px;
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
.patch-grid-sg .patch-swatch {
  border-radius: 3px;
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
.patch-de-sg {
  font-size: 0.55em;
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
