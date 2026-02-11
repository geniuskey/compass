<template>
  <div class="energy-balance-container">
    <h4>{{ t('Energy Balance: R + T + A = 1', '에너지 균형: R + T + A = 1') }}</h4>
    <p class="component-description">
      {{ t(
        'Adjust silicon thickness and BARL quality to see how reflection, transmission, and absorption partition the incident light across the visible spectrum.',
        '실리콘 두께와 BARL 품질을 조정하여 가시광선 스펙트럼에서 반사, 투과, 흡수가 입사광을 어떻게 분배하는지 확인합니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Silicon thickness:', '실리콘 두께:') }} <strong>{{ siliconThickness.toFixed(1) }} um</strong>
        </label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siliconThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('BARL quality:', 'BARL 품질:') }} <strong>{{ barlQuality }}%</strong>
        </label>
        <input type="range" min="0" max="100" step="1" v-model.number="barlQuality" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card" style="border-left: 3px solid #27ae60;">
        <span class="info-label">{{ t('Max error:', '최대 오차:') }}</span>
        <span class="info-value">{{ maxError.toExponential(2) }}</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #3498db;">
        <span class="info-label">{{ t('Mean error:', '평균 오차:') }}</span>
        <span class="info-value">{{ meanError.toExponential(2) }}</span>
      </div>
      <div class="info-card" :style="{ borderLeft: '3px solid ' + (isValid ? '#27ae60' : '#e74c3c') }">
        <span class="info-label">{{ t('Validation:', '검증:') }}</span>
        <span class="info-value" :style="{ color: isValid ? '#27ae60' : '#e74c3c' }">{{ isValid ? '\u2713' : '\u26A0' }} {{ isValid ? t('Pass', '통과') : t('Fail', '실패') }}</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #e67e22;">
        <span class="info-label">{{ t('Peak absorption:', '최대 흡수:') }}</span>
        <span class="info-value">{{ peakAbsorptionWl }} nm</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="energy-svg"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
      >
        <!-- Visible spectrum band on x-axis -->
        <defs>
          <linearGradient id="ebVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>
        <rect
          :x="pad.left"
          :y="pad.top + plotH + 2"
          :width="plotW"
          height="10"
          fill="url(#ebVisSpectrum)"
          rx="2"
        />

        <!-- Grid lines -->
        <line
          v-for="tick in yTicks"
          :key="'yg' + tick"
          :x1="pad.left"
          :y1="yScale(tick)"
          :x2="pad.left + plotW"
          :y2="yScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />
        <line
          v-for="tick in xTicks"
          :key="'xg' + tick"
          :x1="xScale(tick)"
          :y1="pad.top"
          :x2="xScale(tick)"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />

        <!-- Axes -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <!-- Y-axis labels -->
        <text
          v-for="tick in yTicks"
          :key="'yl' + tick"
          :x="pad.left - 6"
          :y="yScale(tick) + 3"
          text-anchor="end"
          class="axis-label"
        >{{ tick.toFixed(1) }}</text>

        <!-- X-axis labels -->
        <text
          v-for="tick in xTicks"
          :key="'xl' + tick"
          :x="xScale(tick)"
          :y="pad.top + plotH + 26"
          text-anchor="middle"
          class="axis-label"
        >{{ tick }}</text>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
        <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`">{{ t('Fraction', '비율') }}</text>

        <!-- Stacked areas: Reflection (bottom), Transmission (middle), Absorption (top) -->
        <!-- Reflection area (from 0 to R) -->
        <path :d="reflectionAreaPath" fill="#3498db" opacity="0.6" />
        <!-- Transmission area (from R to R+T) -->
        <path :d="transmissionAreaPath" fill="#27ae60" opacity="0.6" />
        <!-- Absorption area (from R+T to 1) -->
        <path :d="absorptionAreaPath" fill="#e67e22" opacity="0.6" />

        <!-- Boundary lines -->
        <path :d="reflectionLinePath" fill="none" stroke="#3498db" stroke-width="1.5" />
        <path :d="rtLinePath" fill="none" stroke="#27ae60" stroke-width="1.5" />

        <!-- Top line at y=1 -->
        <line
          :x1="pad.left"
          :y1="yScale(1)"
          :x2="pad.left + plotW"
          :y2="yScale(1)"
          stroke="var(--vp-c-text-2)"
          stroke-width="0.5"
          stroke-dasharray="2,2"
        />

        <!-- Crosshair on hover -->
        <template v-if="hoverWl !== null">
          <line :x1="xScale(hoverWl)" :y1="pad.top" :x2="xScale(hoverWl)" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
          <!-- Dots at R, R+T boundaries -->
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverData.R)" r="3.5" fill="#3498db" stroke="#fff" stroke-width="1" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverData.R + hoverData.T)" r="3.5" fill="#27ae60" stroke="#fff" stroke-width="1" />
          <!-- Tooltip -->
          <rect :x="tooltipX" :y="pad.top + 4" width="130" height="72" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#3498db">R: {{ hoverData.R.toFixed(4) }}</text>
          <text :x="tooltipX + 8" :y="pad.top + 44" class="tooltip-text" fill="#27ae60">T: {{ hoverData.T.toFixed(4) }}</text>
          <text :x="tooltipX + 8" :y="pad.top + 56" class="tooltip-text" fill="#e67e22">A: {{ hoverData.A.toFixed(4) }}</text>
          <text :x="tooltipX + 8" :y="pad.top + 68" class="tooltip-text" fill="var(--vp-c-text-2)">{{ t('Sum:', '합:') }} {{ (hoverData.R + hoverData.T + hoverData.A).toFixed(6) }}</text>
        </template>

        <!-- Legend -->
        <rect :x="pad.left + 8" :y="pad.top + 8" width="10" height="10" fill="#3498db" opacity="0.8" rx="1" />
        <text :x="pad.left + 22" :y="pad.top + 17" class="legend-label">{{ t('Reflection (R)', '반사 (R)') }}</text>
        <rect :x="pad.left + 8" :y="pad.top + 22" width="10" height="10" fill="#27ae60" opacity="0.8" rx="1" />
        <text :x="pad.left + 22" :y="pad.top + 31" class="legend-label">{{ t('Transmission (T)', '투과 (T)') }}</text>
        <rect :x="pad.left + 8" :y="pad.top + 36" width="10" height="10" fill="#e67e22" opacity="0.8" rx="1" />
        <text :x="pad.left + 22" :y="pad.top + 45" class="legend-label">{{ t('Absorption (A)', '흡수 (A)') }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const siliconThickness = ref(3.0)
const barlQuality = ref(70)

const svgW = 520
const svgH = 300
const pad = { top: 20, right: 20, bottom: 40, left: 50 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 380
const wlMax = 780
const wlStep = 2
const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function xScale(wl) {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}

function yScale(frac) {
  return pad.top + plotH - (frac / 1.0) * plotH
}

// Silicon extinction coefficient approximation: k(lambda_um) = 0.028 * exp(3.0 * (0.55 - lambda_um))
function siliconK(wlNm) {
  const wlUm = wlNm / 1000
  return 0.028 * Math.exp(3.0 * (0.55 - wlUm))
}

// Silicon refractive index approximation: n(lambda_um) ~ 3.5 + 1.5 * exp(-5*(lambda_um - 0.4))
function siliconN(wlNm) {
  const wlUm = wlNm / 1000
  return 3.5 + 1.5 * Math.exp(-5 * (wlUm - 0.4))
}

// Compute R, T, A for a given wavelength
function computeRTA(wlNm) {
  const wlUm = wlNm / 1000
  const n = siliconN(wlNm)
  const k = siliconK(wlNm)

  // Fresnel reflection at normal incidence: R_fresnel = ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
  // Simplified as per spec: R_fresnel ~ ((n-1)/(n+1))^2
  const rFresnel = Math.pow((n - 1) / (n + 1), 2)

  // BARL reduces reflection
  const barlFactor = barlQuality.value / 100
  const R = rFresnel * (1 - barlFactor)

  // Absorption coefficient: alpha = 4*pi*k / lambda
  const alpha = 4 * Math.PI * k / wlUm

  // Absorption: A = (1-R) * (1 - exp(-alpha * d))
  const d = siliconThickness.value
  const A = (1 - R) * (1 - Math.exp(-alpha * d))

  // Transmission: T = 1 - R - A, clamped >= 0
  const T = Math.max(0, 1 - R - A)

  return { R, T, A }
}

// Generate data points across the spectrum
const dataPoints = computed(() => {
  const pts = []
  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    const { R, T, A } = computeRTA(wl)
    pts.push({ wl, R, T, A })
  }
  return pts
})

// Stacked area paths
// Reflection: from y=0 to y=R
const reflectionAreaPath = computed(() => {
  const pts = dataPoints.value
  let path = `M${xScale(pts[0].wl).toFixed(1)},${yScale(0).toFixed(1)}`
  // Top edge: R values
  for (let i = 0; i < pts.length; i++) {
    path += ` L${xScale(pts[i].wl).toFixed(1)},${yScale(pts[i].R).toFixed(1)}`
  }
  // Bottom edge: back along y=0
  path += ` L${xScale(pts[pts.length - 1].wl).toFixed(1)},${yScale(0).toFixed(1)} Z`
  return path
})

// Transmission: from y=R to y=R+T
const transmissionAreaPath = computed(() => {
  const pts = dataPoints.value
  // Top edge: R+T values (left to right)
  let path = `M${xScale(pts[0].wl).toFixed(1)},${yScale(pts[0].R + pts[0].T).toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    path += ` L${xScale(pts[i].wl).toFixed(1)},${yScale(pts[i].R + pts[i].T).toFixed(1)}`
  }
  // Bottom edge: R values (right to left)
  for (let i = pts.length - 1; i >= 0; i--) {
    path += ` L${xScale(pts[i].wl).toFixed(1)},${yScale(pts[i].R).toFixed(1)}`
  }
  path += ' Z'
  return path
})

// Absorption: from y=R+T to y=1
const absorptionAreaPath = computed(() => {
  const pts = dataPoints.value
  // Top edge: y=1 (left to right)
  let path = `M${xScale(pts[0].wl).toFixed(1)},${yScale(1).toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    path += ` L${xScale(pts[i].wl).toFixed(1)},${yScale(1).toFixed(1)}`
  }
  // Bottom edge: R+T values (right to left)
  for (let i = pts.length - 1; i >= 0; i--) {
    path += ` L${xScale(pts[i].wl).toFixed(1)},${yScale(pts[i].R + pts[i].T).toFixed(1)}`
  }
  path += ' Z'
  return path
})

// Boundary line paths
const reflectionLinePath = computed(() => {
  const pts = dataPoints.value
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(p.wl).toFixed(1)},${yScale(p.R).toFixed(1)}`
  }).join(' ')
})

const rtLinePath = computed(() => {
  const pts = dataPoints.value
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(p.wl).toFixed(1)},${yScale(p.R + p.T).toFixed(1)}`
  }).join(' ')
})

// Validation metrics
const maxError = computed(() => {
  let max = 0
  for (const p of dataPoints.value) {
    const err = Math.abs(p.R + p.T + p.A - 1)
    if (err > max) max = err
  }
  return max
})

const meanError = computed(() => {
  let sum = 0
  const pts = dataPoints.value
  for (const p of pts) {
    sum += Math.abs(p.R + p.T + p.A - 1)
  }
  return sum / pts.length
})

const isValid = computed(() => maxError.value < 0.01)

const peakAbsorptionWl = computed(() => {
  let maxA = 0
  let peakWl = wlMin
  for (const p of dataPoints.value) {
    if (p.A > maxA) {
      maxA = p.A
      peakWl = p.wl
    }
  }
  return peakWl
})

// Hover crosshair
const hoverWl = ref(null)
const hoverData = ref({ R: 0, T: 0, A: 0 })

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const snapped = Math.round(wl)
    hoverWl.value = snapped
    hoverData.value = computeRTA(snapped)
  } else {
    hoverWl.value = null
  }
}

function onMouseLeave() {
  hoverWl.value = null
}

const tooltipX = computed(() => {
  if (hoverWl.value === null) return 0
  const x = xScale(hoverWl.value)
  return x + 140 > svgW - pad.right ? x - 140 : x + 10
})

// Visible spectrum color stops
function wavelengthToCSS(wl) {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) { r = -(wl - 440) / 60; b = 1 }
  else if (wl >= 440 && wl < 490) { g = (wl - 440) / 50; b = 1 }
  else if (wl >= 490 && wl < 510) { g = 1; b = -(wl - 510) / 20 }
  else if (wl >= 510 && wl < 580) { r = (wl - 510) / 70; g = 1 }
  else if (wl >= 580 && wl < 645) { r = 1; g = -(wl - 645) / 65 }
  else if (wl >= 645 && wl <= 780) { r = 1 }
  let f = 1.0
  if (wl >= 380 && wl < 420) f = 0.3 + 0.7 * (wl - 380) / 40
  else if (wl >= 700 && wl <= 780) f = 0.3 + 0.7 * (780 - wl) / 80
  r = Math.round(255 * Math.pow(r * f, 0.8))
  g = Math.round(255 * Math.pow(g * f, 0.8))
  b = Math.round(255 * Math.pow(b * f, 0.8))
  return `rgb(${r}, ${g}, ${b})`
}

const spectrumStops = computed(() => {
  const stops = []
  for (let wl = wlMin; wl <= wlMax; wl += 20) {
    stops.push({
      offset: ((wl - wlMin) / (wlMax - wlMin) * 100) + '%',
      color: wavelengthToCSS(wl),
    })
  }
  return stops
})
</script>

<style scoped>
.energy-balance-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.energy-balance-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 150px;
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
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
}
.info-label {
  color: var(--vp-c-text-2);
  margin-right: 4px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.svg-wrapper {
  margin-top: 8px;
}
.energy-svg {
  width: 100%;
  max-width: 520px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.legend-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
</style>
