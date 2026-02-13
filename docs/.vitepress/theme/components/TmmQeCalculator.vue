<template>
  <div class="tmm-qe-container">
    <h4>{{ t('TMM Quantum Efficiency Calculator', 'TMM 양자 효율 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Configure a BSI pixel layer stack and see real-time R/T/A/QE spectra computed via the transfer matrix method.',
        'BSI 픽셀 레이어 스택을 구성하고 전달 행렬법으로 계산된 실시간 R/T/A/QE 스펙트럼을 확인합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="select-group">
        <label>{{ t('Preset:', '프리셋:') }}</label>
        <select v-model="preset" class="tmm-qe-select">
          <option value="bsi1um">{{ t('BSI 1\u03BCm (default)', 'BSI 1\u03BCm (기본)') }}</option>
          <option value="bsi08um">{{ t('BSI 0.8\u03BCm', 'BSI 0.8\u03BCm') }}</option>
          <option value="custom">{{ t('Custom', '사용자 정의') }}</option>
        </select>
      </div>
      <div class="slider-group">
        <label>
          {{ t('Silicon thickness:', '실리콘 두께:') }} <strong>{{ siThickness.toFixed(1) }} &micro;m</strong>
        </label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siThickness" class="tmm-qe-range" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Angle of incidence:', '입사각:') }} <strong>{{ angleDeg }}&deg;</strong>
        </label>
        <input type="range" min="0" max="60" step="1" v-model.number="angleDeg" class="tmm-qe-range" />
      </div>
      <div class="radio-group">
        <label class="radio-group-label">{{ t('Polarization:', '편광:') }}</label>
        <label class="radio-item">
          <input type="radio" value="s" v-model="polarization" /> s
        </label>
        <label class="radio-item">
          <input type="radio" value="p" v-model="polarization" /> p
        </label>
        <label class="radio-item">
          <input type="radio" value="avg" v-model="polarization" />
          {{ t('unpolarized', '비편광') }}
        </label>
      </div>
    </div>

    <!-- Info cards -->
    <div class="info-row">
      <div class="info-card" style="border-left: 3px solid #3498db;">
        <span class="info-label">{{ t('Blue peak QE:', 'Blue 최대 QE:') }}</span>
        <span class="info-value">{{ peakQeBlue.toFixed(1) }}%</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #27ae60;">
        <span class="info-label">{{ t('Green peak QE:', 'Green 최대 QE:') }}</span>
        <span class="info-value">{{ peakQeGreen.toFixed(1) }}%</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #e74c3c;">
        <span class="info-label">{{ t('Red peak QE:', 'Red 최대 QE:') }}</span>
        <span class="info-value">{{ peakQeRed.toFixed(1) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Average QE:', '평균 QE:') }}</span>
        <span class="info-value">{{ avgQe.toFixed(1) }}%</span>
      </div>
    </div>

    <!-- Tab buttons -->
    <div class="tab-row">
      <button
        :class="['tab-btn', { active: viewMode === 'qe' }]"
        @click="viewMode = 'qe'"
      >{{ t('QE Spectrum', 'QE 스펙트럼') }}</button>
      <button
        :class="['tab-btn', { active: viewMode === 'rta' }]"
        @click="viewMode = 'rta'"
      >{{ t('R/T/A Spectrum', 'R/T/A 스펙트럼') }}</button>
      <div v-if="viewMode === 'rta'" class="rta-cf-selector">
        <label class="radio-item" v-for="c in cfOptions" :key="c.value">
          <input type="radio" :value="c.value" v-model="rtaCfColor" />
          <span :style="{ color: c.color }">{{ c.label }}</span>
        </label>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="tmm-qe-svg"
        @mousemove="onMouseMove"
        @mouseleave="hoverIdx = -1"
      >
        <!-- Visible spectrum gradient -->
        <defs>
          <linearGradient id="tmmQeVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>
        <rect
          :x="pad.left"
          :y="pad.top + plotH + 2"
          :width="plotW"
          height="10"
          fill="url(#tmmQeVisSpectrum)"
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
        >{{ viewMode === 'qe' ? tick + '%' : tick.toFixed(1) }}</text>

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
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">
          {{ t('Wavelength (nm)', '파장 (nm)') }}
        </text>
        <text
          :x="14"
          :y="pad.top + plotH / 2"
          text-anchor="middle"
          class="axis-title"
          :transform="`rotate(-90, 14, ${pad.top + plotH / 2})`"
        >{{ viewMode === 'qe' ? 'QE (%)' : t('Value (0-1)', '값 (0-1)') }}</text>

        <!-- QE view: 3 color curves -->
        <template v-if="viewMode === 'qe'">
          <path :d="qeBluePath" fill="none" stroke="#3498db" stroke-width="2" opacity="0.9" />
          <path :d="qeGreenPath" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.9" />
          <path :d="qeRedPath" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.9" />
          <path :d="qeBlueArea" fill="#3498db" opacity="0.07" />
          <path :d="qeGreenArea" fill="#27ae60" opacity="0.07" />
          <path :d="qeRedArea" fill="#e74c3c" opacity="0.07" />
        </template>

        <!-- R/T/A view: stacked for selected CF color -->
        <template v-if="viewMode === 'rta'">
          <path :d="rtaAArea" fill="#e74c3c" opacity="0.15" />
          <path :d="rtaRPath" fill="none" stroke="#3498db" stroke-width="2" />
          <path :d="rtaTPath" fill="none" stroke="#27ae60" stroke-width="2" />
          <path :d="rtaAPath" fill="none" stroke="#e74c3c" stroke-width="2" />
        </template>

        <!-- Hover crosshair -->
        <template v-if="hoverIdx >= 0">
          <line
            :x1="xScale(hoverWl)"
            :y1="pad.top"
            :x2="xScale(hoverWl)"
            :y2="pad.top + plotH"
            stroke="var(--vp-c-text-2)"
            stroke-width="0.8"
            stroke-dasharray="4,3"
          />

          <!-- QE mode dots -->
          <template v-if="viewMode === 'qe'">
            <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQeB)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQeG)" r="4" fill="#27ae60" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQeR)" r="4" fill="#e74c3c" stroke="#fff" stroke-width="1" />
          </template>
          <!-- R/T/A mode dots -->
          <template v-if="viewMode === 'rta'">
            <circle :cx="xScale(hoverWl)" :cy="yScaleRta(hoverR)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverWl)" :cy="yScaleRta(hoverT)" r="4" fill="#27ae60" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverWl)" :cy="yScaleRta(hoverA)" r="4" fill="#e74c3c" stroke="#fff" stroke-width="1" />
          </template>

          <!-- Tooltip -->
          <rect
            :x="tooltipX"
            :y="pad.top + 4"
            width="130"
            :height="viewMode === 'qe' ? 58 : 58"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            opacity="0.95"
          />
          <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ hoverWl }} nm</text>
          <template v-if="viewMode === 'qe'">
            <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#3498db">B: {{ hoverQeB.toFixed(1) }}%</text>
            <text :x="tooltipX + 8" :y="pad.top + 44" class="tooltip-text" fill="#27ae60">G: {{ hoverQeG.toFixed(1) }}%</text>
            <text :x="tooltipX + 8" :y="pad.top + 56" class="tooltip-text" fill="#e74c3c">R: {{ hoverQeR.toFixed(1) }}%</text>
          </template>
          <template v-if="viewMode === 'rta'">
            <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#3498db">R: {{ (hoverR * 100).toFixed(1) }}%</text>
            <text :x="tooltipX + 8" :y="pad.top + 44" class="tooltip-text" fill="#27ae60">T: {{ (hoverT * 100).toFixed(1) }}%</text>
            <text :x="tooltipX + 8" :y="pad.top + 56" class="tooltip-text" fill="#e74c3c">A: {{ (hoverA * 100).toFixed(1) }}%</text>
          </template>
        </template>

        <!-- Legend -->
        <template v-if="viewMode === 'qe'">
          <line :x1="pad.left + plotW - 90" :y1="pad.top + 12" :x2="pad.left + plotW - 72" :y2="pad.top + 12" stroke="#3498db" stroke-width="2" />
          <text :x="pad.left + plotW - 68" :y="pad.top + 16" class="legend-label">{{ t('Blue', '파랑') }}</text>
          <line :x1="pad.left + plotW - 90" :y1="pad.top + 26" :x2="pad.left + plotW - 72" :y2="pad.top + 26" stroke="#27ae60" stroke-width="2" />
          <text :x="pad.left + plotW - 68" :y="pad.top + 30" class="legend-label">{{ t('Green', '초록') }}</text>
          <line :x1="pad.left + plotW - 90" :y1="pad.top + 40" :x2="pad.left + plotW - 72" :y2="pad.top + 40" stroke="#e74c3c" stroke-width="2" />
          <text :x="pad.left + plotW - 68" :y="pad.top + 44" class="legend-label">{{ t('Red', '빨강') }}</text>
        </template>
        <template v-if="viewMode === 'rta'">
          <line :x1="pad.left + plotW - 80" :y1="pad.top + 12" :x2="pad.left + plotW - 62" :y2="pad.top + 12" stroke="#3498db" stroke-width="2" />
          <text :x="pad.left + plotW - 58" :y="pad.top + 16" class="legend-label">R</text>
          <line :x1="pad.left + plotW - 80" :y1="pad.top + 26" :x2="pad.left + plotW - 62" :y2="pad.top + 26" stroke="#27ae60" stroke-width="2" />
          <text :x="pad.left + plotW - 58" :y="pad.top + 30" class="legend-label">T</text>
          <line :x1="pad.left + plotW - 80" :y1="pad.top + 40" :x2="pad.left + plotW - 62" :y2="pad.top + 40" stroke="#e74c3c" stroke-width="2" />
          <text :x="pad.left + plotW - 58" :y="pad.top + 44" class="legend-label">A</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useLocale } from '../composables/useLocale'
import {
  tmmSpectrum, wlRange, defaultBsiStack,
  SI_LAYER_IDX,
  type TmmLayer, type TmmResult
} from '../composables/tmm'

const { t } = useLocale()

// --- Controls ---
const preset = ref<'bsi1um' | 'bsi08um' | 'custom'>('bsi1um')
const siThickness = ref(3.0)
const angleDeg = ref(0)
const polarization = ref<'s' | 'p' | 'avg'>('avg')
const viewMode = ref<'qe' | 'rta'>('qe')
const rtaCfColor = ref<'red' | 'green' | 'blue'>('green')

const cfOptions = computed(() => [
  { value: 'red' as const, label: t('Red', '빨강'), color: '#e74c3c' },
  { value: 'green' as const, label: t('Green', '초록'), color: '#27ae60' },
  { value: 'blue' as const, label: t('Blue', '파랑'), color: '#3498db' },
])

// Sync preset -> silicon thickness
watch(preset, (v) => {
  if (v === 'bsi1um') siThickness.value = 3.0
  else if (v === 'bsi08um') siThickness.value = 2.0
})
watch(siThickness, () => {
  if (preset.value !== 'custom') preset.value = 'custom'
})

// --- SVG layout ---
const svgW = 600
const svgH = 360
const pad = { left: 60, right: 20, top: 20, bottom: 40 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 380
const wlMax = 780
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function xScale(wl: number): number {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}

// --- QE mode y-axis: 0-100% ---
const yTicksQe = [0, 20, 40, 60, 80, 100]
// R/T/A mode y-axis: 0-1.0
const yTicksRta = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

const yTicks = computed(() => viewMode.value === 'qe' ? yTicksQe : yTicksRta)

function yScale(v: number): number {
  // QE mode: 0-100%
  return pad.top + plotH - (v / 100) * plotH
}
function yScaleRta(v: number): number {
  // R/T/A mode: 0-1.0
  return pad.top + plotH - v * plotH
}

// --- TMM computation ---
const wavelengths = computed(() => wlRange(0.38, 0.78, 0.005))
const wavelengthsNm = computed(() => wavelengths.value.map(w => Math.round(w * 1000)))

function buildStack(cf: 'red' | 'green' | 'blue'): TmmLayer[] {
  return defaultBsiStack(cf, siThickness.value)
}

const spectraBlue = computed(() =>
  tmmSpectrum(buildStack('blue'), 'air', 'sio2', wavelengths.value, angleDeg.value, polarization.value)
)
const spectraGreen = computed(() =>
  tmmSpectrum(buildStack('green'), 'air', 'sio2', wavelengths.value, angleDeg.value, polarization.value)
)
const spectraRed = computed(() =>
  tmmSpectrum(buildStack('red'), 'air', 'sio2', wavelengths.value, angleDeg.value, polarization.value)
)

// QE = silicon absorption = layerA[SI_LAYER_IDX]
function extractQe(spectra: TmmResult[]): number[] {
  return spectra.map(r => r.layerA[SI_LAYER_IDX] * 100)
}

const qeBlue = computed(() => extractQe(spectraBlue.value))
const qeGreen = computed(() => extractQe(spectraGreen.value))
const qeRed = computed(() => extractQe(spectraRed.value))

// R/T/A for selected CF
const rtaSpectra = computed(() => {
  if (rtaCfColor.value === 'red') return spectraRed.value
  if (rtaCfColor.value === 'blue') return spectraBlue.value
  return spectraGreen.value
})

// --- Info cards ---
const peakQeBlue = computed(() => Math.max(...qeBlue.value))
const peakQeGreen = computed(() => Math.max(...qeGreen.value))
const peakQeRed = computed(() => Math.max(...qeRed.value))
const avgQe = computed(() => (peakQeBlue.value + peakQeGreen.value + peakQeRed.value) / 3)

// --- SVG paths ---
function buildLinePath(values: number[], scaleFn: (v: number) => number): string {
  return values.map((v, i) => {
    const x = xScale(wavelengthsNm.value[i]).toFixed(1)
    const y = scaleFn(v).toFixed(1)
    return (i === 0 ? 'M' : 'L') + x + ',' + y
  }).join(' ')
}

function buildAreaPath(values: number[], scaleFn: (v: number) => number): string {
  const line = buildLinePath(values, scaleFn)
  const lastWl = wavelengthsNm.value[wavelengthsNm.value.length - 1]
  const firstWl = wavelengthsNm.value[0]
  const baseY = scaleFn(0).toFixed(1)
  return line + ` L${xScale(lastWl).toFixed(1)},${baseY} L${xScale(firstWl).toFixed(1)},${baseY} Z`
}

const qeBluePath = computed(() => buildLinePath(qeBlue.value, yScale))
const qeGreenPath = computed(() => buildLinePath(qeGreen.value, yScale))
const qeRedPath = computed(() => buildLinePath(qeRed.value, yScale))
const qeBlueArea = computed(() => buildAreaPath(qeBlue.value, yScale))
const qeGreenArea = computed(() => buildAreaPath(qeGreen.value, yScale))
const qeRedArea = computed(() => buildAreaPath(qeRed.value, yScale))

// R/T/A paths
const rtaRPath = computed(() => buildLinePath(rtaSpectra.value.map(r => r.R), yScaleRta))
const rtaTPath = computed(() => buildLinePath(rtaSpectra.value.map(r => r.T), yScaleRta))
const rtaAPath = computed(() => buildLinePath(rtaSpectra.value.map(r => r.A), yScaleRta))
const rtaAArea = computed(() => buildAreaPath(rtaSpectra.value.map(r => r.A), yScaleRta))

// --- Hover ---
const hoverIdx = ref(-1)

const hoverWl = computed(() => hoverIdx.value >= 0 ? wavelengthsNm.value[hoverIdx.value] : 0)
const hoverQeB = computed(() => hoverIdx.value >= 0 ? qeBlue.value[hoverIdx.value] : 0)
const hoverQeG = computed(() => hoverIdx.value >= 0 ? qeGreen.value[hoverIdx.value] : 0)
const hoverQeR = computed(() => hoverIdx.value >= 0 ? qeRed.value[hoverIdx.value] : 0)
const hoverR = computed(() => hoverIdx.value >= 0 ? rtaSpectra.value[hoverIdx.value].R : 0)
const hoverT = computed(() => hoverIdx.value >= 0 ? rtaSpectra.value[hoverIdx.value].T : 0)
const hoverA = computed(() => hoverIdx.value >= 0 ? rtaSpectra.value[hoverIdx.value].A : 0)

const tooltipX = computed(() => {
  if (hoverIdx.value < 0) return 0
  const x = xScale(hoverWl.value)
  return x + 140 > svgW - pad.right ? x - 140 : x + 10
})

function onMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    // Find closest index
    const targetUm = wl / 1000
    let bestIdx = 0
    let bestDist = Infinity
    for (let i = 0; i < wavelengths.value.length; i++) {
      const d = Math.abs(wavelengths.value[i] - targetUm)
      if (d < bestDist) { bestDist = d; bestIdx = i }
    }
    hoverIdx.value = bestIdx
  } else {
    hoverIdx.value = -1
  }
}

// --- Visible spectrum gradient ---
function wavelengthToCSS(wl: number): string {
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
  const stops: { offset: string; color: string }[] = []
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
.tmm-qe-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.tmm-qe-container h4 {
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
  align-items: flex-end;
  margin-bottom: 12px;
}
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.9em;
  font-weight: 600;
}
.tmm-qe-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.slider-group {
  flex: 1;
  min-width: 160px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.tmm-qe-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.tmm-qe-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.tmm-qe-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.radio-group {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.radio-group-label {
  font-size: 0.85em;
  font-weight: 600;
}
.radio-item {
  font-size: 0.85em;
  display: flex;
  align-items: center;
  gap: 3px;
  cursor: pointer;
}
.radio-item input[type="radio"] {
  margin: 0;
  cursor: pointer;
}
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
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
.tab-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.tab-btn {
  padding: 6px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  cursor: pointer;
  transition: all 0.15s;
}
.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-text-1);
}
.tab-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
.rta-cf-selector {
  display: flex;
  gap: 10px;
  margin-left: 12px;
}
.svg-wrapper {
  margin-top: 8px;
}
.tmm-qe-svg {
  width: 100%;
  max-width: 600px;
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
