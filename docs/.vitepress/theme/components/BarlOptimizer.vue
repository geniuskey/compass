<template>
  <div class="barl-optimizer-container">
    <h4>{{ t('BARL Anti-Reflection Coating Optimizer', 'BARL 반사 방지막 최적화기') }}</h4>
    <p class="component-description">
      {{ t(
        'Adjust the 4 BARL sublayer thicknesses to minimize reflectance in a target wavelength band. Use Auto Optimize to find the optimal configuration.',
        'BARL 4개 서브레이어 두께를 조절하여 목표 파장 대역의 반사율을 최소화합니다. 자동 최적화 버튼으로 최적 구성을 찾을 수 있습니다.'
      ) }}
    </p>

    <!-- Target band selector + optimize button -->
    <div class="controls-row">
      <div class="select-group">
        <label>{{ t('Target band:', '목표 대역:') }}</label>
        <select v-model="targetBand" class="barl-optimizer-select">
          <option value="blue">{{ t('Blue (430-470nm)', 'Blue (430-470nm)') }}</option>
          <option value="green">{{ t('Green (510-560nm)', 'Green (510-560nm)') }}</option>
          <option value="red">{{ t('Red (590-640nm)', 'Red (590-640nm)') }}</option>
          <option value="visible">{{ t('Visible (430-640nm)', 'Visible (430-640nm)') }}</option>
        </select>
      </div>
      <button
        class="optimize-btn"
        :disabled="optimizing"
        @click="runOptimize"
      >
        <template v-if="optimizing">{{ t('Optimizing...', '최적화 중...') }}</template>
        <template v-else>{{ t('Auto Optimize', '자동 최적화') }}</template>
      </button>
    </div>

    <!-- 4 thickness sliders -->
    <div class="layers-row">
      <div class="layer-card" v-for="(layer, idx) in layerDefs" :key="idx">
        <div class="layer-header">
          <span class="layer-swatch" :style="{ background: layer.color }"></span>
          <span class="layer-name" v-html="layer.label"></span>
        </div>
        <div class="layer-slider">
          <label>{{ thicknesses[idx] }} nm</label>
          <input
            type="range"
            :min="0"
            :max="layer.max"
            step="1"
            :value="thicknesses[idx]"
            @input="updateThickness(idx, $event)"
            class="barl-optimizer-range"
          />
        </div>
      </div>
    </div>

    <!-- Info cards -->
    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Avg R in band:', '대역 평균 R:') }}</span>
        <span class="info-value">{{ avgRInBand.toFixed(2) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Min R:', '최소 R:') }}</span>
        <span class="info-value">{{ minR.value.toFixed(2) }}% @ {{ minR.wl }} nm</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Total BARL:', 'BARL 총 두께:') }}</span>
        <span class="info-value">{{ totalThickness }} nm</span>
      </div>
    </div>

    <!-- Layer stack bar visualization -->
    <div class="stack-viz-section">
      <span class="stack-viz-label">{{ t('BARL Stack:', 'BARL 스택:') }}</span>
      <div class="stack-bar">
        <div
          v-for="(layer, idx) in layerDefs"
          :key="'bar' + idx"
          class="stack-segment"
          :style="{
            width: totalThickness > 0 ? (thicknesses[idx] / totalThickness * 100) + '%' : '25%',
            background: layer.color,
          }"
        >
          <span v-if="thicknesses[idx] > 8" class="stack-segment-label">{{ thicknesses[idx] }}</span>
        </div>
      </div>
      <div class="stack-legend">
        <span v-for="(layer, idx) in layerDefs" :key="'leg' + idx" class="stack-legend-item">
          <span class="stack-legend-swatch" :style="{ background: layer.color }"></span>
          <span v-html="layer.label"></span>
        </span>
      </div>
    </div>

    <!-- SVG Chart -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="barl-optimizer-svg"
        @mousemove="onMouseMove"
        @mouseleave="hoverIdx = -1"
      >
        <!-- Visible spectrum gradient -->
        <defs>
          <linearGradient id="barlVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>
        <rect
          :x="pad.left"
          :y="pad.top + plotH + 2"
          :width="plotW"
          height="8"
          fill="url(#barlVisSpectrum)"
          rx="2"
        />

        <!-- Target band shading -->
        <rect
          :x="xScale(bandRange[0])"
          :y="pad.top"
          :width="xScale(bandRange[1]) - xScale(bandRange[0])"
          :height="plotH"
          fill="var(--vp-c-brand-1)"
          opacity="0.08"
        />
        <line
          :x1="xScale(bandRange[0])"
          :y1="pad.top"
          :x2="xScale(bandRange[0])"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-brand-1)"
          stroke-width="1"
          stroke-dasharray="4,3"
          opacity="0.5"
        />
        <line
          :x1="xScale(bandRange[1])"
          :y1="pad.top"
          :x2="xScale(bandRange[1])"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-brand-1)"
          stroke-width="1"
          stroke-dasharray="4,3"
          opacity="0.5"
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
        >{{ tick }}%</text>

        <!-- X-axis labels -->
        <text
          v-for="tick in xTicks"
          :key="'xl' + tick"
          :x="xScale(tick)"
          :y="pad.top + plotH + 24"
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
        >{{ t('Reflectance (%)', '반사율 (%)') }}</text>

        <!-- Reflectance curve -->
        <path :d="reflPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />

        <!-- Minimum R marker -->
        <circle
          :cx="xScale(minR.wl)"
          :cy="yScale(minR.value)"
          r="5"
          fill="var(--vp-c-brand-1)"
          stroke="#fff"
          stroke-width="1.5"
        />

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
          <circle
            :cx="xScale(hoverWl)"
            :cy="yScale(hoverRefl)"
            r="4"
            fill="var(--vp-c-brand-1)"
            stroke="#fff"
            stroke-width="1"
          />
          <!-- Tooltip -->
          <rect
            :x="tooltipX"
            :y="pad.top + 4"
            width="120"
            height="34"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            opacity="0.95"
          />
          <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text">R: {{ hoverRefl.toFixed(2) }}%</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
import { useLocale } from '../composables/useLocale'
import {
  tmmSpectrum, wlRange,
  type TmmLayer, type TmmResult
} from '../composables/tmm'

const { t } = useLocale()

// --- Layer definitions ---
const layerDefs = [
  { label: 'SiO&#x2082; (1)', material: 'sio2', color: '#7fb3d8', max: 50 },
  { label: 'HfO&#x2082;', material: 'hfo2', color: '#6c71c4', max: 50 },
  { label: 'SiO&#x2082; (3)', material: 'sio2', color: '#e8d44d', max: 50 },
  { label: 'Si&#x2083;N&#x2084;', material: 'si3n4', color: '#2aa198', max: 80 },
]

const thicknesses = reactive([10, 25, 15, 30])
const targetBand = ref<'blue' | 'green' | 'red' | 'visible'>('visible')
const optimizing = ref(false)

function updateThickness(idx: number, event: Event) {
  thicknesses[idx] = parseInt((event.target as HTMLInputElement).value)
}

const totalThickness = computed(() => thicknesses.reduce((s, v) => s + v, 0))

// --- Band ranges (nm) ---
const bandRanges: Record<string, [number, number]> = {
  blue: [430, 470],
  green: [510, 560],
  red: [590, 640],
  visible: [430, 640],
}
const bandRange = computed(() => bandRanges[targetBand.value])

// --- SVG layout ---
const svgW = 600
const svgH = 300
const pad = { left: 60, right: 20, top: 20, bottom: 40 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 380
const wlMax = 780
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]
const yTicks = [0, 5, 10, 15, 20, 25, 30]
const yMax = 30

function xScale(wl: number): number {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}
function yScale(r: number): number {
  return pad.top + plotH - (r / yMax) * plotH
}

// --- TMM computation ---
const wavelengths = computed(() => wlRange(0.38, 0.78, 0.005))
const wavelengthsNm = computed(() => wavelengths.value.map(w => Math.round(w * 1000)))

function buildBarlStack(t0: number, t1: number, t2: number, t3: number): TmmLayer[] {
  // BARL layers on silicon: SiO2 → HfO2 → SiO2 → Si3N4 (light hits Si3N4 first, then down to Si)
  // Order in TMM: light enters from air, goes through layers top to bottom, exits into silicon
  return [
    { material: 'si3n4', thickness: t3 / 1000 },
    { material: 'sio2', thickness: t2 / 1000 },
    { material: 'hfo2', thickness: t1 / 1000 },
    { material: 'sio2', thickness: t0 / 1000 },
  ]
}

const spectra = computed((): TmmResult[] => {
  const stack = buildBarlStack(thicknesses[0], thicknesses[1], thicknesses[2], thicknesses[3])
  return tmmSpectrum(stack, 'air', 'silicon', wavelengths.value, 0, 'avg')
})

const reflectances = computed(() => spectra.value.map(r => r.R * 100))

// --- Info cards ---
const avgRInBand = computed(() => {
  const [lo, hi] = bandRange.value
  let sum = 0, cnt = 0
  for (let i = 0; i < wavelengthsNm.value.length; i++) {
    if (wavelengthsNm.value[i] >= lo && wavelengthsNm.value[i] <= hi) {
      sum += reflectances.value[i]
      cnt++
    }
  }
  return cnt > 0 ? sum / cnt : 0
})

const minR = computed(() => {
  let minVal = Infinity, minWl = 550
  for (let i = 0; i < reflectances.value.length; i++) {
    if (reflectances.value[i] < minVal) {
      minVal = reflectances.value[i]
      minWl = wavelengthsNm.value[i]
    }
  }
  return { value: minVal, wl: minWl }
})

// --- SVG path ---
const reflPath = computed(() => {
  return reflectances.value.map((r, i) => {
    const x = xScale(wavelengthsNm.value[i]).toFixed(1)
    const y = yScale(Math.min(r, yMax)).toFixed(1)
    return (i === 0 ? 'M' : 'L') + x + ',' + y
  }).join(' ')
})

// --- Hover ---
const hoverIdx = ref(-1)
const hoverWl = computed(() => hoverIdx.value >= 0 ? wavelengthsNm.value[hoverIdx.value] : 0)
const hoverRefl = computed(() => hoverIdx.value >= 0 ? reflectances.value[hoverIdx.value] : 0)

const tooltipX = computed(() => {
  if (hoverIdx.value < 0) return 0
  const x = xScale(hoverWl.value)
  return x + 130 > svgW - pad.right ? x - 130 : x + 10
})

function onMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const targetUm = wl / 1000
    let bestIdx = 0, bestDist = Infinity
    for (let i = 0; i < wavelengths.value.length; i++) {
      const d = Math.abs(wavelengths.value[i] - targetUm)
      if (d < bestDist) { bestDist = d; bestIdx = i }
    }
    hoverIdx.value = bestIdx
  } else {
    hoverIdx.value = -1
  }
}

// --- Auto Optimize ---
function computeAvgR(t0: number, t1: number, t2: number, t3: number): number {
  const stack = buildBarlStack(t0, t1, t2, t3)
  const results = tmmSpectrum(stack, 'air', 'silicon', wavelengths.value, 0, 'avg')
  const [lo, hi] = bandRange.value
  let sum = 0, cnt = 0
  for (let i = 0; i < wavelengthsNm.value.length; i++) {
    if (wavelengthsNm.value[i] >= lo && wavelengthsNm.value[i] <= hi) {
      sum += results[i].R
      cnt++
    }
  }
  return cnt > 0 ? sum / cnt : 1
}

async function runOptimize() {
  optimizing.value = true

  // Allow UI to update
  await new Promise(r => requestAnimationFrame(r))

  // Phase 1: coarse grid search (step 5nm)
  let bestR = Infinity
  let best = [thicknesses[0], thicknesses[1], thicknesses[2], thicknesses[3]]

  const step1 = 5
  const ranges = layerDefs.map(l => l.max)

  // Use a chunked approach to avoid blocking
  const coarseValues: number[][] = [[], [], [], []]
  for (let i = 0; i < 4; i++) {
    for (let v = 0; v <= ranges[i]; v += step1) {
      coarseValues[i].push(v)
    }
  }

  // Iterate over coarse grid
  for (const v0 of coarseValues[0]) {
    for (const v1 of coarseValues[1]) {
      for (const v2 of coarseValues[2]) {
        for (const v3 of coarseValues[3]) {
          const r = computeAvgR(v0, v1, v2, v3)
          if (r < bestR) {
            bestR = r
            best = [v0, v1, v2, v3]
          }
        }
      }
    }
  }

  // Yield to UI between phases
  await new Promise(r => requestAnimationFrame(r))

  // Phase 2: fine search around best (step 1nm, +/- 5nm)
  const fineRange = 5
  for (let d0 = Math.max(0, best[0] - fineRange); d0 <= Math.min(ranges[0], best[0] + fineRange); d0++) {
    for (let d1 = Math.max(0, best[1] - fineRange); d1 <= Math.min(ranges[1], best[1] + fineRange); d1++) {
      for (let d2 = Math.max(0, best[2] - fineRange); d2 <= Math.min(ranges[2], best[2] + fineRange); d2++) {
        for (let d3 = Math.max(0, best[3] - fineRange); d3 <= Math.min(ranges[3], best[3] + fineRange); d3++) {
          const r = computeAvgR(d0, d1, d2, d3)
          if (r < bestR) {
            bestR = r
            best = [d0, d1, d2, d3]
          }
        }
      }
    }
  }

  // Apply results
  for (let i = 0; i < 4; i++) {
    thicknesses[i] = best[i]
  }

  optimizing.value = false
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
.barl-optimizer-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.barl-optimizer-container h4 {
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
  align-items: center;
  margin-bottom: 14px;
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
.barl-optimizer-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.optimize-btn {
  padding: 8px 20px;
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.optimize-btn:hover:not(:disabled) {
  opacity: 0.85;
}
.optimize-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.layers-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.layer-card {
  flex: 1;
  min-width: 130px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
}
.layer-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.layer-swatch {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  flex-shrink: 0;
}
.layer-name {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.layer-slider label {
  display: block;
  font-size: 0.8em;
  margin-bottom: 4px;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-2);
}
.barl-optimizer-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.barl-optimizer-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.barl-optimizer-range::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
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
.stack-viz-section {
  margin-bottom: 14px;
}
.stack-viz-label {
  font-size: 0.85em;
  font-weight: 600;
  display: block;
  margin-bottom: 6px;
}
.stack-bar {
  display: flex;
  height: 24px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
}
.stack-segment {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 2px;
  transition: width 0.2s;
}
.stack-segment-label {
  font-size: 0.7em;
  font-weight: 600;
  color: #000;
  text-shadow: 0 0 2px rgba(255,255,255,0.6);
}
.stack-legend {
  display: flex;
  gap: 12px;
  margin-top: 6px;
  flex-wrap: wrap;
}
.stack-legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.78em;
  color: var(--vp-c-text-2);
}
.stack-legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}
.svg-wrapper {
  margin-top: 8px;
}
.barl-optimizer-svg {
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
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
</style>
