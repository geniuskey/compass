<template>
  <div class="energy-budget-container">
    <h4>{{ t('Energy Budget Analyzer', '에너지 버짓 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize where photon energy goes at each wavelength — how much is reflected, absorbed in each layer, or transmitted through the pixel stack.',
        '각 파장에서 광자 에너지가 어디로 가는지 시각화합니다 — 반사, 각 층에서의 흡수, 투과 비율을 확인할 수 있습니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="toggle-group">
        <button
          :class="['toggle-btn', { active: mode === 'single' }]"
          @click="mode = 'single'"
        >{{ t('Single Wavelength', '단일 파장') }}</button>
        <button
          :class="['toggle-btn', { active: mode === 'spectrum' }]"
          @click="mode = 'spectrum'"
        >{{ t('Full Spectrum', '전체 스펙트럼') }}</button>
      </div>
      <div class="select-group">
        <label>{{ t('Color Filter:', '컬러 필터:') }}</label>
        <select v-model="cfColor" class="ctrl-select">
          <option value="red">Red</option>
          <option value="green">Green</option>
          <option value="blue">Blue</option>
        </select>
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group" v-if="mode === 'single'">
        <label>
          {{ t('Wavelength:', '파장:') }} <strong>{{ wavelength }} nm</strong>
        </label>
        <input type="range" min="380" max="780" step="5" v-model.number="wavelength" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Silicon thickness:', '실리콘 두께:') }} <strong>{{ siThickness.toFixed(1) }} um</strong>
        </label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siThickness" class="ctrl-range" />
      </div>
    </div>

    <!-- Single Wavelength Mode -->
    <template v-if="mode === 'single'">
      <div class="stacked-bar-wrapper">
        <svg :viewBox="`0 0 ${barW} ${barH}`" class="bar-svg">
          <rect
            v-for="seg in singleBarSegments"
            :key="seg.label"
            :x="seg.x"
            :y="10"
            :width="Math.max(0, seg.w)"
            :height="40"
            :fill="seg.color"
            rx="1"
          />
          <text
            v-for="seg in singleBarSegments"
            :key="'t' + seg.label"
            v-show="seg.pct > 3"
            :x="seg.x + seg.w / 2"
            :y="34"
            text-anchor="middle"
            class="bar-label"
          >{{ seg.pct.toFixed(1) }}%</text>
        </svg>
      </div>
      <div class="legend-row">
        <span v-for="seg in singleBarSegments" :key="'l' + seg.label" class="legend-item">
          <span class="legend-swatch" :style="{ background: seg.color }"></span>
          {{ seg.label }}
        </span>
      </div>
      <div class="summary-text">
        {{ t(
          `At \u03BB=${wavelength}nm: QE=${singleResult.qe.toFixed(1)}%, R=${singleResult.R.toFixed(1)}%, Other losses=${singleResult.otherLoss.toFixed(1)}%`,
          `\u03BB=${wavelength}nm: QE=${singleResult.qe.toFixed(1)}%, R=${singleResult.R.toFixed(1)}%, \uAE30\uD0C0 \uC190\uC2E4=${singleResult.otherLoss.toFixed(1)}%`
        ) }}
      </div>
    </template>

    <!-- Full Spectrum Mode -->
    <template v-if="mode === 'spectrum'">
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${svgW} ${svgH}`"
          class="spectrum-svg"
          @mousemove="onMouseMove"
          @mouseleave="hoverIdx = null"
        >
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
            :y="pad.top + plotH + 20"
            text-anchor="middle"
            class="axis-label"
          >{{ tick }}</text>

          <!-- Axis titles -->
          <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">
            {{ t('Wavelength (nm)', '\uD30C\uC7A5 (nm)') }}
          </text>
          <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
            :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`"
          >{{ t('Energy (%)', '\uC5D0\uB108\uC9C0 (%)') }}</text>

          <!-- Stacked areas (bottom to top): Silicon, BARL, CF, Planarization, Microlens, Reflection, Transmission -->
          <path
            v-for="area in stackedAreas"
            :key="area.label"
            :d="area.path"
            :fill="area.color"
            :opacity="0.75"
          />

          <!-- Hover crosshair -->
          <template v-if="hoverIdx !== null">
            <line
              :x1="xScale(spectrumWls[hoverIdx] * 1000)"
              :y1="pad.top"
              :x2="xScale(spectrumWls[hoverIdx] * 1000)"
              :y2="pad.top + plotH"
              stroke="var(--vp-c-text-1)"
              stroke-width="0.8"
              stroke-dasharray="4,3"
            />
            <!-- Tooltip -->
            <rect
              :x="tooltipX"
              :y="pad.top + 4"
              width="156"
              height="106"
              rx="4"
              fill="var(--vp-c-bg)"
              stroke="var(--vp-c-divider)"
              stroke-width="0.8"
              opacity="0.95"
            />
            <text :x="tooltipX + 6" :y="pad.top + 17" class="tooltip-text" font-weight="600">
              {{ (spectrumWls[hoverIdx] * 1000).toFixed(0) }} nm
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 31" class="tooltip-text" :fill="colors.silicon">
              Silicon (QE): {{ (hoverBreakdown.silicon * 100).toFixed(1) }}%
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 44" class="tooltip-text" :fill="colors.barl">
              BARL: {{ (hoverBreakdown.barl * 100).toFixed(1) }}%
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 57" class="tooltip-text" :fill="cfColorHex">
              CF: {{ (hoverBreakdown.cf * 100).toFixed(1) }}%
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 70" class="tooltip-text" :fill="colors.planarization">
              Plnr: {{ (hoverBreakdown.planarization * 100).toFixed(1) }}%
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 83" class="tooltip-text" :fill="colors.microlens">
              Microlens: {{ (hoverBreakdown.microlens * 100).toFixed(1) }}%
            </text>
            <text :x="tooltipX + 6" :y="pad.top + 96" class="tooltip-text" :fill="colors.reflection">
              R: {{ (hoverBreakdown.reflection * 100).toFixed(1) }}%
              T: {{ (hoverBreakdown.transmission * 100).toFixed(1) }}%
            </text>
          </template>
        </svg>
      </div>
      <div class="legend-row">
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.silicon }"></span>{{ t('Silicon (QE)', '\uC2E4\uB9AC\uCF58 (QE)') }}</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.barl }"></span>BARL</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: cfColorHex }"></span>{{ t('Color Filter', '\uCEEC\uB7EC \uD544\uD130') }}</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.planarization }"></span>{{ t('Planarization', '\uD3C9\uD0C4\uD654\uCE35') }}</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.microlens }"></span>{{ t('Microlens', '\uB9C8\uC774\uD06C\uB85C\uB80C\uC988') }}</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.reflection }"></span>{{ t('Reflection', '\uBC18\uC0AC') }}</span>
        <span class="legend-item"><span class="legend-swatch" :style="{ background: colors.transmission }"></span>{{ t('Transmission', '\uD22C\uACFC') }}</span>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import {
  tmmCalc, tmmSpectrum, wlRange, defaultBsiStack,
  SI_LAYER_IDX, type TmmResult,
} from '../composables/tmm'

const { t } = useLocale()

// --- Controls ---
const mode = ref<'single' | 'spectrum'>('single')
const cfColor = ref<'red' | 'green' | 'blue'>('green')
const wavelength = ref(550)
const siThickness = ref(3.0)

// --- Colors ---
const colors = {
  reflection: '#3498db',
  microlens: '#dda0dd',
  planarization: '#d5dbdb',
  barl: '#8e44ad',
  silicon: '#5d6d7e',
  transmission: '#f39c12',
}
const cfColors: Record<string, string> = { red: '#c0392b', green: '#27ae60', blue: '#2980b9' }
const cfColorHex = computed(() => cfColors[cfColor.value])

// --- Helpers ---
interface Breakdown {
  reflection: number
  microlens: number
  planarization: number
  cf: number
  barl: number
  silicon: number
  transmission: number
}

function resultToBreakdown(r: TmmResult): Breakdown {
  const barlA = r.layerA[3] + r.layerA[4] + r.layerA[5] + r.layerA[6]
  return {
    reflection: r.R,
    microlens: r.layerA[0],
    planarization: r.layerA[1],
    cf: r.layerA[2],
    barl: barlA,
    silicon: r.layerA[SI_LAYER_IDX],
    transmission: r.T,
  }
}

// --- Single Wavelength Mode ---
const barW = 600
const barH = 60

const singleResult = computed(() => {
  const stack = defaultBsiStack(cfColor.value, siThickness.value)
  const wl = wavelength.value / 1000 // nm to um
  const r = tmmCalc(stack, 'air', 'sio2', wl)
  const bd = resultToBreakdown(r)
  const qe = bd.silicon * 100
  const R = bd.reflection * 100
  const otherLoss = 100 - qe - R
  return { ...bd, qe, R, otherLoss }
})

const singleBarSegments = computed(() => {
  const bd = singleResult.value
  const items = [
    { label: t('Reflection', '\uBC18\uC0AC'), pct: bd.reflection * 100, color: colors.reflection },
    { label: t('Microlens', '\uB9C8\uC774\uD06C\uB85C\uB80C\uC988'), pct: bd.microlens * 100, color: colors.microlens },
    { label: t('Planarization', '\uD3C9\uD0C4\uD654\uCE35'), pct: bd.planarization * 100, color: colors.planarization },
    { label: t('Color Filter', '\uCEEC\uB7EC \uD544\uD130'), pct: bd.cf * 100, color: cfColorHex.value },
    { label: 'BARL', pct: bd.barl * 100, color: colors.barl },
    { label: t('Silicon (QE)', '\uC2E4\uB9AC\uCF58 (QE)'), pct: bd.silicon * 100, color: colors.silicon },
    { label: t('Transmission', '\uD22C\uACFC'), pct: bd.transmission * 100, color: colors.transmission },
  ]
  const usable = barW - 20
  let x = 10
  return items.map(it => {
    const w = (it.pct / 100) * usable
    const seg = { ...it, x, w }
    x += w
    return seg
  })
})

// --- Full Spectrum Mode ---
const svgW = 600
const svgH = 320
const pad = { top: 20, right: 20, bottom: 36, left: 48 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom
const yTicks = [0, 20, 40, 60, 80, 100]
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function xScale(wlNm: number): number {
  return pad.left + ((wlNm - 380) / (780 - 380)) * plotW
}
function yScale(pct: number): number {
  return pad.top + plotH - (pct / 100) * plotH
}

const spectrumWls = computed(() => wlRange(0.38, 0.78, 0.005))

const spectrumBreakdowns = computed(() => {
  const stack = defaultBsiStack(cfColor.value, siThickness.value)
  const results = tmmSpectrum(stack, 'air', 'sio2', spectrumWls.value)
  return results.map(r => resultToBreakdown(r))
})

const stackedAreas = computed(() => {
  const bds = spectrumBreakdowns.value
  const wls = spectrumWls.value
  const n = wls.length
  const keys: (keyof Breakdown)[] = ['silicon', 'barl', 'cf', 'planarization', 'microlens', 'reflection', 'transmission']
  const colorMap: Record<string, string> = {
    silicon: colors.silicon,
    barl: colors.barl,
    cf: cfColorHex.value,
    planarization: colors.planarization,
    microlens: colors.microlens,
    reflection: colors.reflection,
    transmission: colors.transmission,
  }

  // Compute cumulative sums
  const cumulative: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    let sum = 0
    for (const k of keys) {
      row.push(sum)
      sum += bds[i][k]
    }
    row.push(sum)
    cumulative.push(row)
  }

  const areas: { label: string; color: string; path: string }[] = []
  for (let ki = 0; ki < keys.length; ki++) {
    const key = keys[ki]
    // Upper boundary: cumulative[i][ki+1], lower boundary: cumulative[i][ki]
    let path = ''
    // Forward pass for upper boundary
    for (let i = 0; i < n; i++) {
      const x = xScale(wls[i] * 1000).toFixed(1)
      const y = yScale(cumulative[i][ki + 1] * 100).toFixed(1)
      path += i === 0 ? `M${x},${y}` : ` L${x},${y}`
    }
    // Backward pass for lower boundary
    for (let i = n - 1; i >= 0; i--) {
      const x = xScale(wls[i] * 1000).toFixed(1)
      const y = yScale(cumulative[i][ki] * 100).toFixed(1)
      path += ` L${x},${y}`
    }
    path += ' Z'
    areas.push({ label: key, color: colorMap[key], path })
  }
  return areas
})

// --- Hover ---
const hoverIdx = ref<number | null>(null)

const hoverBreakdown = computed(() => {
  if (hoverIdx.value === null) return resultToBreakdown({ R: 0, T: 0, A: 0, layerA: [0,0,0,0,0,0,0,0] })
  return spectrumBreakdowns.value[hoverIdx.value]
})

const tooltipX = computed(() => {
  if (hoverIdx.value === null) return 0
  const x = xScale(spectrumWls.value[hoverIdx.value] * 1000)
  return x + 170 > svgW - pad.right ? x - 166 : x + 10
})

function onMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wlNm = 380 + ((mouseX - pad.left) / plotW) * (780 - 380)
  if (wlNm < 380 || wlNm > 780) {
    hoverIdx.value = null
    return
  }
  // Find closest index
  const wls = spectrumWls.value
  let bestIdx = 0
  let bestDist = Infinity
  for (let i = 0; i < wls.length; i++) {
    const dist = Math.abs(wls[i] * 1000 - wlNm)
    if (dist < bestDist) { bestDist = dist; bestIdx = i }
  }
  hoverIdx.value = bestIdx
}
</script>

<style scoped>
.energy-budget-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.energy-budget-container h4 {
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
  margin-bottom: 12px;
}
.toggle-group {
  display: flex;
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  overflow: hidden;
}
.toggle-btn {
  padding: 6px 14px;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.toggle-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.toggle-btn:not(.active):hover {
  background: var(--vp-c-bg-soft);
}
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.85em;
  font-weight: 600;
}
.ctrl-select {
  padding: 5px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.85em;
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
.stacked-bar-wrapper {
  margin: 12px 0 4px 0;
}
.bar-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.bar-label {
  font-size: 9px;
  fill: #fff;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  pointer-events: none;
}
.legend-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 8px 0 4px 0;
  justify-content: center;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.78em;
  color: var(--vp-c-text-2);
}
.legend-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}
.summary-text {
  margin-top: 10px;
  padding: 10px 14px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 0.88em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
  text-align: center;
}
.svg-wrapper {
  margin-top: 8px;
}
.spectrum-svg {
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
