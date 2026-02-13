<template>
  <div class="mtf-container">
    <h4>{{ t('MTF Analyzer', 'MTF 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Interactive Modulation Transfer Function analyzer. Explore how pixel aperture and diffraction optics limit spatial resolution.',
        '인터랙티브 변조 전달 함수 분석기. 픽셀 개구와 회절 광학이 공간 해상도를 어떻게 제한하는지 살펴보세요.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} &mu;m</strong>
        </label>
        <input type="range" min="0.5" max="3.0" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('f-number', 'f값') }}: <strong>f/{{ fNumber.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.4" max="11.0" step="0.1" v-model.number="fNumber" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelengthNm }} nm</strong>
        </label>
        <input type="range" min="400" max="700" step="10" v-model.number="wavelengthNm" class="ctrl-range" />
      </div>
      <div class="check-group">
        <label>
          <input type="checkbox" v-model="showComponents" />
          {{ t('Show components', '구성 요소 표시') }}
        </label>
      </div>
    </div>

    <!-- Info cards -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Nyquist', '나이퀴스트') }}</div>
        <div class="result-value">{{ nyquistLpmm.toFixed(0) }} lp/mm</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Optical Cutoff', '광학 차단') }}</div>
        <div class="result-value">{{ cutoffLpmm.toFixed(0) }} lp/mm</div>
      </div>
      <div class="result-card">
        <div class="result-label">MTF@Nyquist</div>
        <div class="result-value highlight">{{ (mtfAtNyquist * 100).toFixed(1) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">MTF@Nyquist/2</div>
        <div class="result-value">{{ (mtfAtHalfNyquist * 100).toFixed(1) }}%</div>
      </div>
    </div>

    <!-- Main MTF chart -->
    <div class="chart-section">
      <h5>{{ t('MTF Curves', 'MTF 곡선') }}</h5>
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${chartW} ${chartH}`"
          class="mtf-svg"
          @mousemove="onChartMouseMove"
          @mouseleave="chartHover = null"
        >
          <!-- Grid lines -->
          <line
            v-for="tick in yTicks" :key="'yg'+tick"
            :x1="pad.left" :y1="yScale(tick)"
            :x2="pad.left + plotW" :y2="yScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            v-for="tick in xTicks" :key="'xg'+tick"
            :x1="xScale(tick)" :y1="pad.top"
            :x2="xScale(tick)" :y2="pad.top + plotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />

          <!-- Aliasing zone (beyond Nyquist) -->
          <rect
            v-if="nyquistFreq < xMax"
            :x="xScale(nyquistFreq)"
            :y="pad.top"
            :width="Math.max(0, xScale(xMax) - xScale(nyquistFreq))"
            :height="plotH"
            fill="#e74c3c" opacity="0.08"
          />

          <!-- Axes -->
          <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <!-- Top axis line for lp/mm -->
          <line :x1="pad.left" :y1="pad.top" :x2="pad.left + plotW" :y2="pad.top" stroke="var(--vp-c-text-2)" stroke-width="0.5" />

          <!-- X tick labels (cycles/um) -->
          <template v-for="tick in xTicks" :key="'xl'+tick">
            <text :x="xScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">
              {{ tick.toFixed(1) }}
            </text>
          </template>
          <!-- Top X tick labels (lp/mm) -->
          <template v-for="tick in xTicks" :key="'xlt'+tick">
            <text :x="xScale(tick)" :y="pad.top - 5" text-anchor="middle" class="tick-label" fill="var(--vp-c-text-3)">
              {{ (tick * 1000).toFixed(0) }}
            </text>
          </template>

          <!-- Y tick labels -->
          <template v-for="tick in yTicks" :key="'yl'+tick">
            <text :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">
              {{ tick.toFixed(1) }}
            </text>
          </template>

          <!-- Axis titles -->
          <text :x="pad.left + plotW / 2" :y="chartH - 2" text-anchor="middle" class="axis-title">
            {{ t('Spatial Frequency (cycles/\u00B5m)', '\uACF5\uAC04 \uC8FC\uD30C\uC218 (cycles/\u00B5m)') }}
          </text>
          <text :x="pad.left + plotW / 2" :y="pad.top - 16" text-anchor="middle" class="axis-title" fill="var(--vp-c-text-3)">
            (lp/mm)
          </text>
          <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
            :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`">
            MTF
          </text>

          <!-- Pixel MTF curve (dashed blue) -->
          <path v-if="showComponents" :d="pixelMtfPath" fill="none" stroke="#3498db" stroke-width="2" stroke-dasharray="6,3" />
          <!-- Optical MTF curve (dashed green) -->
          <path v-if="showComponents" :d="opticalMtfPath" fill="none" stroke="#27ae60" stroke-width="2" stroke-dasharray="6,3" />
          <!-- Combined MTF curve (solid purple) -->
          <path :d="combinedMtfPath" fill="none" stroke="#8e44ad" stroke-width="2.5" />

          <!-- Nyquist vertical line -->
          <line
            v-if="nyquistFreq <= xMax"
            :x1="xScale(nyquistFreq)" :y1="pad.top"
            :x2="xScale(nyquistFreq)" :y2="pad.top + plotH"
            stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="5,4"
          />
          <text
            v-if="nyquistFreq <= xMax"
            :x="xScale(nyquistFreq)" :y="pad.top + plotH + 26"
            text-anchor="middle" class="marker-label" fill="#e74c3c"
          >Nyquist</text>

          <!-- Optical cutoff vertical line -->
          <line
            v-if="cutoffFreq <= xMax"
            :x1="xScale(cutoffFreq)" :y1="pad.top"
            :x2="xScale(cutoffFreq)" :y2="pad.top + plotH"
            stroke="#27ae60" stroke-width="1.5" stroke-dasharray="5,4"
          />
          <text
            v-if="cutoffFreq <= xMax"
            :x="xScale(cutoffFreq)" :y="pad.top + plotH + 26"
            text-anchor="middle" class="marker-label" fill="#27ae60"
          >{{ t('Cutoff', '차단') }}</text>

          <!-- Legend -->
          <g :transform="`translate(${pad.left + plotW - 150}, ${pad.top + 8})`">
            <rect x="-4" y="-4" width="155" height="56" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.5" opacity="0.9" />
            <template v-if="showComponents">
              <line x1="0" y1="8" x2="18" y2="8" stroke="#3498db" stroke-width="2" stroke-dasharray="6,3" />
              <text x="22" y="12" class="legend-label">{{ t('Pixel MTF', '픽셀 MTF') }}</text>
              <line x1="0" y1="22" x2="18" y2="22" stroke="#27ae60" stroke-width="2" stroke-dasharray="6,3" />
              <text x="22" y="26" class="legend-label">{{ t('Optical MTF', '광학 MTF') }}</text>
            </template>
            <line x1="0" :y1="showComponents ? 36 : 12" x2="18" :y2="showComponents ? 36 : 12" stroke="#8e44ad" stroke-width="2.5" />
            <text x="22" :y="showComponents ? 40 : 16" class="legend-label">{{ t('Combined MTF', '합성 MTF') }}</text>
          </g>

          <!-- Hover tooltip -->
          <template v-if="chartHover">
            <line :x1="chartHover.sx" :y1="pad.top" :x2="chartHover.sx" :y2="pad.top + plotH"
              stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
            <rect :x="chartHover.tx" :y="pad.top + 4" width="160" height="58" rx="4"
              fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
            <text :x="chartHover.tx + 6" :y="pad.top + 17" class="tooltip-text">
              f: {{ chartHover.freq.toFixed(3) }} cy/&mu;m ({{ (chartHover.freq * 1000).toFixed(0) }} lp/mm)
            </text>
            <text :x="chartHover.tx + 6" :y="pad.top + 29" class="tooltip-text" fill="#3498db">
              {{ t('Pixel', '픽셀') }}: {{ (chartHover.pixelMtf * 100).toFixed(1) }}%
            </text>
            <text :x="chartHover.tx + 6" :y="pad.top + 41" class="tooltip-text" fill="#27ae60">
              {{ t('Optical', '광학') }}: {{ (chartHover.optMtf * 100).toFixed(1) }}%
            </text>
            <text :x="chartHover.tx + 6" :y="pad.top + 53" class="tooltip-text" fill="#8e44ad">
              {{ t('Combined', '합성') }}: {{ (chartHover.combMtf * 100).toFixed(1) }}%
            </text>
          </template>
        </svg>
      </div>
    </div>

    <!-- Bar target simulation -->
    <div class="chart-section">
      <h5>{{ t('Bar Target Simulation', '바 타겟 시뮬레이션') }}</h5>
      <p class="bar-description">
        {{ t(
          'Shows how contrast degrades at increasing spatial frequencies. Each group shows bars at a fraction of the Nyquist frequency.',
          '공간 주파수가 증가함에 따라 대비가 어떻게 저하되는지 보여줍니다. 각 그룹은 나이퀴스트 주파수의 일부에서 바를 나타냅니다.'
        ) }}
      </p>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${barW} ${barH}`" class="bar-svg">
          <template v-for="(group, gi) in barGroups" :key="'bg'+gi">
            <!-- Group label -->
            <text
              :x="group.cx"
              :y="barH - 4"
              text-anchor="middle"
              class="bar-label"
            >{{ group.label }}</text>
            <!-- Bar rectangles -->
            <rect
              v-for="(bar, bi) in group.bars" :key="'b'+gi+'-'+bi"
              :x="bar.x"
              :y="barPad.top"
              :width="bar.w"
              :height="barPlotH"
              :fill="bar.fill"
            />
          </template>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// ---- Controls ----
const pitch = ref(1.0)
const fNumber = ref(2.8)
const wavelengthNm = ref(550)
const showComponents = ref(true)

// ---- Physics ----
const wavelengthUm = computed(() => wavelengthNm.value / 1000)
const nyquistFreq = computed(() => 1 / (2 * pitch.value)) // cycles/um
const cutoffFreq = computed(() => 1 / (wavelengthUm.value * fNumber.value)) // cycles/um
const nyquistLpmm = computed(() => nyquistFreq.value * 1000)
const cutoffLpmm = computed(() => cutoffFreq.value * 1000)

function sinc(x: number): number {
  if (Math.abs(x) < 1e-10) return 1
  return Math.sin(Math.PI * x) / (Math.PI * x)
}

function mtfPixel(f: number): number {
  return Math.abs(sinc(f * pitch.value))
}

function mtfOptical(f: number): number {
  const fc = cutoffFreq.value
  if (fc <= 0 || f >= fc) return 0
  const r = f / fc
  return (2 / Math.PI) * (Math.acos(r) - r * Math.sqrt(1 - r * r))
}

function mtfCombined(f: number): number {
  return mtfPixel(f) * mtfOptical(f)
}

const mtfAtNyquist = computed(() => mtfCombined(nyquistFreq.value))
const mtfAtHalfNyquist = computed(() => mtfCombined(nyquistFreq.value / 2))

// ---- Chart dimensions ----
const chartW = 600
const chartH = 320
const pad = { top: 32, right: 20, bottom: 42, left: 45 }
const plotW = chartW - pad.left - pad.right
const plotH = chartH - pad.top - pad.bottom

// X-axis range: 0 to max(2*Nyquist, 1.2*cutoff)
const xMax = computed(() => {
  const v = Math.max(2 * nyquistFreq.value, 1.2 * cutoffFreq.value)
  // Round up to a nice number
  if (v <= 0.5) return 0.5
  if (v <= 1.0) return 1.0
  if (v <= 1.5) return 1.5
  if (v <= 2.0) return 2.0
  if (v <= 3.0) return 3.0
  return Math.ceil(v)
})

const xTicks = computed(() => {
  const max = xMax.value
  const step = max <= 1.0 ? 0.2 : max <= 2.0 ? 0.4 : max <= 3.0 ? 0.5 : 1.0
  const ticks: number[] = []
  for (let v = 0; v <= max + step * 0.01; v += step) {
    ticks.push(Math.round(v * 100) / 100)
  }
  return ticks
})

const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

function xScale(f: number): number {
  return pad.left + (f / xMax.value) * plotW
}
function yScale(mtf: number): number {
  return pad.top + plotH - mtf * plotH
}

// ---- Build curve paths ----
const numSteps = 300

function buildCurvePath(fn: (f: number) => number): string {
  let d = ''
  for (let i = 0; i <= numSteps; i++) {
    const f = (i / numSteps) * xMax.value
    const val = Math.max(0, Math.min(1, fn(f)))
    const x = xScale(f)
    const y = yScale(val)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

const pixelMtfPath = computed(() => buildCurvePath(mtfPixel))
const opticalMtfPath = computed(() => buildCurvePath(mtfOptical))
const combinedMtfPath = computed(() => buildCurvePath(mtfCombined))

// ---- Chart hover ----
const chartHover = ref<{
  sx: number; tx: number; freq: number;
  pixelMtf: number; optMtf: number; combMtf: number
} | null>(null)

function onChartMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = chartW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const freq = ((mouseX - pad.left) / plotW) * xMax.value
  if (freq >= 0 && freq <= xMax.value) {
    const sx = xScale(freq)
    const tx = sx + 170 > chartW - pad.right ? sx - 170 : sx + 10
    chartHover.value = {
      sx, tx, freq,
      pixelMtf: mtfPixel(freq),
      optMtf: mtfOptical(freq),
      combMtf: mtfCombined(freq)
    }
  } else {
    chartHover.value = null
  }
}

// ---- Bar target simulation ----
const barW = 600
const barH = 120
const barPad = { top: 8, bottom: 22, left: 10, right: 10 }
const barPlotH = barH - barPad.top - barPad.bottom

const barFreqMultipliers = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

const barGroups = computed(() => {
  const fN = nyquistFreq.value
  const numGroups = barFreqMultipliers.length
  const totalW = barW - barPad.left - barPad.right
  const groupW = totalW / numGroups
  const gapW = 6 // gap between groups

  return barFreqMultipliers.map((mult, gi) => {
    const freq = mult * fN
    const mtf = mtfCombined(freq)
    const groupStartX = barPad.left + gi * groupW + gapW / 2
    const availW = groupW - gapW

    // Create thin vertical bars to simulate sinusoidal pattern
    const numBars = 40
    const barWidth = availW / numBars
    const bars: { x: number; w: number; fill: string }[] = []

    for (let bi = 0; bi < numBars; bi++) {
      const xPos = groupStartX + bi * barWidth
      // Position within the group as spatial coordinate
      const spatialX = bi / numBars
      // I(x) = 0.5 + 0.5 * MTF * cos(2*pi*f_rel*x)
      // Use a few cycles within the group width
      const numCycles = 3
      const intensity = 0.5 + 0.5 * mtf * Math.cos(2 * Math.PI * numCycles * spatialX)
      const gray = Math.round(intensity * 255)
      bars.push({
        x: xPos,
        w: barWidth + 0.5, // slight overlap to avoid gaps
        fill: `rgb(${gray},${gray},${gray})`
      })
    }

    const label = `${mult.toFixed(1)}fN`
    const cx = groupStartX + availW / 2

    return { bars, label, cx }
  })
})
</script>

<style scoped>
.mtf-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.mtf-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.mtf-container h5 {
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
  align-items: flex-end;
}
.slider-group {
  flex: 1;
  min-width: 180px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.check-group {
  display: flex;
  align-items: center;
  min-width: 140px;
  padding-bottom: 4px;
}
.check-group label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85em;
  cursor: pointer;
}
.check-group input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: var(--vp-c-brand-1);
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
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.mtf-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.bar-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
  background: var(--vp-c-bg);
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
}
.bar-description {
  margin: 0 0 8px 0;
  color: var(--vp-c-text-3);
  font-size: 0.82em;
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
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.marker-label {
  font-size: 9px;
  font-weight: 600;
}
.tooltip-text {
  font-size: 8.5px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.bar-label {
  font-size: 8.5px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
</style>
