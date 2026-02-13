<template>
  <div class="angular-response-container">
    <h4>{{ t('Angular Response Simulator', '\uAC01\uB3C4 \uC751\uB2F5 \uC2DC\uBBAC\uB808\uC774\uD130') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how quantum efficiency changes with angle of incidence — critical for understanding Chief Ray Angle (CRA) effects in image sensors.',
        '\uC785\uC0AC\uAC01\uC5D0 \uB530\uB978 \uC591\uC790 \uD6A8\uC728 \uBCC0\uD654\uB97C \uD0D0\uC0C9\uD569\uB2C8\uB2E4 \u2014 \uC774\uBBF8\uC9C0 \uC13C\uC11C\uC758 CRA(Chief Ray Angle) \uD6A8\uACFC\uB97C \uC774\uD574\uD558\uB294 \uB370 \uC911\uC694\uD569\uB2C8\uB2E4.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="toggle-group">
        <button
          :class="['toggle-btn', { active: displayMode === 'single' }]"
          @click="displayMode = 'single'"
        >{{ t('Single Wavelength', '\uB2E8\uC77C \uD30C\uC7A5') }}</button>
        <button
          :class="['toggle-btn', { active: displayMode === 'rgb' }]"
          @click="displayMode = 'rgb'"
        >{{ t('R/G/B Channels', 'R/G/B \uCC44\uB110') }}</button>
      </div>
      <div class="radio-group">
        <label class="radio-label">
          <input type="radio" value="avg" v-model="polarization" /> {{ t('Unpolarized', '\uBE44\uD3B8\uAD11') }}
        </label>
        <label class="radio-label">
          <input type="radio" value="s" v-model="polarization" /> s-pol
        </label>
        <label class="radio-label">
          <input type="radio" value="p" v-model="polarization" /> p-pol
        </label>
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group" v-if="displayMode === 'single'">
        <label>
          {{ t('Wavelength:', '\uD30C\uC7A5:') }} <strong>{{ singleWl }} nm</strong>
        </label>
        <input type="range" min="380" max="780" step="5" v-model.number="singleWl" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Silicon thickness:', '\uC2E4\uB9AC\uCF58 \uB450\uAED8:') }} <strong>{{ siThickness.toFixed(1) }} um</strong>
        </label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Max angle:', '\uCD5C\uB300 \uAC01\uB3C4:') }} <strong>{{ maxAngle }}\u00B0</strong>
        </label>
        <input type="range" min="30" max="80" step="5" v-model.number="maxAngle" class="ctrl-range" />
      </div>
    </div>

    <!-- Info cards -->
    <div class="info-row">
      <template v-if="displayMode === 'single'">
        <div class="info-card">
          <span class="info-label">QE @ 0\u00B0</span>
          <span class="info-value">{{ infoSingle.qe0.toFixed(1) }}%</span>
        </div>
        <div class="info-card">
          <span class="info-label">QE @ 15\u00B0</span>
          <span class="info-value">{{ infoSingle.qe15.toFixed(1) }}%</span>
        </div>
        <div class="info-card">
          <span class="info-label">QE @ 30\u00B0</span>
          <span class="info-value">{{ infoSingle.qe30.toFixed(1) }}%</span>
        </div>
        <div class="info-card">
          <span class="info-label">{{ t('Half-power angle', '\uBC18\uAC10 \uAC01\uB3C4') }}</span>
          <span class="info-value">{{ infoSingle.halfPower }}\u00B0</span>
        </div>
      </template>
      <template v-else>
        <div class="info-card" style="border-left: 3px solid #e74c3c;">
          <span class="info-label">Red @ 0\u00B0</span>
          <span class="info-value">{{ infoRgb.red0.toFixed(1) }}%</span>
        </div>
        <div class="info-card" style="border-left: 3px solid #27ae60;">
          <span class="info-label">Green @ 0\u00B0</span>
          <span class="info-value">{{ infoRgb.green0.toFixed(1) }}%</span>
        </div>
        <div class="info-card" style="border-left: 3px solid #3498db;">
          <span class="info-label">Blue @ 0\u00B0</span>
          <span class="info-value">{{ infoRgb.blue0.toFixed(1) }}%</span>
        </div>
        <div class="info-card">
          <span class="info-label">{{ t('Worst half-power', '\uCD5C\uC800 \uBC18\uAC10') }}</span>
          <span class="info-value">{{ infoRgb.worstHalf }}\u00B0</span>
        </div>
      </template>
    </div>

    <!-- Chart -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="angular-svg"
        @mousemove="onMouseMove"
        @mouseleave="hoverAngle = null"
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
        >{{ tick }}\u00B0</text>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">
          {{ t('Angle of Incidence (\u00B0)', '\uC785\uC0AC\uAC01 (\u00B0)') }}
        </text>
        <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`"
        >QE (%)</text>

        <!-- Curves -->
        <template v-if="displayMode === 'single'">
          <path :d="singleAreaPath" fill="#6c5ce7" opacity="0.08" />
          <path :d="singleLinePath" fill="none" stroke="#6c5ce7" stroke-width="2.5" />
        </template>
        <template v-else>
          <path :d="redAreaPath" fill="#e74c3c" opacity="0.06" />
          <path :d="greenAreaPath" fill="#27ae60" opacity="0.06" />
          <path :d="blueAreaPath" fill="#3498db" opacity="0.06" />
          <path :d="redLinePath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <path :d="greenLinePath" fill="none" stroke="#27ae60" stroke-width="2" />
          <path :d="blueLinePath" fill="none" stroke="#3498db" stroke-width="2" />
        </template>

        <!-- Hover crosshair -->
        <template v-if="hoverAngle !== null">
          <line
            :x1="xScale(hoverAngle)"
            :y1="pad.top"
            :x2="xScale(hoverAngle)"
            :y2="pad.top + plotH"
            stroke="var(--vp-c-text-1)"
            stroke-width="0.8"
            stroke-dasharray="4,3"
          />
          <template v-if="displayMode === 'single'">
            <circle :cx="xScale(hoverAngle)" :cy="yScale(hoverQeSingle)" r="4" fill="#6c5ce7" stroke="#fff" stroke-width="1" />
          </template>
          <template v-else>
            <circle :cx="xScale(hoverAngle)" :cy="yScale(hoverQeRgb.red)" r="4" fill="#e74c3c" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverAngle)" :cy="yScale(hoverQeRgb.green)" r="4" fill="#27ae60" stroke="#fff" stroke-width="1" />
            <circle :cx="xScale(hoverAngle)" :cy="yScale(hoverQeRgb.blue)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
          </template>
          <!-- Tooltip -->
          <rect
            :x="ttX"
            :y="pad.top + 4"
            :width="displayMode === 'single' ? 110 : 120"
            :height="displayMode === 'single' ? 36 : 62"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            opacity="0.95"
          />
          <text :x="ttX + 6" :y="pad.top + 18" class="tooltip-text" font-weight="600">
            {{ hoverAngle }}\u00B0
          </text>
          <template v-if="displayMode === 'single'">
            <text :x="ttX + 6" :y="pad.top + 32" class="tooltip-text" fill="#6c5ce7">
              QE: {{ hoverQeSingle.toFixed(1) }}%
            </text>
          </template>
          <template v-else>
            <text :x="ttX + 6" :y="pad.top + 32" class="tooltip-text" fill="#e74c3c">
              R: {{ hoverQeRgb.red.toFixed(1) }}%
            </text>
            <text :x="ttX + 6" :y="pad.top + 44" class="tooltip-text" fill="#27ae60">
              G: {{ hoverQeRgb.green.toFixed(1) }}%
            </text>
            <text :x="ttX + 6" :y="pad.top + 56" class="tooltip-text" fill="#3498db">
              B: {{ hoverQeRgb.blue.toFixed(1) }}%
            </text>
          </template>
        </template>

        <!-- Legend -->
        <template v-if="displayMode === 'single'">
          <line :x1="pad.left + plotW - 80" :y1="pad.top + 14" :x2="pad.left + plotW - 62" :y2="pad.top + 14" stroke="#6c5ce7" stroke-width="2" />
          <text :x="pad.left + plotW - 58" :y="pad.top + 18" class="legend-label">{{ singleWl }} nm</text>
        </template>
        <template v-else>
          <line :x1="pad.left + plotW - 66" :y1="pad.top + 12" :x2="pad.left + plotW - 48" :y2="pad.top + 12" stroke="#e74c3c" stroke-width="2" />
          <text :x="pad.left + plotW - 44" :y="pad.top + 16" class="legend-label">Red</text>
          <line :x1="pad.left + plotW - 66" :y1="pad.top + 26" :x2="pad.left + plotW - 48" :y2="pad.top + 26" stroke="#27ae60" stroke-width="2" />
          <text :x="pad.left + plotW - 44" :y="pad.top + 30" class="legend-label">Green</text>
          <line :x1="pad.left + plotW - 66" :y1="pad.top + 40" :x2="pad.left + plotW - 48" :y2="pad.top + 40" stroke="#3498db" stroke-width="2" />
          <text :x="pad.left + plotW - 44" :y="pad.top + 44" class="legend-label">Blue</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import {
  tmmCalc, defaultBsiStack, SI_LAYER_IDX,
} from '../composables/tmm'

const { t } = useLocale()

// --- Controls ---
const displayMode = ref<'single' | 'rgb'>('single')
const singleWl = ref(550)
const polarization = ref<'s' | 'p' | 'avg'>('avg')
const siThickness = ref(3.0)
const maxAngle = ref(60)

// --- SVG Layout ---
const svgW = 600
const svgH = 360
const pad = { top: 20, right: 20, bottom: 36, left: 48 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

// --- Scales ---
function xScale(angle: number): number {
  return pad.left + (angle / maxAngle.value) * plotW
}
const yMax = computed(() => {
  // Find max QE across all data to set appropriate y scale
  const allData = displayMode.value === 'single' ? singleData.value : [...rgbData.value.red, ...rgbData.value.green, ...rgbData.value.blue]
  const maxVal = Math.max(...allData.map(d => d.qe), 10)
  return Math.ceil(maxVal / 10) * 10
})
function yScale(pct: number): number {
  return pad.top + plotH - (pct / yMax.value) * plotH
}

const yTicks = computed(() => {
  const max = yMax.value
  const step = max <= 30 ? 5 : max <= 60 ? 10 : 20
  const ticks: number[] = []
  for (let v = 0; v <= max; v += step) ticks.push(v)
  return ticks
})

const xTicks = computed(() => {
  const ma = maxAngle.value
  const step = ma <= 40 ? 5 : 10
  const ticks: number[] = []
  for (let v = 0; v <= ma; v += step) ticks.push(v)
  return ticks
})

// --- TMM computation ---
type AngleQe = { angle: number; qe: number }

function computeAngularQe(cfColor: 'red' | 'green' | 'blue', wlNm: number): AngleQe[] {
  const stack = defaultBsiStack(cfColor, siThickness.value)
  const wl = wlNm / 1000 // nm to um
  const pol = polarization.value
  const data: AngleQe[] = []
  for (let ang = 0; ang <= maxAngle.value; ang += 1) {
    const result = tmmCalc(stack, 'air', 'sio2', wl, ang, pol)
    data.push({ angle: ang, qe: result.layerA[SI_LAYER_IDX] * 100 })
  }
  return data
}

// Typical wavelengths for R/G/B channels
const rgbWavelengths = { red: 620, green: 530, blue: 450 }

const singleData = computed(() => computeAngularQe('green', singleWl.value))

const rgbData = computed(() => ({
  red: computeAngularQe('red', rgbWavelengths.red),
  green: computeAngularQe('green', rgbWavelengths.green),
  blue: computeAngularQe('blue', rgbWavelengths.blue),
}))

// --- Path generation ---
function linePath(data: AngleQe[]): string {
  return data.map((d, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(d.angle).toFixed(1)},${yScale(d.qe).toFixed(1)}`
  }).join(' ')
}
function areaPath(data: AngleQe[]): string {
  const line = linePath(data)
  const last = data[data.length - 1]
  const first = data[0]
  return line + ` L${xScale(last.angle).toFixed(1)},${yScale(0).toFixed(1)} L${xScale(first.angle).toFixed(1)},${yScale(0).toFixed(1)} Z`
}

const singleLinePath = computed(() => linePath(singleData.value))
const singleAreaPath = computed(() => areaPath(singleData.value))
const redLinePath = computed(() => linePath(rgbData.value.red))
const greenLinePath = computed(() => linePath(rgbData.value.green))
const blueLinePath = computed(() => linePath(rgbData.value.blue))
const redAreaPath = computed(() => areaPath(rgbData.value.red))
const greenAreaPath = computed(() => areaPath(rgbData.value.green))
const blueAreaPath = computed(() => areaPath(rgbData.value.blue))

// --- Info cards ---
function findQeAtAngle(data: AngleQe[], angle: number): number {
  const pt = data.find(d => d.angle === angle)
  return pt ? pt.qe : 0
}
function findHalfPower(data: AngleQe[]): number {
  const qe0 = data[0]?.qe ?? 0
  const threshold = qe0 * 0.5
  for (const d of data) {
    if (d.qe <= threshold) return d.angle
  }
  return maxAngle.value
}

const infoSingle = computed(() => ({
  qe0: findQeAtAngle(singleData.value, 0),
  qe15: findQeAtAngle(singleData.value, 15),
  qe30: findQeAtAngle(singleData.value, 30),
  halfPower: findHalfPower(singleData.value),
}))

const infoRgb = computed(() => {
  const rd = rgbData.value
  return {
    red0: findQeAtAngle(rd.red, 0),
    green0: findQeAtAngle(rd.green, 0),
    blue0: findQeAtAngle(rd.blue, 0),
    worstHalf: Math.min(findHalfPower(rd.red), findHalfPower(rd.green), findHalfPower(rd.blue)),
  }
})

// --- Hover ---
const hoverAngle = ref<number | null>(null)

function interpolateQe(data: AngleQe[], angle: number): number {
  if (angle <= 0) return data[0]?.qe ?? 0
  if (angle >= maxAngle.value) return data[data.length - 1]?.qe ?? 0
  const idx = Math.floor(angle)
  if (idx >= data.length - 1) return data[data.length - 1]?.qe ?? 0
  const frac = angle - idx
  return data[idx].qe * (1 - frac) + data[idx + 1].qe * frac
}

const hoverQeSingle = computed(() => {
  if (hoverAngle.value === null) return 0
  return interpolateQe(singleData.value, hoverAngle.value)
})

const hoverQeRgb = computed(() => {
  if (hoverAngle.value === null) return { red: 0, green: 0, blue: 0 }
  const rd = rgbData.value
  return {
    red: interpolateQe(rd.red, hoverAngle.value),
    green: interpolateQe(rd.green, hoverAngle.value),
    blue: interpolateQe(rd.blue, hoverAngle.value),
  }
})

const ttX = computed(() => {
  if (hoverAngle.value === null) return 0
  const x = xScale(hoverAngle.value)
  const w = displayMode.value === 'single' ? 110 : 120
  return x + w + 10 > svgW - pad.right ? x - w - 6 : x + 10
})

function onMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const angle = ((mouseX - pad.left) / plotW) * maxAngle.value
  if (angle < 0 || angle > maxAngle.value) {
    hoverAngle.value = null
    return
  }
  hoverAngle.value = Math.round(angle)
}
</script>

<style scoped>
.angular-response-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.angular-response-container h4 {
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
.radio-group {
  display: flex;
  gap: 12px;
  align-items: center;
}
.radio-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  cursor: pointer;
}
.radio-label input[type="radio"] {
  accent-color: var(--vp-c-brand-1);
}
.slider-group {
  flex: 1;
  min-width: 140px;
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
  margin-bottom: 14px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
  flex: 1;
  min-width: 100px;
  text-align: center;
}
.info-label {
  display: block;
  color: var(--vp-c-text-2);
  font-size: 0.8em;
  margin-bottom: 2px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.svg-wrapper {
  margin-top: 8px;
}
.angular-svg {
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
