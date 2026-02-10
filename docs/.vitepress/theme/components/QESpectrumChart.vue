<template>
  <div class="qe-spectrum-container">
    <h4>Interactive QE Spectrum Chart</h4>
    <p class="component-description">
      Explore how silicon thickness, BARL quality, and metal grid width affect the quantum efficiency spectrum of Red, Green, and Blue channels.
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          Silicon thickness: <strong>{{ siliconThickness.toFixed(1) }} um</strong>
        </label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siliconThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          BARL quality: <strong>{{ barlQuality }}%</strong>
        </label>
        <input type="range" min="0" max="100" step="1" v-model.number="barlQuality" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          Metal grid width: <strong>{{ gridWidth }} nm</strong>
        </label>
        <input type="range" min="0" max="100" step="1" v-model.number="gridWidth" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card" style="border-left: 3px solid #3498db;">
        <span class="info-label">Blue peak QE:</span>
        <span class="info-value">{{ peakBlue.toFixed(1) }}%</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #27ae60;">
        <span class="info-label">Green peak QE:</span>
        <span class="info-value">{{ peakGreen.toFixed(1) }}%</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #e74c3c;">
        <span class="info-label">Red peak QE:</span>
        <span class="info-value">{{ peakRed.toFixed(1) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">Average QE:</span>
        <span class="info-value">{{ avgQE.toFixed(1) }}%</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="qe-svg"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
      >
        <!-- Visible spectrum band on x-axis -->
        <defs>
          <linearGradient id="qeVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>
        <rect
          :x="pad.left"
          :y="pad.top + plotH + 2"
          :width="plotW"
          height="10"
          fill="url(#qeVisSpectrum)"
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
        >{{ tick }}%</text>

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
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">Wavelength (nm)</text>
        <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" transform="rotate(-90, 12, 125)">QE (%)</text>

        <!-- Blue curve -->
        <path :d="bluePath" fill="none" stroke="#3498db" stroke-width="2" opacity="0.9" />
        <!-- Green curve -->
        <path :d="greenPath" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.9" />
        <!-- Red curve -->
        <path :d="redPath" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.9" />

        <!-- Filled areas -->
        <path :d="blueArea" fill="#3498db" opacity="0.08" />
        <path :d="greenArea" fill="#27ae60" opacity="0.08" />
        <path :d="redArea" fill="#e74c3c" opacity="0.08" />

        <!-- Crosshair on hover -->
        <template v-if="hoverWl !== null">
          <line :x1="xScale(hoverWl)" :y1="pad.top" :x2="xScale(hoverWl)" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQE.blue)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQE.green)" r="4" fill="#27ae60" stroke="#fff" stroke-width="1" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverQE.red)" r="4" fill="#e74c3c" stroke="#fff" stroke-width="1" />
          <!-- Tooltip -->
          <rect :x="tooltipX" :y="pad.top + 4" width="120" height="58" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#3498db">B: {{ hoverQE.blue.toFixed(1) }}%</text>
          <text :x="tooltipX + 8" :y="pad.top + 44" class="tooltip-text" fill="#27ae60">G: {{ hoverQE.green.toFixed(1) }}%</text>
          <text :x="tooltipX + 8" :y="pad.top + 56" class="tooltip-text" fill="#e74c3c">R: {{ hoverQE.red.toFixed(1) }}%</text>
        </template>

        <!-- Legend -->
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 12" :x2="pad.left + plotW - 72" :y2="pad.top + 12" stroke="#3498db" stroke-width="2" />
        <text :x="pad.left + plotW - 68" :y="pad.top + 16" class="legend-label">Blue</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 26" :x2="pad.left + plotW - 72" :y2="pad.top + 26" stroke="#27ae60" stroke-width="2" />
        <text :x="pad.left + plotW - 68" :y="pad.top + 30" class="legend-label">Green</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 40" :x2="pad.left + plotW - 72" :y2="pad.top + 40" stroke="#e74c3c" stroke-width="2" />
        <text :x="pad.left + plotW - 68" :y="pad.top + 44" class="legend-label">Red</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const siliconThickness = ref(3.0)
const barlQuality = ref(70)
const gridWidth = ref(50)

const svgW = 520
const svgH = 300
const pad = { top: 20, right: 20, bottom: 40, left: 50 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 380
const wlMax = 780
const wlStep = 2
const yTicks = [0, 20, 40, 60, 80]
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function xScale(wl) {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}

function yScale(qe) {
  return pad.top + plotH - (qe / 80) * plotH
}

// Gaussian model for each channel
function gaussian(wl, peak, sigma, peakQE) {
  return peakQE * Math.exp(-0.5 * Math.pow((wl - peak) / sigma, 2))
}

// Silicon absorption rolloff: longer wavelengths need thicker silicon
function siAbsorptionFactor(wl, thickness) {
  // Absorption length increases dramatically at long wavelengths
  // At 400nm: ~0.1um, at 550nm: ~1.5um, at 700nm: ~8um
  const absLength = 0.1 * Math.exp(0.0045 * (wl - 400))
  return 1 - Math.exp(-thickness / absLength)
}

function computeQE(wl, channel) {
  let peak, sigma, baseQE
  if (channel === 'blue') { peak = 450; sigma = 30; baseQE = 60 }
  else if (channel === 'green') { peak = 530; sigma = 35; baseQE = 70 }
  else { peak = 610; sigma = 35; baseQE = 55 }

  // Base Gaussian shape
  let qe = gaussian(wl, peak, sigma, baseQE)

  // BARL quality scales overall QE
  const barlFactor = 0.4 + 0.6 * (barlQuality.value / 100)
  qe *= barlFactor

  // Silicon thickness affects long-wavelength QE
  const siFactor = siAbsorptionFactor(wl, siliconThickness.value)
  qe *= siFactor

  // Metal grid width affects channel separation (less overlap when wider)
  // Wider grid narrows the effective filter bandwidth
  const gridFactor = 1.0 - 0.002 * gridWidth.value
  const overlapReduction = 1.0 + 0.005 * gridWidth.value
  const distFromPeak = Math.abs(wl - peak) / sigma
  if (distFromPeak > 1.5) {
    qe *= Math.max(0, gridFactor * Math.exp(-0.1 * gridWidth.value / 100 * (distFromPeak - 1.5)))
  } else {
    qe *= (1.0 - 0.001 * gridWidth.value)
  }

  return Math.max(0, Math.min(80, qe))
}

function computeWlPoints(channel) {
  const pts = []
  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    pts.push({ wl, qe: computeQE(wl, channel) })
  }
  return pts
}

function pathFromPoints(pts) {
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(p.wl).toFixed(1)},${yScale(p.qe).toFixed(1)}`
  }).join(' ')
}

function areaFromPoints(pts) {
  const line = pathFromPoints(pts)
  const lastPt = pts[pts.length - 1]
  const firstPt = pts[0]
  return line + ` L${xScale(lastPt.wl).toFixed(1)},${yScale(0).toFixed(1)} L${xScale(firstPt.wl).toFixed(1)},${yScale(0).toFixed(1)} Z`
}

const bluePoints = computed(() => computeWlPoints('blue'))
const greenPoints = computed(() => computeWlPoints('green'))
const redPoints = computed(() => computeWlPoints('red'))

const bluePath = computed(() => pathFromPoints(bluePoints.value))
const greenPath = computed(() => pathFromPoints(greenPoints.value))
const redPath = computed(() => pathFromPoints(redPoints.value))

const blueArea = computed(() => areaFromPoints(bluePoints.value))
const greenArea = computed(() => areaFromPoints(greenPoints.value))
const redArea = computed(() => areaFromPoints(redPoints.value))

const peakBlue = computed(() => Math.max(...bluePoints.value.map(p => p.qe)))
const peakGreen = computed(() => Math.max(...greenPoints.value.map(p => p.qe)))
const peakRed = computed(() => Math.max(...redPoints.value.map(p => p.qe)))
const avgQE = computed(() => (peakBlue.value + peakGreen.value + peakRed.value) / 3)

// Hover crosshair
const hoverWl = ref(null)
const hoverQE = ref({ blue: 0, green: 0, red: 0 })

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const snapped = Math.round(wl)
    hoverWl.value = snapped
    hoverQE.value = {
      blue: computeQE(snapped, 'blue'),
      green: computeQE(snapped, 'green'),
      red: computeQE(snapped, 'red'),
    }
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
  return x + 130 > svgW - pad.right ? x - 130 : x + 10
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
.qe-spectrum-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.qe-spectrum-container h4 {
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
.qe-svg {
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
