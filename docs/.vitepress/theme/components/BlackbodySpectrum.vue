<template>
  <div class="blackbody-container">
    <h4>Interactive Blackbody Spectrum Viewer</h4>
    <p class="component-description">
      Adjust the color temperature to see how the blackbody spectrum changes. Enable standard illuminant overlays for comparison.
    </p>

    <div class="controls-row">
      <div class="slider-group" style="flex: 2;">
        <label>
          Color Temperature: <strong>{{ temperature }} K</strong>
        </label>
        <input type="range" min="2000" max="10000" step="100" v-model.number="temperature" class="ctrl-range" />
      </div>
    </div>

    <div class="controls-row">
      <label class="checkbox-label">
        <input type="checkbox" v-model="showD65" /> CIE D65
      </label>
      <label class="checkbox-label">
        <input type="checkbox" v-model="showA" /> CIE A
      </label>
      <label class="checkbox-label">
        <input type="checkbox" v-model="showF11" /> CIE F11
      </label>
      <label class="checkbox-label">
        <input type="checkbox" v-model="showLED" /> LED White
      </label>
    </div>

    <div class="info-row">
      <div class="info-card" style="border-left: 3px solid var(--vp-c-brand-1);">
        <span class="info-label">CCT:</span>
        <span class="info-value">{{ temperature }} K</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #e74c3c;">
        <span class="info-label">&lambda;<sub>max</sub>:</span>
        <span class="info-value">{{ lambdaMax }} nm</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #27ae60;">
        <span class="info-label">Visible Power:</span>
        <span class="info-value">{{ visiblePower }}%</span>
      </div>
      <div class="info-card color-swatch-card">
        <div class="color-swatch" :style="{ backgroundColor: bbColor }"></div>
        <span class="info-label">Approx. Color</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="bb-svg"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
      >
        <!-- Visible spectrum band on x-axis -->
        <defs>
          <linearGradient id="bbVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
          <clipPath id="bbVisClip">
            <rect :x="xScale(380)" :y="pad.top" :width="xScale(780) - xScale(380)" :height="plotH" />
          </clipPath>
        </defs>

        <!-- Visible region background -->
        <rect
          :x="xScale(380)"
          :y="pad.top"
          :width="xScale(780) - xScale(380)"
          :height="plotH"
          fill="url(#bbVisSpectrum)"
          opacity="0.06"
        />

        <!-- Visible spectrum color bar on x-axis -->
        <rect
          :x="xScale(380)"
          :y="pad.top + plotH + 2"
          :width="xScale(780) - xScale(380)"
          height="8"
          fill="url(#bbVisSpectrum)"
          rx="2"
        />

        <!-- NIR region label -->
        <text
          :x="(xScale(780) + xScale(1000)) / 2"
          :y="pad.top + plotH + 10"
          text-anchor="middle"
          class="nir-label"
        >NIR</text>

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
          :y="pad.top + plotH + 24"
          text-anchor="middle"
          class="axis-label"
        >{{ tick }}</text>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">Wavelength (nm)</text>
        <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`">Relative Spectral Radiance</text>

        <!-- D65 overlay -->
        <path v-if="showD65" :d="d65Path" fill="none" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.85" />
        <!-- A overlay -->
        <path v-if="showA" :d="aPath" fill="none" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.85" />
        <!-- F11 overlay -->
        <path v-if="showF11" :d="f11Path" fill="none" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.85" />
        <!-- LED overlay -->
        <path v-if="showLED" :d="ledPath" fill="none" stroke="#1abc9c" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.85" />

        <!-- Blackbody curve (main) -->
        <path :d="bbPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" opacity="0.95" />
        <!-- Filled area -->
        <path :d="bbArea" fill="var(--vp-c-brand-1)" opacity="0.07" />

        <!-- Wien's law marker -->
        <line
          :x1="xScale(lambdaMaxNum)"
          :y1="pad.top"
          :x2="xScale(lambdaMaxNum)"
          :y2="yScale(0)"
          stroke="#e74c3c"
          stroke-width="1.2"
          stroke-dasharray="5,3"
        />
        <circle
          :cx="xScale(lambdaMaxNum)"
          :cy="yScale(bbPeakVal)"
          r="4"
          fill="#e74c3c"
          stroke="#fff"
          stroke-width="1.5"
        />
        <text
          :x="xScale(lambdaMaxNum) + 6"
          :y="yScale(bbPeakVal) - 6"
          class="peak-label"
        >&lambda;<tspan baseline-shift="sub" font-size="7">max</tspan> = {{ lambdaMax }} nm</text>

        <!-- Hover crosshair -->
        <template v-if="hoverWl !== null">
          <line :x1="xScale(hoverWl)" :y1="pad.top" :x2="xScale(hoverWl)" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverVal)" r="4" fill="var(--vp-c-brand-1)" stroke="#fff" stroke-width="1" />
          <rect :x="tooltipX" :y="pad.top + 4" width="105" height="32" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="tooltipX + 6" :y="pad.top + 18" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX + 6" :y="pad.top + 30" class="tooltip-text">L: {{ hoverVal.toFixed(3) }}</text>
        </template>

        <!-- Legend -->
        <g :transform="`translate(${pad.left + 8}, ${pad.top + 8})`">
          <line x1="0" y1="6" x2="16" y2="6" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
          <text x="20" y="10" class="legend-label">BB {{ temperature }}K</text>
          <template v-if="showD65">
            <line x1="0" y1="20" x2="16" y2="20" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="6,3" />
            <text x="20" y="24" class="legend-label">D65</text>
          </template>
          <template v-if="showA">
            <line x1="0" :y1="showD65 ? 34 : 20" x2="16" :y2="showD65 ? 34 : 20" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" />
            <text x="20" :y="showD65 ? 38 : 24" class="legend-label">CIE A</text>
          </template>
          <template v-if="showF11">
            <line x1="0" :y1="legendY('f11')" x2="16" :y2="legendY('f11')" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="4,2" />
            <text x="20" :y="legendY('f11') + 4" class="legend-label">F11</text>
          </template>
          <template v-if="showLED">
            <line x1="0" :y1="legendY('led')" x2="16" :y2="legendY('led')" stroke="#1abc9c" stroke-width="1.5" stroke-dasharray="4,2" />
            <text x="20" :y="legendY('led') + 4" class="legend-label">LED</text>
          </template>
        </g>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const temperature = ref(5500)
const showD65 = ref(false)
const showA = ref(false)
const showF11 = ref(false)
const showLED = ref(false)

const svgW = 560
const svgH = 320
const pad = { top: 20, right: 20, bottom: 40, left: 55 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 350
const wlMax = 1000
const wlStep = 5
const xTicks = [400, 500, 600, 700, 800, 900, 1000]
const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

function xScale(wl) {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}

function yScale(val) {
  return pad.top + plotH - (val / 1.0) * plotH
}

// Planck's law (relative, normalized to peak = 1)
function planck(wlNm, T) {
  const lam = wlNm * 1e-9
  const h = 6.626e-34
  const c = 2.998e8
  const kB = 1.381e-23
  const num = 2 * h * c * c / Math.pow(lam, 5)
  const denom = Math.exp((h * c) / (lam * kB * T)) - 1
  return num / denom
}

function computeBBSpectrum(T) {
  const pts = []
  let maxVal = 0
  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    const val = planck(wl, T)
    if (val > maxVal) maxVal = val
    pts.push({ wl, val })
  }
  // Normalize to peak = 1
  for (const p of pts) {
    p.val /= maxVal
  }
  return pts
}

// CIE D65 approximate SPD (normalized, key data points)
function cieD65(wlNm) {
  // Simplified D65 SPD from CIE data, normalized to 560nm = 1.0
  const d65Data = [
    [350, 0.46], [360, 0.58], [370, 0.68], [380, 0.50], [390, 0.56],
    [400, 0.83], [410, 0.91], [420, 0.93], [430, 0.87], [440, 1.05],
    [450, 1.17], [460, 1.18], [470, 1.15], [480, 1.14], [490, 1.08],
    [500, 1.09], [510, 1.09], [520, 1.07], [530, 1.07], [540, 1.04],
    [550, 1.04], [560, 1.00], [570, 0.96], [580, 0.96], [590, 0.89],
    [600, 0.90], [610, 0.88], [620, 0.84], [630, 0.83], [640, 0.78],
    [650, 0.74], [660, 0.69], [670, 0.70], [680, 0.64], [690, 0.55],
    [700, 0.57], [710, 0.52], [720, 0.49], [730, 0.46], [740, 0.47],
    [750, 0.44], [760, 0.34], [770, 0.38], [780, 0.35],
  ]
  if (wlNm < 350 || wlNm > 780) return 0
  for (let i = 0; i < d65Data.length - 1; i++) {
    if (wlNm >= d65Data[i][0] && wlNm <= d65Data[i + 1][0]) {
      const t = (wlNm - d65Data[i][0]) / (d65Data[i + 1][0] - d65Data[i][0])
      return d65Data[i][1] * (1 - t) + d65Data[i + 1][1] * t
    }
  }
  return 0
}

// CIE A (Planck at 2856K, normalized at 560nm)
function cieA(wlNm) {
  if (wlNm < 350 || wlNm > 780) return 0
  const val = planck(wlNm, 2856)
  const ref560 = planck(560, 2856)
  return val / ref560
}

// CIE F11 approximate (triband fluorescent)
function cieF11(wlNm) {
  if (wlNm < 380 || wlNm > 780) return 0
  // Three main peaks: 435nm, 545nm, 610nm plus broad background
  const p1 = 0.7 * Math.exp(-0.5 * Math.pow((wlNm - 435) / 8, 2))
  const p2 = 1.0 * Math.exp(-0.5 * Math.pow((wlNm - 545) / 10, 2))
  const p3 = 0.85 * Math.exp(-0.5 * Math.pow((wlNm - 610) / 8, 2))
  const bg = 0.08 * Math.exp(-0.5 * Math.pow((wlNm - 530) / 80, 2))
  return p1 + p2 + p3 + bg
}

// LED white (blue peak + phosphor)
function ledWhite(wlNm) {
  if (wlNm < 380 || wlNm > 780) return 0
  const blue = 0.75 * Math.exp(-0.5 * Math.pow((wlNm - 450) / 12, 2))
  const phosphor = 0.95 * Math.exp(-0.5 * Math.pow((wlNm - 570) / 55, 2))
  return blue + phosphor
}

function computeIllumSpectrum(fn) {
  const pts = []
  let maxVal = 0
  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    const val = fn(wl)
    if (val > maxVal) maxVal = val
    pts.push({ wl, val })
  }
  if (maxVal > 0) {
    for (const p of pts) p.val /= maxVal
  }
  return pts
}

const bbPoints = computed(() => computeBBSpectrum(temperature.value))

const d65Points = computed(() => computeIllumSpectrum(cieD65))
const aPoints = computed(() => computeIllumSpectrum(cieA))
const f11Points = computed(() => computeIllumSpectrum(cieF11))
const ledPoints = computed(() => computeIllumSpectrum(ledWhite))

function pathFromPoints(pts) {
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(p.wl).toFixed(1)},${yScale(p.val).toFixed(1)}`
  }).join(' ')
}

function areaFromPoints(pts) {
  const line = pathFromPoints(pts)
  const last = pts[pts.length - 1]
  const first = pts[0]
  return line + ` L${xScale(last.wl).toFixed(1)},${yScale(0).toFixed(1)} L${xScale(first.wl).toFixed(1)},${yScale(0).toFixed(1)} Z`
}

const bbPath = computed(() => pathFromPoints(bbPoints.value))
const bbArea = computed(() => areaFromPoints(bbPoints.value))

const d65Path = computed(() => pathFromPoints(d65Points.value))
const aPath = computed(() => pathFromPoints(aPoints.value))
const f11Path = computed(() => pathFromPoints(f11Points.value))
const ledPath = computed(() => pathFromPoints(ledPoints.value))

const lambdaMaxNum = computed(() => Math.round(2898000 / temperature.value))
const lambdaMax = computed(() => lambdaMaxNum.value)

const bbPeakVal = computed(() => {
  // The peak value in normalized spectrum is 1.0
  return 1.0
})

// Visible power fraction (380-780 nm vs total)
const visiblePower = computed(() => {
  let visSum = 0
  let totalSum = 0
  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    const val = planck(wl, temperature.value)
    totalSum += val
    if (wl >= 380 && wl <= 780) visSum += val
  }
  // Total extends to IR; our range only goes to 1000nm, so this is approximate
  return (visSum / totalSum * 100).toFixed(1)
})

// Approximate color of blackbody at given temperature
const bbColor = computed(() => {
  const T = temperature.value
  // Approximate blackbody color using CIE color matching
  let r, g, b
  if (T < 3000) {
    r = 255
    g = Math.round(100 + (T - 2000) * 0.06)
    b = Math.round(20 + (T - 2000) * 0.02)
  } else if (T < 5000) {
    r = 255
    g = Math.round(160 + (T - 3000) * 0.04)
    b = Math.round(40 + (T - 3000) * 0.07)
  } else if (T < 6500) {
    r = Math.round(255 - (T - 5000) * 0.02)
    g = Math.round(240 - (T - 5000) * 0.01)
    b = Math.round(180 + (T - 5000) * 0.05)
  } else {
    r = Math.round(225 - (T - 6500) * 0.015)
    g = Math.round(230 - (T - 6500) * 0.008)
    b = Math.round(255)
  }
  r = Math.max(0, Math.min(255, r))
  g = Math.max(0, Math.min(255, g))
  b = Math.max(0, Math.min(255, b))
  return `rgb(${r}, ${g}, ${b})`
})

// Hover
const hoverWl = ref(null)
const hoverVal = ref(0)

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const snapped = Math.round(wl / wlStep) * wlStep
    hoverWl.value = snapped
    const pts = bbPoints.value
    const match = pts.find(p => p.wl === snapped)
    hoverVal.value = match ? match.val : 0
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
  return x + 115 > svgW - pad.right ? x - 115 : x + 10
})

// Visible spectrum color stops for gradient
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
  for (let wl = 380; wl <= 780; wl += 20) {
    stops.push({
      offset: ((wl - 380) / (780 - 380) * 100) + '%',
      color: wavelengthToCSS(wl),
    })
  }
  return stops
})

// Legend y positions
function legendY(item) {
  let y = 20
  if (showD65.value) y += 14
  if (showA.value) y += 14
  if (item === 'f11') return y
  if (showF11.value) y += 14
  if (item === 'led') return y
  return y
}
</script>

<style scoped>
.blackbody-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.blackbody-container h4 {
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
  margin-bottom: 12px;
  align-items: center;
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
.checkbox-label {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.85em;
  cursor: pointer;
  user-select: none;
}
.checkbox-label input[type="checkbox"] {
  accent-color: var(--vp-c-brand-1);
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
.color-swatch-card {
  display: flex;
  align-items: center;
  gap: 8px;
}
.color-swatch {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 2px solid var(--vp-c-divider);
}
.svg-wrapper {
  margin-top: 8px;
}
.bb-svg {
  width: 100%;
  max-width: 560px;
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
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.peak-label {
  font-size: 9px;
  fill: #e74c3c;
  font-weight: 600;
}
.nir-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
  font-style: italic;
}
</style>
