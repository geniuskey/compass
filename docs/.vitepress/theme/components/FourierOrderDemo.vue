<template>
  <div class="fourier-container">
    <h4>Fourier Order Approximation Demo</h4>
    <p class="component-description">
      See how increasing the number of Fourier harmonics improves the approximation of a
      square wave (representing a DTI trench or metal grid cross-section). Notice the
      Gibbs phenomenon ringing at the edges.
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label for="order-input">
          Fourier Order N: <strong>{{ order }}</strong>
        </label>
        <input
          id="order-input"
          type="range"
          min="1"
          max="25"
          step="1"
          v-model.number="order"
          class="order-range"
        />
      </div>
      <div class="slider-group">
        <label for="duty-input">
          Duty Cycle: <strong>{{ (dutyCycle * 100).toFixed(0) }}%</strong>
        </label>
        <input
          id="duty-input"
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          v-model.number="dutyCycle"
          class="order-range"
        />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">Harmonics used:</span>
        <span class="info-value">{{ 2 * order + 1 }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">RCWA matrix size:</span>
        <span class="info-value">{{ (2 * order + 1) }}&times;{{ (2 * order + 1) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">Max overshoot:</span>
        <span class="info-value">{{ (overshoot * 100).toFixed(1) }}%</span>
      </div>
    </div>

    <div class="chart-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="fourier-svg">
        <!-- Grid lines -->
        <line :x1="padL" :y1="midY" :x2="svgW - padR" :y2="midY" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="padL" :y1="topY" :x2="svgW - padR" :y2="topY" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="padL" :y1="botY" :x2="svgW - padR" :y2="botY" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

        <!-- Axes -->
        <line :x1="padL" :y1="topY - 10" :x2="padL" :y2="botY + 10" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line :x1="padL" :y1="botY + 10" :x2="svgW - padR" :y2="botY + 10" stroke="var(--vp-c-text-3)" stroke-width="1" />

        <!-- Y-axis labels -->
        <text :x="padL - 5" :y="topY + 4" text-anchor="end" class="tick-label">1</text>
        <text :x="padL - 5" :y="midY + 4" text-anchor="end" class="tick-label">0</text>
        <text :x="padL - 5" :y="botY + 4" text-anchor="end" class="tick-label">-1</text>

        <!-- X-axis labels -->
        <text :x="padL" :y="botY + 24" text-anchor="middle" class="tick-label">0</text>
        <text :x="padL + plotW / 2" :y="botY + 24" text-anchor="middle" class="tick-label">&Lambda;/2</text>
        <text :x="padL + plotW" :y="botY + 24" text-anchor="middle" class="tick-label">&Lambda;</text>
        <text :x="padL + plotW / 2" :y="botY + 38" text-anchor="middle" class="axis-label">Position within unit cell</text>

        <!-- Original square wave (two periods shown) -->
        <path :d="squarePath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.7" />

        <!-- Fourier approximation -->
        <path :d="fourierPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />

        <!-- Gibbs overshoot region highlight -->
        <template v-for="(region, idx) in gibbsRegions" :key="idx">
          <rect
            :x="region.x"
            :y="topY - 10"
            :width="region.w"
            :height="plotH + 20"
            fill="#e74c3c"
            opacity="0.06"
            rx="3"
          />
        </template>

        <!-- Legend -->
        <line :x1="svgW - padR - 160" :y1="topY + 2" :x2="svgW - padR - 140" :y2="topY + 2" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="6,4" />
        <text :x="svgW - padR - 135" :y="topY + 6" class="legend-label">Original</text>
        <line :x1="svgW - padR - 160" :y1="topY + 18" :x2="svgW - padR - 140" :y2="topY + 18" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        <text :x="svgW - padR - 135" :y="topY + 22" class="legend-label">Fourier (N={{ order }})</text>
        <rect :x="svgW - padR - 160" :y="topY + 28" width="20" height="10" fill="#e74c3c" opacity="0.15" rx="2" />
        <text :x="svgW - padR - 135" :y="topY + 37" class="legend-label">Gibbs region</text>
      </svg>
    </div>

    <div class="explanation">
      <p>
        <strong>Gibbs phenomenon:</strong> Even with many harmonics, the Fourier series
        overshoots by ~9% at discontinuities. In RCWA, this affects convergence at
        sharp material boundaries (e.g., Si/SiO<sub>2</sub> DTI walls).
        Li's factorization rules mitigate this for TM polarization.
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const order = ref(5)
const dutyCycle = ref(0.5)

const svgW = 520
const svgH = 300
const padL = 45
const padR = 20
const padT = 30
const padB = 50
const plotW = svgW - padL - padR
const plotH = svgH - padT - padB
const topY = padT
const botY = padT + plotH
const midY = padT + plotH / 2

function squareWave(x, dc) {
  // x in [0, 1] representing one period
  const xMod = ((x % 1) + 1) % 1
  return xMod < dc ? 1 : -1
}

function fourierSquare(x, N, dc) {
  // Fourier series of a square wave with duty cycle dc
  // f(x) = a0 + sum_{n=1}^{N} [a_n cos(2*pi*n*x) + b_n sin(2*pi*n*x)]
  // For a square wave from 0 to dc: value=1, dc to 1: value=-1
  // a0 = 2*dc - 1
  // a_n = (2/(n*pi)) * sin(2*pi*n*dc)
  // b_n = (2/(n*pi)) * (1 - cos(2*pi*n*dc))
  let sum = 2 * dc - 1 // a0
  for (let n = 1; n <= N; n++) {
    const arg = 2 * Math.PI * n
    const an = (2 / (n * Math.PI)) * Math.sin(arg * dc)
    const bn = (2 / (n * Math.PI)) * (1 - Math.cos(arg * dc))
    sum += an * Math.cos(arg * x) + bn * Math.sin(arg * x)
  }
  return sum
}

const numPoints = 500

function valToY(val) {
  // val in [-1.2, 1.2] -> y coordinates
  const clamped = Math.max(-1.3, Math.min(1.3, val))
  return midY - (clamped / 1.3) * (plotH / 2)
}

const squarePath = computed(() => {
  let d = ''
  const dc = dutyCycle.value
  for (let i = 0; i <= numPoints; i++) {
    const x = i / numPoints
    const val = squareWave(x, dc)
    const px = padL + x * plotW
    const py = valToY(val)
    d += i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`
  }
  return d
})

const fourierPath = computed(() => {
  let d = ''
  const dc = dutyCycle.value
  for (let i = 0; i <= numPoints; i++) {
    const x = i / numPoints
    const val = fourierSquare(x, order.value, dc)
    const px = padL + x * plotW
    const py = valToY(val)
    d += i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`
  }
  return d
})

const overshoot = computed(() => {
  const dc = dutyCycle.value
  let maxVal = 0
  for (let i = 0; i <= numPoints; i++) {
    const x = i / numPoints
    const val = Math.abs(fourierSquare(x, order.value, dc))
    if (val > maxVal) maxVal = val
  }
  return Math.max(0, maxVal - 1)
})

const gibbsRegions = computed(() => {
  const dc = dutyCycle.value
  const regionWidth = 0.04  // fraction of period to highlight
  const regions = []
  // Transitions at x=0 (wrapped), x=dc
  const transitions = [0, dc]
  for (const t of transitions) {
    const x1 = Math.max(0, t - regionWidth)
    const x2 = Math.min(1, t + regionWidth)
    regions.push({
      x: padL + x1 * plotW,
      w: (x2 - x1) * plotW,
    })
  }
  return regions
})
</script>

<style scoped>
.fourier-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.fourier-container h4 {
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
  gap: 24px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 200px;
}
.slider-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.95em;
}
.order-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.order-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.order-range::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.info-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 0.9em;
}
.info-label {
  color: var(--vp-c-text-2);
  margin-right: 6px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.chart-wrapper {
  margin-bottom: 16px;
}
.fourier-svg {
  width: 100%;
  max-width: 580px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.axis-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.explanation {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 0.88em;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}
.explanation p {
  margin: 0;
}
</style>
