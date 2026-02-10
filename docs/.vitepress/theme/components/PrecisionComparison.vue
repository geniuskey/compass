<template>
  <div class="precision-container">
    <h4>Numerical Precision Comparison: Float32 vs Float64</h4>
    <p class="component-description">
      See how floating-point precision affects phase computation accuracy in wave optics. As phase accumulates over many cycles, float32 errors grow while float64 stays accurate.
    </p>

    <div class="slider-section">
      <label>
        Phase accumulation: <strong>{{ phaseMultiplier.toFixed(0) }} &times; 2&pi;</strong>
        (total phase = {{ totalPhase.toFixed(1) }} rad)
      </label>
      <input type="range" min="0" max="100" step="1" v-model.number="phaseMultiplier" class="ctrl-range" />
    </div>

    <div class="panels-row">
      <!-- Float32 panel -->
      <div class="panel panel-32">
        <div class="panel-header" :class="{ 'error-bad': f32ErrorMag > qeTolerance }">
          Float32
        </div>
        <div class="panel-body">
          <div class="val-row">
            <span class="val-label">cos(phase):</span>
            <span class="val-mono">{{ f32Cos }}</span>
          </div>
          <div class="val-row">
            <span class="val-label">sin(phase):</span>
            <span class="val-mono">{{ f32Sin }}</span>
          </div>
          <div class="val-row">
            <span class="val-label">Error magnitude:</span>
            <span class="val-mono" :style="{ color: f32ErrorColor }">{{ f32ErrorStr }}</span>
          </div>
          <div class="precision-note">
            ~7 significant digits<br />
            Machine epsilon: 1.2e-7
          </div>
        </div>
      </div>

      <!-- Float64 panel -->
      <div class="panel panel-64">
        <div class="panel-header header-good">
          Float64
        </div>
        <div class="panel-body">
          <div class="val-row">
            <span class="val-label">cos(phase):</span>
            <span class="val-mono">{{ f64Cos }}</span>
          </div>
          <div class="val-row">
            <span class="val-label">sin(phase):</span>
            <span class="val-mono">{{ f64Sin }}</span>
          </div>
          <div class="val-row">
            <span class="val-label">Error magnitude:</span>
            <span class="val-mono" style="color: #27ae60;">{{ f64ErrorStr }}</span>
          </div>
          <div class="precision-note">
            ~16 significant digits<br />
            Machine epsilon: 2.2e-16
          </div>
        </div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="precision-svg">
        <!-- Grid -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <!-- Y-axis labels (log scale) -->
        <template v-for="tick in yLogTicks" :key="'yl' + tick.exp">
          <line
            :x1="pad.left"
            :y1="tick.y"
            :x2="pad.left + plotW"
            :y2="tick.y"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            stroke-dasharray="3,3"
          />
          <text :x="pad.left - 6" :y="tick.y + 3" text-anchor="end" class="axis-label">{{ tick.label }}</text>
        </template>

        <!-- X-axis labels -->
        <template v-for="tick in xBarTicks" :key="'xb' + tick.val">
          <text :x="tick.x" :y="pad.top + plotH + 14" text-anchor="middle" class="axis-label">{{ tick.val }}</text>
        </template>

        <!-- QE tolerance threshold line -->
        <line
          :x1="pad.left"
          :y1="yLogScale(qeTolerance)"
          :x2="pad.left + plotW"
          :y2="yLogScale(qeTolerance)"
          stroke="#e74c3c"
          stroke-width="1"
          stroke-dasharray="6,3"
        />
        <text :x="pad.left + plotW + 4" :y="yLogScale(qeTolerance) + 3" class="threshold-label">0.5% QE tolerance</text>

        <!-- Float32 error bars -->
        <template v-for="(pt, idx) in errorData" :key="'f32-' + idx">
          <rect
            :x="pt.x - barW / 2 - barW * 0.3"
            :y="Math.min(yLogScale(pt.f32), pad.top + plotH)"
            :width="barW * 0.55"
            :height="Math.max(0, pad.top + plotH - yLogScale(pt.f32))"
            :fill="pt.f32 > qeTolerance ? '#e74c3c' : '#f39c12'"
            :opacity="0.8"
            rx="1"
          />
        </template>

        <!-- Float64 error bars -->
        <template v-for="(pt, idx) in errorData" :key="'f64-' + idx">
          <rect
            :x="pt.x - barW / 2 + barW * 0.3"
            :y="Math.min(yLogScale(pt.f64), pad.top + plotH)"
            :width="barW * 0.55"
            :height="Math.max(0, pad.top + plotH - yLogScale(pt.f64))"
            fill="#27ae60"
            opacity="0.8"
            rx="1"
          />
        </template>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">Phase (multiples of 2pi)</text>
        <text x="10" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`">Absolute error</text>

        <!-- Legend -->
        <rect :x="pad.left + 10" :y="pad.top + 4" width="10" height="10" fill="#f39c12" rx="1" />
        <text :x="pad.left + 24" :y="pad.top + 13" class="legend-label">Float32</text>
        <rect :x="pad.left + 74" :y="pad.top + 4" width="10" height="10" fill="#27ae60" rx="1" />
        <text :x="pad.left + 88" :y="pad.top + 13" class="legend-label">Float64</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const phaseMultiplier = ref(20)
const qeTolerance = 0.005

const totalPhase = computed(() => phaseMultiplier.value * 2 * Math.PI)

// Simulate float32 errors
// Float32 has ~7.2 significant decimal digits, epsilon ~ 1.19e-7
// When computing cos(large_phase), the error grows because we lose significant digits
// in the phase value itself before the trig function
function simulateFloat32Error(nCycles) {
  const phase = nCycles * 2 * Math.PI
  // Float32 relative precision ~ 1.19e-7
  // Error in phase ~ phase * epsilon32
  // Error in cos/sin ~ |phase * epsilon32| (first-order)
  const epsilon32 = 1.19e-7
  return Math.abs(phase) * epsilon32
}

function simulateFloat64Error(nCycles) {
  const phase = nCycles * 2 * Math.PI
  const epsilon64 = 2.22e-16
  return Math.abs(phase) * epsilon64
}

// Current slider values
const f32Error = computed(() => simulateFloat32Error(phaseMultiplier.value))
const f64Error = computed(() => simulateFloat64Error(phaseMultiplier.value))

const f32ErrorMag = computed(() => f32Error.value)

// Cos/sin display: show the "analytical" value (which is just cos/sin of the exact phase)
// and the f32 "computed" value with noise
const analyticalCos = computed(() => Math.cos(totalPhase.value))
const analyticalSin = computed(() => Math.sin(totalPhase.value))

const f32Cos = computed(() => {
  const err = f32Error.value
  return (analyticalCos.value + err * 0.7).toFixed(7)
})
const f32Sin = computed(() => {
  const err = f32Error.value
  return (analyticalSin.value + err * 0.7).toFixed(7)
})
const f64Cos = computed(() => analyticalCos.value.toFixed(15))
const f64Sin = computed(() => analyticalSin.value.toFixed(15))

const f32ErrorStr = computed(() => f32Error.value.toExponential(2))
const f64ErrorStr = computed(() => f64Error.value.toExponential(2))

const f32ErrorColor = computed(() => {
  if (f32Error.value > qeTolerance) return '#e74c3c'
  if (f32Error.value > qeTolerance * 0.1) return '#f39c12'
  return '#27ae60'
})

// Bar chart data: error at different phase multipliers
const svgW = 500
const svgH = 240
const pad = { top: 24, right: 80, bottom: 30, left: 55 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom
const barW = 20

const samplePoints = [1, 5, 10, 20, 40, 60, 80, 100]

const errorData = computed(() => {
  return samplePoints.map((n, idx) => ({
    n,
    x: pad.left + (idx + 0.5) / samplePoints.length * plotW,
    f32: simulateFloat32Error(n),
    f64: simulateFloat64Error(n),
  }))
})

const xBarTicks = computed(() => {
  return samplePoints.map((n, idx) => ({
    val: n,
    x: pad.left + (idx + 0.5) / samplePoints.length * plotW,
  }))
})

// Log scale for y-axis
const yLogMin = -16
const yLogMax = -1

function yLogScale(val) {
  const logVal = val > 0 ? Math.log10(val) : yLogMin
  const clamped = Math.max(yLogMin, Math.min(yLogMax, logVal))
  const frac = (clamped - yLogMin) / (yLogMax - yLogMin)
  return pad.top + plotH - frac * plotH
}

const yLogTicks = computed(() => {
  const ticks = []
  for (let exp = yLogMin; exp <= yLogMax; exp += 3) {
    ticks.push({
      exp,
      y: yLogScale(Math.pow(10, exp)),
      label: `1e${exp}`,
    })
  }
  return ticks
})
</script>

<style scoped>
.precision-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.precision-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.slider-section {
  margin-bottom: 16px;
}
.slider-section label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.9em;
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
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.panels-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.panel {
  flex: 1;
  min-width: 200px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.panel-header {
  padding: 8px 12px;
  font-weight: 700;
  font-size: 0.95em;
  text-align: center;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}
.panel-header.error-bad {
  background: #f8d7da;
  color: #842029;
}
.dark .panel-header.error-bad {
  background: #4a1c21;
  color: #f5c6cb;
}
.panel-header.header-good {
  background: #d1e7dd;
  color: #0f5132;
}
.dark .panel-header.header-good {
  background: #1a3a2a;
  color: #a3cfbb;
}
.panel-body {
  padding: 10px 12px;
  background: var(--vp-c-bg-soft);
}
.val-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 0.82em;
}
.val-label {
  color: var(--vp-c-text-2);
}
.val-mono {
  font-family: var(--vp-font-family-mono);
  font-weight: 600;
  font-size: 0.9em;
}
.precision-note {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 0.75em;
  color: var(--vp-c-text-2);
  line-height: 1.4;
}
.svg-wrapper {
  margin-top: 8px;
}
.precision-svg {
  width: 100%;
  max-width: 500px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 8px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
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
.threshold-label {
  font-size: 8px;
  fill: #e74c3c;
  font-weight: 600;
}
</style>
