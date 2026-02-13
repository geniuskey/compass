<template>
  <div class="fabry-perot-container">
    <h4>{{ t('Fabry-Perot Thin Film Interference', '파브리-페로 박막 간섭') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize multiple reflections and phasor interference in a single thin film on a substrate.',
        '기판 위 단일 박막에서의 다중 반사와 위상 간섭을 시각화합니다.'
      ) }}
    </p>

    <!-- Controls row -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Film index', '박막 굴절률') }} n<sub>2</sub>: <strong>{{ n2.toFixed(2) }}</strong>
        </label>
        <input
          type="range"
          min="1.0"
          max="4.0"
          step="0.01"
          v-model.number="n2"
          class="ctrl-range"
        />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Substrate index', '기판 굴절률') }} n<sub>3</sub>: <strong>{{ n3.toFixed(2) }}</strong>
        </label>
        <input
          type="range"
          min="1.0"
          max="5.0"
          step="0.01"
          v-model.number="n3"
          class="ctrl-range"
        />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Film thickness', '박막 두께') }}: <strong>{{ thickness }} nm</strong>
        </label>
        <input
          type="range"
          min="10"
          max="300"
          step="1"
          v-model.number="thickness"
          class="ctrl-range"
        />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelength }} nm</strong>
        </label>
        <input
          type="range"
          min="380"
          max="780"
          step="5"
          v-model.number="wavelength"
          class="ctrl-range"
        />
      </div>
    </div>

    <!-- Two-panel display -->
    <div class="panels-row">
      <!-- Left: Wave diagram -->
      <div class="panel">
        <svg :viewBox="`0 0 ${waveW} ${waveH}`" class="wave-svg">
          <defs>
            <marker id="fpRayArrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="var(--vp-c-text-2)" />
            </marker>
            <marker
              v-for="(_, idx) in 6"
              :key="'marker' + idx"
              :id="'fpRayColor' + idx"
              markerWidth="7"
              markerHeight="5"
              refX="7"
              refY="2.5"
              orient="auto"
            >
              <polygon points="0 0, 7 2.5, 0 5" :fill="phasorColors[idx]" />
            </marker>
          </defs>

          <!-- Air region -->
          <rect x="0" y="0" :width="waveW" :height="filmTopY" fill="none" />
          <text x="16" y="20" class="medium-label">n<tspan font-size="7" dy="2">1</tspan><tspan dy="-2">=1.00</tspan></text>

          <!-- Film region -->
          <rect
            x="0"
            :y="filmTopY"
            :width="waveW"
            :height="filmBottomY - filmTopY"
            fill="var(--vp-c-brand-1)"
            opacity="0.12"
          />
          <line
            x1="0"
            :y1="filmTopY"
            :x2="waveW"
            :y2="filmTopY"
            stroke="var(--vp-c-text-3)"
            stroke-width="1.5"
          />
          <line
            x1="0"
            :y1="filmBottomY"
            :x2="waveW"
            :y2="filmBottomY"
            stroke="var(--vp-c-text-3)"
            stroke-width="1.5"
          />
          <text :x="waveW - 8" :y="(filmTopY + filmBottomY) / 2 + 4" text-anchor="end" class="medium-label">
            n<tspan font-size="7" dy="2">2</tspan><tspan dy="-2">={{ n2.toFixed(2) }}</tspan>
          </text>

          <!-- Substrate region -->
          <rect
            x="0"
            :y="filmBottomY"
            :width="waveW"
            :height="waveH - filmBottomY"
            fill="var(--vp-c-text-3)"
            opacity="0.06"
          />
          <text x="16" :y="filmBottomY + 20" class="medium-label">
            n<tspan font-size="7" dy="2">3</tspan><tspan dy="-2">={{ n3.toFixed(2) }}</tspan>
          </text>

          <!-- Incident ray -->
          <line
            :x1="rayStartX - 40"
            :y1="filmTopY - 60"
            :x2="rayStartX"
            :y2="filmTopY"
            stroke="#f39c12"
            stroke-width="2.5"
            marker-end="url(#fpRayArrow)"
          />

          <!-- Bouncing rays inside film -->
          <template v-for="(ray, idx) in bouncingRays" :key="'ray' + idx">
            <!-- Downward leg -->
            <line
              :x1="ray.x1"
              :y1="filmTopY"
              :x2="ray.x2"
              :y2="filmBottomY"
              :stroke="phasorColors[Math.min(idx, 5)]"
              :stroke-width="Math.max(0.5, 2 - idx * 0.3)"
              :opacity="Math.max(0.15, 1 - idx * 0.15)"
            />
            <!-- Upward leg (reflected from bottom) -->
            <line
              :x1="ray.x2"
              :y1="filmBottomY"
              :x2="ray.x3"
              :y2="filmTopY"
              :stroke="phasorColors[Math.min(idx, 5)]"
              :stroke-width="Math.max(0.5, 2 - idx * 0.3)"
              :opacity="Math.max(0.15, 1 - idx * 0.15)"
            />
            <!-- Exiting ray upward -->
            <line
              :x1="ray.x3"
              :y1="filmTopY"
              :x2="ray.x3 + 15"
              :y2="filmTopY - 35"
              :stroke="phasorColors[Math.min(idx, 5)]"
              :stroke-width="Math.max(0.5, 1.5 - idx * 0.2)"
              :opacity="Math.max(0.1, 0.8 - idx * 0.12)"
              :marker-end="`url(#fpRayColor${Math.min(idx, 5)})`"
            />
          </template>

          <!-- First reflected ray (r12, no entry into film) -->
          <line
            :x1="rayStartX"
            :y1="filmTopY"
            :x2="rayStartX + 15"
            :y2="filmTopY - 35"
            stroke="#e74c3c"
            stroke-width="2"
            opacity="0.8"
            marker-end="url(#fpRayColor0)"
          />

          <!-- Phase label -->
          <text
            :x="waveW / 2"
            :y="(filmTopY + filmBottomY) / 2 + 4"
            text-anchor="middle"
            class="phase-label"
          >delta = {{ phaseDelta.toFixed(2) }} rad</text>
        </svg>
      </div>

      <!-- Right: Phasor diagram -->
      <div class="panel">
        <svg :viewBox="`0 0 ${phasorW} ${phasorH}`" class="phasor-svg">
          <!-- Unit circle (faint) -->
          <circle
            :cx="phasorCx"
            :cy="phasorCy"
            :r="phasorScale"
            fill="none"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            stroke-dasharray="4,4"
          />
          <!-- Half unit circle -->
          <circle
            :cx="phasorCx"
            :cy="phasorCy"
            :r="phasorScale * 0.5"
            fill="none"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            stroke-dasharray="3,3"
          />

          <!-- Axes (faint) -->
          <line
            :x1="phasorCx - phasorScale - 10"
            :y1="phasorCy"
            :x2="phasorCx + phasorScale + 10"
            :y2="phasorCy"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
          />
          <line
            :x1="phasorCx"
            :y1="phasorCy - phasorScale - 10"
            :x2="phasorCx"
            :y2="phasorCy + phasorScale + 10"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
          />

          <!-- Axis labels -->
          <text :x="phasorCx + phasorScale + 6" :y="phasorCy + 3" class="phasor-axis-label">Re</text>
          <text :x="phasorCx + 4" :y="phasorCy - phasorScale - 4" class="phasor-axis-label">Im</text>

          <!-- Phasor vectors (partial reflections, head-to-tail) -->
          <template v-for="(vec, idx) in phasorVectors" :key="'pv' + idx">
            <line
              :x1="phasorCx + vec.fromRe * phasorScale"
              :y1="phasorCy - vec.fromIm * phasorScale"
              :x2="phasorCx + vec.toRe * phasorScale"
              :y2="phasorCy - vec.toIm * phasorScale"
              :stroke="phasorColors[idx]"
              :stroke-width="idx === 0 ? 2.5 : Math.max(1, 2.5 - idx * 0.3)"
            />
            <!-- Small circle at tip -->
            <circle
              :cx="phasorCx + vec.toRe * phasorScale"
              :cy="phasorCy - vec.toIm * phasorScale"
              r="2.5"
              :fill="phasorColors[idx]"
            />
          </template>

          <!-- Total r vector (thick, from origin to final point) -->
          <line
            :x1="phasorCx"
            :y1="phasorCy"
            :x2="phasorCx + totalR_re * phasorScale"
            :y2="phasorCy - totalR_im * phasorScale"
            stroke="var(--vp-c-text-1)"
            stroke-width="3"
            opacity="0.85"
          />
          <circle
            :cx="phasorCx + totalR_re * phasorScale"
            :cy="phasorCy - totalR_im * phasorScale"
            r="4"
            fill="var(--vp-c-text-1)"
          />

          <!-- R label -->
          <text
            :x="phasorCx"
            :y="phasorCy + phasorScale + 24"
            text-anchor="middle"
            class="r-total-label"
          >|r|<tspan font-size="7" dy="-4">2</tspan><tspan dy="4"> = R = {{ (reflectanceExact * 100).toFixed(1) }}%</tspan></text>

          <!-- Legend labels for phasor vectors -->
          <template v-for="(vec, idx) in phasorVectors.slice(0, 4)" :key="'pvl' + idx">
            <line
              :x1="phasorCx + phasorScale + 16"
              :y1="phasorCy - phasorScale + 12 + idx * 14"
              :x2="phasorCx + phasorScale + 28"
              :y2="phasorCy - phasorScale + 12 + idx * 14"
              :stroke="phasorColors[idx]"
              stroke-width="2"
            />
            <text
              :x="phasorCx + phasorScale + 32"
              :y="phasorCy - phasorScale + 15 + idx * 14"
              class="phasor-legend"
            >r<tspan font-size="6" dy="2">{{ idx }}</tspan></text>
          </template>
        </svg>
      </div>
    </div>

    <!-- Bottom panel: R vs thickness chart -->
    <div class="r-chart-section">
      <svg :viewBox="`0 0 ${rChartW} ${rChartH}`" class="r-chart-svg">
        <!-- Grid lines at 0, 25, 50, 75, 100 -->
        <template v-for="pct in [0, 25, 50, 75, 100]" :key="'rgy' + pct">
          <line
            :x1="rPadL"
            :y1="rPctToY(pct)"
            :x2="rPadL + rPlotW"
            :y2="rPctToY(pct)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            stroke-dasharray="3,3"
          />
          <text
            :x="rPadL - 6"
            :y="rPctToY(pct) + 3"
            text-anchor="end"
            class="tick-label"
          >{{ pct }}%</text>
        </template>

        <!-- Axes -->
        <line
          :x1="rPadL"
          :y1="rPadT"
          :x2="rPadL"
          :y2="rPadT + rPlotH"
          stroke="var(--vp-c-text-3)"
          stroke-width="1"
        />
        <line
          :x1="rPadL"
          :y1="rPadT + rPlotH"
          :x2="rPadL + rPlotW"
          :y2="rPadT + rPlotH"
          stroke="var(--vp-c-text-3)"
          stroke-width="1"
        />

        <!-- Y-axis label -->
        <text
          x="10"
          :y="rPadT + rPlotH / 2"
          text-anchor="middle"
          :transform="`rotate(-90, 10, ${rPadT + rPlotH / 2})`"
          class="axis-label"
        >R, T (%)</text>

        <!-- X-axis ticks -->
        <template v-for="d in [0, 50, 100, 150, 200, 250, 300]" :key="'rxt' + d">
          <line
            :x1="thickToX(d)"
            :y1="rPadT + rPlotH"
            :x2="thickToX(d)"
            :y2="rPadT + rPlotH + 4"
            stroke="var(--vp-c-text-3)"
            stroke-width="1"
          />
          <text
            :x="thickToX(d)"
            :y="rPadT + rPlotH + 16"
            text-anchor="middle"
            class="tick-label"
          >{{ d }}</text>
        </template>
        <text
          :x="rPadL + rPlotW / 2"
          :y="rChartH - 2"
          text-anchor="middle"
          class="axis-label"
        >{{ t('Thickness (nm)', '두께 (nm)') }}</text>

        <!-- R curve -->
        <path :d="rVsThickPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <!-- T curve -->
        <path :d="tVsThickPath" fill="none" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="5,3" />

        <!-- Current thickness marker -->
        <line
          :x1="thickToX(thickness)"
          :y1="rPadT"
          :x2="thickToX(thickness)"
          :y2="rPadT + rPlotH"
          stroke="var(--vp-c-text-2)"
          stroke-width="1.2"
          stroke-dasharray="4,3"
          opacity="0.6"
        />
        <circle
          :cx="thickToX(thickness)"
          :cy="rPctToY(reflectanceExact * 100)"
          r="4"
          fill="var(--vp-c-brand-1)"
          stroke="#fff"
          stroke-width="1"
        />
        <circle
          :cx="thickToX(thickness)"
          :cy="rPctToY(transmittanceExact * 100)"
          r="3.5"
          fill="#2ecc71"
          stroke="#fff"
          stroke-width="1"
        />

        <!-- Legend -->
        <line :x1="rPadL + rPlotW - 80" :y1="rPadT + 10" :x2="rPadL + rPlotW - 60" :y2="rPadT + 10" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <text :x="rPadL + rPlotW - 55" :y="rPadT + 13" class="legend-label">R</text>
        <line :x1="rPadL + rPlotW - 80" :y1="rPadT + 24" :x2="rPadL + rPlotW - 60" :y2="rPadT + 24" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="5,3" />
        <text :x="rPadL + rPlotW - 55" :y="rPadT + 27" class="legend-label">T</text>
      </svg>
    </div>

    <!-- Info cards -->
    <div class="info-cards">
      <div class="info-card">
        <div class="info-label">R</div>
        <div class="info-value highlight">{{ (reflectanceExact * 100).toFixed(1) }}%</div>
      </div>
      <div class="info-card">
        <div class="info-label">T</div>
        <div class="info-value">{{ (transmittanceExact * 100).toFixed(1) }}%</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Phase', '위상') }} delta</div>
        <div class="info-value">{{ phaseDelta.toFixed(3) }} rad</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Quarter-wave d', '쿼터파 두께') }}</div>
        <div class="info-value">{{ quarterWaveThickness.toFixed(1) }} nm</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// Controls
const n2 = ref(2.0)
const n3 = ref(3.5)
const thickness = ref(100)
const wavelength = ref(550)

// Constants
const n1 = 1.0 // air

// Phasor vector colors (blue to red gradient)
const phasorColors = ['#3498db', '#2980b9', '#8e44ad', '#c0392b', '#e74c3c', '#d35400']

// Wave diagram SVG dimensions
const waveW = 320
const waveH = 300
const filmTopY = 80
const filmBottomY = 200
const rayStartX = 80

// Phasor diagram SVG dimensions
const phasorW = 280
const phasorH = 280
const phasorCx = 130
const phasorCy = 125
const phasorScale = 95

// R vs thickness chart dimensions
const rChartW = 600
const rChartH = 200
const rPadL = 42
const rPadR = 16
const rPadT = 12
const rPadB = 32
const rPlotW = rChartW - rPadL - rPadR
const rPlotH = rChartH - rPadT - rPadB

// Physics: Fresnel coefficients (normal incidence, real indices)
const r12 = computed(() => (n1 - n2.value) / (n1 + n2.value))
const r23 = computed(() => (n2.value - n3.value) / (n2.value + n3.value))

// Phase per round trip
const phaseDelta = computed(() => {
  return (4 * Math.PI * n2.value * thickness.value) / wavelength.value
})

// Exact reflectance from Airy formula
// r = (r12 + r23 * exp(i*delta)) / (1 + r12*r23*exp(i*delta))
function computeExactR(r12v: number, r23v: number, delta: number): { R: number; T: number; rRe: number; rIm: number } {
  const cosDelta = Math.cos(delta)
  const sinDelta = Math.sin(delta)

  // Numerator: r12 + r23 * exp(i*delta) = (r12 + r23*cos(delta)) + i*(r23*sin(delta))
  const numRe = r12v + r23v * cosDelta
  const numIm = r23v * sinDelta

  // Denominator: 1 + r12*r23*exp(i*delta) = (1 + r12*r23*cos(delta)) + i*(r12*r23*sin(delta))
  const denRe = 1 + r12v * r23v * cosDelta
  const denIm = r12v * r23v * sinDelta

  // Complex division
  const denMag2 = denRe * denRe + denIm * denIm
  const rRe = (numRe * denRe + numIm * denIm) / denMag2
  const rIm = (numIm * denRe - numRe * denIm) / denMag2

  const R = rRe * rRe + rIm * rIm
  // For lossless dielectric: T = 1 - R
  const T = 1 - R

  return { R, T, rRe, rIm }
}

const reflectanceExact = computed(() => {
  return computeExactR(r12.value, r23.value, phaseDelta.value).R
})

const transmittanceExact = computed(() => {
  return computeExactR(r12.value, r23.value, phaseDelta.value).T
})

const totalR_re = computed(() => {
  return computeExactR(r12.value, r23.value, phaseDelta.value).rRe
})

const totalR_im = computed(() => {
  return computeExactR(r12.value, r23.value, phaseDelta.value).rIm
})

// Quarter-wave thickness for zero R: d = lambda / (4 * n2)
const quarterWaveThickness = computed(() => {
  return wavelength.value / (4 * n2.value)
})

// Partial reflection phasors for phasor diagram (6 terms)
// r_0 = r12
// r_m = (1 - r12^2) * r23 * (-r12 * r23)^(m-1) * exp(i * m * delta) for m >= 1
const phasorVectors = computed(() => {
  const vecs: { fromRe: number; fromIm: number; toRe: number; toIm: number }[] = []
  const r12v = r12.value
  const r23v = r23.value
  const delta = phaseDelta.value
  const t12t21 = 1 - r12v * r12v // t12 * t21

  let sumRe = 0
  let sumIm = 0

  // m = 0: r_0 = r12 (real)
  vecs.push({ fromRe: 0, fromIm: 0, toRe: r12v, toIm: 0 })
  sumRe = r12v
  sumIm = 0

  // m >= 1
  // r_m = t12t21 * r23 * (-r12 * r23)^(m-1) * exp(i * m * delta)
  const factor = -r12v * r23v
  let powerRe = 1 // (-r12*r23)^0 = 1
  let powerIm = 0

  for (let m = 1; m <= 5; m++) {
    // Multiply power by factor (real): power *= factor
    // Since factor is real: (a + bi) * c = (ac) + (bc)i
    if (m > 1) {
      const newPowerRe = powerRe * factor
      const newPowerIm = powerIm * factor
      powerRe = newPowerRe
      powerIm = newPowerIm
    }

    // exp(i * m * delta)
    const expRe = Math.cos(m * delta)
    const expIm = Math.sin(m * delta)

    // r_m = t12t21 * r23 * power * exp(i*m*delta)
    const coeff = t12t21 * r23v
    // (powerRe + i*powerIm) * (expRe + i*expIm)
    const prodRe = powerRe * expRe - powerIm * expIm
    const prodIm = powerRe * expIm + powerIm * expRe

    const rmRe = coeff * prodRe
    const rmIm = coeff * prodIm

    const fromRe = sumRe
    const fromIm = sumIm
    sumRe += rmRe
    sumIm += rmIm

    vecs.push({ fromRe, fromIm, toRe: sumRe, toIm: sumIm })
  }

  return vecs
})

// Bouncing ray positions for wave diagram
const bouncingRays = computed(() => {
  const rays: { x1: number; x2: number; x3: number }[] = []
  const spacing = 30
  let x = rayStartX + 20
  for (let i = 0; i < 5; i++) {
    rays.push({
      x1: x,
      x2: x + 12,
      x3: x + 24,
    })
    x += spacing
  }
  return rays
})

// R vs thickness chart helpers
function thickToX(d: number): number {
  return rPadL + (d / 300) * rPlotW
}

function rPctToY(pct: number): number {
  return rPadT + rPlotH - (pct / 100) * rPlotH
}

function computeRAtThickness(d: number): number {
  const delta = (4 * Math.PI * n2.value * d) / wavelength.value
  return computeExactR(r12.value, r23.value, delta).R
}

const rVsThickPath = computed(() => {
  let path = ''
  for (let d = 0; d <= 300; d += 2) {
    const R = computeRAtThickness(d) * 100
    const x = thickToX(d)
    const y = rPctToY(R)
    path += path === '' ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return path
})

const tVsThickPath = computed(() => {
  let path = ''
  for (let d = 0; d <= 300; d += 2) {
    const T = (1 - computeRAtThickness(d)) * 100
    const x = thickToX(d)
    const y = rPctToY(T)
    path += path === '' ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return path
})
</script>

<style scoped>
.fabry-perot-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.fabry-perot-container h4 {
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
  min-width: 160px;
}
.slider-group label {
  display: block;
  font-size: 0.88em;
  margin-bottom: 4px;
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
.panels-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.panel {
  flex: 1;
  min-width: 260px;
}
.wave-svg {
  width: 100%;
  max-width: 320px;
  display: block;
  margin: 0 auto;
}
.phasor-svg {
  width: 100%;
  max-width: 280px;
  display: block;
  margin: 0 auto;
}
.medium-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.phase-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.phasor-axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.phasor-legend {
  font-size: 8px;
  fill: var(--vp-c-text-2);
}
.r-total-label {
  font-size: 10px;
  fill: var(--vp-c-text-1);
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.r-chart-section {
  margin-bottom: 16px;
}
.r-chart-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.tick-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.info-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}
.info-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.info-value {
  font-weight: 600;
  font-size: 1em;
  font-family: var(--vp-font-family-mono);
}
.info-value.highlight {
  color: var(--vp-c-brand-1);
}
</style>
