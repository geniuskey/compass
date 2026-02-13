<template>
  <div class="si-absorption-container">
    <h4>{{ t('Silicon Absorption Depth Visualizer', '실리콘 흡수 깊이 시각화') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how light penetration depth in silicon depends on wavelength using Beer-Lambert law.',
        'Beer-Lambert 법칙을 사용하여 실리콘에서의 빛 침투 깊이가 파장에 따라 어떻게 달라지는지 살펴보세요.'
      ) }}
    </p>

    <!-- Controls row -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelength }} nm</strong>
        </label>
        <input
          type="range"
          min="380"
          max="1000"
          step="5"
          v-model.number="wavelength"
          class="ctrl-range"
        />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Si Thickness', 'Si 두께') }}: <strong>{{ siThickness.toFixed(1) }} um</strong>
        </label>
        <input
          type="range"
          min="1"
          max="5"
          step="0.1"
          v-model.number="siThickness"
          class="ctrl-range"
        />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Photodiode Depth', '포토다이오드 깊이') }}: <strong>{{ pdDepth.toFixed(1) }} um</strong>
        </label>
        <input
          type="range"
          min="0.5"
          max="3.0"
          step="0.1"
          v-model.number="pdDepth"
          class="ctrl-range"
        />
      </div>
    </div>

    <!-- Two-panel display -->
    <div class="panels-row">
      <!-- Left panel: Cross-section diagram -->
      <div class="panel">
        <svg :viewBox="`0 0 ${crossW} ${crossH}`" class="cross-section-svg">
          <defs>
            <linearGradient :id="gradientId" x1="0" y1="0" x2="0" y2="1">
              <stop
                v-for="(stop, idx) in gradientStops"
                :key="idx"
                :offset="stop.offset"
                :stop-color="stop.color"
                :stop-opacity="stop.opacity"
              />
            </linearGradient>
            <marker id="siAbsLightArrow" markerWidth="8" markerHeight="6" refX="0" refY="3" orient="auto">
              <polygon :points="'0 0, 8 3, 0 6'" :fill="wlColor" />
            </marker>
          </defs>

          <!-- Light arrow at top -->
          <line
            :x1="siRectX + siRectW / 2"
            :y1="12"
            :x2="siRectX + siRectW / 2"
            :y2="crossTopY - 4"
            :stroke="wlColor"
            stroke-width="2.5"
            marker-end="url(#siAbsLightArrow)"
          />
          <text
            :x="siRectX + siRectW / 2"
            y="10"
            text-anchor="middle"
            class="light-label"
            :fill="wlColor"
          >{{ t('Light', '빛') }} ({{ wavelength }} nm)</text>

          <!-- Silicon rectangle with gradient -->
          <rect
            :x="siRectX"
            :y="crossTopY"
            :width="siRectW"
            :height="siRectH"
            :fill="`url(#${gradientId})`"
            stroke="var(--vp-c-text-3)"
            stroke-width="1"
          />

          <!-- "Silicon" label -->
          <text
            :x="siRectX + siRectW / 2"
            :y="crossTopY + siRectH + 16"
            text-anchor="middle"
            class="si-label"
          >Silicon</text>

          <!-- Photodiode depth dashed line -->
          <line
            :x1="siRectX - 8"
            :y1="pdLineY"
            :x2="siRectX + siRectW + 8"
            :y2="pdLineY"
            stroke="var(--vp-c-brand-1)"
            stroke-width="1.5"
            stroke-dasharray="5,3"
            v-if="pdDepth <= siThickness"
          />
          <text
            :x="siRectX + siRectW + 12"
            :y="pdLineY + 4"
            class="pd-label"
            v-if="pdDepth <= siThickness"
          >{{ t('PD depth', 'PD 깊이') }}</text>

          <!-- Depth axis labels (right side) -->
          <template v-for="d in depthTicks" :key="'dtick' + d">
            <line
              :x1="siRectX - 4"
              :y1="crossTopY + (d / siThickness) * siRectH"
              :x2="siRectX"
              :y2="crossTopY + (d / siThickness) * siRectH"
              stroke="var(--vp-c-text-3)"
              stroke-width="0.8"
            />
            <text
              :x="siRectX - 8"
              :y="crossTopY + (d / siThickness) * siRectH + 3"
              text-anchor="end"
              class="depth-tick-label"
            >{{ d }} um</text>
          </template>

          <!-- Penetration depth marker (if within Si) -->
          <line
            v-if="penetrationDepth <= siThickness"
            :x1="siRectX"
            :y1="crossTopY + (penetrationDepth / siThickness) * siRectH"
            :x2="siRectX + siRectW"
            :y2="crossTopY + (penetrationDepth / siThickness) * siRectH"
            stroke="#e74c3c"
            stroke-width="1"
            stroke-dasharray="3,2"
          />
          <text
            v-if="penetrationDepth <= siThickness"
            :x="siRectX + siRectW + 12"
            :y="crossTopY + (penetrationDepth / siThickness) * siRectH + 4"
            class="delta-label"
          >delta = {{ penetrationDepth.toFixed(2) }} um</text>
        </svg>
      </div>

      <!-- Right panel: Absorption coefficient chart -->
      <div class="panel">
        <svg :viewBox="`0 0 ${chartW} ${chartH}`" class="alpha-chart-svg">
          <!-- Grid -->
          <template v-for="exp in [1, 2, 3, 4, 5, 6]" :key="'gy' + exp">
            <line
              :x1="chartPadL"
              :y1="alphaToY(Math.pow(10, exp))"
              :x2="chartPadL + chartPlotW"
              :y2="alphaToY(Math.pow(10, exp))"
              stroke="var(--vp-c-divider)"
              stroke-width="0.5"
              stroke-dasharray="3,3"
            />
          </template>

          <!-- Axes -->
          <line
            :x1="chartPadL"
            :y1="chartPadT"
            :x2="chartPadL"
            :y2="chartPadT + chartPlotH"
            stroke="var(--vp-c-text-3)"
            stroke-width="1"
          />
          <line
            :x1="chartPadL"
            :y1="chartPadT + chartPlotH"
            :x2="chartPadL + chartPlotW"
            :y2="chartPadT + chartPlotH"
            stroke="var(--vp-c-text-3)"
            stroke-width="1"
          />
          <!-- Right Y-axis -->
          <line
            :x1="chartPadL + chartPlotW"
            :y1="chartPadT"
            :x2="chartPadL + chartPlotW"
            :y2="chartPadT + chartPlotH"
            stroke="var(--vp-c-text-3)"
            stroke-width="1"
          />

          <!-- Y-axis labels (left: alpha in cm^-1) -->
          <template v-for="exp in [1, 2, 3, 4, 5, 6]" :key="'yt' + exp">
            <line
              :x1="chartPadL - 4"
              :y1="alphaToY(Math.pow(10, exp))"
              :x2="chartPadL"
              :y2="alphaToY(Math.pow(10, exp))"
              stroke="var(--vp-c-text-3)"
              stroke-width="1"
            />
            <text
              :x="chartPadL - 7"
              :y="alphaToY(Math.pow(10, exp)) + 3"
              text-anchor="end"
              class="tick-label"
            >10<tspan :dy="-4" font-size="7">{{ exp }}</tspan></text>
          </template>
          <text
            x="10"
            :y="chartPadT + chartPlotH / 2"
            text-anchor="middle"
            :transform="`rotate(-90, 10, ${chartPadT + chartPlotH / 2})`"
            class="axis-label"
          >alpha (cm-1)</text>

          <!-- Right Y-axis labels (penetration depth in um) -->
          <template v-for="exp in [1, 2, 3, 4, 5, 6]" :key="'yr' + exp">
            <line
              :x1="chartPadL + chartPlotW"
              :y1="alphaToY(Math.pow(10, exp))"
              :x2="chartPadL + chartPlotW + 4"
              :y2="alphaToY(Math.pow(10, exp))"
              stroke="var(--vp-c-text-3)"
              stroke-width="1"
            />
            <text
              :x="chartPadL + chartPlotW + 7"
              :y="alphaToY(Math.pow(10, exp)) + 3"
              text-anchor="start"
              class="tick-label"
            >{{ depthLabel(Math.pow(10, exp)) }}</text>
          </template>
          <text
            :x="chartW - 8"
            :y="chartPadT + chartPlotH / 2"
            text-anchor="middle"
            :transform="`rotate(90, ${chartW - 8}, ${chartPadT + chartPlotH / 2})`"
            class="axis-label"
          >delta (um)</text>

          <!-- X-axis ticks -->
          <template v-for="wl in [400, 500, 600, 700, 800, 900, 1000]" :key="'xt' + wl">
            <line
              :x1="wlToChartX(wl)"
              :y1="chartPadT + chartPlotH"
              :x2="wlToChartX(wl)"
              :y2="chartPadT + chartPlotH + 4"
              stroke="var(--vp-c-text-3)"
              stroke-width="1"
            />
            <text
              :x="wlToChartX(wl)"
              :y="chartPadT + chartPlotH + 16"
              text-anchor="middle"
              class="tick-label"
            >{{ wl }}</text>
          </template>
          <text
            :x="chartPadL + chartPlotW / 2"
            :y="chartH - 2"
            text-anchor="middle"
            class="axis-label"
          >{{ t('Wavelength (nm)', '파장 (nm)') }}</text>

          <!-- Alpha curve -->
          <path :d="alphaPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />

          <!-- Current wavelength vertical line -->
          <line
            :x1="wlToChartX(wavelength)"
            :y1="chartPadT"
            :x2="wlToChartX(wavelength)"
            :y2="chartPadT + chartPlotH"
            :stroke="wlColor"
            stroke-width="1.5"
            stroke-dasharray="4,3"
          />
          <circle
            :cx="wlToChartX(wavelength)"
            :cy="alphaToY(alphaCm)"
            r="4"
            :fill="wlColor"
            stroke="#fff"
            stroke-width="1"
          />
        </svg>
      </div>
    </div>

    <!-- Info cards row -->
    <div class="info-cards">
      <div class="info-card">
        <div class="info-label">{{ t('Absorption coeff.', '흡수 계수') }} alpha</div>
        <div class="info-value">{{ alphaDisplay }} cm<sup>-1</sup></div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Penetration depth', '침투 깊이') }} delta</div>
        <div class="info-value">{{ penetrationDepth.toFixed(3) }} um</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Absorbed in PD', 'PD 내 흡수율') }}</div>
        <div class="info-value highlight">{{ (absorbedInPD * 100).toFixed(1) }}%</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Absorbed in full Si', '전체 Si 흡수율') }}</div>
        <div class="info-value">{{ (absorbedInSi * 100).toFixed(1) }}%</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { MATERIALS, getN } from '../composables/tmm'

const { t } = useLocale()

// Controls
const wavelength = ref(550)
const siThickness = ref(3.0)
const pdDepth = ref(2.0)

// Unique gradient ID to avoid collisions if multiple instances
const gradientId = 'siAbsGrad' + Math.random().toString(36).slice(2, 8)

// Cross-section SVG dimensions
const crossW = 300
const crossH = 350
const siRectX = 80
const siRectW = 120
const crossTopY = 40
const siRectH = 250

// Chart SVG dimensions
const chartW = 300
const chartH = 280
const chartPadL = 46
const chartPadR = 46
const chartPadT = 16
const chartPadB = 36
const chartPlotW = chartW - chartPadL - chartPadR
const chartPlotH = chartH - chartPadT - chartPadB

// Physics: compute alpha at current wavelength
const alphaUm = computed(() => {
  const wl_um = wavelength.value / 1000
  const [, k] = getN(MATERIALS['silicon'], wl_um)
  const kSafe = Math.max(k, 0)
  return (4 * Math.PI * kSafe) / wl_um
})

const alphaCm = computed(() => alphaUm.value * 1e4)

const penetrationDepth = computed(() => {
  if (alphaUm.value <= 0) return 999
  return 1 / alphaUm.value
})

const absorbedInPD = computed(() => {
  return 1 - Math.exp(-alphaUm.value * pdDepth.value)
})

const absorbedInSi = computed(() => {
  return 1 - Math.exp(-alphaUm.value * siThickness.value)
})

const alphaDisplay = computed(() => {
  const v = alphaCm.value
  if (v >= 1e5) return v.toExponential(2)
  if (v >= 1000) return Math.round(v).toLocaleString()
  if (v >= 1) return v.toFixed(1)
  return v.toExponential(2)
})

// Wavelength to visible color
function wavelengthToColor(wl: number): string {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) {
    r = -(wl - 440) / (440 - 380)
    g = 0
    b = 1
  } else if (wl >= 440 && wl < 490) {
    r = 0
    g = (wl - 440) / (490 - 440)
    b = 1
  } else if (wl >= 490 && wl < 510) {
    r = 0
    g = 1
    b = -(wl - 510) / (510 - 490)
  } else if (wl >= 510 && wl < 580) {
    r = (wl - 510) / (580 - 510)
    g = 1
    b = 0
  } else if (wl >= 580 && wl < 645) {
    r = 1
    g = -(wl - 645) / (645 - 580)
    b = 0
  } else if (wl >= 645 && wl <= 780) {
    r = 1
    g = 0
    b = 0
  } else if (wl > 780) {
    // Near-IR: deep red fading
    r = 0.6
    g = 0
    b = 0
  }

  // Intensity adjustment at edges of visible spectrum
  let factor = 1.0
  if (wl >= 380 && wl < 420) {
    factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
  } else if (wl >= 700 && wl <= 780) {
    factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
  } else if (wl > 780) {
    factor = 0.25
  }

  r = Math.round(255 * Math.pow(r * factor, 0.8))
  g = Math.round(255 * Math.pow(g * factor, 0.8))
  b = Math.round(255 * Math.pow(b * factor, 0.8))

  return `rgb(${r},${g},${b})`
}

const wlColor = computed(() => wavelengthToColor(wavelength.value))

// Gradient stops for cross-section: Beer-Lambert decay
const gradientStops = computed(() => {
  const stops: { offset: string; color: string; opacity: number }[] = []
  const numStops = 12
  const color = wavelengthToColor(wavelength.value)
  const alpha = alphaUm.value
  const thickness = siThickness.value

  for (let i = 0; i <= numStops; i++) {
    const frac = i / numStops
    const z = frac * thickness
    const intensity = Math.exp(-alpha * z)
    stops.push({
      offset: `${(frac * 100).toFixed(1)}%`,
      color,
      opacity: Math.max(0.03, intensity * 0.9),
    })
  }
  return stops
})

// Photodiode depth line Y position
const pdLineY = computed(() => {
  return crossTopY + (pdDepth.value / siThickness.value) * siRectH
})

// Depth axis ticks
const depthTicks = computed(() => {
  const ticks: number[] = [0]
  const maxTick = Math.floor(siThickness.value)
  for (let d = 1; d <= maxTick; d++) {
    ticks.push(d)
  }
  return ticks
})

// Chart helper: map wavelength to X
function wlToChartX(wl: number): number {
  return chartPadL + ((wl - 380) / (1000 - 380)) * chartPlotW
}

// Chart helper: map alpha (cm^-1) to Y (log scale, 10 to 10^6)
function alphaToY(alphaCmVal: number): number {
  const logMin = 1 // log10(10)
  const logMax = 6 // log10(10^6)
  const logVal = Math.log10(Math.max(alphaCmVal, 10))
  const frac = (logVal - logMin) / (logMax - logMin)
  return chartPadT + chartPlotH - frac * chartPlotH
}

// Depth label for right Y-axis: delta = 1/alpha, convert alpha cm^-1 to delta um
function depthLabel(alphaCmVal: number): string {
  // alpha_um = alpha_cm / 1e4, delta_um = 1 / alpha_um = 1e4 / alpha_cm
  const delta = 1e4 / alphaCmVal
  if (delta >= 100) return `${Math.round(delta)} um`
  if (delta >= 1) return `${delta.toFixed(1)} um`
  if (delta >= 0.01) return `${(delta * 1000).toFixed(0)} nm`
  return `${delta.toExponential(1)} um`
}

// Alpha curve path across all wavelengths
const alphaPath = computed(() => {
  let d = ''
  for (let wl = 380; wl <= 1000; wl += 5) {
    const wl_um = wl / 1000
    const [, k] = getN(MATERIALS['silicon'], wl_um)
    const alphaPerUm = (4 * Math.PI * Math.max(k, 0)) / wl_um
    const alphaPerCm = alphaPerUm * 1e4
    const x = wlToChartX(wl)
    const y = alphaToY(alphaPerCm)
    d += d === '' ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
})
</script>

<style scoped>
.si-absorption-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.si-absorption-container h4 {
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
  min-width: 180px;
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
.cross-section-svg {
  width: 100%;
  max-width: 300px;
  display: block;
  margin: 0 auto;
}
.alpha-chart-svg {
  width: 100%;
  max-width: 300px;
  display: block;
  margin: 0 auto;
}
.light-label {
  font-size: 10px;
  font-weight: 600;
}
.si-label {
  font-size: 11px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.pd-label {
  font-size: 9px;
  fill: var(--vp-c-brand-1);
  font-weight: 600;
}
.delta-label {
  font-size: 8px;
  fill: #e74c3c;
}
.depth-tick-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.tick-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
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
