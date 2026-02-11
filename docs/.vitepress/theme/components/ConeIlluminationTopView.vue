<template>
  <div class="cone-illum-container">
    <h4>{{ t('Cone Illumination – Top View', '콘 조명 – 상면도') }}</h4>
    <p class="component-description">
      {{ t(
        'Bird\'s eye view of cone illumination on a 2×2 Bayer pixel array. Adjust CRA, f-number, and sampling to see how the illumination footprint covers the pixels.',
        '2×2 베이어 픽셀 배열에 대한 콘 조명의 상면도입니다. CRA, f-넘버, 샘플링을 조정하여 조명 풋프린트가 픽셀을 어떻게 덮는지 확인하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('CRA', 'CRA') }}: <strong>{{ cra.toFixed(1) }}&deg;</strong>
        </label>
        <input type="range" min="0" max="30" step="0.5" v-model.number="cra" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('f-number', 'f-넘버') }}: <strong>f/{{ fNumber.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.4" max="8.0" step="0.1" v-model.number="fNumber" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Sampling points', '샘플링 포인트') }}: <strong>{{ nPoints }}</strong>
        </label>
        <input type="range" min="7" max="91" step="1" v-model.number="nPoints" class="ctrl-range" />
      </div>
      <div class="toggle-group">
        <label class="toggle-label">{{ t('Sampling method', '샘플링 방식') }}:</label>
        <div class="toggle-buttons">
          <button
            :class="['toggle-btn', { active: samplingMethod === 'fibonacci' }]"
            @click="samplingMethod = 'fibonacci'"
          >
            {{ t('Fibonacci', '피보나치') }}
          </button>
          <button
            :class="['toggle-btn', { active: samplingMethod === 'grid' }]"
            @click="samplingMethod = 'grid'"
          >
            {{ t('Grid', '격자') }}
          </button>
        </div>
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Footprint diameter', '풋프린트 직경') }}:</span>
        <span class="info-value">{{ footprintDiameter.toFixed(3) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Lens area', '렌즈 면적') }}:</span>
        <span class="info-value">{{ lensArea.toFixed(4) }} um&sup2;</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('CRA shift', 'CRA 시프트') }}:</span>
        <span class="info-value">{{ craShiftUm.toFixed(3) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Coverage ratio', '커버리지 비율') }}:</span>
        <span class="info-value">{{ (coverageRatio * 100).toFixed(1) }}%</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgSize} ${svgSize}`" class="topview-svg">
        <defs>
          <marker id="topviewCraArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#e67e22" />
          </marker>
        </defs>

        <!-- Pixel grid (2x2 Bayer) -->
        <!-- R (top-left) -->
        <rect
          :x="gridOriginX"
          :y="gridOriginY"
          :width="pixelSizePx"
          :height="pixelSizePx"
          fill="#e74c3c33"
          stroke="#e74c3c"
          stroke-width="1.5"
        />
        <text
          :x="gridOriginX + pixelSizePx / 2"
          :y="gridOriginY + pixelSizePx / 2 + 4"
          text-anchor="middle"
          class="pixel-label"
          fill="#e74c3c"
        >R</text>

        <!-- G (top-right) -->
        <rect
          :x="gridOriginX + pixelSizePx"
          :y="gridOriginY"
          :width="pixelSizePx"
          :height="pixelSizePx"
          fill="#27ae6033"
          stroke="#27ae60"
          stroke-width="1.5"
        />
        <text
          :x="gridOriginX + pixelSizePx * 1.5"
          :y="gridOriginY + pixelSizePx / 2 + 4"
          text-anchor="middle"
          class="pixel-label"
          fill="#27ae60"
        >G</text>

        <!-- G (bottom-left) -->
        <rect
          :x="gridOriginX"
          :y="gridOriginY + pixelSizePx"
          :width="pixelSizePx"
          :height="pixelSizePx"
          fill="#27ae6033"
          stroke="#27ae60"
          stroke-width="1.5"
        />
        <text
          :x="gridOriginX + pixelSizePx / 2"
          :y="gridOriginY + pixelSizePx * 1.5 + 4"
          text-anchor="middle"
          class="pixel-label"
          fill="#27ae60"
        >G</text>

        <!-- B (bottom-right) -->
        <rect
          :x="gridOriginX + pixelSizePx"
          :y="gridOriginY + pixelSizePx"
          :width="pixelSizePx"
          :height="pixelSizePx"
          fill="#2980b933"
          stroke="#2980b9"
          stroke-width="1.5"
        />
        <text
          :x="gridOriginX + pixelSizePx * 1.5"
          :y="gridOriginY + pixelSizePx * 1.5 + 4"
          text-anchor="middle"
          class="pixel-label"
          fill="#2980b9"
        >B</text>

        <!-- Cone footprint circle -->
        <circle
          :cx="coneCenterSvgX"
          :cy="coneCenterSvgY"
          :r="footprintRadiusPx"
          fill="#3498db"
          fill-opacity="0.1"
          stroke="#3498db"
          stroke-width="1.5"
          stroke-dasharray="6,3"
        />

        <!-- CRA shift arrow (pixel center to cone center) -->
        <template v-if="cra > 0.5">
          <line
            :x1="pixelArrayCenterX"
            :y1="pixelArrayCenterY"
            :x2="coneCenterSvgX"
            :y2="coneCenterSvgY"
            stroke="#e67e22"
            stroke-width="1.5"
            marker-end="url(#topviewCraArrow)"
          />
          <text
            :x="(pixelArrayCenterX + coneCenterSvgX) / 2 + 8"
            :y="(pixelArrayCenterY + coneCenterSvgY) / 2 - 6"
            class="shift-label"
          >{{ t('CRA shift', 'CRA 시프트') }}</text>
        </template>

        <!-- Pixel center marker -->
        <circle
          :cx="pixelArrayCenterX"
          :cy="pixelArrayCenterY"
          r="3"
          fill="none"
          stroke="var(--vp-c-text-2)"
          stroke-width="1"
        />
        <line
          :x1="pixelArrayCenterX - 5"
          :y1="pixelArrayCenterY"
          :x2="pixelArrayCenterX + 5"
          :y2="pixelArrayCenterY"
          stroke="var(--vp-c-text-2)"
          stroke-width="0.8"
        />
        <line
          :x1="pixelArrayCenterX"
          :y1="pixelArrayCenterY - 5"
          :x2="pixelArrayCenterX"
          :y2="pixelArrayCenterY + 5"
          stroke="var(--vp-c-text-2)"
          stroke-width="0.8"
        />

        <!-- Sampling points -->
        <circle
          v-for="(pt, idx) in samplingPoints"
          :key="idx"
          :cx="pt.svgX"
          :cy="pt.svgY"
          r="3"
          :fill="pt.color"
          :opacity="pt.opacity"
          stroke="none"
        />

        <!-- Scale bar -->
        <line
          :x1="svgSize - 20 - scaleBarLengthPx"
          :y1="svgSize - 25"
          :x2="svgSize - 20"
          :y2="svgSize - 25"
          stroke="var(--vp-c-text-1)"
          stroke-width="2"
        />
        <line
          :x1="svgSize - 20 - scaleBarLengthPx"
          :y1="svgSize - 30"
          :x2="svgSize - 20 - scaleBarLengthPx"
          :y2="svgSize - 20"
          stroke="var(--vp-c-text-1)"
          stroke-width="1.5"
        />
        <line
          :x1="svgSize - 20"
          :y1="svgSize - 30"
          :x2="svgSize - 20"
          :y2="svgSize - 20"
          stroke="var(--vp-c-text-1)"
          stroke-width="1.5"
        />
        <text
          :x="svgSize - 20 - scaleBarLengthPx / 2"
          :y="svgSize - 12"
          text-anchor="middle"
          class="scale-label"
        >1 um</text>

        <!-- Axis labels -->
        <text x="8" :y="svgSize / 2" class="axis-label" text-anchor="middle" transform-origin="center" :transform="`rotate(-90, 8, ${svgSize / 2})`">y (um)</text>
        <text :x="svgSize / 2" :y="svgSize - 4" class="axis-label" text-anchor="middle">x (um)</text>
      </svg>
    </div>

    <div class="legend-row">
      <span class="legend-item">
        <svg width="18" height="12"><circle cx="6" cy="6" r="5" fill="#3498db" fill-opacity="0.1" stroke="#3498db" stroke-width="1" stroke-dasharray="3,2" /></svg>
        {{ t('Cone footprint', '콘 풋프린트') }}
      </span>
      <span class="legend-item">
        <svg width="18" height="12"><circle cx="6" cy="6" r="3" fill="#3498db" opacity="0.8" /></svg>
        {{ t('Sampling point', '샘플링 포인트') }}
      </span>
      <span class="legend-item">
        <svg width="18" height="12"><line x1="1" y1="6" x2="17" y2="6" stroke="#e67e22" stroke-width="1.5" /></svg>
        {{ t('CRA shift', 'CRA 시프트') }}
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

// --- Reactive controls ---
const cra = ref(10)
const fNumber = ref(2.8)
const nPoints = ref(37)
const samplingMethod = ref('fibonacci')

// --- Constants ---
const svgSize = 400
const pixelPitch = 1.0   // um
const domainSize = 2.0   // um (2x2 Bayer)
const stackHeight = 2.5  // um from microlens to pixel plane
const scale = 150         // px per um

// --- Derived pixel grid layout ---
const pixelSizePx = pixelPitch * scale
const gridOriginX = (svgSize - domainSize * scale) / 2
const gridOriginY = (svgSize - domainSize * scale) / 2
const pixelArrayCenterX = svgSize / 2
const pixelArrayCenterY = svgSize / 2

// Scale bar: 1 um
const scaleBarLengthPx = scale

// --- Physics computations ---
const craRad = computed(() => cra.value * Math.PI / 180)
const halfAngle = computed(() => Math.asin(1 / (2 * fNumber.value)))

// Cone footprint radius in um: r = stackHeight * tan(half_angle)
const footprintRadiusUm = computed(() => stackHeight * Math.tan(halfAngle.value))
const footprintRadiusPx = computed(() => footprintRadiusUm.value * scale)
const footprintDiameter = computed(() => 2 * footprintRadiusUm.value)

// Lens area (pi * r^2) in um^2
const lensArea = computed(() => Math.PI * footprintRadiusUm.value * footprintRadiusUm.value)

// CRA shift in um: shift = stackHeight * tan(CRA)
const craShiftUm = computed(() => stackHeight * Math.tan(craRad.value))
const craShiftPx = computed(() => craShiftUm.value * scale)

// Cone center in SVG coords (shifted from pixel array center along +x direction)
const coneCenterSvgX = computed(() => pixelArrayCenterX + craShiftPx.value)
const coneCenterSvgY = computed(() => pixelArrayCenterY)

// Coverage ratio: footprint area / pixel area (single pixel)
const coverageRatio = computed(() => {
  const pixelArea = pixelPitch * pixelPitch
  return lensArea.value / pixelArea
})

// --- Sampling points generation ---
const GOLDEN_RATIO = (1 + Math.sqrt(5)) / 2

const samplingPoints = computed(() => {
  const ha = halfAngle.value
  const n = nPoints.value
  const points = []

  if (samplingMethod.value === 'fibonacci') {
    for (let i = 0; i < n; i++) {
      const thetaFrac = n > 1 ? i / (n - 1) : 0
      const theta = ha * Math.sqrt(thetaFrac)
      const phi = 2 * Math.PI * i / GOLDEN_RATIO

      // Project onto pixel plane
      const rProj = stackHeight * Math.tan(theta)
      const xUm = rProj * Math.cos(phi)
      const yUm = rProj * Math.sin(phi)

      // Weight: points near center have higher weight (uniform area weighting)
      const weight = n > 1 ? 1 - 0.5 * thetaFrac : 1

      points.push({
        svgX: coneCenterSvgX.value + xUm * scale,
        svgY: coneCenterSvgY.value + yUm * scale,
        color: interpolateColor(weight),
        opacity: 0.4 + 0.5 * weight
      })
    }
  } else {
    // Grid sampling
    const nTheta = Math.max(1, Math.floor(Math.sqrt(n)))
    const nPhi = Math.max(1, Math.floor(n / nTheta))

    for (let it = 0; it < nTheta; it++) {
      const theta = nTheta > 1 ? ha * (it / (nTheta - 1)) : 0
      for (let ip = 0; ip < nPhi; ip++) {
        const phi = nPhi > 1 ? 2 * Math.PI * (ip / nPhi) : 0

        const rProj = stackHeight * Math.tan(theta)
        const xUm = rProj * Math.cos(phi)
        const yUm = rProj * Math.sin(phi)

        const thetaFrac = nTheta > 1 ? it / (nTheta - 1) : 0
        const weight = 1 - 0.5 * thetaFrac

        points.push({
          svgX: coneCenterSvgX.value + xUm * scale,
          svgY: coneCenterSvgY.value + yUm * scale,
          color: interpolateColor(weight),
          opacity: 0.4 + 0.5 * weight
        })
      }
    }
  }

  return points
})

// Color interpolation: high weight = deep blue, low weight = light blue
function interpolateColor(weight) {
  // Interpolate between light cyan (#88d4f5) and deep blue (#2471a3)
  const r = Math.round(0x24 + (0x88 - 0x24) * (1 - weight))
  const g = Math.round(0x71 + (0xd4 - 0x71) * (1 - weight))
  const b = Math.round(0xa3 + (0xf5 - 0xa3) * (1 - weight))
  return `rgb(${r}, ${g}, ${b})`
}
</script>

<style scoped>
.cone-illum-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.cone-illum-container h4 {
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
.toggle-group {
  min-width: 160px;
}
.toggle-label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.toggle-buttons {
  display: flex;
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  overflow: hidden;
}
.toggle-btn {
  flex: 1;
  padding: 4px 12px;
  font-size: 0.82em;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.toggle-btn:first-child {
  border-right: 1px solid var(--vp-c-divider);
}
.toggle-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.toggle-btn:hover:not(.active) {
  background: var(--vp-c-bg-soft);
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
  display: flex;
  justify-content: center;
}
.topview-svg {
  width: 100%;
  max-width: 420px;
}
.pixel-label {
  font-size: 12px;
  font-weight: 700;
  pointer-events: none;
}
.shift-label {
  font-size: 9px;
  fill: #e67e22;
  font-weight: 600;
}
.scale-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.legend-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 10px;
  font-size: 0.82em;
  color: var(--vp-c-text-2);
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>
