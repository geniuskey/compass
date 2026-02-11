<template>
  <div class="staircase-ml-container">
    <h4>{{ t('Microlens Staircase Approximation', '마이크로렌즈 계단 근사') }}</h4>
    <p class="component-description">
      {{ t(
        'Adjust the number of slices and lens squareness to see how the staircase approximation changes.',
        '슬라이스 수와 렌즈 사각도를 조절하여 계단 근사가 어떻게 변하는지 확인하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Number of slices', '슬라이스 수') }}: <strong>{{ nSlices }}</strong>
        </label>
        <input type="range" min="5" max="50" step="1" v-model.number="nSlices" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Squareness (n)', '사각도 (n)') }}: <strong>{{ squareness.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.5" max="5.0" step="0.1" v-model.number="squareness" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Lens height', '렌즈 높이') }}:</span>
        <span class="info-value">0.600 um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Number of slices', '슬라이스 수') }}:</span>
        <span class="info-value">{{ nSlices }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Slice thickness', '슬라이스 두께') }}:</span>
        <span class="info-value">{{ sliceThickness.toFixed(4) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Estimated area error', '추정 면적 오차') }}:</span>
        <span class="info-value">{{ areaError.toFixed(2) }}%</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="staircase-svg">
        <!-- Background -->
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />

        <!-- Planarization layer -->
        <rect
          :x="svgW / 2 - scaleR - 20"
          :y="baseY"
          :width="2 * scaleR + 40"
          :height="24"
          fill="#c0c0c0"
          fill-opacity="0.35"
          stroke="var(--vp-c-divider)"
          stroke-width="1"
        />
        <text
          :x="svgW / 2"
          :y="baseY + 15"
          text-anchor="middle"
          class="layer-label"
        >{{ t('Planarization layer', '평탄화층') }}</text>

        <!-- Error shading between smooth curve and staircase -->
        <path
          :d="errorPath"
          fill="#e74c3c"
          fill-opacity="0.2"
          stroke="none"
        />

        <!-- Staircase rectangles -->
        <rect
          v-for="(step, i) in staircaseSteps"
          :key="'step-' + i"
          :x="step.x"
          :y="step.y"
          :width="step.w"
          :height="step.h"
          fill="#d4e6f1"
          fill-opacity="0.7"
          stroke="#85c1e9"
          stroke-width="1"
        />

        <!-- Smooth superellipse curve (dashed) -->
        <path
          :d="smoothCurvePath"
          fill="none"
          stroke="var(--vp-c-brand-1)"
          stroke-width="2"
          stroke-dasharray="6,3"
        />

        <!-- Dimension arrow: height (h) on right side -->
        <line
          :x1="svgW / 2 + scaleR + 20"
          :y1="baseY"
          :x2="svgW / 2 + scaleR + 20"
          :y2="baseY - scaleH"
          stroke="var(--vp-c-text-2)"
          stroke-width="1"
          marker-start="url(#arrowDown)"
          marker-end="url(#arrowUp)"
        />
        <text
          :x="svgW / 2 + scaleR + 30"
          :y="baseY - scaleH / 2 + 4"
          class="dim-label"
        >h</text>

        <!-- Dimension arrow: radius (R) at bottom -->
        <line
          :x1="svgW / 2"
          :y1="baseY + 30"
          :x2="svgW / 2 + scaleR"
          :y2="baseY + 30"
          stroke="var(--vp-c-text-2)"
          stroke-width="1"
          marker-start="url(#arrowLeft)"
          marker-end="url(#arrowRight)"
        />
        <text
          :x="svgW / 2 + scaleR / 2"
          :y="baseY + 44"
          text-anchor="middle"
          class="dim-label"
        >R</text>

        <!-- Axis labels -->
        <text
          :x="svgW / 2 - scaleR - 30"
          :y="baseY - scaleH / 2"
          text-anchor="middle"
          class="axis-label"
        >z</text>
        <text
          :x="svgW / 2"
          :y="baseY + 56"
          text-anchor="middle"
          class="axis-label"
        >x</text>

        <!-- Legend labels on the SVG -->
        <line
          :x1="svgW - 160"
          y1="20"
          :x2="svgW - 130"
          y2="20"
          stroke="var(--vp-c-brand-1)"
          stroke-width="2"
          stroke-dasharray="6,3"
        />
        <text :x="svgW - 125" y="24" class="legend-label">
          {{ t('Smooth profile', '매끄러운 프로파일') }}
        </text>

        <rect
          :x="svgW - 160"
          y="32"
          width="30"
          height="12"
          fill="#d4e6f1"
          fill-opacity="0.7"
          stroke="#85c1e9"
          stroke-width="1"
        />
        <text :x="svgW - 125" y="42" class="legend-label">
          {{ t('Staircase', '계단 근사') }}
        </text>

        <rect
          :x="svgW - 160"
          y="50"
          width="30"
          height="12"
          fill="#e74c3c"
          fill-opacity="0.2"
          stroke="none"
        />
        <text :x="svgW - 125" y="60" class="legend-label">
          {{ t('Error region', '오차 영역') }}
        </text>

        <!-- Arrow markers -->
        <defs>
          <marker id="arrowUp" markerWidth="6" markerHeight="6" refX="3" refY="6" orient="auto">
            <polygon points="0 6, 3 0, 6 6" fill="var(--vp-c-text-2)" />
          </marker>
          <marker id="arrowDown" markerWidth="6" markerHeight="6" refX="3" refY="0" orient="auto">
            <polygon points="0 0, 3 6, 6 0" fill="var(--vp-c-text-2)" />
          </marker>
          <marker id="arrowLeft" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto">
            <polygon points="6 0, 0 3, 6 6" fill="var(--vp-c-text-2)" />
          </marker>
          <marker id="arrowRight" markerWidth="6" markerHeight="6" refX="0" refY="3" orient="auto">
            <polygon points="0 0, 6 3, 0 6" fill="var(--vp-c-text-2)" />
          </marker>
        </defs>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const nSlices = ref(20)
const squareness = ref(2.5)

// Physical parameters (um)
const lensH = 0.6
const lensR = 0.48

// SVG layout
const svgW = 520
const svgH = 300
const baseY = 250 // bottom of the lens (y coordinate in SVG)
const scaleR = 150 // pixels for radius R
const scaleH = 150 // pixels for height h

const sliceThickness = computed(() => lensH / nSlices.value)

// Superellipse profile: z(x) = h * (1 - |x/R|^n)^(1/n)
// x is in physical units [-R, R], returns z in [0, h]
function superellipseZ(x, n) {
  const absXR = Math.abs(x / lensR)
  if (absXR >= 1) return 0
  return lensH * Math.pow(1 - Math.pow(absXR, n), 1 / n)
}

// Map physical coords to SVG
function toSvgX(physX) {
  return svgW / 2 + (physX / lensR) * scaleR
}
function toSvgY(physZ) {
  return baseY - (physZ / lensH) * scaleH
}

// Smooth curve SVG path (left to right, many points)
const smoothCurvePath = computed(() => {
  const n = squareness.value
  const numPts = 200
  const pts = []
  for (let i = 0; i <= numPts; i++) {
    const physX = -lensR + (2 * lensR * i) / numPts
    const physZ = superellipseZ(physX, n)
    pts.push({ x: toSvgX(physX), y: toSvgY(physZ) })
  }
  let d = `M ${pts[0].x.toFixed(2)} ${pts[0].y.toFixed(2)}`
  for (let i = 1; i < pts.length; i++) {
    d += ` L ${pts[i].x.toFixed(2)} ${pts[i].y.toFixed(2)}`
  }
  return d
})

// Staircase steps: divide lens height into nSlices horizontal slices
// For each slice, find the width at that z-height from the superellipse
const staircaseSteps = computed(() => {
  const n = squareness.value
  const steps = []
  const dz = lensH / nSlices.value

  for (let i = 0; i < nSlices.value; i++) {
    const zBot = i * dz
    const zTop = (i + 1) * dz
    // Find half-width at zBot from the inverse superellipse:
    // z = h*(1 - |x/R|^n)^(1/n)  =>  |x/R| = (1 - (z/h)^n)^(1/n)
    const zMid = zBot // use bottom of slice to determine width
    const ratio = zMid / lensH
    let halfWidthPhys
    if (ratio >= 1) {
      halfWidthPhys = 0
    } else {
      halfWidthPhys = lensR * Math.pow(1 - Math.pow(ratio, n), 1 / n)
    }

    const svgX = toSvgX(-halfWidthPhys)
    const svgYTop = toSvgY(zTop)
    const svgYBot = toSvgY(zBot)
    const svgWidth = toSvgX(halfWidthPhys) - toSvgX(-halfWidthPhys)
    const svgHeight = svgYBot - svgYTop

    if (svgWidth > 0 && svgHeight > 0) {
      steps.push({
        x: svgX,
        y: svgYTop,
        w: svgWidth,
        h: svgHeight,
        halfWidthPhys,
        zBot,
        zTop
      })
    }
  }
  return steps
})

// Error path: region between smooth curve and staircase outline
const errorPath = computed(() => {
  const n = squareness.value
  const steps = staircaseSteps.value
  if (steps.length === 0) return ''

  // Build the staircase outline going from left baseline, up the staircase, across top, and down
  // Right half only, then mirror for left half
  // The staircase outline (going up from base):
  const staircaseRight = []
  const staircaseLeft = []
  for (let i = 0; i < steps.length; i++) {
    const s = steps[i]
    const rightX = toSvgX(s.halfWidthPhys)
    const leftX = toSvgX(-s.halfWidthPhys)
    const yBot = toSvgY(s.zBot)
    const yTop = toSvgY(s.zTop)
    // Right side: go right at yBot, then up to yTop
    staircaseRight.push({ x: rightX, y: yBot })
    staircaseRight.push({ x: rightX, y: yTop })
    // Left side: collect in reverse order
    staircaseLeft.push({ x: leftX, y: yTop })
    staircaseLeft.push({ x: leftX, y: yBot })
  }

  // Build smooth curve points (right half, bottom to top)
  const smoothRight = []
  const numPts = 100
  for (let i = 0; i <= numPts; i++) {
    const physX = lensR * i / numPts
    const physZ = superellipseZ(physX, n)
    smoothRight.push({ x: toSvgX(physX), y: toSvgY(physZ) })
  }

  // Smooth curve points (left half, top to bottom)
  const smoothLeft = []
  for (let i = numPts; i >= 0; i--) {
    const physX = -lensR * i / numPts
    const physZ = superellipseZ(physX, n)
    smoothLeft.push({ x: toSvgX(physX), y: toSvgY(physZ) })
  }

  // Right half error region: smooth curve (bottom-to-top) then staircase (top-to-bottom)
  let d = ''

  // Right error region
  d += `M ${smoothRight[0].x.toFixed(2)} ${smoothRight[0].y.toFixed(2)}`
  for (let i = 1; i < smoothRight.length; i++) {
    d += ` L ${smoothRight[i].x.toFixed(2)} ${smoothRight[i].y.toFixed(2)}`
  }
  // Now trace staircase back down (reverse order)
  for (let i = staircaseRight.length - 1; i >= 0; i--) {
    d += ` L ${staircaseRight[i].x.toFixed(2)} ${staircaseRight[i].y.toFixed(2)}`
  }
  d += ' Z'

  // Left error region
  const smoothLeftBotToTop = []
  for (let i = 0; i <= numPts; i++) {
    const physX = -lensR * i / numPts
    const physZ = superellipseZ(physX, n)
    smoothLeftBotToTop.push({ x: toSvgX(physX), y: toSvgY(physZ) })
  }

  d += ` M ${smoothLeftBotToTop[0].x.toFixed(2)} ${smoothLeftBotToTop[0].y.toFixed(2)}`
  for (let i = 1; i < smoothLeftBotToTop.length; i++) {
    d += ` L ${smoothLeftBotToTop[i].x.toFixed(2)} ${smoothLeftBotToTop[i].y.toFixed(2)}`
  }
  // Staircase left side (top to bottom) -- reverse staircaseLeft
  for (let i = staircaseLeft.length - 1; i >= 0; i--) {
    d += ` L ${staircaseLeft[i].x.toFixed(2)} ${staircaseLeft[i].y.toFixed(2)}`
  }
  d += ' Z'

  return d
})

// Compute area error using numerical integration
const smoothArea = computed(() => {
  const n = squareness.value
  // Integrate z(x) from -R to R using trapezoidal rule
  const numPts = 1000
  const dx = (2 * lensR) / numPts
  let area = 0
  for (let i = 0; i <= numPts; i++) {
    const x = -lensR + i * dx
    const z = superellipseZ(x, n)
    const weight = (i === 0 || i === numPts) ? 0.5 : 1
    area += z * weight * dx
  }
  return area
})

const staircaseArea = computed(() => {
  const steps = staircaseSteps.value
  let area = 0
  for (const s of steps) {
    area += 2 * s.halfWidthPhys * (s.zTop - s.zBot)
  }
  return area
})

const areaError = computed(() => {
  const smooth = smoothArea.value
  if (smooth === 0) return 0
  return Math.abs((staircaseArea.value - smooth) / smooth) * 100
})
</script>

<style scoped>
.staircase-ml-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.staircase-ml-container h4 {
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
  display: flex;
  justify-content: center;
}
.staircase-svg {
  width: 100%;
  max-width: 540px;
}
.layer-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.dim-label {
  font-size: 12px;
  fill: var(--vp-c-text-2);
  font-weight: 700;
  font-style: italic;
}
.axis-label {
  font-size: 13px;
  fill: var(--vp-c-text-2);
  font-weight: 700;
  font-style: italic;
}
.legend-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
</style>
