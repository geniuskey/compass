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
            v-for="method in samplingMethods"
            :key="method.id"
            type="button"
            :class="['toggle-btn', { active: samplingMethod === method.id }]"
            :aria-pressed="samplingMethod === method.id"
            @click="samplingMethod = method.id"
          >
            {{ t(method.labelEn, method.labelKo) }}
          </button>
        </div>
      </div>
    </div>

    <p class="method-hint">{{ methodHint }}</p>

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
      <div class="info-card">
        <span class="info-label">{{ t('Rendered samples', '표시 샘플') }}:</span>
        <span class="info-value">{{ samplingPoints.length }}</span>
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

const samplingMethods = [
  { id: 'fibonacci', labelEn: 'Fibonacci', labelKo: '피보나치' },
  { id: 'rings', labelEn: 'Rings', labelKo: '링' },
  { id: 'halton', labelEn: 'Halton', labelKo: '할튼' },
  { id: 'hammersley', labelEn: 'Hammersley', labelKo: '해머슬리' },
  { id: 'gauss', labelEn: 'Gauss', labelKo: '가우스' },
  { id: 'grid', labelEn: 'Grid legacy', labelKo: '격자 legacy' },
]

const methodHint = computed(() => {
  const hints = {
    fibonacci: t(
      'Recommended default: golden-angle, near-uniform coverage without ring artifacts.',
      '권장 기본값: 황금각 기반으로 ring artifact 없이 준균일하게 덮습니다.'
    ),
    rings: t(
      'Equal-area concentric rings: easy to inspect and more balanced than a polar grid.',
      '동일 면적 동심 ring: 눈으로 검토하기 쉽고 polar grid보다 균형이 좋습니다.'
    ),
    halton: t(
      'Low-discrepancy sequence: good for convergence checks without regular angular symmetry.',
      'Low-discrepancy sequence: 규칙적인 각도 대칭 없이 수렴 확인에 유리합니다.'
    ),
    hammersley: t(
      'Fixed-budget low-discrepancy set: often cleaner than Halton when n_points is known upfront.',
      '고정 sample budget용 low-discrepancy set: n_points를 미리 알 때 Halton보다 더 깔끔한 분포가 나옵니다.'
    ),
    gauss: t(
      'Gauss-Legendre radial quadrature: best when the goal is angular integration accuracy.',
      'Gauss-Legendre radial quadrature: 각도 적분 정확도가 목표일 때 유리합니다.'
    ),
    grid: t(
      'Legacy polar grid: useful as a baseline, but it clusters poorly for the same sample budget.',
      'Legacy polar grid: 기준 비교용으로만 유용하며 같은 sample budget에서 분포가 좋지 않습니다.'
    ),
  }
  return hints[samplingMethod.value]
})

// --- Constants ---
const svgSize = 400
const pixelPitch = 1.0   // um
const domainSize = 2.0   // um (2x2 Bayer)
const stackHeight = 2.5  // um used for cone half-angle footprint radius
const scale = 150         // px per um
const stackLayers = [
  { thickness: 0.30, n: 1.46 }, // planarization
  { thickness: 0.60, n: 1.62 }, // green color filter reference
  { thickness: 0.010, n: 1.46 },
  { thickness: 0.025, n: 2.00 },
  { thickness: 0.015, n: 1.46 },
  { thickness: 0.030, n: 2.02 },
  { thickness: 2.50, n: 4.00 }, // silicon top to PD center
]

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

function snellShiftUm(angleRad) {
  const sinCra = Math.sin(angleRad)
  return stackLayers.reduce((sum, layer) => {
    const sinTheta = Math.min(0.999, Math.abs(sinCra) / layer.n)
    const cosTheta = Math.sqrt(1 - sinTheta * sinTheta)
    return sum + layer.thickness * sinTheta / cosTheta
  }, 0)
}

// CRA footprint-center shift in um. Use the refracted stack path rather than
// raw air-path stackHeight*tan(CRA), which overstates shifts in high-index layers.
const craShiftUm = computed(() => snellShiftUm(craRad.value))
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
  let samples = []

  if (samplingMethod.value === 'rings') {
    samples = ringSamples(n, ha)
  } else if (samplingMethod.value === 'halton') {
    samples = haltonSamples(n, ha)
  } else if (samplingMethod.value === 'hammersley') {
    samples = hammersleySamples(n, ha)
  } else if (samplingMethod.value === 'gauss') {
    samples = gaussSamples(n, ha)
  } else if (samplingMethod.value === 'grid') {
    samples = gridSamples(n, ha)
  } else {
    samples = fibonacciSamples(n, ha)
  }

  return finalizeSamples(samples)
})

function thetaFromCapFraction(u, ha) {
  return Math.acos(1 - u * (1 - Math.cos(ha)))
}

function projectedSample(theta, phi, weight) {
  const rProj = stackHeight * Math.tan(theta)
  const xUm = rProj * Math.cos(phi)
  const yUm = rProj * Math.sin(phi)
  return {
    svgX: coneCenterSvgX.value + xUm * scale,
    svgY: coneCenterSvgY.value + yUm * scale,
    weight,
  }
}

function angularWeight(theta) {
  return Math.max(0.05, Math.cos(theta))
}

function finalizeSamples(samples) {
  const maxWeight = Math.max(...samples.map((sample) => sample.weight), 1e-9)
  return samples.map((sample) => {
    const relativeWeight = sample.weight / maxWeight
    return {
      svgX: sample.svgX,
      svgY: sample.svgY,
      color: interpolateColor(relativeWeight),
      opacity: 0.35 + 0.55 * relativeWeight,
    }
  })
}

function fibonacciSamples(n, ha) {
  const count = Math.max(1, n)
  return Array.from({ length: count }, (_, i) => {
    const theta = thetaFromCapFraction((i + 0.5) / count, ha)
    const phi = 2 * Math.PI * i / GOLDEN_RATIO
    return projectedSample(theta, phi, angularWeight(theta))
  })
}

function ringSamples(n, ha) {
  const count = Math.max(1, n)
  const nRings = Math.max(1, Math.round(Math.sqrt(count)))
  const counts = ringCounts(count, nRings)
  const samples = []
  for (let ring = 0; ring < nRings; ring++) {
    const uInner = ring / nRings
    const uOuter = (ring + 1) / nRings
    const theta = thetaFromCapFraction((uInner + uOuter) / 2, ha)
    const ringWeight = (uOuter - uInner) * angularWeight(theta) / counts[ring]
    const offset = ring % 2 === 0 ? 0 : 0.5
    for (let i = 0; i < counts[ring]; i++) {
      samples.push(projectedSample(theta, 2 * Math.PI * ((i + offset) / counts[ring]), ringWeight))
    }
  }
  return samples
}

function haltonSamples(n, ha) {
  const count = Math.max(1, n)
  return Array.from({ length: count }, (_, i) => {
    const theta = thetaFromCapFraction(radicalInverse(i + 1, 2), ha)
    const phi = 2 * Math.PI * radicalInverse(i + 1, 3)
    return projectedSample(theta, phi, angularWeight(theta))
  })
}

function hammersleySamples(n, ha) {
  const count = Math.max(1, n)
  return Array.from({ length: count }, (_, i) => {
    const theta = thetaFromCapFraction((i + 0.5) / count, ha)
    const phi = 2 * Math.PI * radicalInverse(i + 1, 2)
    return projectedSample(theta, phi, angularWeight(theta))
  })
}

function gaussSamples(n, ha) {
  const nTheta = Math.max(2, Math.round(Math.sqrt(n)))
  const nPhi = Math.max(4, Math.floor(n / nTheta))
  const { nodes, weights } = gaussLegendre(nTheta)
  const samples = []
  for (let it = 0; it < nTheta; it++) {
    const theta = 0.5 * ha * (nodes[it] + 1)
    const radialWeight = 0.5 * ha * weights[it] * Math.sin(theta) * angularWeight(theta)
    for (let ip = 0; ip < nPhi; ip++) {
      samples.push(projectedSample(theta, 2 * Math.PI * (ip / nPhi), radialWeight / nPhi))
    }
  }
  return samples
}

function gridSamples(n, ha) {
  const nTheta = Math.max(2, Math.floor(Math.sqrt(n)))
  const nPhi = Math.max(4, Math.floor(n / nTheta))
  const samples = []
  for (let it = 0; it < nTheta; it++) {
    const theta = nTheta > 1 ? ha * (it / (nTheta - 1)) : 0
    for (let ip = 0; ip < nPhi; ip++) {
      samples.push(projectedSample(theta, 2 * Math.PI * (ip / nPhi), angularWeight(theta) * Math.sin(theta + 1e-6)))
    }
  }
  return samples
}

function ringCounts(n, nRings) {
  const weights = Array.from({ length: nRings }, (_, i) => i + 1)
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  const counts = weights.map((weight) => Math.max(1, Math.round(n * weight / total)))
  while (counts.reduce((sum, value) => sum + value, 0) < n) counts[counts.length - 1] += 1
  while (counts.reduce((sum, value) => sum + value, 0) > n) {
    for (let i = counts.length - 1; i >= 0; i--) {
      if (counts[i] > 1) {
        counts[i] -= 1
        break
      }
    }
  }
  return counts
}

function radicalInverse(index, base) {
  let result = 0
  let fraction = 1 / base
  while (index > 0) {
    result += fraction * (index % base)
    index = Math.floor(index / base)
    fraction /= base
  }
  return result
}

function gaussLegendre(n) {
  const nodes = new Array(n)
  const weights = new Array(n)
  const m = Math.floor((n + 1) / 2)
  const eps = 1e-12
  for (let i = 0; i < m; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5))
    let zPrev
    let p1 = 1
    let p2 = 0
    let pp = 0
    do {
      p1 = 1
      p2 = 0
      for (let j = 1; j <= n; j++) {
        const p3 = p2
        p2 = p1
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
      }
      pp = n * (z * p1 - p2) / (z * z - 1)
      zPrev = z
      z = zPrev - p1 / pp
    } while (Math.abs(z - zPrev) > eps)

    nodes[i] = -z
    nodes[n - 1 - i] = z
    const weight = 2 / ((1 - z * z) * pp * pp)
    weights[i] = weight
    weights[n - 1 - i] = weight
  }
  return { nodes, weights }
}

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
  min-width: 280px;
  flex: 1.2;
}
.toggle-label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.toggle-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.toggle-btn {
  flex: 0 0 auto;
  padding: 4px 12px;
  font-size: 0.82em;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
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
.method-hint {
  margin: -4px 0 14px 0;
  color: var(--vp-c-text-2);
  font-size: 0.86em;
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
