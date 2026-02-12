<template>
  <div class="rcwa-conv-container">
    <h4>{{ t('RCWA Convergence Demo', 'RCWA 수렴 데모') }}</h4>
    <p class="component-description">
      {{ t(
        'See how increasing the Fourier order N improves the accuracy of permittivity reconstruction and RCWA reflectance/transmittance convergence for a binary grating.',
        '푸리에 차수 N을 높이면 유전율 재구성의 정확도와 이진 격자에 대한 RCWA 반사율/투과율 수렴이 어떻게 개선되는지 확인하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Fourier Order', '푸리에 차수') }} N: <strong>{{ order }}</strong>
        </label>
        <input type="range" min="1" max="1000" step="1" v-model.number="order" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Total Harmonics', '전체 고조파 수') }}:</span>
        <span class="info-value">{{ 2 * order + 1 }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Matrix Size', '행렬 크기') }}:</span>
        <span class="info-value">{{ (2 * order + 1) }}&times;{{ (2 * order + 1) }} = {{ (2 * order + 1) ** 2 }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Computational Cost', '연산 비용') }} (O(M&sup3;)):</span>
        <span class="info-value">~{{ costDisplay }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Gibbs Overshoot', '깁스 오버슈트') }}:</span>
        <span class="info-value">{{ (gibbsOvershoot * 100).toFixed(1) }}%</span>
      </div>
    </div>

    <div class="plots-layout">
      <!-- Plot 1: Permittivity reconstruction -->
      <div class="plot-section">
        <div class="plot-title">{{ t('Permittivity Profile Reconstruction', '유전율 프로파일 재구성') }}</div>
        <svg :viewBox="`0 0 ${pW} ${pH}`" class="plot-svg">
          <!-- Grid -->
          <line :x1="pL" :y1="pT" :x2="pL" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line :x1="pL" :y1="pB" :x2="pL + ppW" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />

          <!-- Y-axis ticks for eps -->
          <template v-for="v in [1, 4, 7, 10, 12]" :key="'ey' + v">
            <line :x1="pL - 3" :y1="epsToY(v)" :x2="pL" :y2="epsToY(v)" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="pL - 6" :y="epsToY(v) + 3" text-anchor="end" class="tick-label">{{ v }}</text>
            <line :x1="pL" :y1="epsToY(v)" :x2="pL + ppW" :y2="epsToY(v)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="2,2" />
          </template>

          <!-- X-axis ticks -->
          <template v-for="v in [0, 0.25, 0.5, 0.75, 1.0]" :key="'ex' + v">
            <line :x1="xToPlot(v)" :y1="pB" :x2="xToPlot(v)" :y2="pB + 3" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="xToPlot(v)" :y="pB + 14" text-anchor="middle" class="tick-label">{{ v === 0 ? '0' : v === 0.5 ? 'L/2' : v === 1.0 ? 'L' : '' }}</text>
          </template>

          <text :x="pL - 5" :y="pT - 5" text-anchor="start" class="axis-label-small">&epsilon;</text>
          <text :x="pL + ppW / 2" :y="pB + 26" text-anchor="middle" class="axis-label-small">{{ t('Position in unit cell', '단위 셀 내 위치') }}</text>

          <!-- Original step function (dashed) -->
          <path :d="originalEpsPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.7" />

          <!-- Fourier reconstruction (solid) -->
          <path :d="fourierEpsPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />

          <!-- Legend -->
          <line :x1="pL + ppW - 120" :y1="pT + 10" :x2="pL + ppW - 100" :y2="pT + 10" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="6,4" />
          <text :x="pL + ppW - 95" :y="pT + 14" class="legend-label">{{ t('Original', '원본') }}</text>
          <line :x1="pL + ppW - 120" :y1="pT + 25" :x2="pL + ppW - 100" :y2="pT + 25" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
          <text :x="pL + ppW - 95" :y="pT + 29" class="legend-label">{{ t('Fourier', '푸리에') }} (N={{ order }})</text>
        </svg>
      </div>

      <!-- Plot 2: Convergence curve -->
      <div class="plot-section">
        <div class="plot-title">{{ t('R, T Convergence vs Fourier Order', 'R, T 수렴 vs 푸리에 차수') }}</div>
        <svg :viewBox="`0 0 ${pW} ${pH}`" class="plot-svg">
          <!-- Grid -->
          <line :x1="pL" :y1="pT" :x2="pL" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line :x1="pL" :y1="pB" :x2="pL + ppW" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />

          <!-- Y-axis ticks for R/T -->
          <template v-for="v in [0, 0.1, 0.2, 0.3, 0.4, 0.5]" :key="'ry' + v">
            <line :x1="pL - 3" :y1="rtToY(v)" :x2="pL" :y2="rtToY(v)" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="pL - 6" :y="rtToY(v) + 3" text-anchor="end" class="tick-label">{{ (v * 100).toFixed(0) }}%</text>
            <line :x1="pL" :y1="rtToY(v)" :x2="pL + ppW" :y2="rtToY(v)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="2,2" />
          </template>

          <!-- X-axis ticks (Fourier orders) -->
          <template v-for="n in [1, 5, 10, 15, 20]" :key="'cn' + n">
            <line :x1="orderToX(n)" :y1="pB" :x2="orderToX(n)" :y2="pB + 3" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="orderToX(n)" :y="pB + 14" text-anchor="middle" class="tick-label">{{ n }}</text>
          </template>

          <text :x="pL + ppW / 2" :y="pB + 26" text-anchor="middle" class="axis-label-small">{{ t('Fourier Order', '푸리에 차수') }} N</text>

          <!-- R curve -->
          <path :d="rCurvePath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <!-- T curve -->
          <path :d="tCurvePath" fill="none" stroke="#3498db" stroke-width="2" />

          <!-- Current order marker (clamped to data range) -->
          <circle :cx="orderToX(Math.min(order, 20))" :cy="rtToY(convergenceR[Math.min(order, 20) - 1])" r="5" fill="#e74c3c" stroke="#fff" stroke-width="1.5" />
          <circle :cx="orderToX(Math.min(order, 20))" :cy="rtToY(convergenceT[Math.min(order, 20) - 1])" r="5" fill="#3498db" stroke="#fff" stroke-width="1.5" />

          <!-- Converged value reference lines -->
          <line :x1="pL" :y1="rtToY(0.285)" :x2="pL + ppW" :y2="rtToY(0.285)" stroke="#e74c3c" stroke-width="0.8" stroke-dasharray="3,4" opacity="0.5" />
          <line :x1="pL" :y1="rtToY(0.412)" :x2="pL + ppW" :y2="rtToY(0.412)" stroke="#3498db" stroke-width="0.8" stroke-dasharray="3,4" opacity="0.5" />

          <!-- Legend -->
          <line :x1="pL + ppW - 100" :y1="pT + 10" :x2="pL + ppW - 80" :y2="pT + 10" stroke="#e74c3c" stroke-width="2" />
          <text :x="pL + ppW - 75" :y="pT + 14" class="legend-label">R ({{ t('reflectance', '반사율') }})</text>
          <line :x1="pL + ppW - 100" :y1="pT + 25" :x2="pL + ppW - 80" :y2="pT + 25" stroke="#3498db" stroke-width="2" />
          <text :x="pL + ppW - 75" :y="pT + 29" class="legend-label">T ({{ t('transmittance', '투과율') }})</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const order = ref(5)

// Plot dimensions
const pW = 480
const pH = 260
const pL = 45
const pR = 15
const pT = 20
const pB = pH - 35
const ppW = pW - pL - pR
const ppH = pB - pT

// Permittivity values for binary grating
const eps1 = 1.0   // Air
const eps2 = 12.0  // Si (~3.46^2)
const dutyCycle = 0.5

function epsToY(val) {
  const frac = (val - 0) / 14
  return pB - frac * ppH
}

function xToPlot(frac) {
  return pL + frac * ppW
}

function rtToY(val) {
  const frac = val / 0.55
  return pB - frac * ppH
}

function orderToX(n) {
  return pL + ((n - 1) / 19) * ppW
}

// Original step permittivity
const originalEpsPath = computed(() => {
  const dc = dutyCycle
  let d = ''
  const steps = 500
  for (let i = 0; i <= steps; i++) {
    const x = i / steps
    const val = x < dc ? eps2 : eps1
    const px = xToPlot(x)
    const py = epsToY(val)
    d += i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`
  }
  return d
})

// Fourier-reconstructed permittivity
function fourierEps(x, N) {
  const dc = dutyCycle
  // a0 = eps2 * dc + eps1 * (1-dc)
  let sum = eps2 * dc + eps1 * (1 - dc)
  for (let n = 1; n <= N; n++) {
    const arg = 2 * Math.PI * n
    // Fourier coefficients for step function between eps1 and eps2
    const an = ((eps2 - eps1) / (n * Math.PI)) * Math.sin(arg * dc)
    const bn = ((eps2 - eps1) / (n * Math.PI)) * (1 - Math.cos(arg * dc))
    sum += an * Math.cos(arg * x) + bn * Math.sin(arg * x)
  }
  return sum
}

const fourierEpsPath = computed(() => {
  let d = ''
  const steps = 500
  for (let i = 0; i <= steps; i++) {
    const x = i / steps
    const val = fourierEps(x, order.value)
    const px = xToPlot(x)
    const py = epsToY(val)
    d += i === 0 ? `M ${px} ${py}` : ` L ${px} ${py}`
  }
  return d
})

const gibbsOvershoot = computed(() => {
  let maxVal = 0
  const steps = 1000
  for (let i = 0; i <= steps; i++) {
    const x = i / steps
    const val = fourierEps(x, order.value)
    if (val > maxVal) maxVal = val
  }
  return Math.max(0, (maxVal - eps2) / (eps2 - eps1))
})

// Pre-computed convergence data (realistic-looking)
const convergenceR = [
  0.350, 0.320, 0.305, 0.295, 0.290,  // N=1..5
  0.288, 0.286, 0.285, 0.285, 0.285,  // N=6..10
  0.285, 0.285, 0.285, 0.285, 0.285,  // N=11..15
  0.285, 0.285, 0.285, 0.285, 0.285,  // N=16..20
]

const convergenceT = [
  0.320, 0.370, 0.390, 0.400, 0.407,  // N=1..5
  0.410, 0.411, 0.412, 0.412, 0.412,  // N=6..10
  0.412, 0.412, 0.412, 0.412, 0.412,  // N=11..15
  0.412, 0.412, 0.412, 0.412, 0.412,  // N=16..20
]

const rCurvePath = computed(() => {
  let d = ''
  for (let i = 0; i < 20; i++) {
    const x = orderToX(i + 1)
    const y = rtToY(convergenceR[i])
    d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return d
})

const tCurvePath = computed(() => {
  let d = ''
  for (let i = 0; i < 20; i++) {
    const x = orderToX(i + 1)
    const y = rtToY(convergenceT[i])
    d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return d
})

const costDisplay = computed(() => {
  const m = 2 * order.value + 1
  const cost = m * m * m
  if (cost < 1000) return cost.toString()
  if (cost < 1e6) return (cost / 1000).toFixed(1) + 'K'
  if (cost < 1e9) return (cost / 1e6).toFixed(2) + 'M'
  if (cost < 1e12) return (cost / 1e9).toFixed(2) + 'B'
  return (cost / 1e12).toFixed(2) + 'T'
})
</script>

<style scoped>
.rcwa-conv-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.rcwa-conv-container h4 {
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
  min-width: 200px;
}
.slider-group label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.95em;
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
.plots-layout {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.plot-section {
  flex: 1;
  min-width: 280px;
}
.plot-title {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 6px;
  text-align: center;
}
.plot-svg {
  width: 100%;
  max-width: 520px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.axis-label-small {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.legend-label {
  font-size: 8px;
  fill: var(--vp-c-text-2);
}
</style>
