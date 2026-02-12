<template>
  <div class="solver-cmp-container">
    <h4>{{ t('RCWA vs FDTD Solver Comparison', 'RCWA vs FDTD 솔버 비교') }}</h4>
    <p class="component-description">
      {{ t(
        'Compare simulated quantum efficiency (QE) curves from RCWA and FDTD solvers. Adjust pixel pitch and solver parameters to see how results and performance change.',
        'RCWA와 FDTD 솔버의 양자 효율(QE) 시뮬레이션 곡선을 비교합니다. 픽셀 피치와 솔버 매개변수를 조정하여 결과와 성능 변화를 확인하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label for="sc-pitch">{{ t('Pixel pitch:', '픽셀 피치:') }}</label>
        <select id="sc-pitch" v-model.number="pitch" class="param-select">
          <option v-for="p in pitchOptions" :key="p" :value="p">{{ p }} um</option>
        </select>
      </div>
      <div class="slider-group">
        <label>{{ t('RCWA Fourier order:', 'RCWA Fourier 차수:') }} <strong>{{ fourierOrder }}</strong></label>
        <select v-model.number="fourierOrder" class="param-select">
          <option v-for="o in fourierOptions" :key="o" :value="o">{{ o }}</option>
        </select>
      </div>
      <div class="slider-group">
        <label>{{ t('FDTD grid resolution:', 'FDTD 격자 해상도:') }} <strong>{{ fdtdGrid }} nm</strong></label>
        <select v-model.number="fdtdGrid" class="param-select">
          <option v-for="g in gridOptions" :key="g" :value="g">{{ g }} nm</option>
        </select>
      </div>
    </div>

    <div class="plots-layout">
      <!-- RCWA panel -->
      <div class="plot-section">
        <div class="plot-title">RCWA (Fourier order = {{ fourierOrder }})</div>
        <svg :viewBox="`0 0 ${pW} ${pH}`" class="plot-svg">
          <!-- Grid and axes -->
          <line :x1="pL" :y1="pT" :x2="pL" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line :x1="pL" :y1="pB" :x2="pL + ppW" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />

          <!-- Y-axis ticks -->
          <template v-for="v in [0, 20, 40, 60, 80, 100]" :key="'ry' + v">
            <line :x1="pL - 3" :y1="qeToY(v)" :x2="pL" :y2="qeToY(v)" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="pL - 6" :y="qeToY(v) + 3" text-anchor="end" class="tick-label">{{ v }}%</text>
            <line :x1="pL" :y1="qeToY(v)" :x2="pL + ppW" :y2="qeToY(v)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="2,2" />
          </template>

          <!-- X-axis ticks -->
          <template v-for="wl in [400, 500, 600, 700]" :key="'rx' + wl">
            <line :x1="wlToX(wl)" :y1="pB" :x2="wlToX(wl)" :y2="pB + 3" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="wlToX(wl)" :y="pB + 14" text-anchor="middle" class="tick-label">{{ wl }}</text>
          </template>

          <text :x="pL + ppW / 2" :y="pB + 26" text-anchor="middle" class="axis-label-small">Wavelength (nm)</text>
          <text :x="8" :y="pT + ppH / 2" text-anchor="middle" :transform="`rotate(-90, 8, ${pT + ppH / 2})`" class="axis-label-small">QE (%)</text>

          <!-- QE curves -->
          <path :d="rcwaRedPath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <path :d="rcwaGreenPath" fill="none" stroke="#27ae60" stroke-width="2" />
          <path :d="rcwaBluePath" fill="none" stroke="#2980b9" stroke-width="2" />

          <!-- Legend -->
          <line :x1="pL + ppW - 70" :y1="pT + 10" :x2="pL + ppW - 55" :y2="pT + 10" stroke="#e74c3c" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 14" class="legend-label">Red</text>
          <line :x1="pL + ppW - 70" :y1="pT + 23" :x2="pL + ppW - 55" :y2="pT + 23" stroke="#27ae60" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 27" class="legend-label">Green</text>
          <line :x1="pL + ppW - 70" :y1="pT + 36" :x2="pL + ppW - 55" :y2="pT + 36" stroke="#2980b9" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 40" class="legend-label">Blue</text>
        </svg>
      </div>

      <!-- FDTD panel -->
      <div class="plot-section">
        <div class="plot-title">FDTD (grid = {{ fdtdGrid }} nm)</div>
        <svg :viewBox="`0 0 ${pW} ${pH}`" class="plot-svg">
          <!-- Grid and axes -->
          <line :x1="pL" :y1="pT" :x2="pL" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line :x1="pL" :y1="pB" :x2="pL + ppW" :y2="pB" stroke="var(--vp-c-text-3)" stroke-width="1" />

          <!-- Y-axis ticks -->
          <template v-for="v in [0, 20, 40, 60, 80, 100]" :key="'fy' + v">
            <line :x1="pL - 3" :y1="qeToY(v)" :x2="pL" :y2="qeToY(v)" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="pL - 6" :y="qeToY(v) + 3" text-anchor="end" class="tick-label">{{ v }}%</text>
            <line :x1="pL" :y1="qeToY(v)" :x2="pL + ppW" :y2="qeToY(v)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="2,2" />
          </template>

          <!-- X-axis ticks -->
          <template v-for="wl in [400, 500, 600, 700]" :key="'fx' + wl">
            <line :x1="wlToX(wl)" :y1="pB" :x2="wlToX(wl)" :y2="pB + 3" stroke="var(--vp-c-text-3)" stroke-width="1" />
            <text :x="wlToX(wl)" :y="pB + 14" text-anchor="middle" class="tick-label">{{ wl }}</text>
          </template>

          <text :x="pL + ppW / 2" :y="pB + 26" text-anchor="middle" class="axis-label-small">Wavelength (nm)</text>
          <text :x="8" :y="pT + ppH / 2" text-anchor="middle" :transform="`rotate(-90, 8, ${pT + ppH / 2})`" class="axis-label-small">QE (%)</text>

          <!-- QE curves -->
          <path :d="fdtdRedPath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <path :d="fdtdGreenPath" fill="none" stroke="#27ae60" stroke-width="2" />
          <path :d="fdtdBluePath" fill="none" stroke="#2980b9" stroke-width="2" />

          <!-- Legend -->
          <line :x1="pL + ppW - 70" :y1="pT + 10" :x2="pL + ppW - 55" :y2="pT + 10" stroke="#e74c3c" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 14" class="legend-label">Red</text>
          <line :x1="pL + ppW - 70" :y1="pT + 23" :x2="pL + ppW - 55" :y2="pT + 23" stroke="#27ae60" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 27" class="legend-label">Green</text>
          <line :x1="pL + ppW - 70" :y1="pT + 36" :x2="pL + ppW - 55" :y2="pT + 36" stroke="#2980b9" stroke-width="2" />
          <text :x="pL + ppW - 50" :y="pT + 40" class="legend-label">Blue</text>
        </svg>
      </div>
    </div>

    <div class="comparison-cards">
      <div class="cmp-card">
        <div class="cmp-title">RCWA</div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Time estimate:', '예상 시간:') }}</span>
          <span class="cmp-value">{{ rcwaTime }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Memory:', '메모리:') }}</span>
          <span class="cmp-value">{{ rcwaMemory }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Periodic structures:', '주기 구조:') }}</span>
          <span class="cmp-value check">{{ t('Yes', '예') }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Arbitrary geometry:', '임의 형상:') }}</span>
          <span class="cmp-value cross">{{ t('Limited', '제한적') }}</span>
        </div>
      </div>

      <div class="cmp-card">
        <div class="cmp-title">FDTD</div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Time estimate:', '예상 시간:') }}</span>
          <span class="cmp-value">{{ fdtdTime }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Memory:', '메모리:') }}</span>
          <span class="cmp-value">{{ fdtdMemory }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Periodic structures:', '주기 구조:') }}</span>
          <span class="cmp-value check">{{ t('Yes', '예') }}</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Arbitrary geometry:', '임의 형상:') }}</span>
          <span class="cmp-value check">{{ t('Yes', '예') }}</span>
        </div>
      </div>

      <div class="cmp-card agreement-card">
        <div class="cmp-title">{{ t('Agreement', '일치도') }}</div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Max |Delta QE|:', '최대 |Delta QE|:') }}</span>
          <span class="cmp-value" :class="{ warn: maxDeltaQE > 5 }">{{ maxDeltaQE.toFixed(1) }}%</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Avg |Delta QE|:', '평균 |Delta QE|:') }}</span>
          <span class="cmp-value">{{ avgDeltaQE.toFixed(1) }}%</span>
        </div>
        <div class="cmp-row">
          <span class="cmp-label">{{ t('Status:', '상태:') }}</span>
          <span class="cmp-value" :class="maxDeltaQE <= 3 ? 'check' : 'warn'">{{ maxDeltaQE <= 3 ? t('Good agreement', '양호한 일치') : t('Check parameters', '매개변수 확인 필요') }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// Plot dimensions
const pW = 460
const pH = 260
const pL = 42
const pR = 15
const pT = 20
const pB = pH - 35
const ppW = pW - pL - pR
const ppH = pB - pT

const pitchOptions = [0.7, 0.9, 1.0, 1.2, 1.4]
const fourierOptions = [3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 201, 501, 1000]
const gridOptions = [10, 20, 30, 50]

const pitch = ref(1.0)
const fourierOrder = ref(9)
const fdtdGrid = ref(20)

// Wavelength points
const wls = []
for (let w = 380; w <= 780; w += 5) wls.push(w)

// Generate Gaussian-like QE peak
function gaussianQE(wl, center, sigma, peakQE, baseline) {
  const x = (wl - center) / sigma
  return baseline + (peakQE - baseline) * Math.exp(-0.5 * x * x)
}

// Pitch-dependent parameters
function getPitchFactor(p) {
  // Larger pitch -> higher QE (less crosstalk)
  return 0.7 + (p - 0.7) * 0.43
}

// RCWA QE model
function rcwaQE(wl, channel) {
  const pf = getPitchFactor(pitch.value)
  // Fourier order effect: lower order -> slightly shifted/broader peaks (converged at ~15)
  const orderNoise = Math.max(0, (15 - fourierOrder.value)) * 0.3
  const orderShift = Math.max(0, (15 - fourierOrder.value)) * 0.2

  switch (channel) {
    case 'red':
      return gaussianQE(wl, 620 + orderShift, 55 + orderNoise, 72 * pf, 3 + orderNoise * 0.3)
    case 'green':
      return gaussianQE(wl, 530 + orderShift * 0.5, 48 + orderNoise, 78 * pf, 4 + orderNoise * 0.2)
    case 'blue':
      return gaussianQE(wl, 460 + orderShift * 0.3, 38 + orderNoise, 55 * pf, 2 + orderNoise * 0.4)
    default:
      return 0
  }
}

// FDTD QE model (with per-voxel absorption and two-pass normalization, close RCWA agreement)
function fdtdQE(wl, channel) {
  const pf = getPitchFactor(pitch.value)
  // Coarser grid -> small numerical dispersion -> minor shifts
  const gridNoise = (fdtdGrid.value - 10) * 0.02
  const gridShift = (fdtdGrid.value - 10) * 0.05

  // Minor ripple from finite grid discretization
  const ripple = gridNoise * Math.sin(wl * 0.1 + fdtdGrid.value) * 0.3

  switch (channel) {
    case 'red':
      return gaussianQE(wl, 620 - gridShift * 0.3, 55.5 + gridNoise * 0.3, 71 * pf + ripple, 3.2 + gridNoise * 0.1)
    case 'green':
      return gaussianQE(wl, 530 - gridShift * 0.2, 48.5 + gridNoise * 0.2, 77 * pf + ripple, 4.1 + gridNoise * 0.08)
    case 'blue':
      return gaussianQE(wl, 460 - gridShift * 0.1, 38.5 + gridNoise * 0.15, 54.5 * pf + ripple, 2.2 + gridNoise * 0.15)
    default:
      return 0
  }
}

function qeToY(v) {
  return pB - (v / 100) * ppH
}

function wlToX(wl) {
  return pL + ((wl - 380) / (780 - 380)) * ppW
}

function buildPath(qeFn, channel) {
  let d = ''
  for (let i = 0; i < wls.length; i++) {
    const x = wlToX(wls[i])
    const y = qeToY(Math.max(0, Math.min(100, qeFn(wls[i], channel))))
    d += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
}

const rcwaRedPath = computed(() => buildPath(rcwaQE, 'red'))
const rcwaGreenPath = computed(() => buildPath(rcwaQE, 'green'))
const rcwaBluePath = computed(() => buildPath(rcwaQE, 'blue'))

const fdtdRedPath = computed(() => buildPath(fdtdQE, 'red'))
const fdtdGreenPath = computed(() => buildPath(fdtdQE, 'green'))
const fdtdBluePath = computed(() => buildPath(fdtdQE, 'blue'))

// Max delta QE
const maxDeltaQE = computed(() => {
  let maxD = 0
  for (const wl of wls) {
    for (const ch of ['red', 'green', 'blue']) {
      const diff = Math.abs(rcwaQE(wl, ch) - fdtdQE(wl, ch))
      if (diff > maxD) maxD = diff
    }
  }
  return maxD
})

const avgDeltaQE = computed(() => {
  let sum = 0
  let count = 0
  for (const wl of wls) {
    for (const ch of ['red', 'green', 'blue']) {
      sum += Math.abs(rcwaQE(wl, ch) - fdtdQE(wl, ch))
      count++
    }
  }
  return sum / count
})

// Performance estimates
const rcwaTime = computed(() => {
  const M = 2 * fourierOrder.value + 1
  const baseTime = (M * M * M) / 1000 * 0.02 * (pitch.value / 1.0)
  if (baseTime < 1) return `${(baseTime * 1000).toFixed(0)} ms`
  if (baseTime < 60) return `${baseTime.toFixed(1)} s`
  if (baseTime < 3600) return `${(baseTime / 60).toFixed(1)} min`
  return `${(baseTime / 3600).toFixed(1)} hr`
})

const rcwaMemory = computed(() => {
  const M = 2 * fourierOrder.value + 1
  const mem = M * M * 16 / 1024 // MB estimate
  if (mem < 1) return `${(mem * 1024).toFixed(0)} KB`
  if (mem < 1024) return `${mem.toFixed(0)} MB`
  return `${(mem / 1024).toFixed(1)} GB`
})

const fdtdTime = computed(() => {
  const pitchNm = pitch.value * 1000
  const cells = (pitchNm / fdtdGrid.value)
  const totalCells = cells * cells * cells * 3 // 3D with z extent
  const baseTime = totalCells / 1e6 * 0.5
  if (baseTime < 1) return `${(baseTime * 1000).toFixed(0)} ms`
  if (baseTime < 60) return `${baseTime.toFixed(1)} s`
  return `${(baseTime / 60).toFixed(1)} min`
})

const fdtdMemory = computed(() => {
  const pitchNm = pitch.value * 1000
  const cells = pitchNm / fdtdGrid.value
  const totalCells = cells * cells * cells * 3
  const mem = totalCells * 8 / (1024 * 1024) // bytes to MB (8 bytes per float64)
  if (mem < 1) return `${(mem * 1024).toFixed(0)} KB`
  if (mem < 1024) return `${mem.toFixed(0)} MB`
  return `${(mem / 1024).toFixed(1)} GB`
})
</script>

<style scoped>
.solver-cmp-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.solver-cmp-container h4 {
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
  align-items: flex-end;
  margin-bottom: 16px;
}
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.9em;
  font-weight: 600;
}
.param-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.slider-group {
  flex: 1;
  min-width: 180px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.plots-layout {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
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
  max-width: 500px;
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
  font-weight: 600;
}
.comparison-cards {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
.cmp-card {
  flex: 1;
  min-width: 200px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 14px;
}
.agreement-card {
  border-color: var(--vp-c-brand-1);
}
.cmp-title {
  font-size: 0.95em;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  margin-bottom: 10px;
}
.cmp-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  font-size: 0.85em;
}
.cmp-row:last-child {
  margin-bottom: 0;
}
.cmp-label {
  color: var(--vp-c-text-2);
}
.cmp-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
}
.cmp-value.check {
  color: #27ae60;
}
.cmp-value.cross {
  color: #f39c12;
}
.cmp-value.warn {
  color: #e74c3c;
}
</style>
