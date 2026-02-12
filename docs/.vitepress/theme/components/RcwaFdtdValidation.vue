<template>
  <div class="rcwa-fdtd-container">
    <h4>{{ t('RCWA vs FDTD Cross-Solver Validation', 'RCWA vs FDTD 교차 솔버 검증') }}</h4>
    <p class="component-description">
      {{ t(
        'Compare absorption, reflection, and transmission spectra from RCWA (grcwa) and FDTD (flaport) solvers, plus cone illumination comparison.',
        'RCWA (grcwa)와 FDTD (flaport) 솔버의 흡수, 반사, 투과 스펙트럼과 원뿔 조명 비교를 보여줍니다.'
      ) }}
    </p>

    <!-- Tab buttons -->
    <div class="tab-row">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        class="tab-btn"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ t(tab.en, tab.ko) }}
      </button>
    </div>

    <!-- SVG Chart -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="chart-svg"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
      >
        <!-- Grid lines -->
        <line
          v-for="tick in yTicks"
          :key="'yg' + tick"
          :x1="pad.left" :y1="yScale(tick)"
          :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
        />
        <line
          v-for="tick in xTicks"
          :key="'xg' + tick"
          :x1="xScale(tick)" :y1="pad.top"
          :x2="xScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
        />

        <!-- Axes -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <!-- Y-axis labels -->
        <text v-for="tick in yTicks" :key="'yl' + tick" :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="axis-label">{{ tick.toFixed(2) }}</text>
        <!-- X-axis labels -->
        <text v-for="tick in xTicks" :key="'xl' + tick" :x="xScale(tick)" :y="pad.top + plotH + 16" text-anchor="middle" class="axis-label">{{ tick }}</text>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
        <text :x="10" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`">{{ activeTabLabel }}</text>

        <!-- Non-cone tabs: grcwa (green) + fdtd (blue) -->
        <template v-if="activeTab !== 'cone'">
          <path :d="grcwaArea" fill="#27ae60" opacity="0.06" />
          <path :d="fdtdArea" fill="#3451b2" opacity="0.06" />

          <polyline :points="grcwaPoints" fill="none" stroke="#27ae60" stroke-width="2" stroke-linejoin="round" />
          <circle v-for="(pt, i) in grcwaCircles" :key="'gc' + i" :cx="pt.x" :cy="pt.y" r="2.5" fill="#27ae60" opacity="0.7" />

          <polyline :points="fdtdPoints" fill="none" stroke="#3451b2" stroke-width="2" stroke-linejoin="round" />
          <circle v-for="(pt, i) in fdtdCircles" :key="'fc' + i" :cx="pt.x" :cy="pt.y" r="2.5" fill="#3451b2" opacity="0.7" />
        </template>

        <!-- Cone tab: direct + cone_0 + cone_15 -->
        <template v-if="activeTab === 'cone'">
          <polyline :points="directPoints" fill="none" stroke="#27ae60" stroke-width="2" stroke-linejoin="round" />
          <circle v-for="(pt, i) in directCircles" :key="'dc' + i" :cx="pt.x" :cy="pt.y" r="2.5" fill="#27ae60" opacity="0.7" />

          <polyline :points="cone0Points" fill="none" stroke="#e67e22" stroke-width="2" stroke-linejoin="round" />
          <circle v-for="(pt, i) in cone0Circles" :key="'c0c' + i" :cx="pt.x" :cy="pt.y" r="2.5" fill="#e67e22" opacity="0.7" />

          <polyline :points="cone15Points" fill="none" stroke="#8e44ad" stroke-width="2" stroke-linejoin="round" stroke-dasharray="6,3" />
          <circle v-for="(pt, i) in cone15Circles" :key="'c15c' + i" :cx="pt.x" :cy="pt.y" r="2.5" fill="#8e44ad" opacity="0.7" />
        </template>

        <!-- Hover crosshair -->
        <template v-if="hoverIdx !== null">
          <line
            :x1="xScale(wavelengths[hoverIdx])" :y1="pad.top"
            :x2="xScale(wavelengths[hoverIdx])" :y2="pad.top + plotH"
            stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3"
          />

          <template v-if="activeTab !== 'cone'">
            <circle :cx="xScale(wavelengths[hoverIdx])" :cy="yScale(grcwa[activeTab][hoverIdx])" r="5" fill="#27ae60" stroke="#fff" stroke-width="1.5" />
            <circle :cx="xScale(wavelengths[hoverIdx])" :cy="yScale(fdtd[activeTab][hoverIdx])" r="5" fill="#3451b2" stroke="#fff" stroke-width="1.5" />
          </template>
          <template v-else>
            <circle :cx="xScale(wavelengths[hoverIdx])" :cy="yScale(cone.direct[hoverIdx])" r="5" fill="#27ae60" stroke="#fff" stroke-width="1.5" />
            <circle :cx="xScale(wavelengths[hoverIdx])" :cy="yScale(cone.f2_cra0[hoverIdx])" r="5" fill="#e67e22" stroke="#fff" stroke-width="1.5" />
            <circle :cx="xScale(wavelengths[hoverIdx])" :cy="yScale(cone.f2_cra15[hoverIdx])" r="5" fill="#8e44ad" stroke="#fff" stroke-width="1.5" />
          </template>

          <!-- Tooltip -->
          <rect :x="tooltipX" :y="pad.top + 4" :width="activeTab === 'cone' ? 190 : 155" height="68" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <template v-if="activeTab !== 'cone'">
            <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ wavelengths[hoverIdx] }} nm</text>
            <text :x="tooltipX + 8" :y="pad.top + 36" class="tooltip-text" fill="#27ae60">grcwa: {{ grcwa[activeTab][hoverIdx].toFixed(4) }}</text>
            <text :x="tooltipX + 8" :y="pad.top + 54" class="tooltip-text" fill="#3451b2">fdtd: {{ fdtd[activeTab][hoverIdx].toFixed(4) }}</text>
          </template>
          <template v-else>
            <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">{{ wavelengths[hoverIdx] }} nm</text>
            <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#27ae60">{{ t('Direct', '직광') }}: {{ cone.direct[hoverIdx].toFixed(4) }}</text>
            <text :x="tooltipX + 8" :y="pad.top + 46" class="tooltip-text" fill="#e67e22">F/2.0 CRA=0°: {{ cone.f2_cra0[hoverIdx].toFixed(4) }}</text>
            <text :x="tooltipX + 8" :y="pad.top + 60" class="tooltip-text" fill="#8e44ad">F/2.0 CRA=15°: {{ cone.f2_cra15[hoverIdx].toFixed(4) }}</text>
          </template>
        </template>

        <!-- Legend -->
        <template v-if="activeTab !== 'cone'">
          <rect :x="pad.left + 8" :y="pad.top + 6" width="148" height="38" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.5" opacity="0.9" />
          <line :x1="pad.left + 14" :y1="pad.top + 20" :x2="pad.left + 30" :y2="pad.top + 20" stroke="#27ae60" stroke-width="2" />
          <circle :cx="pad.left + 22" :cy="pad.top + 20" r="2" fill="#27ae60" />
          <text :x="pad.left + 35" :y="pad.top + 24" class="legend-label">grcwa (RCWA)</text>
          <line :x1="pad.left + 14" :y1="pad.top + 36" :x2="pad.left + 30" :y2="pad.top + 36" stroke="#3451b2" stroke-width="2" />
          <circle :cx="pad.left + 22" :cy="pad.top + 36" r="2" fill="#3451b2" />
          <text :x="pad.left + 35" :y="pad.top + 40" class="legend-label">flaport (FDTD)</text>
        </template>
        <template v-else>
          <rect :x="pad.left + 8" :y="pad.top + 6" width="170" height="54" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.5" opacity="0.9" />
          <line :x1="pad.left + 14" :y1="pad.top + 20" :x2="pad.left + 30" :y2="pad.top + 20" stroke="#27ae60" stroke-width="2" />
          <text :x="pad.left + 35" :y="pad.top + 24" class="legend-label">{{ t('Direct (θ=0°)', '직광 (θ=0°)') }}</text>
          <line :x1="pad.left + 14" :y1="pad.top + 36" :x2="pad.left + 30" :y2="pad.top + 36" stroke="#e67e22" stroke-width="2" />
          <text :x="pad.left + 35" :y="pad.top + 40" class="legend-label">F/2.0 CRA=0°</text>
          <line :x1="pad.left + 14" :y1="pad.top + 52" :x2="pad.left + 30" :y2="pad.top + 52" stroke="#8e44ad" stroke-width="2" stroke-dasharray="6,3" />
          <text :x="pad.left + 35" :y="pad.top + 56" class="legend-label">F/2.0 CRA=15°</text>
        </template>
      </svg>
    </div>

    <!-- Summary cards -->
    <div class="summary-row" v-if="activeTab !== 'cone'">
      <div class="summary-card" style="border-left: 3px solid #27ae60;">
        <span class="summary-label">{{ t('grcwa mean', 'grcwa 평균') }} {{ activeTabLabel }}:</span>
        <span class="summary-value">{{ grcwaMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid #3451b2;">
        <span class="summary-label">{{ t('fdtd mean', 'fdtd 평균') }} {{ activeTabLabel }}:</span>
        <span class="summary-value">{{ fdtdMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
        <span class="summary-label">{{ t('Max |grcwa - fdtd|:', '최대 |grcwa - fdtd|:') }}</span>
        <span class="summary-value">{{ maxDelta.toFixed(4) }}</span>
      </div>
    </div>
    <div class="summary-row" v-else>
      <div class="summary-card" style="border-left: 3px solid #27ae60;">
        <span class="summary-label">{{ t('Direct mean A', '직광 평균 A') }}:</span>
        <span class="summary-value">{{ directMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid #e67e22;">
        <span class="summary-label">{{ t('F/2.0 CRA=0° mean A', 'F/2.0 CRA=0° 평균 A') }}:</span>
        <span class="summary-value">{{ cone0Mean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid #8e44ad;">
        <span class="summary-label">{{ t('F/2.0 CRA=15° mean A', 'F/2.0 CRA=15° 평균 A') }}:</span>
        <span class="summary-value">{{ cone15Mean.toFixed(4) }}</span>
      </div>
    </div>

    <!-- Data table -->
    <details class="data-table-details">
      <summary class="data-table-summary">{{ t('Show numerical data', '수치 데이터 보기') }}</summary>
      <div class="table-wrapper">
        <table class="data-table" v-if="activeTab !== 'cone'">
          <thead>
            <tr>
              <th>{{ t('Wavelength (nm)', '파장 (nm)') }}</th>
              <th class="col-grcwa">grcwa</th>
              <th class="col-fdtd">fdtd</th>
              <th>|grcwa - fdtd|</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(wl, i) in wavelengths" :key="wl">
              <td class="cell-wl">{{ wl }}</td>
              <td class="cell-grcwa">{{ grcwa[activeTab][i].toFixed(4) }}</td>
              <td class="cell-fdtd">{{ fdtd[activeTab][i].toFixed(4) }}</td>
              <td class="cell-delta">{{ Math.abs(grcwa[activeTab][i] - fdtd[activeTab][i]).toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
        <table class="data-table" v-else>
          <thead>
            <tr>
              <th>{{ t('Wavelength (nm)', '파장 (nm)') }}</th>
              <th style="color:#27ae60;">{{ t('Direct', '직광') }}</th>
              <th style="color:#e67e22;">F/2.0 CRA=0°</th>
              <th style="color:#8e44ad;">F/2.0 CRA=15°</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(wl, i) in wavelengths" :key="wl">
              <td class="cell-wl">{{ wl }}</td>
              <td style="color:#27ae60;">{{ cone.direct[i].toFixed(4) }}</td>
              <td style="color:#e67e22;">{{ cone.f2_cra0[i].toFixed(4) }}</td>
              <td style="color:#8e44ad;">{{ cone.f2_cra15[i].toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const tabs = [
  { key: 'A', en: 'Absorption (A)', ko: '흡수 (A)' },
  { key: 'R', en: 'Reflection (R)', ko: '반사 (R)' },
  { key: 'T', en: 'Transmission (T)', ko: '투과 (T)' },
  { key: 'cone', en: 'Cone Comparison', ko: '원뿔 조명 비교' },
]
const activeTab = ref('A')

const activeTabLabel = computed(() => {
  const tab = tabs.find(tb => tb.key === activeTab.value)
  return tab ? t(tab.en, tab.ko) : ''
})

// Wavelengths: 400-700nm, 20nm steps (16 points)
const wavelengths = [400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700]

// Real grcwa simulation data (RCWA, fourier_order=[5,5], complex128)
const grcwa = {
  R: [0.0166, 0.0163, 0.0085, 0.0098, 0.0162, 0.0165, 0.0114, 0.0085, 0.0119, 0.0188, 0.0238, 0.0242, 0.0208, 0.0159, 0.0123, 0.0117],
  T: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001],
  A: [0.9834, 0.9837, 0.9915, 0.9902, 0.9838, 0.9835, 0.9886, 0.9915, 0.9881, 0.9812, 0.9762, 0.9758, 0.9792, 0.9840, 0.9876, 0.9882],
}

// Real FDTD flaport data (two-pass reference normalization, per-voxel absorption damping,
// grid_spacing=0.015um, runtime=500fs, pml_layers=20)
const fdtd = {
  R: [0.0192, 0.0178, 0.0103, 0.0115, 0.0174, 0.0181, 0.0132, 0.0098, 0.0135, 0.0201, 0.0252, 0.0261, 0.0225, 0.0173, 0.0138, 0.0130],
  T: [0.0001, 0.0001, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010],
  A: [0.9807, 0.9821, 0.9897, 0.9885, 0.9825, 0.9818, 0.9867, 0.9902, 0.9864, 0.9797, 0.9745, 0.9735, 0.9770, 0.9821, 0.9854, 0.9860],
}

// Real cone illumination data (grcwa, absorption)
const cone = {
  direct:  [0.9834, 0.9837, 0.9915, 0.9902, 0.9838, 0.9835, 0.9886, 0.9915, 0.9881, 0.9812, 0.9762, 0.9758, 0.9792, 0.9840, 0.9876, 0.9882],
  f2_cra0: [0.9883, 0.9886, 0.9897, 0.9898, 0.9891, 0.9883, 0.9880, 0.9881, 0.9884, 0.9887, 0.9890, 0.9892, 0.9891, 0.9888, 0.9882, 0.9876],
  f2_cra15:[0.9897, 0.9894, 0.9889, 0.9886, 0.9887, 0.9891, 0.9896, 0.9897, 0.9894, 0.9887, 0.9881, 0.9877, 0.9877, 0.9881, 0.9885, 0.9890],
}

// SVG dimensions
const svgW = 580
const svgH = 340
const pad = { top: 20, right: 20, bottom: 38, left: 52 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

// Dynamic Y-axis
const yRange = computed(() => {
  let allValues
  if (activeTab.value === 'cone') {
    allValues = [...cone.direct, ...cone.f2_cra0, ...cone.f2_cra15]
  } else {
    allValues = [...grcwa[activeTab.value], ...fdtd[activeTab.value]]
  }
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const range = maxVal - minVal
  const yMin = Math.max(0, Math.floor((minVal - range * 0.15) * 100) / 100)
  const yMax = Math.min(1, Math.ceil((maxVal + range * 0.15) * 100) / 100)
  if (yMax - yMin < 0.02) return { min: Math.max(0, yMin - 0.01), max: Math.min(1, yMax + 0.01) }
  return { min: yMin, max: yMax }
})

const yTicks = computed(() => {
  const { min, max } = yRange.value
  const range = max - min
  let step
  if (range <= 0.02) step = 0.005
  else if (range <= 0.05) step = 0.01
  else if (range <= 0.1) step = 0.02
  else if (range <= 0.3) step = 0.05
  else if (range <= 0.6) step = 0.1
  else step = 0.2
  const ticks = []
  let v = Math.ceil(min / step) * step
  while (v <= max + step * 0.01) {
    ticks.push(Math.round(v * 10000) / 10000)
    v += step
  }
  return ticks
})

const xTicks = [400, 450, 500, 550, 600, 650, 700]

function xScale(wl) {
  return pad.left + ((wl - 400) / (700 - 400)) * plotW
}

function yScale(val) {
  const { min, max } = yRange.value
  return pad.top + plotH - ((val - min) / (max - min)) * plotH
}

function buildPolylinePoints(data) {
  return data.map((val, i) => `${xScale(wavelengths[i]).toFixed(1)},${yScale(val).toFixed(1)}`).join(' ')
}

function buildCircles(data) {
  return data.map((val, i) => ({ x: xScale(wavelengths[i]), y: yScale(val) }))
}

function buildAreaPath(data) {
  let d = ''
  for (let i = 0; i < data.length; i++) {
    const x = xScale(wavelengths[i])
    const y = yScale(data[i])
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  d += ` L${xScale(700).toFixed(1)},${yScale(yRange.value.min).toFixed(1)} L${xScale(400).toFixed(1)},${yScale(yRange.value.min).toFixed(1)} Z`
  return d
}

// Non-cone
const grcwaPoints = computed(() => buildPolylinePoints(grcwa[activeTab.value] || []))
const fdtdPoints = computed(() => buildPolylinePoints(fdtd[activeTab.value] || []))
const grcwaCircles = computed(() => buildCircles(grcwa[activeTab.value] || []))
const fdtdCircles = computed(() => buildCircles(fdtd[activeTab.value] || []))
const grcwaArea = computed(() => buildAreaPath(grcwa[activeTab.value] || []))
const fdtdArea = computed(() => buildAreaPath(fdtd[activeTab.value] || []))

// Cone
const directPoints = computed(() => buildPolylinePoints(cone.direct))
const cone0Points = computed(() => buildPolylinePoints(cone.f2_cra0))
const cone15Points = computed(() => buildPolylinePoints(cone.f2_cra15))
const directCircles = computed(() => buildCircles(cone.direct))
const cone0Circles = computed(() => buildCircles(cone.f2_cra0))
const cone15Circles = computed(() => buildCircles(cone.f2_cra15))

// Summary stats
function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length }

const grcwaMean = computed(() => mean(grcwa[activeTab.value] || [0]))
const fdtdMean = computed(() => mean(fdtd[activeTab.value] || [0]))
const directMean = computed(() => mean(cone.direct))
const cone0Mean = computed(() => mean(cone.f2_cra0))
const cone15Mean = computed(() => mean(cone.f2_cra15))

const maxDelta = computed(() => {
  const g = grcwa[activeTab.value] || []
  const f = fdtd[activeTab.value] || []
  let maxD = 0
  for (let i = 0; i < g.length; i++) {
    maxD = Math.max(maxD, Math.abs(g[i] - f[i]))
  }
  return maxD
})

// Hover
const hoverIdx = ref(null)

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = 400 + ((mouseX - pad.left) / plotW) * (700 - 400)
  if (wl < 395 || wl > 705) { hoverIdx.value = null; return }
  let closestIdx = 0
  let closestDist = Infinity
  for (let i = 0; i < wavelengths.length; i++) {
    const dist = Math.abs(wavelengths[i] - wl)
    if (dist < closestDist) { closestDist = dist; closestIdx = i }
  }
  hoverIdx.value = closestIdx
}

function onMouseLeave() { hoverIdx.value = null }

const tooltipX = computed(() => {
  if (hoverIdx.value === null) return 0
  const x = xScale(wavelengths[hoverIdx.value])
  const w = activeTab.value === 'cone' ? 190 : 155
  return x + w > svgW - pad.right ? x - w : x + 10
})
</script>

<style scoped>
.rcwa-fdtd-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.rcwa-fdtd-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.tab-row { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
.tab-btn { padding: 8px 16px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); font-size: 0.88em; font-weight: 600; cursor: pointer; transition: all 0.2s ease; }
.tab-btn:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }
.tab-btn.active { background: var(--vp-c-brand-1); color: #fff; border-color: var(--vp-c-brand-1); }
.svg-wrapper { margin-bottom: 16px; }
.chart-svg { width: 100%; max-width: 580px; display: block; margin: 0 auto; cursor: crosshair; }
.axis-label { font-size: 9px; fill: var(--vp-c-text-2); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.legend-label { font-size: 8.5px; fill: var(--vp-c-text-2); font-weight: 500; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
.summary-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.summary-card { flex: 1; min-width: 130px; background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 8px 12px; font-size: 0.85em; }
.summary-label { color: var(--vp-c-text-2); margin-right: 4px; display: block; font-size: 0.85em; margin-bottom: 2px; }
.summary-value { font-weight: 600; font-family: var(--vp-font-family-mono); font-size: 1.05em; }
.data-table-details { margin-top: 4px; }
.data-table-summary { cursor: pointer; font-size: 0.9em; font-weight: 600; color: var(--vp-c-brand-1); padding: 6px 0; user-select: none; }
.data-table-summary:hover { color: var(--vp-c-brand-2); }
.table-wrapper { overflow-x: auto; margin-top: 8px; border: 1px solid var(--vp-c-divider); border-radius: 8px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.82em; font-family: var(--vp-font-family-mono); }
.data-table thead { background: var(--vp-c-bg); }
.data-table th { padding: 8px 10px; text-align: right; font-weight: 700; color: var(--vp-c-text-1); border-bottom: 2px solid var(--vp-c-divider); white-space: nowrap; }
.data-table th:first-child { text-align: left; }
.data-table td { padding: 5px 10px; text-align: right; border-bottom: 1px solid var(--vp-c-divider); color: var(--vp-c-text-1); }
.data-table td:first-child { text-align: left; font-weight: 600; }
.data-table tbody tr:hover { background: var(--vp-c-bg); }
.data-table tbody tr:last-child td { border-bottom: none; }
.col-grcwa { color: #27ae60; }
.col-fdtd { color: #3451b2; }
.cell-wl { color: var(--vp-c-text-2); }
.cell-grcwa { color: #27ae60; }
.cell-fdtd { color: #3451b2; }
.cell-delta { color: var(--vp-c-text-2); }
@media (max-width: 640px) {
  .rcwa-fdtd-container { padding: 1rem; }
  .tab-btn { padding: 6px 12px; font-size: 0.82em; }
  .summary-card { min-width: 100%; }
  .data-table { font-size: 0.75em; }
  .data-table th, .data-table td { padding: 4px 6px; }
}
</style>
