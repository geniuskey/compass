<template>
  <div class="cross-solver-container">
    <h4>{{ t('Cross-Solver Validation: TMM vs torcwa vs grcwa', '교차 솔버 검증: TMM vs torcwa vs grcwa') }}</h4>
    <p class="component-description">
      {{ t(
        'Compare absorption, reflection, and transmission spectra from three different electromagnetic solvers. TMM is a fast 1D method, while torcwa and grcwa are full 2D RCWA solvers.',
        '세 가지 전자기 솔버의 흡수, 반사, 투과 스펙트럼을 비교합니다. TMM은 빠른 1D 방법이고, torcwa와 grcwa는 완전한 2D RCWA 솔버입니다.'
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
          :x1="pad.left"
          :y1="yScale(tick)"
          :x2="pad.left + plotW"
          :y2="yScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />
        <line
          v-for="tick in xTicks"
          :key="'xg' + tick"
          :x1="xScale(tick)"
          :y1="pad.top"
          :x2="xScale(tick)"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />

        <!-- Axes -->
        <line
          :x1="pad.left" :y1="pad.top"
          :x2="pad.left" :y2="pad.top + plotH"
          stroke="var(--vp-c-text-2)" stroke-width="1"
        />
        <line
          :x1="pad.left" :y1="pad.top + plotH"
          :x2="pad.left + plotW" :y2="pad.top + plotH"
          stroke="var(--vp-c-text-2)" stroke-width="1"
        />

        <!-- Y-axis labels -->
        <text
          v-for="tick in yTicks"
          :key="'yl' + tick"
          :x="pad.left - 6"
          :y="yScale(tick) + 3"
          text-anchor="end"
          class="axis-label"
        >{{ tick.toFixed(1) }}</text>

        <!-- X-axis labels -->
        <text
          v-for="tick in xTicks"
          :key="'xl' + tick"
          :x="xScale(tick)"
          :y="pad.top + plotH + 16"
          text-anchor="middle"
          class="axis-label"
        >{{ tick }}</text>

        <!-- Axis titles -->
        <text
          :x="pad.left + plotW / 2"
          :y="svgH - 2"
          text-anchor="middle"
          class="axis-title"
        >{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
        <text
          :x="10"
          :y="pad.top + plotH / 2"
          text-anchor="middle"
          class="axis-title"
          :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`"
        >{{ activeTabLabel }}</text>

        <!-- Filled areas (subtle) -->
        <path :d="tmmArea" fill="#3451b2" opacity="0.06" />
        <path :d="torcwaArea" fill="#e67e22" opacity="0.06" />
        <path :d="grcwaArea" fill="#27ae60" opacity="0.06" />

        <!-- TMM polyline -->
        <polyline
          :points="tmmPoints"
          fill="none"
          stroke="#3451b2"
          stroke-width="2"
          stroke-linejoin="round"
        />
        <!-- TMM data circles -->
        <circle
          v-for="(pt, i) in tmmCircles"
          :key="'tc' + i"
          :cx="pt.x"
          :cy="pt.y"
          r="2.5"
          fill="#3451b2"
          opacity="0.7"
        />

        <!-- torcwa polyline -->
        <polyline
          :points="torcwaPoints"
          fill="none"
          stroke="#e67e22"
          stroke-width="2"
          stroke-linejoin="round"
        />
        <!-- torcwa data circles -->
        <circle
          v-for="(pt, i) in torcwaCircles"
          :key="'oc' + i"
          :cx="pt.x"
          :cy="pt.y"
          r="2.5"
          fill="#e67e22"
          opacity="0.7"
        />

        <!-- grcwa polyline -->
        <polyline
          :points="grcwaPoints"
          fill="none"
          stroke="#27ae60"
          stroke-width="2"
          stroke-linejoin="round"
        />
        <!-- grcwa data circles -->
        <circle
          v-for="(pt, i) in grcwaCircles"
          :key="'gc' + i"
          :cx="pt.x"
          :cy="pt.y"
          r="2.5"
          fill="#27ae60"
          opacity="0.7"
        />

        <!-- Hover crosshair -->
        <template v-if="hoverIdx !== null">
          <line
            :x1="xScale(wavelengths[hoverIdx])"
            :y1="pad.top"
            :x2="xScale(wavelengths[hoverIdx])"
            :y2="pad.top + plotH"
            stroke="var(--vp-c-text-2)"
            stroke-width="0.8"
            stroke-dasharray="4,3"
          />
          <circle
            :cx="xScale(wavelengths[hoverIdx])"
            :cy="yScale(tmm[activeTab][hoverIdx])"
            r="5" fill="#3451b2" stroke="#fff" stroke-width="1.5"
          />
          <circle
            :cx="xScale(wavelengths[hoverIdx])"
            :cy="yScale(torcwa[activeTab][hoverIdx])"
            r="5" fill="#e67e22" stroke="#fff" stroke-width="1.5"
          />
          <circle
            :cx="xScale(wavelengths[hoverIdx])"
            :cy="yScale(grcwa[activeTab][hoverIdx])"
            r="5" fill="#27ae60" stroke="#fff" stroke-width="1.5"
          />

          <!-- Tooltip -->
          <rect
            :x="tooltipX"
            :y="pad.top + 4"
            width="155"
            height="68"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            opacity="0.95"
          />
          <text :x="tooltipX + 8" :y="pad.top + 18" class="tooltip-text">
            {{ wavelengths[hoverIdx] }} nm
          </text>
          <text :x="tooltipX + 8" :y="pad.top + 32" class="tooltip-text" fill="#3451b2">
            TMM: {{ tmm[activeTab][hoverIdx].toFixed(4) }}
          </text>
          <text :x="tooltipX + 8" :y="pad.top + 46" class="tooltip-text" fill="#e67e22">
            torcwa: {{ torcwa[activeTab][hoverIdx].toFixed(4) }}
          </text>
          <text :x="tooltipX + 8" :y="pad.top + 60" class="tooltip-text" fill="#27ae60">
            grcwa: {{ grcwa[activeTab][hoverIdx].toFixed(4) }}
          </text>
        </template>

        <!-- Legend -->
        <rect
          :x="pad.left + 8"
          :y="pad.top + 6"
          width="148"
          height="54"
          rx="4"
          fill="var(--vp-c-bg)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          opacity="0.9"
        />
        <line
          :x1="pad.left + 14" :y1="pad.top + 20"
          :x2="pad.left + 30" :y2="pad.top + 20"
          stroke="#3451b2" stroke-width="2"
        />
        <circle :cx="pad.left + 22" :cy="pad.top + 20" r="2" fill="#3451b2" />
        <text :x="pad.left + 35" :y="pad.top + 24" class="legend-label">
          TMM (1D) - {{ tmm.runtime }}
        </text>

        <line
          :x1="pad.left + 14" :y1="pad.top + 36"
          :x2="pad.left + 30" :y2="pad.top + 36"
          stroke="#e67e22" stroke-width="2"
        />
        <circle :cx="pad.left + 22" :cy="pad.top + 36" r="2" fill="#e67e22" />
        <text :x="pad.left + 35" :y="pad.top + 40" class="legend-label">
          torcwa (2D RCWA) - {{ torcwa.runtime }}
        </text>

        <line
          :x1="pad.left + 14" :y1="pad.top + 52"
          :x2="pad.left + 30" :y2="pad.top + 52"
          stroke="#27ae60" stroke-width="2"
        />
        <circle :cx="pad.left + 22" :cy="pad.top + 52" r="2" fill="#27ae60" />
        <text :x="pad.left + 35" :y="pad.top + 56" class="legend-label">
          grcwa (2D RCWA) - {{ grcwa.runtime }}
        </text>
      </svg>
    </div>

    <!-- Summary cards -->
    <div class="summary-row">
      <div class="summary-card" style="border-left: 3px solid #3451b2;">
        <span class="summary-label">{{ t('TMM mean', 'TMM 평균') }} {{ activeTabLabel }}:</span>
        <span class="summary-value">{{ tmmMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid #e67e22;">
        <span class="summary-label">{{ t('torcwa mean', 'torcwa 평균') }} {{ activeTabLabel }}:</span>
        <span class="summary-value">{{ torcwaMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid #27ae60;">
        <span class="summary-label">{{ t('grcwa mean', 'grcwa 평균') }} {{ activeTabLabel }}:</span>
        <span class="summary-value">{{ grcwaMean.toFixed(4) }}</span>
      </div>
      <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
        <span class="summary-label">{{ t('Max |torcwa - grcwa|:', '최대 |torcwa - grcwa|:') }}</span>
        <span class="summary-value">{{ maxRcwaDelta.toFixed(4) }}</span>
      </div>
    </div>

    <!-- Data table -->
    <details class="data-table-details">
      <summary class="data-table-summary">
        {{ t('Show numerical data', '수치 데이터 보기') }}
      </summary>
      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>{{ t('Wavelength (nm)', '파장 (nm)') }}</th>
              <th class="col-tmm">TMM</th>
              <th class="col-torcwa">torcwa</th>
              <th class="col-grcwa">grcwa</th>
              <th>|torcwa - grcwa|</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(wl, i) in wavelengths" :key="wl">
              <td class="cell-wl">{{ wl }}</td>
              <td class="cell-tmm">{{ tmm[activeTab][i].toFixed(4) }}</td>
              <td class="cell-torcwa">{{ torcwa[activeTab][i].toFixed(4) }}</td>
              <td class="cell-grcwa">{{ grcwa[activeTab][i].toFixed(4) }}</td>
              <td class="cell-delta">{{ Math.abs(torcwa[activeTab][i] - grcwa[activeTab][i]).toFixed(4) }}</td>
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

// --- Tabs ---
const tabs = [
  { key: 'A', en: 'Absorption (A)', ko: '흡수 (A)' },
  { key: 'R', en: 'Reflection (R)', ko: '반사 (R)' },
  { key: 'T', en: 'Transmission (T)', ko: '투과 (T)' },
]
const activeTab = ref('A')

const activeTabLabel = computed(() => {
  const tab = tabs.find(tb => tb.key === activeTab.value)
  return tab ? t(tab.en, tab.ko) : ''
})

// --- Simulation data ---
const wavelengths = [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780]

const tmm = {
  R: [0.0401, 0.0529, 0.0961, 0.0348, 0.0633, 0.0324, 0.1461, 0.0088, 0.2405, 0.1697, 0.0280, 0.0387, 0.0477, 0.0180, 0.0209, 0.0781, 0.1199, 0.1091, 0.0664, 0.0372, 0.0430],
  T: [0.0517, 0.0722, 0.0830, 0.1085, 0.1443, 0.2549, 0.4549, 0.9135, 0.7023, 0.4775, 0.3331, 0.2423, 0.2175, 0.2258, 0.2336, 0.2284, 0.2252, 0.2345, 0.2522, 0.2667, 0.2721],
  A: [0.9081, 0.8750, 0.8210, 0.8566, 0.7923, 0.7127, 0.3990, 0.0777, 0.0572, 0.3528, 0.6389, 0.7190, 0.7349, 0.7563, 0.7455, 0.6935, 0.6549, 0.6564, 0.6815, 0.6961, 0.6849],
  runtime: '2.9ms',
  label: 'TMM (1D)',
}

const torcwa = {
  R: [0.0147, 0.0239, 0.0021, 0.0069, 0.0018, 0.0138, 0.0186, 0.0258, 0.0131, 0.0013, 0.0021, 0.0077, 0.0137, 0.0223, 0.0126, 0.0116, 0.0093, 0.0055, 0.0032, 0.0045, 0.0116],
  T: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0003, 0.0008, 0.0026, 0.0061, 0.0093, 0.0113, 0.0126, 0.0130, 0.0140, 0.0147, 0.0142, 0.0145],
  A: [0.9853, 0.9761, 0.9979, 0.9931, 0.9982, 0.9862, 0.9814, 0.9742, 0.9868, 0.9984, 0.9971, 0.9897, 0.9802, 0.9684, 0.9761, 0.9757, 0.9777, 0.9805, 0.9821, 0.9813, 0.9739],
  runtime: '15.7s',
  label: 'torcwa (2D RCWA)',
}

const grcwa = {
  R: [0.0195, 0.0139, 0.0131, 0.0125, 0.0102, 0.0119, 0.0148, 0.0144, 0.0121, 0.0114, 0.0139, 0.0176, 0.0202, 0.0206, 0.0193, 0.0173, 0.0159, 0.0161, 0.0178, 0.0205, 0.0231],
  T: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0003, 0.0004, 0.0004],
  A: [0.9805, 0.9861, 0.9869, 0.9875, 0.9898, 0.9881, 0.9852, 0.9856, 0.9879, 0.9886, 0.9861, 0.9824, 0.9797, 0.9793, 0.9807, 0.9826, 0.9840, 0.9837, 0.9819, 0.9792, 0.9765],
  runtime: '0.1s',
  label: 'grcwa (2D RCWA)',
}

// --- SVG dimensions ---
const svgW = 580
const svgH = 340
const pad = { top: 20, right: 20, bottom: 38, left: 48 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

// --- Dynamic Y-axis range based on active tab ---
const yRange = computed(() => {
  const allValues = [
    ...tmm[activeTab.value],
    ...torcwa[activeTab.value],
    ...grcwa[activeTab.value],
  ]
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  // Add some padding
  const range = maxVal - minVal
  const yMin = Math.max(0, Math.floor((minVal - range * 0.1) * 10) / 10)
  const yMax = Math.min(1, Math.ceil((maxVal + range * 0.1) * 10) / 10)
  // Ensure at least some range
  if (yMax - yMin < 0.1) {
    return { min: Math.max(0, yMin - 0.05), max: Math.min(1, yMax + 0.05) }
  }
  return { min: yMin, max: yMax }
})

const yTicks = computed(() => {
  const { min, max } = yRange.value
  const range = max - min
  let step
  if (range <= 0.1) step = 0.02
  else if (range <= 0.3) step = 0.05
  else if (range <= 0.6) step = 0.1
  else step = 0.2
  const ticks = []
  let v = Math.ceil(min / step) * step
  while (v <= max + step * 0.01) {
    ticks.push(Math.round(v * 1000) / 1000)
    v += step
  }
  return ticks
})

const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function xScale(wl) {
  return pad.left + ((wl - 380) / (780 - 380)) * plotW
}

function yScale(val) {
  const { min, max } = yRange.value
  return pad.top + plotH - ((val - min) / (max - min)) * plotH
}

// --- Build polyline points and circle positions ---
function buildPolylinePoints(data) {
  return data.map((val, i) => {
    const x = xScale(wavelengths[i])
    const y = yScale(val)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function buildCircles(data) {
  return data.map((val, i) => ({
    x: xScale(wavelengths[i]),
    y: yScale(val),
  }))
}

function buildAreaPath(data) {
  let d = ''
  for (let i = 0; i < data.length; i++) {
    const x = xScale(wavelengths[i])
    const y = yScale(data[i])
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  const lastX = xScale(wavelengths[wavelengths.length - 1])
  const firstX = xScale(wavelengths[0])
  const baseY = yScale(yRange.value.min)
  d += ` L${lastX.toFixed(1)},${baseY.toFixed(1)} L${firstX.toFixed(1)},${baseY.toFixed(1)} Z`
  return d
}

const tmmPoints = computed(() => buildPolylinePoints(tmm[activeTab.value]))
const torcwaPoints = computed(() => buildPolylinePoints(torcwa[activeTab.value]))
const grcwaPoints = computed(() => buildPolylinePoints(grcwa[activeTab.value]))

const tmmCircles = computed(() => buildCircles(tmm[activeTab.value]))
const torcwaCircles = computed(() => buildCircles(torcwa[activeTab.value]))
const grcwaCircles = computed(() => buildCircles(grcwa[activeTab.value]))

const tmmArea = computed(() => buildAreaPath(tmm[activeTab.value]))
const torcwaArea = computed(() => buildAreaPath(torcwa[activeTab.value]))
const grcwaArea = computed(() => buildAreaPath(grcwa[activeTab.value]))

// --- Summary stats ---
function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

const tmmMean = computed(() => mean(tmm[activeTab.value]))
const torcwaMean = computed(() => mean(torcwa[activeTab.value]))
const grcwaMean = computed(() => mean(grcwa[activeTab.value]))

const maxRcwaDelta = computed(() => {
  let maxD = 0
  for (let i = 0; i < wavelengths.length; i++) {
    const diff = Math.abs(torcwa[activeTab.value][i] - grcwa[activeTab.value][i])
    if (diff > maxD) maxD = diff
  }
  return maxD
})

// --- Hover interaction ---
const hoverIdx = ref(null)

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = 380 + ((mouseX - pad.left) / plotW) * (780 - 380)

  if (wl < 380 || wl > 780) {
    hoverIdx.value = null
    return
  }

  // Find the closest wavelength index
  let closestIdx = 0
  let closestDist = Infinity
  for (let i = 0; i < wavelengths.length; i++) {
    const dist = Math.abs(wavelengths[i] - wl)
    if (dist < closestDist) {
      closestDist = dist
      closestIdx = i
    }
  }
  hoverIdx.value = closestIdx
}

function onMouseLeave() {
  hoverIdx.value = null
}

const tooltipX = computed(() => {
  if (hoverIdx.value === null) return 0
  const x = xScale(wavelengths[hoverIdx.value])
  return x + 165 > svgW - pad.right ? x - 165 : x + 10
})
</script>

<style scoped>
.cross-solver-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.cross-solver-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

/* Tabs */
.tab-row {
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.tab-btn {
  padding: 8px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.88em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.tab-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}

/* SVG Chart */
.svg-wrapper {
  margin-bottom: 16px;
}

.chart-svg {
  width: 100%;
  max-width: 580px;
  display: block;
  margin: 0 auto;
  cursor: crosshair;
}

.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}

.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}

.legend-label {
  font-size: 8.5px;
  fill: var(--vp-c-text-2);
  font-weight: 500;
}

.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

/* Summary cards */
.summary-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.summary-card {
  flex: 1;
  min-width: 130px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
}

.summary-label {
  color: var(--vp-c-text-2);
  margin-right: 4px;
  display: block;
  font-size: 0.85em;
  margin-bottom: 2px;
}

.summary-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  font-size: 1.05em;
}

/* Data table */
.data-table-details {
  margin-top: 4px;
}

.data-table-summary {
  cursor: pointer;
  font-size: 0.9em;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  padding: 6px 0;
  user-select: none;
}

.data-table-summary:hover {
  color: var(--vp-c-brand-2);
}

.table-wrapper {
  overflow-x: auto;
  margin-top: 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82em;
  font-family: var(--vp-font-family-mono);
}

.data-table thead {
  background: var(--vp-c-bg);
}

.data-table th {
  padding: 8px 10px;
  text-align: right;
  font-weight: 700;
  color: var(--vp-c-text-1);
  border-bottom: 2px solid var(--vp-c-divider);
  white-space: nowrap;
}

.data-table th:first-child {
  text-align: left;
}

.data-table td {
  padding: 5px 10px;
  text-align: right;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}

.data-table td:first-child {
  text-align: left;
  font-weight: 600;
}

.data-table tbody tr:hover {
  background: var(--vp-c-bg);
}

.data-table tbody tr:last-child td {
  border-bottom: none;
}

.col-tmm {
  color: #3451b2;
}

.col-torcwa {
  color: #e67e22;
}

.col-grcwa {
  color: #27ae60;
}

.cell-wl {
  color: var(--vp-c-text-2);
}

.cell-tmm {
  color: #3451b2;
}

.cell-torcwa {
  color: #e67e22;
}

.cell-grcwa {
  color: #27ae60;
}

.cell-delta {
  color: var(--vp-c-text-2);
}

/* Responsive */
@media (max-width: 640px) {
  .cross-solver-container {
    padding: 1rem;
  }

  .tab-btn {
    padding: 6px 12px;
    font-size: 0.82em;
  }

  .summary-card {
    min-width: 100%;
  }

  .data-table {
    font-size: 0.75em;
  }

  .data-table th,
  .data-table td {
    padding: 4px 6px;
  }
}
</style>
