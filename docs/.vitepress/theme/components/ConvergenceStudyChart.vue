<template>
  <div class="convergence-container">
    <h4>{{ t('Convergence Study: RCWA Parameter Sensitivity', '수렴 연구: RCWA 파라미터 민감도') }}</h4>
    <p class="component-description">
      {{ t(
        'Analyze how simulation accuracy converges as key RCWA parameters increase. Identify the minimum parameter values needed for converged results.',
        'RCWA 주요 파라미터 증가에 따른 시뮬레이션 정확도 수렴을 분석합니다. 수렴된 결과에 필요한 최소 파라미터 값을 확인합니다.'
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
          v-for="tick in currentYTicks"
          :key="'yg' + tick"
          :x1="pad.left"
          :y1="currentYScale(tick)"
          :x2="pad.left + plotW"
          :y2="currentYScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />
        <line
          v-for="tick in currentXTicks"
          :key="'xg' + tick"
          :x1="currentXScale(tick)"
          :y1="pad.top"
          :x2="currentXScale(tick)"
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
          v-for="tick in currentYTicks"
          :key="'yl' + tick"
          :x="pad.left - 6"
          :y="currentYScale(tick) + 3"
          text-anchor="end"
          class="axis-label"
        >{{ currentYFormat(tick) }}</text>

        <!-- X-axis labels -->
        <text
          v-for="tick in currentXTicks"
          :key="'xl' + tick"
          :x="currentXScale(tick)"
          :y="pad.top + plotH + 16"
          text-anchor="middle"
          class="axis-label"
        >{{ currentXFormat(tick) }}</text>

        <!-- Axis titles -->
        <text
          :x="pad.left + plotW / 2"
          :y="svgH - 2"
          text-anchor="middle"
          class="axis-title"
        >{{ currentXLabel }}</text>
        <text
          :x="12"
          :y="pad.top + plotH / 2"
          text-anchor="middle"
          class="axis-title"
          :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`"
        >{{ currentYLabel }}</text>

        <!-- Tab: Fourier Order -->
        <template v-if="activeTab === 'fourier'">
          <!-- Convergence threshold band -->
          <rect
            :x="pad.left"
            :y="currentYScale(convergedValue + 0.001)"
            :width="plotW"
            :height="currentYScale(convergedValue - 0.001) - currentYScale(convergedValue + 0.001)"
            fill="var(--vp-c-brand-1)"
            opacity="0.08"
          />
          <line
            :x1="pad.left" :y1="currentYScale(convergedValue)"
            :x2="pad.left + plotW" :y2="currentYScale(convergedValue)"
            stroke="var(--vp-c-brand-1)"
            stroke-width="1"
            stroke-dasharray="6,3"
            opacity="0.5"
          />

          <!-- grcwa line and points -->
          <polyline
            :points="grcwaFourierPoints"
            fill="none"
            stroke="#2196F3"
            stroke-width="2"
            stroke-linejoin="round"
          />
          <circle
            v-for="(pt, i) in grcwaFourierCircles"
            :key="'gf' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="3"
            :fill="pt.converged ? '#2196F3' : '#2196F3'"
            :stroke="pt.converged ? '#fff' : 'none'"
            :stroke-width="pt.converged ? 2 : 0"
            :opacity="pt.converged ? 1 : 0.7"
          />
          <!-- grcwa convergence point highlight -->
          <circle
            v-if="grcwaConvergePt"
            :cx="grcwaConvergePt.x"
            :cy="grcwaConvergePt.y"
            r="7"
            fill="none"
            stroke="#2196F3"
            stroke-width="2"
            stroke-dasharray="3,2"
            opacity="0.6"
          />

          <!-- torcwa line and points -->
          <polyline
            :points="torcwaFourierPoints"
            fill="none"
            stroke="#FF9800"
            stroke-width="2"
            stroke-linejoin="round"
          />
          <circle
            v-for="(pt, i) in torcwaFourierCircles"
            :key="'tf' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="3"
            :fill="pt.converged ? '#FF9800' : '#FF9800'"
            :stroke="pt.converged ? '#fff' : 'none'"
            :stroke-width="pt.converged ? 2 : 0"
            :opacity="pt.converged ? 1 : 0.7"
          />
          <!-- torcwa convergence point highlight -->
          <circle
            v-if="torcwaConvergePt"
            :cx="torcwaConvergePt.x"
            :cy="torcwaConvergePt.y"
            r="7"
            fill="none"
            stroke="#FF9800"
            stroke-width="2"
            stroke-dasharray="3,2"
            opacity="0.6"
          />

          <!-- Legend -->
          <rect
            :x="pad.left + plotW - 170"
            :y="pad.top + 6"
            width="164"
            height="56"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            opacity="0.9"
          />
          <line
            :x1="pad.left + plotW - 162" :y1="pad.top + 21"
            :x2="pad.left + plotW - 148" :y2="pad.top + 21"
            stroke="#2196F3" stroke-width="2"
          />
          <circle :cx="pad.left + plotW - 155" :cy="pad.top + 21" r="2.5" fill="#2196F3" />
          <text :x="pad.left + plotW - 142" :y="pad.top + 25" class="legend-label">
            grcwa (nG)
          </text>
          <line
            :x1="pad.left + plotW - 162" :y1="pad.top + 37"
            :x2="pad.left + plotW - 148" :y2="pad.top + 37"
            stroke="#FF9800" stroke-width="2"
          />
          <circle :cx="pad.left + plotW - 155" :cy="pad.top + 37" r="2.5" fill="#FF9800" />
          <text :x="pad.left + plotW - 142" :y="pad.top + 41" class="legend-label">
            torcwa ([N,N])
          </text>
          <line
            :x1="pad.left + plotW - 162" :y1="pad.top + 52"
            :x2="pad.left + plotW - 148" :y2="pad.top + 52"
            stroke="var(--vp-c-brand-1)" stroke-width="1" stroke-dasharray="4,2"
          />
          <text :x="pad.left + plotW - 142" :y="pad.top + 56" class="legend-label">
            {{ t('Converged ±0.001', '수렴 ±0.001') }}
          </text>
        </template>

        <!-- Tab: Lens Slices -->
        <template v-if="activeTab === 'lens'">
          <polyline
            :points="lensPoints"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-linejoin="round"
          />
          <circle
            v-for="(pt, i) in lensCircles"
            :key="'lc' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="3"
            fill="var(--vp-c-brand-1)"
            opacity="0.8"
          />
        </template>

        <!-- Tab: Grid Resolution -->
        <template v-if="activeTab === 'grid'">
          <polyline
            :points="gridPoints"
            fill="none"
            stroke="var(--vp-c-brand-1)"
            stroke-width="2"
            stroke-linejoin="round"
          />
          <circle
            v-for="(pt, i) in gridCircles"
            :key="'gc' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="3"
            fill="var(--vp-c-brand-1)"
            opacity="0.8"
          />
        </template>

        <!-- Tab: Runtime vs Accuracy -->
        <template v-if="activeTab === 'runtime'">
          <!-- grcwa points -->
          <circle
            v-for="(pt, i) in runtimeGrcwaCircles"
            :key="'rg' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="4"
            fill="#2196F3"
            opacity="0.8"
          />
          <text
            v-for="(pt, i) in runtimeGrcwaCircles"
            :key="'rgl' + i"
            :x="pt.x + 6"
            :y="pt.y - 6"
            class="point-label"
            fill="#2196F3"
          >nG={{ pt.label }}</text>

          <!-- torcwa points -->
          <circle
            v-for="(pt, i) in runtimeTorcwaCircles"
            :key="'rt' + i"
            :cx="pt.x"
            :cy="pt.y"
            r="4"
            fill="#FF9800"
            opacity="0.8"
          />
          <text
            v-for="(pt, i) in runtimeTorcwaCircles"
            :key="'rtl' + i"
            :x="pt.x + 6"
            :y="pt.y - 6"
            class="point-label"
            fill="#FF9800"
          >[{{ pt.label }},{{ pt.label }}]</text>

          <!-- Legend -->
          <rect
            :x="pad.left + plotW - 130"
            :y="pad.top + 6"
            width="124"
            height="38"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            opacity="0.9"
          />
          <circle :cx="pad.left + plotW - 118" :cy="pad.top + 20" r="3" fill="#2196F3" />
          <text :x="pad.left + plotW - 110" :y="pad.top + 24" class="legend-label">grcwa</text>
          <circle :cx="pad.left + plotW - 118" :cy="pad.top + 36" r="3" fill="#FF9800" />
          <text :x="pad.left + plotW - 110" :y="pad.top + 40" class="legend-label">torcwa</text>
        </template>

        <!-- Hover tooltip -->
        <template v-if="hoverInfo">
          <circle
            :cx="hoverInfo.x"
            :cy="hoverInfo.y"
            r="6"
            :fill="hoverInfo.color"
            stroke="#fff"
            stroke-width="1.5"
          />
          <rect
            :x="hoverInfo.tooltipX"
            :y="hoverInfo.tooltipY"
            :width="hoverInfo.tooltipW"
            height="38"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.8"
            opacity="0.95"
          />
          <text
            :x="hoverInfo.tooltipX + 8"
            :y="hoverInfo.tooltipY + 15"
            class="tooltip-text"
          >{{ hoverInfo.line1 }}</text>
          <text
            :x="hoverInfo.tooltipX + 8"
            :y="hoverInfo.tooltipY + 30"
            class="tooltip-text"
            :fill="hoverInfo.color"
          >{{ hoverInfo.line2 }}</text>
        </template>
      </svg>
    </div>

    <!-- Summary cards -->
    <div class="summary-row">
      <template v-if="activeTab === 'fourier'">
        <div class="summary-card" style="border-left: 3px solid #2196F3;">
          <span class="summary-label">{{ t('grcwa converged at', 'grcwa 수렴 지점') }}:</span>
          <span class="summary-value">nG = 49 (A = 0.970, 0.95s)</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #FF9800;">
          <span class="summary-label">{{ t('torcwa converged at', 'torcwa 수렴 지점') }}:</span>
          <span class="summary-value">[5,5] = 121 (A = 0.986)</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Converged value', '수렴 값') }}:</span>
          <span class="summary-value">A = {{ convergedValue }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'lens'">
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Converged at', '수렴 지점') }}:</span>
          <span class="summary-value">n_lens_slices = 15</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Max variation (15-80)', '최대 변동 (15-80)') }}:</span>
          <span class="summary-value">{{ (0.9700 - 0.9696).toFixed(4) }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'grid'">
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Converged at', '수렴 지점') }}:</span>
          <span class="summary-value">grid_multiplier = 3</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Max variation (3-6)', '최대 변동 (3-6)') }}:</span>
          <span class="summary-value">{{ (0.9699 - 0.9698).toFixed(4) }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'runtime'">
        <div class="summary-card" style="border-left: 3px solid #2196F3;">
          <span class="summary-label">{{ t('grcwa fastest converged', 'grcwa 최소 수렴 시간') }}:</span>
          <span class="summary-value">0.95s (nG=49)</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #FF9800;">
          <span class="summary-label">{{ t('torcwa fastest converged', 'torcwa 최소 수렴 시간') }}:</span>
          <span class="summary-value">23.12s ([5,5])</span>
        </div>
      </template>
    </div>

    <!-- Data table -->
    <details class="data-table-details">
      <summary class="data-table-summary">
        {{ t('Show numerical data', '수치 데이터 보기') }}
      </summary>
      <div class="table-wrapper">
        <table v-if="activeTab === 'fourier'" class="data-table">
          <thead>
            <tr>
              <th>{{ t('Solver', '솔버') }}</th>
              <th>{{ t('Parameter', '파라미터') }}</th>
              <th>{{ t('Harmonics', '하모닉스') }}</th>
              <th>{{ t('Absorption', '흡수') }}</th>
              <th>{{ t('Runtime', '실행 시간') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in grcwaData" :key="'gd' + d.nG">
              <td class="cell-grcwa">grcwa</td>
              <td>nG = {{ d.nG }}</td>
              <td>{{ d.harmonics }}</td>
              <td>{{ d.A.toFixed(4) }}</td>
              <td>{{ d.time }}s</td>
            </tr>
            <tr v-for="d in torcwaData" :key="'td' + d.N">
              <td class="cell-torcwa">torcwa</td>
              <td>[{{ d.N }},{{ d.N }}]</td>
              <td>{{ d.harmonics }}</td>
              <td>{{ d.A.toFixed(4) }}</td>
              <td>{{ d.time }}s</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'lens'" class="data-table">
          <thead>
            <tr>
              <th>n_lens_slices</th>
              <th>{{ t('Absorption', '흡수') }}</th>
              <th>{{ t('Delta from max', '최대값 차이') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in lensSlicesData" :key="'ld' + d.slices">
              <td>{{ d.slices }}</td>
              <td>{{ d.A.toFixed(4) }}</td>
              <td>{{ (d.A - 0.9700).toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'grid'" class="data-table">
          <thead>
            <tr>
              <th>grid_multiplier</th>
              <th>{{ t('Absorption', '흡수') }}</th>
              <th>{{ t('Delta from max', '최대값 차이') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in gridData" :key="'grd' + d.multiplier">
              <td>{{ d.multiplier }}</td>
              <td>{{ d.A.toFixed(4) }}</td>
              <td>{{ (d.A - 0.9699).toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'runtime'" class="data-table">
          <thead>
            <tr>
              <th>{{ t('Solver', '솔버') }}</th>
              <th>{{ t('Parameter', '파라미터') }}</th>
              <th>{{ t('Runtime (s)', '실행 시간 (s)') }}</th>
              <th>{{ t('Absorption Error', '흡수 오차') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in grcwaData" :key="'rgd' + d.nG">
              <td class="cell-grcwa">grcwa</td>
              <td>nG = {{ d.nG }}</td>
              <td>{{ d.time }}</td>
              <td>{{ Math.abs(d.A - convergedValue).toFixed(4) }}</td>
            </tr>
            <tr v-for="d in torcwaData" :key="'rtd' + d.N">
              <td class="cell-torcwa">torcwa</td>
              <td>[{{ d.N }},{{ d.N }}]</td>
              <td>{{ d.time }}</td>
              <td>{{ Math.abs(d.A - convergedValue).toFixed(4) }}</td>
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
  { key: 'fourier', en: 'Fourier Order', ko: '푸리에 차수' },
  { key: 'lens', en: 'Lens Slices', ko: '렌즈 슬라이스' },
  { key: 'grid', en: 'Grid Resolution', ko: '그리드 해상도' },
  { key: 'runtime', en: 'Runtime vs Accuracy', ko: '실행 시간 vs 정확도' },
]
const activeTab = ref('fourier')

// --- Data ---
// Actual convergence data from COMPASS convergence study runs (550nm, BSI 1um pixel)
// Note: grcwa has numerical instability at certain nG values (81, 169, 225, etc.)
// Only numerically stable data points are shown
const grcwaData = [
  { nG: 9, harmonics: 9, A: 0.9905, time: 0.02 },
  { nG: 25, harmonics: 25, A: 0.9868, time: 0.17 },
  { nG: 49, harmonics: 49, A: 0.9699, time: 0.95 },
  { nG: 121, harmonics: 121, A: 0.9712, time: 5.42 },
  { nG: 625, harmonics: 625, A: 0.9749, time: 356.24 },
]

const torcwaData = [
  { N: 3, harmonics: 49, A: 0.98202, time: 2.77 },
  { N: 5, harmonics: 121, A: 0.98561, time: 23.12 },
  { N: 7, harmonics: 225, A: 0.98581, time: 100.13 },
  { N: 9, harmonics: 361, A: 0.98305, time: 349.79 },
  { N: 11, harmonics: 529, A: 0.98582, time: 846.12 },
]

const lensSlicesData = [
  { slices: 5, A: 0.9662 },
  { slices: 10, A: 0.9691 },
  { slices: 15, A: 0.9696 },
  { slices: 20, A: 0.9698 },
  { slices: 30, A: 0.9699 },
  { slices: 50, A: 0.9700 },
  { slices: 80, A: 0.9700 },
]

// Note: grid_multiplier 2 and 4 cause grcwa instability; stable values shown
const gridData = [
  { multiplier: 3, A: 0.9699 },
  { multiplier: 5, A: 0.9698 },
  { multiplier: 6, A: 0.9699 },
]

const convergedValue = 0.970

// --- SVG dimensions ---
const svgW = 600
const svgH = 400
const pad = { top: 24, right: 24, bottom: 42, left: 56 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

// --- Scale helpers ---
function linearScale(val, domain, range) {
  const t = (val - domain[0]) / (domain[1] - domain[0])
  return range[0] + t * (range[1] - range[0])
}

function logScale(val, domain, range) {
  if (val <= 0) val = domain[0]
  const logMin = Math.log10(domain[0])
  const logMax = Math.log10(domain[1])
  const t = (Math.log10(val) - logMin) / (logMax - logMin)
  return range[0] + t * (range[1] - range[0])
}

// --- Fourier Order tab ---
const fourierXDomain = [0, 650]
const fourierYDomain = [0.96, 1.00]

function fourierXScale(val) {
  return linearScale(val, fourierXDomain, [pad.left, pad.left + plotW])
}
function fourierYScale(val) {
  return linearScale(val, fourierYDomain, [pad.top + plotH, pad.top])
}

const fourierXTicks = [0, 100, 200, 300, 400, 500, 600]
const fourierYTicks = [0.96, 0.97, 0.98, 0.99, 1.00]

const grcwaFourierPoints = computed(() =>
  grcwaData.map(d => `${fourierXScale(d.harmonics).toFixed(1)},${fourierYScale(d.A).toFixed(1)}`).join(' ')
)
const grcwaFourierCircles = computed(() =>
  grcwaData.map(d => ({
    x: fourierXScale(d.harmonics),
    y: fourierYScale(d.A),
    converged: Math.abs(d.A - convergedValue) <= 0.001,
  }))
)
const grcwaConvergePt = computed(() => {
  const d = grcwaData.find(d => Math.abs(d.A - convergedValue) <= 0.001)
  if (!d) return null
  return { x: fourierXScale(d.harmonics), y: fourierYScale(d.A) }
})

const torcwaFourierPoints = computed(() =>
  torcwaData.map(d => `${fourierXScale(d.harmonics).toFixed(1)},${fourierYScale(d.A).toFixed(1)}`).join(' ')
)
const torcwaFourierCircles = computed(() =>
  torcwaData.map(d => ({
    x: fourierXScale(d.harmonics),
    y: fourierYScale(d.A),
    converged: Math.abs(d.A - convergedValue) <= 0.001,
  }))
)
const torcwaConvergePt = computed(() => {
  const d = torcwaData.find(d => Math.abs(d.A - convergedValue) <= 0.001)
  if (!d) return null
  return { x: fourierXScale(d.harmonics), y: fourierYScale(d.A) }
})

// --- Lens Slices tab ---
const lensXDomain = [0, 85]
const lensYDomain = [0.964, 0.972]

function lensXScale(val) {
  return linearScale(val, lensXDomain, [pad.left, pad.left + plotW])
}
function lensYScale(val) {
  return linearScale(val, lensYDomain, [pad.top + plotH, pad.top])
}

const lensXTicks = [10, 20, 30, 40, 50, 60, 70, 80]
const lensYTicks = [0.964, 0.966, 0.968, 0.970, 0.972]

const lensPoints = computed(() =>
  lensSlicesData.map(d => `${lensXScale(d.slices).toFixed(1)},${lensYScale(d.A).toFixed(1)}`).join(' ')
)
const lensCircles = computed(() =>
  lensSlicesData.map(d => ({ x: lensXScale(d.slices), y: lensYScale(d.A) }))
)

// --- Grid Resolution tab ---
const gridXDomain = [2.5, 6.5]
const gridYDomain = [0.968, 0.972]

function gridXScale(val) {
  return linearScale(val, gridXDomain, [pad.left, pad.left + plotW])
}
function gridYScale(val) {
  return linearScale(val, gridYDomain, [pad.top + plotH, pad.top])
}

const gridXTicks = [3, 4, 5, 6]
const gridYTicks = [0.968, 0.969, 0.970, 0.971, 0.972]

const gridPoints = computed(() =>
  gridData.map(d => `${gridXScale(d.multiplier).toFixed(1)},${gridYScale(d.A).toFixed(1)}`).join(' ')
)
const gridCircles = computed(() =>
  gridData.map(d => ({ x: gridXScale(d.multiplier), y: gridYScale(d.A) }))
)

// --- Runtime vs Accuracy tab ---
const runtimeXDomain = [0.01, 400]
const runtimeYDomain = [-0.002, 0.025]

function runtimeXScale(val) {
  return logScale(val, runtimeXDomain, [pad.left, pad.left + plotW])
}
function runtimeYScale(val) {
  return linearScale(val, runtimeYDomain, [pad.top + plotH, pad.top])
}

const runtimeXTicks = [0.01, 0.1, 1, 10, 100]
const runtimeYTicks = [0, 0.005, 0.010, 0.015, 0.020]

const runtimeGrcwaCircles = computed(() =>
  grcwaData.map(d => ({
    x: runtimeXScale(d.time),
    y: runtimeYScale(Math.abs(d.A - convergedValue)),
    label: d.nG,
  }))
)
const runtimeTorcwaCircles = computed(() =>
  torcwaData.map(d => ({
    x: runtimeXScale(d.time),
    y: runtimeYScale(Math.abs(d.A - convergedValue)),
    label: d.N,
  }))
)

// --- Dynamic axis config per tab ---
const currentXScale = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierXScale
    case 'lens': return lensXScale
    case 'grid': return gridXScale
    case 'runtime': return runtimeXScale
    default: return fourierXScale
  }
})

const currentYScale = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierYScale
    case 'lens': return lensYScale
    case 'grid': return gridYScale
    case 'runtime': return runtimeYScale
    default: return fourierYScale
  }
})

const currentXTicks = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierXTicks
    case 'lens': return lensXTicks
    case 'grid': return gridXTicks
    case 'runtime': return runtimeXTicks
    default: return fourierXTicks
  }
})

const currentYTicks = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierYTicks
    case 'lens': return lensYTicks
    case 'grid': return gridYTicks
    case 'runtime': return runtimeYTicks
    default: return fourierYTicks
  }
})

const currentXLabel = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return t('Number of Harmonics', '하모닉스 수')
    case 'lens': return t('n_lens_slices', 'n_lens_slices')
    case 'grid': return t('grid_multiplier', 'grid_multiplier')
    case 'runtime': return t('Runtime (seconds, log scale)', '실행 시간 (초, 로그 스케일)')
    default: return ''
  }
})

const currentYLabel = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return t('Absorption', '흡수')
    case 'lens': return t('Absorption', '흡수')
    case 'grid': return t('Absorption', '흡수')
    case 'runtime': return t('Absorption Error', '흡수 오차')
    default: return ''
  }
})

function currentXFormat(tick) {
  if (activeTab.value === 'runtime') {
    if (tick < 1) return tick.toString()
    return tick.toString()
  }
  if (activeTab.value === 'lens' || activeTab.value === 'grid') return tick.toString()
  return tick.toString()
}

function currentYFormat(tick) {
  if (activeTab.value === 'runtime') return tick.toFixed(2)
  if (activeTab.value === 'lens' || activeTab.value === 'grid') return tick.toFixed(3)
  return tick.toFixed(2)
}

// --- Hover interaction ---
const hoverInfo = ref(null)

function getDataPoints() {
  switch (activeTab.value) {
    case 'fourier':
      return [
        ...grcwaData.map(d => ({
          x: fourierXScale(d.harmonics),
          y: fourierYScale(d.A),
          color: '#2196F3',
          line1: `${t('Harmonics', '하모닉스')}: ${d.harmonics}`,
          line2: `grcwa A = ${d.A.toFixed(4)} (${d.time}s)`,
        })),
        ...torcwaData.map(d => ({
          x: fourierXScale(d.harmonics),
          y: fourierYScale(d.A),
          color: '#FF9800',
          line1: `[${d.N},${d.N}] → ${d.harmonics} ${t('harmonics', '하모닉스')}`,
          line2: `torcwa A = ${d.A.toFixed(4)} (${d.time}s)`,
        })),
      ]
    case 'lens':
      return lensSlicesData.map(d => ({
        x: lensXScale(d.slices),
        y: lensYScale(d.A),
        color: 'var(--vp-c-brand-1)',
        line1: `n_lens_slices = ${d.slices}`,
        line2: `A = ${d.A.toFixed(4)}`,
      }))
    case 'grid':
      return gridData.map(d => ({
        x: gridXScale(d.multiplier),
        y: gridYScale(d.A),
        color: 'var(--vp-c-brand-1)',
        line1: `grid_multiplier = ${d.multiplier}`,
        line2: `A = ${d.A.toFixed(4)}`,
      }))
    case 'runtime':
      return [
        ...grcwaData.map(d => ({
          x: runtimeXScale(d.time),
          y: runtimeYScale(Math.abs(d.A - convergedValue)),
          color: '#2196F3',
          line1: `grcwa nG=${d.nG} (${d.time}s)`,
          line2: `${t('Error', '오차')} = ${Math.abs(d.A - convergedValue).toFixed(4)}`,
        })),
        ...torcwaData.map(d => ({
          x: runtimeXScale(d.time),
          y: runtimeYScale(Math.abs(d.A - convergedValue)),
          color: '#FF9800',
          line1: `torcwa [${d.N},${d.N}] (${d.time}s)`,
          line2: `${t('Error', '오차')} = ${Math.abs(d.A - convergedValue).toFixed(4)}`,
        })),
      ]
    default:
      return []
  }
}

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const scaleY = svgH / rect.height
  const mouseX = (event.clientX - rect.left) * scaleX
  const mouseY = (event.clientY - rect.top) * scaleY

  const points = getDataPoints()
  let closest = null
  let closestDist = Infinity

  for (const pt of points) {
    const dx = mouseX - pt.x
    const dy = mouseY - pt.y
    const dist = Math.sqrt(dx * dx + dy * dy)
    if (dist < closestDist && dist < 30) {
      closestDist = dist
      closest = pt
    }
  }

  if (closest) {
    const tooltipW = 180
    hoverInfo.value = {
      ...closest,
      tooltipX: closest.x + tooltipW + 10 > svgW - pad.right ? closest.x - tooltipW - 10 : closest.x + 10,
      tooltipY: Math.max(pad.top, Math.min(closest.y - 19, pad.top + plotH - 40)),
      tooltipW,
    }
  } else {
    hoverInfo.value = null
  }
}

function onMouseLeave() {
  hoverInfo.value = null
}
</script>

<style scoped>
.convergence-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.convergence-container h4 {
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
  max-width: 600px;
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

.point-label {
  font-size: 7.5px;
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
  min-width: 150px;
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

.cell-grcwa {
  color: #2196F3;
}

.cell-torcwa {
  color: #FF9800;
}

/* Responsive */
@media (max-width: 640px) {
  .convergence-container {
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
