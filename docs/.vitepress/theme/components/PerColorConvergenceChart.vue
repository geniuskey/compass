<template>
  <div class="per-color-container">
    <h4>{{ t('Per-Color QE Convergence (Uniform CF)', '색상별 QE 수렴 (단일 CF)') }}</h4>
    <p class="component-description">
      {{ t(
        'Per-color QE analysis using uniform color filter simulations. Each color is measured separately with its CF material applied to all pixels, giving true per-channel spectral response.',
        '단일 컬러 필터 시뮬레이션을 사용한 색상별 QE 분석. 각 색상은 해당 CF 재료를 모든 픽셀에 적용하여 별도로 측정하며, 실제 채널별 분광 감도를 제공합니다.'
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

        <!-- Tab: Per-Color Fourier -->
        <template v-if="activeTab === 'fourier'">
          <template v-for="color in ['B', 'G', 'R']" :key="'fc' + color">
            <polyline
              :points="fourierColorPoints(color)"
              fill="none"
              :stroke="colorMap[color]"
              stroke-width="2"
              stroke-linejoin="round"
            />
            <circle
              v-for="(pt, i) in fourierColorCircles(color)"
              :key="'fcc' + color + i"
              :cx="pt.x"
              :cy="pt.y"
              r="3.5"
              :fill="colorMap[color]"
              opacity="0.85"
            />
          </template>

          <!-- Legend -->
          <rect
            :x="pad.left + plotW - 140"
            :y="pad.top + 6"
            width="134"
            height="52"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            opacity="0.92"
          />
          <template v-for="(color, ci) in ['R', 'G', 'B']" :key="'leg' + color">
            <line
              :x1="pad.left + plotW - 132" :y1="pad.top + 19 + ci * 15"
              :x2="pad.left + plotW - 118" :y2="pad.top + 19 + ci * 15"
              :stroke="colorMap[color]" stroke-width="2"
            />
            <circle
              :cx="pad.left + plotW - 125" :cy="pad.top + 19 + ci * 15"
              r="2.5" :fill="colorMap[color]"
            />
            <text
              :x="pad.left + plotW - 112" :y="pad.top + 22 + ci * 15"
              class="legend-label"
            >{{ colorLabels[color] }}</text>
          </template>
        </template>

        <!-- Tab: grcwa Fourier -->
        <template v-if="activeTab === 'grcwaFourier'">
          <template v-for="color in ['B', 'G', 'R']" :key="'gfc' + color">
            <polyline
              :points="grcwaFourierColorPoints(color)"
              fill="none"
              :stroke="colorMap[color]"
              stroke-width="2"
              stroke-linejoin="round"
              :stroke-dasharray="color === 'R' ? '6,3' : 'none'"
            />
            <circle
              v-for="(pt, i) in grcwaFourierColorCircles(color)"
              :key="'gfcc' + color + i"
              :cx="pt.x"
              :cy="pt.y"
              r="3.5"
              :fill="pt.unstable ? '#999' : colorMap[color]"
              :opacity="pt.unstable ? 0.4 : 0.85"
            />
          </template>

          <!-- Instability marker -->
          <text
            :x="grcwaXScale(81) + 4"
            :y="grcwaYScale(0.5) - 6"
            class="bar-label"
            fill="#E53935"
          >{{ t('TM unstable', 'TM 불안정') }}</text>

          <!-- Legend -->
          <rect
            :x="pad.left + plotW - 155"
            :y="pad.top + 6"
            width="149"
            height="68"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            opacity="0.92"
          />
          <template v-for="(color, ci) in ['R', 'G', 'B']" :key="'gleg' + color">
            <line
              :x1="pad.left + plotW - 147" :y1="pad.top + 19 + ci * 15"
              :x2="pad.left + plotW - 133" :y2="pad.top + 19 + ci * 15"
              :stroke="colorMap[color]" stroke-width="2"
              :stroke-dasharray="color === 'R' ? '6,3' : 'none'"
            />
            <circle
              :cx="pad.left + plotW - 140" :cy="pad.top + 19 + ci * 15"
              r="2.5" :fill="colorMap[color]"
            />
            <text
              :x="pad.left + plotW - 127" :y="pad.top + 22 + ci * 15"
              class="legend-label"
            >{{ colorLabels[color] }}</text>
          </template>
          <circle :cx="pad.left + plotW - 140" :cy="pad.top + 64" r="2.5" fill="#999" opacity="0.4" />
          <text :x="pad.left + plotW - 127" :y="pad.top + 67" class="legend-label">
            {{ t('Unstable', '불안정') }}
          </text>
        </template>

        <!-- Tab: Cross-Solver -->
        <template v-if="activeTab === 'crossSolver'">
          <template v-for="(color, ci) in ['B', 'G', 'R']" :key="'cs' + color">
            <!-- grcwa bar -->
            <rect
              :x="barX(ci, 0)"
              :y="crossYScale(crossSolverData.grcwa[color])"
              :width="barWidth"
              :height="pad.top + plotH - crossYScale(crossSolverData.grcwa[color])"
              :fill="colorMap[color]"
              opacity="0.85"
            />
            <!-- torcwa bar -->
            <rect
              :x="barX(ci, 1)"
              :y="crossYScale(crossSolverData.torcwa[color])"
              :width="barWidth"
              :height="pad.top + plotH - crossYScale(crossSolverData.torcwa[color])"
              :fill="colorMap[color]"
              opacity="0.45"
              stroke="var(--vp-c-text-2)"
              stroke-width="0.5"
            />
            <!-- Value labels -->
            <text
              :x="barX(ci, 0) + barWidth / 2"
              :y="crossYScale(crossSolverData.grcwa[color]) - 5"
              text-anchor="middle"
              class="bar-label"
            >{{ crossSolverData.grcwa[color].toFixed(3) }}</text>
            <text
              :x="barX(ci, 1) + barWidth / 2"
              :y="crossYScale(crossSolverData.torcwa[color]) - 5"
              text-anchor="middle"
              class="bar-label"
            >{{ crossSolverData.torcwa[color].toFixed(3) }}</text>
            <!-- Color label -->
            <text
              :x="barX(ci, 0) + barWidth + 2"
              :y="pad.top + plotH + 16"
              text-anchor="middle"
              class="axis-label"
            >{{ color }} ({{ { B: '450', G: '530', R: '620' }[color] }}nm)</text>
          </template>

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
            opacity="0.92"
          />
          <rect :x="pad.left + plotW - 122" :y="pad.top + 15" width="10" height="8" fill="var(--vp-c-text-2)" opacity="0.85" />
          <text :x="pad.left + plotW - 108" :y="pad.top + 23" class="legend-label">
            grcwa (nG=49)
          </text>
          <rect :x="pad.left + plotW - 122" :y="pad.top + 29" width="10" height="8" fill="var(--vp-c-text-2)" opacity="0.35" stroke="var(--vp-c-text-2)" stroke-width="0.5" />
          <text :x="pad.left + plotW - 108" :y="pad.top + 37" class="legend-label">
            torcwa ([5,5])
          </text>
        </template>

        <!-- Tab: Angle Dependence -->
        <template v-if="activeTab === 'angle'">
          <template v-for="color in ['B', 'G', 'R']" :key="'ac' + color">
            <polyline
              :points="angleColorPoints(color)"
              fill="none"
              :stroke="colorMap[color]"
              stroke-width="2"
              stroke-linejoin="round"
            />
            <circle
              v-for="(pt, i) in angleColorCircles(color)"
              :key="'acc' + color + i"
              :cx="pt.x"
              :cy="pt.y"
              r="3.5"
              :fill="colorMap[color]"
              opacity="0.85"
            />
          </template>

          <!-- Legend -->
          <rect
            :x="pad.left + plotW - 140"
            :y="pad.top + 6"
            width="134"
            height="52"
            rx="4"
            fill="var(--vp-c-bg)"
            stroke="var(--vp-c-divider)"
            stroke-width="0.5"
            opacity="0.92"
          />
          <template v-for="(color, ci) in ['R', 'G', 'B']" :key="'aleg' + color">
            <line
              :x1="pad.left + plotW - 132" :y1="pad.top + 19 + ci * 15"
              :x2="pad.left + plotW - 118" :y2="pad.top + 19 + ci * 15"
              :stroke="colorMap[color]" stroke-width="2"
            />
            <text
              :x="pad.left + plotW - 112" :y="pad.top + 22 + ci * 15"
              class="legend-label"
            >{{ colorLabels[color] }}</text>
          </template>
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
        <div class="summary-card" style="border-left: 3px solid #E53935;">
          <span class="summary-label">{{ t('Red (620nm)', '적색 (620nm)') }}:</span>
          <span class="summary-value">{{ t('Converged at N=5 (QE≈0.962)', 'N=5에서 수렴 (QE≈0.962)') }}</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #43A047;">
          <span class="summary-label">{{ t('Green (530nm)', '녹색 (530nm)') }}:</span>
          <span class="summary-value">{{ t('Converged at N=7 (QE≈0.974)', 'N=7에서 수렴 (QE≈0.974)') }}</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #1E88E5;">
          <span class="summary-label">{{ t('Blue (450nm)', '청색 (450nm)') }}:</span>
          <span class="summary-value">{{ t('Non-monotonic: settles at QE≈0.949', '비단조적: QE≈0.949로 안정화') }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'grcwaFourier'">
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('Stable range', '안정 범위') }}:</span>
          <span class="summary-value">{{ t('nG≤49 for all colors', '모든 색상에서 nG≤49') }}</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #E53935;">
          <span class="summary-label">{{ t('Red instability', '적색 불안정') }}:</span>
          <span class="summary-value">{{ t('nG=81: TM diverges (QE=0.50)', 'nG=81: TM 발산 (QE=0.50)') }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'crossSolver'">
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('grcwa vs torcwa difference', 'grcwa 대 torcwa 차이') }}:</span>
          <span class="summary-value">2.9–3.8% ({{ t('torcwa higher', 'torcwa가 더 높음') }})</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #1E88E5;">
          <span class="summary-label">{{ t('Largest gap', '최대 차이') }}:</span>
          <span class="summary-value">{{ t('Blue channel (3.8%)', '청색 채널 (3.8%)') }}</span>
        </div>
      </template>
      <template v-if="activeTab === 'angle'">
        <div class="summary-card" style="border-left: 3px solid var(--vp-c-brand-1);">
          <span class="summary-label">{{ t('CRA effect', 'CRA 효과') }}:</span>
          <span class="summary-value">{{ t('QE increases with CRA (auto_cra shift)', 'CRA 증가 시 QE 증가 (auto_cra 시프트)') }}</span>
        </div>
        <div class="summary-card" style="border-left: 3px solid #43A047;">
          <span class="summary-label">{{ t('Relative Illumination', '상대 조도') }}:</span>
          <span class="summary-value">{{ t('RI > 1.0 at all angles', '모든 각도에서 RI > 1.0') }}</span>
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
              <th>N</th>
              <th>{{ t('Harmonics', '하모닉스') }}</th>
              <th>QE_R@620</th>
              <th>QE_G@530</th>
              <th>QE_B@450</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in torcwaFourierData" :key="'tfd' + d.N">
              <td>[{{ d.N }},{{ d.N }}]</td>
              <td>{{ d.harmonics }}</td>
              <td :style="{ color: colorMap.R }">{{ d.QE_R.toFixed(4) }}</td>
              <td :style="{ color: colorMap.G }">{{ d.QE_G.toFixed(4) }}</td>
              <td :style="{ color: colorMap.B }">{{ d.QE_B.toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'grcwaFourier'" class="data-table">
          <thead>
            <tr>
              <th>nG</th>
              <th>QE_R@620</th>
              <th>QE_G@530</th>
              <th>QE_B@450</th>
              <th>{{ t('Status', '상태') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in grcwaFourierData" :key="'gfd' + d.nG">
              <td>{{ d.nG }}</td>
              <td :style="{ color: d.unstable_R ? '#999' : colorMap.R }">{{ d.QE_R.toFixed(4) }}</td>
              <td :style="{ color: colorMap.G }">{{ d.QE_G.toFixed(4) }}</td>
              <td :style="{ color: colorMap.B }">{{ d.QE_B.toFixed(4) }}</td>
              <td>{{ d.unstable_R ? t('R: TM unstable', 'R: TM 불안정') : t('Stable', '안정') }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'crossSolver'" class="data-table">
          <thead>
            <tr>
              <th>{{ t('Solver', '솔버') }}</th>
              <th>QE_R@620</th>
              <th>QE_G@530</th>
              <th>QE_B@450</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="cell-grcwa">grcwa (nG=49)</td>
              <td>{{ crossSolverData.grcwa.R.toFixed(4) }}</td>
              <td>{{ crossSolverData.grcwa.G.toFixed(4) }}</td>
              <td>{{ crossSolverData.grcwa.B.toFixed(4) }}</td>
            </tr>
            <tr>
              <td class="cell-torcwa">torcwa ([5,5])</td>
              <td>{{ crossSolverData.torcwa.R.toFixed(4) }}</td>
              <td>{{ crossSolverData.torcwa.G.toFixed(4) }}</td>
              <td>{{ crossSolverData.torcwa.B.toFixed(4) }}</td>
            </tr>
            <tr>
              <td style="font-weight: 600;">{{ t('Difference', '차이') }}</td>
              <td>{{ (crossSolverData.grcwa.R - crossSolverData.torcwa.R).toFixed(4) }}</td>
              <td>{{ (crossSolverData.grcwa.G - crossSolverData.torcwa.G).toFixed(4) }}</td>
              <td>{{ (crossSolverData.grcwa.B - crossSolverData.torcwa.B).toFixed(4) }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="activeTab === 'angle'" class="data-table">
          <thead>
            <tr>
              <th>CRA</th>
              <th>QE_R@620</th>
              <th>QE_G@530</th>
              <th>QE_B@450</th>
              <th>RI_R</th>
              <th>RI_G</th>
              <th>RI_B</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in angleData" :key="'ad' + d.cra">
              <td>{{ d.cra }}°</td>
              <td :style="{ color: colorMap.R }">{{ d.QE_R.toFixed(4) }}</td>
              <td :style="{ color: colorMap.G }">{{ d.QE_G.toFixed(4) }}</td>
              <td :style="{ color: colorMap.B }">{{ d.QE_B.toFixed(4) }}</td>
              <td>{{ (d.QE_R / angleData[0].QE_R).toFixed(3) }}</td>
              <td>{{ (d.QE_G / angleData[0].QE_G).toFixed(3) }}</td>
              <td>{{ (d.QE_B / angleData[0].QE_B).toFixed(3) }}</td>
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

// --- Color scheme (CIS convention) ---
const colorMap = { R: '#E53935', G: '#43A047', B: '#1E88E5' }
const colorLabels = {
  R: 'Red (620nm)',
  G: 'Green (530nm)',
  B: 'Blue (450nm)',
}

// --- Tabs ---
const tabs = [
  { key: 'fourier', en: 'torcwa Fourier', ko: 'torcwa 푸리에' },
  { key: 'grcwaFourier', en: 'grcwa Fourier', ko: 'grcwa 푸리에' },
  { key: 'crossSolver', en: 'Cross-Solver', ko: '솔버 비교' },
  { key: 'angle', en: 'Angle Dependence', ko: '각도 의존성' },
]
const activeTab = ref('fourier')

// --- Data (from COMPASS per-color convergence study, uniform CF, BSI 1um pixel) ---
// torcwa Fourier convergence: QE = total absorption through uniform CF at peak wavelength
const torcwaFourierData = [
  { N: 3, harmonics: 49, QE_R: 0.9289, QE_G: 0.9471, QE_B: 0.9196 },
  { N: 5, harmonics: 121, QE_R: 0.9622, QE_G: 0.9770, QE_B: 0.9697 },
  { N: 7, harmonics: 225, QE_R: 0.9615, QE_G: 0.9737, QE_B: 0.9475 },
  { N: 9, harmonics: 361, QE_R: 0.9617, QE_G: 0.9737, QE_B: 0.9486 },
]

// grcwa Fourier convergence (nG=81 Red is unstable: R_R=14.33, A=0.500)
const grcwaFourierData = [
  { nG: 9, harmonics: 9, QE_R: 0.9811, QE_G: 0.9931, QE_B: 0.9949 },
  { nG: 25, harmonics: 25, QE_R: 0.9834, QE_G: 0.9883, QE_B: 0.9734 },
  { nG: 49, harmonics: 49, QE_R: 0.9333, QE_G: 0.9432, QE_B: 0.9316 },
  { nG: 81, harmonics: 81, QE_R: 0.5000, QE_G: 0.9670, QE_B: 0.9340, unstable_R: true },
]

// Cross-solver comparison (uniform CF, converged params)
const crossSolverData = {
  grcwa: { R: 0.9333, G: 0.9432, B: 0.9316 },
  torcwa: { R: 0.9622, G: 0.9770, B: 0.9697 },
}

// Angle-dependent QE (torcwa N=5, uniform CF)
// Note: QE increases with CRA — microlens auto_cra shift improves coupling at oblique angles
const angleData = [
  { cra: 0, QE_R: 0.9622, QE_G: 0.9770, QE_B: 0.9697 },
  { cra: 15, QE_R: 0.9960, QE_G: 1.0000, QE_B: 0.9995 },
  { cra: 30, QE_R: 0.9980, QE_G: 0.9997, QE_B: 0.9954 },
]

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

// --- Fourier tab ---
const fourierXDomain = [0, 400]
const fourierYDomain = [0.88, 1.0]

function fourierXScale(val) {
  return linearScale(val, fourierXDomain, [pad.left, pad.left + plotW])
}
function fourierYScale(val) {
  return linearScale(val, fourierYDomain, [pad.top + plotH, pad.top])
}

const fourierXTicks = [0, 50, 100, 150, 200, 250, 300, 350]
const fourierYTicks = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

function fourierColorPoints(color) {
  const key = `QE_${color}`
  return torcwaFourierData
    .map(d => `${fourierXScale(d.harmonics).toFixed(1)},${fourierYScale(d[key]).toFixed(1)}`)
    .join(' ')
}

function fourierColorCircles(color) {
  const key = `QE_${color}`
  return torcwaFourierData.map(d => ({
    x: fourierXScale(d.harmonics),
    y: fourierYScale(d[key]),
  }))
}

// --- grcwa Fourier tab ---
const grcwaXDomain = [0, 90]
const grcwaYDomain = [0.45, 1.0]

function grcwaXScale(val) {
  return linearScale(val, grcwaXDomain, [pad.left, pad.left + plotW])
}
function grcwaYScale(val) {
  return linearScale(val, grcwaYDomain, [pad.top + plotH, pad.top])
}

const grcwaXTicks = [0, 9, 25, 49, 81]
const grcwaYTicks = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

function grcwaFourierColorPoints(color) {
  const key = `QE_${color}`
  return grcwaFourierData
    .map(d => `${grcwaXScale(d.nG).toFixed(1)},${grcwaYScale(d[key]).toFixed(1)}`)
    .join(' ')
}

function grcwaFourierColorCircles(color) {
  const key = `QE_${color}`
  return grcwaFourierData.map(d => ({
    x: grcwaXScale(d.nG),
    y: grcwaYScale(d[key]),
    unstable: color === 'R' && d.unstable_R,
  }))
}

// --- Cross-Solver tab ---
const crossYDomain = [0.90, 1.0]
const crossYTicks = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
const barWidth = 35
const barGap = 6

function crossYScale(val) {
  return linearScale(val, crossYDomain, [pad.top + plotH, pad.top])
}

function barX(colorIndex, solverIndex) {
  const groupW = barWidth * 2 + barGap
  const totalW = groupW * 3 + 40
  const startX = pad.left + (plotW - totalW) / 2
  return startX + colorIndex * (groupW + 20) + solverIndex * (barWidth + barGap)
}

// --- Angle tab ---
const angleXDomain = [0, 35]
const angleYDomain = [0.95, 1.005]

function angleXScale(val) {
  return linearScale(val, angleXDomain, [pad.left, pad.left + plotW])
}
function angleYScale(val) {
  return linearScale(val, angleYDomain, [pad.top + plotH, pad.top])
}

const angleXTicks = [0, 5, 10, 15, 20, 25, 30]
const angleYTicks = [0.95, 0.96, 0.97, 0.98, 0.99, 1.00]

function angleColorPoints(color) {
  const key = `QE_${color}`
  return angleData
    .map(d => `${angleXScale(d.cra).toFixed(1)},${angleYScale(d[key]).toFixed(1)}`)
    .join(' ')
}

function angleColorCircles(color) {
  const key = `QE_${color}`
  return angleData.map(d => ({
    x: angleXScale(d.cra),
    y: angleYScale(d[key]),
  }))
}

// --- Dynamic axis config per tab ---
const currentXScale = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierXScale
    case 'grcwaFourier': return grcwaXScale
    case 'crossSolver': return (v) => linearScale(v, [0, 3], [pad.left, pad.left + plotW])
    case 'angle': return angleXScale
    default: return fourierXScale
  }
})

const currentYScale = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierYScale
    case 'grcwaFourier': return grcwaYScale
    case 'crossSolver': return crossYScale
    case 'angle': return angleYScale
    default: return fourierYScale
  }
})

const currentXTicks = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierXTicks
    case 'grcwaFourier': return grcwaXTicks
    case 'crossSolver': return []
    case 'angle': return angleXTicks
    default: return fourierXTicks
  }
})

const currentYTicks = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return fourierYTicks
    case 'grcwaFourier': return grcwaYTicks
    case 'crossSolver': return crossYTicks
    case 'angle': return angleYTicks
    default: return fourierYTicks
  }
})

const currentXLabel = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return t('Number of Harmonics (torcwa)', '하모닉스 수 (torcwa)')
    case 'grcwaFourier': return t('nG (grcwa)', 'nG (grcwa)')
    case 'crossSolver': return t('Color Channel', '색상 채널')
    case 'angle': return t('CRA (degrees)', 'CRA (도)')
    default: return ''
  }
})

const currentYLabel = computed(() => {
  switch (activeTab.value) {
    case 'fourier': return t('QE (uniform CF)', 'QE (단일 CF)')
    case 'grcwaFourier': return t('QE (uniform CF)', 'QE (단일 CF)')
    case 'crossSolver': return t('QE at peak wavelength', '피크 파장 QE')
    case 'angle': return t('QE (uniform CF)', 'QE (단일 CF)')
    default: return ''
  }
})

function currentXFormat(tick) {
  return tick.toString()
}

function currentYFormat(tick) {
  return tick.toFixed(2)
}

// --- Hover interaction ---
const hoverInfo = ref(null)

function getDataPoints() {
  switch (activeTab.value) {
    case 'fourier':
      return ['R', 'G', 'B'].flatMap(color => {
        const key = `QE_${color}`
        return torcwaFourierData.map(d => ({
          x: fourierXScale(d.harmonics),
          y: fourierYScale(d[key]),
          color: colorMap[color],
          line1: `[${d.N},${d.N}] → ${d.harmonics} ${t('harmonics', '하모닉스')}`,
          line2: `${color} QE = ${d[key].toFixed(4)}`,
        }))
      })
    case 'grcwaFourier':
      return ['R', 'G', 'B'].flatMap(color => {
        const key = `QE_${color}`
        return grcwaFourierData.map(d => ({
          x: grcwaXScale(d.nG),
          y: grcwaYScale(d[key]),
          color: colorMap[color],
          line1: `nG=${d.nG} (${d.harmonics} ${t('harmonics', '하모닉스')})`,
          line2: `${color} QE = ${d[key].toFixed(4)}${d.unstable_R && color === 'R' ? ' ⚠' : ''}`,
        }))
      })
    case 'angle':
      return ['R', 'G', 'B'].flatMap(color => {
        const key = `QE_${color}`
        return angleData.map(d => ({
          x: angleXScale(d.cra),
          y: angleYScale(d[key]),
          color: colorMap[color],
          line1: `CRA = ${d.cra}°`,
          line2: `${color} QE = ${d[key].toFixed(4)}`,
        }))
      })
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
.per-color-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.per-color-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

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

.svg-wrapper { margin-bottom: 16px; }

.chart-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
  cursor: crosshair;
}

.axis-label { font-size: 9px; fill: var(--vp-c-text-2); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.legend-label { font-size: 8.5px; fill: var(--vp-c-text-2); font-weight: 500; }
.bar-label { font-size: 8px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }

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
  display: block;
  font-size: 0.85em;
  margin-bottom: 2px;
}

.summary-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  font-size: 1.05em;
}

.data-table-details { margin-top: 4px; }

.data-table-summary {
  cursor: pointer;
  font-size: 0.9em;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  padding: 6px 0;
  user-select: none;
}

.data-table-summary:hover { color: var(--vp-c-brand-2); }

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

.data-table thead { background: var(--vp-c-bg); }

.data-table th {
  padding: 8px 10px;
  text-align: right;
  font-weight: 700;
  color: var(--vp-c-text-1);
  border-bottom: 2px solid var(--vp-c-divider);
  white-space: nowrap;
}

.data-table th:first-child { text-align: left; }

.data-table td {
  padding: 5px 10px;
  text-align: right;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-1);
}

.data-table td:first-child { text-align: left; font-weight: 600; }
.data-table tbody tr:hover { background: var(--vp-c-bg); }
.data-table tbody tr:last-child td { border-bottom: none; }
.cell-grcwa { color: #2196F3; }
.cell-torcwa { color: #FF9800; }

@media (max-width: 640px) {
  .per-color-container { padding: 1rem; }
  .tab-btn { padding: 6px 12px; font-size: 0.82em; }
  .summary-card { min-width: 100%; }
  .data-table { font-size: 0.75em; }
  .data-table th, .data-table td { padding: 4px 6px; }
}
</style>
