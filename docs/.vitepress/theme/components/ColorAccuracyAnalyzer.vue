<template>
  <div class="ca-container">
    <h4>{{ t('Color Accuracy Analyzer', '색 정확도 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Compute color reproduction accuracy (deltaE) for ColorChecker patches using TMM-based QE spectra and a least-squares Color Correction Matrix.',
        'TMM 기반 QE 스펙트럼과 최소제곱 색 보정 행렬(CCM)을 사용하여 ColorChecker 패치의 색 재현 정확도(deltaE)를 계산합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThickness.toFixed(1) }} &mu;m</strong>
        </label>
        <input type="range" min="1" max="5" step="0.1" v-model.number="siThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('CF Bandwidth (FWHM)', 'CF 대역폭 (FWHM)') }}: <strong>{{ cfBandwidth }} nm</strong>
        </label>
        <input type="range" min="50" max="150" step="5" v-model.number="cfBandwidth" class="ctrl-range" />
      </div>
      <div class="chart-toggle">
        <button
          :class="['toggle-btn', { active: chartType === 'classic' }]"
          @click="chartType = 'classic'"
        >Classic 24</button>
        <button
          :class="['toggle-btn', { active: chartType === 'sg' }]"
          @click="chartType = 'sg'"
        >SG 140</button>
      </div>
      <div class="chart-toggle">
        <button
          v-for="m in DE_METHODS" :key="m.key"
          :class="['toggle-btn', { active: deMethod === m.key }]"
          @click="deMethod = m.key"
        >{{ m.label }}</button>
      </div>
    </div>

    <!-- Summary bar -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Average deltaE', '평균 deltaE') }}</div>
        <div class="result-value highlight">{{ avgDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Max deltaE', '최대 deltaE') }}</div>
        <div class="result-value">{{ maxDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Excellent (deltaE < 3)', '우수 (deltaE < 3)') }}</div>
        <div class="result-value highlight">{{ excellentPct.toFixed(0) }}%</div>
      </div>
    </div>

    <!-- QE Spectrum -->
    <div class="chart-section">
      <h5>{{ t('Sensor QE Spectrum', '센서 QE 스펙트럼') }}</h5>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${qeW} ${qeH}`" class="de-svg">
          <line
            v-for="tick in qeYTicks" :key="'qyg'+tick"
            :x1="qePad.left" :y1="qeYScale(tick)"
            :x2="qePad.left + qePlotW" :y2="qeYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line :x1="qePad.left" :y1="qePad.top" :x2="qePad.left" :y2="qePad.top + qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="qePad.left" :y1="qePad.top + qePlotH" :x2="qePad.left + qePlotW" :y2="qePad.top + qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <text
            v-for="tick in qeYTicks" :key="'qyl'+tick"
            :x="qePad.left - 6" :y="qeYScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick.toFixed(1) }}</text>
          <text
            v-for="wl in qeXTicks" :key="'qxl'+wl"
            :x="qeXScale(wl)" :y="qePad.top + qePlotH + 14"
            text-anchor="middle" class="tick-label"
          >{{ wl }}</text>
          <path :d="qePathR" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.8" />
          <path :d="qePathG" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.8" />
          <path :d="qePathB" fill="none" stroke="#3498db" stroke-width="2" opacity="0.8" />
          <text :x="10" :y="qePad.top + qePlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${qePad.top + qePlotH / 2})`">QE</text>
          <text :x="qePad.left + qePlotW / 2" :y="qePad.top + qePlotH + 26" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
          <!-- Legend -->
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 6" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 6" stroke="#e74c3c" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 9" class="tick-label">Red</text>
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 18" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 18" stroke="#27ae60" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 21" class="tick-label">Green</text>
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 30" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 30" stroke="#3498db" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 33" class="tick-label">Blue</text>
        </svg>
      </div>
    </div>

    <!-- CCM Matrix -->
    <div class="ccm-section" v-if="ccmMatrix">
      <h5>{{ t('Color Correction Matrix (CCM)', '색 보정 행렬 (CCM)') }}</h5>
      <table class="ccm-table">
        <thead>
          <tr><th></th><th>R<sub>in</sub></th><th>G<sub>in</sub></th><th>B<sub>in</sub></th></tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in ccmMatrix" :key="i">
            <th>{{ ['R','G','B'][i] }}<sub>out</sub></th>
            <td v-for="(v, j) in row" :key="j" :class="{ 'ccm-diag': i === j }">{{ v.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Patch grid -->
    <div class="patch-section">
      <div class="section-header">
        <h5>{{ t('ColorChecker Patches', 'ColorChecker 패치') }}
          <span class="patch-count">({{ activePatches.length }})</span>
        </h5>
        <div class="chart-toggle view-toggle">
          <button
            :class="['toggle-btn', { active: patchView === 'swatch' }]"
            @click="patchView = 'swatch'"
          >{{ t('Swatch', '스와치') }}</button>
          <button
            :class="['toggle-btn', { active: patchView === 'heatmap' }]"
            @click="patchView = 'heatmap'"
          >{{ t('Heatmap', '히트맵') }}</button>
        </div>
      </div>
      <div :class="['patch-grid', chartType === 'sg' ? 'patch-grid-sg' : 'patch-grid-classic']">
        <div v-for="(patch, idx) in patchResults" :key="idx" class="patch-cell">
          <!-- Swatch view -->
          <template v-if="patchView === 'swatch'">
            <div class="patch-swatch">
              <div class="patch-ref" :style="{ background: rgbStr(patch.refSrgb) }"></div>
              <div class="patch-sensor" :style="{ background: rgbStr(patch.corrSrgb) }"></div>
            </div>
          </template>
          <!-- Heatmap view -->
          <template v-else>
            <div class="heatmap-cell" :style="{ background: deHeatBg(patch.deltaE) }">
              <span class="heatmap-val" :style="{ color: deHeatText(patch.deltaE) }">{{ patch.deltaE.toFixed(chartType === 'sg' ? 0 : 1) }}</span>
            </div>
          </template>
          <div v-if="chartType === 'classic'" class="patch-name">{{ patch.name }}</div>
          <div
            v-if="patchView === 'swatch'"
            class="patch-de"
            :class="{ 'patch-de-sg': chartType === 'sg' }"
            :style="{ color: deColor(patch.deltaE) }"
          >{{ patch.deltaE.toFixed(chartType === 'sg' ? 0 : 1) }}</div>
        </div>
      </div>
    </div>

    <!-- DeltaE bar chart -->
    <div class="chart-section">
      <h5>{{ t('deltaE per Patch', '패치별 deltaE') }}</h5>
      <div class="svg-wrapper">
        <svg
          ref="deChartSvg"
          :viewBox="`0 0 ${chartW} ${chartH}`"
          class="de-svg"
          @mousemove="onChartMouseMove"
          @mouseleave="chartHover = null"
        >
          <line
            v-for="tick in yTicks" :key="'yg'+tick"
            :x1="pad.left" :y1="yScale(tick)"
            :x2="pad.left + plotW" :y2="yScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            :x1="pad.left" :y1="yScale(3)"
            :x2="pad.left + plotW" :y2="yScale(3)"
            stroke="#27ae60" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(3) + 3" class="ref-label" fill="#27ae60">3</text>
          <line
            :x1="pad.left" :y1="yScale(6)"
            :x2="pad.left + plotW" :y2="yScale(6)"
            stroke="#e67e22" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(6) + 3" class="ref-label" fill="#e67e22">6</text>
          <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <text
            v-for="tick in yTicks" :key="'yl'+tick"
            :x="pad.left - 6" :y="yScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick }}</text>
          <rect
            v-for="(patch, idx) in patchResults" :key="'bar'+idx"
            :x="barX(idx)"
            :y="yScale(patch.deltaE)"
            :width="barWidth"
            :height="Math.max(0, pad.top + plotH - yScale(patch.deltaE))"
            :fill="deColor(patch.deltaE)"
            opacity="0.8"
            rx="1"
          />
          <template v-if="chartType === 'classic'">
            <text
              v-for="(patch, idx) in patchResults" :key="'xl'+idx"
              :x="barX(idx) + barWidth / 2"
              :y="pad.top + plotH + 10"
              text-anchor="end"
              class="tick-label"
              :transform="`rotate(-45, ${barX(idx) + barWidth / 2}, ${pad.top + plotH + 10})`"
            >{{ patch.name.substring(0, 8) }}</text>
          </template>
          <text :x="10" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`">{{ deAxisLabel }}</text>
          <template v-if="chartHover">
            <rect
              :x="chartHover.tx" :y="pad.top + 4"
              width="140" height="34" rx="4"
              fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95"
            />
            <text :x="chartHover.tx + 6" :y="pad.top + 18" class="tooltip-text">{{ chartHover.name }}</text>
            <text :x="chartHover.tx + 6" :y="pad.top + 30" class="tooltip-text">&Delta;E = {{ chartHover.de.toFixed(2) }}</text>
          </template>
        </svg>
      </div>
    </div>

    <!-- Export -->
    <div class="export-row">
      <button class="export-btn" @click="exportCsv">
        {{ t('Export CSV', 'CSV 내보내기') }}
      </button>
      <button class="export-btn" @click="exportPng">
        {{ t('Export PNG', 'PNG 내보내기') }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'

const { t } = useLocale()

// ---- Controls ----
const siThickness = ref(3.0)
const cfBandwidth = ref(100)
const chartType = ref<'classic' | 'sg'>('classic')
const patchView = ref<'swatch' | 'heatmap'>('swatch')

type DEMethod = 'cie76' | 'cie94' | 'ciede2000'
const deMethod = ref<DEMethod>('ciede2000')
const DE_METHODS = [
  { key: 'cie76' as DEMethod, label: 'CIE76' },
  { key: 'cie94' as DEMethod, label: 'CIE94' },
  { key: 'ciede2000' as DEMethod, label: 'CIEDE2000' },
]

// ---- ColorChecker Classic 24 data (standard 24 patches, 7 wavelengths 400-700nm @ 50nm) ----
const CLASSIC_PATCHES: { name: string; srgb: number[]; refl: number[] }[] = [
  { name: 'Dark Skin',     srgb: [115,82,68],   refl: [0.055,0.058,0.069,0.099,0.132,0.143,0.146] },
  { name: 'Light Skin',    srgb: [194,150,130],  refl: [0.092,0.107,0.152,0.191,0.260,0.286,0.275] },
  { name: 'Blue Sky',      srgb: [98,122,157],   refl: [0.117,0.143,0.178,0.193,0.149,0.115,0.108] },
  { name: 'Foliage',       srgb: [87,108,67],    refl: [0.042,0.051,0.085,0.131,0.099,0.068,0.060] },
  { name: 'Blue Flower',   srgb: [133,128,177],  refl: [0.137,0.132,0.128,0.120,0.133,0.154,0.228] },
  { name: 'Bluish Green',  srgb: [103,189,170],  refl: [0.131,0.226,0.318,0.341,0.301,0.232,0.218] },
  { name: 'Orange',        srgb: [214,126,44],   refl: [0.050,0.054,0.063,0.170,0.395,0.413,0.266] },
  { name: 'Purplish Blue', srgb: [80,91,166],    refl: [0.153,0.128,0.091,0.058,0.042,0.060,0.157] },
  { name: 'Moderate Red',  srgb: [193,90,99],    refl: [0.065,0.053,0.058,0.078,0.200,0.305,0.231] },
  { name: 'Purple',        srgb: [94,60,108],    refl: [0.065,0.052,0.040,0.038,0.043,0.065,0.100] },
  { name: 'Yellow Green',  srgb: [157,188,64],   refl: [0.047,0.065,0.172,0.331,0.350,0.237,0.120] },
  { name: 'Orange Yellow', srgb: [224,163,46],   refl: [0.050,0.058,0.093,0.284,0.481,0.465,0.287] },
  { name: 'Blue',          srgb: [56,61,150],    refl: [0.142,0.103,0.056,0.033,0.026,0.032,0.088] },
  { name: 'Green',         srgb: [70,148,73],    refl: [0.035,0.060,0.140,0.175,0.099,0.058,0.044] },
  { name: 'Red',           srgb: [175,54,60],    refl: [0.043,0.036,0.035,0.047,0.153,0.331,0.271] },
  { name: 'Yellow',        srgb: [231,199,31],   refl: [0.042,0.054,0.124,0.397,0.559,0.504,0.316] },
  { name: 'Magenta',       srgb: [187,86,149],   refl: [0.094,0.065,0.067,0.073,0.127,0.219,0.276] },
  { name: 'Cyan',          srgb: [8,133,161],    refl: [0.073,0.139,0.210,0.219,0.137,0.083,0.076] },
  { name: 'White',         srgb: [243,243,242],  refl: [0.875,0.886,0.892,0.894,0.892,0.882,0.870] },
  { name: 'Neutral 8',     srgb: [200,200,200],  refl: [0.570,0.578,0.584,0.586,0.585,0.578,0.572] },
  { name: 'Neutral 6.5',   srgb: [160,160,160],  refl: [0.354,0.362,0.366,0.368,0.366,0.361,0.356] },
  { name: 'Neutral 5',     srgb: [122,122,121],  refl: [0.195,0.200,0.204,0.206,0.205,0.201,0.197] },
  { name: 'Neutral 3.5',   srgb: [85,85,85],     refl: [0.091,0.094,0.096,0.097,0.096,0.094,0.092] },
  { name: 'Black',         srgb: [52,52,52],      refl: [0.032,0.033,0.034,0.034,0.034,0.033,0.032] },
]

// ---- ColorChecker SG 140 data (L*a*b* D50/2°) ----
const SG_LAB: number[][] = [
  [96.04,-0.12,0.31],[53.35,-36.85,15.22],[70.48,-32.43,0.55],[48.52,-28.76,-8.49],[39.43,-16.58,-26.34],
  [55.28,9.35,-34.09],[53.34,14.25,-13.53],[80.57,3.73,-7.71],[50.72,51.66,-14.77],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[46.47,52.54,19.94],[50.76,50.96,27.64],[66.47,35.85,60.44],[62.27,33.28,57.71],
  [72.70,-0.97,69.36],[53.61,10.59,53.90],[44.19,-14.25,38.96],[35.17,-13.17,22.74],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[35.75,60.42,34.16],[40.06,48.08,27.44],[30.87,23.44,22.47],[25.53,13.84,15.33],
  [52.91,-1.37,55.42],[39.77,17.41,47.95],[27.38,-0.63,30.60],[20.17,-0.67,13.69],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[42.35,12.63,-44.77],[52.17,2.07,-30.04],[51.20,49.36,-16.78],[60.21,25.51,2.46],
  [53.13,12.31,17.49],[71.82,-23.83,57.10],[60.94,-29.79,41.53],[49.48,-29.99,22.66],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[38.63,12.21,-45.58],[62.86,36.15,57.33],[71.93,-23.53,57.21],[55.76,-38.34,31.73],
  [40.02,10.05,-44.52],[30.36,22.89,-20.64],[72.39,-27.57,1.35],[49.06,30.18,-4.58],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[42.53,52.21,28.87],[63.85,18.23,18.15],[71.15,11.26,17.58],[78.36,0.41,0.10],
  [64.16,-18.45,-17.62],[60.06,26.25,-19.86],[61.77,0.63,0.51],[50.50,-31.76,-27.86],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[83.68,3.73,79.79],[55.09,-38.56,32.12],[31.77,1.29,-23.25],[81.68,1.02,79.96],
  [52.00,48.63,-14.84],[32.79,18.35,21.36],[72.91,0.69,0.10],[79.66,-1.02,75.21],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[62.67,37.32,68.03],[39.46,49.57,31.76],[72.67,-23.21,58.12],[51.69,-28.81,49.32],
  [51.51,54.10,25.56],[81.22,2.51,80.51],[40.02,50.40,27.55],[30.25,26.53,-22.62],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[43.25,15.29,24.73],[61.50,10.04,17.26],[68.80,12.52,15.81],[78.36,0.41,0.10],
  [57.44,-8.73,-10.74],[50.62,-28.32,-1.05],[48.97,-1.02,-0.31],[23.53,2.10,-2.48],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[53.72,7.22,-25.45],[59.75,-28.34,-26.83],[49.30,-1.96,44.83],[34.95,14.40,24.37],
  [38.93,55.97,29.59],[67.52,-30.41,-0.67],[92.71,0.06,1.45],[32.65,-23.41,0.65],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[44.44,23.12,31.85],[38.52,11.43,-45.87],[32.97,16.78,-30.95],[53.53,-41.67,34.15],
  [42.68,15.63,-44.01],[31.50,24.37,19.33],[68.48,0.37,0.37],[60.58,37.49,67.63],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[66.84,17.81,-21.56],[42.15,53.06,28.98],[43.26,48.54,-5.29],[42.83,54.65,26.66],
  [62.32,-5.55,63.37],[50.45,50.81,-14.56],[34.63,10.14,27.26],[26.47,15.39,12.83],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[49.81,-3.25,49.37],[38.73,-16.30,30.58],[28.61,-10.42,20.02],[21.17,-7.13,10.92],
  [52.72,47.41,56.15],[40.22,42.73,28.43],[30.49,28.63,16.00],[24.54,15.14,6.43],[96.04,-0.12,0.31],
  [96.04,-0.12,0.31],[81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],[96.04,-0.12,0.31],
]

const SG_NAMES: string[] = (() => {
  const rows = 'ABCDEFGHIJKLMN'
  const names: string[] = []
  for (let r = 0; r < 14; r++)
    for (let c = 1; c <= 10; c++)
      names.push(`${rows[r]}${c}`)
  return names
})()

// ---- Constants ----
const D65 = [82.75, 109.35, 117.01, 114.86, 100.0, 90.01, 71.61]
const WL_UM = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
const LN2 = Math.log(2)
const CF_CENTERS = { red: 0.620, green: 0.530, blue: 0.450 }

// ---- QE Spectrum (31 points, 400-700nm @ 10nm) ----
const QE_WL_NM = Array.from({ length: 31 }, (_, i) => 400 + i * 10)
const QE_WL_UM = QE_WL_NM.map(w => w / 1000)

// ---- Color filter & TMM ----
function cfTransmittance(wlUm: number, centerUm: number, fwhmNm: number): number {
  const fwhmUm = fwhmNm / 1000
  return Math.exp(-4 * LN2 * ((wlUm - centerUm) / fwhmUm) ** 2)
}

function computeQE(color: 'red' | 'green' | 'blue', wlUm: number): number {
  const stack = defaultBsiStack(color, siThickness.value)
  const result = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
  return result.layerA[SI_LAYER_IDX]
}

function sensorResponse(color: 'red' | 'green' | 'blue', wlUm: number): number {
  const qe = computeQE(color, wlUm)
  const cfT = cfTransmittance(wlUm, CF_CENTERS[color], cfBandwidth.value)
  return qe * cfT
}

// ---- QE Spectra computed ----
const qeSpectra = computed(() => ({
  r: QE_WL_UM.map(wl => sensorResponse('red', wl)),
  g: QE_WL_UM.map(wl => sensorResponse('green', wl)),
  b: QE_WL_UM.map(wl => sensorResponse('blue', wl)),
}))

// QE chart dimensions
const qeW = 600
const qeH = 180
const qePad = { top: 12, right: 20, bottom: 32, left: 40 }
const qePlotW = qeW - qePad.left - qePad.right
const qePlotH = qeH - qePad.top - qePad.bottom

const qeYMax = computed(() => {
  const all = [...qeSpectra.value.r, ...qeSpectra.value.g, ...qeSpectra.value.b]
  return Math.max(0.1, Math.ceil(Math.max(...all) * 10) / 10)
})

const qeYTicks = computed(() => {
  const ticks: number[] = []
  const step = qeYMax.value <= 0.5 ? 0.1 : 0.2
  for (let v = 0; v <= qeYMax.value + 0.001; v += step) ticks.push(Math.round(v * 100) / 100)
  return ticks
})

const qeXTicks = [400, 450, 500, 550, 600, 650, 700]

function qeYScale(v: number): number {
  return qePad.top + qePlotH - (v / qeYMax.value) * qePlotH
}

function qeXScale(wlNm: number): number {
  return qePad.left + ((wlNm - 400) / 300) * qePlotW
}

function qePathD(data: number[]): string {
  return data.map((v, i) => {
    const x = qePad.left + (i / (data.length - 1)) * qePlotW
    const y = qePad.top + qePlotH - (v / qeYMax.value) * qePlotH
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join('')
}

const qePathR = computed(() => qePathD(qeSpectra.value.r))
const qePathG = computed(() => qePathD(qeSpectra.value.g))
const qePathB = computed(() => qePathD(qeSpectra.value.b))

// ---- sRGB / linear color math ----
function srgbToLinear(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
}

function linearToSrgb(c: number): number {
  const v = Math.max(0, Math.min(1, c))
  const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055
  return Math.round(Math.max(0, Math.min(255, s * 255)))
}

const SRGB_TO_XYZ = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
]

function srgbToXYZ(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0]), g = srgbToLinear(srgb[1]), b = srgbToLinear(srgb[2])
  return [
    SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b,
    SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b,
    SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b,
  ]
}

const XYZ_TO_SRGB = [
  [ 3.2404542, -1.5371385, -0.4985314],
  [-0.9692660,  1.8760108,  0.0415560],
  [ 0.0556434, -0.2040259,  1.0572252],
]

function xyzToLinearRgb(xyz: number[]): number[] {
  return [
    XYZ_TO_SRGB[0][0] * xyz[0] + XYZ_TO_SRGB[0][1] * xyz[1] + XYZ_TO_SRGB[0][2] * xyz[2],
    XYZ_TO_SRGB[1][0] * xyz[0] + XYZ_TO_SRGB[1][1] * xyz[1] + XYZ_TO_SRGB[1][2] * xyz[2],
    XYZ_TO_SRGB[2][0] * xyz[0] + XYZ_TO_SRGB[2][1] * xyz[1] + XYZ_TO_SRGB[2][2] * xyz[2],
  ]
}

// ---- XYZ ↔ Lab (D65) ----
const D65_WP = { Xn: 0.9505, Yn: 1.0, Zn: 1.089 }

function labF(t: number): number {
  return t > 0.008856 ? Math.pow(t, 1 / 3) : 7.787 * t + 16 / 116
}

function xyzToLab(xyz: number[]): number[] {
  const fx = labF(xyz[0] / D65_WP.Xn)
  const fy = labF(xyz[1] / D65_WP.Yn)
  const fz = labF(xyz[2] / D65_WP.Zn)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

// ---- deltaE methods ----
function deltaE76(lab1: number[], lab2: number[]): number {
  return Math.sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2)
}

function deltaE94(lab1: number[], lab2: number[]): number {
  const dL = lab1[0] - lab2[0]
  const da = lab1[1] - lab2[1], db = lab1[2] - lab2[2]
  const C1 = Math.sqrt(lab1[1] ** 2 + lab1[2] ** 2)
  const dC = C1 - Math.sqrt(lab2[1] ** 2 + lab2[2] ** 2)
  const dH2 = da ** 2 + db ** 2 - dC ** 2
  const SC = 1 + 0.045 * C1, SH = 1 + 0.015 * C1
  return Math.sqrt(dL ** 2 + (dC / SC) ** 2 + (dH2 > 0 ? dH2 : 0) / SH ** 2)
}

function deltaE2000(lab1: number[], lab2: number[]): number {
  const L1 = lab1[0], a1 = lab1[1], b1 = lab1[2]
  const L2 = lab2[0], a2 = lab2[1], b2 = lab2[2]
  const deg = Math.PI / 180
  const Cab1 = Math.sqrt(a1 ** 2 + b1 ** 2), Cab2 = Math.sqrt(a2 ** 2 + b2 ** 2)
  const CabAvg = (Cab1 + Cab2) / 2, CabAvg7 = CabAvg ** 7
  const G = 0.5 * (1 - Math.sqrt(CabAvg7 / (CabAvg7 + 25 ** 7)))
  const a1p = a1 * (1 + G), a2p = a2 * (1 + G)
  const C1p = Math.sqrt(a1p ** 2 + b1 ** 2), C2p = Math.sqrt(a2p ** 2 + b2 ** 2)
  let h1p = Math.atan2(b1, a1p); if (h1p < 0) h1p += 2 * Math.PI
  let h2p = Math.atan2(b2, a2p); if (h2p < 0) h2p += 2 * Math.PI
  const dLp = L2 - L1, dCp = C2p - C1p
  let dhp: number
  if (C1p * C2p === 0) dhp = 0
  else if (Math.abs(h2p - h1p) <= Math.PI) dhp = h2p - h1p
  else if (h2p - h1p > Math.PI) dhp = h2p - h1p - 2 * Math.PI
  else dhp = h2p - h1p + 2 * Math.PI
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dhp / 2)
  const Lpm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2
  let hpm: number
  if (C1p * C2p === 0) hpm = h1p + h2p
  else if (Math.abs(h1p - h2p) <= Math.PI) hpm = (h1p + h2p) / 2
  else if (h1p + h2p < 2 * Math.PI) hpm = (h1p + h2p + 2 * Math.PI) / 2
  else hpm = (h1p + h2p - 2 * Math.PI) / 2
  const T = 1 - 0.17 * Math.cos(hpm - 30 * deg) + 0.24 * Math.cos(2 * hpm) + 0.32 * Math.cos(3 * hpm + 6 * deg) - 0.20 * Math.cos(4 * hpm - 63 * deg)
  const SL = 1 + 0.015 * (Lpm - 50) ** 2 / Math.sqrt(20 + (Lpm - 50) ** 2)
  const SC = 1 + 0.045 * Cpm, SH = 1 + 0.015 * Cpm * T
  const Cpm7 = Cpm ** 7, RC = 2 * Math.sqrt(Cpm7 / (Cpm7 + 25 ** 7))
  const dTheta = 30 * deg * Math.exp(-(((hpm / deg - 275) / 25) ** 2))
  const RT = -Math.sin(2 * dTheta) * RC
  return Math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))
}

function calcDeltaE(lab1: number[], lab2: number[]): number {
  switch (deMethod.value) {
    case 'cie94': return deltaE94(lab1, lab2)
    case 'ciede2000': return deltaE2000(lab1, lab2)
    default: return deltaE76(lab1, lab2)
  }
}

const deAxisLabel = computed(() => {
  switch (deMethod.value) {
    case 'cie94': return '\u0394E*94'
    case 'ciede2000': return '\u0394E00'
    default: return '\u0394E*ab'
  }
})

// ---- Lab (D50) → sRGB (D65) via Bradford ----
const D50_WP = { Xn: 0.9642, Yn: 1.0, Zn: 0.8251 }
const BRAD_M = [[ 0.8951, 0.2664,-0.1614],[-0.7502, 1.7135, 0.0367],[ 0.0389,-0.0685, 1.0296]]
const BRAD_MI = [[ 0.9870,-0.1471, 0.1600],[ 0.4323, 0.5184, 0.0493],[-0.0085, 0.0400, 0.9685]]
const D50_XYZ = [0.9642, 1.0, 0.8251]
const D65_XYZ = [0.9505, 1.0, 1.089]

const bradAdapt: number[][] = (() => {
  const coneS = [0, 0, 0].map((_, r) => BRAD_M[r][0] * D50_XYZ[0] + BRAD_M[r][1] * D50_XYZ[1] + BRAD_M[r][2] * D50_XYZ[2])
  const coneD = [0, 0, 0].map((_, r) => BRAD_M[r][0] * D65_XYZ[0] + BRAD_M[r][1] * D65_XYZ[1] + BRAD_M[r][2] * D65_XYZ[2])
  const scale = [[coneD[0]/coneS[0],0,0],[0,coneD[1]/coneS[1],0],[0,0,coneD[2]/coneS[2]]]
  const tmp: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) tmp[i][j] += scale[i][k] * BRAD_M[k][j]
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) result[i][j] += BRAD_MI[i][k] * tmp[k][j]
  return result
})()

function labFInv(t: number): number {
  return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787
}

function labToSrgb(lab: number[]): number[] {
  const fy = (lab[0] + 16) / 116, fx = lab[1] / 500 + fy, fz = fy - lab[2] / 200
  const xyzD50 = [D50_WP.Xn * labFInv(fx), D50_WP.Yn * labFInv(fy), D50_WP.Zn * labFInv(fz)]
  const xyzD65 = [0, 1, 2].map(i => bradAdapt[i][0] * xyzD50[0] + bradAdapt[i][1] * xyzD50[1] + bradAdapt[i][2] * xyzD50[2])
  const lin = xyzToLinearRgb(xyzD65)
  return [linearToSrgb(lin[0]), linearToSrgb(lin[1]), linearToSrgb(lin[2])]
}

// ---- Gaussian basis spectral reconstruction ----
function reconstructRefl(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0]), g = srgbToLinear(srgb[1]), b = srgbToLinear(srgb[2])
  const wlNm = [400, 450, 500, 550, 600, 650, 700]
  return wlNm.map(wl => {
    const bR = Math.exp(-0.5 * ((wl - 610) / 55) ** 2)
    const bG = Math.exp(-0.5 * ((wl - 535) / 50) ** 2)
    const bB = Math.exp(-0.5 * ((wl - 445) / 35) ** 2)
    return Math.max(0.01, Math.min(1, r * bR + g * bG + b * bB))
  })
}

// ---- Active patches ----
interface PatchData { name: string; srgb: number[]; refl: number[] }

const activePatches = computed<PatchData[]>(() => {
  if (chartType.value === 'classic') return CLASSIC_PATCHES
  return SG_LAB.map((lab, idx) => {
    const srgb = labToSrgb(lab)
    return { name: SG_NAMES[idx], srgb, refl: reconstructRefl(srgb) }
  })
})

// ---- 3x3 matrix ops ----
function mat3x3Inverse(m: number[][]): number[][] | null {
  const [a,b,c] = m[0], [d,e,f] = m[1], [g,h,i] = m[2]
  const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
  if (Math.abs(det) < 1e-12) return null
  const inv = 1 / det
  return [
    [(e*i-f*h)*inv, (c*h-b*i)*inv, (b*f-c*e)*inv],
    [(f*g-d*i)*inv, (a*i-c*g)*inv, (c*d-a*f)*inv],
    [(d*h-e*g)*inv, (b*g-a*h)*inv, (a*e-b*d)*inv],
  ]
}

function matMul3x3(a: number[][], b: number[][]): number[][] {
  const r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i][j] += a[i][k] * b[k][j]
  return r
}

function matTransposeMulNx3(s: number[][]): number[][] {
  const n = s.length, r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < n; k++) r[i][j] += s[k][i] * s[k][j]
  return r
}

function matSTmulT(s: number[][], tgt: number[][]): number[][] {
  const n = s.length, r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < n; k++) r[i][j] += s[k][i] * tgt[k][j]
  return r
}

function mat3Vec(m: number[][], v: number[]): number[] {
  return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2], m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2], m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]
}

// ---- Pipeline: compute CCM + sensor data once ----
const pipeline = computed(() => {
  const patches = activePatches.value
  const sensorR = WL_UM.map(wl => sensorResponse('red', wl))
  const sensorG = WL_UM.map(wl => sensorResponse('green', wl))
  const sensorB = WL_UM.map(wl => sensorResponse('blue', wl))
  const sensorRGB = patches.map(patch => {
    let rS = 0, gS = 0, bS = 0
    for (let i = 0; i < 7; i++) { const w = patch.refl[i] * D65[i]; rS += w * sensorR[i]; gS += w * sensorG[i]; bS += w * sensorB[i] }
    return [rS, gS, bS]
  })
  const targetLinear = patches.map(patch => [srgbToLinear(patch.srgb[0]), srgbToLinear(patch.srgb[1]), srgbToLinear(patch.srgb[2])])
  const STS = matTransposeMulNx3(sensorRGB)
  const STSinv = mat3x3Inverse(STS)
  if (!STSinv) return { ccm: null as number[][] | null, sensorRGB, targetLinear }
  const CCM = matMul3x3(STSinv, matSTmulT(sensorRGB, targetLinear))
  return { ccm: CCM, sensorRGB, targetLinear }
})

const ccmMatrix = computed(() => pipeline.value.ccm)

// ---- Main patch results ----
interface PatchResult { name: string; refSrgb: number[]; corrSrgb: number[]; deltaE: number }

const patchResults = computed<PatchResult[]>(() => {
  const patches = activePatches.value
  const { ccm, sensorRGB } = pipeline.value
  if (!ccm) return patches.map(p => ({ name: p.name, refSrgb: p.srgb, corrSrgb: [128,128,128], deltaE: 99 }))
  return patches.map((patch, idx) => {
    const corrLinear = mat3Vec(ccm, sensorRGB[idx])
    const corrSrgb = [linearToSrgb(corrLinear[0]), linearToSrgb(corrLinear[1]), linearToSrgb(corrLinear[2])]
    const refLab = xyzToLab(srgbToXYZ(patch.srgb))
    const cc = corrLinear.map(v => Math.max(0, Math.min(1, v)))
    const corrXYZ = [0, 1, 2].map(r => SRGB_TO_XYZ[r][0]*cc[0] + SRGB_TO_XYZ[r][1]*cc[1] + SRGB_TO_XYZ[r][2]*cc[2])
    const corrLab = xyzToLab(corrXYZ)
    return { name: patch.name, refSrgb: patch.srgb, corrSrgb, deltaE: calcDeltaE(refLab, corrLab) }
  })
})

const avgDeltaE = computed(() => { const r = patchResults.value; return r.reduce((s, p) => s + p.deltaE, 0) / r.length })
const maxDeltaE = computed(() => Math.max(...patchResults.value.map(p => p.deltaE)))
const excellentPct = computed(() => { const r = patchResults.value; return (r.filter(p => p.deltaE < 3).length / r.length) * 100 })

// ---- Helpers ----
function rgbStr(rgb: number[]): string { return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})` }

function deColor(de: number): string {
  if (de < 3) return '#27ae60'
  if (de < 6) return '#e67e22'
  return '#e74c3c'
}

function deHeatBg(de: number): string {
  const hue = Math.max(0, 120 * (1 - Math.min(de, 12) / 12))
  return `hsl(${hue}, 65%, 72%)`
}

function deHeatText(de: number): string {
  return de > 9 ? '#fff' : '#222'
}

// ---- DeltaE bar chart ----
const chartW = 600
const chartH = 200
const pad = { top: 16, right: 30, bottom: 60, left: 40 }
const plotW = chartW - pad.left - pad.right
const plotH = chartH - pad.top - pad.bottom

const yMax = computed(() => Math.max(8, Math.ceil(maxDeltaE.value + 2)))

const yTicks = computed(() => {
  const ticks: number[] = []
  const step = yMax.value <= 12 ? 2 : 5
  for (let v = 0; v <= yMax.value; v += step) ticks.push(v)
  return ticks
})

function yScale(v: number): number { return pad.top + plotH - (v / yMax.value) * plotH }

const barGap = computed(() => chartType.value === 'sg' ? 0.5 : 2)
const barWidth = computed(() => Math.max(2, (plotW - barGap.value * (activePatches.value.length - 1)) / activePatches.value.length))

function barX(idx: number): number { return pad.left + idx * (barWidth.value + barGap.value) }

const chartHover = ref<{ tx: number; name: string; de: number } | null>(null)
const deChartSvg = ref<SVGSVGElement | null>(null)

function onChartMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mouseX = (event.clientX - rect.left) * (chartW / rect.width)
  const idx = Math.floor((mouseX - pad.left) / (barWidth.value + barGap.value))
  if (idx >= 0 && idx < patchResults.value.length) {
    const patch = patchResults.value[idx]
    const bx = barX(idx) + barWidth.value / 2
    chartHover.value = { tx: bx + 150 > chartW - pad.right ? bx - 150 : bx + 10, name: patch.name, de: patch.deltaE }
  } else {
    chartHover.value = null
  }
}

// ---- Export CSV ----
function exportCsv() {
  const method = deMethod.value.toUpperCase()
  const header = `Patch,Ref_R,Ref_G,Ref_B,Corr_R,Corr_G,Corr_B,${method}\n`
  const rows = patchResults.value.map(p =>
    `${p.name},${p.refSrgb.join(',')},${p.corrSrgb.join(',')},${p.deltaE.toFixed(4)}`
  ).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `colorchecker-${chartType.value}-${method}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

// ---- Export PNG (canvas redraw) ----
function exportPng() {
  const results = patchResults.value
  const dpr = 2
  const canvas = document.createElement('canvas')
  canvas.width = chartW * dpr
  canvas.height = chartH * dpr
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)

  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, chartW, chartH)

  // Grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 0.5
  ctx.setLineDash([3, 3])
  for (const tick of yTicks.value) {
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(tick)); ctx.lineTo(pad.left + plotW, yScale(tick)); ctx.stroke()
  }
  ctx.setLineDash([])

  // Threshold lines
  ctx.setLineDash([6, 3])
  ctx.lineWidth = 1
  ctx.strokeStyle = '#27ae60'
  ctx.beginPath(); ctx.moveTo(pad.left, yScale(3)); ctx.lineTo(pad.left + plotW, yScale(3)); ctx.stroke()
  ctx.strokeStyle = '#e67e22'
  ctx.beginPath(); ctx.moveTo(pad.left, yScale(6)); ctx.lineTo(pad.left + plotW, yScale(6)); ctx.stroke()
  ctx.setLineDash([])

  // Axes
  ctx.strokeStyle = '#333'
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke()

  // Y ticks
  ctx.fillStyle = '#666'
  ctx.font = '9px sans-serif'
  ctx.textAlign = 'right'
  for (const tick of yTicks.value) ctx.fillText(String(tick), pad.left - 6, yScale(tick) + 3)

  // Bars
  for (let i = 0; i < results.length; i++) {
    const p = results[i]
    ctx.fillStyle = deColor(p.deltaE)
    ctx.globalAlpha = 0.8
    const by = yScale(p.deltaE)
    ctx.fillRect(barX(i), by, barWidth.value, Math.max(0, pad.top + plotH - by))
  }
  ctx.globalAlpha = 1

  // Axis title
  ctx.save()
  ctx.fillStyle = '#333'
  ctx.font = 'bold 10px sans-serif'
  ctx.textAlign = 'center'
  ctx.translate(10, pad.top + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillText(deAxisLabel.value, 0, 0)
  ctx.restore()

  // X labels for classic
  if (chartType.value === 'classic') {
    ctx.fillStyle = '#666'
    ctx.font = '8px sans-serif'
    for (let i = 0; i < results.length; i++) {
      ctx.save()
      const x = barX(i) + barWidth.value / 2
      const y = pad.top + plotH + 10
      ctx.translate(x, y)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'end'
      ctx.fillText(results[i].name.substring(0, 8), 0, 0)
      ctx.restore()
    }
  }

  // Summary text
  ctx.fillStyle = '#333'
  ctx.font = 'bold 9px sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(`Avg ${deAxisLabel.value}=${avgDeltaE.value.toFixed(2)}  Max=${maxDeltaE.value.toFixed(2)}  <3: ${excellentPct.value.toFixed(0)}%`, pad.left, pad.top - 2)

  canvas.toBlob(blob => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `colorchecker-${chartType.value}-${deMethod.value}.png`
    a.click()
    URL.revokeObjectURL(url)
  })
}
</script>

<style scoped>
.ca-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.ca-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.ca-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.patch-count {
  font-weight: 400;
  color: var(--vp-c-text-3);
  font-size: 0.85em;
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
  align-items: flex-end;
}
.slider-group {
  flex: 1;
  min-width: 200px;
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
.chart-toggle {
  display: flex;
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.toggle-btn {
  padding: 6px 14px;
  font-size: 0.82em;
  font-weight: 500;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}
.toggle-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.toggle-btn:not(.active):hover {
  background: var(--vp-c-bg-soft);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.0em;
  font-family: var(--vp-font-family-mono);
}
.result-value.highlight {
  color: var(--vp-c-brand-1);
}

/* CCM table */
.ccm-section {
  margin-bottom: 20px;
}
.ccm-table {
  border-collapse: collapse;
  font-family: var(--vp-font-family-mono);
  font-size: 0.82em;
  margin: 0 auto;
}
.ccm-table th, .ccm-table td {
  border: 1px solid var(--vp-c-divider);
  padding: 6px 12px;
  text-align: center;
}
.ccm-table th {
  background: var(--vp-c-bg);
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.ccm-table td {
  background: var(--vp-c-bg-soft);
}
.ccm-diag {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

/* Patch grid */
.patch-section {
  margin-bottom: 20px;
}
.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.section-header h5 {
  margin: 0;
}
.view-toggle {
  flex-shrink: 0;
}
.view-toggle .toggle-btn {
  padding: 4px 10px;
  font-size: 0.75em;
}
.patch-grid-classic {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
}
.patch-grid-sg {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
}
@media (max-width: 640px) {
  .patch-grid-classic {
    grid-template-columns: repeat(4, 1fr);
  }
  .patch-grid-sg {
    grid-template-columns: repeat(10, 1fr);
    gap: 2px;
  }
}
.patch-cell {
  text-align: center;
}
.patch-swatch {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  display: flex;
  flex-direction: column;
}
.patch-grid-sg .patch-swatch {
  border-radius: 3px;
}
.patch-ref {
  flex: 1;
}
.patch-sensor {
  flex: 1;
}
.patch-name {
  font-size: 0.6em;
  color: var(--vp-c-text-3);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.patch-de {
  font-size: 0.75em;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.patch-de-sg {
  font-size: 0.55em;
}

/* Heatmap */
.heatmap-cell {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  display: flex;
  align-items: center;
  justify-content: center;
}
.patch-grid-sg .heatmap-cell {
  border-radius: 3px;
}
.heatmap-val {
  font-size: 0.72em;
  font-weight: 700;
  font-family: var(--vp-font-family-mono);
}
.patch-grid-sg .heatmap-val {
  font-size: 0.5em;
}

/* Chart */
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.de-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.ref-label {
  font-size: 8px;
  font-weight: 600;
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

/* Export */
.export-row {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}
.export-btn {
  padding: 6px 16px;
  font-size: 0.82em;
  font-weight: 500;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}
.export-btn:hover {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
</style>
