<template>
  <div class="lin-container">
    <h4>{{ t('Linearity Analyzer', '선형성 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Analyze sensor response linearity. Visualize the input-output transfer curve, deviation from ideal, and maximum non-linearity.',
        '센서 응답 선형성을 분석합니다. 입출력 전달 곡선, 이상 대비 편차, 최대 비선형성을 시각화합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Non-linearity', '비선형성') }}: <strong>{{ nonlin.toFixed(1) }}%</strong></label>
        <input type="range" min="0" max="10" step="0.1" v-model.number="nonlin" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Knee Point', '무릎 지점') }}: <strong>{{ kneePoint }}%</strong></label>
        <input type="range" min="50" max="100" step="1" v-model.number="kneePoint" class="ctrl-range" />
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Max Non-linearity', '최대 비선형성') }}</div>
        <div class="result-value highlight">{{ maxNL.toFixed(2) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Linear Range', '선형 범위') }}</div>
        <div class="result-value">{{ linearRange.toFixed(0) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('RMS Error', 'RMS 오차') }}</div>
        <div class="result-value">{{ rmsError.toFixed(3) }}%</div>
      </div>
    </div>

    <div class="charts-row">
      <!-- Transfer curve -->
      <div class="chart-col">
        <h5>{{ t('Transfer Curve', '전달 곡선') }}</h5>
        <svg :viewBox="`0 0 ${CW} ${CH}`" class="lin-svg">
          <line v-for="tick in aTicks" :key="'ag'+tick"
            :x1="axScale(tick)" :y1="aPad.top" :x2="axScale(tick)" :y2="aPad.top + aPlotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line v-for="tick in aTicks" :key="'ayg'+tick"
            :x1="aPad.left" :y1="ayScale(tick)" :x2="aPad.left + aPlotW" :y2="ayScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line :x1="aPad.left" :y1="aPad.top" :x2="aPad.left" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="aPad.left" :y1="aPad.top + aPlotH" :x2="aPad.left + aPlotW" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <!-- Ideal line -->
          <line :x1="aPad.left" :y1="aPad.top + aPlotH" :x2="aPad.left + aPlotW" :y2="aPad.top"
            stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
          <!-- Actual curve -->
          <path :d="transferPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
          <text v-for="tick in aTicks" :key="'axl'+tick"
            :x="axScale(tick)" :y="aPad.top + aPlotH + 12" text-anchor="middle" class="tick-label">{{ tick }}%</text>
          <text v-for="tick in aTicks" :key="'ayl'+tick"
            :x="aPad.left - 4" :y="ayScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}%</text>
          <text :x="aPad.left + aPlotW / 2" :y="aPad.top + aPlotH + 24" text-anchor="middle" class="axis-title">{{ t('Input (%FWC)', '입력 (%FWC)') }}</text>
        </svg>
      </div>

      <!-- Residual -->
      <div class="chart-col">
        <h5>{{ t('Residual (Deviation)', '잔차 (편차)') }}</h5>
        <svg :viewBox="`0 0 ${CW} ${CH}`" class="lin-svg">
          <line :x1="aPad.left" :y1="resYScale(0)" :x2="aPad.left + aPlotW" :y2="resYScale(0)"
            stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line v-for="tick in resTicks" :key="'rg'+tick"
            :x1="aPad.left" :y1="resYScale(tick)" :x2="aPad.left + aPlotW" :y2="resYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line :x1="aPad.left" :y1="aPad.top" :x2="aPad.left" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="aPad.left" :y1="aPad.top + aPlotH" :x2="aPad.left + aPlotW" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <path :d="residualPath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <text v-for="tick in aTicks" :key="'rxl'+tick"
            :x="axScale(tick)" :y="aPad.top + aPlotH + 12" text-anchor="middle" class="tick-label">{{ tick }}%</text>
          <text v-for="tick in resTicks" :key="'ryl'+tick"
            :x="aPad.left - 4" :y="resYScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}%</text>
          <text :x="aPad.left + aPlotW / 2" :y="aPad.top + aPlotH + 24" text-anchor="middle" class="axis-title">{{ t('Input (%FWC)', '입력 (%FWC)') }}</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const fwc = ref(10000)
const nonlin = ref(2.0)
const kneePoint = ref(80)

function sensorOutput(inputPct: number): number {
  const knee = kneePoint.value / 100
  const nl = nonlin.value / 100
  if (inputPct <= knee) {
    return inputPct * (1 - nl * (inputPct / knee) ** 2)
  }
  const atKnee = knee * (1 - nl)
  const remaining = 1 - knee
  const compressionFactor = (1 - nl * 3)
  const aboveKnee = (inputPct - knee) * Math.max(0.1, compressionFactor)
  return Math.min(1, atKnee + aboveKnee)
}

const STEPS = 100
const transferData = computed(() => {
  const pts: { input: number; output: number; residual: number }[] = []
  for (let i = 0; i <= STEPS; i++) {
    const input = i / STEPS
    const output = sensorOutput(input)
    pts.push({ input, output, residual: (output - input) * 100 })
  }
  return pts
})

const maxNL = computed(() => Math.max(...transferData.value.map(p => Math.abs(p.residual))))
const rmsError = computed(() => {
  const sum = transferData.value.reduce((s, p) => s + p.residual ** 2, 0)
  return Math.sqrt(sum / transferData.value.length)
})
const linearRange = computed(() => {
  let last = 100
  for (let i = transferData.value.length - 1; i >= 0; i--) {
    if (Math.abs(transferData.value[i].residual) < 1) { last = transferData.value[i].input * 100; break }
  }
  return last
})

const CW = 280, CH = 200
const aPad = { top: 12, right: 8, bottom: 30, left: 36 }
const aPlotW = CW - aPad.left - aPad.right
const aPlotH = CH - aPad.top - aPad.bottom
const aTicks = [0, 25, 50, 75, 100]

function axScale(pct: number): number { return aPad.left + (pct / 100) * aPlotW }
function ayScale(pct: number): number { return aPad.top + aPlotH - (pct / 100) * aPlotH }

const transferPath = computed(() =>
  transferData.value.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${axScale(p.input * 100).toFixed(1)},${ayScale(p.output * 100).toFixed(1)}`
  ).join('')
)

const resRange = computed(() => Math.max(1, Math.ceil(maxNL.value + 0.5)))
const resTicks = computed(() => {
  const r = resRange.value
  return [-r, -r / 2, 0, r / 2, r]
})

function resYScale(v: number): number {
  return aPad.top + aPlotH / 2 - (v / resRange.value) * (aPlotH / 2)
}

const residualPath = computed(() =>
  transferData.value.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${axScale(p.input * 100).toFixed(1)},${resYScale(p.residual).toFixed(1)}`
  ).join('')
)
</script>

<style scoped>
.lin-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.lin-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.lin-container h5 { margin: 0 0 4px 0; font-size: 0.9em; color: var(--vp-c-text-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.charts-row { display: flex; gap: 16px; flex-wrap: wrap; }
.chart-col { flex: 1; min-width: 260px; }
.lin-svg { width: 100%; max-width: 280px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
