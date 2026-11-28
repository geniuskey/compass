<template>
  <div :class="['lin-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Linearity Analyzer', '선형성 분석기') }}</h4>
    <button
      type="button"
      class="sim-fs-btn"
      :aria-label="t('Toggle fullscreen', '전체화면 전환')"
      :aria-pressed="isFullscreen"
      :title="t('Toggle fullscreen', '전체화면 전환')"
      @click="toggleFullscreen"
    >{{ isFullscreen ? '\u00d7' : '\u26f6' }}</button>
    <p class="component-description">
      {{ t(
        'Analyze sensor response linearity. Visualize the input-output transfer curve, deviation from ideal, and maximum non-linearity.',
        '센서 응답 선형성을 분석합니다. 입출력 전달 곡선, 이상 대비 편차, 최대 비선형성을 시각화합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" :aria-label="t('Full Well Capacity', '풀 웰 용량')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Non-linearity', '비선형성') }}: <strong>{{ nonlin.toFixed(1) }}%</strong></label>
        <input type="range" min="0" max="10" step="0.1" v-model.number="nonlin" class="ctrl-range" :aria-label="t('Non-linearity', '비선형성')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Knee Point', '무릎 지점') }}: <strong>{{ kneePoint }}%</strong></label>
        <input type="range" min="50" max="100" step="1" v-model.number="kneePoint" class="ctrl-range" :aria-label="t('Knee Point', '무릎 지점')" />
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Linearity Error LE (EMVA)', '선형성 오차 LE (EMVA)') }}</div>
        <div class="result-value highlight">&plusmn;{{ linearityError.toFixed(2) }}%</div>
        <div class="result-sub">{{ t('fit 5–95 % of saturation', '포화의 5–95 % 구간 피팅') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Max deviation from ideal', '이상 직선 대비 최대 편차') }}</div>
        <div class="result-value">{{ maxNL.toFixed(2) }}% FS</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Linear range (<1 % FS)', '선형 범위 (<1 % FS)') }}</div>
        <div class="result-value">{{ linearRange.toFixed(0) }}%</div>
        <div class="result-sub">{{ fmtE(linearRange / 100 * fwc) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Output saturates at', '출력 포화 입력') }}</div>
        <div class="result-value">{{ satInput.toFixed(0) }}%</div>
        <div class="result-sub">{{ fmtE(satInput / 100 * fwc) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Fitted responsivity', '피팅 응답도') }}</div>
        <div class="result-value">{{ fit.slope.toFixed(3) }}</div>
        <div class="result-sub">{{ t('output/input slope', '출력/입력 기울기') }}</div>
      </div>
    </div>

    <div class="charts-row">
      <!-- Transfer curve -->
      <div class="chart-col">
        <h5>{{ t('Transfer Curve', '전달 곡선') }}</h5>
        <svg :viewBox="`0 0 ${CW} ${CH}`" class="lin-svg">
          <rect :x="axScale(fitWindow.x0)" :y="aPad.top" :width="Math.max(0, axScale(fitWindow.x1) - axScale(fitWindow.x0))" :height="aPlotH"
            fill="var(--vp-c-brand-1)" opacity="0.06" />
          <line v-for="tick in xTicks" :key="'ag'+tick"
            :x1="axScale(tick)" :y1="aPad.top" :x2="axScale(tick)" :y2="aPad.top + aPlotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line v-for="tick in yTicks" :key="'ayg'+tick"
            :x1="aPad.left" :y1="ayScale(tick)" :x2="aPad.left + aPlotW" :y2="ayScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line :x1="aPad.left" :y1="aPad.top" :x2="aPad.left" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="aPad.left" :y1="aPad.top + aPlotH" :x2="aPad.left + aPlotW" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <!-- Ideal line -->
          <line :x1="axScale(0)" :y1="ayScale(0)" :x2="axScale(100)" :y2="ayScale(100)"
            stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
          <!-- Least-squares fit over the EMVA window -->
          <line :x1="axScale(0)" :y1="ayScale(fit.intercept * 100)" :x2="axScale(X_MAX)" :y2="ayScale((fit.intercept + fit.slope * X_MAX / 100) * 100)"
            stroke="#8e44ad" stroke-width="1" stroke-dasharray="2,2" />
          <!-- Actual curve -->
          <path :d="transferPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
          <text v-for="tick in xTicks" :key="'axl'+tick"
            :x="axScale(tick)" :y="aPad.top + aPlotH + 12" text-anchor="middle" class="tick-label">{{ tick }}%</text>
          <text v-for="tick in yTicks" :key="'ayl'+tick"
            :x="aPad.left - 4" :y="ayScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}%</text>
          <text :x="aPad.left + aPlotW / 2" :y="aPad.top + aPlotH + 24" text-anchor="middle" class="axis-title">{{ t('Input (%FWC)', '입력 (%FWC)') }}</text>
          <text :x="9" :y="aPad.top + aPlotH / 2" text-anchor="middle" class="axis-title"
            :transform="`rotate(-90, 9, ${aPad.top + aPlotH / 2})`">{{ t('Output (%FS)', '출력 (%FS)') }}</text>
        </svg>
      </div>

      <!-- Relative deviation from the fit -->
      <div class="chart-col">
        <h5>{{ t('Deviation from linear fit', '선형 피팅 대비 편차') }}</h5>
        <svg :viewBox="`0 0 ${CW} ${CH}`" class="lin-svg">
          <rect :x="axScale(fitWindow.x0)" :y="aPad.top" :width="Math.max(0, axScale(fitWindow.x1) - axScale(fitWindow.x0))" :height="aPlotH"
            fill="var(--vp-c-brand-1)" opacity="0.06" />
          <line :x1="aPad.left" :y1="resYScale(0)" :x2="aPad.left + aPlotW" :y2="resYScale(0)"
            stroke="var(--vp-c-text-3)" stroke-width="1" />
          <line v-for="tick in resTicks" :key="'rg'+tick"
            :x1="aPad.left" :y1="resYScale(tick)" :x2="aPad.left + aPlotW" :y2="resYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line :x1="aPad.left" :y1="aPad.top" :x2="aPad.left" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="aPad.left" :y1="aPad.top + aPlotH" :x2="aPad.left + aPlotW" :y2="aPad.top + aPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <path :d="residualPath" fill="none" stroke="#e74c3c" stroke-width="2" />
          <text v-for="tick in xTicks" :key="'rxl'+tick"
            :x="axScale(tick)" :y="aPad.top + aPlotH + 12" text-anchor="middle" class="tick-label">{{ tick }}%</text>
          <text v-for="tick in resTicks" :key="'ryl'+tick"
            :x="aPad.left - 4" :y="resYScale(tick) + 3" text-anchor="end" class="tick-label">{{ fmtTick(tick) }}%</text>
          <text :x="aPad.left + aPlotW / 2" :y="aPad.top + aPlotH + 24" text-anchor="middle" class="axis-title">{{ t('Input (%FWC)', '입력 (%FWC)') }}</text>
          <text :x="9" :y="aPad.top + aPlotH / 2" text-anchor="middle" class="axis-title"
            :transform="`rotate(-90, 9, ${aPad.top + aPlotH / 2})`">&delta; (%)</text>
        </svg>
      </div>
    </div>
    <p class="hint">{{ t('Shaded: EMVA 1288 fit window (output between 5 % and 95 % of saturation). δ = (y − y_fit)/y_fit and LE = (max δ − min δ)/2 inside the window.', '음영: EMVA 1288 피팅 구간 (출력이 포화의 5 %–95 %). δ = (y − y_fit)/y_fit 이며 LE = (max δ − min δ)/2 (구간 내).') }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

const fwc = ref(10000)
const nonlin = ref(2.0)
const kneePoint = ref(80)

function resetControls() {
  fwc.value = 10000
  nonlin.value = 2.0
  kneePoint.value = 80
}

/** Normalised output (fraction of full scale) for normalised input (fraction of FWC); clips at 1. */
function sensorOutput(input: number): number {
  const knee = kneePoint.value / 100
  const nl = nonlin.value / 100
  if (input <= knee) {
    return Math.min(1, input * (1 - nl * (input / knee) ** 2))
  }
  const atKnee = knee * (1 - nl)
  // Slope just below the knee is (1 − 3·nl); keep it continuous above the knee.
  const compressionFactor = 1 - nl * 3
  const aboveKnee = (input - knee) * Math.max(0.1, compressionFactor)
  return Math.min(1, atKnee + aboveKnee)
}

const X_MAX = 120 // % of FWC on the input axis, so output clipping is visible
const STEPS = 240
const transferData = computed(() => {
  const pts: { input: number; output: number }[] = []
  for (let i = 0; i <= STEPS; i++) {
    const input = (i / STEPS) * (X_MAX / 100)
    pts.push({ input, output: sensorOutput(input) })
  }
  return pts
})

const satInput = computed(() => {
  const hit = transferData.value.find(p => p.output >= 0.9995)
  return hit ? hit.input * 100 : X_MAX
})

/** EMVA 1288 window: output between 5 % and 95 % of saturation output. */
const windowPts = computed(() => transferData.value.filter(p => p.output >= 0.05 && p.output <= 0.95 && p.input * 100 < satInput.value))
const fitWindow = computed(() => {
  const w = windowPts.value
  return w.length ? { x0: w[0].input * 100, x1: w[w.length - 1].input * 100 } : { x0: 0, x1: 0 }
})

const fit = computed(() => {
  const w = windowPts.value
  const n = w.length
  if (n < 2) return { slope: 1, intercept: 0 }
  let sx = 0, sy = 0, sxx = 0, sxy = 0
  for (const p of w) { sx += p.input; sy += p.output; sxx += p.input ** 2; sxy += p.input * p.output }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
  return { slope, intercept: (sy - slope * sx) / n }
})

const deviations = computed(() =>
  windowPts.value.map(p => {
    const yFit = fit.value.intercept + fit.value.slope * p.input
    return { input: p.input, delta: yFit > 0 ? ((p.output - yFit) / yFit) * 100 : 0 }
  })
)

const linearityError = computed(() => {
  const d = deviations.value.map(v => v.delta)
  if (!d.length) return 0
  return (Math.max(...d) - Math.min(...d)) / 2
})

/** Deviation from the ideal y = x, in % of full scale, below saturation. */
const maxNL = computed(() =>
  Math.max(0, ...transferData.value.filter(p => p.input <= 1).map(p => Math.abs(p.output - p.input) * 100))
)
const linearRange = computed(() => {
  let last = 0
  for (const p of transferData.value) {
    if (p.input > 1) break
    if (Math.abs(p.output - p.input) * 100 < 1) last = p.input * 100
    else break
  }
  return last
})

function fmtE(v: number): string {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)
}
function fmtTick(v: number): string {
  return Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2)
}

const CW = 280, CH = 200
const aPad = { top: 12, right: 16, bottom: 30, left: 40 }
const aPlotW = CW - aPad.left - aPad.right
const aPlotH = CH - aPad.top - aPad.bottom
const xTicks = [0, 30, 60, 90, 120]
const yTicks = [0, 25, 50, 75, 100]

function axScale(pct: number): number { return aPad.left + (pct / X_MAX) * aPlotW }
function ayScale(pct: number): number { return aPad.top + aPlotH - (pct / 110) * aPlotH }

const transferPath = computed(() =>
  transferData.value.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${axScale(p.input * 100).toFixed(1)},${ayScale(p.output * 100).toFixed(1)}`
  ).join('')
)

const resRange = computed(() => {
  const m = Math.max(0.1, ...deviations.value.map(d => Math.abs(d.delta)))
  const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
  return nice.find(v => v >= m * 1.1) ?? 50
})
const resTicks = computed(() => {
  const r = resRange.value
  return [-r, -r / 2, 0, r / 2, r]
})

function resYScale(v: number): number {
  return aPad.top + aPlotH / 2 - (v / resRange.value) * (aPlotH / 2)
}

const residualPath = computed(() =>
  deviations.value.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${axScale(p.input * 100).toFixed(1)},${resYScale(p.delta).toFixed(1)}`
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
.ctrl-range::-moz-range-thumb { width: 18px; height: 18px; border: none; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.reset-btn { align-self: end; justify-self: start; padding: 4px 12px; font-size: 0.8em; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; transition: color 0.15s, border-color 0.15s; }
.reset-btn:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.result-sub { font-size: 0.72em; color: var(--vp-c-text-3); margin-top: 2px; }
.hint { margin: 8px 0 0; font-size: 0.78em; color: var(--vp-c-text-2); }
.charts-row { display: flex; gap: 16px; flex-wrap: wrap; }
.chart-col { flex: 1; min-width: 260px; }
.lin-svg { width: 100%; max-width: 280px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
