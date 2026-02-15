<template>
  <div class="ptc-container">
    <h4>{{ t('Photon Transfer Curve (PTC)', '광자 전달 곡선 (PTC)') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize the relationship between signal and noise to extract read noise, conversion gain, FWC, and PRNU in a single log-log plot.',
        '신호와 노이즈의 관계를 로그-로그 플롯으로 시각화하여 읽기 노이즈, 변환 이득, 풀 웰 용량, PRNU를 통합 분석합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0.1" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Conversion Gain', '변환 이득') }}: <strong>{{ cg.toFixed(2) }} &mu;V/e&minus;</strong></label>
        <input type="range" min="0.5" max="10" step="0.1" v-model.number="cg" class="ctrl-range" />
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Read Noise Floor', '읽기 노이즈') }}</div>
        <div class="result-value highlight">{{ readNoise.toFixed(1) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">SNR<sub>max</sub></div>
        <div class="result-value">{{ snrMax.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Shot/Read Crossover', 'Shot/Read 교차') }}</div>
        <div class="result-value">{{ shotReadCross.toFixed(0) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('PRNU/Shot Crossover', 'PRNU/Shot 교차') }}</div>
        <div class="result-value">{{ prnuShotCross.toFixed(0) }} e&minus;</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="ptc-svg"
        @mousemove="onHover" @mouseleave="hover = null">
        <!-- Grid -->
        <line v-for="tick in xTicks" :key="'xg'+tick"
          :x1="xScale(tick)" :y1="pad.top" :x2="xScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <!-- Axes -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <!-- Tick labels -->
        <text v-for="tick in xTicks" :key="'xl'+tick"
          :x="xScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ tick >= 1000 ? (tick/1000)+'k' : tick }}</text>
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <!-- Region backgrounds -->
        <rect :x="pad.left" :y="pad.top" :width="xScale(shotReadCross) - pad.left" :height="plotH"
          fill="#3498db" opacity="0.05" />
        <rect :x="xScale(shotReadCross)" :y="pad.top"
          :width="Math.max(0, xScale(Math.min(prnuShotCross, fwc)) - xScale(shotReadCross))" :height="plotH"
          fill="#27ae60" opacity="0.05" />
        <!-- Individual noise curves -->
        <path :d="readPath" fill="none" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.6" />
        <path :d="shotPath" fill="none" stroke="#27ae60" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.6" />
        <path :d="prnuPath" fill="none" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.6" />
        <!-- Total noise -->
        <path :d="totalPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        <!-- FWC vertical line -->
        <line :x1="xScale(fwc)" :y1="pad.top" :x2="xScale(fwc)" :y2="pad.top + plotH"
          stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="xScale(fwc) + 4" :y="pad.top + 12" class="tick-label" fill="#e74c3c">FWC</text>
        <!-- Legend -->
        <line :x1="pad.left + 8" :y1="pad.top + 8" :x2="pad.left + 22" :y2="pad.top + 8" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        <text :x="pad.left + 26" :y="pad.top + 11" class="tick-label">{{ t('Total', '전체') }}</text>
        <line :x1="pad.left + 8" :y1="pad.top + 20" :x2="pad.left + 22" :y2="pad.top + 20" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 23" class="tick-label">Read</text>
        <line :x1="pad.left + 8" :y1="pad.top + 32" :x2="pad.left + 22" :y2="pad.top + 32" stroke="#27ae60" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 35" class="tick-label">Shot</text>
        <line :x1="pad.left + 8" :y1="pad.top + 44" :x2="pad.left + 22" :y2="pad.top + 44" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 47" class="tick-label">PRNU</text>
        <!-- Axis labels -->
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Signal (e-)', '신호 (e-)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">&sigma; (e&minus;)</text>
        <!-- Hover -->
        <template v-if="hover">
          <circle :cx="hover.cx" :cy="hover.cy" r="4" fill="var(--vp-c-brand-1)" />
          <rect :x="hover.tx" :y="pad.top + 4" width="130" height="32" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="hover.tx + 6" :y="pad.top + 18" class="tooltip-text">N = {{ hover.n.toFixed(0) }} e&minus;</text>
          <text :x="hover.tx + 6" :y="pad.top + 30" class="tooltip-text">&sigma; = {{ hover.sigma.toFixed(2) }} e&minus;</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const readNoise = ref(5.0)
const fwc = ref(10000)
const prnu = ref(1.0)
const cg = ref(3.0)

const snrMax = computed(() => 20 * Math.log10(Math.sqrt(fwc.value)))
const shotReadCross = computed(() => readNoise.value ** 2)
const prnuShotCross = computed(() => 1 / (prnu.value / 100) ** 2)

function totalNoise(n: number): number {
  return Math.sqrt(readNoise.value ** 2 + n + (prnu.value / 100 * n) ** 2)
}

const W = 560, H = 280
const pad = { top: 16, right: 16, bottom: 36, left: 50 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const logMin = 0, logMaxX = computed(() => Math.log10(fwc.value * 1.5))
const logMinY = computed(() => Math.floor(Math.log10(Math.max(0.5, readNoise.value * 0.5))))
const logMaxY = computed(() => Math.ceil(Math.log10(Math.max(10, totalNoise(fwc.value) * 1.3))))

function xScale(n: number): number {
  const logN = Math.log10(Math.max(1, n))
  return pad.left + (logN / logMaxX.value) * plotW
}
function yScale(sigma: number): number {
  const logS = Math.log10(Math.max(0.1, sigma))
  return pad.top + plotH - ((logS - logMinY.value) / (logMaxY.value - logMinY.value)) * plotH
}

const xTicks = computed(() => {
  const ticks: number[] = []
  for (let e = 0; e <= logMaxX.value; e++) { ticks.push(10 ** e); if (10 ** e * 3 < fwc.value * 1.5) ticks.push(10 ** e * 3) }
  return ticks.filter(v => v >= 1 && v <= fwc.value * 1.5)
})
const yTicks = computed(() => {
  const ticks: number[] = []
  for (let e = logMinY.value; e <= logMaxY.value; e++) { ticks.push(10 ** e); if (10 ** e * 3 < 10 ** logMaxY.value) ticks.push(10 ** e * 3) }
  return ticks.filter(v => v > 0)
})

const STEPS = 200
function makePath(fn: (n: number) => number): string {
  let d = ''
  for (let i = 0; i <= STEPS; i++) {
    const logN = (i / STEPS) * logMaxX.value
    const n = 10 ** logN
    if (n > fwc.value * 1.2) break
    const sigma = fn(n)
    const x = xScale(n), y = yScale(sigma)
    d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

const readPath = computed(() => makePath(() => readNoise.value))
const shotPath = computed(() => makePath(n => Math.sqrt(n)))
const prnuPath = computed(() => makePath(n => prnu.value / 100 * n))
const totalPath = computed(() => makePath(n => totalNoise(n)))

const hover = ref<{ cx: number; cy: number; tx: number; n: number; sigma: number } | null>(null)
function onHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * (W / rect.width)
  const logN = ((mx - pad.left) / plotW) * logMaxX.value
  if (logN < 0 || logN > logMaxX.value) { hover.value = null; return }
  const n = 10 ** logN
  const sigma = totalNoise(n)
  const cx = xScale(n), cy = yScale(sigma)
  hover.value = { cx, cy, tx: cx + 140 > W - pad.right ? cx - 140 : cx + 10, n, sigma }
}
</script>

<style scoped>
.ptc-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.ptc-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-size: 1.0em; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.svg-wrapper { margin-top: 4px; }
.ptc-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
</style>
