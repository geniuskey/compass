<template>
  <div class="snri-container">
    <h4>{{ t('SNR vs Illuminance', 'SNR vs 조도') }}</h4>
    <p class="component-description">
      {{ t(
        'Analyze SNR across signal levels with noise breakdown. Compare different pixel configurations and identify noise-limited regions.',
        '신호 레벨별 SNR을 노이즈 성분 분해와 함께 분석합니다. 다른 픽셀 구성을 비교하고 노이즈 제한 영역을 식별합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('QE (550nm)', 'QE (550nm)') }}: <strong>{{ qe }}%</strong></label>
        <input type="range" min="10" max="95" step="1" v-model.number="qe" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Dark Current', '암전류') }}: <strong>{{ darkCurrent }} e&minus;/s</strong></label>
        <input type="range" min="0" max="100" step="1" v-model.number="darkCurrent" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Exposure Time', '노출 시간') }}: <strong>{{ expTime }} ms</strong></label>
        <input type="range" min="1" max="500" step="1" v-model.number="expTime" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Pixel Area', '픽셀 면적') }}: <strong>{{ pixelArea.toFixed(2) }} &mu;m&sup2;</strong></label>
        <input type="range" min="0.25" max="25" step="0.25" v-model.number="pixelArea" class="ctrl-range" />
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">SNR<sub>max</sub></div>
        <div class="result-value highlight">{{ snrMaxDB.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Unity SNR', '단위 SNR') }}</div>
        <div class="result-value">{{ unityPhotons.toFixed(0) }} {{ t('photons', '광자') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('20dB Threshold', '20dB 임계') }}</div>
        <div class="result-value">{{ thresh20dB.toFixed(0) }} {{ t('photons', '광자') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Saturation', '포화') }}</div>
        <div class="result-value">{{ satPhotons.toFixed(0) }} {{ t('photons', '광자') }}</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="snri-svg"
        @mousemove="onHover" @mouseleave="hover = null">
        <line v-for="tick in xTicks" :key="'xg'+tick"
          :x1="xScale(tick)" :y1="pad.top" :x2="xScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in xTicks" :key="'xl'+tick"
          :x="xScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ tick >= 1000 ? (tick/1000)+'k' : tick }}</text>
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <!-- Reference lines -->
        <line :x1="pad.left" :y1="yScale(0)" :x2="pad.left + plotW" :y2="yScale(0)"
          stroke="#e67e22" stroke-width="1" stroke-dasharray="4,3" opacity="0.5" />
        <text :x="pad.left + plotW + 4" :y="yScale(0) + 3" class="tick-label" fill="#e67e22">0 dB</text>
        <line :x1="pad.left" :y1="yScale(20)" :x2="pad.left + plotW" :y2="yScale(20)"
          stroke="#27ae60" stroke-width="1" stroke-dasharray="4,3" opacity="0.5" />
        <text :x="pad.left + plotW + 4" :y="yScale(20) + 3" class="tick-label" fill="#27ae60">20 dB</text>
        <!-- Noise breakdown (filled areas) -->
        <path :d="readArea" fill="#3498db" opacity="0.15" />
        <path :d="shotArea" fill="#27ae60" opacity="0.15" />
        <!-- SNR curve -->
        <path :d="snrCurve" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        <!-- Ideal (shot noise only) -->
        <path :d="idealCurve" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Photons / pixel / frame', '광자 / 픽셀 / 프레임') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">SNR (dB)</text>
        <!-- Hover -->
        <template v-if="hover">
          <circle :cx="hover.cx" :cy="hover.cy" r="4" fill="var(--vp-c-brand-1)" />
          <rect :x="hover.tx" :y="pad.top + 4" width="150" height="44" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="hover.tx + 6" :y="pad.top + 16" class="tooltip-text">{{ hover.photons.toFixed(0) }} {{ t('photons', '광자') }}</text>
          <text :x="hover.tx + 6" :y="pad.top + 28" class="tooltip-text">SNR = {{ hover.snr.toFixed(1) }} dB</text>
          <text :x="hover.tx + 6" :y="pad.top + 40" class="tooltip-text">{{ hover.sig.toFixed(0) }} e&minus; / &sigma;={{ hover.noise.toFixed(1) }}</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const qe = ref(70)
const fwc = ref(10000)
const readNoise = ref(3.0)
const darkCurrent = ref(10)
const expTime = ref(33)
const pixelArea = ref(1.0)

const darkCharge = computed(() => darkCurrent.value * expTime.value / 1000)
const satPhotons = computed(() => fwc.value / (qe.value / 100))
const snrMaxDB = computed(() => 20 * Math.log10(Math.sqrt(fwc.value)))

function computeSNR(nPhotons: number): { signal: number; noise: number; snr: number; snrDB: number } {
  const signal = nPhotons * (qe.value / 100)
  const noise = Math.sqrt(signal + darkCharge.value + readNoise.value ** 2)
  const snr = signal / noise
  return { signal, noise, snr, snrDB: snr > 0 ? 20 * Math.log10(snr) : -20 }
}

const unityPhotons = computed(() => {
  for (let n = 1; n < 1e6; n++) {
    if (computeSNR(n).snr >= 1) return n
  }
  return 0
})

const thresh20dB = computed(() => {
  for (let n = 1; n < 1e6; n++) {
    if (computeSNR(n).snrDB >= 20) return n
  }
  return 0
})

const W = 560, H = 260
const pad = { top: 16, right: 30, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const logMax = computed(() => Math.ceil(Math.log10(satPhotons.value * 1.3)))
const yMin = -10, yMaxVal = computed(() => Math.ceil(snrMaxDB.value / 10) * 10 + 5)

const xTicks = computed(() => {
  const ticks: number[] = []
  for (let e = 0; e <= logMax.value; e++) ticks.push(10 ** e)
  return ticks
})
const yTicks = computed(() => {
  const ticks: number[] = []
  for (let v = yMin; v <= yMaxVal.value; v += 10) ticks.push(v)
  return ticks
})

function xScale(n: number): number { return pad.left + (Math.log10(Math.max(1, n)) / logMax.value) * plotW }
function yScale(snr: number): number { return pad.top + plotH - ((snr - yMin) / (yMaxVal.value - yMin)) * plotH }

const STEPS = 200
function buildCurve(): string {
  let d = ''
  for (let i = 0; i <= STEPS; i++) {
    const nPh = 10 ** ((i / STEPS) * logMax.value)
    if (nPh * (qe.value / 100) > fwc.value) break
    const { snrDB } = computeSNR(nPh)
    d += `${i === 0 ? 'M' : 'L'}${xScale(nPh).toFixed(1)},${yScale(snrDB).toFixed(1)}`
  }
  return d
}

const snrCurve = computed(() => buildCurve())

const idealCurve = computed(() => {
  let d = ''
  for (let i = 0; i <= STEPS; i++) {
    const nPh = 10 ** ((i / STEPS) * logMax.value)
    const sig = nPh * (qe.value / 100)
    if (sig > fwc.value) break
    const snr = sig > 0 ? 20 * Math.log10(Math.sqrt(sig)) : -20
    d += `${i === 0 ? 'M' : 'L'}${xScale(nPh).toFixed(1)},${yScale(snr).toFixed(1)}`
  }
  return d
})

const readArea = computed(() => {
  const nfDB = 20 * Math.log10(readNoise.value)
  const y0 = yScale(yMin)
  const y1 = yScale(nfDB)
  return `M${pad.left},${y0}L${pad.left + plotW},${y0}L${pad.left + plotW},${y1}L${pad.left},${y1}Z`
})

const shotArea = computed(() => {
  const lo = yScale(20 * Math.log10(readNoise.value))
  const hi = yScale(snrMaxDB.value)
  return `M${pad.left},${lo}L${pad.left + plotW},${lo}L${pad.left + plotW},${hi}L${pad.left},${hi}Z`
})

const hover = ref<{ cx: number; cy: number; tx: number; photons: number; snr: number; sig: number; noise: number } | null>(null)
function onHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * (W / rect.width)
  const logN = ((mx - pad.left) / plotW) * logMax.value
  if (logN < 0 || logN > logMax.value) { hover.value = null; return }
  const nPh = 10 ** logN
  const { signal, noise, snrDB } = computeSNR(nPh)
  const cx = xScale(nPh), cy = yScale(snrDB)
  hover.value = { cx, cy, tx: cx + 160 > W - pad.right ? cx - 160 : cx + 10, photons: nPh, snr: snrDB, sig: signal, noise }
}
</script>

<style scoped>
.snri-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.snri-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.svg-wrapper { margin-top: 4px; }
.snri-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
</style>
