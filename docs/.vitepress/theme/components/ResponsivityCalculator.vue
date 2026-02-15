<template>
  <div class="resp-container">
    <h4>{{ t('Spectral Responsivity Calculator', '분광 응답도 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Convert QE spectrum to spectral responsivity R(λ) = QE × qλ/(hc). Compare R/G/B channels with ideal Si photodiode.',
        'QE 스펙트럼을 분광 응답도 R(λ) = QE × qλ/(hc)로 변환합니다. R/G/B 채널과 이상적 Si 포토다이오드를 비교합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThick.toFixed(1) }} &mu;m</strong></label>
        <input type="range" min="1" max="6" step="0.1" v-model.number="siThick" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('CF Bandwidth', 'CF 대역폭') }}: <strong>{{ cfBw }} nm</strong></label>
        <input type="range" min="50" max="150" step="5" v-model.number="cfBw" class="ctrl-range" />
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Peak R (Red)', 'Red 최대') }}</div>
        <div class="result-value" style="color:#e74c3c">{{ peakR.toFixed(3) }} A/W</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Peak R (Green)', 'Green 최대') }}</div>
        <div class="result-value" style="color:#27ae60">{{ peakG.toFixed(3) }} A/W</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Peak R (Blue)', 'Blue 최대') }}</div>
        <div class="result-value" style="color:#3498db">{{ peakB.toFixed(3) }} A/W</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="resp-svg">
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
        <text v-for="wl in xTicks" :key="'xl'+wl"
          :x="xScale(wl)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ wl }}</text>
        <!-- Ideal Si -->
        <path :d="idealPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6" />
        <!-- R/G/B -->
        <path :d="pathR" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.8" />
        <path :d="pathG" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.8" />
        <path :d="pathB" fill="none" stroke="#3498db" stroke-width="2" opacity="0.8" />
        <!-- Axis labels -->
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">R (A/W)</text>
        <!-- Legend -->
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 8" :x2="pad.left + plotW - 75" :y2="pad.top + 8" stroke="#e74c3c" stroke-width="2" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 11" class="tick-label">Red</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 20" :x2="pad.left + plotW - 75" :y2="pad.top + 20" stroke="#27ae60" stroke-width="2" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 23" class="tick-label">Green</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 32" :x2="pad.left + plotW - 75" :y2="pad.top + 32" stroke="#3498db" stroke-width="2" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 35" class="tick-label">Blue</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 44" :x2="pad.left + plotW - 75" :y2="pad.top + 44" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="4,3" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 47" class="tick-label">{{ t('Ideal Si', '이상 Si') }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'

const { t } = useLocale()

const siThick = ref(3.0)
const cfBw = ref(100)

const WL_NM = Array.from({ length: 31 }, (_, i) => 400 + i * 10)
const CF_CENTERS: Record<string, number> = { red: 0.620, green: 0.530, blue: 0.450 }
const LN2 = Math.log(2)

function cfT(wlUm: number, center: number, bw: number): number {
  return Math.exp(-4 * LN2 * ((wlUm - center) / (bw / 1000)) ** 2)
}

function qeAt(color: string, wlUm: number): number {
  const stack = defaultBsiStack(color as 'red' | 'green' | 'blue', siThick.value)
  const r = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
  return r.layerA[SI_LAYER_IDX] * cfT(wlUm, CF_CENTERS[color], cfBw.value)
}

function qeToResp(qe: number, wlNm: number): number {
  return qe * wlNm / 1240
}

const spectra = computed(() => {
  const r: number[] = [], g: number[] = [], b: number[] = [], ideal: number[] = []
  for (const wl of WL_NM) {
    const wlUm = wl / 1000
    r.push(qeToResp(qeAt('red', wlUm), wl))
    g.push(qeToResp(qeAt('green', wlUm), wl))
    b.push(qeToResp(qeAt('blue', wlUm), wl))
    const stack = defaultBsiStack('green', siThick.value)
    const res = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
    ideal.push(qeToResp(res.layerA[SI_LAYER_IDX], wl))
  }
  return { r, g, b, ideal }
})

const peakR = computed(() => Math.max(...spectra.value.r))
const peakG = computed(() => Math.max(...spectra.value.g))
const peakB = computed(() => Math.max(...spectra.value.b))

const W = 560, H = 220
const pad = { top: 12, right: 16, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const yMax = computed(() => {
  const all = [...spectra.value.r, ...spectra.value.g, ...spectra.value.b, ...spectra.value.ideal]
  return Math.ceil(Math.max(...all) * 10) / 10 + 0.05
})

const yTicks = computed(() => {
  const ticks: number[] = []
  const step = yMax.value <= 0.3 ? 0.05 : 0.1
  for (let v = 0; v <= yMax.value + 0.001; v += step) ticks.push(Math.round(v * 100) / 100)
  return ticks
})

const xTicks = [400, 450, 500, 550, 600, 650, 700]

function xScale(wl: number): number { return pad.left + ((wl - 400) / 300) * plotW }
function yScale(v: number): number { return pad.top + plotH - (v / yMax.value) * plotH }

function makePath(data: number[]): string {
  return data.map((v, i) => {
    const x = pad.left + (i / (data.length - 1)) * plotW
    const y = yScale(v)
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join('')
}

const pathR = computed(() => makePath(spectra.value.r))
const pathG = computed(() => makePath(spectra.value.g))
const pathB = computed(() => makePath(spectra.value.b))
const idealPath = computed(() => makePath(spectra.value.ideal))
</script>

<style scoped>
.resp-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.resp-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.svg-wrapper { margin-top: 4px; }
.resp-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
