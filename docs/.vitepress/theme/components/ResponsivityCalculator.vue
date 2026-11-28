<template>
  <div :class="['resp-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Spectral Responsivity Calculator', '분광 응답도 계산기') }}</h4>
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
        'Convert QE spectrum to spectral responsivity R(λ) = QE × qλ/(hc). The TMM stack (no CF layer) is multiplied by a Gaussian colour filter; compare with the bare stack and the ideal QE = 1 limit.',
        'QE 스펙트럼을 분광 응답도 R(λ) = QE × qλ/(hc)로 변환합니다. CF 층을 뺀 TMM 스택에 가우시안 컬러 필터를 곱하며, CF 없는 스택 및 QE = 1 이상 한계와 비교합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThick.toFixed(1) }} &mu;m</strong></label>
        <input type="range" min="1" max="6" step="0.1" v-model.number="siThick" class="ctrl-range" :aria-label="t('Si Thickness', '실리콘 두께')" />
      </div>
      <div class="slider-group">
        <label>{{ t('CF Bandwidth', 'CF 대역폭') }}: <strong>{{ cfBw }} nm</strong></label>
        <input type="range" min="50" max="150" step="5" v-model.number="cfBw" class="ctrl-range" :aria-label="t('CF Bandwidth', 'CF 대역폭')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Conversion Gain', '변환 이득') }}: <strong>{{ cg.toFixed(0) }} &mu;V/e&minus;</strong></label>
        <input type="range" min="20" max="200" step="5" v-model.number="cg" class="ctrl-range" :aria-label="t('Conversion Gain', '변환 이득')" />
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
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
      <div class="result-card">
        <div class="result-label">{{ t('Green per lux·s·µm² (555 nm)', 'Green, lux·s·µm²당 (555 nm)') }}</div>
        <div class="result-value">{{ photometricG.toFixed(0) }} e&minus; &middot; {{ (photometricG * cg / 1000).toFixed(1) }} mV</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="resp-svg" @mousemove="onHover" @mouseleave="hoverIdx = null">
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
        <text v-for="wl in xTicks" :key="'xl'+wl"
          :x="xScale(wl)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ wl }}</text>
        <!-- Ideal QE = 1 and bare stack (no CF) -->
        <path :d="idealPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="2,3" opacity="0.7" />
        <path :d="bareSiPath" fill="none" stroke="var(--vp-c-text-2)" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" />
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
        <text :x="pad.left + plotW - 72" :y="pad.top + 11" class="tick-label">{{ t('Red', '빨강') }}</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 20" :x2="pad.left + plotW - 75" :y2="pad.top + 20" stroke="#27ae60" stroke-width="2" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 23" class="tick-label">{{ t('Green', '초록') }}</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 32" :x2="pad.left + plotW - 75" :y2="pad.top + 32" stroke="#3498db" stroke-width="2" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 35" class="tick-label">{{ t('Blue', '파랑') }}</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 44" :x2="pad.left + plotW - 75" :y2="pad.top + 44" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="4,3" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 47" class="tick-label">{{ t('No CF', 'CF 없음') }}</text>
        <line :x1="pad.left + plotW - 90" :y1="pad.top + 56" :x2="pad.left + plotW - 75" :y2="pad.top + 56" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="2,3" />
        <text :x="pad.left + plotW - 72" :y="pad.top + 59" class="tick-label">QE = 1</text>
        <!-- Hover -->
        <template v-if="hoverIdx !== null">
          <line :x1="xScale(WL_NM[hoverIdx])" :y1="pad.top" :x2="xScale(WL_NM[hoverIdx])" :y2="pad.top + plotH"
            stroke="var(--vp-c-text-3)" stroke-width="0.8" />
          <rect :x="hoverBoxX" :y="pad.top + 4" width="112" height="58" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="hoverBoxX + 6" :y="pad.top + 16" class="tooltip-text">&lambda; = {{ WL_NM[hoverIdx] }} nm</text>
          <text :x="hoverBoxX + 6" :y="pad.top + 28" class="tooltip-text" :style="{ fill: '#e74c3c' }">R {{ spectra.r[hoverIdx].toFixed(3) }} A/W</text>
          <text :x="hoverBoxX + 6" :y="pad.top + 40" class="tooltip-text" :style="{ fill: '#27ae60' }">G {{ spectra.g[hoverIdx].toFixed(3) }} A/W</text>
          <text :x="hoverBoxX + 6" :y="pad.top + 52" class="tooltip-text" :style="{ fill: '#3498db' }">B {{ spectra.b[hoverIdx].toFixed(3) }} A/W</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

const siThick = ref(3.0)
const cfBw = ref(100)
const cg = ref(80)

function resetControls() {
  siThick.value = 3.0
  cfBw.value = 100
  cg.value = 80
}

const WL_NM = Array.from({ length: 31 }, (_, i) => 400 + i * 10)
const CF_CENTERS: Record<string, number> = { red: 0.620, green: 0.530, blue: 0.450 }
const LN2 = Math.log(2)

function cfT(wlUm: number, center: number, bw: number): number {
  return Math.exp(-4 * LN2 * ((wlUm - center) / (bw / 1000)) ** 2)
}

/**
 * Stack without the tabulated CF absorber: the CF layer is replaced by a
 * transparent polymer of the same thickness so interference in the stack is
 * kept, and the colour response comes only from the Gaussian filter below.
 * (Applying both would count the colour filter twice.)
 */
function bareStack() {
  const stack = defaultBsiStack('green', siThick.value)
  stack[2] = { material: 'polymer', thickness: stack[2].thickness }
  return stack
}

function qeToResp(qe: number, wlNm: number): number {
  return qe * wlNm / 1239.84
}

// For red, the real CF is a long-pass filter: keep the Gaussian on the blue
// side but hold transmission flat above the centre wavelength.
function channelT(color: 'red' | 'green' | 'blue', wlUm: number): number {
  const c = CF_CENTERS[color]
  if (color === 'red' && wlUm > c) return 1
  return cfT(wlUm, c, cfBw.value)
}

const spectra = computed(() => {
  const stack = bareStack()
  const r: number[] = [], g: number[] = [], b: number[] = [], bare: number[] = [], ideal: number[] = []
  for (const wl of WL_NM) {
    const wlUm = wl / 1000
    const qe = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg').layerA[SI_LAYER_IDX]
    r.push(qeToResp(qe * channelT('red', wlUm), wl))
    g.push(qeToResp(qe * channelT('green', wlUm), wl))
    b.push(qeToResp(qe * channelT('blue', wlUm), wl))
    bare.push(qeToResp(qe, wl))
    ideal.push(qeToResp(1, wl))
  }
  return { r, g, b, bare, ideal }
})

/**
 * Electrons per (lux·s) per µm² at 555 nm for the green channel:
 * 1 lux at 555 nm = 1/683 W/m² → photon flux 4.09e11 ph/s/cm² per lux
 * = 4090 ph/s/µm² per lux. Uses the green QE at 550 nm (nearest sample) on a 1 µm² pixel.
 */
const photometricG = computed(() => {
  const i = WL_NM.indexOf(550)
  const qe = spectra.value.g[i] / qeToResp(1, 550)
  return 4090 * qe
})

const hoverIdx = ref<number | null>(null)
function onHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * (W / rect.width)
  const frac = (mx - pad.left) / plotW
  if (frac < 0 || frac > 1) { hoverIdx.value = null; return }
  hoverIdx.value = Math.round(frac * (WL_NM.length - 1))
}
const hoverBoxX = computed(() => {
  if (hoverIdx.value === null) return 0
  const x = xScale(WL_NM[hoverIdx.value])
  return x + 122 > W - pad.right ? x - 118 : x + 6
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
const bareSiPath = computed(() => makePath(spectra.value.bare))
</script>

<style scoped>
.resp-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.resp-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.ctrl-range::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; border: none; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.svg-wrapper { margin-top: 4px; }
.resp-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
.reset-btn { align-self: end; justify-self: start; padding: 4px 12px; font-size: 0.8em; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; }
.reset-btn:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
</style>
