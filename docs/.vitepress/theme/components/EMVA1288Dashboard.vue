<template>
  <div :class="['emva-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('EMVA 1288 Dashboard', 'EMVA 1288 대시보드') }}</h4>
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
        'Integrated characterization dashboard based on EMVA 1288 standard. Simulates key sensor parameters from pixel design.',
        'EMVA 1288 표준 기반 통합 특성화 대시보드. 픽셀 설계로부터 주요 센서 파라미터를 시뮬레이션합니다.'
      ) }}
    </p>

    <div class="sim-fs-controls">
    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThick.toFixed(1) }} &mu;m</strong></label>
        <input type="range" min="1" max="6" step="0.1" v-model.number="siThick" class="ctrl-range" :aria-label="t('Si Thickness', '실리콘 두께')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} &mu;m</strong></label>
        <input type="range" min="0.5" max="5" step="0.05" v-model.number="pitch" class="ctrl-range" :aria-label="t('Pixel Pitch', '픽셀 피치')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" :aria-label="t('Read Noise', '읽기 노이즈')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Dark Current @25C', '암전류 @25C') }}: <strong>{{ darkCurrent }} e&minus;/s</strong></label>
        <input type="range" min="0" max="100" step="1" v-model.number="darkCurrent" class="ctrl-range" :aria-label="t('Dark Current @25C', '암전류 @25C')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Exposure Time', '노출 시간') }}: <strong>{{ fmtExposure(exposureMs) }}</strong></label>
        <input type="range" min="-1" max="3" step="0.05" v-model.number="exposureLog" class="ctrl-range" :aria-label="t('Exposure Time', '노출 시간')" />
      </div>
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0.1" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" aria-label="PRNU" />
      </div>
      <div class="slider-group">
        <label>{{ t('Bit Depth', '비트 심도') }}: <strong>{{ bitDepth }} bit</strong></label>
        <input type="range" min="8" max="16" step="1" v-model.number="bitDepth" class="ctrl-range" :aria-label="t('Bit Depth', '비트 심도')" />
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
    </div>
    </div>

    <div class="sim-fs-view">
    <div class="param-grid">
      <div v-for="param in emvaParams" :key="param.key" class="param-card">
        <div class="param-label">{{ param.label }}</div>
        <div class="param-value" :class="{ good: param.grade === 'good', mid: param.grade === 'mid', bad: param.grade === 'bad' }">
          {{ param.value }}
        </div>
        <div class="param-unit">{{ param.unit }}</div>
      </div>
    </div>

    <h5>{{ t('SNR Curve', 'SNR 곡선') }}</h5>
    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="emva-svg">
        <defs>
          <clipPath :id="clipId"><rect :x="pad.left" :y="pad.top" :width="plotW" :height="plotH" /></clipPath>
        </defs>
        <line v-for="tick in snrXTicks" :key="'sxg'+tick"
          :x1="sxScale(tick)" :y1="pad.top" :x2="sxScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in snrYTicks" :key="'syg'+tick"
          :x1="pad.left" :y1="syScale(tick)" :x2="pad.left + plotW" :y2="syScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in snrXTicks" :key="'sxl'+tick"
          :x="sxScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ tick >= 1000 ? (tick/1000)+'k' : tick }}</text>
        <text v-for="tick in snrYTicks" :key="'syl'+tick"
          :x="pad.left - 6" :y="syScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <g :clip-path="`url(#${clipId})`">
          <!-- 0 dB (SNR = 1) reference -->
          <line :x1="pad.left" :y1="syScale(0)" :x2="pad.left + plotW" :y2="syScale(0)" stroke="var(--vp-c-text-2)" stroke-width="0.8" />
          <!-- Ideal shot noise limited -->
          <path :d="idealSnrPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
          <!-- Actual SNR -->
          <path :d="snrPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
          <!-- Absolute sensitivity threshold -->
          <line :x1="sxScale(thresholdE)" :y1="pad.top" :x2="sxScale(thresholdE)" :y2="pad.top + plotH" stroke="#8e44ad" stroke-width="1" stroke-dasharray="3,3" />
        </g>
        <text :x="sxScale(thresholdE) + 4" :y="pad.top + plotH - 6" class="tick-label" :style="{ fill: '#8e44ad' }">&mu;<tspan baseline-shift="sub" font-size="7">e,min</tspan></text>
        <!-- SNR_max line -->
        <line :x1="pad.left" :y1="syScale(snrMax)" :x2="pad.left + plotW" :y2="syScale(snrMax)"
          stroke="#e67e22" stroke-width="1" stroke-dasharray="6,3" />
        <text :x="pad.left + 4" :y="syScale(snrMax) - 4" class="tick-label" :style="{ fill: '#e67e22' }">SNRmax = {{ snrMax.toFixed(1) }} dB</text>
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Signal above dark (e-)', '암신호 제외 신호 (e-)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">SNR (dB)</text>
      </svg>
    </div>
    <p v-if="darkSaturates" class="warn-note">{{ t('Dark signal fills the full well at this exposure: the pixel saturates without light.', '이 노출에서는 암신호가 풀 웰을 채워, 빛이 없어도 픽셀이 포화됩니다.') }}</p>
    <p class="hint">{{ t('EMVA 1288 definitions: σ_d² = σ_read² + μ_dark + σ_q² (temporal dark noise in e⁻), μ_e,min = ½ + √(σ_d² + ¼), DR = μ_e,sat / μ_e,min, SNR_max = μ_e,sat / σ(μ_e,sat). ADC full scale is assumed to equal the full well.', 'EMVA 1288 정의: σ_d² = σ_read² + μ_dark + σ_q² (e⁻ 단위 시간 암잡음), μ_e,min = ½ + √(σ_d² + ¼), DR = μ_e,sat / μ_e,min, SNR_max = μ_e,sat / σ(μ_e,sat). ADC 풀스케일은 풀 웰과 같다고 가정합니다.') }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, useId } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'
import { emvaThresholdE, snrDb, totalNoiseE } from '../composables/sensorNoise'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

const clipId = `emva-clip-${useId()}`

const siThick = ref(3.0)
const pitch = ref(1.0)
const readNoise = ref(3.0)
const darkCurrent = ref(10)
const prnu = ref(1.0)
const bitDepth = ref(12)
const exposureLog = ref(1) // log10(ms): 10 ms

function resetControls() {
  siThick.value = 3.0
  pitch.value = 1.0
  readNoise.value = 3.0
  darkCurrent.value = 10
  prnu.value = 1.0
  bitDepth.value = 12
  exposureLog.value = 1
}

const exposureMs = computed(() => 10 ** exposureLog.value)
function fmtExposure(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 10 ? `${ms.toFixed(0)} ms` : `${ms.toFixed(2)} ms`
}

const fwc = computed(() => Math.round(pitch.value ** 2 * 5000))
/** 1 V output swing at full well. */
const convGain = computed(() => 1 / fwc.value) // V/e⁻
/** ADC full scale = full well → system gain K = 2^bits / FWC [DN/e⁻]. */
const systemGain = computed(() => 2 ** bitDepth.value / fwc.value)
const quantNoise = computed(() => Math.sqrt(1 / 12) / systemGain.value) // e⁻
const darkElectrons = computed(() => darkCurrent.value * exposureMs.value / 1000)
const sigmaDark = computed(() => Math.sqrt(readNoise.value ** 2 + darkElectrons.value + quantNoise.value ** 2))
/** Saturation capacity measured above the dark level. */
const satE = computed(() => Math.max(1, fwc.value - darkElectrons.value))
const darkSaturates = computed(() => darkElectrons.value >= fwc.value)
const p = computed(() => prnu.value / 100)

function noiseAt(n: number): number {
  return totalNoiseE(n, { readNoise: sigmaDark.value, prnu: p.value })
}
function computeSNR(n: number): number {
  return snrDb(n, noiseAt(n))
}

const thresholdE = computed(() => emvaThresholdE(sigmaDark.value))
const snrMax = computed(() => computeSNR(satE.value))
const drRatio = computed(() => satE.value / thresholdE.value)
const dr = computed(() => 20 * Math.log10(drRatio.value))
const drStops = computed(() => Math.log2(drRatio.value))
const absThreshold = computed(() => thresholdE.value / peakQE.value.qe)

const peakQE = computed(() => {
  // Search the visible band: interference in the stack shifts the QE peak
  // away from 550nm as silicon thickness changes.
  const stack = defaultBsiStack('green', siThick.value)
  let best = { qe: 0, wl: 0.55 }
  for (let wl = 0.45; wl <= 0.65 + 1e-9; wl += 0.01) {
    const qe = tmmCalc(stack, 'air', 'sio2', wl, 0, 'avg').layerA[SI_LAYER_IDX]
    if (qe > best.qe) best = { qe, wl }
  }
  return best
})

const peakResp = computed(() => peakQE.value.qe * (peakQE.value.wl * 1000) / 1239.84)

function gradeValue(val: number, goodThresh: number, badThresh: number, higherBetter: boolean): 'good' | 'mid' | 'bad' {
  if (higherBetter) return val >= goodThresh ? 'good' : val >= badThresh ? 'mid' : 'bad'
  return val <= goodThresh ? 'good' : val <= badThresh ? 'mid' : 'bad'
}

const emvaParams = computed(() => [
  { key: 'qe', label: t(`Peak QE (${Math.round(peakQE.value.wl * 1000)}nm)`, `피크 QE (${Math.round(peakQE.value.wl * 1000)}nm)`), value: (peakQE.value.qe * 100).toFixed(1), unit: '%', grade: gradeValue(peakQE.value.qe * 100, 70, 40, true) },
  { key: 'resp', label: t('Peak Responsivity', '피크 응답도'), value: peakResp.value.toFixed(3), unit: 'A/W', grade: 'mid' as const },
  { key: 'fwc', label: t('Saturation Capacity', '포화 용량'), value: Math.round(satE.value).toLocaleString(), unit: t('e- (above dark)', 'e- (암신호 제외)'), grade: gradeValue(satE.value, 8000, 3000, true) },
  { key: 'read', label: t('Temporal Dark Noise σd', '시간적 암노이즈 σd'), value: sigmaDark.value.toFixed(2), unit: t('e- rms (read+dark+quant.)', 'e- rms (읽기+암+양자화)'), grade: gradeValue(sigmaDark.value, 2, 10, false) },
  { key: 'snrmax', label: 'SNR max', value: snrMax.value.toFixed(1), unit: 'dB', grade: gradeValue(snrMax.value, 40, 30, true) },
  { key: 'dr', label: t('Dynamic Range', '다이나믹 레인지'), value: dr.value.toFixed(1), unit: `dB (${drStops.value.toFixed(1)} EV)`, grade: gradeValue(dr.value, 60, 45, true) },
  { key: 'dark', label: t('Dark Signal', '암신호'), value: darkElectrons.value < 10 ? darkElectrons.value.toFixed(2) : darkElectrons.value.toFixed(0), unit: t(`e- (${darkCurrent.value} e-/s × ${fmtExposure(exposureMs.value)})`, `e- (${darkCurrent.value} e-/s × ${fmtExposure(exposureMs.value)})`), grade: gradeValue(Math.sqrt(darkElectrons.value), readNoise.value * 0.5, readNoise.value * 1.5, false) },
  { key: 'prnu', label: 'PRNU', value: prnu.value.toFixed(1), unit: '%', grade: gradeValue(prnu.value, 0.5, 2, false) },
  { key: 'absth', label: t('Abs. Sensitivity Threshold', '절대 감도 임계값'), value: absThreshold.value.toFixed(1), unit: t('photons (μp,min)', '광자 (μp,min)'), grade: gradeValue(absThreshold.value, 5, 15, false) },
  { key: 'quant', label: t('Quantization Noise', '양자화 노이즈'), value: quantNoise.value.toFixed(2), unit: 'e-', grade: gradeValue(quantNoise.value, readNoise.value * 0.3, readNoise.value, false) },
  { key: 'gain', label: t('System Gain K', '시스템 이득 K'), value: systemGain.value.toFixed(3), unit: `DN/e- · CG ${(convGain.value * 1e6).toFixed(0)} µV/e-`, grade: 'mid' as const },
  { key: 'bits', label: t('Useful Bits', '유효 비트'), value: Math.min(bitDepth.value, drStops.value).toFixed(1), unit: 'bit', grade: 'mid' as const },
])

const W = 560, H = 240
const pad = { top: 16, right: 16, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const logMax = computed(() => Math.log10(satE.value * 1.5))
const snrYMax = computed(() => Math.ceil(snrMax.value / 10) * 10 + 5)
const snrYMin = computed(() => Math.max(-40, Math.floor(computeSNR(1) / 10) * 10))

const snrXTicks = computed(() => {
  const ticks: number[] = []
  for (let e = 0; e <= Math.floor(logMax.value); e++) ticks.push(10 ** e)
  return ticks
})
const snrYTicks = computed(() => {
  const ticks: number[] = []
  for (let v = snrYMin.value; v <= snrYMax.value; v += 10) ticks.push(v)
  return ticks
})

function sxScale(n: number): number { return pad.left + (Math.log10(Math.max(1, n)) / logMax.value) * plotW }
function syScale(snr: number): number {
  const v = Math.max(snrYMin.value - 5, snr)
  return pad.top + plotH - ((v - snrYMin.value) / (snrYMax.value - snrYMin.value)) * plotH
}

function curvePath(fn: (n: number) => number): string {
  let d = ''
  for (let i = 0; i <= 200; i++) {
    const n = 10 ** ((i / 200) * Math.log10(satE.value))
    d += `${i === 0 ? 'M' : 'L'}${sxScale(n).toFixed(1)},${syScale(fn(n)).toFixed(1)}`
  }
  return d
}

const snrPath = computed(() => curvePath(computeSNR))
const idealSnrPath = computed(() => curvePath(n => 10 * Math.log10(n)))
</script>

<style scoped>
.emva-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.emva-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.emva-container h5 { margin: 16px 0 8px 0; font-size: 0.95em; color: var(--vp-c-text-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.ctrl-range::-moz-range-thumb { width: 18px; height: 18px; border: none; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.reset-btn { align-self: end; justify-self: start; padding: 4px 12px; font-size: 0.8em; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; transition: color 0.15s, border-color 0.15s; }
.reset-btn:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
.param-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 10px; text-align: center; }
.param-label { font-size: 0.75em; color: var(--vp-c-text-2); margin-bottom: 4px; line-height: 1.2; }
.param-value { font-weight: 700; font-size: 1.05em; font-family: var(--vp-font-family-mono); }
.param-value.good { color: #27ae60; }
.param-value.mid { color: var(--vp-c-brand-1); }
.param-value.bad { color: #e74c3c; }
.param-unit { font-size: 0.7em; color: var(--vp-c-text-3); margin-top: 2px; }
.svg-wrapper { margin-top: 4px; }
.emva-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.hint { margin: 8px 0 0; font-size: 0.78em; color: var(--vp-c-text-2); }
.warn-note { margin: 8px 0 0; font-size: 0.82em; color: #e74c3c; font-weight: 600; }
</style>
