<template>
  <div class="emva-container">
    <h4>{{ t('EMVA 1288 Dashboard', 'EMVA 1288 대시보드') }}</h4>
    <p class="component-description">
      {{ t(
        'Integrated characterization dashboard based on EMVA 1288 standard. Simulates key sensor parameters from pixel design.',
        'EMVA 1288 표준 기반 통합 특성화 대시보드. 픽셀 설계로부터 주요 센서 파라미터를 시뮬레이션합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThick.toFixed(1) }} &mu;m</strong></label>
        <input type="range" min="1" max="6" step="0.1" v-model.number="siThick" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} &mu;m</strong></label>
        <input type="range" min="0.5" max="5" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Dark Current @25C', '암전류 @25C') }}: <strong>{{ darkCurrent }} e&minus;/s</strong></label>
        <input type="range" min="0" max="100" step="1" v-model.number="darkCurrent" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0.1" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Bit Depth', '비트 심도') }}: <strong>{{ bitDepth }} bit</strong></label>
        <input type="range" min="8" max="16" step="1" v-model.number="bitDepth" class="ctrl-range" />
      </div>
    </div>

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
        <!-- Ideal shot noise limited -->
        <path :d="idealSnrPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
        <!-- Actual SNR -->
        <path :d="snrPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <!-- SNR_max line -->
        <line :x1="pad.left" :y1="syScale(snrMax)" :x2="pad.left + plotW" :y2="syScale(snrMax)"
          stroke="#e67e22" stroke-width="1" stroke-dasharray="6,3" />
        <text :x="pad.left + plotW - 4" :y="syScale(snrMax) - 4" text-anchor="end" class="tick-label" fill="#e67e22">SNR_max</text>
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Signal (e-)', '신호 (e-)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">SNR (dB)</text>
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
const pitch = ref(1.0)
const readNoise = ref(3.0)
const darkCurrent = ref(10)
const prnu = ref(1.0)
const bitDepth = ref(12)

const fwc = computed(() => Math.round(pitch.value ** 2 * 5000))
const convGain = computed(() => 1000 / fwc.value)
const snrMax = computed(() => 20 * Math.log10(Math.sqrt(fwc.value)))
const dr = computed(() => 20 * Math.log10(fwc.value / readNoise.value))
const drStops = computed(() => dr.value / 6.02)
const absThreshold = computed(() => readNoise.value / peakQE.value)

const peakQE = computed(() => {
  const stack = defaultBsiStack('green', siThick.value)
  const r = tmmCalc(stack, 'air', 'sio2', 0.55, 0, 'avg')
  return r.layerA[SI_LAYER_IDX]
})

const peakResp = computed(() => peakQE.value * 550 / 1240)
const adcSteps = computed(() => 2 ** bitDepth.value)
const quantNoise = computed(() => (fwc.value / adcSteps.value) / Math.sqrt(12))

function gradeValue(val: number, goodThresh: number, badThresh: number, higherBetter: boolean): 'good' | 'mid' | 'bad' {
  if (higherBetter) return val >= goodThresh ? 'good' : val >= badThresh ? 'mid' : 'bad'
  return val <= goodThresh ? 'good' : val <= badThresh ? 'mid' : 'bad'
}

const emvaParams = computed(() => [
  { key: 'qe', label: t('Peak QE (550nm)', '피크 QE (550nm)'), value: (peakQE.value * 100).toFixed(1), unit: '%', grade: gradeValue(peakQE.value * 100, 70, 40, true) },
  { key: 'resp', label: t('Peak Responsivity', '피크 응답도'), value: peakResp.value.toFixed(3), unit: 'A/W', grade: 'mid' as const },
  { key: 'fwc', label: t('Saturation Capacity', '포화 용량'), value: fwc.value.toLocaleString(), unit: 'e-', grade: gradeValue(fwc.value, 8000, 3000, true) },
  { key: 'read', label: t('Temporal Dark Noise', '시간적 암노이즈'), value: readNoise.value.toFixed(1), unit: 'e- rms', grade: gradeValue(readNoise.value, 2, 10, false) },
  { key: 'snrmax', label: 'SNR_max', value: snrMax.value.toFixed(1), unit: 'dB', grade: gradeValue(snrMax.value, 40, 30, true) },
  { key: 'dr', label: t('Dynamic Range', '다이나믹 레인지'), value: dr.value.toFixed(1), unit: `dB (${drStops.value.toFixed(1)} EV)`, grade: gradeValue(dr.value, 60, 45, true) },
  { key: 'dark', label: t('Dark Current @25C', '암전류 @25C'), value: darkCurrent.value.toString(), unit: 'e-/s', grade: gradeValue(darkCurrent.value, 5, 30, false) },
  { key: 'prnu', label: 'PRNU', value: prnu.value.toFixed(1), unit: '%', grade: gradeValue(prnu.value, 0.5, 2, false) },
  { key: 'absth', label: t('Abs. Sensitivity', '절대 감도'), value: absThreshold.value.toFixed(1), unit: t('photons', '광자'), grade: gradeValue(absThreshold.value, 5, 15, false) },
  { key: 'quant', label: t('Quantization Noise', '양자화 노이즈'), value: quantNoise.value.toFixed(2), unit: 'e-', grade: 'mid' as const },
  { key: 'cg', label: t('Conversion Gain', '변환 이득'), value: (convGain.value * 1000).toFixed(1), unit: 'µV/e-', grade: 'mid' as const },
  { key: 'bits', label: t('Useful Bits', '유효 비트'), value: Math.min(bitDepth.value, Math.log2(fwc.value / readNoise.value)).toFixed(1), unit: 'bit', grade: 'mid' as const },
])

const W = 560, H = 220
const pad = { top: 16, right: 16, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const logMax = computed(() => Math.ceil(Math.log10(fwc.value * 1.2)))
const snrYMax = computed(() => Math.ceil(snrMax.value / 10) * 10 + 5)

const snrXTicks = computed(() => {
  const ticks: number[] = []
  for (let e = 0; e <= logMax.value; e++) ticks.push(10 ** e)
  return ticks
})
const snrYTicks = computed(() => {
  const ticks: number[] = []
  for (let v = 0; v <= snrYMax.value; v += 10) ticks.push(v)
  return ticks
})

function sxScale(n: number): number { return pad.left + (Math.log10(Math.max(1, n)) / logMax.value) * plotW }
function syScale(snr: number): number { return pad.top + plotH - (snr / snrYMax.value) * plotH }

function computeSNR(n: number): number {
  const noise = Math.sqrt(readNoise.value ** 2 + n + (prnu.value / 100 * n) ** 2)
  return n > 0 ? 20 * Math.log10(n / noise) : 0
}

const snrPath = computed(() => {
  let d = ''
  for (let i = 0; i <= 200; i++) {
    const n = 10 ** ((i / 200) * logMax.value)
    if (n > fwc.value) break
    const snr = computeSNR(n)
    d += `${i === 0 ? 'M' : 'L'}${sxScale(n).toFixed(1)},${syScale(snr).toFixed(1)}`
  }
  return d
})

const idealSnrPath = computed(() => {
  let d = ''
  for (let i = 0; i <= 200; i++) {
    const n = 10 ** ((i / 200) * logMax.value)
    if (n > fwc.value) break
    const snr = n > 0 ? 20 * Math.log10(Math.sqrt(n)) : 0
    d += `${i === 0 ? 'M' : 'L'}${sxScale(n).toFixed(1)},${syScale(snr).toFixed(1)}`
  }
  return d
})
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
</style>
