<template>
  <div :class="['dr-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Dynamic Range Calculator', '다이나믹 레인지 계산기') }}</h4>
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
        'Calculate sensor dynamic range from FWC, read noise, dark current, and exposure time. Compare single vs HDR modes.',
        '풀 웰 용량, 읽기 노이즈, 암전류, 노출 시간으로 다이나믹 레인지를 계산합니다. 단일 노출과 HDR 모드를 비교합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" :aria-label="t('Full Well Capacity', '풀 웰 용량')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" :aria-label="t('Read Noise', '읽기 노이즈')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Dark Current', '암전류') }}: <strong>{{ darkCurrent }} e&minus;/s</strong></label>
        <input type="range" min="0" max="100" step="1" v-model.number="darkCurrent" class="ctrl-range" :aria-label="t('Dark Current', '암전류')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Exposure Time', '노출 시간') }}: <strong>{{ expTime }} ms</strong></label>
        <input type="range" min="1" max="1000" step="1" v-model.number="expTime" class="ctrl-range" :aria-label="t('Exposure Time', '노출 시간')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Temperature', '온도') }}: <strong>{{ temperature }}&deg;C</strong></label>
        <input type="range" min="-20" max="70" step="1" v-model.number="temperature" class="ctrl-range" :aria-label="t('Temperature', '온도')" />
      </div>
      <div class="slider-group">
        <label>{{ t('HDR Exposure Ratio', 'HDR 노출 비') }}: <strong>1:{{ hdrRatio }}</strong></label>
        <input type="range" min="2" max="32" step="1" v-model.number="hdrRatio" class="ctrl-range" :aria-label="t('HDR Exposure Ratio', 'HDR 노출 비')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Dark-current doubling', '암전류 2배 온도') }}: <strong>{{ doublingT.toFixed(1) }}&deg;C</strong></label>
        <input type="range" min="4" max="12" step="0.5" v-model.number="doublingT" class="ctrl-range" :aria-label="t('Dark-current doubling', '암전류 2배 온도')" />
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Single Exposure DR', '단일 노출 DR') }}</div>
        <div class="result-value highlight">{{ singleDR.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Single (stops)', '단일 (스톱)') }}</div>
        <div class="result-value">{{ singleStops.toFixed(1) }} EV</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('HDR DR', 'HDR DR') }}</div>
        <div class="result-value highlight">{{ hdrDR.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('HDR (stops)', 'HDR (스톱)') }}</div>
        <div class="result-value">{{ hdrStops.toFixed(1) }} EV</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Noise Floor', '노이즈 플로어') }}</div>
        <div class="result-value">{{ noiseFloor.toFixed(2) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dark Charge', '암전하') }}</div>
        <div class="result-value">{{ darkCharge.toFixed(1) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Usable Full Well', '유효 풀 웰') }}</div>
        <div class="result-value" :class="{ bad: darkSaturates }">{{ usableFwc.toFixed(0) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('HDR merge SNR dip', 'HDR 합성 SNR 저하') }}</div>
        <div class="result-value">{{ snrDipBefore.toFixed(1) }} &rarr; {{ snrDipAfter.toFixed(1) }} dB</div>
      </div>
    </div>
    <p v-if="darkSaturates" class="warn-note">{{ t('Dark charge fills the full well: the pixel saturates without light. Shorten the exposure or cool the sensor.', '암전하가 풀 웰을 채워 빛 없이도 픽셀이 포화됩니다. 노출을 줄이거나 센서를 냉각하세요.') }}</p>
    <p v-else-if="darkCharge > 0.1 * fwc" class="warn-note soft">{{ t('Dark charge uses more than 10 % of the full well.', '암전하가 풀 웰의 10 % 이상을 차지합니다.') }}</p>

    <h5>{{ t('DR vs Temperature', 'DR vs 온도') }}</h5>
    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="dr-svg">
        <line v-for="tick in tempTicks" :key="'tg'+tick"
          :x1="txScale(tick)" :y1="pad.top" :x2="txScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in drTicks" :key="'dg'+tick"
          :x1="pad.left" :y1="dyScale(tick)" :x2="pad.left + plotW" :y2="dyScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in tempTicks" :key="'tl'+tick"
          :x="txScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ tick }}&deg;C</text>
        <text v-for="tick in drTicks" :key="'dl'+tick"
          :x="pad.left - 6" :y="dyScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <path :d="singleTempPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <path :d="hdrTempPath" fill="none" stroke="#e67e22" stroke-width="2" stroke-dasharray="6,3" />
        <!-- Current temp marker -->
        <circle :cx="txScale(temperature)" :cy="dyScale(singleDR)" r="5" fill="var(--vp-c-brand-1)" />
        <circle :cx="txScale(temperature)" :cy="dyScale(hdrDR)" r="5" fill="#e67e22" />
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Temperature', '온도') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">DR (dB)</text>
        <!-- Legend -->
        <line :x1="pad.left + plotW - 100" :y1="pad.top + 8" :x2="pad.left + plotW - 85" :y2="pad.top + 8" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <text :x="pad.left + plotW - 82" :y="pad.top + 11" class="tick-label">{{ t('Single', '단일') }}</text>
        <line :x1="pad.left + plotW - 100" :y1="pad.top + 20" :x2="pad.left + plotW - 85" :y2="pad.top + 20" stroke="#e67e22" stroke-width="2" stroke-dasharray="6,3" />
        <text :x="pad.left + plotW - 82" :y="pad.top + 23" class="tick-label">HDR</text>
      </svg>
    </div>

    <h5>{{ t('HDR SNR vs Scene Exposure', 'HDR SNR vs 장면 노출') }}</h5>
    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="dr-svg">
        <line v-for="tick in sTicksX" :key="'sxg'+tick"
          :x1="sxScale(tick)" :y1="pad.top" :x2="sxScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in sTicksY" :key="'syg'+tick"
          :x1="pad.left" :y1="syScale(tick)" :x2="pad.left + plotW" :y2="syScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in sTicksX" :key="'sxl'+tick"
          :x="sxScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ fmtE(tick) }}</text>
        <text v-for="tick in sTicksY" :key="'syl'+tick"
          :x="pad.left - 6" :y="syScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <path :d="hdrSnrPath" fill="none" stroke="#e67e22" stroke-width="2" />
        <path :d="singleSnrPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <line :x1="sxScale(usableFwc)" :y1="pad.top" :x2="sxScale(usableFwc)" :y2="pad.top + plotH" stroke="#e74c3c" stroke-width="1" stroke-dasharray="4,3" />
        <text :x="sxScale(usableFwc) + 4" :y="pad.top + 12" class="tick-label" :style="{ fill: '#e74c3c' }">{{ t('long exp. saturates', '장노출 포화') }}</text>
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Scene exposure (long-exposure e⁻ equivalent)', '장면 노출 (장노출 e⁻ 등가)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">SNR (dB)</text>
        <line :x1="pad.left + 8" :y1="pad.top + 8" :x2="pad.left + 23" :y2="pad.top + 8" stroke="var(--vp-c-brand-1)" stroke-width="2" />
        <text :x="pad.left + 26" :y="pad.top + 11" class="tick-label">{{ t('Single', '단일') }}</text>
        <line :x1="pad.left + 8" :y1="pad.top + 20" :x2="pad.left + 23" :y2="pad.top + 20" stroke="#e67e22" stroke-width="2" />
        <text :x="pad.left + 26" :y="pad.top + 23" class="tick-label">{{ t('HDR (long + short)', 'HDR (장 + 단노출)') }}</text>
      </svg>
    </div>
    <p class="hint">{{ t('Above the long-exposure saturation the merged image switches to the short exposure (1/ratio of the light), so SNR drops by about 10·log10(ratio) dB. A larger ratio extends DR but deepens this dip.', '장노출이 포화되는 지점부터는 단노출(빛의 1/비율)로 전환되므로 SNR이 약 10·log10(비율) dB 떨어집니다. 비율이 크면 DR은 넓어지지만 이 저하가 깊어집니다.') }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

const fwc = ref(10000)
const readNoise = ref(5.0)
const darkCurrent = ref(10)
const expTime = ref(33)
const temperature = ref(25)
const hdrRatio = ref(8)
const doublingT = ref(6)

function resetControls() {
  fwc.value = 10000
  readNoise.value = 5.0
  darkCurrent.value = 10
  expTime.value = 33
  temperature.value = 25
  hdrRatio.value = 8
  doublingT.value = 6
}

/** Dark current referenced to 25 °C, doubling every `doublingT` °C. */
function darkAtTemp(tempC: number): number {
  return darkCurrent.value * Math.pow(2, (tempC - 25) / doublingT.value)
}

function budget(tempC: number) {
  const dc = darkAtTemp(tempC) * expTime.value / 1000
  const nf = Math.sqrt(readNoise.value ** 2 + dc)
  // Dark charge occupies the well, so less is left for photo-signal.
  const usable = Math.max(1, fwc.value - dc)
  return { dc, nf, usable }
}

const darkCharge = computed(() => budget(temperature.value).dc)
const noiseFloor = computed(() => budget(temperature.value).nf)
const usableFwc = computed(() => budget(temperature.value).usable)
const darkSaturates = computed(() => darkCharge.value >= fwc.value)
const singleDR = computed(() => 20 * Math.log10(usableFwc.value / noiseFloor.value))
const singleStops = computed(() => Math.log2(usableFwc.value / noiseFloor.value))
const hdrDR = computed(() => 20 * Math.log10(usableFwc.value * hdrRatio.value / noiseFloor.value))
const hdrStops = computed(() => Math.log2(usableFwc.value * hdrRatio.value / noiseFloor.value))

function drAtTemp(tempC: number, isHdr: boolean): number {
  const { nf, usable } = budget(tempC)
  return 20 * Math.log10((isHdr ? usable * hdrRatio.value : usable) / nf)
}

/** SNR of a single exposure at x electrons (long exposure). */
function snrSingle(x: number): number {
  return 20 * Math.log10(x / Math.sqrt(x + noiseFloor.value ** 2))
}
/** SNR of the HDR merge: long exposure until saturation, then the short exposure (x/ratio electrons). */
function snrHdr(x: number): number {
  if (x <= usableFwc.value) return snrSingle(x)
  const r = hdrRatio.value
  const sShort = x / r
  const darkShort = darkCharge.value / r
  return 20 * Math.log10(sShort / Math.sqrt(sShort + readNoise.value ** 2 + darkShort))
}
const snrDipBefore = computed(() => snrSingle(usableFwc.value))
const snrDipAfter = computed(() => snrHdr(usableFwc.value * 1.0001))

function fmtE(v: number): string {
  if (v >= 1e6) return `${+(v / 1e6).toPrecision(2)}M`
  if (v >= 1e3) return `${+(v / 1e3).toPrecision(2)}k`
  return `${v}`
}

const W = 560, H = 220
const pad = { top: 16, right: 16, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const tempRange = [-20, 70]
const tempTicks = [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70]

function txScale(t: number): number { return pad.left + ((t - tempRange[0]) / (tempRange[1] - tempRange[0])) * plotW }

const drMax = computed(() => Math.ceil(Math.max(hdrDR.value, drAtTemp(-20, true)) / 10) * 10 + 10)
const drMin = computed(() => Math.floor(Math.min(singleDR.value, drAtTemp(70, false)) / 10) * 10 - 10)
const drTicks = computed(() => {
  const ticks: number[] = []
  for (let v = drMin.value; v <= drMax.value; v += 10) ticks.push(v)
  return ticks
})

function dyScale(dr: number): number {
  return pad.top + plotH - ((dr - drMin.value) / (drMax.value - drMin.value)) * plotH
}

function tempPath(isHdr: boolean): string {
  let d = ''
  for (let t = -20; t <= 70; t += 2) {
    const dr = drAtTemp(t, isHdr)
    d += `${t === -20 ? 'M' : 'L'}${txScale(t).toFixed(1)},${dyScale(dr).toFixed(1)}`
  }
  return d
}

const singleTempPath = computed(() => tempPath(false))
const hdrTempPath = computed(() => tempPath(true))

// --- HDR SNR chart ---
const sLogMax = computed(() => Math.log10(usableFwc.value * hdrRatio.value * 1.05))
const sTicksX = computed(() => {
  const ticks: number[] = []
  for (let e = 0; e <= Math.floor(sLogMax.value); e++) ticks.push(10 ** e)
  return ticks
})
const sYMin = computed(() => Math.floor(snrSingle(1) / 10) * 10)
const sYMax = computed(() => Math.ceil(snrHdr(usableFwc.value * hdrRatio.value) / 10) * 10 + 5)
const sTicksY = computed(() => {
  const ticks: number[] = []
  for (let v = sYMin.value; v <= sYMax.value; v += 10) ticks.push(v)
  return ticks
})
function sxScale(x: number): number { return pad.left + (Math.log10(Math.max(1, x)) / sLogMax.value) * plotW }
function syScale(v: number): number { return pad.top + plotH - ((v - sYMin.value) / (sYMax.value - sYMin.value)) * plotH }
function snrPathFor(fn: (x: number) => number, xMax: number): string {
  let d = ''
  const N = 240
  const lmax = Math.log10(xMax)
  for (let i = 0; i <= N; i++) {
    const x = 10 ** ((i / N) * lmax)
    d += `${i === 0 ? 'M' : 'L'}${sxScale(x).toFixed(1)},${syScale(fn(x)).toFixed(1)}`
  }
  return d
}
const singleSnrPath = computed(() => snrPathFor(snrSingle, usableFwc.value))
const hdrSnrPath = computed(() => snrPathFor(snrHdr, usableFwc.value * hdrRatio.value))
</script>

<style scoped>
.dr-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.dr-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.dr-container h5 { margin: 16px 0 8px 0; font-size: 0.95em; color: var(--vp-c-text-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.ctrl-range::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; border: none; }
.reset-btn { align-self: end; justify-self: start; padding: 4px 12px; font-size: 0.8em; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; transition: color 0.15s, border-color 0.15s; }
.reset-btn:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.result-value.bad { color: #e74c3c; }
.warn-note { margin: 0 0 12px; font-size: 0.82em; color: #e74c3c; font-weight: 600; }
.warn-note.soft { color: #e67e22; }
.hint { margin: 8px 0 0; font-size: 0.78em; color: var(--vp-c-text-2); }
.svg-wrapper { margin-top: 4px; }
.dr-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
