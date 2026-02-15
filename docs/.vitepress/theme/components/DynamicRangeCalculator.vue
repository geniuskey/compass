<template>
  <div class="dr-container">
    <h4>{{ t('Dynamic Range Calculator', '다이나믹 레인지 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Calculate sensor dynamic range from FWC, read noise, dark current, and exposure time. Compare single vs HDR modes.',
        '풀 웰 용량, 읽기 노이즈, 암전류, 노출 시간으로 다이나믹 레인지를 계산합니다. 단일 노출과 HDR 모드를 비교합니다.'
      ) }}
    </p>

    <div class="controls-grid">
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
        <input type="range" min="1" max="1000" step="1" v-model.number="expTime" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Temperature', '온도') }}: <strong>{{ temperature }}&deg;C</strong></label>
        <input type="range" min="-20" max="70" step="1" v-model.number="temperature" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('HDR Exposure Ratio', 'HDR 노출 비') }}: <strong>1:{{ hdrRatio }}</strong></label>
        <input type="range" min="2" max="32" step="1" v-model.number="hdrRatio" class="ctrl-range" />
      </div>
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
    </div>

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
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const fwc = ref(10000)
const readNoise = ref(5.0)
const darkCurrent = ref(10)
const expTime = ref(33)
const temperature = ref(25)
const hdrRatio = ref(8)

function darkAtTemp(tempC: number): number {
  return darkCurrent.value * Math.pow(2, (tempC - 25) / 5.5)
}

const darkCharge = computed(() => darkAtTemp(temperature.value) * expTime.value / 1000)
const noiseFloor = computed(() => Math.sqrt(readNoise.value ** 2 + darkCharge.value))
const singleDR = computed(() => 20 * Math.log10(fwc.value / noiseFloor.value))
const singleStops = computed(() => singleDR.value / 6.02)
const hdrDR = computed(() => 20 * Math.log10(fwc.value * hdrRatio.value / noiseFloor.value))
const hdrStops = computed(() => hdrDR.value / 6.02)

function drAtTemp(tempC: number, isHdr: boolean): number {
  const dc = darkAtTemp(tempC) * expTime.value / 1000
  const nf = Math.sqrt(readNoise.value ** 2 + dc)
  return 20 * Math.log10((isHdr ? fwc.value * hdrRatio.value : fwc.value) / nf)
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
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.svg-wrapper { margin-top: 4px; }
.dr-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
