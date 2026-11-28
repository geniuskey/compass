<template>
  <div :class="['snri-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('SNR vs Illuminance', 'SNR vs 조도') }}</h4>
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
        'SNR versus sensor-plane illuminance (555 nm equivalent). Pixel area and exposure set how many photons each lux delivers. Shading shows which noise term dominates.',
        '센서면 조도(555 nm 등가)에 따른 SNR입니다. 픽셀 면적과 노출 시간이 1 lux당 광자 수를 정하고, 음영은 어떤 노이즈 성분이 지배적인지 보여줍니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('QE (550nm)', 'QE (550nm)') }}: <strong>{{ qe }}%</strong></label>
        <input type="range" min="10" max="95" step="1" v-model.number="qe" class="ctrl-range" aria-label="QE (550nm)" />
      </div>
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
        <input type="range" min="1" max="500" step="1" v-model.number="expTime" class="ctrl-range" :aria-label="t('Exposure Time', '노출 시간')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Pixel Area', '픽셀 면적') }}: <strong>{{ pixelArea.toFixed(2) }} &mu;m&sup2;</strong></label>
        <input type="range" min="0.25" max="25" step="0.25" v-model.number="pixelArea" class="ctrl-range" :aria-label="t('Pixel Area', '픽셀 면적')" />
      </div>
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" aria-label="PRNU" />
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">SNR<sub>max</sub></div>
        <div class="result-value highlight">{{ snrMaxDB.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('SNR = 1 (0 dB)', 'SNR = 1 (0 dB)') }}</div>
        <div class="result-value">{{ fmtLux(luxUnity) }} lux</div>
        <div class="result-sub">{{ fmtNum(luxUnity * photonsPerLux) }} {{ t('photons', '광자') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">SNR10 (20 dB)</div>
        <div class="result-value">{{ fmtLux(lux20dB) }} lux</div>
        <div class="result-sub">{{ fmtNum(lux20dB * photonsPerLux) }} {{ t('photons', '광자') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Saturation', '포화') }}</div>
        <div class="result-value">{{ fmtLux(luxSat) }} lux</div>
        <div class="result-sub">{{ fmtNum(satPhotons) }} {{ t('photons', '광자') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Photons per lux', 'lux당 광자') }}</div>
        <div class="result-value">{{ fmtNum(photonsPerLux) }}</div>
        <div class="result-sub">{{ t('per pixel per frame', '픽셀·프레임당') }}</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="snri-svg"
        @mousemove="onHover" @mouseleave="hover = null">
        <defs>
          <clipPath :id="clipId"><rect :x="pad.left" :y="pad.top" :width="plotW" :height="plotH" /></clipPath>
        </defs>
        <!-- Regime shading along x -->
        <g :clip-path="`url(#${clipId})`">
          <rect v-for="band in regimeBands" :key="band.key" :x="band.x0" :y="pad.top" :width="Math.max(0, band.x1 - band.x0)" :height="plotH"
            :fill="band.color" opacity="0.08" />
        </g>
        <line v-for="tick in xTicks" :key="'xg'+tick"
          :x1="xScale(tick)" :y1="pad.top" :x2="xScale(tick)" :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left + plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text v-for="tick in xTicks" :key="'xl'+tick"
          :x="xScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ fmtLux(tick) }}</text>
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        <!-- Reference lines -->
        <line :x1="pad.left" :y1="yScale(0)" :x2="pad.left + plotW" :y2="yScale(0)"
          stroke="#e67e22" stroke-width="1" stroke-dasharray="4,3" opacity="0.6" />
        <text :x="pad.left + plotW + 4" :y="yScale(0) + 3" class="tick-label" :style="{ fill: '#e67e22' }">0 dB</text>
        <line :x1="pad.left" :y1="yScale(20)" :x2="pad.left + plotW" :y2="yScale(20)"
          stroke="#27ae60" stroke-width="1" stroke-dasharray="4,3" opacity="0.6" />
        <text :x="pad.left + plotW + 4" :y="yScale(20) + 3" class="tick-label" :style="{ fill: '#27ae60' }">20 dB</text>
        <g :clip-path="`url(#${clipId})`">
          <!-- Ideal (shot noise only) -->
          <path :d="idealCurve" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="4,3" />
          <!-- SNR curve -->
          <path :d="snrCurve" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        </g>
        <!-- Saturation -->
        <line :x1="xScale(luxSat)" :y1="pad.top" :x2="xScale(luxSat)" :y2="pad.top + plotH" stroke="#e74c3c" stroke-width="1.2" stroke-dasharray="5,3" />
        <!-- Legend -->
        <g :transform="`translate(${pad.left + 8}, ${pad.top + 6})`">
          <rect x="0" y="0" width="10" height="8" fill="#3498db" opacity="0.35" />
          <text x="14" y="7" class="tick-label">{{ t('Read/dark-limited', '읽기/암 노이즈 제한') }}</text>
          <rect x="0" y="12" width="10" height="8" fill="#27ae60" opacity="0.35" />
          <text x="14" y="19" class="tick-label">{{ t('Shot-limited', '샷 노이즈 제한') }}</text>
          <rect x="0" y="24" width="10" height="8" fill="#e67e22" opacity="0.35" />
          <text x="14" y="31" class="tick-label">{{ t('PRNU-limited', 'PRNU 제한') }}</text>
        </g>
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ t('Sensor-plane illuminance (lux)', '센서면 조도 (lux)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">SNR (dB)</text>
        <!-- Hover -->
        <template v-if="hover">
          <circle :cx="hover.cx" :cy="hover.cy" r="4" fill="var(--vp-c-brand-1)" />
          <rect :x="hover.tx" :y="pad.top + 4" width="150" height="56" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="hover.tx + 6" :y="pad.top + 16" class="tooltip-text">{{ fmtLux(hover.lux) }} lux</text>
          <text :x="hover.tx + 6" :y="pad.top + 28" class="tooltip-text">{{ fmtNum(hover.photons) }} {{ t('photons', '광자') }}</text>
          <text :x="hover.tx + 6" :y="pad.top + 40" class="tooltip-text">SNR = {{ hover.snr.toFixed(1) }} dB</text>
          <text :x="hover.tx + 6" :y="pad.top + 52" class="tooltip-text">{{ fmtNum(hover.sig) }} e&minus; / &sigma;={{ hover.noise.toFixed(1) }}</text>
        </template>
      </svg>
    </div>
    <p class="hint">{{ t('Illuminance is at the sensor plane. For scene illuminance divide by ≈ 4N²/(ρ·T): about 50× at f/2 with an 18 % grey scene.', '조도는 센서면 기준입니다. 장면 조도로 환산하려면 ≈ 4N²/(ρ·T)로 나눕니다 (f/2, 18 % 회색 장면에서 약 50배).') }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, useId } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'
import { PHOTONS_PER_LUX_S_UM2 } from '../composables/sensorNoise'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()
const clipId = `snri-clip-${useId()}`

const qe = ref(70)
const fwc = ref(10000)
const readNoise = ref(3.0)
const darkCurrent = ref(10)
const expTime = ref(33)
const pixelArea = ref(1.0)
const prnu = ref(1.0)

function resetControls() {
  qe.value = 70
  fwc.value = 10000
  readNoise.value = 3.0
  darkCurrent.value = 10
  expTime.value = 33
  pixelArea.value = 1.0
  prnu.value = 1.0
}

/** Photons per pixel per frame for 1 lux at the sensor plane (555 nm equivalent). */
const photonsPerLux = computed(() => PHOTONS_PER_LUX_S_UM2 * pixelArea.value * expTime.value / 1000)
const darkCharge = computed(() => darkCurrent.value * expTime.value / 1000)
const eta = computed(() => qe.value / 100)
const p = computed(() => prnu.value / 100)
const satPhotons = computed(() => fwc.value / eta.value)
const luxSat = computed(() => satPhotons.value / photonsPerLux.value)

function computeSNR(nPhotons: number): { signal: number; noise: number; snr: number; snrDB: number } {
  const signal = Math.min(nPhotons * eta.value, fwc.value)
  const noise = Math.sqrt(signal + darkCharge.value + readNoise.value ** 2 + (p.value * signal) ** 2)
  const snr = signal / noise
  return { signal, noise, snr, snrDB: snr > 0 ? 20 * Math.log10(snr) : -Infinity }
}
function snrAtLux(lux: number) { return computeSNR(lux * photonsPerLux.value) }

const snrMaxDB = computed(() => computeSNR(satPhotons.value).snrDB)

/** Smallest lux reaching the target SNR (bisection on log-lux; SNR is monotonic below saturation). */
function luxForSnr(target: number): number {
  let lo = Math.log10(luxSat.value) - 12, hi = Math.log10(luxSat.value)
  if (snrAtLux(10 ** hi).snr < target) return NaN
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    if (snrAtLux(10 ** mid).snr >= target) hi = mid
    else lo = mid
  }
  return 10 ** hi
}
const luxUnity = computed(() => luxForSnr(1))
const lux20dB = computed(() => luxForSnr(10))

function fmtLux(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v >= 1000) return `${+(v / 1000).toPrecision(3)}k`
  if (v >= 1) return `${+v.toPrecision(3)}`
  return `${+v.toPrecision(2)}`
}
function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (v >= 1e4) return `${(v / 1e3).toFixed(1)}k`
  if (v >= 10) return v.toFixed(0)
  return v.toFixed(2)
}

const W = 560, H = 270
const pad = { top: 16, right: 30, bottom: 36, left: 46 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

// x-range: from 0.3 photon per pixel up to 1.5× saturation
const logMin = computed(() => Math.floor(Math.log10(0.3 / photonsPerLux.value)))
const logMax = computed(() => Math.log10(luxSat.value * 1.5))
const yMin = computed(() => Math.max(-40, Math.floor(snrAtLux(10 ** logMin.value).snrDB / 10) * 10))
const yMaxVal = computed(() => Math.ceil(snrMaxDB.value / 10) * 10 + 5)

const xTicks = computed(() => {
  const ticks: number[] = []
  for (let e = Math.ceil(logMin.value); e <= Math.floor(logMax.value); e++) ticks.push(10 ** e)
  return ticks
})
const yTicks = computed(() => {
  const ticks: number[] = []
  for (let v = yMin.value; v <= yMaxVal.value; v += 10) ticks.push(v)
  return ticks
})

function xScale(lux: number): number {
  return pad.left + ((Math.log10(Math.max(1e-12, lux)) - logMin.value) / (logMax.value - logMin.value)) * plotW
}
function yScale(snr: number): number {
  const v = Number.isFinite(snr) ? Math.max(yMin.value - 5, snr) : yMin.value - 5
  return pad.top + plotH - ((v - yMin.value) / (yMaxVal.value - yMin.value)) * plotH
}

/** Regime boundaries in electrons: shot = read²+dark; PRNU = 1/p². */
const regimeBands = computed(() => {
  const toLux = (e: number) => e / eta.value / photonsPerLux.value
  const xLo = pad.left, xSat = xScale(luxSat.value)
  const xShot = Math.min(xSat, xScale(toLux(readNoise.value ** 2 + darkCharge.value)))
  const xPrnu = p.value > 0 ? Math.min(xSat, xScale(toLux(1 / p.value ** 2))) : xSat
  return [
    { key: 'read', x0: xLo, x1: Math.max(xLo, xShot), color: '#3498db' },
    { key: 'shot', x0: Math.max(xLo, xShot), x1: Math.max(xLo, xPrnu), color: '#27ae60' },
    { key: 'prnu', x0: Math.max(xLo, xPrnu), x1: xSat, color: '#e67e22' },
  ]
})

const STEPS = 240
function buildCurve(fn: (lux: number) => number): string {
  let d = ''
  for (let i = 0; i <= STEPS; i++) {
    const lux = 10 ** (logMin.value + (i / STEPS) * (Math.log10(luxSat.value) - logMin.value))
    d += `${i === 0 ? 'M' : 'L'}${xScale(lux).toFixed(1)},${yScale(fn(lux)).toFixed(1)}`
  }
  return d
}

const snrCurve = computed(() => buildCurve(lux => snrAtLux(lux).snrDB))
const idealCurve = computed(() => buildCurve(lux => 10 * Math.log10(Math.max(1e-12, lux * photonsPerLux.value * eta.value))))

const hover = ref<{ cx: number; cy: number; tx: number; lux: number; photons: number; snr: number; sig: number; noise: number } | null>(null)
function onHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * (W / rect.width)
  const frac = (mx - pad.left) / plotW
  if (frac < 0 || frac > 1) { hover.value = null; return }
  const lux = 10 ** (logMin.value + frac * (logMax.value - logMin.value))
  const nPh = lux * photonsPerLux.value
  const { signal, noise, snrDB } = computeSNR(nPh)
  const cx = xScale(lux), cy = Math.min(pad.top + plotH, yScale(snrDB))
  hover.value = { cx, cy, tx: cx + 160 > W - pad.right ? cx - 160 : cx + 10, lux, photons: nPh, snr: snrDB, sig: signal, noise }
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
.svg-wrapper { margin-top: 4px; }
.snri-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
</style>
