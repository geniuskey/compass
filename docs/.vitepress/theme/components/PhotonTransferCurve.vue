<template>
  <div :class="['ptc-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Photon Transfer Curve (PTC)', '광자 전달 곡선 (PTC)') }}</h4>
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
        'Visualize the relationship between signal and noise to extract read noise, conversion gain, FWC, and PRNU in a single log-log plot.',
        '신호와 노이즈의 관계를 로그-로그 플롯으로 시각화하여 읽기 노이즈, 변환 이득, 풀 웰 용량, PRNU를 통합 분석합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Read Noise', '읽기 노이즈') }}: <strong>{{ readNoise.toFixed(1) }} e&minus;</strong></label>
        <input type="range" min="0.5" max="30" step="0.5" v-model.number="readNoise" class="ctrl-range" :aria-label="t('Read Noise', '읽기 노이즈')" />
      </div>
      <div class="slider-group">
        <label>{{ t('Full Well Capacity', '풀 웰 용량') }}: <strong>{{ fwc.toLocaleString() }} e&minus;</strong></label>
        <input type="range" min="1000" max="100000" step="1000" v-model.number="fwc" class="ctrl-range" :aria-label="t('Full Well Capacity', '풀 웰 용량')" />
      </div>
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0.1" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" aria-label="PRNU" />
      </div>
      <div class="slider-group">
        <label>{{ t('Conversion Gain', '변환 이득') }}: <strong>{{ cg.toFixed(0) }} &mu;V/e&minus;</strong></label>
        <input type="range" min="5" max="200" step="1" v-model.number="cg" class="ctrl-range" :aria-label="t('Conversion Gain', '변환 이득')" />
      </div>
      <div class="slider-group">
        <label>{{ t('ADC', 'ADC') }}: <strong>{{ adcBits }} bit / {{ adcRange.toFixed(1) }} V</strong></label>
        <div class="inline-row">
          <select v-model.number="adcBits" class="ctrl-select" :aria-label="t('ADC bit depth', 'ADC 비트 수')">
            <option :value="10">10 bit</option>
            <option :value="12">12 bit</option>
            <option :value="14">14 bit</option>
          </select>
          <input type="range" min="0.2" max="2" step="0.1" v-model.number="adcRange" class="ctrl-range" :aria-label="t('ADC input range', 'ADC 입력 범위')" />
        </div>
      </div>
      <div class="slider-group">
        <label>{{ t('Plot units', '표시 단위') }}</label>
        <div class="seg">
          <button type="button" :class="{ active: unit === 'e' }" :aria-pressed="unit === 'e'" @click="unit = 'e'">e&minus;</button>
          <button type="button" :class="{ active: unit === 'dn' }" :aria-pressed="unit === 'dn'" @click="unit = 'dn'">DN</button>
        </div>
      </div>
      <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('System Gain K', '시스템 이득 K') }}</div>
        <div class="result-value highlight">{{ K.toFixed(3) }} DN/e&minus;</div>
        <div class="result-sub">{{ (1 / K).toFixed(2) }} e&minus;/DN</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Temporal Dark Noise', '시간 암잡음') }}</div>
        <div class="result-value">{{ sigmaDarkE.toFixed(2) }} e&minus;</div>
        <div class="result-sub">{{ (sigmaDarkE * K).toFixed(2) }} DN</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Saturation', '포화') }}</div>
        <div class="result-value">{{ satE.toFixed(0) }} e&minus;</div>
        <div class="result-sub" :class="{ warn: adcLimited }">{{ adcLimited ? t('ADC-limited (clips before FWC)', 'ADC 제한 (FWC 전에 클리핑)') : t('FWC-limited', 'FWC 제한') }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">SNR<sub>max</sub></div>
        <div class="result-value">{{ snrMaxDb.toFixed(1) }} dB</div>
        <div class="result-sub">{{ t('ideal', '이상') }} &radic;sat: {{ (20 * Math.log10(Math.sqrt(satE))).toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dynamic Range', '동적 범위') }}</div>
        <div class="result-value">{{ drDb.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Read/Shot · Shot/PRNU', 'Read/Shot · Shot/PRNU') }}</div>
        <div class="result-value small">{{ fmtCross(shotReadCross) }} &middot; {{ fmtCross(prnuShotCross) }}</div>
        <div class="result-sub">e&minus;</div>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="ptc-svg"
        @mousemove="onHover" @mouseleave="hover = null">
        <defs>
          <clipPath :id="clipId">
            <rect :x="pad.left" :y="pad.top" :width="plotW" :height="plotH" />
          </clipPath>
        </defs>
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
          :x="xScale(tick)" :y="pad.top + plotH + 14" text-anchor="middle" class="tick-label">{{ fmtTick(tick) }}</text>
        <text v-for="tick in yTicks" :key="'yl'+tick"
          :x="pad.left - 6" :y="yScale(tick) + 3" text-anchor="end" class="tick-label">{{ fmtTick(tick) }}</text>
        <g :clip-path="`url(#${clipId})`">
          <!-- Region backgrounds -->
          <rect :x="pad.left" :y="pad.top" :width="Math.max(0, xScale(u(Math.min(shotReadCross, satE))) - pad.left)" :height="plotH"
            fill="#3498db" opacity="0.06" />
          <rect :x="xScale(u(Math.min(shotReadCross, satE)))" :y="pad.top"
            :width="Math.max(0, xScale(u(Math.min(prnuShotCross, satE))) - xScale(u(Math.min(shotReadCross, satE))))" :height="plotH"
            fill="#27ae60" opacity="0.06" />
          <rect :x="xScale(u(Math.min(prnuShotCross, satE)))" :y="pad.top"
            :width="Math.max(0, xScale(u(satE)) - xScale(u(Math.min(prnuShotCross, satE))))" :height="plotH"
            fill="#e67e22" opacity="0.06" />
          <!-- Individual noise curves (below saturation) -->
          <path :d="readPath" fill="none" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" />
          <path :d="shotPath" fill="none" stroke="#27ae60" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" />
          <path :d="prnuPath" fill="none" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" />
          <!-- Total noise incl. saturation roll-off -->
          <path :d="totalPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        </g>
        <!-- Saturation vertical line -->
        <line :x1="xScale(u(satE))" :y1="pad.top" :x2="xScale(u(satE))" :y2="pad.top + plotH"
          stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="xScale(u(satE)) - 4" :y="pad.top + 12" text-anchor="end" class="tick-label" :style="{ fill: '#e74c3c' }">{{ adcLimited ? 'ADC sat' : 'FWC' }}</text>
        <!-- Legend -->
        <line :x1="pad.left + 8" :y1="pad.top + 8" :x2="pad.left + 22" :y2="pad.top + 8" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
        <text :x="pad.left + 26" :y="pad.top + 11" class="tick-label">{{ t('Total', '전체') }}</text>
        <line :x1="pad.left + 8" :y1="pad.top + 20" :x2="pad.left + 22" :y2="pad.top + 20" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 23" class="tick-label">{{ unit === 'dn' ? t('Read + quant.', '읽기 + 양자화') : t('Read', '읽기') }}</text>
        <line :x1="pad.left + 8" :y1="pad.top + 32" :x2="pad.left + 22" :y2="pad.top + 32" stroke="#27ae60" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 35" class="tick-label">{{ t('Shot (slope ½)', '샷 (기울기 ½)') }}</text>
        <line :x1="pad.left + 8" :y1="pad.top + 44" :x2="pad.left + 22" :y2="pad.top + 44" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="6,3" />
        <text :x="pad.left + 26" :y="pad.top + 47" class="tick-label">{{ t('PRNU (slope 1)', 'PRNU (기울기 1)') }}</text>
        <!-- Axis labels -->
        <text :x="pad.left + plotW / 2" :y="pad.top + plotH + 28" text-anchor="middle" class="axis-title">{{ unit === 'dn' ? t('Mean signal (DN)', '평균 신호 (DN)') : t('Mean signal (e-)', '평균 신호 (e-)') }}</text>
        <text :x="8" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title"
          :transform="`rotate(-90, 8, ${pad.top + plotH / 2})`">&sigma; ({{ unit === 'dn' ? 'DN' : 'e&minus;' }})</text>
        <!-- Hover -->
        <template v-if="hover">
          <circle :cx="hover.cx" :cy="hover.cy" r="4" fill="var(--vp-c-brand-1)" />
          <rect :x="hover.tx" :y="pad.top + 4" width="140" height="44" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="hover.tx + 6" :y="pad.top + 18" class="tooltip-text">&mu; = {{ fmtVal(hover.mean) }} {{ unitLabel }}</text>
          <text :x="hover.tx + 6" :y="pad.top + 30" class="tooltip-text">&sigma; = {{ fmtVal(hover.sigma) }} {{ unitLabel }}</text>
          <text :x="hover.tx + 6" :y="pad.top + 42" class="tooltip-text">SNR = {{ (20 * Math.log10(Math.max(1e-9, hover.mean / Math.max(hover.sigma, 1e-9)))).toFixed(1) }} dB</text>
        </template>
      </svg>
    </div>
    <p class="hint">
      {{ t(
        'Mean-variance method: in the shot-noise region σ²(DN) = K·μ(DN), with K in DN/e⁻ (EMVA 1288). The Janesick convention quotes 1/K in e⁻/DN. Switch to DN to read K from the slope ½ line. Near saturation the output clips, so σ collapses. That roll-off marks the saturation point.',
        '평균-분산법: 샷 노이즈 영역에서 σ²(DN) = K·μ(DN)이며 K의 단위는 DN/e⁻입니다 (EMVA 1288). Janesick 관례는 1/K를 e⁻/DN으로 표기합니다. DN으로 전환하면 기울기 ½ 직선에서 K를 읽을 수 있습니다. 포화 근처에서는 출력이 클리핑되어 σ가 급감하며, 이 지점이 포화점입니다.'
      ) }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, useId } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'
import { clippedGaussianMoments } from '../composables/sensorNoise'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()
const clipId = `ptc-clip-${useId()}`

const DEFAULTS = { readNoise: 5.0, fwc: 10000, prnu: 1.0, cg: 60, adcBits: 12, adcRange: 1.0 }
const readNoise = ref(DEFAULTS.readNoise)
const fwc = ref(DEFAULTS.fwc)
const prnu = ref(DEFAULTS.prnu)
const cg = ref(DEFAULTS.cg)
const adcBits = ref(DEFAULTS.adcBits)
const adcRange = ref(DEFAULTS.adcRange)
const unit = ref<'e' | 'dn'>('e')

function resetControls() {
  readNoise.value = DEFAULTS.readNoise
  fwc.value = DEFAULTS.fwc
  prnu.value = DEFAULTS.prnu
  cg.value = DEFAULTS.cg
  adcBits.value = DEFAULTS.adcBits
  adcRange.value = DEFAULTS.adcRange
  unit.value = 'e'
}

/** System gain K [DN/e⁻] = CG / LSB. */
const lsbUv = computed(() => (adcRange.value * 1e6) / 2 ** adcBits.value)
const K = computed(() => cg.value / lsbUv.value)
const adcMaxE = computed(() => (2 ** adcBits.value - 1) / K.value)
const satE = computed(() => Math.min(fwc.value, adcMaxE.value))
const adcLimited = computed(() => adcMaxE.value < fwc.value)
/** Quantisation noise of 1/12 DN² referred to electrons. */
const sigmaQE = computed(() => Math.sqrt(1 / 12) / K.value)
const sigmaDarkE = computed(() => Math.sqrt(readNoise.value ** 2 + sigmaQE.value ** 2))
const p = computed(() => prnu.value / 100)

/** Unclipped total noise [e⁻] at mean signal n [e⁻]. */
function sigmaE(n: number): number {
  return Math.sqrt(sigmaDarkE.value ** 2 + n + (p.value * n) ** 2)
}

const snrMaxDb = computed(() => 20 * Math.log10(satE.value / sigmaE(satE.value)))
const drDb = computed(() => 20 * Math.log10(satE.value / sigmaDarkE.value))
const shotReadCross = computed(() => sigmaDarkE.value ** 2)
const prnuShotCross = computed(() => 1 / p.value ** 2)

function fmtCross(n: number): string {
  if (n > satE.value) return '> sat'
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : n.toFixed(0)
}

/** Unit conversion: electrons → current plot unit. */
function u(n: number): number { return unit.value === 'dn' ? n * K.value : n }
const unitLabel = computed(() => (unit.value === 'dn' ? 'DN' : 'e⁻'))

/**
 * Output mean and σ (in electrons) for an input mean μ, including clipping at saturation.
 * Output = min(X, sat) with X ~ N(μ, σ²).
 */
function clippedAt(mu: number): { mean: number; sigma: number } {
  return clippedGaussianMoments(mu, sigmaE(mu), satE.value)
}

const W = 560, H = 300
const pad = { top: 16, right: 16, bottom: 36, left: 50 }
const plotW = W - pad.left - pad.right
const plotH = H - pad.top - pad.bottom

const xMinLog = computed(() => Math.floor(Math.log10(Math.max(1e-3, u(1)))))
const xMaxLog = computed(() => Math.log10(u(satE.value) * 1.6))
const yMinLog = computed(() => Math.floor(Math.log10(Math.max(1e-3, u(sigmaDarkE.value) * 0.4))))
const yMaxLog = computed(() => Math.ceil(Math.log10(u(sigmaE(satE.value)) * 1.3)))

function xScale(v: number): number {
  const lv = Math.log10(Math.max(1e-12, v))
  return pad.left + ((lv - xMinLog.value) / (xMaxLog.value - xMinLog.value)) * plotW
}
function yScale(v: number): number {
  // Clamp a little below the floor so collapsing curves exit through the bottom (clipped).
  const lv = Math.max(yMinLog.value - 0.5, Math.log10(Math.max(1e-12, v)))
  return pad.top + plotH - ((lv - yMinLog.value) / (yMaxLog.value - yMinLog.value)) * plotH
}

function decadeTicks(lo: number, hi: number): number[] {
  const ticks: number[] = []
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    for (const m of [1, 3]) {
      const v = m * 10 ** e
      const lv = Math.log10(v)
      if (lv >= lo - 1e-9 && lv <= hi + 1e-9) ticks.push(v)
    }
  }
  return ticks
}
const xTicks = computed(() => decadeTicks(xMinLog.value, xMaxLog.value))
const yTicks = computed(() => decadeTicks(yMinLog.value, yMaxLog.value))

function fmtTick(v: number): string {
  if (v >= 1000) return `${+(v / 1000).toPrecision(3)}k`
  if (v < 1) return v.toPrecision(1)
  return `${+v.toPrecision(3)}`
}
function fmtVal(v: number): string {
  if (v >= 100) return v.toFixed(0)
  if (v >= 1) return v.toFixed(2)
  return v.toPrecision(2)
}

const STEPS = 240
/** Sample input means (electrons) log-spaced from 1 e⁻ to 1.6 × sat. */
const inputMeans = computed(() => {
  const lo = 0, hi = Math.log10(satE.value * 1.6)
  return Array.from({ length: STEPS + 1 }, (_, i) => 10 ** (lo + (i / STEPS) * (hi - lo)))
})

function pathFrom(points: [number, number][]): string {
  return points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join('')
}
function componentPath(fn: (n: number) => number): string {
  const pts: [number, number][] = []
  for (const n of inputMeans.value) {
    if (n > satE.value) break
    pts.push([xScale(u(n)), yScale(u(fn(n)))])
  }
  return pathFrom(pts)
}

const readPath = computed(() => componentPath(() => (unit.value === 'dn' ? sigmaDarkE.value : readNoise.value)))
const shotPath = computed(() => componentPath(n => Math.sqrt(n)))
const prnuPath = computed(() => componentPath(n => p.value * n))
const totalCurve = computed(() => inputMeans.value.map(mu => clippedAt(mu)))
const totalPath = computed(() => pathFrom(totalCurve.value.map(c => [xScale(u(c.mean)), yScale(u(c.sigma))])))

const hover = ref<{ cx: number; cy: number; tx: number; mean: number; sigma: number } | null>(null)
function onHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mx = (e.clientX - rect.left) * (W / rect.width)
  if (mx < pad.left || mx > pad.left + plotW) { hover.value = null; return }
  // Nearest sampled point on the total curve by x position.
  let best = totalCurve.value[0], bestDx = Infinity
  for (const c of totalCurve.value) {
    const dx = Math.abs(xScale(u(c.mean)) - mx)
    if (dx < bestDx) { bestDx = dx; best = c }
  }
  const mean = u(best.mean), sigma = u(best.sigma)
  const cx = xScale(mean), cy = Math.min(pad.top + plotH, yScale(sigma))
  hover.value = { cx, cy, tx: cx + 150 > W - pad.right ? cx - 150 : cx + 10, mean, sigma }
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
.ctrl-range::-moz-range-thumb { width: 18px; height: 18px; border: none; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.reset-btn { align-self: end; justify-self: start; padding: 4px 12px; font-size: 0.8em; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; transition: color 0.15s, border-color 0.15s; }
.reset-btn:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-size: 1.0em; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.result-value.small { font-size: 0.9em; }
.result-sub { font-size: 0.72em; color: var(--vp-c-text-3); margin-top: 2px; }
.result-sub.warn { color: #e67e22; font-weight: 600; }
.inline-row { display: flex; gap: 8px; align-items: center; }
.ctrl-select { padding: 2px 6px; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); color: var(--vp-c-text-1); font-size: 0.85em; }
.seg { display: inline-flex; border: 1px solid var(--vp-c-divider); border-radius: 6px; overflow: hidden; }
.seg button { padding: 3px 12px; font-size: 0.82em; background: var(--vp-c-bg); color: var(--vp-c-text-2); border: none; cursor: pointer; }
.seg button.active { background: var(--vp-c-brand-1); color: var(--vp-c-white, #fff); }
.hint { margin: 8px 0 0; font-size: 0.8em; color: var(--vp-c-text-2); }
.svg-wrapper { margin-top: 4px; }
.ptc-svg { width: 100%; max-width: 560px; display: block; margin: 0 auto; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
</style>
