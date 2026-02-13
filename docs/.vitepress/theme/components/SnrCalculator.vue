<template>
  <div class="snr-container">
    <h4>{{ t('SNR & Photon Transfer Curve Calculator', 'SNR 및 광자 전달 곡선 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Calculate signal-to-noise ratio and related metrics for a CMOS pixel given sensor parameters.',
        'CMOS 픽셀의 센서 파라미터를 기반으로 신호대잡음비 및 관련 지표를 계산합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>
          {{ t('Quantum Efficiency', '양자 효율') }}: <strong>{{ qe }}%</strong>
        </label>
        <input type="range" min="10" max="90" step="1" v-model.number="qe" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(1) }} &mu;m</strong>
        </label>
        <input type="range" min="0.5" max="3.0" step="0.1" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Integration Time', '적분 시간') }}: <strong>{{ integrationTime }} ms</strong>
        </label>
        <input type="range" min="1" max="100" step="1" v-model.number="integrationTime" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Scene Illuminance', '조도') }}: <strong>{{ illuminance.toFixed(0) }} lux</strong>
        </label>
        <input type="range" min="0" max="4" step="0.01" v-model.number="logIlluminance" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Dark Current', '암전류') }}: <strong>{{ darkCurrent.toFixed(1) }} e⁻/s</strong>
        </label>
        <input type="range" min="0.1" max="50" step="0.1" v-model.number="darkCurrent" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Read Noise', '읽기 잡음') }}: <strong>{{ readNoise.toFixed(1) }} e⁻ rms</strong>
        </label>
        <input type="range" min="0.5" max="10" step="0.1" v-model.number="readNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Full Well Capacity', '풀웰 용량') }}: <strong>{{ fullWell }} e⁻</strong>
        </label>
        <input type="range" min="1000" max="30000" step="100" v-model.number="fullWell" class="ctrl-range" />
      </div>
    </div>

    <!-- Info cards -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Signal', '신호') }}</div>
        <div class="result-value">{{ signalElectrons.toFixed(0) }} e⁻</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Total Noise', '총 잡음') }}</div>
        <div class="result-value">{{ totalNoise.toFixed(2) }} e⁻</div>
      </div>
      <div class="result-card">
        <div class="result-label">SNR</div>
        <div class="result-value highlight">{{ snrLinear.toFixed(1) }} ({{ snrDb.toFixed(1) }} dB)</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dynamic Range', '다이나믹 레인지') }}</div>
        <div class="result-value">{{ dynamicRangeDb.toFixed(1) }} dB</div>
      </div>
    </div>

    <!-- Saturation warning -->
    <div v-if="isSaturated" class="sat-notice">
      {{ t('Pixel is saturated! Signal capped at full well capacity.', '픽셀이 포화 상태입니다! 신호가 풀웰 용량에서 제한됩니다.') }}
    </div>

    <!-- PTC Chart -->
    <div class="chart-section">
      <h5>{{ t('Photon Transfer Curve (PTC)', '광자 전달 곡선 (PTC)') }}</h5>
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${ptcW} ${ptcH}`"
          class="ptc-svg"
          @mousemove="onPtcMouseMove"
          @mouseleave="ptcHover = null"
        >
          <!-- Grid -->
          <line
            v-for="tick in ptcXTicks" :key="'pxg'+tick"
            :x1="ptcXScale(tick)" :y1="ptcPad.top"
            :x2="ptcXScale(tick)" :y2="ptcPad.top + ptcPlotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            v-for="tick in ptcYTicks" :key="'pyg'+tick"
            :x1="ptcPad.left" :y1="ptcYScale(tick)"
            :x2="ptcPad.left + ptcPlotW" :y2="ptcYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />

          <!-- Axes -->
          <line :x1="ptcPad.left" :y1="ptcPad.top" :x2="ptcPad.left" :y2="ptcPad.top + ptcPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="ptcPad.left" :y1="ptcPad.top + ptcPlotH" :x2="ptcPad.left + ptcPlotW" :y2="ptcPad.top + ptcPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

          <!-- X tick labels -->
          <template v-for="tick in ptcXTicks" :key="'pxl'+tick">
            <text :x="ptcXScale(tick)" :y="ptcPad.top + ptcPlotH + 14" text-anchor="middle" class="tick-label">
              {{ formatPow10(tick) }}
            </text>
          </template>

          <!-- Y tick labels -->
          <template v-for="tick in ptcYTicks" :key="'pyl'+tick">
            <text :x="ptcPad.left - 6" :y="ptcYScale(tick) + 3" text-anchor="end" class="tick-label">
              {{ formatPow10(tick) }}
            </text>
          </template>

          <!-- Axis titles -->
          <text :x="ptcPad.left + ptcPlotW / 2" :y="ptcH - 2" text-anchor="middle" class="axis-title">{{ t('Signal (e⁻)', '신호 (e⁻)') }}</text>
          <text :x="12" :y="ptcPad.top + ptcPlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${ptcPad.top + ptcPlotH / 2})`">{{ t('Noise (e⁻)', '잡음 (e⁻)') }}</text>

          <!-- Read noise floor -->
          <path :d="readNoisePath" fill="none" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="6,3" />
          <!-- Dark noise -->
          <path :d="darkNoisePath" fill="none" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="4,2" />
          <!-- Shot noise -->
          <path :d="shotNoisePath" fill="none" stroke="#3498db" stroke-width="1.5" stroke-dasharray="8,4" />
          <!-- Total noise -->
          <path :d="totalNoisePath" fill="none" stroke="#e74c3c" stroke-width="2.5" />

          <!-- Operating point -->
          <circle
            v-if="signalElectrons > 0"
            :cx="ptcXScale(Math.log10(Math.max(1, signalElectrons)))"
            :cy="ptcYScale(Math.log10(Math.max(0.1, totalNoise)))"
            r="5" fill="#e74c3c" stroke="#fff" stroke-width="1.5"
          />

          <!-- Legend -->
          <g :transform="`translate(${ptcPad.left + ptcPlotW - 140}, ${ptcPad.top + 8})`">
            <line x1="0" y1="6" x2="16" y2="6" stroke="#e74c3c" stroke-width="2.5" />
            <text x="20" y="10" class="legend-label">{{ t('Total', '합계') }}</text>
            <line x1="0" y1="20" x2="16" y2="20" stroke="#3498db" stroke-width="1.5" stroke-dasharray="8,4" />
            <text x="20" y="24" class="legend-label">{{ t('Shot', '샷') }}</text>
            <line x1="0" y1="34" x2="16" y2="34" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="6,3" />
            <text x="20" y="38" class="legend-label">{{ t('Read', '읽기') }}</text>
            <line x1="0" y1="48" x2="16" y2="48" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="4,2" />
            <text x="20" y="52" class="legend-label">{{ t('Dark', '암전류') }}</text>
          </g>

          <!-- PTC Hover tooltip -->
          <template v-if="ptcHover">
            <line :x1="ptcHover.sx" :y1="ptcPad.top" :x2="ptcHover.sx" :y2="ptcPad.top + ptcPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
            <rect :x="ptcHover.tx" :y="ptcPad.top + 4" width="120" height="34" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
            <text :x="ptcHover.tx + 6" :y="ptcPad.top + 18" class="tooltip-text">Signal: {{ ptcHover.sig.toFixed(0) }} e⁻</text>
            <text :x="ptcHover.tx + 6" :y="ptcPad.top + 30" class="tooltip-text">Noise: {{ ptcHover.noise.toFixed(2) }} e⁻</text>
          </template>
        </svg>
      </div>
    </div>

    <!-- SNR vs Illuminance Chart -->
    <div class="chart-section">
      <h5>{{ t('SNR vs Illuminance', 'SNR 대 조도') }}</h5>
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${snrW} ${snrH}`"
          class="snr-svg"
          @mousemove="onSnrMouseMove"
          @mouseleave="snrHover = null"
        >
          <!-- Grid -->
          <line
            v-for="tick in snrXTicks" :key="'sxg'+tick"
            :x1="snrXScale(tick)" :y1="snrPad.top"
            :x2="snrXScale(tick)" :y2="snrPad.top + snrPlotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            v-for="tick in snrYTicks" :key="'syg'+tick"
            :x1="snrPad.left" :y1="snrYScale(tick)"
            :x2="snrPad.left + snrPlotW" :y2="snrYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />

          <!-- Reference lines at 20dB, 30dB, 40dB -->
          <template v-for="refDb in [20, 30, 40]" :key="'ref'+refDb">
            <line
              v-if="refDb >= snrYMin && refDb <= snrYMax"
              :x1="snrPad.left" :y1="snrYScale(refDb)"
              :x2="snrPad.left + snrPlotW" :y2="snrYScale(refDb)"
              stroke="#27ae60" stroke-width="1" stroke-dasharray="8,4" opacity="0.5"
            />
            <text
              v-if="refDb >= snrYMin && refDb <= snrYMax"
              :x="snrPad.left + snrPlotW + 4" :y="snrYScale(refDb) + 3"
              class="ref-label"
            >{{ refDb }} dB</text>
          </template>

          <!-- Axes -->
          <line :x1="snrPad.left" :y1="snrPad.top" :x2="snrPad.left" :y2="snrPad.top + snrPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="snrPad.left" :y1="snrPad.top + snrPlotH" :x2="snrPad.left + snrPlotW" :y2="snrPad.top + snrPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

          <!-- X tick labels -->
          <template v-for="tick in snrXTicks" :key="'sxl'+tick">
            <text :x="snrXScale(tick)" :y="snrPad.top + snrPlotH + 14" text-anchor="middle" class="tick-label">
              {{ formatPow10(tick) }}
            </text>
          </template>

          <!-- Y tick labels -->
          <template v-for="tick in snrYTicks" :key="'syl'+tick">
            <text :x="snrPad.left - 6" :y="snrYScale(tick) + 3" text-anchor="end" class="tick-label">
              {{ tick }}
            </text>
          </template>

          <!-- Axis titles -->
          <text :x="snrPad.left + snrPlotW / 2" :y="snrH - 2" text-anchor="middle" class="axis-title">{{ t('Illuminance (lux)', '조도 (lux)') }}</text>
          <text :x="12" :y="snrPad.top + snrPlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${snrPad.top + snrPlotH / 2})`">{{ t('SNR (dB)', 'SNR (dB)') }}</text>

          <!-- SNR curve -->
          <path :d="snrCurvePath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />

          <!-- Operating point -->
          <circle
            :cx="snrXScale(Math.log10(illuminance))"
            :cy="snrYScale(Math.max(snrYMin, Math.min(snrYMax, snrDb)))"
            r="5" fill="var(--vp-c-brand-1)" stroke="#fff" stroke-width="1.5"
          />

          <!-- SNR Hover tooltip -->
          <template v-if="snrHover">
            <line :x1="snrHover.sx" :y1="snrPad.top" :x2="snrHover.sx" :y2="snrPad.top + snrPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
            <rect :x="snrHover.tx" :y="snrPad.top + 4" width="120" height="34" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
            <text :x="snrHover.tx + 6" :y="snrPad.top + 18" class="tooltip-text">{{ snrHover.lux.toFixed(0) }} lux</text>
            <text :x="snrHover.tx + 6" :y="snrPad.top + 30" class="tooltip-text">SNR: {{ snrHover.db.toFixed(1) }} dB</text>
          </template>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// ---- Controls ----
const qe = ref(60)
const pitch = ref(1.0)
const integrationTime = ref(33)
const logIlluminance = ref(Math.log10(500))
const darkCurrent = ref(5.0)
const readNoise = ref(1.5)
const fullWell = ref(6000)

const illuminance = computed(() => Math.pow(10, logIlluminance.value))

// ---- Physics ----
const PHOTON_FLUX_PER_LUX = 4.09e11 // photons/s/cm²/lux at 555nm equiv

function calcSignalAndNoise(lux: number) {
  const pixelAreaCm2 = (pitch.value * 1e-4) ** 2
  const tInt = integrationTime.value / 1000
  const photonFlux = lux * PHOTON_FLUX_PER_LUX
  const signalPhotons = photonFlux * pixelAreaCm2 * tInt
  const rawSignal = (qe.value / 100) * signalPhotons
  const signal = Math.min(rawSignal, fullWell.value)
  const shotNoise = Math.sqrt(signal)
  const darkNoise = Math.sqrt(darkCurrent.value * tInt)
  const totalNoise = Math.sqrt(shotNoise ** 2 + darkNoise ** 2 + readNoise.value ** 2)
  const snr = signal > 0 ? signal / totalNoise : 0
  const snrDb = snr > 0 ? 20 * Math.log10(snr) : 0
  return { signal, shotNoise, darkNoise, totalNoise, snr, snrDb, saturated: rawSignal > fullWell.value }
}

const metrics = computed(() => calcSignalAndNoise(illuminance.value))
const signalElectrons = computed(() => metrics.value.signal)
const totalNoise = computed(() => metrics.value.totalNoise)
const snrLinear = computed(() => metrics.value.snr)
const snrDb = computed(() => metrics.value.snrDb)
const isSaturated = computed(() => metrics.value.saturated)
const dynamicRangeDb = computed(() => 20 * Math.log10(fullWell.value / readNoise.value))

// ---- PTC Chart ----
const ptcW = 560
const ptcH = 320
const ptcPad = { top: 20, right: 20, bottom: 40, left: 55 }
const ptcPlotW = ptcW - ptcPad.left - ptcPad.right
const ptcPlotH = ptcH - ptcPad.top - ptcPad.bottom

// Log10 range
const ptcXMin = computed(() => 0)  // 1 e-
const ptcXMax = computed(() => Math.ceil(Math.log10(fullWell.value * 1.1)))
const ptcYMin = computed(() => {
  const v = Math.log10(readNoise.value / 2)
  return Math.floor(v * 2) / 2
})
const ptcYMax = computed(() => {
  const v = Math.log10(Math.sqrt(fullWell.value) * 1.2)
  return Math.ceil(v * 2) / 2
})

const ptcXTicks = computed(() => {
  const ticks: number[] = []
  for (let i = Math.ceil(ptcXMin.value); i <= ptcXMax.value; i++) ticks.push(i)
  return ticks
})
const ptcYTicks = computed(() => {
  const ticks: number[] = []
  const step = 0.5
  for (let i = Math.ceil(ptcYMin.value / step) * step; i <= ptcYMax.value; i += step) {
    ticks.push(Math.round(i * 10) / 10)
  }
  return ticks
})

function ptcXScale(logVal: number): number {
  return ptcPad.left + ((logVal - ptcXMin.value) / (ptcXMax.value - ptcXMin.value)) * ptcPlotW
}
function ptcYScale(logVal: number): number {
  return ptcPad.top + ptcPlotH - ((logVal - ptcYMin.value) / (ptcYMax.value - ptcYMin.value)) * ptcPlotH
}

function formatPow10(v: number): string {
  if (Number.isInteger(v)) return `10${superscript(v)}`
  return `${Math.pow(10, v).toPrecision(2)}`
}
function superscript(n: number): string {
  const map: Record<string, string> = { '0': '\u2070', '1': '\u00B9', '2': '\u00B2', '3': '\u00B3', '4': '\u2074', '5': '\u2075' }
  if (n < 0) return '\u207B' + superscript(-n)
  return String(n).split('').map(c => map[c] || c).join('')
}

function buildPtcPath(noiseFn: (sig: number) => number): string {
  const steps = 200
  let d = ''
  const xMin = ptcXMin.value
  const xMax = ptcXMax.value
  for (let i = 0; i <= steps; i++) {
    const logSig = xMin + (i / steps) * (xMax - xMin)
    const sig = Math.pow(10, logSig)
    const noise = noiseFn(sig)
    if (noise <= 0) continue
    const logNoise = Math.log10(noise)
    const clampedY = Math.max(ptcYMin.value, Math.min(ptcYMax.value, logNoise))
    const x = ptcXScale(logSig)
    const y = ptcYScale(clampedY)
    d += d === '' ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

const tIntSec = computed(() => integrationTime.value / 1000)
const darkNoiseVal = computed(() => Math.sqrt(darkCurrent.value * tIntSec.value))

const readNoisePath = computed(() => buildPtcPath(() => readNoise.value))
const darkNoisePath = computed(() => buildPtcPath(() => darkNoiseVal.value))
const shotNoisePath = computed(() => buildPtcPath((sig) => Math.sqrt(sig)))
const totalNoisePath = computed(() => buildPtcPath((sig) => {
  const shot = Math.sqrt(sig)
  return Math.sqrt(shot ** 2 + darkNoiseVal.value ** 2 + readNoise.value ** 2)
}))

// PTC hover
const ptcHover = ref<{ sx: number; tx: number; sig: number; noise: number } | null>(null)

function onPtcMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = ptcW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const logSig = ptcXMin.value + ((mouseX - ptcPad.left) / ptcPlotW) * (ptcXMax.value - ptcXMin.value)
  if (logSig >= ptcXMin.value && logSig <= ptcXMax.value) {
    const sig = Math.pow(10, logSig)
    const shot = Math.sqrt(sig)
    const noise = Math.sqrt(shot ** 2 + darkNoiseVal.value ** 2 + readNoise.value ** 2)
    const sx = ptcXScale(logSig)
    const tx = sx + 130 > ptcW - ptcPad.right ? sx - 130 : sx + 10
    ptcHover.value = { sx, tx, sig, noise }
  } else {
    ptcHover.value = null
  }
}

// ---- SNR vs Illuminance Chart ----
const snrW = 560
const snrH = 250
const snrPad = { top: 20, right: 40, bottom: 40, left: 55 }
const snrPlotW = snrW - snrPad.left - snrPad.right
const snrPlotH = snrH - snrPad.top - snrPad.bottom

const snrXMin = 0  // log10(1) = 0
const snrXMax = 4  // log10(10000) = 4
const snrYMin = 0
const snrYMax = 60

const snrXTicks = [0, 1, 2, 3, 4]
const snrYTicks = [0, 10, 20, 30, 40, 50, 60]

function snrXScale(logLux: number): number {
  return snrPad.left + ((logLux - snrXMin) / (snrXMax - snrXMin)) * snrPlotW
}
function snrYScale(db: number): number {
  return snrPad.top + snrPlotH - ((db - snrYMin) / (snrYMax - snrYMin)) * snrPlotH
}

const snrCurvePath = computed(() => {
  const steps = 200
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const logLux = snrXMin + (i / steps) * (snrXMax - snrXMin)
    const lux = Math.pow(10, logLux)
    const { snrDb: db } = calcSignalAndNoise(lux)
    const clampedDb = Math.max(snrYMin, Math.min(snrYMax, db))
    const x = snrXScale(logLux)
    const y = snrYScale(clampedDb)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
})

// SNR hover
const snrHover = ref<{ sx: number; tx: number; lux: number; db: number } | null>(null)

function onSnrMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = snrW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const logLux = snrXMin + ((mouseX - snrPad.left) / snrPlotW) * (snrXMax - snrXMin)
  if (logLux >= snrXMin && logLux <= snrXMax) {
    const lux = Math.pow(10, logLux)
    const { snrDb: db } = calcSignalAndNoise(lux)
    const sx = snrXScale(logLux)
    const tx = sx + 130 > snrW - snrPad.right ? sx - 130 : sx + 10
    snrHover.value = { sx, tx, lux, db }
  } else {
    snrHover.value = null
  }
}
</script>

<style scoped>
.snr-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.snr-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.snr-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.slider-group {
  min-width: 200px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.0em;
  font-family: var(--vp-font-family-mono);
}
.result-value.highlight {
  color: var(--vp-c-brand-1);
}
.sat-notice {
  background: #fff3cd;
  color: #664d03;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 0.9em;
  margin-bottom: 16px;
}
.dark .sat-notice {
  background: #4a3f1f;
  color: #ffda6a;
}
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.ptc-svg, .snr-svg {
  width: 100%;
  max-width: 560px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.ref-label {
  font-size: 8px;
  fill: #27ae60;
  font-weight: 600;
}
</style>
