<template>
  <div class="dc-container">
    <h4>{{ t('Dark Current & Noise Simulator', '암전류 및 잡음 시뮬레이터') }}</h4>
    <p class="component-description">
      {{ t(
        'Simulate temperature-dependent dark current, noise budget, and dark frame pattern for CMOS image sensor pixels.',
        'CMOS 이미지 센서 픽셀의 온도 의존 암전류, 잡음 분석, 암전류 프레임 패턴을 시뮬레이션합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Temperature', '온도') }}: <strong>{{ temperature }} &deg;C</strong>
        </label>
        <input type="range" min="-20" max="85" step="1" v-model.number="temperature" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} &mu;m</strong>
        </label>
        <input type="range" min="0.5" max="2.0" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Integration Time', '적분 시간') }}: <strong>{{ integrationTime }} ms</strong>
        </label>
        <input type="range" min="1" max="100" step="1" v-model.number="integrationTime" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Read Noise', '읽기 잡음') }}: <strong>{{ readNoise.toFixed(1) }} e&minus; rms</strong>
        </label>
        <input type="range" min="0.5" max="5.0" step="0.1" v-model.number="readNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Scene Illuminance', '조도') }}: <strong>{{ illuminance }} lux</strong>
        </label>
        <input type="range" min="1" max="1000" step="1" v-model.number="illuminance" class="ctrl-range" />
      </div>
    </div>

    <!-- Info cards -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Dark Current', '암전류') }}</div>
        <div class="result-value">{{ darkCurrentAtT.toFixed(3) }} e&minus;/s</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dark Signal', '암전류 신호') }}</div>
        <div class="result-value">{{ darkSignal.toFixed(3) }} e&minus;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dark Noise', '암전류 잡음') }}</div>
        <div class="result-value">{{ darkNoise.toFixed(3) }} e&minus; rms</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Total Noise', '총 잡음') }}</div>
        <div class="result-value">{{ totalNoise.toFixed(3) }} e&minus; rms</div>
      </div>
      <div class="result-card">
        <div class="result-label">SNR</div>
        <div class="result-value highlight">{{ snrDb.toFixed(1) }} dB</div>
      </div>
    </div>

    <!-- Charts side by side -->
    <div class="charts-row">
      <!-- Left: Dark current vs Temperature -->
      <div class="chart-section chart-half">
        <h5>{{ t('Dark Current vs Temperature', '암전류 대 온도') }}</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${dcW} ${dcH}`"
            class="dc-svg"
            @mousemove="onDcMouseMove"
            @mouseleave="dcHover = null"
          >
            <!-- Grid -->
            <line
              v-for="tick in dcXTicks" :key="'dcxg'+tick"
              :x1="dcXScale(tick)" :y1="dcPad.top"
              :x2="dcXScale(tick)" :y2="dcPad.top + dcPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
            />
            <!-- Reference horizontal lines at 1, 10, 100, 1000 -->
            <template v-for="refVal in [0.01, 0.1, 1, 10, 100, 1000, 10000]" :key="'dcref'+refVal">
              <line
                v-if="Math.log10(refVal) >= dcYMin && Math.log10(refVal) <= dcYMax"
                :x1="dcPad.left" :y1="dcYScale(Math.log10(refVal))"
                :x2="dcPad.left + dcPlotW" :y2="dcYScale(Math.log10(refVal))"
                stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
              />
            </template>

            <!-- Axes -->
            <line :x1="dcPad.left" :y1="dcPad.top" :x2="dcPad.left" :y2="dcPad.top + dcPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="dcPad.left" :y1="dcPad.top + dcPlotH" :x2="dcPad.left + dcPlotW" :y2="dcPad.top + dcPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- X tick labels -->
            <text
              v-for="tick in dcXTicks" :key="'dcxl'+tick"
              :x="dcXScale(tick)" :y="dcPad.top + dcPlotH + 14"
              text-anchor="middle" class="tick-label"
            >{{ tick }}&deg;</text>

            <!-- Y tick labels -->
            <template v-for="refVal in [0.01, 0.1, 1, 10, 100, 1000, 10000]" :key="'dcyl'+refVal">
              <text
                v-if="Math.log10(refVal) >= dcYMin && Math.log10(refVal) <= dcYMax"
                :x="dcPad.left - 6" :y="dcYScale(Math.log10(refVal)) + 3"
                text-anchor="end" class="tick-label"
              >{{ refVal >= 1 ? refVal : refVal.toString() }}</text>
            </template>

            <!-- Axis titles -->
            <text :x="dcPad.left + dcPlotW / 2" :y="dcH - 2" text-anchor="middle" class="axis-title">{{ t('Temperature (C)', '온도 (C)') }}</text>
            <text :x="10" :y="dcPad.top + dcPlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${dcPad.top + dcPlotH / 2})`">{{ t('Dark current (e-/s)', '암전류 (e-/s)') }}</text>

            <!-- Curve -->
            <path :d="dcCurvePath" fill="none" stroke="#e74c3c" stroke-width="2.5" />

            <!-- Vertical marker at current temperature -->
            <line
              :x1="dcXScale(temperature)" :y1="dcPad.top"
              :x2="dcXScale(temperature)" :y2="dcPad.top + dcPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="6,3"
            />
            <circle
              :cx="dcXScale(temperature)"
              :cy="dcYScale(clampLog(darkCurrentAtT))"
              r="5" fill="#e74c3c" stroke="#fff" stroke-width="1.5"
            />

            <!-- Hover tooltip -->
            <template v-if="dcHover">
              <line :x1="dcHover.sx" :y1="dcPad.top" :x2="dcHover.sx" :y2="dcPad.top + dcPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
              <rect :x="dcHover.tx" :y="dcPad.top + 4" width="120" height="34" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
              <text :x="dcHover.tx + 6" :y="dcPad.top + 18" class="tooltip-text">T = {{ dcHover.temp.toFixed(0) }}&deg;C</text>
              <text :x="dcHover.tx + 6" :y="dcPad.top + 30" class="tooltip-text">{{ dcHover.dc.toFixed(3) }} e&minus;/s</text>
            </template>
          </svg>
        </div>
      </div>

      <!-- Right: Noise budget vs Temperature -->
      <div class="chart-section chart-half">
        <h5>{{ t('Noise Budget vs Temperature', '잡음 분석 대 온도') }}</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${nbW} ${nbH}`"
            class="nb-svg"
            @mousemove="onNbMouseMove"
            @mouseleave="nbHover = null"
          >
            <!-- Grid -->
            <line
              v-for="tick in nbXTicks" :key="'nbxg'+tick"
              :x1="nbXScale(tick)" :y1="nbPad.top"
              :x2="nbXScale(tick)" :y2="nbPad.top + nbPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
            />
            <template v-for="refVal in [0.1, 1, 10, 100]" :key="'nbref'+refVal">
              <line
                v-if="Math.log10(refVal) >= nbYMin && Math.log10(refVal) <= nbYMax"
                :x1="nbPad.left" :y1="nbYScale(Math.log10(refVal))"
                :x2="nbPad.left + nbPlotW" :y2="nbYScale(Math.log10(refVal))"
                stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
              />
            </template>

            <!-- Axes -->
            <line :x1="nbPad.left" :y1="nbPad.top" :x2="nbPad.left" :y2="nbPad.top + nbPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="nbPad.left" :y1="nbPad.top + nbPlotH" :x2="nbPad.left + nbPlotW" :y2="nbPad.top + nbPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- X tick labels -->
            <text
              v-for="tick in nbXTicks" :key="'nbxl'+tick"
              :x="nbXScale(tick)" :y="nbPad.top + nbPlotH + 14"
              text-anchor="middle" class="tick-label"
            >{{ tick }}&deg;</text>

            <!-- Y tick labels -->
            <template v-for="refVal in [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]" :key="'nbyl'+refVal">
              <text
                v-if="Math.log10(refVal) >= nbYMin && Math.log10(refVal) <= nbYMax"
                :x="nbPad.left - 6" :y="nbYScale(Math.log10(refVal)) + 3"
                text-anchor="end" class="tick-label"
              >{{ refVal }}</text>
            </template>

            <!-- Axis titles -->
            <text :x="nbPad.left + nbPlotW / 2" :y="nbH - 2" text-anchor="middle" class="axis-title">{{ t('Temperature (C)', '온도 (C)') }}</text>
            <text :x="10" :y="nbPad.top + nbPlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${nbPad.top + nbPlotH / 2})`">{{ t('Noise (e- rms)', '잡음 (e- rms)') }}</text>

            <!-- Read noise (horizontal dashed line) -->
            <path :d="readNoisePath" fill="none" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" />
            <!-- Dark noise curve -->
            <path :d="darkNoisePath" fill="none" stroke="#e74c3c" stroke-width="1.5" />
            <!-- Total noise curve -->
            <path :d="totalNoisePath" fill="none" stroke="var(--vp-c-text-1)" stroke-width="2.5" />

            <!-- Vertical marker -->
            <line
              :x1="nbXScale(temperature)" :y1="nbPad.top"
              :x2="nbXScale(temperature)" :y2="nbPad.top + nbPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="6,3"
            />
            <circle
              :cx="nbXScale(temperature)"
              :cy="nbYScale(clampLogNb(totalNoise))"
              r="5" fill="var(--vp-c-text-1)" stroke="#fff" stroke-width="1.5"
            />

            <!-- Legend -->
            <g :transform="`translate(${nbPad.left + 8}, ${nbPad.top + 8})`">
              <line x1="0" y1="6" x2="16" y2="6" stroke="var(--vp-c-text-1)" stroke-width="2.5" />
              <text x="20" y="10" class="legend-label">{{ t('Total', '합계') }}</text>
              <line x1="0" y1="20" x2="16" y2="20" stroke="#e74c3c" stroke-width="1.5" />
              <text x="20" y="24" class="legend-label">{{ t('Dark', '암전류') }}</text>
              <line x1="0" y1="34" x2="16" y2="34" stroke="#3498db" stroke-width="1.5" stroke-dasharray="6,3" />
              <text x="20" y="38" class="legend-label">{{ t('Read', '읽기') }}</text>
            </g>

            <!-- Hover tooltip -->
            <template v-if="nbHover">
              <line :x1="nbHover.sx" :y1="nbPad.top" :x2="nbHover.sx" :y2="nbPad.top + nbPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
              <rect :x="nbHover.tx" :y="nbPad.top + 4" width="130" height="46" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
              <text :x="nbHover.tx + 6" :y="nbPad.top + 16" class="tooltip-text">T = {{ nbHover.temp.toFixed(0) }}&deg;C</text>
              <text :x="nbHover.tx + 6" :y="nbPad.top + 28" class="tooltip-text">{{ t('Dark', '암전류') }}: {{ nbHover.dark.toFixed(3) }} e&minus;</text>
              <text :x="nbHover.tx + 6" :y="nbPad.top + 40" class="tooltip-text">{{ t('Total', '합계') }}: {{ nbHover.total.toFixed(3) }} e&minus;</text>
            </template>
          </svg>
        </div>
      </div>
    </div>

    <!-- Dark frame visualization -->
    <div class="chart-section">
      <h5>{{ t('Dark Frame Visualization', '암전류 프레임 시각화') }}</h5>
      <p class="frame-desc">
        {{ t(
          'Simulated dark frame image showing noise pattern at current temperature and integration time.',
          '현재 온도와 적분 시간에서의 잡음 패턴을 보여주는 시뮬레이션된 암전류 프레임 이미지.'
        ) }}
      </p>
      <div class="canvas-wrapper">
        <canvas ref="darkFrameCanvas" :width="canvasW" :height="canvasH" class="dark-frame-canvas"></canvas>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// ---- Controls ----
const temperature = ref(25)
const pitch = ref(1.0)
const integrationTime = ref(33)
const readNoise = ref(1.5)
const illuminance = ref(100)

// ---- Physics constants ----
const Eg = 1.12       // eV, Si bandgap
const kB = 8.617e-5   // eV/K, Boltzmann constant

// Calibrate J0 so Jd = 5 e-/s/um^2 at T=300K
// Jd(T) = J0 * T^1.5 * exp(-Eg/(2*kB*T))
// J0 = 5 / (300^1.5 * exp(-Eg/(2*kB*300)))
const T_ref = 300
const J0 = 5 / (Math.pow(T_ref, 1.5) * Math.exp(-Eg / (2 * kB * T_ref)))

function darkCurrentDensity(tempC: number): number {
  const T = tempC + 273.15
  return J0 * Math.pow(T, 1.5) * Math.exp(-Eg / (2 * kB * T))
}

function darkCurrentPixel(tempC: number, pitchUm: number): number {
  return darkCurrentDensity(tempC) * pitchUm * pitchUm
}

// ---- Computed values ----
const darkCurrentAtT = computed(() => darkCurrentPixel(temperature.value, pitch.value))
const tIntSec = computed(() => integrationTime.value / 1000)
const darkSignal = computed(() => darkCurrentAtT.value * tIntSec.value)
const darkNoise = computed(() => Math.sqrt(darkSignal.value))
const totalNoise = computed(() => Math.sqrt(readNoise.value ** 2 + darkSignal.value))

// Photon signal (simplified)
const photonSignal = computed(() => {
  const QE = 0.6 // assume 60% average QE
  return QE * illuminance.value * (pitch.value * 1e-4) ** 2 * tIntSec.value * 4e11
})

const snrDb = computed(() => {
  const sig = photonSignal.value
  const noise = Math.sqrt(readNoise.value ** 2 + darkSignal.value + sig) // shot noise from signal too
  if (sig <= 0 || noise <= 0) return 0
  return 20 * Math.log10(sig / noise)
})

// ---- Dark current vs Temperature chart ----
const dcW = 300
const dcH = 260
const dcPad = { top: 16, right: 16, bottom: 36, left: 50 }
const dcPlotW = dcW - dcPad.left - dcPad.right
const dcPlotH = dcH - dcPad.top - dcPad.bottom

const dcXMin = -20
const dcXMax = 85
const dcXTicks = [-20, 0, 20, 40, 60, 85]

// Log scale Y: 0.01 to 10000
const dcYMin = -2  // log10(0.01)
const dcYMax = 4   // log10(10000)

function dcXScale(tempC: number): number {
  return dcPad.left + ((tempC - dcXMin) / (dcXMax - dcXMin)) * dcPlotW
}
function dcYScale(logVal: number): number {
  return dcPad.top + dcPlotH - ((logVal - dcYMin) / (dcYMax - dcYMin)) * dcPlotH
}

function clampLog(v: number): number {
  if (v <= 0) return dcYMin
  const lv = Math.log10(v)
  return Math.max(dcYMin, Math.min(dcYMax, lv))
}

const dcCurvePath = computed(() => {
  const steps = 200
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const tempC = dcXMin + (i / steps) * (dcXMax - dcXMin)
    const dc = darkCurrentPixel(tempC, pitch.value)
    const logDc = clampLog(dc)
    const x = dcXScale(tempC)
    const y = dcYScale(logDc)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
})

// DC chart hover
const dcHover = ref<{ sx: number; tx: number; temp: number; dc: number } | null>(null)

function onDcMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = dcW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const tempC = dcXMin + ((mouseX - dcPad.left) / dcPlotW) * (dcXMax - dcXMin)
  if (tempC >= dcXMin && tempC <= dcXMax) {
    const dc = darkCurrentPixel(tempC, pitch.value)
    const sx = dcXScale(tempC)
    const tx = sx + 130 > dcW - dcPad.right ? sx - 130 : sx + 10
    dcHover.value = { sx, tx, temp: tempC, dc }
  } else {
    dcHover.value = null
  }
}

// ---- Noise budget vs Temperature chart ----
const nbW = 300
const nbH = 260
const nbPad = { top: 16, right: 16, bottom: 36, left: 46 }
const nbPlotW = nbW - nbPad.left - nbPad.right
const nbPlotH = nbH - nbPad.top - nbPad.bottom

const nbXMin = -20
const nbXMax = 85
const nbXTicks = [-20, 0, 20, 40, 60, 85]

// Log scale Y: 0.1 to 100
const nbYMin = -1  // log10(0.1)
const nbYMax = 2   // log10(100)

function nbXScale(tempC: number): number {
  return nbPad.left + ((tempC - nbXMin) / (nbXMax - nbXMin)) * nbPlotW
}
function nbYScale(logVal: number): number {
  return nbPad.top + nbPlotH - ((logVal - nbYMin) / (nbYMax - nbYMin)) * nbPlotH
}

function clampLogNb(v: number): number {
  if (v <= 0) return nbYMin
  const lv = Math.log10(v)
  return Math.max(nbYMin, Math.min(nbYMax, lv))
}

function buildNoisePath(noiseFn: (tempC: number) => number): string {
  const steps = 200
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const tempC = nbXMin + (i / steps) * (nbXMax - nbXMin)
    const noise = noiseFn(tempC)
    const logN = clampLogNb(noise)
    const x = nbXScale(tempC)
    const y = nbYScale(logN)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

const readNoisePath = computed(() => buildNoisePath(() => readNoise.value))

const darkNoisePath = computed(() => buildNoisePath((tempC) => {
  const dc = darkCurrentPixel(tempC, pitch.value)
  const sd = dc * tIntSec.value
  return Math.sqrt(sd)
}))

const totalNoisePath = computed(() => buildNoisePath((tempC) => {
  const dc = darkCurrentPixel(tempC, pitch.value)
  const sd = dc * tIntSec.value
  return Math.sqrt(readNoise.value ** 2 + sd)
}))

// NB chart hover
const nbHover = ref<{ sx: number; tx: number; temp: number; dark: number; total: number } | null>(null)

function onNbMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = nbW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const tempC = nbXMin + ((mouseX - nbPad.left) / nbPlotW) * (nbXMax - nbXMin)
  if (tempC >= nbXMin && tempC <= nbXMax) {
    const dc = darkCurrentPixel(tempC, pitch.value)
    const sd = dc * tIntSec.value
    const dn = Math.sqrt(sd)
    const tn = Math.sqrt(readNoise.value ** 2 + sd)
    const sx = nbXScale(tempC)
    const tx = sx + 140 > nbW - nbPad.right ? sx - 140 : sx + 10
    nbHover.value = { sx, tx, temp: tempC, dark: dn, total: tn }
  } else {
    nbHover.value = null
  }
}

// ---- Dark frame visualization (Canvas) ----
const darkFrameCanvas = ref<HTMLCanvasElement | null>(null)
const canvasW = 280
const canvasH = 200
const pixelSize = 4  // each simulated pixel is 4x4 screen pixels
const frameW = Math.floor(canvasW / pixelSize)
const frameH = Math.floor(canvasH / pixelSize)

// Simple LCG for deterministic hot pixel positions
function lcg(seed: number): () => number {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    return (s >>> 0) / 0xFFFFFFFF
  }
}

// Generate deterministic hot pixel positions (15 hot pixels)
const HOT_PIXEL_COUNT = 15
const hotPixels: { x: number; y: number; factor: number }[] = []
{
  const rng = lcg(42)
  for (let i = 0; i < HOT_PIXEL_COUNT; i++) {
    hotPixels.push({
      x: Math.floor(rng() * frameW),
      y: Math.floor(rng() * frameH),
      factor: 5 + rng() * 5, // 5-10x average dark current
    })
  }
}

// Box-Muller gaussian
function gaussRandom(): number {
  let u = 0, v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function renderDarkFrame() {
  const canvas = darkFrameCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const ds = darkSignal.value
  const dn = darkNoise.value

  // Build pixel values
  const values: number[] = new Array(frameW * frameH)
  let maxVal = 0

  // Create a set for fast hot pixel lookup
  const hotSet = new Map<number, number>()
  for (const hp of hotPixels) {
    hotSet.set(hp.y * frameW + hp.x, hp.factor)
  }

  for (let y = 0; y < frameH; y++) {
    for (let x = 0; x < frameW; x++) {
      const idx = y * frameW + x
      const factor = hotSet.get(idx) || 1
      const pixelDarkSignal = ds * factor
      const pixelDarkNoise = Math.sqrt(pixelDarkSignal > 0 ? pixelDarkSignal : 0)
      const noiseVal = pixelDarkNoise > 0 ? gaussRandom() * pixelDarkNoise : 0
      const val = Math.max(0, pixelDarkSignal + noiseVal)
      values[idx] = val
      if (val > maxVal) maxVal = val
    }
  }

  // Auto-scale to 0-255
  const scale = maxVal > 0 ? 255 / maxVal : 0
  const imageData = ctx.createImageData(canvasW, canvasH)

  for (let y = 0; y < frameH; y++) {
    for (let x = 0; x < frameW; x++) {
      const v = Math.min(255, Math.round(values[y * frameW + x] * scale))
      // Fill a pixelSize x pixelSize block
      for (let dy = 0; dy < pixelSize; dy++) {
        for (let dx = 0; dx < pixelSize; dx++) {
          const px = x * pixelSize + dx
          const py = y * pixelSize + dy
          if (px < canvasW && py < canvasH) {
            const off = (py * canvasW + px) * 4
            imageData.data[off] = v
            imageData.data[off + 1] = v
            imageData.data[off + 2] = v
            imageData.data[off + 3] = 255
          }
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0)
}

// Watch reactive values and re-render
watch(
  [temperature, pitch, integrationTime],
  () => { nextTick(renderDarkFrame) },
  { immediate: false }
)

onMounted(() => {
  nextTick(renderDarkFrame)
})
</script>

<style scoped>
.dc-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.dc-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.dc-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 180px;
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
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
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

/* Charts row */
.charts-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 20px;
}
.chart-half {
  flex: 1;
  min-width: 280px;
}
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.dc-svg, .nb-svg {
  width: 100%;
  max-width: 300px;
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

/* Dark frame canvas */
.frame-desc {
  font-size: 0.85em;
  color: var(--vp-c-text-3);
  margin: 0 0 8px 0;
}
.canvas-wrapper {
  display: flex;
  justify-content: center;
}
.dark-frame-canvas {
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: #000;
  image-rendering: pixelated;
}
</style>
