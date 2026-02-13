<template>
  <div class="psf-container">
    <h4>{{ t('Diffraction PSF Viewer', '회절 PSF 뷰어') }}</h4>
    <p class="component-description">
      {{ t(
        'Interactive Airy pattern viewer with pixel grid overlay. Explore how f-number, wavelength, and pixel pitch affect the diffraction-limited PSF.',
        '픽셀 그리드 오버레이가 있는 대화형 에어리 패턴 뷰어입니다. f-수, 파장, 픽셀 피치가 회절 제한 PSF에 미치는 영향을 탐색하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('f-number', 'f-수') }}: <strong>f/{{ fNumber.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.0" max="8.0" step="0.1" v-model.number="fNumber" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelengthNm }} nm</strong>
          <span class="color-dot" :style="{ backgroundColor: wavelengthToCSS(wavelengthNm) }"></span>
        </label>
        <input type="range" min="400" max="700" step="10" v-model.number="wavelengthNm" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Pixel pitch', '픽셀 피치') }}: <strong>{{ pixelPitch.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.5" max="2.0" step="0.05" v-model.number="pixelPitch" class="ctrl-range" />
      </div>
      <div class="toggle-group">
        <label class="toggle-label">
          <input type="checkbox" v-model="showGrid" />
          {{ t('Pixel grid', '픽셀 그리드') }}
        </label>
      </div>
    </div>

    <div class="display-layout">
      <!-- Left: 2D PSF heatmap -->
      <div class="heatmap-panel">
        <div class="panel-title">{{ t('2D PSF (log scale)', '2D PSF (로그 스케일)') }}</div>
        <div class="canvas-wrapper">
          <canvas ref="canvasRef" width="280" height="280" class="psf-canvas"></canvas>
        </div>
      </div>

      <!-- Right panels -->
      <div class="right-panels">
        <!-- Radial profile -->
        <div class="chart-panel">
          <div class="panel-title">{{ t('Radial Profile', '방사형 프로파일') }}</div>
          <svg :viewBox="`0 0 ${profileW} ${profileH}`" class="profile-svg">
            <!-- Background -->
            <rect x="0" y="0" :width="profileW" :height="profileH" fill="var(--vp-c-bg)" />

            <!-- Pixel aperture shading -->
            <rect
              :x="profMarginL"
              :y="profMarginT"
              :width="profScaleX(pixelPitch / 2) - profMarginL"
              :height="profPlotH"
              fill="var(--vp-c-brand-1)"
              opacity="0.08"
            />
            <text
              :x="(profMarginL + profScaleX(pixelPitch / 2)) / 2"
              :y="profMarginT + profPlotH - 4"
              text-anchor="middle"
              class="tiny-label"
              fill="var(--vp-c-brand-1)"
            >{{ t('pixel', '픽셀') }}</text>

            <!-- Grid lines -->
            <line
              v-for="tick in profYTicks"
              :key="'profgy-' + tick.val"
              :x1="profMarginL"
              :y1="profScaleY(tick.val)"
              :x2="profMarginL + profPlotW"
              :y2="profScaleY(tick.val)"
              stroke="var(--vp-c-divider)"
              stroke-width="0.5"
              opacity="0.5"
            />

            <!-- X axis ticks -->
            <template v-for="xt in profXTicks" :key="'profgx-' + xt">
              <line
                :x1="profScaleX(xt)"
                :y1="profMarginT + profPlotH"
                :x2="profScaleX(xt)"
                :y2="profMarginT + profPlotH + 4"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.5"
              />
              <text
                :x="profScaleX(xt)"
                :y="profMarginT + profPlotH + 14"
                text-anchor="middle"
                class="axis-tick"
              >{{ xt }}</text>
            </template>

            <!-- Y axis labels -->
            <text
              v-for="tick in profYTicks"
              :key="'profyl-' + tick.val"
              :x="profMarginL - 4"
              :y="profScaleY(tick.val) + 3"
              text-anchor="end"
              class="axis-tick"
            >{{ tick.label }}</text>

            <!-- Airy pattern curve -->
            <path :d="radialProfilePath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="1.8" />

            <!-- Airy radius vertical line -->
            <line
              :x1="profScaleX(airyRadius)"
              :y1="profMarginT"
              :x2="profScaleX(airyRadius)"
              :y2="profMarginT + profPlotH"
              stroke="#e74c3c"
              stroke-width="1"
              stroke-dasharray="4,3"
            />
            <text
              :x="profScaleX(airyRadius) + 3"
              :y="profMarginT + 12"
              class="tiny-label"
              fill="#e74c3c"
            >{{ t('Airy r', '에어리 r') }}</text>

            <!-- Axis labels -->
            <text
              :x="profMarginL + profPlotW / 2"
              :y="profileH - 2"
              text-anchor="middle"
              class="axis-label"
            >{{ t('Radius (um)', '반경 (um)') }}</text>
            <text
              :x="8"
              :y="profMarginT + profPlotH / 2"
              text-anchor="middle"
              class="axis-label"
              :transform="`rotate(-90, 8, ${profMarginT + profPlotH / 2})`"
            >{{ t('Intensity', '강도') }}</text>
          </svg>
        </div>

        <!-- Encircled energy -->
        <div class="chart-panel">
          <div class="panel-title">{{ t('Encircled Energy', '원형 에너지') }}</div>
          <svg :viewBox="`0 0 ${eeW} ${eeH}`" class="ee-svg">
            <rect x="0" y="0" :width="eeW" :height="eeH" fill="var(--vp-c-bg)" />

            <!-- Grid lines -->
            <line
              v-for="yp in [20, 40, 60, 80, 100]"
              :key="'eegy-' + yp"
              :x1="eeMarginL"
              :y1="eeScaleY(yp)"
              :x2="eeMarginL + eePlotW"
              :y2="eeScaleY(yp)"
              stroke="var(--vp-c-divider)"
              stroke-width="0.5"
              opacity="0.5"
            />

            <!-- 84% line (Airy disk) -->
            <line
              :x1="eeMarginL"
              :y1="eeScaleY(84)"
              :x2="eeMarginL + eePlotW"
              :y2="eeScaleY(84)"
              stroke="#9b59b6"
              stroke-width="0.8"
              stroke-dasharray="4,3"
            />
            <text
              :x="eeMarginL + eePlotW - 2"
              :y="eeScaleY(84) - 3"
              text-anchor="end"
              class="tiny-label"
              fill="#9b59b6"
            >84%</text>

            <!-- Pixel boundary vertical line -->
            <line
              :x1="eeScaleX(pixelPitch / 2)"
              :y1="eeMarginT"
              :x2="eeScaleX(pixelPitch / 2)"
              :y2="eeMarginT + eePlotH"
              stroke="var(--vp-c-brand-1)"
              stroke-width="0.8"
              stroke-dasharray="4,3"
            />
            <text
              :x="eeScaleX(pixelPitch / 2) + 3"
              :y="eeMarginT + 10"
              class="tiny-label"
              fill="var(--vp-c-brand-1)"
            >{{ t('pixel', '픽셀') }}</text>

            <!-- X axis ticks -->
            <template v-for="xt in eeXTicks" :key="'eegx-' + xt">
              <line
                :x1="eeScaleX(xt)"
                :y1="eeMarginT + eePlotH"
                :x2="eeScaleX(xt)"
                :y2="eeMarginT + eePlotH + 4"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.5"
              />
              <text
                :x="eeScaleX(xt)"
                :y="eeMarginT + eePlotH + 14"
                text-anchor="middle"
                class="axis-tick"
              >{{ xt }}</text>
            </template>

            <!-- Y axis labels -->
            <text
              v-for="yp in [0, 20, 40, 60, 80, 100]"
              :key="'eeyl-' + yp"
              :x="eeMarginL - 4"
              :y="eeScaleY(yp) + 3"
              text-anchor="end"
              class="axis-tick"
            >{{ yp }}%</text>

            <!-- EE curve -->
            <path :d="eeCurvePath" fill="none" stroke="#27ae60" stroke-width="1.8" />

            <!-- EE at pixel boundary marker -->
            <circle
              :cx="eeScaleX(pixelPitch / 2)"
              :cy="eeScaleY(energyInPixel * 100)"
              r="3"
              fill="#27ae60"
              stroke="#fff"
              stroke-width="1"
            />
            <text
              :x="eeScaleX(pixelPitch / 2) + 8"
              :y="eeScaleY(energyInPixel * 100) + 3"
              class="tiny-label"
              fill="#27ae60"
            >{{ (energyInPixel * 100).toFixed(1) }}%</text>

            <!-- Axis labels -->
            <text
              :x="eeMarginL + eePlotW / 2"
              :y="eeH - 2"
              text-anchor="middle"
              class="axis-label"
            >{{ t('Radius (um)', '반경 (um)') }}</text>
            <text
              :x="8"
              :y="eeMarginT + eePlotH / 2"
              text-anchor="middle"
              class="axis-label"
              :transform="`rotate(-90, 8, ${eeMarginT + eePlotH / 2})`"
            >EE (%)</text>
          </svg>
        </div>
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Airy radius', '에어리 반경') }}</span>
        <span class="info-value">{{ airyRadius.toFixed(3) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Airy disk / pixel', '에어리 디스크 / 픽셀') }}</span>
        <span class="info-value">{{ airyPixelRatio.toFixed(2) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Energy in pixel', '픽셀 내 에너지') }}</span>
        <span class="info-value">{{ (energyInPixel * 100).toFixed(1) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Resolution (lp/mm)', '분해능 (lp/mm)') }}</span>
        <span class="info-value">{{ diffractionResolution.toFixed(0) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watchEffect, onMounted } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// Controls
const fNumber = ref(2.8)
const wavelengthNm = ref(550)
const pixelPitch = ref(1.0)
const showGrid = ref(true)

// Derived
const wavelengthUm = computed(() => wavelengthNm.value / 1000)
const airyRadius = computed(() => 1.22 * wavelengthUm.value * fNumber.value)
const airyPixelRatio = computed(() => (2 * airyRadius.value) / pixelPitch.value)
const diffractionResolution = computed(() => {
  // lp/mm = 1/(2.44 * lambda * F) * 1000, with lambda in mm
  const lambdaMm = wavelengthNm.value * 1e-6
  return 1 / (2.44 * lambdaMm * fNumber.value)
})

// Bessel functions
function besselJ0(x: number): number {
  let sum = 1
  let term = 1
  for (let m = 1; m <= 25; m++) {
    term *= -(x * x) / (4 * m * m)
    sum += term
    if (Math.abs(term) < 1e-15) break
  }
  return sum
}

function besselJ1(x: number): number {
  if (Math.abs(x) < 1e-10) return 0
  let sum = x / 2
  let term = x / 2
  for (let m = 1; m <= 25; m++) {
    term *= -(x * x) / (4 * m * (m + 1))
    sum += term
    if (Math.abs(term) < 1e-15) break
  }
  return sum
}

function airyIntensity(r: number, wlUm: number, fNum: number): number {
  if (r < 1e-10) return 1.0
  const x = Math.PI * r / (wlUm * fNum)
  const j1 = besselJ1(x)
  return Math.pow(2 * j1 / x, 2)
}

// Encircled energy: EE(r) = 1 - J0(x)^2 - J1(x)^2
function encircledEnergy(r: number, wlUm: number, fNum: number): number {
  if (r < 1e-10) return 0
  const x = Math.PI * r / (wlUm * fNum)
  const j0 = besselJ0(x)
  const j1 = besselJ1(x)
  return 1 - j0 * j0 - j1 * j1
}

const energyInPixel = computed(() => {
  return encircledEnergy(pixelPitch.value / 2, wavelengthUm.value, fNumber.value)
})

// Canvas rendering
const canvasRef = ref<HTMLCanvasElement | null>(null)

function logColormap(val: number): [number, number, number] {
  // black -> blue -> cyan -> yellow -> white
  const t = Math.max(0, Math.min(1, val))
  let r = 0, g = 0, b = 0
  if (t < 0.25) {
    // black to blue
    const s = t / 0.25
    b = Math.round(s * 200)
  } else if (t < 0.5) {
    // blue to cyan
    const s = (t - 0.25) / 0.25
    g = Math.round(s * 220)
    b = 200
  } else if (t < 0.75) {
    // cyan to yellow
    const s = (t - 0.5) / 0.25
    r = Math.round(s * 255)
    g = 220 + Math.round(s * 35)
    b = Math.round(200 * (1 - s))
  } else {
    // yellow to white
    const s = (t - 0.75) / 0.25
    r = 255
    g = 255
    b = Math.round(s * 255)
  }
  return [r, g, b]
}

function renderCanvas() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const size = 280
  canvas.width = size
  canvas.height = size

  const wlUm = wavelengthUm.value
  const fNum = fNumber.value
  const aR = airyRadius.value
  const range = Math.max(3, 3 * aR)

  const imgData = ctx.createImageData(size, size)
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const x = (px / size - 0.5) * 2 * range
      const y = (py / size - 0.5) * 2 * range
      const r = Math.sqrt(x * x + y * y)
      const I = airyIntensity(r, wlUm, fNum)
      // Log colormap: map log10(I) from [-4, 0] to [0, 1]
      const logI = Math.max(0, (Math.log10(I + 1e-4) + 4) / 4)
      const [cr, cg, cb] = logColormap(logI)
      const idx = (py * size + px) * 4
      imgData.data[idx] = cr
      imgData.data[idx + 1] = cg
      imgData.data[idx + 2] = cb
      imgData.data[idx + 3] = 255
    }
  }
  ctx.putImageData(imgData, 0, 0)

  // Overlay pixel grid
  if (showGrid.value) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1
    const pp = pixelPitch.value

    // Vertical lines
    const startX = -Math.ceil(range / pp) * pp
    for (let gx = startX; gx <= range; gx += pp) {
      const canvasX = ((gx / (2 * range)) + 0.5) * size
      ctx.beginPath()
      ctx.moveTo(canvasX, 0)
      ctx.lineTo(canvasX, size)
      ctx.stroke()
    }
    // Horizontal lines
    for (let gy = startX; gy <= range; gy += pp) {
      const canvasY = ((gy / (2 * range)) + 0.5) * size
      ctx.beginPath()
      ctx.moveTo(0, canvasY)
      ctx.lineTo(size, canvasY)
      ctx.stroke()
    }
  }

  // Airy disk circle overlay
  const airyPx = (aR / range) * (size / 2)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
  ctx.lineWidth = 1
  ctx.setLineDash([4, 3])
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, airyPx, 0, 2 * Math.PI)
  ctx.stroke()
  ctx.setLineDash([])
}

onMounted(() => {
  renderCanvas()
})

watchEffect(() => {
  // Touch reactive deps to trigger re-render
  const _f = fNumber.value
  const _w = wavelengthNm.value
  const _p = pixelPitch.value
  const _g = showGrid.value
  renderCanvas()
})

// Radial profile SVG
const profileW = 300
const profileH = 200
const profMarginL = 38
const profMarginT = 12
const profMarginR = 10
const profMarginB = 28
const profPlotW = profileW - profMarginL - profMarginR
const profPlotH = profileH - profMarginT - profMarginB
const profXMax = 3 // um
const profYMinLog = -4 // log10

const profXTicks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
const profYTicks = [
  { val: 0, label: '1' },
  { val: -1, label: '10\u207B\u00B9' },
  { val: -2, label: '10\u207B\u00B2' },
  { val: -3, label: '10\u207B\u00B3' },
  { val: -4, label: '10\u207B\u2074' },
]

function profScaleX(r: number): number {
  return profMarginL + (r / profXMax) * profPlotW
}

function profScaleY(logVal: number): number {
  // logVal is 0 (top) to -4 (bottom)
  const frac = (0 - logVal) / (0 - profYMinLog)
  return profMarginT + frac * profPlotH
}

const radialProfilePath = computed(() => {
  const wlUm = wavelengthUm.value
  const fNum = fNumber.value
  const numPts = 300
  const pts: string[] = []

  for (let i = 0; i <= numPts; i++) {
    const r = (profXMax * i) / numPts
    const I = airyIntensity(r, wlUm, fNum)
    const logI = Math.max(profYMinLog, Math.log10(Math.max(I, 1e-10)))
    const sx = profScaleX(r)
    const sy = profScaleY(logI)
    pts.push(`${i === 0 ? 'M' : 'L'} ${sx.toFixed(1)} ${sy.toFixed(1)}`)
  }
  return pts.join(' ')
})

// Encircled energy SVG
const eeW = 300
const eeH = 180
const eeMarginL = 38
const eeMarginT = 12
const eeMarginR = 10
const eeMarginB = 28
const eePlotW = eeW - eeMarginL - eeMarginR
const eePlotH = eeH - eeMarginT - eeMarginB
const eeXMax = 3

const eeXTicks = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

function eeScaleX(r: number): number {
  return eeMarginL + (r / eeXMax) * eePlotW
}

function eeScaleY(pct: number): number {
  return eeMarginT + eePlotH - (pct / 100) * eePlotH
}

const eeCurvePath = computed(() => {
  const wlUm = wavelengthUm.value
  const fNum = fNumber.value
  const numPts = 300
  const pts: string[] = []

  for (let i = 0; i <= numPts; i++) {
    const r = (eeXMax * i) / numPts
    const ee = encircledEnergy(r, wlUm, fNum) * 100
    const sx = eeScaleX(r)
    const sy = eeScaleY(Math.min(100, Math.max(0, ee)))
    pts.push(`${i === 0 ? 'M' : 'L'} ${sx.toFixed(1)} ${sy.toFixed(1)}`)
  }
  return pts.join(' ')
})

// Wavelength to CSS color
function wavelengthToCSS(wl: number): string {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) { r = -(wl - 440) / 60; b = 1 }
  else if (wl >= 440 && wl < 490) { g = (wl - 440) / 50; b = 1 }
  else if (wl >= 490 && wl < 510) { g = 1; b = -(wl - 510) / 20 }
  else if (wl >= 510 && wl < 580) { r = (wl - 510) / 70; g = 1 }
  else if (wl >= 580 && wl < 645) { r = 1; g = -(wl - 645) / 65 }
  else if (wl >= 645 && wl <= 780) { r = 1 }
  let f = 1.0
  if (wl >= 380 && wl < 420) f = 0.3 + 0.7 * (wl - 380) / 40
  else if (wl >= 700 && wl <= 780) f = 0.3 + 0.7 * (780 - wl) / 80
  r = Math.round(255 * Math.pow(r * f, 0.8))
  g = Math.round(255 * Math.pow(g * f, 0.8))
  b = Math.round(255 * Math.pow(b * f, 0.8))
  return `rgb(${r}, ${g}, ${b})`
}
</script>

<style scoped>
.psf-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.psf-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  align-items: flex-end;
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 140px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
  color: var(--vp-c-text-1);
}
.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  vertical-align: middle;
  margin-left: 4px;
  border: 1px solid #888;
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
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.toggle-group {
  display: flex;
  align-items: center;
  padding-bottom: 4px;
}
.toggle-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85em;
  cursor: pointer;
  color: var(--vp-c-text-1);
}
.toggle-label input {
  cursor: pointer;
}
.display-layout {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.heatmap-panel {
  flex: 0 0 auto;
}
.right-panels {
  flex: 1;
  min-width: 280px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.chart-panel {
  flex: 0 0 auto;
}
.panel-title {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.canvas-wrapper {
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  overflow: hidden;
  display: inline-block;
}
.psf-canvas {
  display: block;
  width: 280px;
  height: 280px;
}
.profile-svg {
  width: 100%;
  max-width: 320px;
}
.ee-svg {
  width: 100%;
  max-width: 320px;
}
.axis-tick {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.tiny-label {
  font-size: 8px;
  font-weight: 600;
}
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
  flex: 1;
  min-width: 120px;
  text-align: center;
}
.info-label {
  display: block;
  color: var(--vp-c-text-2);
  margin-bottom: 2px;
  font-size: 0.85em;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}
</style>
