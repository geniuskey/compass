<template>
  <div class="wl-explorer-container">
    <h4>{{ t('Wavelength, Silicon Absorption & Pixel Response', '파장, 실리콘 흡수 및 픽셀 응답') }}</h4>
    <p class="component-description">
      {{ t(
        'Drag the marker along the spectrum to explore how wavelength affects silicon absorption depth, optical properties, and pixel performance.',
        '스펙트럼을 따라 마커를 드래그하여 파장이 실리콘 흡수 깊이, 광학 특성 및 픽셀 성능에 미치는 영향을 살펴보세요.'
      ) }}
    </p>

    <!-- Panel 1: Wavelength Selector -->
    <div class="panel spectrum-panel">
      <div class="panel-title">{{ t('Wavelength Selector', '파장 선택기') }}</div>
      <div class="wl-display">
        <span class="wl-lambda">&lambda; = {{ wavelength }} nm</span>
        <span class="wl-color-name" :style="{ color: wavelengthCSS }">
          ({{ colorName }})
        </span>
      </div>
      <div class="spectrum-bar-wrapper">
        <div class="spectrum-bar">
          <div class="spectrum-gradient-vis"></div>
          <div class="spectrum-nir-fade"></div>
        </div>
        <input
          type="range"
          :min="380"
          :max="1100"
          step="1"
          v-model.number="wavelength"
          class="spectrum-range"
        />
        <div class="spectrum-labels">
          <span>380</span>
          <span>500</span>
          <span>600</span>
          <span>700</span>
          <span>800</span>
          <span>900</span>
          <span>1000</span>
          <span>1100</span>
        </div>
      </div>
      <div class="spectrum-region-label">
        <span v-if="wavelength <= 780">{{ t('Visible', '가시광') }}</span>
        <span v-else>{{ t('Near Infrared (NIR)', '근적외선 (NIR)') }}</span>
      </div>
    </div>

    <!-- Panel 2: Silicon Absorption Depth Visualization -->
    <div class="panel absorption-panel">
      <div class="panel-title">{{ t('Silicon Absorption Profile', '실리콘 흡수 프로파일') }}</div>
      <div class="absorption-info">
        {{ t('Absorption depth', '흡수 깊이') }} (1/&alpha;): <strong>{{ absorptionDepthDisplay }}</strong>
      </div>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${absW} ${absH}`" class="absorption-svg">
          <defs>
            <linearGradient id="weGradSi" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#6b2c2c" />
              <stop offset="100%" stop-color="#8B6060" />
            </linearGradient>
          </defs>

          <!-- Silicon block -->
          <rect x="70" y="15" width="320" height="80" fill="url(#weGradSi)" rx="3" opacity="0.85" />
          <text x="230" y="108" text-anchor="middle" class="si-label">{{ t('Silicon (Si)', '실리콘 (Si)') }}</text>

          <!-- Light entry arrow -->
          <line x1="30" y1="55" x2="65" y2="55" :stroke="wavelengthCSS" stroke-width="3" />
          <polygon :points="'65,49 65,61 73,55'" :fill="wavelengthCSS" />
          <text x="25" y="45" text-anchor="middle" class="arrow-label">{{ t('Light', '빛') }}</text>

          <!-- Exponential decay curve -->
          <path :d="decayCurvePath" :stroke="wavelengthCSS" stroke-width="2.5" fill="none" />

          <!-- Decay fill area (semi-transparent) -->
          <path :d="decayFillPath" :fill="wavelengthCSS" opacity="0.15" />

          <!-- 1/alpha depth marker -->
          <line
            :x1="depthMarkerX"
            y1="17"
            :x2="depthMarkerX"
            y2="93"
            stroke="#ff6b6b"
            stroke-width="1.5"
            stroke-dasharray="4,3"
          />
          <text :x="depthMarkerX" y="12" text-anchor="middle" class="depth-marker-label">1/&alpha;</text>

          <!-- Depth scale -->
          <line x1="75" y1="100" x2="386" y2="100" stroke="var(--vp-c-text-3)" stroke-width="0.5" />
          <text x="75" y="118" text-anchor="start" class="scale-text">0</text>
          <text x="386" y="118" text-anchor="end" class="scale-text">{{ scaleMaxLabel }}</text>
          <text x="230" y="128" text-anchor="middle" class="scale-text">{{ t('Depth (um)', '깊이 (um)') }}</text>

          <!-- Intensity labels -->
          <text x="67" y="24" text-anchor="end" class="intensity-label">I<tspan font-size="6" dy="2">0</tspan></text>
          <text x="67" y="92" text-anchor="end" class="intensity-label">0</text>
        </svg>
      </div>
    </div>

    <!-- Panel 3: Key Numbers Cards -->
    <div class="panel numbers-panel">
      <div class="panel-title">{{ t('Optical Properties at', '광학 특성 기준') }} &lambda; = {{ wavelength }} nm</div>
      <div class="numbers-grid">
        <div class="number-card">
          <div class="number-label">{{ t('Refractive index', '굴절률') }} n(&lambda;)</div>
          <div class="number-value">{{ nValue.toFixed(3) }}</div>
        </div>
        <div class="number-card">
          <div class="number-label">{{ t('Extinction coeff.', '소광 계수') }} k(&lambda;)</div>
          <div class="number-value">{{ kValue.toExponential(3) }}</div>
        </div>
        <div class="number-card">
          <div class="number-label">{{ t('Absorption depth', '흡수 깊이') }} 1/&alpha;</div>
          <div class="number-value">{{ absorptionDepthDisplay }}</div>
        </div>
        <div class="number-card">
          <div class="number-label">{{ t('Absorption coeff.', '흡수 계수') }} &alpha;</div>
          <div class="number-value">{{ alphaDisplay }} cm<sup style="font-size:0.7em">-1</sup></div>
        </div>
        <div class="number-card">
          <div class="number-label">{{ t('Typical BSI QE', '일반적인 BSI QE') }}</div>
          <div class="number-value">{{ typicalQE }}%</div>
        </div>
        <div class="number-card">
          <div class="number-label">{{ t('Color region', '색상 영역') }}</div>
          <div class="number-value" :style="{ color: wavelengthCSS }">{{ colorName }}</div>
        </div>
      </div>

      <div class="formula-note">
        <code>&alpha; = 4&pi;k / &lambda;</code>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <code>{{ t('Absorption depth', '흡수 깊이') }} = 1/&alpha; = &lambda; / (4&pi;k)</code>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const wavelength = ref(550)

// Silicon optical constants lookup table
const siData = [
  { wl: 380, n: 6.50, k: 3.600 },
  { wl: 400, n: 5.57, k: 0.387 },
  { wl: 450, n: 4.67, k: 0.148 },
  { wl: 500, n: 4.30, k: 0.073 },
  { wl: 550, n: 4.08, k: 0.028 },
  { wl: 600, n: 3.94, k: 0.016 },
  { wl: 650, n: 3.85, k: 0.010 },
  { wl: 700, n: 3.78, k: 0.007 },
  { wl: 750, n: 3.73, k: 0.005 },
  { wl: 800, n: 3.69, k: 0.004 },
  { wl: 850, n: 3.67, k: 0.003 },
  { wl: 900, n: 3.65, k: 0.0025 },
  { wl: 950, n: 3.63, k: 0.0015 },
  { wl: 1000, n: 3.62, k: 0.0008 },
  { wl: 1100, n: 3.60, k: 0.0001 },
]

function interpolate(wl: number, key: 'n' | 'k'): number {
  if (wl <= siData[0].wl) return siData[0][key]
  if (wl >= siData[siData.length - 1].wl) return siData[siData.length - 1][key]
  for (let i = 0; i < siData.length - 1; i++) {
    if (wl >= siData[i].wl && wl <= siData[i + 1].wl) {
      const frac = (wl - siData[i].wl) / (siData[i + 1].wl - siData[i].wl)
      if (key === 'k') {
        // Log-interpolation for k (spans orders of magnitude)
        const v0 = Math.log(siData[i].k)
        const v1 = Math.log(siData[i + 1].k)
        return Math.exp(v0 + frac * (v1 - v0))
      }
      // Linear interpolation for n
      return siData[i].n + frac * (siData[i + 1].n - siData[i].n)
    }
  }
  return siData[0][key]
}

const nValue = computed(() => interpolate(wavelength.value, 'n'))
const kValue = computed(() => interpolate(wavelength.value, 'k'))

// alpha in cm^-1
const alpha = computed(() => {
  const lambdaCm = wavelength.value * 1e-7
  return (4 * Math.PI * kValue.value) / lambdaCm
})

const alphaDisplay = computed(() => {
  const a = alpha.value
  if (a >= 1e6) return a.toExponential(2)
  if (a >= 1000) return Math.round(a).toLocaleString()
  return a.toFixed(1)
})

// Absorption depth in um
const absorptionDepthUm = computed(() => {
  if (alpha.value === 0) return Infinity
  return 1 / (alpha.value * 1e-4) // cm to um
})

const absorptionDepthDisplay = computed(() => {
  const d = absorptionDepthUm.value
  if (!isFinite(d)) return '> 1000 um'
  if (d < 0.01) return `${(d * 1000).toFixed(0)} nm`
  if (d < 0.1) return `${(d * 1000).toFixed(0)} nm`
  if (d < 10) return `${d.toFixed(2)} um`
  if (d < 100) return `${d.toFixed(1)} um`
  return `${d.toFixed(0)} um`
})

// Typical QE lookup (approximate values for standard BSI pixel)
const typicalQE = computed(() => {
  const wl = wavelength.value
  if (wl < 400) return 15
  if (wl < 420) return 30
  if (wl < 450) return 45
  if (wl < 480) return 55
  if (wl < 520) return 65
  if (wl < 560) return 70
  if (wl < 600) return 65
  if (wl < 640) return 58
  if (wl < 680) return 48
  if (wl < 720) return 35
  if (wl < 780) return 22
  if (wl < 850) return 12
  if (wl < 950) return 5
  return 1
})

// Color name
const colorName = computed(() => {
  const wl = wavelength.value
  if (wl < 420) return t('Violet', '보라색')
  if (wl < 450) return t('Indigo', '남색')
  if (wl < 495) return t('Blue', '파란색')
  if (wl < 520) return t('Cyan', '청록색')
  if (wl < 565) return t('Green', '초록색')
  if (wl < 590) return t('Yellow', '노란색')
  if (wl < 625) return t('Orange', '주황색')
  if (wl < 780) return t('Red', '빨간색')
  if (wl < 1000) return t('Near IR', '근적외선')
  return t('IR', '적외선')
})

// Wavelength to CSS color
function wavelengthToRGB(wl: number): { r: number; g: number; b: number } {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) {
    r = -(wl - 440) / (440 - 380); g = 0; b = 1
  } else if (wl >= 440 && wl < 490) {
    r = 0; g = (wl - 440) / (490 - 440); b = 1
  } else if (wl >= 490 && wl < 510) {
    r = 0; g = 1; b = -(wl - 510) / (510 - 490)
  } else if (wl >= 510 && wl < 580) {
    r = (wl - 510) / (580 - 510); g = 1; b = 0
  } else if (wl >= 580 && wl < 645) {
    r = 1; g = -(wl - 645) / (645 - 580); b = 0
  } else if (wl >= 645 && wl <= 780) {
    r = 1; g = 0; b = 0
  } else if (wl > 780) {
    // NIR: show as dim red/gray
    const fade = Math.max(0, 1 - (wl - 780) / 320)
    r = 0.6 * fade; g = 0.1 * fade; b = 0.1 * fade
  }

  let factor = 1.0
  if (wl >= 380 && wl < 420) factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
  else if (wl >= 700 && wl <= 780) factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
  else if (wl > 780) factor = 1.0

  r = Math.round(255 * Math.pow(Math.max(0, r * factor), 0.8))
  g = Math.round(255 * Math.pow(Math.max(0, g * factor), 0.8))
  b = Math.round(255 * Math.pow(Math.max(0, b * factor), 0.8))
  return { r, g, b }
}

const wavelengthCSS = computed(() => {
  const { r, g, b } = wavelengthToRGB(wavelength.value)
  if (r === 0 && g === 0 && b === 0) return '#888'
  return `rgb(${r}, ${g}, ${b})`
})

// Absorption SVG parameters
const absW = 420
const absH = 135

// Dynamic scale for the depth visualization
const scaleMax = computed(() => {
  const d = absorptionDepthUm.value
  if (!isFinite(d) || d > 50) return 100
  if (d < 0.3) return 0.5
  if (d < 1) return 2
  if (d < 3) return 5
  if (d < 8) return 15
  if (d < 20) return 30
  return 60
})

const scaleMaxLabel = computed(() => {
  const s = scaleMax.value
  if (s < 1) return `${(s * 1000).toFixed(0)} nm`
  return `${s} um`
})

const depthMarkerX = computed(() => {
  if (!isFinite(absorptionDepthUm.value)) return 390
  const frac = Math.min(absorptionDepthUm.value / scaleMax.value, 1)
  return 75 + frac * 311
})

// Exponential decay curve path
const decayCurvePath = computed(() => {
  const steps = 200
  let d = ''
  const alphaUm = alpha.value * 1e-4 // cm^-1 to um^-1
  for (let i = 0; i <= steps; i++) {
    const x = 75 + (311 * i) / steps
    const distUm = (scaleMax.value * i) / steps
    const intensity = Math.exp(-alphaUm * distUm)
    // Map intensity 1 -> y=20, intensity 0 -> y=90
    const y = 20 + (1 - intensity) * 70
    d += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
})

// Fill path for area under curve
const decayFillPath = computed(() => {
  const steps = 200
  const alphaUm = alpha.value * 1e-4
  let d = 'M 75 90' // start bottom-left
  for (let i = 0; i <= steps; i++) {
    const x = 75 + (311 * i) / steps
    const distUm = (scaleMax.value * i) / steps
    const intensity = Math.exp(-alphaUm * distUm)
    const y = 20 + (1 - intensity) * 70
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  d += ` L 386 90 Z` // close bottom-right
  return d
})
</script>

<style scoped>
.wl-explorer-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.wl-explorer-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.panel {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 16px 18px;
  margin-bottom: 14px;
}
.panel-title {
  font-size: 0.92em;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  margin-bottom: 10px;
}

/* Panel 1: Spectrum */
.wl-display {
  font-size: 1.1em;
  margin-bottom: 10px;
}
.wl-lambda {
  font-family: var(--vp-font-family-mono);
  font-weight: 700;
  color: var(--vp-c-text-1);
}
.wl-color-name {
  font-weight: 600;
  margin-left: 6px;
}
.spectrum-bar-wrapper {
  position: relative;
}
.spectrum-bar {
  position: relative;
  height: 24px;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
}
.spectrum-gradient-vis {
  position: absolute;
  left: 0;
  top: 0;
  width: 55.6%; /* 380-780 of 380-1100 range = ~55.6% */
  height: 100%;
  background: linear-gradient(
    to right,
    #7700ff 0%,
    #0000ff 12%,
    #0077ff 22%,
    #00ffff 30%,
    #00ff00 40%,
    #ffff00 55%,
    #ff7700 70%,
    #ff0000 85%,
    #990000 100%
  );
}
.spectrum-nir-fade {
  position: absolute;
  left: 55.6%;
  top: 0;
  right: 0;
  height: 100%;
  background: linear-gradient(to right, #440000, #333 30%, #aaa 70%, #ddd);
  opacity: 0.4;
}
.spectrum-range {
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 28px;
  background: transparent;
  margin-top: -26px;
  z-index: 2;
  display: block;
}
.spectrum-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 28px;
  border-radius: 4px;
  background: var(--vp-c-bg);
  border: 3px solid var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.spectrum-range::-moz-range-thumb {
  width: 20px;
  height: 28px;
  border-radius: 4px;
  background: var(--vp-c-bg);
  border: 3px solid var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.3);
}
.spectrum-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.7em;
  color: var(--vp-c-text-3);
  padding: 2px 0;
}
.spectrum-region-label {
  text-align: center;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-top: 2px;
}

/* Panel 2: Absorption */
.absorption-info {
  font-size: 0.92em;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}
.absorption-info strong {
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-brand-1);
}
.svg-wrapper {
  overflow-x: auto;
}
.absorption-svg {
  width: 100%;
  max-width: 500px;
  display: block;
  margin: 0 auto;
}
.si-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.arrow-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.depth-marker-label {
  font-size: 9px;
  fill: #ff6b6b;
  font-weight: 600;
}
.scale-text {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.intensity-label {
  font-size: 8px;
  fill: var(--vp-c-text-2);
}

/* Panel 3: Numbers */
.numbers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.number-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
  text-align: center;
}
.number-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.number-value {
  font-weight: 700;
  font-size: 1.05em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}
.formula-note {
  text-align: center;
  padding: 8px 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  font-size: 0.85em;
  color: var(--vp-c-text-2);
}
.formula-note code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.92em;
  background: transparent;
  padding: 0;
}

@media (max-width: 600px) {
  .numbers-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .wl-display {
    font-size: 0.95em;
  }
}
</style>
