<template>
  <div class="wavelength-slider-container">
    <h4>{{ t('Interactive Wavelength Explorer', '대화형 파장 탐색기') }}</h4>
    <p class="component-description">
      {{ t('Drag the slider to explore how wavelength affects color and silicon absorption depth.', '슬라이더를 드래그하여 파장이 색상과 실리콘 흡수 깊이에 미치는 영향을 살펴보세요.') }}
    </p>

    <div class="slider-section">
      <label for="wavelength-input">
        {{ t('Wavelength', '파장') }}: <strong>{{ wavelength }} nm</strong>
      </label>
      <div class="slider-with-gradient">
        <div class="spectrum-gradient" ref="gradientBar"></div>
        <input
          id="wavelength-input"
          type="range"
          min="380"
          max="780"
          step="1"
          v-model.number="wavelength"
          class="wavelength-range"
        />
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card color-card">
        <div class="color-swatch" :style="{ backgroundColor: wavelengthToCSS }"></div>
        <div class="color-label">{{ colorName }}</div>
      </div>

      <div class="result-card">
        <div class="result-label">{{ t('Extinction Coefficient', '소광 계수') }} (k)</div>
        <div class="result-value">{{ kValue.toExponential(3) }}</div>
      </div>

      <div class="result-card">
        <div class="result-label">{{ t('Absorption Coefficient', '흡수 계수') }}</div>
        <div class="result-value">&alpha; = {{ alpha.toFixed(0) }} cm<sup>-1</sup></div>
      </div>

      <div class="result-card">
        <div class="result-label">{{ t('Absorption Depth', '흡수 깊이') }} (1/&alpha;)</div>
        <div class="result-value">{{ absorptionDepthDisplay }}</div>
      </div>
    </div>

    <div class="formula-section">
      <div class="formula">
        &alpha; = 4&pi;k / &lambda;
        &nbsp;&nbsp;&rarr;&nbsp;&nbsp;
        {{ t('Absorption depth', '흡수 깊이') }} &approx; &lambda; / (4&pi;k)
      </div>
    </div>

    <div class="depth-visualization">
      <svg :viewBox="`0 0 400 120`" class="depth-svg">
        <!-- Silicon block -->
        <rect x="50" y="10" width="300" height="80" fill="#8B8682" stroke="#555" stroke-width="1" rx="2" />
        <text x="200" y="105" text-anchor="middle" class="svg-label">Silicon</text>

        <!-- Incoming light arrow -->
        <line x1="55" y1="50" x2="80" y2="50" :stroke="wavelengthToCSS" stroke-width="3" />
        <polygon :points="arrowHead" :fill="wavelengthToCSS" />

        <!-- Exponential decay visualization -->
        <path :d="decayPath" :stroke="wavelengthToCSS" stroke-width="2.5" fill="none" opacity="0.9" />

        <!-- Depth marker -->
        <line :x1="depthMarkerX" y1="12" :x2="depthMarkerX" y2="88" stroke="#ff6b6b" stroke-width="1.5" stroke-dasharray="4,3" />
        <text :x="depthMarkerX" y="8" text-anchor="middle" class="svg-depth-label">1/&alpha;</text>

        <!-- Scale bar -->
        <line x1="85" y1="95" x2="345" y2="95" stroke="#888" stroke-width="0.5" />
        <text x="85" y="105" text-anchor="start" class="svg-scale">0</text>
        <text x="345" y="105" text-anchor="end" class="svg-scale">{{ scaleLabel }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const wavelength = ref(550)

// Silicon extinction coefficient (k) at key wavelengths.
// Interpolated from published Si optical constants (Palik / Green).
const siKData = [
  { wl: 380, k: 2.215 },
  { wl: 400, k: 0.370 },
  { wl: 420, k: 0.194 },
  { wl: 450, k: 0.092 },
  { wl: 480, k: 0.058 },
  { wl: 500, k: 0.044 },
  { wl: 520, k: 0.035 },
  { wl: 550, k: 0.028 },
  { wl: 580, k: 0.021 },
  { wl: 600, k: 0.017 },
  { wl: 620, k: 0.013 },
  { wl: 650, k: 0.0096 },
  { wl: 680, k: 0.0073 },
  { wl: 700, k: 0.0058 },
  { wl: 730, k: 0.0044 },
  { wl: 750, k: 0.0036 },
  { wl: 780, k: 0.0027 },
]

function interpolateK(wl) {
  if (wl <= siKData[0].wl) return siKData[0].k
  if (wl >= siKData[siKData.length - 1].wl) return siKData[siKData.length - 1].k
  for (let i = 0; i < siKData.length - 1; i++) {
    if (wl >= siKData[i].wl && wl <= siKData[i + 1].wl) {
      const t = (wl - siKData[i].wl) / (siKData[i + 1].wl - siKData[i].wl)
      // Log-interpolation for k since it spans orders of magnitude
      const logK = Math.log(siKData[i].k) * (1 - t) + Math.log(siKData[i + 1].k) * t
      return Math.exp(logK)
    }
  }
  return 0.028
}

const kValue = computed(() => interpolateK(wavelength.value))

// alpha in cm^-1: alpha = 4 * pi * k / lambda, with lambda in cm
const alpha = computed(() => {
  const lambdaCm = wavelength.value * 1e-7  // nm to cm
  return (4 * Math.PI * kValue.value) / lambdaCm
})

// Absorption depth in um
const absorptionDepthUm = computed(() => {
  return 1 / (alpha.value * 1e-4)  // 1/alpha in cm, converted to um
})

const absorptionDepthDisplay = computed(() => {
  const d = absorptionDepthUm.value
  if (d < 0.1) return `${(d * 1000).toFixed(0)} nm`
  if (d < 10) return `${d.toFixed(2)} um`
  return `${d.toFixed(1)} um`
})

// The scale of the depth visualization in um
const scaleMax = computed(() => {
  const d = absorptionDepthUm.value
  if (d < 0.5) return 1
  if (d < 2) return 5
  if (d < 5) return 10
  if (d < 15) return 30
  return 50
})

const scaleLabel = computed(() => `${scaleMax.value} um`)

const depthMarkerX = computed(() => {
  const frac = absorptionDepthUm.value / scaleMax.value
  const clampedFrac = Math.min(frac, 1)
  return 85 + clampedFrac * 260
})

const arrowHead = computed(() => {
  return '80,44 80,56 88,50'
})

const decayPath = computed(() => {
  const steps = 100
  let d = 'M 88 50'
  for (let i = 1; i <= steps; i++) {
    const x = 88 + (260 * i) / steps
    const distUm = (scaleMax.value * i) / steps
    const alphaUm = alpha.value * 1e-4  // convert cm^-1 to um^-1
    const intensity = Math.exp(-alphaUm * distUm)
    // Map intensity to y: intensity=1 -> y=50 (center), intensity=0 -> approaches y=50 (flat)
    // Show as amplitude oscillation that decays
    const envelope = intensity * 35
    const phase = (distUm / (wavelength.value * 1e-3)) * 2 * Math.PI
    const y = 50 - envelope * Math.sin(phase)
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
})

function wavelengthToRGB(wl) {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) {
    r = -(wl - 440) / (440 - 380)
    g = 0
    b = 1
  } else if (wl >= 440 && wl < 490) {
    r = 0
    g = (wl - 440) / (490 - 440)
    b = 1
  } else if (wl >= 490 && wl < 510) {
    r = 0
    g = 1
    b = -(wl - 510) / (510 - 490)
  } else if (wl >= 510 && wl < 580) {
    r = (wl - 510) / (580 - 510)
    g = 1
    b = 0
  } else if (wl >= 580 && wl < 645) {
    r = 1
    g = -(wl - 645) / (645 - 580)
    b = 0
  } else if (wl >= 645 && wl <= 780) {
    r = 1
    g = 0
    b = 0
  }
  // Intensity falloff at edges of visible spectrum
  let factor = 1.0
  if (wl >= 380 && wl < 420) factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
  else if (wl >= 700 && wl <= 780) factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

  r = Math.round(255 * Math.pow(r * factor, 0.8))
  g = Math.round(255 * Math.pow(g * factor, 0.8))
  b = Math.round(255 * Math.pow(b * factor, 0.8))
  return { r, g, b }
}

const wavelengthToCSS = computed(() => {
  const { r, g, b } = wavelengthToRGB(wavelength.value)
  return `rgb(${r}, ${g}, ${b})`
})

const colorName = computed(() => {
  const wl = wavelength.value
  if (wl < 420) return 'Violet'
  if (wl < 450) return 'Indigo'
  if (wl < 495) return 'Blue'
  if (wl < 520) return 'Cyan'
  if (wl < 565) return 'Green'
  if (wl < 590) return 'Yellow'
  if (wl < 625) return 'Orange'
  return 'Red'
})
</script>

<style scoped>
.wavelength-slider-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.wavelength-slider-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.slider-section {
  margin-bottom: 20px;
}
.slider-section label {
  display: block;
  margin-bottom: 8px;
  font-size: 0.95em;
}
.slider-with-gradient {
  position: relative;
  height: 36px;
}
.spectrum-gradient {
  position: absolute;
  top: 8px;
  left: 0;
  right: 0;
  height: 20px;
  border-radius: 10px;
  background: linear-gradient(
    to right,
    #7700ff, #0000ff, #0077ff, #00ffff, #00ff00, #ffff00, #ff7700, #ff0000, #990000
  );
  opacity: 0.6;
  pointer-events: none;
}
.wavelength-range {
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 36px;
  background: transparent;
  z-index: 1;
}
.wavelength-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--vp-c-bg);
  border: 3px solid var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.wavelength-range::-moz-range-thumb {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: var(--vp-c-bg);
  border: 3px solid var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
.color-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.color-swatch {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: 2px solid var(--vp-c-divider);
}
.color-label {
  font-weight: 600;
  font-size: 0.95em;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.05em;
  font-family: var(--vp-font-family-mono);
}
.formula-section {
  margin: 16px 0;
  text-align: center;
}
.formula {
  display: inline-block;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 20px;
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}
.depth-visualization {
  margin-top: 16px;
}
.depth-svg {
  width: 100%;
  max-width: 500px;
  display: block;
  margin: 0 auto;
}
.svg-label {
  font-size: 12px;
  fill: var(--vp-c-text-2);
}
.svg-depth-label {
  font-size: 10px;
  fill: #ff6b6b;
  font-weight: 600;
}
.svg-scale {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
</style>
