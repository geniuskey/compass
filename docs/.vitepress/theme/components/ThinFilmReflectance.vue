<template>
  <div class="thinfilm-container">
    <h4>{{ t('Interactive Thin Film Reflectance Calculator', '대화형 박막 반사율 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Compute reflectance spectra using the transfer matrix method for common anti-reflection coating configurations.',
        '전달 행렬법을 사용하여 일반적인 반사 방지막 구성의 반사율 스펙트럼을 계산합니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label for="thinfilm-preset">{{ t('Preset:', '프리셋:') }}</label>
        <select id="thinfilm-preset" v-model="selectedPreset" class="preset-select">
          <option v-for="p in presets" :key="p.key" :value="p.key">{{ t(p.label, p.labelKo) }}</option>
        </select>
      </div>
    </div>

    <div class="layers-section">
      <div class="layer-card" v-for="(layer, idx) in currentLayers" :key="idx">
        <div class="layer-header">
          <span class="layer-index">{{ t('Layer', '레이어') }} {{ idx + 1 }}</span>
          <span class="layer-material">{{ layer.name }} (n={{ layer.n.toFixed(2) }})</span>
        </div>
        <div class="layer-slider">
          <label>{{ t('Thickness:', '두께:') }} <strong>{{ layer.d }} nm</strong></label>
          <input
            type="range"
            min="10"
            max="500"
            step="1"
            :value="layer.d"
            @input="updateThickness(idx, $event)"
            class="ctrl-range"
          />
        </div>
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Min Reflectance', '최소 반사율') }}</div>
        <div class="result-value">{{ minReflectance.value.toFixed(2) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('At Wavelength', '해당 파장') }}</div>
        <div class="result-value">{{ minReflectance.wl }} nm</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Substrate', '기판') }}</div>
        <div class="result-value">Si (n=4.0)</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Incident Medium', '입사 매질') }}</div>
        <div class="result-value">Air (n=1.0)</div>
      </div>
    </div>

    <div class="chart-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="thinfilm-svg" @mousemove="onMouseMove" @mouseleave="hoverData = null">
        <defs>
          <linearGradient id="tfSpectrumBand" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#7700ff" />
            <stop offset="12%" stop-color="#0000ff" />
            <stop offset="25%" stop-color="#0077ff" />
            <stop offset="32%" stop-color="#00ffff" />
            <stop offset="42%" stop-color="#00ff00" />
            <stop offset="55%" stop-color="#ffff00" />
            <stop offset="70%" stop-color="#ff7700" />
            <stop offset="85%" stop-color="#ff0000" />
            <stop offset="100%" stop-color="#990000" />
          </linearGradient>
        </defs>

        <!-- Spectrum color band -->
        <rect :x="padL" :y="plotBottom + 2" :width="plotW" height="8" fill="url(#tfSpectrumBand)" opacity="0.5" rx="2" />

        <!-- Grid lines -->
        <template v-for="tick in yTicks" :key="'gy' + tick">
          <line :x1="padL" :y1="rToY(tick)" :x2="padL + plotW" :y2="rToY(tick)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        </template>

        <!-- Axes -->
        <line :x1="padL" :y1="padT" :x2="padL" :y2="plotBottom" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line :x1="padL" :y1="plotBottom" :x2="padL + plotW" :y2="plotBottom" stroke="var(--vp-c-text-3)" stroke-width="1" />

        <!-- Y-axis label -->
        <text :x="12" :y="padT + plotH / 2" text-anchor="middle" :transform="`rotate(-90, 12, ${padT + plotH / 2})`" class="axis-label">Reflectance (%)</text>

        <!-- Y-axis ticks -->
        <template v-for="tick in yTicks" :key="'yt' + tick">
          <line :x1="padL - 4" :y1="rToY(tick)" :x2="padL" :y2="rToY(tick)" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="padL - 7" :y="rToY(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}</text>
        </template>

        <!-- X-axis ticks -->
        <template v-for="wl in [400, 450, 500, 550, 600, 650, 700, 750]" :key="'xt' + wl">
          <line :x1="wlToX(wl)" :y1="plotBottom" :x2="wlToX(wl)" :y2="plotBottom + 4" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="wlToX(wl)" :y="plotBottom + 22" text-anchor="middle" class="tick-label">{{ wl }}</text>
        </template>
        <text :x="padL + plotW / 2" :y="H - 4" text-anchor="middle" class="axis-label">Wavelength (nm)</text>

        <!-- Reflectance curve -->
        <path :d="reflectancePath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />

        <!-- Min reflectance marker -->
        <circle
          :cx="wlToX(minReflectance.wl)"
          :cy="rToY(minReflectance.value)"
          r="5"
          fill="var(--vp-c-brand-1)"
          stroke="#fff"
          stroke-width="1.5"
        />

        <!-- Hover line -->
        <template v-if="hoverData">
          <line
            :x1="wlToX(hoverData.wl)"
            :y1="padT"
            :x2="wlToX(hoverData.wl)"
            :y2="plotBottom"
            stroke="var(--vp-c-text-2)"
            stroke-width="1"
            stroke-dasharray="3,3"
            opacity="0.6"
          />
          <circle
            :cx="wlToX(hoverData.wl)"
            :cy="rToY(hoverData.r)"
            r="4"
            fill="var(--vp-c-brand-1)"
          />
          <text
            :x="wlToX(hoverData.wl) + 8"
            :y="rToY(hoverData.r) - 8"
            class="hover-label"
          >{{ hoverData.wl }} nm, {{ hoverData.r.toFixed(2) }}%</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, reactive } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const W = 540
const H = 340
const padL = 50
const padR = 20
const padT = 25
const padB = 50
const plotW = W - padL - padR
const plotH = H - padT - padB
const plotBottom = padT + plotH

const nAir = 1.0
const nSi = 4.0

const materialDB = {
  SiO2: { n: 1.46, label: 'SiO2' },
  Si3N4: { n: 2.0, label: 'Si3N4' },
  HfO2: { n: 1.9, label: 'HfO2' },
  TiO2: { n: 2.4, label: 'TiO2' },
}

const presetConfigs = {
  single_arc: {
    label: 'Single layer ARC',
    layers: [{ mat: 'Si3N4', d: 69 }],
  },
  barl: {
    label: 'BARL (SiO2/HfO2)',
    layers: [
      { mat: 'SiO2', d: 80 },
      { mat: 'HfO2', d: 72 },
      { mat: 'SiO2', d: 45 },
    ],
  },
  qw_stack: {
    label: 'Quarter-wave stack',
    layers: [
      { mat: 'TiO2', d: 57 },
      { mat: 'SiO2', d: 94 },
      { mat: 'TiO2', d: 57 },
      { mat: 'SiO2', d: 94 },
    ],
  },
}

const presets = [
  { key: 'single_arc', label: 'Single layer ARC', labelKo: '단층 반사 방지막' },
  { key: 'barl', label: 'BARL (SiO2/HfO2)', labelKo: 'BARL (SiO2/HfO2)' },
  { key: 'qw_stack', label: 'Quarter-wave stack', labelKo: '쿼터파 스택' },
]

const selectedPreset = ref('single_arc')
const hoverData = ref(null)

const layerThicknesses = reactive({
  single_arc: [69],
  barl: [80, 72, 45],
  qw_stack: [57, 94, 57, 94],
})

const currentLayers = computed(() => {
  const cfg = presetConfigs[selectedPreset.value]
  const thicknesses = layerThicknesses[selectedPreset.value]
  return cfg.layers.map((l, i) => ({
    name: materialDB[l.mat].label,
    n: materialDB[l.mat].n,
    d: thicknesses[i],
  }))
})

function updateThickness(idx, event) {
  layerThicknesses[selectedPreset.value][idx] = parseInt(event.target.value)
}

// Transfer matrix method
function computeReflectance(wavelengthNm) {
  const lambda = wavelengthNm
  const layers = currentLayers.value

  // Build characteristic matrix product
  // M = product of M_j for each layer
  // M_j = [[cos(delta), -i*sin(delta)/eta], [-i*eta*sin(delta), cos(delta)]]
  // For normal incidence TE: eta = n
  // delta = 2*pi*n*d / lambda

  // Start with identity matrix (complex 2x2)
  let m00r = 1, m00i = 0
  let m01r = 0, m01i = 0
  let m10r = 0, m10i = 0
  let m11r = 1, m11i = 0

  for (let j = 0; j < layers.length; j++) {
    const n = layers[j].n
    const d = layers[j].d
    const delta = (2 * Math.PI * n * d) / lambda
    const cosd = Math.cos(delta)
    const sind = Math.sin(delta)
    const eta = n

    // Layer matrix:
    // [[cosd, -i*sind/eta], [-i*eta*sind, cosd]]
    const a00r = cosd, a00i = 0
    const a01r = 0,    a01i = -sind / eta
    const a10r = 0,    a10i = -eta * sind
    const a11r = cosd, a11i = 0

    // Multiply M = M * A (complex matrix multiplication)
    const n00r = m00r * a00r - m00i * a00i + m01r * a10r - m01i * a10i
    const n00i = m00r * a00i + m00i * a00r + m01r * a10i + m01i * a10r
    const n01r = m00r * a01r - m00i * a01i + m01r * a11r - m01i * a11i
    const n01i = m00r * a01i + m00i * a01r + m01r * a11i + m01i * a11r
    const n10r = m10r * a00r - m10i * a00i + m11r * a10r - m11i * a10i
    const n10i = m10r * a00i + m10i * a00r + m11r * a10i + m11i * a10r
    const n11r = m10r * a01r - m10i * a01i + m11r * a11r - m11i * a11i
    const n11i = m10r * a01i + m10i * a01r + m11r * a11i + m11i * a11r

    m00r = n00r; m00i = n00i
    m01r = n01r; m01i = n01i
    m10r = n10r; m10i = n10i
    m11r = n11r; m11i = n11i
  }

  // Reflection coefficient:
  // r = (m00*eta_s + m01*eta_0*eta_s - m10 - m11*eta_0) / (m00*eta_s + m01*eta_0*eta_s + m10 + m11*eta_0)
  // For normal incidence: eta_0 = n_air, eta_s = n_substrate
  const eta0 = nAir
  const etaS = nSi

  // Numerator: (m00 + m01*eta_s)*eta0 - (m10 + m11*eta_s)
  // Actually the standard formula is:
  // r = (m00*eta_s + m01*eta0*eta_s - m10 - m11*eta0) / (m00*eta_s + m01*eta0*eta_s + m10 + m11*eta0)
  // Wait, let me use the correct formula.
  // With the system matrix M connecting fields at top and bottom:
  // [B; C] = M * [1; eta_s]  where B and C are normalized
  // Actually for the standard TMM:
  // The total matrix M relates (E_forward + E_backward) at the top interface
  // r = (eta_0 * m00 + eta_0 * eta_s * m01 - m10 - eta_s * m11) /
  //     (eta_0 * m00 + eta_0 * eta_s * m01 + m10 + eta_s * m11)

  // Numerator = eta0*(m00 + etaS*m01) - (m10 + etaS*m11)
  const nr = eta0 * (m00r + etaS * m01r) - (m10r + etaS * m11r)
  const ni = eta0 * (m00i + etaS * m01i) - (m10i + etaS * m11i)

  // Denominator = eta0*(m00 + etaS*m01) + (m10 + etaS*m11)
  const dr = eta0 * (m00r + etaS * m01r) + (m10r + etaS * m11r)
  const di = eta0 * (m00i + etaS * m01i) + (m10i + etaS * m11i)

  // |r|^2 = (nr^2 + ni^2) / (dr^2 + di^2)
  const R = (nr * nr + ni * ni) / (dr * dr + di * di)
  return R * 100 // percentage
}

const wavelengths = computed(() => {
  const wls = []
  for (let wl = 380; wl <= 780; wl += 2) {
    wls.push(wl)
  }
  return wls
})

const reflectanceData = computed(() => {
  return wavelengths.value.map(wl => ({
    wl,
    r: computeReflectance(wl),
  }))
})

const yMax = computed(() => {
  const maxR = Math.max(...reflectanceData.value.map(d => d.r))
  return Math.ceil(maxR / 10) * 10 || 50
})

const yTicks = computed(() => {
  const max = yMax.value
  const step = max <= 20 ? 5 : max <= 50 ? 10 : 20
  const ticks = []
  for (let v = 0; v <= max; v += step) {
    ticks.push(v)
  }
  return ticks
})

const minReflectance = computed(() => {
  let minR = Infinity
  let minWl = 550
  for (const d of reflectanceData.value) {
    if (d.r < minR) {
      minR = d.r
      minWl = d.wl
    }
  }
  return { value: minR, wl: minWl }
})

function wlToX(wl) {
  return padL + ((wl - 380) / (780 - 380)) * plotW
}

function rToY(r) {
  const frac = r / yMax.value
  return plotBottom - frac * plotH
}

const reflectancePath = computed(() => {
  let d = ''
  for (let i = 0; i < reflectanceData.value.length; i++) {
    const pt = reflectanceData.value[i]
    const x = wlToX(pt.wl)
    const y = rToY(pt.r)
    d += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
})

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = W / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = 380 + ((mouseX - padL) / plotW) * (780 - 380)
  if (wl < 380 || wl > 780) {
    hoverData.value = null
    return
  }
  const rounded = Math.round(wl)
  const r = computeReflectance(rounded)
  hoverData.value = { wl: rounded, r }
}
</script>

<style scoped>
.thinfilm-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.thinfilm-container h4 {
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
  align-items: center;
  margin-bottom: 16px;
}
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.9em;
  font-weight: 600;
}
.preset-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.layers-section {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.layer-card {
  flex: 1;
  min-width: 180px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
}
.layer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.layer-index {
  font-size: 0.8em;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}
.layer-material {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.layer-slider label {
  display: block;
  font-size: 0.82em;
  margin-bottom: 4px;
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
  gap: 10px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}
.result-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1em;
  font-family: var(--vp-font-family-mono);
}
.chart-wrapper {
  margin-top: 8px;
}
.thinfilm-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.hover-label {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
</style>
