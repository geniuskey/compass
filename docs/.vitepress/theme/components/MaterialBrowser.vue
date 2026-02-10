<template>
  <div class="material-browser-container">
    <h4>{{ t('Interactive Material Optical Properties', '대화형 재료 광학 특성') }}</h4>
    <p class="component-description">
      {{ t('Select a material to view its refractive index (n) and extinction coefficient (k) across the visible spectrum.', '재료를 선택하여 가시광선 영역에서의 굴절률(n)과 소광 계수(k)를 확인하세요.') }}
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label for="material-select">{{ t('Material', '재료') }}:</label>
        <select id="material-select" v-model="selectedMaterial" class="material-select">
          <option v-for="m in materialList" :key="m.key" :value="m.key">{{ m.label }}</option>
        </select>
      </div>
      <div v-if="hoverData" class="hover-info">
        <span class="hover-wl">&lambda; = {{ hoverData.wl }} nm</span>
        <span class="hover-n">n = {{ hoverData.n.toFixed(4) }}</span>
        <span class="hover-k">k = {{ hoverData.k.toFixed(4) }}</span>
        <span class="hover-eps">&epsilon; = {{ epsDisplay }}</span>
      </div>
    </div>

    <div class="chart-wrapper">
      <svg
        :viewBox="`0 0 ${W} ${H}`"
        class="material-svg"
        @mousemove="onMouseMove"
        @mouseleave="hoverData = null"
      >
        <!-- Spectrum color band at bottom of plot area -->
        <defs>
          <linearGradient id="spectrumBand" x1="0" y1="0" x2="1" y2="0">
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
        <rect :x="padL" :y="plotBottom + 2" :width="plotW" height="8" fill="url(#spectrumBand)" opacity="0.6" rx="2" />

        <!-- Grid lines -->
        <template v-for="tick in yTicksN" :key="'gn' + tick">
          <line :x1="padL" :y1="nToY(tick)" :x2="padL + plotW" :y2="nToY(tick)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        </template>

        <!-- Axes -->
        <line :x1="padL" :y1="padT" :x2="padL" :y2="plotBottom" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line :x1="padL + plotW" :y1="padT" :x2="padL + plotW" :y2="plotBottom" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line :x1="padL" :y1="plotBottom" :x2="padL + plotW" :y2="plotBottom" stroke="var(--vp-c-text-3)" stroke-width="1" />

        <!-- Left Y-axis label (n) -->
        <text :x="10" :y="padT + plotH / 2" text-anchor="middle" transform-origin="10 160" :transform="`rotate(-90, 10, ${padT + plotH / 2})`" class="axis-label" fill="#3498db">n (refractive index)</text>

        <!-- Left Y-axis ticks -->
        <template v-for="tick in yTicksN" :key="'yn' + tick">
          <line :x1="padL - 4" :y1="nToY(tick)" :x2="padL" :y2="nToY(tick)" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="padL - 7" :y="nToY(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
        </template>

        <!-- Right Y-axis label (k) -->
        <text :x="W - 6" :y="padT + plotH / 2" text-anchor="middle" :transform="`rotate(90, ${W - 6}, ${padT + plotH / 2})`" class="axis-label" fill="#e74c3c">k (extinction coeff.)</text>

        <!-- Right Y-axis ticks -->
        <template v-for="tick in yTicksK" :key="'yk' + tick">
          <line :x1="padL + plotW" :y1="kToY(tick)" :x2="padL + plotW + 4" :y2="kToY(tick)" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="padL + plotW + 7" :y="kToY(tick) + 3" text-anchor="start" class="tick-label">{{ tick.toFixed(2) }}</text>
        </template>

        <!-- X-axis ticks -->
        <template v-for="wl in [400, 450, 500, 550, 600, 650, 700, 750]" :key="'xwl' + wl">
          <line :x1="wlToX(wl)" y1="" :y1="plotBottom" :x2="wlToX(wl)" :y2="plotBottom + 4" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="wlToX(wl)" :y="plotBottom + 22" text-anchor="middle" class="tick-label">{{ wl }}</text>
        </template>
        <text :x="padL + plotW / 2" :y="H - 4" text-anchor="middle" class="axis-label">Wavelength (nm)</text>

        <!-- n curve (blue) -->
        <path :d="nPath" fill="none" stroke="#3498db" stroke-width="2.5" />

        <!-- k curve (red) -->
        <path :d="kPath" fill="none" stroke="#e74c3c" stroke-width="2.5" />

        <!-- Hover vertical line -->
        <line
          v-if="hoverData"
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
          v-if="hoverData"
          :cx="wlToX(hoverData.wl)"
          :cy="nToY(hoverData.n)"
          r="4"
          fill="#3498db"
        />
        <circle
          v-if="hoverData"
          :cx="wlToX(hoverData.wl)"
          :cy="kToY(hoverData.k)"
          r="4"
          fill="#e74c3c"
        />

        <!-- Legend -->
        <line :x1="padL + 15" :y1="padT + 12" :x2="padL + 35" :y2="padT + 12" stroke="#3498db" stroke-width="2.5" />
        <text :x="padL + 40" :y="padT + 16" class="legend-label">n</text>
        <line :x1="padL + 60" :y1="padT + 12" :x2="padL + 80" :y2="padT + 12" stroke="#e74c3c" stroke-width="2.5" />
        <text :x="padL + 85" :y="padT + 16" class="legend-label">k</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const W = 520
const H = 340
const padL = 55
const padR = 55
const padT = 30
const padB = 50
const plotW = W - padL - padR
const plotH = H - padT - padB
const plotBottom = padT + plotH

const materialList = [
  { key: 'silicon', label: 'Silicon (Si)' },
  { key: 'sio2', label: 'Silicon Dioxide (SiO2)' },
  { key: 'si3n4', label: 'Silicon Nitride (Si3N4)' },
  { key: 'air', label: 'Air' },
  { key: 'tungsten', label: 'Tungsten (W)' },
  { key: 'cf_red', label: 'Color Filter - Red' },
  { key: 'cf_green', label: 'Color Filter - Green' },
  { key: 'cf_blue', label: 'Color Filter - Blue' },
]

// Wavelength points (nm)
const wls = [380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780]

const materials = {
  silicon: {
    n: [5.97, 5.57, 5.10, 4.73, 4.46, 4.26, 4.13, 4.06, 4.02, 3.98, 3.95, 3.92, 3.89, 3.87, 3.85, 3.83, 3.80, 3.78, 3.77, 3.75, 3.73],
    k: [2.50, 0.39, 0.31, 0.24, 0.18, 0.13, 0.06, 0.043, 0.032, 0.026, 0.021, 0.017, 0.014, 0.011, 0.009, 0.0074, 0.0059, 0.0048, 0.0038, 0.0030, 0.005],
  },
  sio2: {
    n: wls.map(() => 1.46),
    k: wls.map(() => 0.0),
  },
  si3n4: {
    n: [2.10, 2.08, 2.07, 2.06, 2.06, 2.05, 2.05, 2.04, 2.04, 2.04, 2.03, 2.03, 2.03, 2.03, 2.02, 2.02, 2.02, 2.02, 2.02, 2.01, 2.01],
    k: wls.map(() => 0.0),
  },
  air: {
    n: wls.map(() => 1.0),
    k: wls.map(() => 0.0),
  },
  tungsten: {
    n: [3.49, 3.50, 3.51, 3.52, 3.53, 3.54, 3.56, 3.58, 3.60, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68, 3.69, 3.70, 3.71, 3.72, 3.73],
    k: [2.80, 2.84, 2.88, 2.91, 2.94, 2.97, 3.00, 3.03, 3.06, 3.08, 3.10, 3.13, 3.15, 3.17, 3.19, 3.21, 3.23, 3.25, 3.27, 3.29, 3.30],
  },
  cf_red: {
    n: [1.62, 1.61, 1.61, 1.60, 1.60, 1.60, 1.59, 1.59, 1.59, 1.59, 1.59, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60],
    k: [0.40, 0.38, 0.35, 0.32, 0.28, 0.24, 0.18, 0.12, 0.07, 0.03, 0.01, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
  },
  cf_green: {
    n: [1.56, 1.56, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.56, 1.56, 1.56, 1.56, 1.56, 1.57, 1.57, 1.57],
    k: [0.15, 0.12, 0.08, 0.04, 0.015, 0.005, 0.002, 0.001, 0.001, 0.001, 0.005, 0.015, 0.04, 0.08, 0.12, 0.16, 0.20, 0.22, 0.24, 0.25, 0.26],
  },
  cf_blue: {
    n: [1.61, 1.61, 1.61, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61],
    k: [0.001, 0.001, 0.001, 0.002, 0.003, 0.005, 0.02, 0.06, 0.12, 0.18, 0.24, 0.28, 0.32, 0.35, 0.37, 0.38, 0.39, 0.40, 0.40, 0.40, 0.40],
  },
}

const selectedMaterial = ref('silicon')
const hoverData = ref(null)

const currentData = computed(() => materials[selectedMaterial.value])

// Dynamic y-axis ranges
const nRange = computed(() => {
  const d = currentData.value.n
  const min = Math.min(...d)
  const max = Math.max(...d)
  const pad = Math.max((max - min) * 0.15, 0.1)
  return { min: Math.max(0, min - pad), max: max + pad }
})

const kRange = computed(() => {
  const d = currentData.value.k
  const max = Math.max(...d)
  if (max < 0.001) return { min: 0, max: 0.01 }
  const pad = max * 0.15
  return { min: 0, max: max + pad }
})

const yTicksN = computed(() => {
  const r = nRange.value
  const span = r.max - r.min
  const step = span <= 0.5 ? 0.1 : span <= 2 ? 0.5 : span <= 4 ? 1 : 2
  const ticks = []
  let t = Math.ceil(r.min / step) * step
  while (t <= r.max) {
    ticks.push(parseFloat(t.toFixed(2)))
    t += step
  }
  return ticks
})

const yTicksK = computed(() => {
  const r = kRange.value
  const span = r.max
  const step = span <= 0.05 ? 0.01 : span <= 0.5 ? 0.1 : span <= 2 ? 0.5 : 1
  const ticks = []
  let t = 0
  while (t <= r.max) {
    ticks.push(parseFloat(t.toFixed(3)))
    t += step
  }
  return ticks
})

function wlToX(wl) {
  return padL + ((wl - 380) / (780 - 380)) * plotW
}

function nToY(n) {
  const r = nRange.value
  const frac = (n - r.min) / (r.max - r.min)
  return plotBottom - frac * plotH
}

function kToY(k) {
  const r = kRange.value
  if (r.max === 0) return plotBottom
  const frac = (k - r.min) / (r.max - r.min)
  return plotBottom - frac * plotH
}

function interpolate(arr, wl) {
  if (wl <= wls[0]) return arr[0]
  if (wl >= wls[wls.length - 1]) return arr[arr.length - 1]
  for (let i = 0; i < wls.length - 1; i++) {
    if (wl >= wls[i] && wl <= wls[i + 1]) {
      const t = (wl - wls[i]) / (wls[i + 1] - wls[i])
      return arr[i] * (1 - t) + arr[i + 1] * t
    }
  }
  return arr[0]
}

const nPath = computed(() => {
  const d = currentData.value.n
  let path = ''
  for (let i = 0; i < wls.length; i++) {
    const x = wlToX(wls[i])
    const y = nToY(d[i])
    path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return path
})

const kPath = computed(() => {
  const d = currentData.value.k
  let path = ''
  for (let i = 0; i < wls.length; i++) {
    const x = wlToX(wls[i])
    const y = kToY(d[i])
    path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return path
})

const epsDisplay = computed(() => {
  if (!hoverData.value) return ''
  const n = hoverData.value.n
  const k = hoverData.value.k
  const epsR = n * n - k * k
  const epsI = 2 * n * k
  const sign = epsI >= 0 ? '+' : '-'
  return `${epsR.toFixed(2)} ${sign} ${Math.abs(epsI).toFixed(3)}i`
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
  const n = interpolate(currentData.value.n, rounded)
  const k = interpolate(currentData.value.k, rounded)
  hoverData.value = { wl: rounded, n, k }
}
</script>

<style scoped>
.material-browser-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.material-browser-container h4 {
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
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
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
.material-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.hover-info {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 0.82em;
  font-family: var(--vp-font-family-mono);
}
.hover-wl { color: var(--vp-c-text-2); }
.hover-n { color: #3498db; font-weight: 600; }
.hover-k { color: #e74c3c; font-weight: 600; }
.hover-eps { color: var(--vp-c-text-2); }
.chart-wrapper {
  margin-top: 8px;
}
.material-svg {
  width: 100%;
  max-width: 580px;
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
.legend-label {
  font-size: 11px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
</style>
