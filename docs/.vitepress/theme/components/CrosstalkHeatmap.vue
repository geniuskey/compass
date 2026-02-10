<template>
  <div class="crosstalk-container">
    <h4>{{ t('Interactive Pixel Crosstalk Heatmap', '인터랙티브 픽셀 크로스토크 히트맵') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how wavelength and pixel pitch affect optical crosstalk between neighboring pixels. The center pixel is illuminated; surrounding pixels show crosstalk intensity.',
        '파장과 픽셀 피치가 인접 픽셀 간 광학 크로스토크에 미치는 영향을 살펴보세요. 중앙 픽셀이 조명되며, 주변 픽셀은 크로스토크 강도를 나타냅니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelength }} nm</strong>
          <span class="color-dot" :style="{ backgroundColor: wavelengthToCSS(wavelength) }"></span>
        </label>
        <input type="range" min="400" max="700" step="10" v-model.number="wavelength" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Pixel pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.6" max="1.4" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="toggle-group">
        <label class="toggle-label">
          <input type="checkbox" v-model="showBayer" />
          {{ t('Show Bayer overlay', '베이어 오버레이 표시') }}
        </label>
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Absorption Depth', '흡수 깊이') }} (Si):</span>
        <span class="info-value">{{ absorptionDepthDisplay }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Nearest-neighbor Crosstalk', '인접 픽셀 크로스토크') }}:</span>
        <span class="info-value">{{ (nearestCT * 100).toFixed(1) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Total Crosstalk', '전체 크로스토크') }}:</span>
        <span class="info-value">{{ (totalCT * 100).toFixed(1) }}%</span>
      </div>
    </div>

    <div class="heatmap-layout">
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${svgSize} ${svgSize + 40}`" class="heatmap-svg">
          <!-- Grid cells -->
          <template v-for="(cell, idx) in gridCells" :key="idx">
            <rect
              :x="cell.x"
              :y="cell.y"
              :width="cellPx"
              :height="cellPx"
              :fill="cell.bgColor"
              :stroke="cell.isCenter ? 'var(--vp-c-brand-1)' : '#888'"
              :stroke-width="cell.isCenter ? 2.5 : 0.8"
            />
            <!-- Bayer overlay -->
            <rect
              v-if="showBayer"
              :x="cell.x + 2"
              :y="cell.y + 2"
              :width="cellPx - 4"
              :height="cellPx - 4"
              :fill="cell.bayerColor"
              opacity="0.25"
              rx="2"
              style="pointer-events: none"
            />
            <!-- Crosstalk value -->
            <text
              :x="cell.x + cellPx / 2"
              :y="cell.y + cellPx / 2 - 3"
              text-anchor="middle"
              :class="['ct-value', { 'center-label': cell.isCenter }]"
            >{{ cell.isCenter ? t('Source', '소스') : cell.ctPercent + '%' }}</text>
            <!-- Pixel coords -->
            <text
              :x="cell.x + cellPx / 2"
              :y="cell.y + cellPx / 2 + 10"
              text-anchor="middle"
              class="ct-coord"
            >({{ cell.row }},{{ cell.col }})</text>
            <!-- Bayer letter -->
            <text
              v-if="showBayer"
              :x="cell.x + cellPx - 6"
              :y="cell.y + 12"
              text-anchor="end"
              class="bayer-letter"
              :fill="cell.bayerColor"
            >{{ cell.bayerLetter }}</text>
          </template>

          <!-- Color scale bar -->
          <defs>
            <linearGradient id="ctScaleGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#ffffff" />
              <stop offset="50%" stop-color="#f5a0a0" />
              <stop offset="100%" stop-color="#c0392b" />
            </linearGradient>
          </defs>
          <rect :x="gridOffset" :y="svgSize + 8" :width="gridTotalPx" height="12" fill="url(#ctScaleGrad)" rx="3" stroke="var(--vp-c-divider)" stroke-width="0.5" />
          <text :x="gridOffset" :y="svgSize + 34" text-anchor="start" class="scale-label">0%</text>
          <text :x="gridOffset + gridTotalPx" :y="svgSize + 34" text-anchor="end" class="scale-label">{{ maxScalePercent }}%</text>
          <text :x="gridOffset + gridTotalPx / 2" :y="svgSize + 34" text-anchor="middle" class="scale-label">{{ t('Crosstalk', '크로스토크') }}</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const wavelength = ref(550)
const pitch = ref(1.0)
const showBayer = ref(true)
const gridN = 5  // 5x5 grid
const svgSize = 340
const gridOffset = 20
const gridTotalPx = svgSize - gridOffset * 2
const cellPx = gridTotalPx / gridN
const centerIdx = 2  // center of 5x5

// Silicon absorption depth model (approximate)
// absorption depth (um) = 1/alpha, where alpha = 4*pi*k / lambda
const siKData = [
  { wl: 400, k: 0.370 },
  { wl: 450, k: 0.092 },
  { wl: 500, k: 0.044 },
  { wl: 550, k: 0.028 },
  { wl: 600, k: 0.017 },
  { wl: 650, k: 0.0096 },
  { wl: 700, k: 0.0058 },
]

function interpolateK(wl) {
  if (wl <= siKData[0].wl) return siKData[0].k
  if (wl >= siKData[siKData.length - 1].wl) return siKData[siKData.length - 1].k
  for (let i = 0; i < siKData.length - 1; i++) {
    if (wl >= siKData[i].wl && wl <= siKData[i + 1].wl) {
      const t = (wl - siKData[i].wl) / (siKData[i + 1].wl - siKData[i].wl)
      const logK = Math.log(siKData[i].k) * (1 - t) + Math.log(siKData[i + 1].k) * t
      return Math.exp(logK)
    }
  }
  return 0.028
}

const absorptionDepthUm = computed(() => {
  const k = interpolateK(wavelength.value)
  const lambdaCm = wavelength.value * 1e-7
  const alpha = (4 * Math.PI * k) / lambdaCm
  return 1 / (alpha * 1e-4) // convert to um
})

const absorptionDepthDisplay = computed(() => {
  const d = absorptionDepthUm.value
  if (d < 0.1) return `${(d * 1000).toFixed(0)} nm`
  if (d < 10) return `${d.toFixed(2)} um`
  return `${d.toFixed(1)} um`
})

// Crosstalk model: CT proportional to exp(-pitch * distance / absorption_depth)
// Distance is in units of pixel pitch from center
function computeCT(row, col) {
  const dr = row - centerIdx
  const dc = col - centerIdx
  if (dr === 0 && dc === 0) return 1.0 // center pixel
  const dist = Math.sqrt(dr * dr + dc * dc)
  const depthRatio = pitch.value / absorptionDepthUm.value
  // Crosstalk decreases with larger pitch (better isolation) and shorter absorption depth
  // Also decreases with distance
  const ct = 0.15 * Math.exp(-depthRatio * 1.2) * Math.exp(-dist * 0.6)
  return Math.min(ct, 0.5)
}

const nearestCT = computed(() => computeCT(centerIdx, centerIdx + 1))

const totalCT = computed(() => {
  let total = 0
  for (let r = 0; r < gridN; r++) {
    for (let c = 0; c < gridN; c++) {
      if (r === centerIdx && c === centerIdx) continue
      total += computeCT(r, c)
    }
  }
  return total
})

const maxScalePercent = computed(() => {
  const maxCT = nearestCT.value
  return Math.max(1, Math.ceil(maxCT * 100))
})

// Bayer pattern (RGGB, repeated)
const bayerPattern = [
  ['R', 'G', 'R', 'G', 'R'],
  ['G', 'B', 'G', 'B', 'G'],
  ['R', 'G', 'R', 'G', 'R'],
  ['G', 'B', 'G', 'B', 'G'],
  ['R', 'G', 'R', 'G', 'R'],
]

const bayerColors = {
  R: '#e74c3c',
  G: '#27ae60',
  B: '#3498db',
}

const gridCells = computed(() => {
  const cells = []
  const maxCT = nearestCT.value
  for (let r = 0; r < gridN; r++) {
    for (let c = 0; c < gridN; c++) {
      const ct = computeCT(r, c)
      const isCenter = r === centerIdx && c === centerIdx

      let bgColor
      if (isCenter) {
        bgColor = wavelengthToCSS(wavelength.value)
      } else {
        // White to red scale based on crosstalk
        const intensity = maxCT > 0 ? ct / maxCT : 0
        const rr = 255
        const gg = Math.round(255 * (1 - intensity * 0.85))
        const bb = Math.round(255 * (1 - intensity * 0.85))
        bgColor = `rgb(${rr}, ${gg}, ${bb})`
      }

      const bLetter = bayerPattern[r][c]
      cells.push({
        x: gridOffset + c * cellPx,
        y: gridOffset + r * cellPx,
        row: r,
        col: c,
        isCenter,
        ct,
        ctPercent: (ct * 100).toFixed(1),
        bgColor,
        bayerColor: bayerColors[bLetter],
        bayerLetter: bLetter,
      })
    }
  }
  return cells
})

function wavelengthToCSS(wl) {
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
.crosstalk-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.crosstalk-container h4 {
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
  min-width: 160px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.color-dot {
  display: inline-block;
  width: 12px;
  height: 12px;
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
.toggle-group {
  display: flex;
  align-items: center;
}
.toggle-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85em;
  cursor: pointer;
}
.toggle-label input {
  cursor: pointer;
}
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
}
.info-label {
  color: var(--vp-c-text-2);
  margin-right: 4px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.heatmap-layout {
  display: flex;
  justify-content: center;
}
.svg-wrapper {
  flex: 0 0 auto;
}
.heatmap-svg {
  width: 340px;
  max-width: 100%;
}
.ct-value {
  font-size: 10px;
  fill: #333;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  pointer-events: none;
}
.center-label {
  font-size: 11px;
  fill: #fff;
  text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
.ct-coord {
  font-size: 7px;
  fill: #666;
  pointer-events: none;
}
.bayer-letter {
  font-size: 9px;
  font-weight: 700;
  pointer-events: none;
}
.scale-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
</style>
