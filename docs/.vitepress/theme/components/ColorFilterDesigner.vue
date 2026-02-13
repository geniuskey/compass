<template>
  <div class="cf-container">
    <h4>{{ t('Color Filter Designer & Gamut Viewer', '컬러 필터 설계 및 색역 뷰어') }}</h4>
    <p class="component-description">
      {{ t(
        'Design color filter spectral responses and visualize the resulting color gamut on a CIE 1931 chromaticity diagram.',
        '컬러 필터의 분광 응답을 설계하고 CIE 1931 색도도에서 결과 색역을 시각화합니다.'
      ) }}
    </p>

    <!-- Controls for R, G, B filters -->
    <div class="filter-controls">
      <div v-for="f in filters" :key="f.id" class="filter-group" :style="{ borderLeftColor: f.color }">
        <div class="filter-header">
          <span class="filter-dot" :style="{ background: f.color }"></span>
          <strong>{{ t(f.nameEn, f.nameKo) }}</strong>
        </div>
        <div class="filter-sliders">
          <div class="slider-row">
            <label>{{ t('Center', '중심') }}: <strong>{{ f.center.value }} nm</strong></label>
            <input type="range" :min="f.centerMin" :max="f.centerMax" step="1" v-model.number="f.center.value" class="ctrl-range" />
          </div>
          <div class="slider-row">
            <label>{{ t('FWHM', '반치폭') }}: <strong>{{ f.fwhm.value }} nm</strong></label>
            <input type="range" min="20" max="120" step="2" v-model.number="f.fwhm.value" class="ctrl-range" />
          </div>
          <div class="slider-row">
            <label>{{ t('Peak', '피크') }}: <strong>{{ f.peak.value }}%</strong></label>
            <input type="range" min="50" max="100" step="1" v-model.number="f.peak.value" class="ctrl-range" />
          </div>
        </div>
      </div>
    </div>

    <!-- Info cards -->
    <div class="results-grid">
      <div class="result-card gamut-card">
        <div class="result-label">{{ t('Gamut Area', '색역 면적') }}</div>
        <div class="result-value highlight">{{ gamutAreaPct.toFixed(1) }}% {{ t('of sRGB', 'sRGB 대비') }}</div>
      </div>
      <div v-for="f in filters" :key="'info-'+f.id" class="result-card" :style="{ borderTop: `3px solid ${f.color}` }">
        <div class="result-label">{{ t(f.nameEn, f.nameKo) }}</div>
        <div class="result-value">{{ filterChroma[f.id].domWl }} nm</div>
        <div class="result-sub">{{ t('Purity', '순도') }}: {{ (filterChroma[f.id].purity * 100).toFixed(1) }}%</div>
      </div>
    </div>

    <!-- Chart 1: Filter Spectra -->
    <div class="chart-section">
      <h5>{{ t('Filter Spectra', '필터 스펙트럼') }}</h5>
      <div class="svg-wrapper">
        <svg
          :viewBox="`0 0 ${specW} ${specH}`"
          class="spec-svg"
          @mousemove="onSpecMouseMove"
          @mouseleave="specHover = null"
        >
          <!-- Visible spectrum gradient bar -->
          <defs>
            <linearGradient id="cfVisSpectrum" x1="0" y1="0" x2="1" y2="0">
              <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
            </linearGradient>
          </defs>
          <rect
            :x="specXScale(380)" :y="specPad.top + specPlotH + 2"
            :width="specXScale(780) - specXScale(380)" height="6"
            fill="url(#cfVisSpectrum)" rx="2"
          />

          <!-- Grid -->
          <line
            v-for="tick in specXTicks" :key="'spxg'+tick"
            :x1="specXScale(tick)" :y1="specPad.top"
            :x2="specXScale(tick)" :y2="specPad.top + specPlotH"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            v-for="tick in specYTicks" :key="'spyg'+tick"
            :x1="specPad.left" :y1="specYScale(tick)"
            :x2="specPad.left + specPlotW" :y2="specYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />

          <!-- Axes -->
          <line :x1="specPad.left" :y1="specPad.top" :x2="specPad.left" :y2="specPad.top + specPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="specPad.left" :y1="specPad.top + specPlotH" :x2="specPad.left + specPlotW" :y2="specPad.top + specPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

          <!-- X tick labels -->
          <text
            v-for="tick in specXTicks" :key="'spxl'+tick"
            :x="specXScale(tick)" :y="specPad.top + specPlotH + 20"
            text-anchor="middle" class="tick-label"
          >{{ tick }}</text>

          <!-- Y tick labels -->
          <text
            v-for="tick in specYTicks" :key="'spyl'+tick"
            :x="specPad.left - 6" :y="specYScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick }}%</text>

          <!-- Axis titles -->
          <text :x="specPad.left + specPlotW / 2" :y="specH - 2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
          <text :x="12" :y="specPad.top + specPlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${specPad.top + specPlotH / 2})`">{{ t('Transmission (%)', '투과율 (%)') }}</text>

          <!-- Filter fills -->
          <path v-for="f in filters" :key="'fill-'+f.id" :d="filterAreaPath(f)" :fill="f.color" opacity="0.15" />
          <!-- Filter curves -->
          <path v-for="f in filters" :key="'curve-'+f.id" :d="filterCurvePath(f)" fill="none" :stroke="f.color" stroke-width="2" />

          <!-- Hover tooltip -->
          <template v-if="specHover">
            <line :x1="specHover.sx" :y1="specPad.top" :x2="specHover.sx" :y2="specPad.top + specPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
            <rect :x="specHover.tx" :y="specPad.top + 4" width="110" height="50" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
            <text :x="specHover.tx + 6" :y="specPad.top + 16" class="tooltip-text">&lambda; = {{ specHover.wl }} nm</text>
            <text :x="specHover.tx + 6" :y="specPad.top + 28" class="tooltip-text" fill="#e74c3c">R: {{ specHover.r.toFixed(1) }}%</text>
            <text :x="specHover.tx + 6" :y="specPad.top + 40" class="tooltip-text" fill="#27ae60">G: {{ specHover.g.toFixed(1) }}%</text>
            <text :x="specHover.tx + 6" :y="specPad.top + 52" class="tooltip-text" fill="#3498db">B: {{ specHover.b.toFixed(1) }}%</text>
          </template>
        </svg>
      </div>
    </div>

    <!-- Chart 2: CIE 1931 Chromaticity Diagram -->
    <div class="chart-section">
      <h5>{{ t('CIE 1931 Chromaticity Diagram', 'CIE 1931 색도도') }}</h5>
      <div class="svg-wrapper cie-wrapper">
        <svg
          :viewBox="`0 0 ${cieW} ${cieH}`"
          class="cie-svg"
        >
          <!-- Axes -->
          <line :x1="ciePad.left" :y1="ciePad.top" :x2="ciePad.left" :y2="ciePad.top + ciePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="ciePad.left" :y1="ciePad.top + ciePlotH" :x2="ciePad.left + ciePlotW" :y2="ciePad.top + ciePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

          <!-- Grid -->
          <template v-for="tick in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]" :key="'cxg'+tick">
            <line
              :x1="cieXScale(tick)" :y1="ciePad.top"
              :x2="cieXScale(tick)" :y2="ciePad.top + ciePlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
            />
            <text :x="cieXScale(tick)" :y="ciePad.top + ciePlotH + 14" text-anchor="middle" class="tick-label">{{ tick.toFixed(1) }}</text>
          </template>
          <template v-for="tick in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]" :key="'cyg'+tick">
            <line
              :x1="ciePad.left" :y1="cieYScale(tick)"
              :x2="ciePad.left + ciePlotW" :y2="cieYScale(tick)"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
            />
            <text :x="ciePad.left - 6" :y="cieYScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
          </template>

          <!-- Axis titles -->
          <text :x="ciePad.left + ciePlotW / 2" :y="cieH - 2" text-anchor="middle" class="axis-title">x</text>
          <text :x="10" :y="ciePad.top + ciePlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${ciePad.top + ciePlotH / 2})`">y</text>

          <!-- Spectral locus (horseshoe) -->
          <path :d="locusPath" fill="none" stroke="var(--vp-c-text-2)" stroke-width="1.5" />
          <!-- Purple line (close the horseshoe) -->
          <line
            :x1="cieXScale(locusPoints[0][0])" :y1="cieYScale(locusPoints[0][1])"
            :x2="cieXScale(locusPoints[locusPoints.length - 1][0])" :y2="cieYScale(locusPoints[locusPoints.length - 1][1])"
            stroke="var(--vp-c-text-2)" stroke-width="1.5" stroke-dasharray="4,3"
          />

          <!-- Wavelength labels along locus -->
          <template v-for="lbl in locusLabels" :key="'ll'+lbl.wl">
            <circle :cx="cieXScale(lbl.x)" :cy="cieYScale(lbl.y)" r="2" fill="var(--vp-c-text-3)" />
            <text :x="cieXScale(lbl.x) + lbl.dx" :y="cieYScale(lbl.y) + lbl.dy" class="locus-label">{{ lbl.wl }}</text>
          </template>

          <!-- sRGB reference triangle -->
          <polygon
            :points="srgbTrianglePoints"
            fill="none" stroke="#888" stroke-width="1" stroke-dasharray="6,3" opacity="0.6"
          />
          <text :x="cieXScale(0.64) + 4" :y="cieYScale(0.33) + 3" class="srgb-label">sRGB</text>

          <!-- Filter gamut triangle -->
          <polygon
            :points="gamutTrianglePoints"
            fill="var(--vp-c-brand-1)" fill-opacity="0.12"
            stroke="var(--vp-c-brand-1)" stroke-width="2"
          />

          <!-- Filter chromaticity points -->
          <template v-for="f in filters" :key="'pt-'+f.id">
            <circle
              :cx="cieXScale(filterChroma[f.id].x)"
              :cy="cieYScale(filterChroma[f.id].y)"
              r="5" :fill="f.color" stroke="#fff" stroke-width="1.5"
            />
            <text
              :x="cieXScale(filterChroma[f.id].x) + (f.id === 'r' ? 8 : f.id === 'b' ? -8 : 0)"
              :y="cieYScale(filterChroma[f.id].y) + (f.id === 'g' ? -8 : 12)"
              :text-anchor="f.id === 'b' ? 'end' : 'start'"
              class="point-label"
              :fill="f.color"
            >{{ t(f.nameEn, f.nameKo) }}</text>
          </template>

          <!-- D65 white point -->
          <line :x1="cieXScale(0.3127) - 5" :y1="cieYScale(0.3290)" :x2="cieXScale(0.3127) + 5" :y2="cieYScale(0.3290)" stroke="#555" stroke-width="1.5" />
          <line :x1="cieXScale(0.3127)" :y1="cieYScale(0.3290) - 5" :x2="cieXScale(0.3127)" :y2="cieYScale(0.3290) + 5" stroke="#555" stroke-width="1.5" />
          <text :x="cieXScale(0.3127) + 7" :y="cieYScale(0.3290) - 4" class="d65-label">D65</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { CIE_WL, CIE_X, CIE_Y, CIE_Z, spectrumToXYZ, xyzToXy } from '../composables/tmm'

const { t } = useLocale()

// ---- Filter definitions ----
interface FilterDef {
  id: string
  nameEn: string
  nameKo: string
  color: string
  center: ReturnType<typeof ref<number>>
  centerMin: number
  centerMax: number
  fwhm: ReturnType<typeof ref<number>>
  peak: ReturnType<typeof ref<number>>
}

const filters: FilterDef[] = [
  { id: 'r', nameEn: 'Red', nameKo: '빨강', color: '#e74c3c', center: ref(620), centerMin: 580, centerMax: 660, fwhm: ref(60), peak: ref(90) },
  { id: 'g', nameEn: 'Green', nameKo: '초록', color: '#27ae60', center: ref(530), centerMin: 500, centerMax: 560, fwhm: ref(50), peak: ref(90) },
  { id: 'b', nameEn: 'Blue', nameKo: '파랑', color: '#3498db', center: ref(450), centerMin: 420, centerMax: 480, fwhm: ref(50), peak: ref(90) },
]

// ---- Gaussian filter model ----
const LN2 = Math.log(2)

function gaussian(wlNm: number, center: number, fwhm: number, peakPct: number): number {
  const peak = peakPct / 100
  return peak * Math.exp(-4 * LN2 * ((wlNm - center) / fwhm) ** 2)
}

function filterTransmission(f: FilterDef, wlNm: number): number {
  return gaussian(wlNm, f.center.value, f.fwhm.value, f.peak.value)
}

// ---- Spectra chart ----
const specW = 600
const specH = 260
const specPad = { top: 16, right: 16, bottom: 36, left: 46 }
const specPlotW = specW - specPad.left - specPad.right
const specPlotH = specH - specPad.top - specPad.bottom

const specXMin = 380
const specXMax = 780
const specXTicks = [400, 450, 500, 550, 600, 650, 700, 750]
const specYTicks = [0, 25, 50, 75, 100]

function specXScale(wl: number): number {
  return specPad.left + ((wl - specXMin) / (specXMax - specXMin)) * specPlotW
}
function specYScale(pct: number): number {
  return specPad.top + specPlotH - (pct / 100) * specPlotH
}

function filterCurvePath(f: FilterDef): string {
  let d = ''
  for (let wl = specXMin; wl <= specXMax; wl += 2) {
    const val = filterTransmission(f, wl) * 100
    const x = specXScale(wl)
    const y = specYScale(val)
    d += d === '' ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

function filterAreaPath(f: FilterDef): string {
  const curve = filterCurvePath(f)
  const firstX = specXScale(specXMin)
  const lastX = specXScale(specXMax)
  const baseY = specYScale(0)
  return curve + ` L${lastX.toFixed(1)},${baseY.toFixed(1)} L${firstX.toFixed(1)},${baseY.toFixed(1)} Z`
}

// Spectrum gradient
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
  return `rgb(${r},${g},${b})`
}

const spectrumStops = computed(() => {
  const stops: { offset: string; color: string }[] = []
  for (let wl = 380; wl <= 780; wl += 20) {
    stops.push({
      offset: ((wl - 380) / (780 - 380) * 100) + '%',
      color: wavelengthToCSS(wl),
    })
  }
  return stops
})

// Spec hover
const specHover = ref<{ sx: number; tx: number; wl: number; r: number; g: number; b: number } | null>(null)

function onSpecMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = specW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = specXMin + ((mouseX - specPad.left) / specPlotW) * (specXMax - specXMin)
  if (wl >= specXMin && wl <= specXMax) {
    const snapped = Math.round(wl)
    const rv = filterTransmission(filters[0], snapped) * 100
    const gv = filterTransmission(filters[1], snapped) * 100
    const bv = filterTransmission(filters[2], snapped) * 100
    const sx = specXScale(snapped)
    const tx = sx + 120 > specW - specPad.right ? sx - 120 : sx + 10
    specHover.value = { sx, tx, wl: snapped, r: rv, g: gv, b: bv }
  } else {
    specHover.value = null
  }
}

// ---- CIE Chromaticity ----
const cieW = 400
const cieH = 400
const ciePad = { top: 16, right: 16, bottom: 32, left: 36 }
const ciePlotW = cieW - ciePad.left - ciePad.right
const ciePlotH = cieH - ciePad.top - ciePad.bottom

const cieXRange = [0, 0.8]
const cieYRange = [0, 0.9]

function cieXScale(x: number): number {
  return ciePad.left + ((x - cieXRange[0]) / (cieXRange[1] - cieXRange[0])) * ciePlotW
}
function cieYScale(y: number): number {
  return ciePad.top + ciePlotH - ((y - cieYRange[0]) / (cieYRange[1] - cieYRange[0])) * ciePlotH
}

// Spectral locus points
const locusPoints = computed(() => {
  const pts: [number, number][] = []
  for (let i = 0; i < CIE_WL.length; i++) {
    const s = CIE_X[i] + CIE_Y[i] + CIE_Z[i]
    if (s > 0.001) {
      pts.push([CIE_X[i] / s, CIE_Y[i] / s])
    }
  }
  return pts
})

const locusPath = computed(() => {
  const pts = locusPoints.value
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${cieXScale(p[0]).toFixed(1)},${cieYScale(p[1]).toFixed(1)}`
  }).join(' ')
})

// Wavelength labels along locus
const locusLabels = computed(() => {
  const labelWls = [460, 480, 500, 520, 540, 560, 580, 600, 620, 650, 700]
  const labels: { wl: number; x: number; y: number; dx: number; dy: number }[] = []
  for (const wl of labelWls) {
    const idx = Math.round((wl - 380) / 5)
    if (idx >= 0 && idx < CIE_WL.length) {
      const s = CIE_X[idx] + CIE_Y[idx] + CIE_Z[idx]
      if (s > 0.001) {
        const x = CIE_X[idx] / s
        const y = CIE_Y[idx] / s
        // Offset labels outward from center
        let dx = 0, dy = 0
        if (wl <= 490) { dx = -10; dy = 4 }
        else if (wl <= 520) { dx = -8; dy = -6 }
        else if (wl <= 560) { dx = 0; dy = -10 }
        else if (wl <= 600) { dx = 6; dy = -6 }
        else { dx = 8; dy = 4 }
        labels.push({ wl, x, y, dx, dy })
      }
    }
  }
  return labels
})

// sRGB triangle
const srgbTrianglePoints = computed(() => {
  const pts = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]]
  return pts.map(p => `${cieXScale(p[0]).toFixed(1)},${cieYScale(p[1]).toFixed(1)}`).join(' ')
})

// Compute filter chromaticity
const filterChroma = computed(() => {
  const result: Record<string, { x: number; y: number; domWl: number; purity: number }> = {}
  for (const f of filters) {
    // Build spectrum at CIE wavelengths
    const spectrum: number[] = CIE_WL.map(wl => filterTransmission(f, wl * 1000))
    const [X, Y, Z] = spectrumToXYZ(spectrum)
    const [x, y] = xyzToXy(X, Y, Z)

    // Dominant wavelength: find intersection of line from D65 through (x,y) with locus
    const d65x = 0.3127, d65y = 0.3290
    let domWl = f.center.value
    let bestDist = Infinity
    for (let i = 0; i < CIE_WL.length; i++) {
      const s = CIE_X[i] + CIE_Y[i] + CIE_Z[i]
      if (s < 0.001) continue
      const lx = CIE_X[i] / s
      const ly = CIE_Y[i] / s
      // Direction from D65 to filter point
      const dx1 = x - d65x, dy1 = y - d65y
      const dx2 = lx - d65x, dy2 = ly - d65y
      // Check if locus point is roughly in same direction
      const dot = dx1 * dx2 + dy1 * dy2
      if (dot > 0) {
        const cross = Math.abs(dx1 * dy2 - dy1 * dx2)
        const len = Math.sqrt(dx2 * dx2 + dy2 * dy2)
        const dist = len > 0 ? cross / len : Infinity
        if (dist < bestDist) {
          bestDist = dist
          domWl = Math.round(CIE_WL[i] * 1000)
        }
      }
    }

    // Excitation purity
    const distFilter = Math.sqrt((x - d65x) ** 2 + (y - d65y) ** 2)
    // Find the locus point at dominant wavelength
    const domIdx = Math.round((domWl / 1000 - 0.380) / 0.005)
    let purity = 0
    if (domIdx >= 0 && domIdx < CIE_WL.length) {
      const s = CIE_X[domIdx] + CIE_Y[domIdx] + CIE_Z[domIdx]
      if (s > 0.001) {
        const lx = CIE_X[domIdx] / s
        const ly = CIE_Y[domIdx] / s
        const distLocus = Math.sqrt((lx - d65x) ** 2 + (ly - d65y) ** 2)
        purity = distLocus > 0 ? Math.min(1, distFilter / distLocus) : 0
      }
    }

    result[f.id] = { x, y, domWl, purity }
  }
  return result
})

// Gamut triangle points
const gamutTrianglePoints = computed(() => {
  return filters.map(f => {
    const ch = filterChroma.value[f.id]
    return `${cieXScale(ch.x).toFixed(1)},${cieYScale(ch.y).toFixed(1)}`
  }).join(' ')
})

// Gamut area as % of sRGB
function triangleArea(pts: [number, number][]): number {
  const [a, b, c] = pts
  return 0.5 * Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
}

const sRGBArea = triangleArea([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])

const gamutAreaPct = computed(() => {
  const pts: [number, number][] = filters.map(f => [filterChroma.value[f.id].x, filterChroma.value[f.id].y])
  const area = triangleArea(pts)
  return (area / sRGBArea) * 100
})
</script>

<style scoped>
.cf-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.cf-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.cf-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.filter-controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.filter-group {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-left: 4px solid;
  border-radius: 8px;
  padding: 12px;
}
.filter-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
  font-size: 0.9em;
}
.filter-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.filter-sliders {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.slider-row label {
  display: block;
  font-size: 0.8em;
  margin-bottom: 2px;
  color: var(--vp-c-text-2);
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 5px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
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
.result-card.gamut-card {
  border-top: 3px solid var(--vp-c-brand-1);
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
.result-sub {
  font-size: 0.75em;
  color: var(--vp-c-text-3);
  margin-top: 2px;
}
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.spec-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.cie-wrapper {
  display: flex;
  justify-content: center;
}
.cie-svg {
  width: 100%;
  max-width: 400px;
  display: block;
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
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.locus-label {
  font-size: 7px;
  fill: var(--vp-c-text-3);
}
.srgb-label {
  font-size: 8px;
  fill: #888;
  font-style: italic;
}
.point-label {
  font-size: 9px;
  font-weight: 600;
}
.d65-label {
  font-size: 8px;
  fill: #555;
  font-weight: 600;
}
</style>
