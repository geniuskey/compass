<template>
  <div class="mla-container">
    <h4>{{ t('Micro Lens Array Visualizer', '마이크로 렌즈 어레이 시각화') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize superellipse microlens array geometry with configurable array patterns, asymmetric radii, and curvature. Switch between top-view height map, cross-section profiles, and 2D ray tracing.',
        '설정 가능한 배열 패턴, 비대칭 반경, 곡률을 가진 초타원 마이크로렌즈 어레이를 시각화합니다. 높이맵, 단면 프로파일, 2D 광선 추적을 전환할 수 있습니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>{{ t('Array', '배열') }}:
          <select v-model="arrayConfig" class="ctrl-select">
            <option value="1x1">1x1</option>
            <option value="2x2">2x2</option>
            <option value="3x3">3x3</option>
            <option value="4x4">4x4</option>
          </select>
        </label>
      </div>
      <div class="slider-group">
        <label>{{ t('Radius X', '반경 X') }}: <strong>{{ Rx.toFixed(2) }} um</strong></label>
        <input type="range" min="0.2" max="2.0" step="0.02" v-model.number="Rx" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Radius Y', '반경 Y') }}: <strong>{{ Ry.toFixed(2) }} um</strong></label>
        <input type="range" min="0.2" max="2.0" step="0.02" v-model.number="Ry" class="ctrl-range" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>{{ t('Height h', '높이 h') }}: <strong>{{ h.toFixed(2) }} um</strong></label>
        <input type="range" min="0.1" max="1.0" step="0.01" v-model.number="h" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Shape n', '형상 n') }}: <strong>{{ n.toFixed(1) }}</strong></label>
        <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="n" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Curvature α', '곡률 α') }}: <strong>{{ alpha.toFixed(2) }}</strong></label>
        <input type="range" min="0.2" max="2.0" step="0.02" v-model.number="alpha" class="ctrl-range" />
      </div>
    </div>

    <div class="controls-row" v-if="arrayConfig !== '1x1'">
      <div class="slider-group">
        <label>{{ t('Spacing X', '간격 X') }}: <strong>{{ spacingX.toFixed(2) }} um</strong></label>
        <input type="range" min="0.3" max="3.0" step="0.02" v-model.number="spacingX" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Spacing Y', '간격 Y') }}: <strong>{{ spacingY.toFixed(2) }} um</strong></label>
        <input type="range" min="0.3" max="3.0" step="0.02" v-model.number="spacingY" class="ctrl-range" />
      </div>
    </div>

    <div v-if="viewMode === 'ray'" class="controls-row">
      <div class="slider-group">
        <label>{{ t('Rays per lens', '렌즈당 광선') }}: <strong>{{ numRays }}</strong></label>
        <input type="range" min="5" max="30" step="1" v-model.number="numRays" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Refractive index', '굴절률') }}: <strong>{{ refIdx.toFixed(2) }}</strong></label>
        <input type="range" min="1.2" max="2.5" step="0.05" v-model.number="refIdx" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Propagation', '전파 거리') }}: <strong>{{ propDist.toFixed(1) }} um</strong></label>
        <input type="range" min="1" max="15" step="0.5" v-model.number="propDist" class="ctrl-range" />
      </div>
    </div>

    <div class="tab-row">
      <button
        v-for="tab in tabs" :key="tab.key"
        :class="['tab-btn', { active: viewMode === tab.key }]"
        @click="viewMode = tab.key"
      >{{ t(tab.en, tab.ko) }}</button>
    </div>

    <div class="svg-wrapper">
      <!-- Top View: Height Map -->
      <svg v-if="viewMode === 'top'" :viewBox="`0 0 ${svgW} ${svgH}`" class="mla-svg">
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />
        <rect
          v-for="(cell, ci) in heatmapCells" :key="ci"
          :x="cell.x" :y="cell.y" :width="cell.w" :height="cell.h"
          :fill="cell.color"
        />
        <!-- Lens boundary circles -->
        <ellipse
          v-for="(lb, li) in lensBoundaries" :key="'lb-' + li"
          :cx="lb.cx" :cy="lb.cy" :rx="lb.rx" :ry="lb.ry"
          fill="none" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="3,2" opacity="0.5"
        />
        <!-- Colorbar -->
        <rect
          v-for="(cb, cbi) in colorbarCells" :key="'cb-' + cbi"
          :x="svgW - 30" :y="cb.y" width="14" :height="cb.h"
          :fill="cb.color"
        />
        <text :x="svgW - 13" :y="pad.top - 2" text-anchor="middle" class="axis-label">{{ h.toFixed(2) }}</text>
        <text :x="svgW - 13" :y="svgH - pad.bottom + 12" text-anchor="middle" class="axis-label">0</text>
        <text :x="svgW - 13" :y="(pad.top + svgH - pad.bottom) / 2 + 3" text-anchor="middle" class="axis-label">um</text>
        <!-- Axes labels -->
        <text :x="(pad.left + svgW - 44) / 2" :y="svgH - 4" text-anchor="middle" class="axis-label">X (um)</text>
        <text :x="12" :y="(pad.top + svgH - pad.bottom) / 2" text-anchor="middle" class="axis-label" transform-origin="12 200" :transform="`rotate(-90, 12, ${(pad.top + svgH - pad.bottom) / 2})`">Y (um)</text>
      </svg>

      <!-- Cross-Section View -->
      <svg v-if="viewMode === 'section'" :viewBox="`0 0 ${svgW} ${svgH}`" class="mla-svg"
        @mousemove="onSectionHover" @mouseleave="sectionHoverIdx = -1">
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />
        <!-- Grid -->
        <line v-for="gy in sectionGridY" :key="'gy-' + gy.v"
          :x1="pad.left" :y1="gy.y" :x2="svgW - pad.right" :y2="gy.y"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gy in sectionGridY" :key="'gyt-' + gy.v"
          :x="pad.left - 4" :y="gy.y + 3" text-anchor="end" class="axis-label">{{ gy.label }}</text>
        <line v-for="gx in sectionGridX" :key="'gx-' + gx.v"
          :x1="gx.x" :y1="pad.top" :x2="gx.x" :y2="svgH - pad.bottom"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gx in sectionGridX" :key="'gxt-' + gx.v"
          :x="gx.x" :y="svgH - pad.bottom + 14" text-anchor="middle" class="axis-label">{{ gx.label }}</text>
        <!-- XZ path -->
        <path :d="xzPath" fill="none" stroke="#3498db" stroke-width="2.5" />
        <!-- YZ path -->
        <path :d="yzPath" fill="none" stroke="#e74c3c" stroke-width="2.5" />
        <!-- Diagonal path -->
        <path :d="diagPath" fill="none" stroke="#27ae60" stroke-width="2.5" stroke-dasharray="6,3" />
        <!-- Legend -->
        <line :x1="svgW - 150" y1="16" :x2="svgW - 130" y2="16" stroke="#3498db" stroke-width="2.5" />
        <text :x="svgW - 126" y="20" class="legend-label">{{ t('XZ (y=0)', 'XZ (y=0)') }}</text>
        <line :x1="svgW - 150" y1="30" :x2="svgW - 130" y2="30" stroke="#e74c3c" stroke-width="2.5" />
        <text :x="svgW - 126" y="34" class="legend-label">{{ t('YZ (x=0)', 'YZ (x=0)') }}</text>
        <line :x1="svgW - 150" y1="44" :x2="svgW - 130" y2="44" stroke="#27ae60" stroke-width="2.5" stroke-dasharray="6,3" />
        <text :x="svgW - 126" y="48" class="legend-label">{{ t('Diagonal', '대각선') }}</text>
        <!-- Hover crosshair -->
        <template v-if="sectionHoverIdx >= 0">
          <line :x1="sectionHoverX" y1="0" :x2="sectionHoverX" :y2="svgH" stroke="var(--vp-c-text-2)" stroke-width="0.5" stroke-dasharray="3,2" />
          <rect :x="sectionHoverX + 6" :y="pad.top" width="90" height="46" rx="4" fill="var(--vp-c-bg-soft)" stroke="var(--vp-c-divider)" />
          <text :x="sectionHoverX + 10" :y="pad.top + 13" class="tooltip-text">r = {{ sectionHoverR }}</text>
          <text :x="sectionHoverX + 10" :y="pad.top + 26" class="tooltip-text" fill="#3498db">XZ: {{ sectionHoverXZ }}</text>
          <text :x="sectionHoverX + 10" :y="pad.top + 39" class="tooltip-text" fill="#e74c3c">YZ: {{ sectionHoverYZ }}</text>
        </template>
        <!-- Axis labels -->
        <text :x="(pad.left + svgW - pad.right) / 2" :y="svgH - 4" text-anchor="middle" class="axis-label">{{ t('Radial distance (um)', '반경 거리 (um)') }}</text>
        <text x="12" :y="(pad.top + svgH - pad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 12, ${(pad.top + svgH - pad.bottom) / 2})`">Z (um)</text>
      </svg>

      <!-- 2D Ray Trace View -->
      <svg v-if="viewMode === 'ray'" :viewBox="`0 0 ${svgW} ${svgH}`" class="mla-svg">
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />
        <!-- Grid -->
        <line v-for="gy in rayGridY" :key="'rgy-' + gy.v"
          :x1="pad.left" :y1="gy.y" :x2="svgW - pad.right" :y2="gy.y"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gy in rayGridY" :key="'rgyt-' + gy.v"
          :x="pad.left - 4" :y="gy.y + 3" text-anchor="end" class="axis-label">{{ gy.label }}</text>
        <line v-for="gx in rayGridX" :key="'rgx-' + gx.v"
          :x1="gx.x" :y1="pad.top" :x2="gx.x" :y2="svgH - pad.bottom"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gx in rayGridX" :key="'rgxt-' + gx.v"
          :x="gx.x" :y="svgH - pad.bottom + 14" text-anchor="middle" class="axis-label">{{ gx.label }}</text>
        <!-- Lens profiles -->
        <path
          v-for="(lp, lpi) in lensProfiles" :key="'lp-' + lpi"
          :d="lp" fill="#dda0dd" fill-opacity="0.4" stroke="#8e44ad" stroke-width="2"
        />
        <!-- Rays -->
        <template v-for="(ray, ri) in rays2D" :key="'ray-' + ri">
          <line
            :x1="ray.x0" :y1="ray.y0" :x2="ray.x1" :y2="ray.y1"
            :stroke="ray.color" stroke-width="1.2" opacity="0.7"
          />
          <circle v-if="ray.hitSurface" :cx="ray.x1" :cy="ray.y1" r="2" :fill="ray.color" opacity="0.8" />
          <line v-if="ray.hitSurface"
            :x1="ray.x1" :y1="ray.y1" :x2="ray.x2" :y2="ray.y2"
            :stroke="ray.color" stroke-width="1.2" opacity="0.7"
          />
        </template>
        <!-- Focal point -->
        <template v-if="focalPoint2D">
          <circle :cx="focalPoint2D.svgX" :cy="focalPoint2D.svgY" r="5" fill="none" stroke="#e74c3c" stroke-width="2" />
          <line :x1="focalPoint2D.svgX - 7" :y1="focalPoint2D.svgY" :x2="focalPoint2D.svgX + 7" :y2="focalPoint2D.svgY" stroke="#e74c3c" stroke-width="1.5" />
          <line :x1="focalPoint2D.svgX" :y1="focalPoint2D.svgY - 7" :x2="focalPoint2D.svgX" :y2="focalPoint2D.svgY + 7" stroke="#e74c3c" stroke-width="1.5" />
        </template>
        <!-- Legend -->
        <line :x1="svgW - 140" y1="16" :x2="svgW - 120" y2="16" stroke="#3498db" stroke-width="1.5" opacity="0.7" />
        <text :x="svgW - 116" y="20" class="legend-label">{{ t('Focused', '집속') }}</text>
        <line :x1="svgW - 140" y1="30" :x2="svgW - 120" y2="30" stroke="#95a5a6" stroke-width="1.5" opacity="0.7" />
        <text :x="svgW - 116" y="34" class="legend-label">{{ t('Missed', '비집속') }}</text>
        <circle :cx="svgW - 130" cy="44" r="4" fill="none" stroke="#e74c3c" stroke-width="1.5" />
        <text :x="svgW - 116" y="48" class="legend-label">{{ t('Focal point', '집속점') }}</text>
        <!-- Axes -->
        <text :x="(pad.left + svgW - pad.right) / 2" :y="svgH - 4" text-anchor="middle" class="axis-label">X (um)</text>
        <text x="12" :y="(pad.top + svgH - pad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 12, ${(pad.top + svgH - pad.bottom) / 2})`">Z (um)</text>
      </svg>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Sag height', '렌즈 새그') }}</span>
        <span class="info-value">{{ sagHeight.toFixed(3) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Focal length (est.)', '초점 거리 (추정)') }}</span>
        <span class="info-value">{{ estFocalLength.toFixed(2) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('f-number', 'f-넘버') }}</span>
        <span class="info-value">f/{{ fNumber.toFixed(2) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Fill factor', '충전률') }}</span>
        <span class="info-value">{{ fillFactor.toFixed(1) }}%</span>
      </div>
    </div>

    <div class="formula-box">
      <strong>{{ t('Superellipse', '초타원') }}:</strong> |x/Rx|<sup>n</sup> + |y/Ry|<sup>n</sup> = 1 &nbsp;|&nbsp;
      <strong>{{ t('Height', '높이') }}:</strong> z(r) = h(1 - r<sup>2</sup>)<sup>1/(2α)</sup> &nbsp;|&nbsp;
      <strong>{{ t('Snell', '스넬') }}:</strong> n<sub>1</sub>sin&theta;<sub>1</sub> = n<sub>2</sub>sin&theta;<sub>2</sub>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const tabs = [
  { key: 'top', en: 'Height Map', ko: '높이맵' },
  { key: 'section', en: 'Cross-Section', ko: '단면' },
  { key: 'ray', en: 'Ray Trace', ko: '광선 추적' },
]

// Controls
const viewMode = ref<'top' | 'section' | 'ray'>('top')
const arrayConfig = ref('2x2')
const Rx = ref(0.8)
const Ry = ref(0.8)
const h = ref(0.5)
const n = ref(2.0)
const alpha = ref(1.0)
const spacingX = ref(1.0)
const spacingY = ref(1.0)
const numRays = ref(15)
const refIdx = ref(1.56)
const propDist = ref(8.0)

// SVG layout
const svgW = 560
const svgH = 400
const pad = { top: 24, right: 44, bottom: 36, left: 48 }

const arrRows = computed(() => parseInt(arrayConfig.value.split('x')[0]))
const arrCols = computed(() => parseInt(arrayConfig.value.split('x')[1]))

// Superellipse distance
function superDist(x: number, y: number, rx: number, ry: number, nn: number): number {
  return Math.pow(Math.pow(Math.abs(x / rx), nn) + Math.pow(Math.abs(y / ry), nn), 1 / nn)
}

// Lens height at normalized distance r
function lensZ(r: number, hh: number, aa: number): number {
  if (r >= 1) return 0
  return hh * Math.pow(1 - r * r, 1 / (2 * aa))
}

// ---- Top View (Height Map) ----
const hmRes = 80

const heatmapCells = computed(() => {
  const rows = arrRows.value
  const cols = arrCols.value
  const sx = arrayConfig.value === '1x1' ? Rx.value * 2.5 : cols * spacingX.value
  const sy = arrayConfig.value === '1x1' ? Ry.value * 2.5 : rows * spacingY.value
  const drawW = svgW - pad.left - pad.right - 30
  const drawH = svgH - pad.top - pad.bottom
  const cellW = drawW / hmRes
  const cellH = drawH / hmRes
  const cells: Array<{ x: number; y: number; w: number; h: number; color: string }> = []
  const hMax = h.value

  for (let i = 0; i < hmRes; i++) {
    for (let j = 0; j < hmRes; j++) {
      const px = -sx / 2 + (i + 0.5) * sx / hmRes
      const py = -sy / 2 + (j + 0.5) * sy / hmRes
      let maxZ = 0

      for (let lr = 0; lr < rows; lr++) {
        for (let lc = 0; lc < cols; lc++) {
          const cx = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value
          const cy = arrayConfig.value === '1x1' ? 0 : (lr - (rows - 1) / 2) * spacingY.value
          const r = superDist(px - cx, py - cy, Rx.value, Ry.value, n.value)
          const z = lensZ(r, h.value, alpha.value)
          if (z > maxZ) maxZ = z
        }
      }

      const ratio = hMax > 0 ? maxZ / hMax : 0
      cells.push({
        x: pad.left + i * cellW,
        y: pad.top + (hmRes - 1 - j) * cellH,
        w: cellW + 0.5,
        h: cellH + 0.5,
        color: viridis(ratio),
      })
    }
  }
  return cells
})

const lensBoundaries = computed(() => {
  const rows = arrRows.value
  const cols = arrCols.value
  const sx = arrayConfig.value === '1x1' ? Rx.value * 2.5 : cols * spacingX.value
  const sy = arrayConfig.value === '1x1' ? Ry.value * 2.5 : rows * spacingY.value
  const drawW = svgW - pad.left - pad.right - 30
  const drawH = svgH - pad.top - pad.bottom

  const bounds: Array<{ cx: number; cy: number; rx: number; ry: number }> = []
  for (let lr = 0; lr < rows; lr++) {
    for (let lc = 0; lc < cols; lc++) {
      const px = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value
      const py = arrayConfig.value === '1x1' ? 0 : (lr - (rows - 1) / 2) * spacingY.value
      bounds.push({
        cx: pad.left + ((px + sx / 2) / sx) * drawW,
        cy: pad.top + ((sy / 2 - py) / sy) * drawH,
        rx: (Rx.value / sx) * drawW,
        ry: (Ry.value / sy) * drawH,
      })
    }
  }
  return bounds
})

const colorbarCells = computed(() => {
  const drawH = svgH - pad.top - pad.bottom
  const numSteps = 40
  const stepH = drawH / numSteps
  const cells: Array<{ y: number; h: number; color: string }> = []
  for (let i = 0; i < numSteps; i++) {
    const ratio = 1 - i / (numSteps - 1)
    cells.push({
      y: pad.top + i * stepH,
      h: stepH + 0.5,
      color: viridis(ratio),
    })
  }
  return cells
})

// ---- Cross-Section View ----
const sectionRes = 200

const sectionRange = computed(() => Math.max(Rx.value, Ry.value) * 1.4)
const sectionDrawW = computed(() => svgW - pad.left - pad.right)
const sectionDrawH = computed(() => svgH - pad.top - pad.bottom)

function secXScale(v: number): number {
  return pad.left + ((v + sectionRange.value) / (2 * sectionRange.value)) * sectionDrawW.value
}
function secYScale(v: number): number {
  const maxZ = h.value * 1.3
  return svgH - pad.bottom - (v / maxZ) * sectionDrawH.value
}

const xzProfile = computed(() => {
  const pts: Array<{ x: number; z: number }> = []
  const range = sectionRange.value
  for (let i = 0; i <= sectionRes; i++) {
    const px = -range + (2 * range * i) / sectionRes
    const r = superDist(px, 0, Rx.value, Ry.value, n.value)
    pts.push({ x: px, z: lensZ(r, h.value, alpha.value) })
  }
  return pts
})

const yzProfile = computed(() => {
  const pts: Array<{ x: number; z: number }> = []
  const range = sectionRange.value
  for (let i = 0; i <= sectionRes; i++) {
    const py = -range + (2 * range * i) / sectionRes
    const r = superDist(0, py, Rx.value, Ry.value, n.value)
    pts.push({ x: py, z: lensZ(r, h.value, alpha.value) })
  }
  return pts
})

const diagProfile = computed(() => {
  const pts: Array<{ x: number; z: number }> = []
  const range = sectionRange.value
  for (let i = 0; i <= sectionRes; i++) {
    const p = -range + (2 * range * i) / sectionRes
    const r = superDist(p, p, Rx.value, Ry.value, n.value)
    pts.push({ x: p, z: lensZ(r, h.value, alpha.value) })
  }
  return pts
})

function buildPath(pts: Array<{ x: number; z: number }>): string {
  if (pts.length === 0) return ''
  let d = `M ${secXScale(pts[0].x).toFixed(1)} ${secYScale(pts[0].z).toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    d += ` L ${secXScale(pts[i].x).toFixed(1)} ${secYScale(pts[i].z).toFixed(1)}`
  }
  return d
}

const xzPath = computed(() => buildPath(xzProfile.value))
const yzPath = computed(() => buildPath(yzProfile.value))
const diagPath = computed(() => buildPath(diagProfile.value))

const sectionGridY = computed(() => {
  const maxZ = h.value * 1.3
  const steps = 5
  const arr: Array<{ v: number; y: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = (maxZ * i) / steps
    arr.push({ v, y: secYScale(v), label: v.toFixed(2) })
  }
  return arr
})

const sectionGridX = computed(() => {
  const range = sectionRange.value
  const steps = 6
  const arr: Array<{ v: number; x: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = -range + (2 * range * i) / steps
    arr.push({ v, x: secXScale(v), label: v.toFixed(2) })
  }
  return arr
})

// Section hover
const sectionHoverIdx = ref(-1)
const sectionHoverX = ref(0)

function onSectionHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mx = (e.clientX - rect.left) * scaleX
  if (mx < pad.left || mx > svgW - pad.right) {
    sectionHoverIdx.value = -1
    return
  }
  sectionHoverIdx.value = 1
  sectionHoverX.value = mx
}

const sectionHoverR = computed(() => {
  if (sectionHoverIdx.value < 0) return ''
  const range = sectionRange.value
  const v = ((sectionHoverX.value - pad.left) / sectionDrawW.value) * 2 * range - range
  return v.toFixed(3)
})

const sectionHoverXZ = computed(() => {
  if (sectionHoverIdx.value < 0) return ''
  const range = sectionRange.value
  const v = ((sectionHoverX.value - pad.left) / sectionDrawW.value) * 2 * range - range
  const r = superDist(v, 0, Rx.value, Ry.value, n.value)
  return lensZ(r, h.value, alpha.value).toFixed(4)
})

const sectionHoverYZ = computed(() => {
  if (sectionHoverIdx.value < 0) return ''
  const range = sectionRange.value
  const v = ((sectionHoverX.value - pad.left) / sectionDrawW.value) * 2 * range - range
  const r = superDist(0, v, Rx.value, Ry.value, n.value)
  return lensZ(r, h.value, alpha.value).toFixed(4)
})

// ---- 2D Ray Trace View (XZ cross-section, y=0) ----

const rayXRange = computed(() => {
  const cols = arrCols.value
  if (arrayConfig.value === '1x1') return Rx.value * 1.8
  return (cols * spacingX.value) / 2 + Rx.value * 0.3
})

const rayZMin = computed(() => -propDist.value * 0.8)
const rayZMax = computed(() => h.value + 1.5)

const rayDrawW = computed(() => svgW - pad.left - pad.right)
const rayDrawH = computed(() => svgH - pad.top - pad.bottom)

function rayXScale(v: number): number {
  return pad.left + ((v + rayXRange.value) / (2 * rayXRange.value)) * rayDrawW.value
}
function rayYScale(z: number): number {
  return svgH - pad.bottom - ((z - rayZMin.value) / (rayZMax.value - rayZMin.value)) * rayDrawH.value
}

const rayGridY = computed(() => {
  const steps = 6
  const arr: Array<{ v: number; y: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = rayZMin.value + ((rayZMax.value - rayZMin.value) * i) / steps
    arr.push({ v, y: rayYScale(v), label: v.toFixed(1) })
  }
  return arr
})

const rayGridX = computed(() => {
  const steps = 6
  const range = rayXRange.value
  const arr: Array<{ v: number; x: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = -range + (2 * range * i) / steps
    arr.push({ v, x: rayXScale(v), label: v.toFixed(1) })
  }
  return arr
})

const lensProfiles = computed(() => {
  const cols = arrCols.value
  const paths: string[] = []
  for (let lc = 0; lc < cols; lc++) {
    const cx = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value
    const pts: string[] = []
    const numPts = 80
    const startX = cx - Rx.value
    const endX = cx + Rx.value
    pts.push(`M ${rayXScale(startX).toFixed(1)} ${rayYScale(0).toFixed(1)}`)
    for (let i = 0; i <= numPts; i++) {
      const px = startX + ((endX - startX) * i) / numPts
      const localX = px - cx
      const r = Math.abs(localX / Rx.value)
      const z = lensZ(r, h.value, alpha.value)
      pts.push(`L ${rayXScale(px).toFixed(1)} ${rayYScale(z).toFixed(1)}`)
    }
    pts.push(`L ${rayXScale(endX).toFixed(1)} ${rayYScale(0).toFixed(1)} Z`)
    paths.push(pts.join(' '))
  }
  return paths
})

interface Ray2D {
  x0: number; y0: number
  x1: number; y1: number
  x2: number; y2: number
  hitSurface: boolean
  color: string
  endPhysZ: number
  endPhysX: number
}

const rays2D = computed((): Ray2D[] => {
  const cols = arrCols.value
  const result: Ray2D[] = []
  const nLens = refIdx.value

  for (let lc = 0; lc < cols; lc++) {
    const cx = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value

    for (let ri = 0; ri < numRays.value; ri++) {
      const localX = ((ri + 0.5) / numRays.value - 0.5) * Rx.value * 1.6
      const startX = cx + localX
      const startZ = rayZMax.value - 0.1

      const r = Math.abs(localX / Rx.value)
      if (r >= 1) {
        result.push({
          x0: rayXScale(startX), y0: rayYScale(startZ),
          x1: rayXScale(startX), y1: rayYScale(-propDist.value),
          x2: rayXScale(startX), y2: rayYScale(-propDist.value),
          hitSurface: false, color: '#95a5a6',
          endPhysZ: -propDist.value, endPhysX: startX,
        })
        continue
      }

      const surfZ = lensZ(r, h.value, alpha.value)

      // Numerical normal (XZ plane, y=0)
      const eps = 0.001
      const rPlus = Math.abs((localX + eps) / Rx.value)
      const zPlus = lensZ(rPlus < 1 ? superDist(localX + eps, 0, Rx.value, Ry.value, n.value) : 1, h.value, alpha.value)
      const dzdx = (zPlus - surfZ) / eps

      const nLen = Math.sqrt(dzdx * dzdx + 1)
      let normX = -dzdx / nLen
      let normZ = 1 / nLen

      // Incident: straight down
      const incDx = 0
      const incDz = -1
      const cosI = -(incDx * normX + incDz * normZ)
      if (cosI < 0) { normX = -normX; normZ = -normZ }
      const cosI2 = -(incDx * normX + incDz * normZ)

      const eta = 1.0 / nLens
      const k = 1 - eta * eta * (1 - cosI2 * cosI2)

      let refDx: number, refDz: number
      if (k < 0) {
        refDx = incDx + 2 * cosI2 * normX
        refDz = incDz + 2 * cosI2 * normZ
      } else {
        const sqrtK = Math.sqrt(k)
        refDx = eta * incDx + (eta * cosI2 - sqrtK) * normX
        refDz = eta * incDz + (eta * cosI2 - sqrtK) * normZ
      }

      const rl = Math.sqrt(refDx * refDx + refDz * refDz)
      refDx /= rl
      refDz /= rl

      const tProp = propDist.value / Math.max(Math.abs(refDz), 0.01)
      const endX = startX + refDx * tProp
      const endZ = surfZ + refDz * tProp
      const focused = Math.abs(endX - cx) < Rx.value * 0.3

      result.push({
        x0: rayXScale(startX), y0: rayYScale(startZ),
        x1: rayXScale(startX), y1: rayYScale(surfZ),
        x2: rayXScale(endX), y2: rayYScale(endZ),
        hitSurface: true,
        color: focused ? '#3498db' : '#95a5a6',
        endPhysZ: endZ, endPhysX: endX,
      })
    }
  }
  return result
})

const focalPoint2D = computed(() => {
  const focused = rays2D.value.filter(r => r.hitSurface && r.color === '#3498db')
  if (focused.length < 2) return null
  let sumX = 0, sumZ = 0
  for (const r of focused) {
    sumX += r.endPhysX
    sumZ += r.endPhysZ
  }
  const fpx = sumX / focused.length
  const fpz = sumZ / focused.length
  return { svgX: rayXScale(fpx), svgY: rayYScale(fpz), x: fpx, z: fpz }
})

// ---- Info Metrics ----
const sagHeight = computed(() => h.value)

const estFocalLength = computed(() => {
  const nLens = refIdx.value
  const avgR = (Rx.value + Ry.value) / 2
  return (avgR * avgR) / (2 * h.value * (nLens - 1))
})

const fNumber = computed(() => {
  const avgR = (Rx.value + Ry.value) / 2
  return estFocalLength.value / (2 * avgR)
})

const fillFactor = computed(() => {
  if (arrayConfig.value === '1x1') return 100
  const lensArea = Math.PI * Rx.value * Ry.value
  const cellArea = spacingX.value * spacingY.value
  return Math.min(100, (lensArea / cellArea) * 100)
})

// ---- Viridis Colormap ----
function viridis(t: number): string {
  const c = Math.max(0, Math.min(1, t))
  // Simplified viridis: dark purple → teal → yellow
  const r = Math.round(255 * Math.min(1, Math.max(0, -0.35 + 2.5 * c * c)))
  const g = Math.round(255 * Math.min(1, Math.max(0, -0.05 + 1.2 * c)))
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.5 + 0.8 * Math.sin(Math.PI * (0.35 + 0.65 * c)))))
  if (c < 0.01) return '#440154'
  if (c > 0.99) return '#fde725'
  return `rgb(${r},${g},${b})`
}
</script>

<style scoped>
.mla-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.mla-container h4 {
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
  margin-bottom: 12px;
}
.slider-group {
  flex: 1;
  min-width: 150px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
  color: var(--vp-c-text-1);
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
.ctrl-select {
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.85em;
  cursor: pointer;
}
.tab-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.tab-btn {
  padding: 6px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  cursor: pointer;
  transition: all 0.2s;
}
.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.tab-btn.active {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
}
.svg-wrapper {
  display: flex;
  justify-content: center;
  margin: 12px 0;
}
.mla-svg {
  width: 100%;
  max-width: 580px;
}
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 500;
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
  flex: 1;
  min-width: 110px;
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
.formula-box {
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 0.82em;
  color: var(--vp-c-text-2);
  font-family: 'Times New Roman', serif;
  line-height: 1.6;
}
@media (max-width: 640px) {
  .controls-row { flex-direction: column; gap: 8px; }
  .info-row { flex-direction: column; }
  .tab-row { justify-content: center; }
}
</style>
