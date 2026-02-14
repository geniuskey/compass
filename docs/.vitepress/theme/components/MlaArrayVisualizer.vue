<template>
  <div class="mla-container">
    <h4>{{ t('Micro Lens Array Visualizer', '마이크로 렌즈 어레이 시각화') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize superellipse microlens array geometry with contour maps, equal-aspect cross-sections, interactive 3D surface, and 2D ray tracing.',
        '등고선, 등비율 단면, 인터랙티브 3D 표면, 2D 광선 추적으로 초타원 마이크로렌즈 어레이를 시각화합니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>{{ t('Array', '배열') }}:
          <select v-model="arrayConfig" class="ctrl-select">
            <option value="1x1">1x1</option>
            <option value="2x1">2x1</option>
            <option value="1x2">1x2</option>
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

    <div v-if="viewMode === '3d'" class="controls-row">
      <div class="slider-group">
        <label>{{ t('Resolution', '해상도') }}: <strong>{{ meshRes3d }}</strong></label>
        <input type="range" min="30" max="120" step="10" v-model.number="meshRes3d" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Colormap', '색상맵') }}:
          <select v-model="colormap3d" class="ctrl-select">
            <option value="Viridis">Viridis</option>
            <option value="Plasma">Plasma</option>
            <option value="Inferno">Inferno</option>
            <option value="Jet">Jet</option>
            <option value="Hot">Hot</option>
          </select>
        </label>
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
      <!-- Contour View -->
      <svg v-if="viewMode === 'contour'" :viewBox="`0 0 ${svgW} ${svgH}`" class="mla-svg">
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />
        <!-- Contour lines for each lens -->
        <path
          v-for="(cp, ci) in contourPaths" :key="'cp-' + ci"
          :d="cp.d" fill="none" :stroke="cp.color" :stroke-width="cp.width"
        />
        <!-- Lens boundary -->
        <path
          v-for="(bp, bi) in boundaryPaths" :key="'bp-' + bi"
          :d="bp" fill="none" stroke="var(--vp-c-text-2)" stroke-width="1" stroke-dasharray="4,2" opacity="0.5"
        />
        <!-- Axis ticks & labels -->
        <line v-for="gx in contourGridX" :key="'cgx-' + gx.v"
          :x1="gx.x" :y1="svgH - cPad.bottom" :x2="gx.x" :y2="svgH - cPad.bottom + 4"
          stroke="var(--vp-c-text-2)" stroke-width="0.5" />
        <text v-for="gx in contourGridX" :key="'cgxt-' + gx.v"
          :x="gx.x" :y="svgH - cPad.bottom + 15" text-anchor="middle" class="axis-label">{{ gx.label }}</text>
        <line v-for="gy in contourGridY" :key="'cgy-' + gy.v"
          :x1="cPad.left" :y1="gy.y" :x2="cPad.left - 4" :y2="gy.y"
          stroke="var(--vp-c-text-2)" stroke-width="0.5" />
        <text v-for="gy in contourGridY" :key="'cgyt-' + gy.v"
          :x="cPad.left - 6" :y="gy.y + 3" text-anchor="end" class="axis-label">{{ gy.label }}</text>
        <!-- Legend -->
        <line v-for="(lv, li) in contourLegend" :key="'cl-' + li"
          :x1="svgW - 70" :y1="cPad.top + 8 + li * 14" :x2="svgW - 52" :y2="cPad.top + 8 + li * 14"
          :stroke="lv.color" stroke-width="1.5" />
        <text v-for="(lv, li) in contourLegend" :key="'clt-' + li"
          :x="svgW - 48" :y="cPad.top + 12 + li * 14" class="legend-label">{{ lv.label }}</text>
        <text :x="(cPad.left + svgW - cPad.right) / 2" :y="svgH - 2" text-anchor="middle" class="axis-label">X (um)</text>
        <text x="10" :y="(cPad.top + svgH - cPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 10, ${(cPad.top + svgH - cPad.bottom) / 2})`">Y (um)</text>
      </svg>

      <!-- Cross-Section View (equal aspect) -->
      <svg v-if="viewMode === 'section'" :viewBox="`0 0 ${secSvgW} ${secSvgH}`" class="mla-svg"
        @mousemove="onSectionHover" @mouseleave="sectionHoverIdx = -1">
        <rect x="0" y="0" :width="secSvgW" :height="secSvgH" fill="var(--vp-c-bg)" />
        <!-- Grid -->
        <line v-for="gy in sectionGridY" :key="'gy-' + gy.v"
          :x1="secPad.left" :y1="gy.y" :x2="secSvgW - secPad.right" :y2="gy.y"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gy in sectionGridY" :key="'gyt-' + gy.v"
          :x="secPad.left - 4" :y="gy.y + 3" text-anchor="end" class="axis-label">{{ gy.label }}</text>
        <line v-for="gx in sectionGridX" :key="'gx-' + gx.v"
          :x1="gx.x" :y1="secPad.top" :x2="gx.x" :y2="secSvgH - secPad.bottom"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gx in sectionGridX" :key="'gxt-' + gx.v"
          :x="gx.x" :y="secSvgH - secPad.bottom + 14" text-anchor="middle" class="axis-label">{{ gx.label }}</text>
        <!-- Filled area under curves -->
        <path :d="xzFillPath" fill="#3498db" fill-opacity="0.08" />
        <path :d="yzFillPath" fill="#e74c3c" fill-opacity="0.08" />
        <!-- XZ path -->
        <path :d="xzPath" fill="none" stroke="#3498db" stroke-width="2.5" />
        <!-- YZ path -->
        <path :d="yzPath" fill="none" stroke="#e74c3c" stroke-width="2.5" />
        <!-- Diagonal path -->
        <path :d="diagPath" fill="none" stroke="#27ae60" stroke-width="2" stroke-dasharray="6,3" />
        <!-- Legend -->
        <line :x1="secSvgW - 150" y1="16" :x2="secSvgW - 130" y2="16" stroke="#3498db" stroke-width="2.5" />
        <text :x="secSvgW - 126" y="20" class="legend-label">{{ t('XZ (y=0)', 'XZ (y=0)') }}</text>
        <line :x1="secSvgW - 150" y1="30" :x2="secSvgW - 130" y2="30" stroke="#e74c3c" stroke-width="2.5" />
        <text :x="secSvgW - 126" y="34" class="legend-label">{{ t('YZ (x=0)', 'YZ (x=0)') }}</text>
        <line :x1="secSvgW - 150" y1="44" :x2="secSvgW - 130" y2="44" stroke="#27ae60" stroke-width="2" stroke-dasharray="6,3" />
        <text :x="secSvgW - 126" y="48" class="legend-label">{{ t('Diagonal', '대각선') }}</text>
        <!-- Hover -->
        <template v-if="sectionHoverIdx >= 0">
          <line :x1="sectionHoverSvgX" :y1="secPad.top" :x2="sectionHoverSvgX" :y2="secSvgH - secPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="0.5" stroke-dasharray="3,2" />
          <rect :x="Math.min(sectionHoverSvgX + 6, secSvgW - 106)" :y="secPad.top" width="100" height="46" rx="4" fill="var(--vp-c-bg-soft)" stroke="var(--vp-c-divider)" />
          <text :x="Math.min(sectionHoverSvgX + 10, secSvgW - 102)" :y="secPad.top + 13" class="tooltip-text">r = {{ sectionHoverR }}</text>
          <text :x="Math.min(sectionHoverSvgX + 10, secSvgW - 102)" :y="secPad.top + 26" class="tooltip-text" fill="#3498db">XZ: {{ sectionHoverXZ }}</text>
          <text :x="Math.min(sectionHoverSvgX + 10, secSvgW - 102)" :y="secPad.top + 39" class="tooltip-text" fill="#e74c3c">YZ: {{ sectionHoverYZ }}</text>
        </template>
        <text :x="(secPad.left + secSvgW - secPad.right) / 2" :y="secSvgH - 2" text-anchor="middle" class="axis-label">{{ t('Distance (um)', '거리 (um)') }}</text>
        <text x="10" :y="(secPad.top + secSvgH - secPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 10, ${(secPad.top + secSvgH - secPad.bottom) / 2})`">Z (um)</text>
      </svg>

      <!-- 3D View (Plotly) -->
      <div v-show="viewMode === '3d'" ref="plotly3dDiv" class="plotly-wrapper"></div>
      <p v-if="viewMode === '3d' && plotlyLoading" class="loading-text">{{ t('Loading 3D engine...', '3D 엔진 로딩 중...') }}</p>
      <p v-if="viewMode === '3d' && plotlyFailed" class="loading-text" style="color: var(--vp-c-danger-1)">{{ t('Failed to load 3D library. Check network or ad-blocker.', '3D 라이브러리 로드 실패. 네트워크 또는 광고 차단기를 확인하세요.') }}</p>

      <!-- 2D Ray Trace View -->
      <svg v-if="viewMode === 'ray'" :viewBox="`0 0 ${svgW} ${svgH}`" class="mla-svg">
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />
        <line v-for="gy in rayGridY" :key="'rgy-' + gy.v"
          :x1="rPad.left" :y1="gy.y" :x2="svgW - rPad.right" :y2="gy.y"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gy in rayGridY" :key="'rgyt-' + gy.v"
          :x="rPad.left - 4" :y="gy.y + 3" text-anchor="end" class="axis-label">{{ gy.label }}</text>
        <line v-for="gx in rayGridX" :key="'rgx-' + gx.v"
          :x1="gx.x" :y1="rPad.top" :x2="gx.x" :y2="svgH - rPad.bottom"
          stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text v-for="gx in rayGridX" :key="'rgxt-' + gx.v"
          :x="gx.x" :y="svgH - rPad.bottom + 14" text-anchor="middle" class="axis-label">{{ gx.label }}</text>
        <!-- Lens profiles -->
        <path v-for="(lp, lpi) in lensProfiles" :key="'lp-' + lpi"
          :d="lp" fill="#dda0dd" fill-opacity="0.4" stroke="#8e44ad" stroke-width="2" />
        <!-- Rays -->
        <template v-for="(ray, ri) in rays2D" :key="'ray-' + ri">
          <line :x1="ray.x0" :y1="ray.y0" :x2="ray.x1" :y2="ray.y1"
            :stroke="ray.color" stroke-width="1.2" opacity="0.7" />
          <circle v-if="ray.hitSurface" :cx="ray.x1" :cy="ray.y1" r="2" :fill="ray.color" opacity="0.8" />
          <line v-if="ray.hitSurface" :x1="ray.x1" :y1="ray.y1" :x2="ray.x2" :y2="ray.y2"
            :stroke="ray.color" stroke-width="1.2" opacity="0.7" />
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
        <text :x="(rPad.left + svgW - rPad.right) / 2" :y="svgH - 2" text-anchor="middle" class="axis-label">X (um)</text>
        <text x="10" :y="(rPad.top + svgH - rPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 10, ${(rPad.top + svgH - rPad.bottom) / 2})`">Z (um)</text>
      </svg>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Sag height', '렌즈 새그') }}</span>
        <span class="info-value">{{ h.toFixed(3) }} um</span>
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
import { ref, computed, watch, onMounted, nextTick } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const tabs = [
  { key: 'contour', en: 'Contour', ko: '등고선' },
  { key: 'section', en: 'Cross-Section', ko: '단면' },
  { key: '3d', en: '3D Surface', ko: '3D 표면' },
  { key: 'ray', en: 'Ray Trace', ko: '광선 추적' },
]

// Controls
const viewMode = ref<'contour' | 'section' | '3d' | 'ray'>('contour')
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
const meshRes3d = ref(60)
const colormap3d = ref('Viridis')

const svgW = 560
const svgH = 400

const arrRows = computed(() => parseInt(arrayConfig.value.split('x')[0]))
const arrCols = computed(() => parseInt(arrayConfig.value.split('x')[1]))

function superDist(x: number, y: number, rx: number, ry: number, nn: number): number {
  return Math.pow(Math.pow(Math.abs(x / rx), nn) + Math.pow(Math.abs(y / ry), nn), 1 / nn)
}

function lensZ(r: number, hh: number, aa: number): number {
  if (r >= 1) return 0
  return hh * Math.pow(1 - r * r, 1 / (2 * aa))
}

function arrayZ(px: number, py: number): number {
  const rows = arrRows.value, cols = arrCols.value
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
  return maxZ
}

// ===== Contour View =====
const cPad = { top: 24, right: 20, bottom: 36, left: 48 }

const contourExtent = computed(() => {
  const rows = arrRows.value, cols = arrCols.value
  if (arrayConfig.value === '1x1') {
    const ext = Math.max(Rx.value, Ry.value) * 1.4
    return { xMin: -ext, xMax: ext, yMin: -ext, yMax: ext }
  }
  const hw = (cols * spacingX.value) / 2 + Rx.value * 0.2
  const hh = (rows * spacingY.value) / 2 + Ry.value * 0.2
  return { xMin: -hw, xMax: hw, yMin: -hh, yMax: hh }
})

const contourDrawW = computed(() => svgW - cPad.left - cPad.right)
const contourDrawH = computed(() => svgH - cPad.top - cPad.bottom)

// Use equal aspect: compute usable area
const contourScale = computed(() => {
  const ext = contourExtent.value
  const physW = ext.xMax - ext.xMin
  const physH = ext.yMax - ext.yMin
  const scaleX = contourDrawW.value / physW
  const scaleY = contourDrawH.value / physH
  return Math.min(scaleX, scaleY)
})

function cX(v: number): number {
  const ext = contourExtent.value
  const physW = ext.xMax - ext.xMin
  const usedW = physW * contourScale.value
  const offset = (contourDrawW.value - usedW) / 2
  return cPad.left + offset + (v - ext.xMin) * contourScale.value
}
function cY(v: number): number {
  const ext = contourExtent.value
  const physH = ext.yMax - ext.yMin
  const usedH = physH * contourScale.value
  const offset = (contourDrawH.value - usedH) / 2
  return cPad.top + offset + (ext.yMax - v) * contourScale.value
}

const contourLevels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

function viridisColor(t: number): string {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.01) return '#440154'
  if (c > 0.99) return '#fde725'
  const r = Math.round(255 * Math.min(1, Math.max(0, -0.35 + 2.5 * c * c)))
  const g = Math.round(255 * Math.min(1, Math.max(0, -0.05 + 1.2 * c)))
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.5 + 0.8 * Math.sin(Math.PI * (0.35 + 0.65 * c)))))
  return `rgb(${r},${g},${b})`
}

// Analytical contour: at height z0, the contour radius is r_c = sqrt(1-(z0/h)^(2*alpha))
// The contour is a superellipse with radii Rx*r_c, Ry*r_c
function superellipsePathSvg(cx: number, cy: number, rx: number, ry: number, nn: number, numPts: number): string {
  const pts: string[] = []
  for (let i = 0; i <= numPts; i++) {
    const t = (2 * Math.PI * i) / numPts
    const cosT = Math.cos(t)
    const sinT = Math.sin(t)
    const x = cx + rx * Math.sign(cosT) * Math.pow(Math.abs(cosT), 2 / nn)
    const y = cy + ry * Math.sign(sinT) * Math.pow(Math.abs(sinT), 2 / nn)
    pts.push(`${i === 0 ? 'M' : 'L'} ${cX(x).toFixed(1)} ${cY(y).toFixed(1)}`)
  }
  return pts.join(' ') + ' Z'
}

const contourPaths = computed(() => {
  const rows = arrRows.value, cols = arrCols.value
  const paths: Array<{ d: string; color: string; width: number }> = []

  for (let lr = 0; lr < rows; lr++) {
    for (let lc = 0; lc < cols; lc++) {
      const lcx = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value
      const lcy = arrayConfig.value === '1x1' ? 0 : (lr - (rows - 1) / 2) * spacingY.value

      for (const level of contourLevels) {
        const z0 = level * h.value
        const rr = Math.pow(z0 / h.value, 2 * alpha.value)
        if (rr >= 1) continue
        const rc = Math.sqrt(1 - rr)
        const rxC = Rx.value * rc
        const ryC = Ry.value * rc
        if (rxC < 0.005 || ryC < 0.005) continue

        paths.push({
          d: superellipsePathSvg(lcx, lcy, rxC, ryC, n.value, 80),
          color: viridisColor(level),
          width: level === 0.5 ? 2 : 1.2,
        })
      }
    }
  }
  return paths
})

const boundaryPaths = computed(() => {
  const rows = arrRows.value, cols = arrCols.value
  const paths: string[] = []
  for (let lr = 0; lr < rows; lr++) {
    for (let lc = 0; lc < cols; lc++) {
      const lcx = arrayConfig.value === '1x1' ? 0 : (lc - (cols - 1) / 2) * spacingX.value
      const lcy = arrayConfig.value === '1x1' ? 0 : (lr - (rows - 1) / 2) * spacingY.value
      paths.push(superellipsePathSvg(lcx, lcy, Rx.value, Ry.value, n.value, 80))
    }
  }
  return paths
})

const contourGridX = computed(() => {
  const ext = contourExtent.value
  const range = ext.xMax - ext.xMin
  const steps = Math.min(8, Math.max(4, Math.round(range / 0.5)))
  const arr: Array<{ v: number; x: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = ext.xMin + (range * i) / steps
    const x = cX(v)
    if (x >= cPad.left - 5 && x <= svgW - cPad.right + 5)
      arr.push({ v, x, label: v.toFixed(1) })
  }
  return arr
})

const contourGridY = computed(() => {
  const ext = contourExtent.value
  const range = ext.yMax - ext.yMin
  const steps = Math.min(8, Math.max(4, Math.round(range / 0.5)))
  const arr: Array<{ v: number; y: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = ext.yMin + (range * i) / steps
    const y = cY(v)
    if (y >= cPad.top - 5 && y <= svgH - cPad.bottom + 5)
      arr.push({ v, y, label: v.toFixed(1) })
  }
  return arr
})

const contourLegend = computed(() =>
  contourLevels.filter((_, i) => i % 2 === 0).map(lv => ({
    color: viridisColor(lv),
    label: `${(lv * 100).toFixed(0)}%`,
  }))
)

// ===== Cross-Section View (equal aspect ratio) =====
const secPad = { top: 24, right: 20, bottom: 36, left: 48 }
const sectionRange = computed(() => Math.max(Rx.value, Ry.value) * 1.4)
const sectionRes = 200

// Equal aspect ratio: compute SVG dimensions dynamically
const secPhysW = computed(() => 2 * sectionRange.value)
const secPhysH = computed(() => h.value * 1.3)
const secAspect = computed(() => secPhysH.value / secPhysW.value)

const secSvgW = svgW
const secSvgH = computed(() => {
  const drawW = secSvgW - secPad.left - secPad.right
  const drawH = drawW * secAspect.value
  const total = drawH + secPad.top + secPad.bottom
  return Math.max(160, Math.min(500, total))
})

const secDrawW = computed(() => secSvgW - secPad.left - secPad.right)
const secDrawH = computed(() => secSvgH.value - secPad.top - secPad.bottom)

// Uniform scale (pixels per um)
const secScale = computed(() => {
  const scaleX = secDrawW.value / secPhysW.value
  const scaleY = secDrawH.value / secPhysH.value
  return Math.min(scaleX, scaleY)
})

function secXScale(v: number): number {
  const usedW = secPhysW.value * secScale.value
  const offset = (secDrawW.value - usedW) / 2
  return secPad.left + offset + (v + sectionRange.value) * secScale.value
}
function secYScale(v: number): number {
  const usedH = secPhysH.value * secScale.value
  const offset = (secDrawH.value - usedH) / 2
  return secSvgH.value - secPad.bottom - offset - v * secScale.value
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

function buildLinePath(pts: Array<{ x: number; z: number }>): string {
  if (!pts.length) return ''
  let d = `M ${secXScale(pts[0].x).toFixed(1)} ${secYScale(pts[0].z).toFixed(1)}`
  for (let i = 1; i < pts.length; i++) {
    d += ` L ${secXScale(pts[i].x).toFixed(1)} ${secYScale(pts[i].z).toFixed(1)}`
  }
  return d
}

function buildFillPath(pts: Array<{ x: number; z: number }>): string {
  if (!pts.length) return ''
  let d = `M ${secXScale(pts[0].x).toFixed(1)} ${secYScale(0).toFixed(1)}`
  for (const pt of pts) {
    d += ` L ${secXScale(pt.x).toFixed(1)} ${secYScale(pt.z).toFixed(1)}`
  }
  d += ` L ${secXScale(pts[pts.length - 1].x).toFixed(1)} ${secYScale(0).toFixed(1)} Z`
  return d
}

const xzPath = computed(() => buildLinePath(xzProfile.value))
const yzPath = computed(() => buildLinePath(yzProfile.value))
const diagPath = computed(() => buildLinePath(diagProfile.value))
const xzFillPath = computed(() => buildFillPath(xzProfile.value))
const yzFillPath = computed(() => buildFillPath(yzProfile.value))

const sectionGridY = computed(() => {
  const maxZ = secPhysH.value
  const steps = Math.max(3, Math.round(maxZ / 0.1))
  const arr: Array<{ v: number; y: number; label: string }> = []
  for (let i = 0; i <= Math.min(steps, 8); i++) {
    const v = (maxZ * i) / Math.min(steps, 8)
    arr.push({ v, y: secYScale(v), label: v.toFixed(2) })
  }
  return arr
})

const sectionGridX = computed(() => {
  const range = sectionRange.value
  const steps = Math.min(8, Math.max(4, Math.round(2 * range / 0.3)))
  const arr: Array<{ v: number; x: number; label: string }> = []
  for (let i = 0; i <= steps; i++) {
    const v = -range + (2 * range * i) / steps
    arr.push({ v, x: secXScale(v), label: v.toFixed(2) })
  }
  return arr
})

const sectionHoverIdx = ref(-1)
const sectionHoverSvgX = ref(0)

function onSectionHover(e: MouseEvent) {
  const svg = e.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = secSvgW / rect.width
  const mx = (e.clientX - rect.left) * scaleX
  if (mx < secPad.left || mx > secSvgW - secPad.right) { sectionHoverIdx.value = -1; return }
  sectionHoverIdx.value = 1
  sectionHoverSvgX.value = mx
}

function hoverPhysX(): number {
  const usedW = secPhysW.value * secScale.value
  const offset = (secDrawW.value - usedW) / 2
  return (sectionHoverSvgX.value - secPad.left - offset) / secScale.value - sectionRange.value
}

const sectionHoverR = computed(() => sectionHoverIdx.value < 0 ? '' : hoverPhysX().toFixed(3))
const sectionHoverXZ = computed(() => {
  if (sectionHoverIdx.value < 0) return ''
  const v = hoverPhysX()
  return lensZ(superDist(v, 0, Rx.value, Ry.value, n.value), h.value, alpha.value).toFixed(4)
})
const sectionHoverYZ = computed(() => {
  if (sectionHoverIdx.value < 0) return ''
  const v = hoverPhysX()
  return lensZ(superDist(0, v, Rx.value, Ry.value, n.value), h.value, alpha.value).toFixed(4)
})

// ===== 3D Plotly View =====
const plotly3dDiv = ref<HTMLElement | null>(null)
const plotlyLoading = ref(false)
const plotlyFailed = ref(false)
let plotlyLib: any = null

const PLOTLY_CDNS = [
  'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.32.0/plotly.min.js',
  'https://cdn.plot.ly/plotly-2.32.0.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js',
]

function tryLoadScript(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    const script = document.createElement('script')
    script.src = url
    script.onload = () => resolve(true)
    script.onerror = () => { script.remove(); resolve(false) }
    document.head.appendChild(script)
  })
}

async function loadPlotly(): Promise<any> {
  if (typeof window === 'undefined') return null
  if ((window as any).Plotly) return (window as any).Plotly
  if (plotlyLib) return plotlyLib

  plotlyLoading.value = true
  for (const url of PLOTLY_CDNS) {
    const ok = await tryLoadScript(url)
    if (ok && (window as any).Plotly) {
      plotlyLib = (window as any).Plotly
      plotlyLoading.value = false
      return plotlyLib
    }
  }
  plotlyLoading.value = false
  plotlyFailed.value = true
  return null
}

async function render3D() {
  const div = plotly3dDiv.value
  if (!div) return

  const Plotly = await loadPlotly()
  if (!Plotly) return

  const rows = arrRows.value, cols = arrCols.value
  const res = meshRes3d.value
  const ext = (() => {
    if (arrayConfig.value === '1x1') {
      const e = Math.max(Rx.value, Ry.value) * 1.4
      return { xMin: -e, xMax: e, yMin: -e, yMax: e }
    }
    const hw = (cols * spacingX.value) / 2 + Rx.value * 0.2
    const hh = (rows * spacingY.value) / 2 + Ry.value * 0.2
    return { xMin: -hw, xMax: hw, yMin: -hh, yMax: hh }
  })()

  const physW = ext.xMax - ext.xMin
  const physH = ext.yMax - ext.yMin

  const xArr: number[] = []
  const yArr: number[] = []
  const zArr: number[][] = []

  for (let i = 0; i <= res; i++) {
    const px = ext.xMin + (physW * i) / res
    xArr.push(px)
  }
  for (let j = 0; j <= res; j++) {
    const py = ext.yMin + (physH * j) / res
    yArr.push(py)
  }
  for (let i = 0; i <= res; i++) {
    const row: number[] = []
    for (let j = 0; j <= res; j++) {
      row.push(arrayZ(xArr[i], yArr[j]))
    }
    zArr.push(row)
  }

  const trace = {
    x: xArr,
    y: yArr,
    z: zArr,
    type: 'surface',
    colorscale: colormap3d.value,
    showscale: true,
    colorbar: { title: 'Z (um)', thickness: 15, len: 0.7 },
  }

  const maxZ = h.value * 2
  const layout = {
    margin: { l: 0, r: 0, t: 30, b: 0 },
    scene: {
      xaxis: { title: 'X (um)' },
      yaxis: { title: 'Y (um)' },
      zaxis: { title: 'Z (um)', range: [0, maxZ] },
      aspectmode: 'cube' as const,
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    height: 480,
  }

  Plotly.react(div, [trace], layout, { responsive: true })
}

let render3dTimer: ReturnType<typeof setTimeout> | null = null

function debouncedRender3D() {
  if (render3dTimer) clearTimeout(render3dTimer)
  render3dTimer = setTimeout(render3D, 80)
}

watch(
  [arrayConfig, Rx, Ry, h, n, alpha, spacingX, spacingY, meshRes3d, colormap3d],
  () => { if (viewMode.value === '3d') debouncedRender3D() },
)

watch(viewMode, (nv) => { if (nv === '3d') nextTick(render3D) })

// ===== 2D Ray Trace View =====
const rPad = { top: 24, right: 20, bottom: 36, left: 48 }

const rayXRange = computed(() => {
  const cols = arrCols.value
  if (arrayConfig.value === '1x1') return Rx.value * 1.8
  return (cols * spacingX.value) / 2 + Rx.value * 0.3
})

const rayZMin = computed(() => -propDist.value * 0.8)
const rayZMax = computed(() => h.value + 1.5)
const rayDrawW = computed(() => svgW - rPad.left - rPad.right)
const rayDrawH = computed(() => svgH - rPad.top - rPad.bottom)

function rayXScale(v: number): number {
  return rPad.left + ((v + rayXRange.value) / (2 * rayXRange.value)) * rayDrawW.value
}
function rayYScale(z: number): number {
  return svgH - rPad.bottom - ((z - rayZMin.value) / (rayZMax.value - rayZMin.value)) * rayDrawH.value
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
  x0: number; y0: number; x1: number; y1: number; x2: number; y2: number
  hitSurface: boolean; color: string; endPhysZ: number; endPhysX: number
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
          hitSurface: false, color: '#95a5a6', endPhysZ: -propDist.value, endPhysX: startX,
        })
        continue
      }

      const surfZ = lensZ(r, h.value, alpha.value)
      const eps = 0.001
      const rPlus = superDist(localX + eps, 0, Rx.value, Ry.value, n.value)
      const zPlus = lensZ(rPlus < 1 ? rPlus : 1, h.value, alpha.value)
      const dzdx = (zPlus - surfZ) / eps

      const nLen = Math.sqrt(dzdx * dzdx + 1)
      let normX = -dzdx / nLen, normZ = 1 / nLen
      const cosI = normZ // incident is (0,-1), dot with normal
      if (cosI < 0) { normX = -normX; normZ = -normZ }
      const cosI2 = Math.abs(normZ)

      const eta = 1.0 / nLens
      const k = 1 - eta * eta * (1 - cosI2 * cosI2)

      let refDx: number, refDz: number
      if (k < 0) {
        refDx = 2 * cosI2 * normX; refDz = -1 + 2 * cosI2 * normZ
      } else {
        const sqrtK = Math.sqrt(k)
        refDx = (eta * cosI2 - sqrtK) * normX
        refDz = eta * (-1) + (eta * cosI2 - sqrtK) * normZ
      }
      const rl = Math.sqrt(refDx * refDx + refDz * refDz)
      refDx /= rl; refDz /= rl

      const tProp = propDist.value / Math.max(Math.abs(refDz), 0.01)
      const endX = startX + refDx * tProp
      const endZ = surfZ + refDz * tProp
      const focused = Math.abs(endX - cx) < Rx.value * 0.3

      result.push({
        x0: rayXScale(startX), y0: rayYScale(startZ),
        x1: rayXScale(startX), y1: rayYScale(surfZ),
        x2: rayXScale(endX), y2: rayYScale(endZ),
        hitSurface: true, color: focused ? '#3498db' : '#95a5a6',
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
  for (const r of focused) { sumX += r.endPhysX; sumZ += r.endPhysZ }
  const fpx = sumX / focused.length, fpz = sumZ / focused.length
  return { svgX: rayXScale(fpx), svgY: rayYScale(fpz), x: fpx, z: fpz }
})

// ===== Info Metrics =====
const estFocalLength = computed(() => {
  const avgR = (Rx.value + Ry.value) / 2
  return (avgR * avgR) / (2 * h.value * (refIdx.value - 1))
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
.plotly-wrapper {
  width: 100%;
  max-width: 620px;
  margin: 0 auto;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  overflow: hidden;
}
.loading-text {
  text-align: center;
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  margin: 8px 0 0;
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
