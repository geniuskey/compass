<template>
  <div class="yee-container">
    <h4>{{ t('Interactive Yee Cell Viewer', '대화형 Yee 셀 뷰어') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize the staggered Yee grid cell used in FDTD simulations. E-field components sit on cell edges, while H-field components sit on face centers.',
        'FDTD 시뮬레이션에서 사용되는 엇갈린 Yee 격자 셀을 시각화합니다. 전기장 성분은 셀 모서리에, 자기장 성분은 면 중심에 위치합니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="toggle-group">
        <label class="toggle-label">
          <input type="checkbox" v-model="showE" /> {{ t('Show E-fields', '전기장 표시') }}
        </label>
        <label class="toggle-label">
          <input type="checkbox" v-model="showH" /> {{ t('Show H-fields', '자기장 표시') }}
        </label>
      </div>
      <div class="slider-group">
        <label>{{ t('Grid spacing', '격자 간격') }} dx: <strong>{{ dx }} nm</strong></label>
        <input type="range" min="10" max="100" step="5" v-model.number="dx" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Cells per wavelength (550 nm):', '파장당 셀 수 (550 nm):') }}</span>
        <span class="info-value" :class="{ warn: cellsPerWl < 10 }">{{ cellsPerWl.toFixed(1) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Courant number S:', '쿠랑 수 S:') }}</span>
        <span class="info-value" :class="{ warn: courant >= courantLimit }">{{ courant.toFixed(4) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Stability limit (1/sqrt(3)):', '안정성 한계 (1/sqrt(3)):') }}</span>
        <span class="info-value">{{ courantLimit.toFixed(4) }}</span>
      </div>
      <div class="info-card" v-if="cellsPerWl < 10">
        <span class="info-value warn">{{ t('Warning: Need 10+ cells/wavelength for accuracy', '경고: 정확도를 위해 파장당 10개 이상의 셀이 필요합니다') }}</span>
      </div>
    </div>

    <div class="content-layout">
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${W} ${H}`" class="yee-svg">
          <!-- Cube wireframe (back edges) -->
          <line :x1="iso(0,0,0).x" :y1="iso(0,0,0).y" :x2="iso(S,0,0).x" :y2="iso(S,0,0).y" stroke="var(--vp-c-divider)" stroke-width="1" stroke-dasharray="4,3" />
          <line :x1="iso(0,0,0).x" :y1="iso(0,0,0).y" :x2="iso(0,S,0).x" :y2="iso(0,S,0).y" stroke="var(--vp-c-divider)" stroke-width="1" stroke-dasharray="4,3" />
          <line :x1="iso(0,0,0).x" :y1="iso(0,0,0).y" :x2="iso(0,0,S).x" :y2="iso(0,0,S).y" stroke="var(--vp-c-divider)" stroke-width="1" stroke-dasharray="4,3" />

          <!-- Cube wireframe (front edges) -->
          <line :x1="iso(S,0,0).x" :y1="iso(S,0,0).y" :x2="iso(S,S,0).x" :y2="iso(S,S,0).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(S,0,0).x" :y1="iso(S,0,0).y" :x2="iso(S,0,S).x" :y2="iso(S,0,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(0,S,0).x" :y1="iso(0,S,0).y" :x2="iso(S,S,0).x" :y2="iso(S,S,0).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(0,S,0).x" :y1="iso(0,S,0).y" :x2="iso(0,S,S).x" :y2="iso(0,S,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(0,0,S).x" :y1="iso(0,0,S).y" :x2="iso(S,0,S).x" :y2="iso(S,0,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(0,0,S).x" :y1="iso(0,0,S).y" :x2="iso(0,S,S).x" :y2="iso(0,S,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(S,S,0).x" :y1="iso(S,S,0).y" :x2="iso(S,S,S).x" :y2="iso(S,S,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(S,0,S).x" :y1="iso(S,0,S).y" :x2="iso(S,S,S).x" :y2="iso(S,S,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />
          <line :x1="iso(0,S,S).x" :y1="iso(0,S,S).y" :x2="iso(S,S,S).x" :y2="iso(S,S,S).y" stroke="var(--vp-c-text-3)" stroke-width="1.2" />

          <!-- Axis labels -->
          <text :x="iso(S+15,0,0).x" :y="iso(S+15,0,0).y" class="axis-label-3d">x</text>
          <text :x="iso(0,S+15,0).x" :y="iso(0,S+15,0).y" class="axis-label-3d">y</text>
          <text :x="iso(0,0,S+15).x" :y="iso(0,0,S+15).y" class="axis-label-3d">z</text>

          <!-- E-field components (on edges) -->
          <template v-if="showE">
            <!-- Ex (red) - edges parallel to x -->
            <line v-for="(edge, i) in exEdges" :key="'ex'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#e74c3c" stroke-width="3" marker-end="url(#arrowEx)"
            />
            <!-- Ey (green) - edges parallel to y -->
            <line v-for="(edge, i) in eyEdges" :key="'ey'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#27ae60" stroke-width="3" marker-end="url(#arrowEy)"
            />
            <!-- Ez (blue) - edges parallel to z -->
            <line v-for="(edge, i) in ezEdges" :key="'ez'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#2980b9" stroke-width="3" marker-end="url(#arrowEz)"
            />
          </template>

          <!-- H-field components (on face centers) -->
          <template v-if="showH">
            <!-- Hx (orange) - face perpendicular to x -->
            <line v-for="(edge, i) in hxArrows" :key="'hx'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#f39c12" stroke-width="2.5" marker-end="url(#arrowHx)"
            />
            <!-- Hy (cyan) - face perpendicular to y -->
            <line v-for="(edge, i) in hyArrows" :key="'hy'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#00bcd4" stroke-width="2.5" marker-end="url(#arrowHy)"
            />
            <!-- Hz (purple) - face perpendicular to z -->
            <line v-for="(edge, i) in hzArrows" :key="'hz'+i"
              :x1="edge.x1" :y1="edge.y1" :x2="edge.x2" :y2="edge.y2"
              stroke="#8e44ad" stroke-width="2.5" marker-end="url(#arrowHz)"
            />
          </template>

          <!-- Arrow markers -->
          <defs>
            <marker id="arrowEx" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#e74c3c" />
            </marker>
            <marker id="arrowEy" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#27ae60" />
            </marker>
            <marker id="arrowEz" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#2980b9" />
            </marker>
            <marker id="arrowHx" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#f39c12" />
            </marker>
            <marker id="arrowHy" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#00bcd4" />
            </marker>
            <marker id="arrowHz" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
              <polygon points="0 0, 7 2.5, 0 5" fill="#8e44ad" />
            </marker>
          </defs>

          <!-- Legend -->
          <template v-if="showE">
            <line :x1="W - 130" y1="20" :x2="W - 110" y2="20" stroke="#e74c3c" stroke-width="3" />
            <text :x="W - 105" y="24" class="legend-label">Ex</text>
            <line :x1="W - 130" y1="35" :x2="W - 110" y2="35" stroke="#27ae60" stroke-width="3" />
            <text :x="W - 105" y="39" class="legend-label">Ey</text>
            <line :x1="W - 130" y1="50" :x2="W - 110" y2="50" stroke="#2980b9" stroke-width="3" />
            <text :x="W - 105" y="54" class="legend-label">Ez</text>
          </template>
          <template v-if="showH">
            <line :x1="W - 65" y1="20" :x2="W - 45" y2="20" stroke="#f39c12" stroke-width="2.5" />
            <text :x="W - 40" y="24" class="legend-label">Hx</text>
            <line :x1="W - 65" y1="35" :x2="W - 45" y2="35" stroke="#00bcd4" stroke-width="2.5" />
            <text :x="W - 40" y="39" class="legend-label">Hy</text>
            <line :x1="W - 65" y1="50" :x2="W - 45" y2="50" stroke="#8e44ad" stroke-width="2.5" />
            <text :x="W - 40" y="54" class="legend-label">Hz</text>
          </template>
        </svg>
      </div>

      <div class="explain-panel">
        <div class="explain-title">{{ t('Staggered Grid Concept', '엇갈린 격자 개념') }}</div>
        <p class="explain-text">
          {{ t(
            'The Yee cell staggers E and H field components in both space and time. E-field components are located on the',
            'Yee 셀은 E장과 H장 성분을 공간과 시간 모두에서 엇갈리게 배치합니다. 전기장 성분은 셀의'
          ) }}
          <strong>{{ t('edges', '모서리') }}</strong>
          {{ t(
            'of the cell, while H-field components are located at the',
            '에, 자기장 성분은'
          ) }}
          <strong>{{ t('face centers', '면 중심') }}</strong>{{ t('.', '에 위치합니다.') }}
        </p>
        <p class="explain-text">
          {{ t(
            'This arrangement ensures that every curl finite-difference is centered, giving second-order accuracy. The spatial staggering by half a grid cell naturally satisfies the divergence-free conditions for both E and B fields.',
            '이 배치는 모든 curl 유한 차분이 중심에 위치하도록 보장하여 2차 정확도를 제공합니다. 반 격자 셀만큼의 공간적 엇갈림은 E장과 B장 모두의 발산 없는 조건을 자연스럽게 만족시킵니다.'
          ) }}
        </p>
        <p class="explain-text">
          {{ t(
            `For a grid spacing of`,
            `격자 간격`
          ) }}
          <strong>{{ dx }} nm</strong>{{ t(', the cell resolves', '에서, 셀은') }}
          <strong>{{ cellsPerWl.toFixed(1) }}</strong>
          {{ t(
            'points per wavelength at 550 nm in silicon (n=4). A minimum of 10 cells per wavelength is recommended.',
            '개의 포인트를 파장당 분해합니다 (550 nm, silicon, n=4). 파장당 최소 10개의 셀을 권장합니다.'
          ) }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const W = 480
const H = 360
const S = 120 // cube size in projected units
const ox = W / 2 - 30 // origin offset x
const oy = H / 2 + 40 // origin offset y

const showE = ref(true)
const showH = ref(true)
const dx = ref(25)

// Isometric projection
function iso(x, y, z) {
  const px = ox + (x - z) * 0.866
  const py = oy - y + (x + z) * 0.35
  return { x: px, y: py }
}

// Cells per wavelength at 550nm in silicon (n~4)
const cellsPerWl = computed(() => {
  const lambdaMedium = 550 / 4.0 // wavelength in silicon ~137.5 nm
  return lambdaMedium / dx.value
})

// Courant number: S = c*dt/dx, with dt = dx/(c*sqrt(3)) for cubic grid at stability limit
// We display what the Courant number would be if dt is set to 0.99 * limit
const courantLimit = computed(() => 1 / Math.sqrt(3))
const courant = computed(() => 0.99 * courantLimit.value)

const arrowLen = S * 0.4 // length of field arrows

// E-field arrows on edges
// Ex: on edges parallel to x-axis (at midpoints of x-edges)
const exEdges = computed(() => {
  const half = S / 2
  const al = arrowLen
  const positions = [
    // Bottom face, y=0 edges
    { sx: half - al / 2, sy: 0, sz: 0 },
    { sx: half - al / 2, sy: 0, sz: S },
    // Top face, y=S edges
    { sx: half - al / 2, sy: S, sz: 0 },
    { sx: half - al / 2, sy: S, sz: S },
  ]
  return positions.map(p => {
    const p1 = iso(p.sx, p.sy, p.sz)
    const p2 = iso(p.sx + al, p.sy, p.sz)
    return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
  })
})

// Ey: on edges parallel to y-axis
const eyEdges = computed(() => {
  const half = S / 2
  const al = arrowLen
  const positions = [
    { sx: 0, sy: half - al / 2, sz: 0 },
    { sx: S, sy: half - al / 2, sz: 0 },
    { sx: 0, sy: half - al / 2, sz: S },
    { sx: S, sy: half - al / 2, sz: S },
  ]
  return positions.map(p => {
    const p1 = iso(p.sx, p.sy, p.sz)
    const p2 = iso(p.sx, p.sy + al, p.sz)
    return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
  })
})

// Ez: on edges parallel to z-axis
const ezEdges = computed(() => {
  const half = S / 2
  const al = arrowLen
  const positions = [
    { sx: 0, sy: 0, sz: half - al / 2 },
    { sx: S, sy: 0, sz: half - al / 2 },
    { sx: 0, sy: S, sz: half - al / 2 },
    { sx: S, sy: S, sz: half - al / 2 },
  ]
  return positions.map(p => {
    const p1 = iso(p.sx, p.sy, p.sz)
    const p2 = iso(p.sx, p.sy, p.sz + al)
    return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
  })
})

// H-field arrows at face centers
// Hx: on faces perpendicular to x (at x=0 and x=S face centers)
const hxArrows = computed(() => {
  const half = S / 2
  const al = arrowLen * 0.7
  return [
    // Face at x=0: center is (0, S/2, S/2), arrow along x direction on the face (actually along y or z)
    // Hx is perpendicular to x-face, so it points in x direction from face center
    (() => {
      const p1 = iso(-al / 2, half, half)
      const p2 = iso(al / 2, half, half)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
    (() => {
      const p1 = iso(S - al / 2, half, half)
      const p2 = iso(S + al / 2, half, half)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
  ]
})

// Hy: on faces perpendicular to y
const hyArrows = computed(() => {
  const half = S / 2
  const al = arrowLen * 0.7
  return [
    (() => {
      const p1 = iso(half, -al / 2, half)
      const p2 = iso(half, al / 2, half)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
    (() => {
      const p1 = iso(half, S - al / 2, half)
      const p2 = iso(half, S + al / 2, half)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
  ]
})

// Hz: on faces perpendicular to z
const hzArrows = computed(() => {
  const half = S / 2
  const al = arrowLen * 0.7
  return [
    (() => {
      const p1 = iso(half, half, -al / 2)
      const p2 = iso(half, half, al / 2)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
    (() => {
      const p1 = iso(half, half, S - al / 2)
      const p2 = iso(half, half, S + al / 2)
      return { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    })(),
  ]
})
</script>

<style scoped>
.yee-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.yee-container h4 {
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
  gap: 20px;
  flex-wrap: wrap;
  align-items: flex-end;
  margin-bottom: 16px;
}
.toggle-group {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.toggle-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
}
.toggle-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: var(--vp-c-brand-1);
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
.info-value.warn {
  color: #e74c3c;
}
.content-layout {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.svg-wrapper {
  flex: 1;
  min-width: 300px;
}
.yee-svg {
  width: 100%;
  max-width: 500px;
  display: block;
  margin: 0 auto;
}
.axis-label-3d {
  font-size: 13px;
  fill: var(--vp-c-text-2);
  font-weight: 700;
  font-style: italic;
}
.legend-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.explain-panel {
  flex: 1;
  min-width: 220px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 16px;
}
.explain-title {
  font-size: 0.95em;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  margin-bottom: 10px;
}
.explain-text {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  margin: 0 0 10px 0;
  line-height: 1.5;
}
.explain-text:last-child {
  margin-bottom: 0;
}
</style>
