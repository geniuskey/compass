<template>
  <div class="pixel-3d-viewer">
    <div class="controls-row">
      <label class="control-item">
        <input type="checkbox" v-model="exploded" />
        <span>Exploded View</span>
      </label>
      <div class="layer-toggles">
        <label
          v-for="layer in layerDefs"
          :key="layer.id"
          class="toggle-item"
        >
          <input type="checkbox" v-model="layer.visible" />
          <span class="toggle-swatch" :style="{ background: layer.color }"></span>
          <span class="toggle-label">{{ layer.label }}</span>
        </label>
      </div>
    </div>

    <svg
      ref="svgEl"
      :viewBox="`0 0 ${svgW} ${svgH}`"
      class="viewer-svg"
      @mousedown="onDragStart"
      @mousemove="onDragMove"
      @mouseup="onDragEnd"
      @mouseleave="onDragEnd"
      @touchstart.prevent="onTouchStart"
      @touchmove.prevent="onTouchMove"
      @touchend="onTouchEnd"
    >
      <!-- Render layers back to front -->
      <template v-for="layer in sortedLayers" :key="'3d-' + layer.id">
        <template v-if="layer.visible">
          <!-- Special: CF layer top face = Bayer pattern -->
          <template v-if="layer.id === 'colorfilter'">
            <!-- Side face -->
            <polygon
              :points="polyPoints(layer, 'side')"
              :fill="darken(layer.color, 0.7)"
              opacity="0.4"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Front face -->
            <polygon
              :points="polyPoints(layer, 'front')"
              :fill="darken(layer.color, 0.85)"
              opacity="0.5"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Bayer top faces -->
            <polygon
              v-for="(cell, ci) in bayerTopFaces(layer)"
              :key="'bayer-' + ci"
              :points="cell.points"
              :fill="cell.color"
              opacity="0.6"
              stroke="#444"
              stroke-width="0.3"
            />
            <!-- Metal grid lines on top -->
            <line
              v-for="(ml, mi) in metalGridLines(layer)"
              :key="'mg3d-' + mi"
              :x1="ml.x1"
              :y1="ml.y1"
              :x2="ml.x2"
              :y2="ml.y2"
              stroke="#555555"
              stroke-width="1.5"
              opacity="0.7"
            />
          </template>

          <!-- Special: Silicon layer with DTI -->
          <template v-else-if="layer.id === 'silicon'">
            <!-- Side face -->
            <polygon
              :points="polyPoints(layer, 'side')"
              :fill="darken(layer.color, 0.7)"
              opacity="0.4"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Front face -->
            <polygon
              :points="polyPoints(layer, 'front')"
              :fill="darken(layer.color, 0.85)"
              opacity="0.5"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Top face -->
            <polygon
              :points="polyPoints(layer, 'top')"
              :fill="layer.color"
              opacity="0.6"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- DTI walls on top face -->
            <line
              v-for="(dl, di) in dtiLines(layer)"
              :key="'dti3d-' + di"
              :x1="dl.x1"
              :y1="dl.y1"
              :x2="dl.x2"
              :y2="dl.y2"
              stroke="#aed6f1"
              stroke-width="2"
              opacity="0.8"
            />
          </template>

          <!-- Special: Microlens with dome -->
          <template v-else-if="layer.id === 'microlens'">
            <!-- Side face -->
            <polygon
              :points="polyPoints(layer, 'side')"
              :fill="darken(layer.color, 0.7)"
              opacity="0.4"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Front face -->
            <polygon
              :points="polyPoints(layer, 'front')"
              :fill="darken(layer.color, 0.85)"
              opacity="0.5"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Top face (base) -->
            <polygon
              :points="polyPoints(layer, 'top')"
              :fill="layer.color"
              opacity="0.4"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Microlens domes -->
            <path
              v-for="(dome, di) in microlensDomes(layer)"
              :key="'dome3d-' + di"
              :d="dome.path"
              fill="#dda0dd"
              opacity="0.55"
              stroke="#b07eb0"
              stroke-width="0.8"
            />
          </template>

          <!-- Normal layers: 3 faces -->
          <template v-else>
            <!-- Side face -->
            <polygon
              :points="polyPoints(layer, 'side')"
              :fill="darken(layer.color, 0.7)"
              opacity="0.4"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Front face -->
            <polygon
              :points="polyPoints(layer, 'front')"
              :fill="darken(layer.color, 0.85)"
              opacity="0.5"
              stroke="#444"
              stroke-width="0.5"
            />
            <!-- Top face -->
            <polygon
              :points="polyPoints(layer, 'top')"
              :fill="layer.color"
              opacity="0.6"
              stroke="#444"
              stroke-width="0.5"
            />
          </template>

          <!-- Layer label -->
          <text
            :x="labelPos(layer).x"
            :y="labelPos(layer).y"
            class="layer-label-3d"
            text-anchor="end"
          >{{ layer.label }} ({{ layer.thickness }}µm)</text>
        </template>
      </template>

      <!-- Hover tooltip -->
      <template v-if="hoveredLayer">
        <rect
          :x="hoverPos.x"
          :y="hoverPos.y"
          width="170"
          height="42"
          rx="4"
          fill="var(--vp-c-bg)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.8"
          opacity="0.95"
        />
        <text
          :x="hoverPos.x + 8"
          :y="hoverPos.y + 16"
          class="tooltip-text-3d"
          font-weight="600"
        >{{ hoveredLayer.label }}</text>
        <text
          :x="hoverPos.x + 8"
          :y="hoverPos.y + 33"
          class="tooltip-text-3d"
          fill="var(--vp-c-text-2)"
        >{{ hoveredLayer.material }} · {{ hoveredLayer.thickness }}µm</text>
      </template>

      <!-- Drag hint -->
      <text
        :x="svgW / 2"
        :y="svgH - 8"
        class="hint-text"
        text-anchor="middle"
      >Drag to rotate</text>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'

// SVG dimensions
const svgW = 560
const svgH = 480

// Camera
const azimuth = ref(30) // degrees
const dragging = ref(false)
const dragStartX = ref(0)
const dragStartAz = ref(0)
const svgEl = ref<SVGSVGElement | null>(null)

// Controls
const exploded = ref(false)

// Hover
const hoveredLayer = ref<LayerDef | null>(null)
const hoverPos = reactive({ x: 0, y: 0 })

// Layer definitions
interface LayerDef {
  id: string
  label: string
  color: string
  zBot: number
  zTop: number
  thickness: string
  material: string
  visible: boolean
}

const layerDefs = reactive<LayerDef[]>([
  { id: 'silicon', label: 'Silicon', color: '#5d6d7e', zBot: 0, zTop: 3.0, thickness: '3.0', material: 'Si', visible: true },
  { id: 'barl', label: 'BARL', color: '#8e44ad', zBot: 3.0, zTop: 3.08, thickness: '0.08', material: 'SiO2/HfO2/SiO2/Si3N4', visible: true },
  { id: 'colorfilter', label: 'Color Filter', color: '#27ae60', zBot: 3.08, zTop: 3.68, thickness: '0.6', material: 'Organic dye', visible: true },
  { id: 'planarization', label: 'Planarization', color: '#d5dbdb', zBot: 3.68, zTop: 3.98, thickness: '0.3', material: 'SiO2', visible: true },
  { id: 'microlens', label: 'Microlens', color: '#dda0dd', zBot: 3.98, zTop: 4.58, thickness: '0.6', material: 'Polymer (n=1.56)', visible: true },
  { id: 'air', label: 'Air', color: '#d6eaf8', zBot: 4.58, zTop: 5.58, thickness: '1.0', material: 'Air', visible: false },
])

// 3D projection
const scale = 80
const centerX = svgW / 2
const centerY = svgH * 0.75
const zScale = 0.5 // compress z axis

function project(x: number, y: number, z: number): { sx: number; sy: number } {
  const theta = (azimuth.value * Math.PI) / 180
  const cosT = Math.cos(theta)
  const sinT = Math.sin(theta)
  return {
    sx: centerX + scale * (x * cosT - y * sinT),
    sy: centerY - scale * z * zScale + scale * 0.5 * (x * sinT + y * cosT),
  }
}

// Exploded offsets
function layerOffset(layer: LayerDef): number {
  if (!exploded.value) return 0
  const idx = layerDefs.indexOf(layer)
  return idx * 0.3
}

function effectiveZBot(layer: LayerDef): number {
  return layer.zBot + layerOffset(layer)
}

function effectiveZTop(layer: LayerDef): number {
  return layer.zTop + layerOffset(layer)
}

// Sort layers: painter's algorithm (back to front by z)
const sortedLayers = computed(() => {
  const visible = layerDefs.filter(l => l.visible)
  return [...visible].sort((a, b) => effectiveZBot(a) - effectiveZBot(b))
})

// Box corners: (0,0)-(2,2) in x,y
function getCorners(layer: LayerDef) {
  const zb = effectiveZBot(layer)
  const zt = effectiveZTop(layer)
  return {
    // bottom face corners
    b00: project(0, 0, zb),
    b20: project(2, 0, zb),
    b22: project(2, 2, zb),
    b02: project(0, 2, zb),
    // top face corners
    t00: project(0, 0, zt),
    t20: project(2, 0, zt),
    t22: project(2, 2, zt),
    t02: project(0, 2, zt),
  }
}

function polyPoints(layer: LayerDef, face: 'top' | 'front' | 'side'): string {
  const c = getCorners(layer)
  const theta = azimuth.value % 360
  const norm = ((theta % 360) + 360) % 360

  if (face === 'top') {
    return `${c.t00.sx},${c.t00.sy} ${c.t20.sx},${c.t20.sy} ${c.t22.sx},${c.t22.sy} ${c.t02.sx},${c.t02.sy}`
  }

  // Determine which faces are visible based on azimuth
  if (face === 'front') {
    if (norm >= 0 && norm < 180) {
      // Show front face (y=0 side)
      return `${c.b00.sx},${c.b00.sy} ${c.b20.sx},${c.b20.sy} ${c.t20.sx},${c.t20.sy} ${c.t00.sx},${c.t00.sy}`
    } else {
      // Show back face (y=2 side)
      return `${c.b02.sx},${c.b02.sy} ${c.b22.sx},${c.b22.sy} ${c.t22.sx},${c.t22.sy} ${c.t02.sx},${c.t02.sy}`
    }
  }

  // side
  if (norm >= 270 || norm < 90) {
    // Show right side (x=2)
    return `${c.b20.sx},${c.b20.sy} ${c.b22.sx},${c.b22.sy} ${c.t22.sx},${c.t22.sy} ${c.t20.sx},${c.t20.sy}`
  } else {
    // Show left side (x=0)
    return `${c.b00.sx},${c.b00.sy} ${c.b02.sx},${c.b02.sy} ${c.t02.sx},${c.t02.sy} ${c.t00.sx},${c.t00.sy}`
  }
}

// Bayer top faces for CF layer
function bayerTopFaces(layer: LayerDef) {
  const zt = effectiveZTop(layer)
  const cells = [
    { x0: 0, y0: 0, x1: 1, y1: 1, color: '#c0392b', label: 'R' },
    { x0: 1, y0: 0, x1: 2, y1: 1, color: '#27ae60', label: 'G' },
    { x0: 0, y0: 1, x1: 1, y1: 2, color: '#27ae60', label: 'G' },
    { x0: 1, y0: 1, x1: 2, y1: 2, color: '#2980b9', label: 'B' },
  ]
  return cells.map(c => {
    const p0 = project(c.x0, c.y0, zt)
    const p1 = project(c.x1, c.y0, zt)
    const p2 = project(c.x1, c.y1, zt)
    const p3 = project(c.x0, c.y1, zt)
    return {
      points: `${p0.sx},${p0.sy} ${p1.sx},${p1.sy} ${p2.sx},${p2.sy} ${p3.sx},${p3.sy}`,
      color: c.color,
    }
  })
}

// Metal grid lines on CF top
function metalGridLines(layer: LayerDef) {
  const zt = effectiveZTop(layer)
  const lines: { x1: number; y1: number; x2: number; y2: number }[] = []
  // Vertical line at x=1
  const p0 = project(1, 0, zt)
  const p1 = project(1, 2, zt)
  lines.push({ x1: p0.sx, y1: p0.sy, x2: p1.sx, y2: p1.sy })
  // Horizontal line at y=1
  const p2 = project(0, 1, zt)
  const p3 = project(2, 1, zt)
  lines.push({ x1: p2.sx, y1: p2.sy, x2: p3.sx, y2: p3.sy })
  return lines
}

// DTI lines on silicon top
function dtiLines(layer: LayerDef) {
  const zt = effectiveZTop(layer)
  const lines: { x1: number; y1: number; x2: number; y2: number }[] = []
  // Vertical at x=1
  const p0 = project(1, 0, zt)
  const p1 = project(1, 2, zt)
  lines.push({ x1: p0.sx, y1: p0.sy, x2: p1.sx, y2: p1.sy })
  // Horizontal at y=1
  const p2 = project(0, 1, zt)
  const p3 = project(2, 1, zt)
  lines.push({ x1: p2.sx, y1: p2.sy, x2: p3.sx, y2: p3.sy })
  return lines
}

// Microlens domes
function microlensDomes(layer: LayerDef) {
  const zBase = effectiveZTop(layer) - 0.6 // base of the dome within the microlens layer
  const mlH = 0.6
  const mlR = 0.48
  const mlN = 2.5
  const centers = [
    { cx: 0.5, cy: 0.5 },
    { cx: 1.5, cy: 0.5 },
    { cx: 0.5, cy: 1.5 },
    { cx: 1.5, cy: 1.5 },
  ]

  return centers.map(({ cx, cy }) => {
    const numPts = 40
    const pts: { sx: number; sy: number }[] = []

    // Generate dome profile along x-axis at cy
    for (let i = 0; i <= numPts; i++) {
      const t = -1 + (2 * i) / numPts
      const physX = cx + t * mlR
      const absT = Math.abs(t)
      const dz = absT >= 1 ? 0 : mlH * Math.pow(1 - Math.pow(absT, mlN), 1 / mlN)
      const p = project(physX, cy, zBase + dz)
      pts.push(p)
    }

    // Close with base line
    const pStart = project(cx - mlR, cy, zBase)
    const pEnd = project(cx + mlR, cy, zBase)

    let d = `M ${pStart.sx.toFixed(1)} ${pStart.sy.toFixed(1)}`
    for (const p of pts) {
      d += ` L ${p.sx.toFixed(1)} ${p.sy.toFixed(1)}`
    }
    d += ` L ${pEnd.sx.toFixed(1)} ${pEnd.sy.toFixed(1)} Z`

    return { path: d }
  })
}

// Label position
function labelPos(layer: LayerDef) {
  const zMid = (effectiveZBot(layer) + effectiveZTop(layer)) / 2
  const p = project(-0.3, 1, zMid)
  return { x: p.sx, y: p.sy }
}

// Darken color
function darken(hex: string, factor: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const nr = Math.round(r * factor)
  const ng = Math.round(g * factor)
  const nb = Math.round(b * factor)
  return `#${nr.toString(16).padStart(2, '0')}${ng.toString(16).padStart(2, '0')}${nb.toString(16).padStart(2, '0')}`
}

// Drag rotation
function onDragStart(e: MouseEvent) {
  dragging.value = true
  dragStartX.value = e.clientX
  dragStartAz.value = azimuth.value
}

function onDragMove(e: MouseEvent) {
  if (!dragging.value) return
  const dx = e.clientX - dragStartX.value
  azimuth.value = dragStartAz.value + dx * 0.5
}

function onDragEnd() {
  dragging.value = false
}

// Touch support
function onTouchStart(e: TouchEvent) {
  if (e.touches.length === 1) {
    dragging.value = true
    dragStartX.value = e.touches[0].clientX
    dragStartAz.value = azimuth.value
  }
}

function onTouchMove(e: TouchEvent) {
  if (!dragging.value || e.touches.length !== 1) return
  const dx = e.touches[0].clientX - dragStartX.value
  azimuth.value = dragStartAz.value + dx * 0.5
}

function onTouchEnd() {
  dragging.value = false
}
</script>

<style scoped>
.pixel-3d-viewer {
  padding: 16px 0;
}
.controls-row {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.control-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.88em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  cursor: pointer;
  white-space: nowrap;
}
.control-item input[type="checkbox"] {
  accent-color: var(--vp-c-brand-1);
}
.layer-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 14px;
}
.toggle-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.82em;
  color: var(--vp-c-text-2);
  cursor: pointer;
}
.toggle-item input[type="checkbox"] {
  accent-color: var(--vp-c-brand-1);
  width: 14px;
  height: 14px;
}
.toggle-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  border: 1px solid var(--vp-c-divider);
}
.toggle-label {
  white-space: nowrap;
}
.viewer-svg {
  width: 100%;
  max-width: 600px;
  height: auto;
  display: block;
  margin: 0 auto;
  cursor: grab;
  user-select: none;
  -webkit-user-select: none;
}
.viewer-svg:active {
  cursor: grabbing;
}
.layer-label-3d {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 500;
  pointer-events: none;
}
.tooltip-text-3d {
  font-size: 11px;
  fill: var(--vp-c-text-1);
}
.hint-text {
  font-size: 11px;
  fill: var(--vp-c-text-3);
  font-style: italic;
}
</style>
