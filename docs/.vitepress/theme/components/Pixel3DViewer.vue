<template>
  <div class="pixel-3d-viewer">
    <div class="controls-row">
      <label class="ctrl-item">
        <input type="checkbox" v-model="exploded" />
        <span>Exploded View</span>
      </label>
      <div class="layer-toggles">
        <label v-for="l in layerDefs" :key="l.id" class="toggle-item">
          <input type="checkbox" v-model="l.visible" />
          <span class="toggle-sw" :style="{ background: l.color }"></span>
          <span class="toggle-lbl">{{ l.label }}</span>
        </label>
      </div>
    </div>

    <svg
      :viewBox="`0 0 ${W} ${H}`"
      class="viewer-svg"
      @mousedown.prevent="startDrag"
      @mousemove="onDrag"
      @mouseup="endDrag"
      @mouseleave="endDrag"
      @touchstart.prevent="startTouch"
      @touchmove.prevent="onTouch"
      @touchend="endDrag"
    >
      <!-- Ground grid -->
      <line
        v-for="g in groundLines"
        :key="g.k"
        :x1="g.x1" :y1="g.y1" :x2="g.x2" :y2="g.y2"
        stroke="var(--vp-c-divider)" stroke-width="0.5" opacity="0.35"
      />

      <!-- All polygons: depth-sorted back to front -->
      <polygon
        v-for="(p, i) in sortedPolys"
        :key="i"
        :points="p.pts"
        :fill="p.fill"
        :fill-opacity="p.fop"
        :stroke="p.stk"
        :stroke-width="p.sw"
        :stroke-dasharray="p.dash || ''"
      />

      <!-- Layer labels with leader lines -->
      <template v-for="lb in labels" :key="'lb-' + lb.id">
        <line
          :x1="lb.lx1" :y1="lb.ly1" :x2="lb.lx2" :y2="lb.ly2"
          stroke="var(--vp-c-text-3)" stroke-width="0.5" opacity="0.5"
        />
        <text :x="lb.tx" :y="lb.ty" class="label-3d" text-anchor="end">
          {{ lb.text }}
        </text>
      </template>

      <text :x="W / 2" :y="H - 6" class="hint" text-anchor="middle">
        Drag to rotate
      </text>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'

// --- Constants ---
const W = 600, H = 520
const S = 55 // px per µm
const CX = W / 2, CY = H * 0.68
const ZS = 0.5 // z-axis compression

// --- State ---
const azimuth = ref(35)
const exploded = ref(false)
const dragging = ref(false)
const dragX0 = ref(0)
const dragAz0 = ref(0)

// --- Layer data ---
interface LayerDef {
  id: string; label: string; color: string
  zBot: number; zTop: number; thickness: string; material: string
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

// --- Projection ---
interface V3 { x: number; y: number; z: number }

function proj(x: number, y: number, z: number) {
  const t = azimuth.value * Math.PI / 180
  return {
    x: CX + S * (x * Math.cos(t) - y * Math.sin(t)),
    y: CY - S * z * ZS + S * 0.5 * (x * Math.sin(t) + y * Math.cos(t)),
  }
}

// --- Color utilities ---
function shade(hex: string, f: number): string {
  const r = Math.min(255, Math.round(parseInt(hex.slice(1, 3), 16) * f))
  const g = Math.min(255, Math.round(parseInt(hex.slice(3, 5), 16) * f))
  const b = Math.min(255, Math.round(parseInt(hex.slice(5, 7), 16) * f))
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')
}

// --- Polygon builder ---
interface Poly {
  pts: string; fill: string; fop: number
  stk: string; sw: number; depth: number; dash?: string
}

function vToStr(verts: V3[]): string {
  return verts.map(v => { const p = proj(v.x, v.y, v.z); return `${p.x.toFixed(1)},${p.y.toFixed(1)}` }).join(' ')
}

function depthOf(verts: V3[]): number {
  const t = azimuth.value * Math.PI / 180
  const n = verts.length
  let sx = 0, sy = 0, sz = 0
  for (const v of verts) { sx += v.x; sy += v.y; sz += v.z }
  sx /= n; sy /= n; sz /= n
  return sz * 5 - (sx * Math.sin(t) + sy * Math.cos(t))
}

function pushPoly(arr: Poly[], verts: V3[], fill: string, fop: number, stk = '#333', sw = 0.5, dash?: string) {
  arr.push({ pts: vToStr(verts), fill, fop, stk, sw, depth: depthOf(verts), dash })
}

// --- Box faces (3 visible faces based on azimuth) ---
function addBox(
  arr: Poly[], x0: number, y0: number, z0: number, x1: number, y1: number, z1: number,
  col: string, topOp = 0.85, fOp = 0.8, sOp = 0.75,
  topQuads?: { x0: number; y0: number; x1: number; y1: number; color: string }[],
) {
  const t = azimuth.value * Math.PI / 180
  const sT = Math.sin(t), cT = Math.cos(t)

  // Top face(s)
  if (topQuads) {
    for (const q of topQuads) {
      pushPoly(arr,
        [{ x: q.x0, y: q.y0, z: z1 }, { x: q.x1, y: q.y0, z: z1 }, { x: q.x1, y: q.y1, z: z1 }, { x: q.x0, y: q.y1, z: z1 }],
        q.color, topOp)
    }
  } else {
    pushPoly(arr,
      [{ x: x0, y: y0, z: z1 }, { x: x1, y: y0, z: z1 }, { x: x1, y: y1, z: z1 }, { x: x0, y: y1, z: z1 }],
      col, topOp)
  }

  // Y-face (front or back)
  const fy = cT >= 0 ? y0 : y1
  pushPoly(arr,
    [{ x: x0, y: fy, z: z0 }, { x: x1, y: fy, z: z0 }, { x: x1, y: fy, z: z1 }, { x: x0, y: fy, z: z1 }],
    shade(col, 0.78), fOp)

  // X-face (left or right)
  const fx = sT >= 0 ? x0 : x1
  pushPoly(arr,
    [{ x: fx, y: y0, z: z0 }, { x: fx, y: y1, z: z0 }, { x: fx, y: y1, z: z1 }, { x: fx, y: y0, z: z1 }],
    shade(col, 0.55), sOp)
}

// --- Dome mesh with per-face shading ---
function addDome(arr: Poly[], cx: number, cy: number, zBase: number) {
  const h = 0.6, R = 0.48, n = 2.5
  const N = 12
  const step = 2 * R / N
  // Light direction (normalized)
  const lx = -0.35, ly = -0.50, lz = 0.79

  // Build height grid
  const grid: (V3 | null)[][] = []
  for (let i = 0; i <= N; i++) {
    const row: (V3 | null)[] = []
    for (let j = 0; j <= N; j++) {
      const dx = -R + i * step
      const dy = -R + j * step
      const u = Math.abs(dx / R), v = Math.abs(dy / R)
      const val = Math.pow(u, n) + Math.pow(v, n)
      if (val >= 1) { row.push(null) } else {
        row.push({ x: cx + dx, y: cy + dy, z: zBase + h * Math.pow(1 - val, 1 / n) })
      }
    }
    grid.push(row)
  }

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v00 = grid[i][j], v10 = grid[i + 1][j], v11 = grid[i + 1][j + 1], v01 = grid[i][j + 1]
      if (!v00 || !v10 || !v11 || !v01) continue

      // Cross product for face normal
      const e1x = v10.x - v00.x, e1y = v10.y - v00.y, e1z = v10.z - v00.z
      const e2x = v01.x - v00.x, e2y = v01.y - v00.y, e2z = v01.z - v00.z
      let nx = e1y * e2z - e1z * e2y
      let ny = e1z * e2x - e1x * e2z
      let nz = e1x * e2y - e1y * e2x
      if (nz < 0) { nx = -nx; ny = -ny; nz = -nz }
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
      if (len < 1e-10) continue
      nx /= len; ny /= len; nz /= len

      const dot = nx * lx + ny * ly + nz * lz
      const bright = 0.4 + 0.6 * Math.max(0, dot)
      pushPoly(arr, [v00, v10, v11, v01],
        shade('#dda0dd', bright), 0.88,
        shade('#b07eb0', bright * 0.7), 0.25)
    }
  }
}

// --- Main polygon generation ---
const sortedPolys = computed(() => {
  const polys: Poly[] = []

  for (const layer of layerDefs) {
    if (!layer.visible) continue
    const idx = layerDefs.indexOf(layer)
    const off = exploded.value ? idx * 0.35 : 0
    const zb = layer.zBot + off, zt = layer.zTop + off

    switch (layer.id) {
      case 'silicon': {
        // Semi-transparent box
        addBox(polys, 0, 0, zb, 2, 2, zt, layer.color, 0.6, 0.55, 0.5)
        // DTI walls (3D boxes)
        const dw = 0.05
        addBox(polys, 1 - dw, 0, zb, 1 + dw, 2, zt, '#aed6f1', 0.75, 0.65, 0.6)
        addBox(polys, 0, 1 - dw, zb, 2, 1 + dw, zt, '#aed6f1', 0.75, 0.65, 0.6)
        // Photodiode outlines on top face
        for (const c of [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]) {
          const px = c[0] - 0.35, py = c[1] - 0.35
          const verts: V3[] = [
            { x: px, y: py, z: zt }, { x: px + 0.7, y: py, z: zt },
            { x: px + 0.7, y: py + 0.7, z: zt }, { x: px, y: py + 0.7, z: zt },
          ]
          pushPoly(polys, verts, '#b85c5c', 0.2, '#b85c5c', 1.2, '3 2')
        }
        break
      }
      case 'barl': {
        // Render 4 BARL sublayers — each material gets a distinct color
        const subs = [
          { dz: 0, t: 0.01, col: '#7fb3d8' },      // SiO2
          { dz: 0.01, t: 0.025, col: '#6c71c4' },   // HfO2
          { dz: 0.035, t: 0.015, col: '#e8d44d' },   // SiO2
          { dz: 0.05, t: 0.030, col: '#2aa198' },    // Si3N4
        ]
        for (const s of subs) {
          addBox(polys, 0, 0, zb + s.dz, 2, 2, zb + s.dz + s.t, s.col, 0.85, 0.75, 0.65)
        }
        break
      }
      case 'colorfilter': {
        const bayer = [
          { x0: 0, y0: 0, x1: 1, y1: 1, color: '#c0392b' },
          { x0: 1, y0: 0, x1: 2, y1: 1, color: '#27ae60' },
          { x0: 0, y0: 1, x1: 1, y1: 2, color: '#27ae60' },
          { x0: 1, y0: 1, x1: 2, y1: 2, color: '#2980b9' },
        ]
        addBox(polys, 0, 0, zb, 2, 2, zt, layer.color, 0.8, 0.7, 0.65, bayer)
        // Metal grid walls
        const mw = 0.025
        addBox(polys, 1 - mw, 0, zb, 1 + mw, 2, zt, '#555555', 0.85, 0.75, 0.7)
        addBox(polys, 0, 1 - mw, zb, 2, 1 + mw, zt, '#555555', 0.85, 0.75, 0.7)
        break
      }
      case 'microlens': {
        // Only render domes — no enclosing box
        for (const c of [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]) {
          addDome(polys, c[0], c[1], zb)
        }
        break
      }
      default:
        addBox(polys, 0, 0, zb, 2, 2, zt, layer.color)
    }
  }

  polys.sort((a, b) => a.depth - b.depth)
  return polys
})

// --- Ground grid ---
const groundLines = computed(() => {
  const lines: { k: string; x1: number; y1: number; x2: number; y2: number }[] = []
  for (let v = 0; v <= 2; v++) {
    const a = proj(v, 0, 0), b = proj(v, 2, 0)
    lines.push({ k: `gx${v}`, x1: a.x, y1: a.y, x2: b.x, y2: b.y })
    const c = proj(0, v, 0), d = proj(2, v, 0)
    lines.push({ k: `gy${v}`, x1: c.x, y1: c.y, x2: d.x, y2: d.y })
  }
  return lines
})

// --- Labels ---
const labels = computed(() => {
  const out: {
    id: string; text: string
    lx1: number; ly1: number; lx2: number; ly2: number
    tx: number; ty: number
  }[] = []
  for (const l of layerDefs) {
    if (!l.visible) continue
    const idx = layerDefs.indexOf(l)
    const off = exploded.value ? idx * 0.35 : 0
    const zMid = (l.zBot + l.zTop) / 2 + off
    const edge = proj(-0.15, -0.15, zMid)
    const lbl = { x: edge.x - 55, y: edge.y }
    out.push({
      id: l.id,
      text: `${l.label} (${l.thickness}µm)`,
      lx1: lbl.x + 50, ly1: lbl.y,
      lx2: edge.x, ly2: edge.y,
      tx: lbl.x + 47, ty: lbl.y + 4,
    })
  }
  return out
})

// --- Drag interaction ---
function startDrag(e: MouseEvent) {
  dragging.value = true; dragX0.value = e.clientX; dragAz0.value = azimuth.value
}
function onDrag(e: MouseEvent) {
  if (!dragging.value) return
  azimuth.value = dragAz0.value + (e.clientX - dragX0.value) * 0.5
}
function endDrag() { dragging.value = false }
function startTouch(e: TouchEvent) {
  if (e.touches.length !== 1) return
  dragging.value = true; dragX0.value = e.touches[0].clientX; dragAz0.value = azimuth.value
}
function onTouch(e: TouchEvent) {
  if (!dragging.value || e.touches.length !== 1) return
  azimuth.value = dragAz0.value + (e.touches[0].clientX - dragX0.value) * 0.5
}
</script>

<style scoped>
.pixel-3d-viewer { padding: 16px 0; }
.controls-row {
  display: flex; align-items: flex-start; gap: 16px;
  margin-bottom: 12px; flex-wrap: wrap;
}
.ctrl-item {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.88em; font-weight: 600; color: var(--vp-c-text-1);
  cursor: pointer; white-space: nowrap;
}
.ctrl-item input[type="checkbox"] { accent-color: var(--vp-c-brand-1); }
.layer-toggles { display: flex; flex-wrap: wrap; gap: 6px 14px; }
.toggle-item {
  display: flex; align-items: center; gap: 4px;
  font-size: 0.82em; color: var(--vp-c-text-2); cursor: pointer;
}
.toggle-item input[type="checkbox"] { accent-color: var(--vp-c-brand-1); width: 14px; height: 14px; }
.toggle-sw { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--vp-c-divider); }
.toggle-lbl { white-space: nowrap; }
.viewer-svg {
  width: 100%; max-width: 640px; height: auto;
  display: block; margin: 0 auto;
  cursor: grab; user-select: none; -webkit-user-select: none;
}
.viewer-svg:active { cursor: grabbing; }
.label-3d { font-size: 9.5px; fill: var(--vp-c-text-2); font-weight: 500; pointer-events: none; }
.hint { font-size: 11px; fill: var(--vp-c-text-3); font-style: italic; }
</style>
