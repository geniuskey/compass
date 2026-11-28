<template>
  <div :class="['raytrace-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Microlens Ray Tracing Simulator', '마이크로렌즈 광선 추적 시뮬레이터') }}</h4>
    <button
      type="button"
      class="sim-fs-btn"
      :aria-label="t('Toggle fullscreen', '전체화면 전환')"
      :aria-pressed="isFullscreen"
      :title="t('Toggle fullscreen', '전체화면 전환')"
      @click="toggleFullscreen"
    >{{ isFullscreen ? '\u00d7' : '\u26f6' }}</button>
    <p class="component-description">
      {{ t(
        'Trace rays through a superellipse microlens onto a pixel photodiode. Adjust lens geometry and CRA to observe focusing and crosstalk.',
        '초타원 마이크로렌즈를 통한 광선 추적으로 픽셀 포토다이오드를 시뮬레이션합니다. 렌즈 형상과 CRA를 조절하여 집광 및 크로스토크를 관찰하세요.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Pixel pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.7" max="2.0" step="0.05" v-model.number="pitch" class="ctrl-range" :aria-label="t('Pixel pitch', '픽셀 피치')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Lens height', '렌즈 높이') }}: <strong>{{ lensHeight.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.1" max="1.0" step="0.01" v-model.number="lensHeight" class="ctrl-range" :aria-label="t('Lens height', '렌즈 높이')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Lens radius R', '렌즈 반경 R') }}: <strong>{{ lensRadius.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.2" :max="lensRadiusMax" step="0.01" v-model.number="lensRadius" class="ctrl-range" :aria-label="t('Lens radius R', '렌즈 반경 R')" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Shape param n', '형상 파라미터 n') }}: <strong>{{ shapeN.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.5" max="6.0" step="0.1" v-model.number="shapeN" class="ctrl-range" :aria-label="t('Shape param n', '형상 파라미터 n')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Planarization', '평탄화층') }}: <strong>{{ planThickness.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.05" max="0.8" step="0.01" v-model.number="planThickness" class="ctrl-range" :aria-label="t('Planarization', '평탄화층')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Color filter', '컬러 필터') }}: <strong>{{ cfThickness.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.1" max="1.5" step="0.01" v-model.number="cfThickness" class="ctrl-range" :aria-label="t('Color filter', '컬러 필터')" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('CRA angle', 'CRA 각도') }}: <strong>{{ cra }}&deg;</strong>
        </label>
        <input type="range" min="0" max="30" step="1" v-model.number="cra" class="ctrl-range" :aria-label="t('CRA angle', 'CRA 각도')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Microlens shift', '마이크로렌즈 시프트') }}: <strong>{{ lensShift.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0" :max="lensShiftMax" step="0.01" v-model.number="lensShift" class="ctrl-range" :aria-label="t('Microlens shift', '마이크로렌즈 시프트')" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Number of rays', '광선 수') }}: <strong>{{ numRays }}</strong>
        </label>
        <input type="range" min="5" max="30" step="1" v-model.number="numRays" class="ctrl-range" :aria-label="t('Number of rays', '광선 수')" />
      </div>
      <div class="slider-group reset-group">
        <button type="button" class="reset-btn" @click="resetControls">{{ t('Reset', '초기화') }}</button>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="raytrace-svg">
        <defs>
          <marker id="rtArrowGreen" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#27ae60" opacity="0.7" />
          </marker>
          <marker id="rtArrowRed" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
            <polygon points="0 0, 6 2, 0 4" fill="#e74c3c" opacity="0.7" />
          </marker>
        </defs>

        <!-- Background -->
        <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />

        <!-- Air region label -->
        <text :x="svgW - 10" :y="toSvgY(totalHeight - 0.2)" text-anchor="end" class="layer-label-text">
          {{ t('Air', '공기') }}
        </text>

        <!-- Silicon layer -->
        <rect
          :x="margin"
          :y="toSvgY(siThickness)"
          :width="drawWidth"
          :height="toSvgY(0) - toSvgY(siThickness)"
          fill="#5d6d7e"
          opacity="0.6"
        />
        <text :x="margin + 6" :y="toSvgY(siThickness / 2)" class="layer-label-text" fill="#fff">
          {{ t('Silicon', '실리콘') }}
        </text>

        <!-- BARL zone -->
        <rect
          :x="margin"
          :y="toSvgY(siThickness + barlThickness)"
          :width="drawWidth"
          :height="toSvgY(siThickness) - toSvgY(siThickness + barlThickness)"
          fill="#8e44ad"
          opacity="0.5"
        />

        <!-- Color filter -->
        <rect
          :x="margin"
          :y="toSvgY(siThickness + barlThickness + cfThickness)"
          :width="drawWidth"
          :height="toSvgY(siThickness + barlThickness) - toSvgY(siThickness + barlThickness + cfThickness)"
          fill="#27ae60"
          opacity="0.35"
        />
        <text :x="margin + 6" :y="toSvgY(siThickness + barlThickness + cfThickness / 2)" class="layer-label-text" fill="#27ae60">
          {{ t('Color filter', '컬러 필터') }}
        </text>

        <!-- Planarization -->
        <rect
          :x="margin"
          :y="toSvgY(planTop)"
          :width="drawWidth"
          :height="toSvgY(siThickness + barlThickness + cfThickness) - toSvgY(planTop)"
          fill="#d5dbdb"
          opacity="0.4"
        />

        <!-- DTI wall -->
        <line
          :x1="toSvgX(pitch)"
          :y1="toSvgY(0)"
          :x2="toSvgX(pitch)"
          :y2="toSvgY(siThickness)"
          stroke="#fff"
          stroke-width="2"
          opacity="0.8"
        />
        <text :x="toSvgX(pitch) + 3" :y="toSvgY(siThickness * 0.7)" class="tiny-label" fill="#fff">DTI</text>

        <!-- Photodiode boxes -->
        <rect
          v-for="(pd, i) in photodiodes"
          :key="'pd-' + i"
          :x="pd.x"
          :y="pd.y"
          :width="pd.w"
          :height="pd.h"
          fill="none"
          stroke="#f1c40f"
          stroke-width="1.5"
          stroke-dasharray="4,3"
        />
        <text
          v-for="(pd, i) in photodiodes"
          :key="'pdlabel-' + i"
          :x="pd.x + pd.w / 2"
          :y="pd.y + pd.h / 2 + 3"
          text-anchor="middle"
          class="tiny-label"
          fill="#f1c40f"
        >PD{{ i }}</text>

        <!-- Collection test depth (PD mid-depth) -->
        <line
          :x1="margin" :y1="toSvgY(pdTestZ)" :x2="margin + drawWidth" :y2="toSvgY(pdTestZ)"
          stroke="#f1c40f" stroke-width="0.6" stroke-dasharray="2,3" opacity="0.6"
        />

        <!-- Microlens domes -->
        <path
          v-for="(dome, i) in lensDomes"
          :key="'dome-' + i"
          :d="dome"
          fill="#dda0dd"
          fill-opacity="0.5"
          stroke="#b07eb0"
          stroke-width="1.5"
        />

        <!-- Rays -->
        <template v-for="(ray, i) in tracedRays" :key="'ray-' + i">
          <!-- Incident segment (above lens) -->
          <line
            :x1="ray.startX" :y1="ray.startY"
            :x2="ray.hitX" :y2="ray.hitY"
            :stroke="ray.hitsTarget ? '#27ae60' : '#e74c3c'"
            stroke-width="1.5"
            :opacity="0.6"
          />
          <!-- Refracted segment in the stack (lens → Si surface) -->
          <line
            :x1="ray.hitX" :y1="ray.hitY"
            :x2="ray.siX" :y2="ray.siY"
            :stroke="ray.hitsTarget ? '#27ae60' : '#e74c3c'"
            stroke-width="1.5"
            :opacity="0.6"
          />
          <!-- Refracted segment inside silicon -->
          <line
            :x1="ray.siX" :y1="ray.siY"
            :x2="ray.endX" :y2="ray.endY"
            :stroke="ray.hitsTarget ? '#27ae60' : '#e74c3c'"
            stroke-width="1.5"
            :opacity="0.6"
            :marker-end="ray.hitsTarget ? 'url(#rtArrowGreen)' : 'url(#rtArrowRed)'"
          />
          <!-- Intersection point -->
          <circle
            :cx="ray.hitX" :cy="ray.hitY"
            r="2.5"
            :fill="ray.hitsTarget ? '#27ae60' : '#e74c3c'"
            opacity="0.8"
          />
        </template>

        <!-- Legend -->
        <line :x1="svgW - 130" y1="16" :x2="svgW - 110" y2="16" stroke="#27ae60" stroke-width="2" opacity="0.7" />
        <text :x="svgW - 106" y="20" class="legend-label">{{ t('Collected', '수집') }}</text>
        <line :x1="svgW - 130" y1="30" :x2="svgW - 110" y2="30" stroke="#e74c3c" stroke-width="2" opacity="0.7" />
        <text :x="svgW - 106" y="34" class="legend-label">{{ t('Lost / Crosstalk', '손실 / 크로스토크') }}</text>

        <!-- Pixel boundary labels -->
        <text :x="toSvgX(pitch / 2)" :y="toSvgY(-0.3)" text-anchor="middle" class="pixel-label">
          {{ t('Pixel 0', '픽셀 0') }}
        </text>
        <text :x="toSvgX(pitch * 1.5)" :y="toSvgY(-0.3)" text-anchor="middle" class="pixel-label">
          {{ t('Pixel 1', '픽셀 1') }}
        </text>
      </svg>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Collection efficiency', '수집 효율') }}</span>
        <span class="info-value">{{ collectionEfficiency.toFixed(1) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Focal length (est.)', '초점 거리 (추정)') }}</span>
        <span class="info-value">{{ focalLength.toFixed(2) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Lens f-number', '렌즈 f-수') }}</span>
        <span class="info-value">f/{{ fNumber.toFixed(2) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Crosstalk rays', '크로스토크 광선') }}</span>
        <span class="info-value">{{ crosstalkCount }}</span>
      </div>
    </div>
    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Traced focus (z, x)', '추적 초점 (z, x)') }}</span>
        <span class="info-value">{{ tracedFocus ? `${tracedFocus.z.toFixed(2)}, ${tracedFocus.x.toFixed(2)} um` : '—' }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('PD span (z)', 'PD 범위 (z)') }}</span>
        <span class="info-value">{{ pdBottom.toFixed(1) }}–{{ pdTop.toFixed(1) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Focus vs PD top', '초점 − PD 상단') }}</span>
        <span class="info-value">{{ tracedFocus ? (tracedFocus.z - pdTop >= 0 ? '+' : '') + (tracedFocus.z - pdTop).toFixed(2) + ' um' : '—' }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Focus offset from PD0 centre', 'PD0 중심 대비 초점 오프셋') }}</span>
        <span class="info-value">{{ tracedFocus ? (tracedFocus.x - pitch / 2).toFixed(2) + ' um' : '—' }}</span>
      </div>
    </div>
    <p class="rt-note">
      {{ t(
        'Rays refract at the lens (n = 1.56; the stack above Si is treated as index-matched) and again at the Si surface (n ≈ ' + nSi.toFixed(2) + ' at 550 nm). Collection is tested at the PD mid-depth (dotted line). Shift the microlens toward the light source to compensate CRA.',
        '광선은 렌즈(n = 1.56; Si 위 스택은 굴절률 정합으로 가정)와 Si 표면(550 nm에서 n ≈ ' + nSi.toFixed(2) + ')에서 굴절됩니다. 수집 여부는 PD 중간 깊이(점선)에서 판정합니다. CRA 보상을 위해 마이크로렌즈를 광원 쪽으로 시프트하세요.'
      ) }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useLocale } from '../composables/useLocale'
import { getN, MATERIALS } from '../composables/tmm'
import { useFullscreen } from '../composables/useFullscreen'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

// Controls
const pitch = ref(1.0)
const lensHeight = ref(0.6)
const lensRadius = ref(0.48)
const shapeN = ref(2.5)
const cra = ref(0)
const numRays = ref(15)
const lensShift = ref(0)

// Lens footprint must fit inside the pixel: R ≤ pitch / 2
const lensRadiusMax = computed(() => Math.round((pitch.value / 2) * 100) / 100)
const lensShiftMax = computed(() => Math.round((pitch.value / 2) * 100) / 100)
watch(pitch, () => {
  if (lensRadius.value > lensRadiusMax.value) lensRadius.value = lensRadiusMax.value
  if (lensShift.value > lensShiftMax.value) lensShift.value = lensShiftMax.value
})

// Silicon index at 550 nm (component has no wavelength control)
const nSi = getN(MATERIALS.silicon, 0.55)[0]
const nLens = 1.56

// Photodiode geometry (um, z from Si bottom)
const pdBottom = 0.3
const pdHeight = 2.0
const pdTop = pdBottom + pdHeight
const pdTestZ = pdBottom + pdHeight / 2

// Layer thicknesses (um)
const siThickness = 3.0
const barlThickness = 0.15
const cfThickness = ref(0.8)
const planThickness = ref(0.3)

function resetControls() {
  pitch.value = 1.0
  lensHeight.value = 0.6
  lensRadius.value = 0.48
  shapeN.value = 2.5
  cra.value = 0
  numRays.value = 15
  lensShift.value = 0
  cfThickness.value = 0.8
  planThickness.value = 0.3
}

const planTop = computed(() => siThickness + barlThickness + cfThickness.value + planThickness.value)

// Total height includes lens and some air above
const totalHeight = computed(() => planTop.value + lensHeight.value + 1.5)

// SVG dimensions
const svgW = 560
const svgH = 450
const margin = 30
const topMargin = 30
const bottomMargin = 30

const drawWidth = computed(() => svgW - 2 * margin)

// Physical x-range: 0 to 2*pitch
// Physical z-range: 0 to totalHeight
function toSvgX(physX: number): number {
  return margin + (physX / (2 * pitch.value)) * drawWidth.value
}

function toSvgY(physZ: number): number {
  const drawH = svgH - topMargin - bottomMargin
  return svgH - bottomMargin - (physZ / totalHeight.value) * drawH
}

// Superellipse z(x) relative to lens base
function superellipseZ(x: number, cx: number, h: number, R: number, n: number): number {
  const dx = Math.abs((x - cx) / R)
  if (dx >= 1) return 0
  return h * Math.pow(1 - Math.pow(dx, n), 1 / n)
}

// Derivative dz/dx of superellipse
function superellipseDzDx(x: number, cx: number, h: number, R: number, n: number): number {
  const xr = (x - cx) / R
  const absXr = Math.abs(xr)
  if (absXr < 1e-10 || absXr >= 0.999) return 0
  const term1 = Math.pow(absXr, n - 2)
  const term2 = Math.pow(1 - Math.pow(absXr, n), 1 / n - 1)
  const dz = -h * (xr / (R)) * term1 * term2
  return dz
}

// Lens centre; shifted toward −x (toward the incoming chief ray) by lensShift
function lensCenter(idx: number): number {
  return pitch.value * (idx + 0.5) - lensShift.value
}

// Lens dome SVG paths
const lensDomes = computed(() => {
  const h = lensHeight.value
  const R = lensRadius.value
  const n = shapeN.value
  const lensBase = planTop.value

  const domes: string[] = []
  for (let lensIdx = 0; lensIdx < 2; lensIdx++) {
    const cx = lensCenter(lensIdx)
    const pts: Array<{ x: number; y: number }> = []
    const numPts = 100
    for (let i = 0; i <= numPts; i++) {
      const px = cx - R + (2 * R * i) / numPts
      const pz = superellipseZ(px, cx, h, R, n)
      pts.push({ x: toSvgX(px), y: toSvgY(lensBase + pz) })
    }
    // Close the path along the base
    let d = `M ${toSvgX(cx - R).toFixed(1)} ${toSvgY(lensBase).toFixed(1)}`
    for (const pt of pts) {
      d += ` L ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`
    }
    d += ` L ${toSvgX(cx + R).toFixed(1)} ${toSvgY(lensBase).toFixed(1)} Z`
    domes.push(d)
  }
  return domes
})

// Photodiode rectangles
const photodiodes = computed(() => {
  const pdWidth = 0.7 * pitch.value

  const pds = []
  for (let i = 0; i < 2; i++) {
    const cx = pitch.value * (i + 0.5)
    pds.push({
      x: toSvgX(cx - pdWidth / 2),
      y: toSvgY(pdBottom + pdHeight),
      w: toSvgX(cx + pdWidth / 2) - toSvgX(cx - pdWidth / 2),
      h: toSvgY(pdBottom) - toSvgY(pdBottom + pdHeight),
    })
  }
  return pds
})

// Ray tracing
interface TracedRay {
  startX: number
  startY: number
  hitX: number
  hitY: number
  siX: number
  siY: number
  endX: number
  endY: number
  hitsTarget: boolean
  testX: number
}

// Vector Snell refraction of unit direction d at a surface with unit normal nrm
// (any orientation), from index n1 into n2. Returns null on TIR.
function refract(dx: number, dz: number, nx: number, nz: number, n1: number, n2: number):
  [number, number] | null {
  if (dx * nx + dz * nz > 0) { nx = -nx; nz = -nz }
  const cosI = -(dx * nx + dz * nz)
  const ratio = n1 / n2
  const sin2T = ratio * ratio * (1 - cosI * cosI)
  if (sin2T > 1) return null
  const cosT = Math.sqrt(1 - sin2T)
  let rx = ratio * dx + (ratio * cosI - cosT) * nx
  let rz = ratio * dz + (ratio * cosI - cosT) * nz
  const rl = Math.hypot(rx, rz)
  rx /= rl; rz /= rl
  return [rx, rz]
}

interface PhysRay {
  xStart: number; zStart: number
  hitX: number; hitZ: number; hitLens: number
  mDx: number; mDz: number // direction in the stack (n = nLens)
  siX: number; siDx: number; siDz: number // entry x at Si top and direction in Si
  testX: number; endX: number
}

const physRays = computed((): PhysRay[] => {
  const h = lensHeight.value
  const R = lensRadius.value
  const n = shapeN.value
  const lensBase = planTop.value
  const craRad = (cra.value * Math.PI) / 180
  const nAir = 1.0

  // Ray direction (unit vector): tilted right by CRA from vertical downward
  const dx = Math.sin(craRad)
  const dz = -Math.cos(craRad)

  const rays: PhysRay[] = []
  const nR = numRays.value

  for (let i = 0; i < nR; i++) {
    // Distribute rays across left pixel (0 to pitch)
    const xStart = (pitch.value * (i + 0.5)) / nR
    const zStart = totalHeight.value - 0.1

    let hitX = -1
    let hitZ = -1
    let hitLens = -1

    for (let lensIdx = 0; lensIdx < 2; lensIdx++) {
      const cx = lensCenter(lensIdx)
      const tMax = (zStart - lensBase + 1) / Math.abs(dz)
      const dt = 0.005

      for (let step = 0; step * dt <= tMax; step++) {
        const tt = step * dt
        const rx = xStart + tt * dx
        const rz = zStart + tt * dz
        const lensZ = lensBase + superellipseZ(rx, cx, h, R, n)

        if (rz <= lensZ) {
          let tLow = Math.max(0, (step - 1) * dt)
          let tHigh = tt
          for (let b = 0; b < 20; b++) {
            const tMid = (tLow + tHigh) / 2
            const mx = xStart + tMid * dx
            const mz = zStart + tMid * dz
            const mlz = lensBase + superellipseZ(mx, cx, h, R, n)
            if (mz <= mlz) tHigh = tMid
            else tLow = tMid
          }
          const tFinal = (tLow + tHigh) / 2
          const fx = xStart + tFinal * dx
          const fz = zStart + tFinal * dz
          if (Math.abs(fx - cx) < R && fz > lensBase + 1e-6) {
            if (hitZ < 0 || tFinal < ((hitZ - zStart) / dz)) {
              hitX = fx
              hitZ = fz
              hitLens = lensIdx
            }
          }
          break
        }
      }
    }

    // Refraction into the lens / planarization medium
    let mDx: number, mDz: number
    if (hitZ < 0) {
      // Missed both domes: flat planarization surface at lensBase
      const tTop = (zStart - lensBase) / Math.abs(dz)
      hitX = xStart + tTop * dx
      hitZ = lensBase
      const r = refract(dx, dz, 0, 1, nAir, nLens) ?? [dx, dz]
      ;[mDx, mDz] = r
    } else {
      const cx = lensCenter(hitLens)
      const dzdx = superellipseDzDx(hitX, cx, h, R, n)
      const nLen = Math.sqrt(dzdx * dzdx + 1)
      const r = refract(dx, dz, -dzdx / nLen, 1 / nLen, nAir, nLens) ?? [dx, dz]
      ;[mDx, mDz] = r
    }
    if (mDz > -1e-6) { mDz = -1e-6 } // keep marching downward

    // Propagate to the Si surface (z = siThickness) and refract into silicon
    const tSi = (hitZ - siThickness) / -mDz
    const siX = hitX + tSi * mDx
    const rSi = refract(mDx, mDz, 0, 1, nLens, nSi) ?? [mDx, mDz]
    const [siDx, siDz] = rSi

    // Collection is tested at the PD mid-depth; drawing continues to Si bottom
    const testX = siX + ((siThickness - pdTestZ) / -siDz) * siDx
    const endX = siX + (siThickness / -siDz) * siDx

    rays.push({ xStart, zStart, hitX, hitZ, hitLens, mDx, mDz, siX, siDx, siDz, testX, endX })
  }
  return rays
})

const tracedRays = computed((): TracedRay[] => {
  const pdWidth = 0.7 * pitch.value
  const pd0Left = pitch.value * 0.5 - pdWidth / 2
  const pd0Right = pitch.value * 0.5 + pdWidth / 2
  return physRays.value.map((r) => ({
    startX: toSvgX(r.xStart),
    startY: toSvgY(r.zStart),
    hitX: toSvgX(r.hitX),
    hitY: toSvgY(r.hitZ),
    siX: toSvgX(r.siX),
    siY: toSvgY(siThickness),
    endX: toSvgX(r.endX),
    endY: toSvgY(0),
    hitsTarget: r.testX >= pd0Left && r.testX <= pd0Right,
    testX: r.testX,
  }))
})

// Least-squares point closest to a set of 2D lines (point p, unit direction d)
function lsCrossing(lines: Array<{ px: number; pz: number; dx: number; dz: number }>):
  { x: number; z: number } | null {
  if (lines.length < 2) return null
  let a11 = 0, a12 = 0, a22 = 0, b1 = 0, b2 = 0
  for (const l of lines) {
    const m11 = 1 - l.dx * l.dx, m12 = -l.dx * l.dz, m22 = 1 - l.dz * l.dz
    a11 += m11; a12 += m12; a22 += m22
    b1 += m11 * l.px + m12 * l.pz
    b2 += m12 * l.px + m22 * l.pz
  }
  const det = a11 * a22 - a12 * a12
  if (Math.abs(det) < 1e-12) return null
  return { x: (a22 * b1 - a12 * b2) / det, z: (a11 * b2 - a12 * b1) / det }
}

// Focus of lens 0 from the actually traced rays: first in the stack medium;
// if that lies below the Si surface, re-solve with the rays refracted into Si.
const tracedFocus = computed(() => {
  const lens0 = physRays.value.filter((r) => r.hitLens === 0)
  if (lens0.length < 2) return null
  const inStack = lsCrossing(lens0.map((r) => ({ px: r.hitX, pz: r.hitZ, dx: r.mDx, dz: r.mDz })))
  if (!inStack) return null
  if (inStack.z >= siThickness) return inStack
  return lsCrossing(lens0.map((r) => ({ px: r.siX, pz: siThickness, dx: r.siDx, dz: r.siDz })))
})

// Statistics
const collectionEfficiency = computed(() => {
  if (tracedRays.value.length === 0) return 0
  const hits = tracedRays.value.filter((r) => r.hitsTarget).length
  return (hits / tracedRays.value.length) * 100
})

const crosstalkCount = computed(() => {
  // Rays that land in the neighbor pixel photodiode area
  const pdWidth = 0.7 * pitch.value
  const pd1Left = pitch.value * 1.5 - pdWidth / 2
  const pd1Right = pitch.value * 1.5 + pdWidth / 2

  let count = 0
  for (const ray of tracedRays.value) {
    if (ray.testX >= pd1Left && ray.testX <= pd1Right) {
      count++
    }
  }
  return count
})

const focalLength = computed(() => {
  // Best-fit spherical cap through apex and rim: R_c = (a² + h²) / 2h.
  // Single refracting surface with the image formed inside the lens medium
  // (object at infinity): f = n_lens · R_c / (n_lens − 1), measured from the vertex.
  const h = Math.max(lensHeight.value, 1e-6)
  const a = lensRadius.value
  const Rc = (a * a + h * h) / (2 * h)
  return nLens * Rc / (nLens - 1)
})

const fNumber = computed(() => {
  return focalLength.value / (2 * lensRadius.value)
})
</script>

<style scoped>
.raytrace-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.raytrace-container h4 {
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
.reset-group {
  display: flex;
  align-items: flex-end;
}
.reset-btn {
  padding: 4px 12px;
  font-size: 0.8em;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.reset-btn:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}
.svg-wrapper {
  display: flex;
  justify-content: center;
  margin: 12px 0;
}
.raytrace-svg {
  width: 100%;
  max-width: 580px;
}
.layer-label-text {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.tiny-label {
  font-size: 7px;
  font-weight: 600;
}
.pixel-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.rt-note {
  margin: 8px 0 0;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
}
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
  flex: 1;
  min-width: 120px;
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
</style>
