<template>
  <div class="raytrace-container">
    <h4>{{ t('Microlens Ray Tracing Simulator', '마이크로렌즈 광선 추적 시뮬레이터') }}</h4>
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
        <input type="range" min="0.7" max="2.0" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Lens height', '렌즈 높이') }}: <strong>{{ lensHeight.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.1" max="1.0" step="0.01" v-model.number="lensHeight" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Lens radius R', '렌즈 반경 R') }}: <strong>{{ lensRadius.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.2" max="1.0" step="0.01" v-model.number="lensRadius" class="ctrl-range" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Shape param n', '형상 파라미터 n') }}: <strong>{{ shapeN.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.5" max="6.0" step="0.1" v-model.number="shapeN" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Planarization', '평탄화층') }}: <strong>{{ planThickness.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.05" max="0.8" step="0.01" v-model.number="planThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Color filter', '컬러 필터') }}: <strong>{{ cfThickness.toFixed(2) }} um</strong>
        </label>
        <input type="range" min="0.1" max="1.5" step="0.01" v-model.number="cfThickness" class="ctrl-range" />
      </div>
    </div>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('CRA angle', 'CRA 각도') }}: <strong>{{ cra }}&deg;</strong>
        </label>
        <input type="range" min="0" max="30" step="1" v-model.number="cra" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Number of rays', '광선 수') }}: <strong>{{ numRays }}</strong>
        </label>
        <input type="range" min="5" max="30" step="1" v-model.number="numRays" class="ctrl-range" />
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
          <!-- Refracted segment (below lens) -->
          <line
            :x1="ray.hitX" :y1="ray.hitY"
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
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// Controls
const pitch = ref(1.0)
const lensHeight = ref(0.6)
const lensRadius = ref(0.48)
const shapeN = ref(2.5)
const cra = ref(0)
const numRays = ref(15)

// Layer thicknesses (um)
const siThickness = 3.0
const barlThickness = 0.15
const cfThickness = ref(0.8)
const planThickness = ref(0.3)

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

// Lens dome SVG paths
const lensDomes = computed(() => {
  const h = lensHeight.value
  const R = lensRadius.value
  const n = shapeN.value
  const lensBase = planTop.value

  const domes: string[] = []
  for (let lensIdx = 0; lensIdx < 2; lensIdx++) {
    const cx = pitch.value * (lensIdx + 0.5)
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
  const pdHeight = 2.0
  const pdBottom = 0.3

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
  endX: number
  endY: number
  hitsTarget: boolean
}

const tracedRays = computed((): TracedRay[] => {
  const h = lensHeight.value
  const R = lensRadius.value
  const n = shapeN.value
  const lensBase = planTop.value
  const craRad = (cra.value * Math.PI) / 180
  const nLens = 1.56
  const nAir = 1.0

  // Ray direction (unit vector): tilted right by CRA from vertical downward
  const dx = Math.sin(craRad)
  const dz = -Math.cos(craRad)

  const rays: TracedRay[] = []
  const nR = numRays.value
  const pdWidth = 0.7 * pitch.value
  const pdBottom = 0.3
  const pdTop = pdBottom + 2.0
  const pd0Left = pitch.value * 0.5 - pdWidth / 2
  const pd0Right = pitch.value * 0.5 + pdWidth / 2

  for (let i = 0; i < nR; i++) {
    // Distribute rays across left pixel (0 to pitch), with some margin
    const xStart = (pitch.value * (i + 0.5)) / nR
    const zStart = totalHeight.value - 0.1

    // Find intersection with either microlens dome
    let hitX = -1
    let hitZ = -1
    let hitLens = -1

    for (let lensIdx = 0; lensIdx < 2; lensIdx++) {
      const cx = pitch.value * (lensIdx + 0.5)

      // March along the ray from top downward to find lens surface intersection
      // Ray parametric: x(t) = xStart + t*dx, z(t) = zStart + t*dz
      // We need to find when z(t) = lensBase + superellipseZ(x(t))
      const tMax = (zStart - lensBase + 1) / Math.abs(dz)
      const dt = 0.005

      for (let step = 0; step * dt <= tMax; step++) {
        const tt = step * dt
        const rx = xStart + tt * dx
        const rz = zStart + tt * dz
        const lensZ = lensBase + superellipseZ(rx, cx, h, R, n)

        if (rz <= lensZ) {
          // Bisection refinement
          let tLow = Math.max(0, (step - 1) * dt)
          let tHigh = tt
          for (let b = 0; b < 20; b++) {
            const tMid = (tLow + tHigh) / 2
            const mx = xStart + tMid * dx
            const mz = zStart + tMid * dz
            const mlz = lensBase + superellipseZ(mx, cx, h, R, n)
            if (mz <= mlz) {
              tHigh = tMid
            } else {
              tLow = tMid
            }
          }
          const tFinal = (tLow + tHigh) / 2
          const fx = xStart + tFinal * dx
          const fz = zStart + tFinal * dz
          // Only accept if within this lens region
          if (Math.abs(fx - cx) < R) {
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

    // If no lens hit, ray goes straight through
    if (hitZ < 0) {
      // No lens intersection, straight ray to bottom
      const tBottom = (zStart - 0) / Math.abs(dz)
      const endXPhys = xStart + tBottom * dx
      const hitsTarget = endXPhys >= pd0Left && endXPhys <= pd0Right
      rays.push({
        startX: toSvgX(xStart),
        startY: toSvgY(zStart),
        hitX: toSvgX(xStart + (zStart - lensBase) / Math.abs(dz) * dx),
        hitY: toSvgY(lensBase),
        endX: toSvgX(endXPhys),
        endY: toSvgY(0),
        hitsTarget,
      })
      continue
    }

    // Compute refracted direction using vector Snell's law
    const cx = pitch.value * (hitLens + 0.5)
    const dzdx = superellipseDzDx(hitX, cx, h, R, n)

    // Surface tangent: (1, dzdx), Normal pointing outward (into air): (-dzdx, 1), normalized
    const nLen = Math.sqrt(dzdx * dzdx + 1)
    let normX = -dzdx / nLen
    let normZ = 1 / nLen

    // Ensure normal points towards the incoming ray (into air, upward)
    // Dot product of normal with incident direction should be negative
    const dotDN = dx * normX + dz * normZ
    if (dotDN > 0) {
      normX = -normX
      normZ = -normZ
    }

    const cosI = -(dx * normX + dz * normZ)
    const ratio = nAir / nLens
    const sin2T = ratio * ratio * (1 - cosI * cosI)

    let refDx: number
    let refDz: number

    if (sin2T > 1) {
      // Total internal reflection (shouldn't normally happen air->lens)
      refDx = dx
      refDz = dz
    } else {
      const cosT = Math.sqrt(1 - sin2T)
      refDx = ratio * dx + (ratio * cosI - cosT) * normX
      refDz = ratio * dz + (ratio * cosI - cosT) * normZ
      // Normalize
      const rl = Math.sqrt(refDx * refDx + refDz * refDz)
      refDx /= rl
      refDz /= rl
    }

    // Trace refracted ray to the bottom of silicon (z=0)
    // Compute t where hitZ + t*refDz = 0
    let endXPhys: number
    let endZPhys: number
    if (Math.abs(refDz) < 1e-10) {
      // Nearly horizontal, just extend
      endXPhys = hitX + refDx * 10
      endZPhys = 0
    } else {
      const tEnd = -hitZ / refDz
      if (tEnd < 0) {
        // Ray going upward after refraction, extend downward anyway
        const tDown = hitZ / Math.abs(refDz)
        endXPhys = hitX + refDx * tDown
        endZPhys = 0
      } else {
        endXPhys = hitX + tEnd * refDx
        endZPhys = hitZ + tEnd * refDz
      }
    }

    // Clamp end position to silicon bottom
    if (endZPhys > 0 && refDz < 0) {
      const tBottom = (hitZ - 0) / Math.abs(refDz)
      endXPhys = hitX + tBottom * refDx
      endZPhys = 0
    } else if (endZPhys < 0) {
      endZPhys = 0
    }

    // Check if ray hits PD0
    const hitsTarget = endXPhys >= pd0Left && endXPhys <= pd0Right && endZPhys <= pdTop

    rays.push({
      startX: toSvgX(xStart),
      startY: toSvgY(zStart),
      hitX: toSvgX(hitX),
      hitY: toSvgY(hitZ),
      endX: toSvgX(endXPhys),
      endY: toSvgY(Math.max(0, endZPhys)),
      hitsTarget,
    })
  }

  return rays
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
    // Convert end SVG position back to physical x
    const physX = ((ray.endX - margin) / drawWidth.value) * 2 * pitch.value
    if (physX >= pd1Left && physX <= pd1Right) {
      count++
    }
  }
  return count
})

const focalLength = computed(() => {
  // Approximate focal length: f = h * n_lens / (n_lens - 1)
  const nLens = 1.56
  return lensHeight.value * nLens / (nLens - 1)
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
