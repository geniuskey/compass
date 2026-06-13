<template>
  <figure class="section-view" :aria-label="meta.aria">
    <div class="section-view-head">
      <strong>{{ meta.title }}&nbsp;</strong>
      <span>{{ meta.note }}</span>
    </div>

    <svg viewBox="0 0 640 270" class="section-svg" role="img">
      <defs>
        <marker :id="arrowId" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#2563eb" />
        </marker>
        <radialGradient :id="lensGradId" cx="45%" cy="40%" r="65%">
          <stop offset="0%" stop-color="#f4fbff" />
          <stop offset="100%" stop-color="#75c7e8" />
        </radialGradient>
        <pattern :id="filmPatternId" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="9" height="9" fill="#fff3c4" />
          <rect width="3" height="9" fill="#f8d76d" opacity="0.72" />
        </pattern>
        <clipPath :id="clipId">
          <rect x="0" y="0" :width="plot.size" :height="plot.size" />
        </clipPath>
      </defs>

      <rect x="0" y="0" width="640" height="270" rx="8" class="canvas" />

      <g :transform="`translate(${plot.x} ${plot.y})`">
        <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" class="domain" />

        <g :clip-path="`url(#${clipId})`">
          <template v-if="variant === 'pixel'">
            <rect
              v-for="cell in cells"
              :key="'pixel-' + cell.id"
              :x="rectX(cell.x0)"
              :y="rectY(cell.y1)"
              :width="rectW(cell.x0, cell.x1)"
              :height="rectH(cell.y0, cell.y1)"
              :fill="cell.fill"
              opacity="0.55"
            />
          </template>

          <template v-else-if="variant === 'air'">
            <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" fill="#e8f2ff" opacity="0.9" />
          </template>

          <template v-else-if="variant === 'microlens'">
            <path
              v-for="lens in microlenses"
              :key="'lens-' + lens.id"
              :d="superellipsePath(lens.cx, lens.cy, mlRadiusX, mlRadiusY, mlN)"
              :fill="`url(#${lensGradId})`"
              class="lens-footprint"
              stroke="#2f9bb8"
              stroke-width="1.5"
              opacity="0.92"
            />
            <path
              v-for="lens in microlenses"
              :key="'lens-shift-' + lens.id"
              :d="superellipsePath(lens.cx + illustrativeCraShift, lens.cy, mlRadiusX, mlRadiusY, mlN)"
              class="shifted-lens"
              fill="none"
              stroke="#f59e0b"
              stroke-width="1.3"
              stroke-dasharray="5 4"
              opacity="0.6"
            />
            <path
              :d="superellipsePath(1, 1, 0.98, 0.98, mlN)"
              class="shared-lens"
              fill="none"
              stroke="#2563eb"
              stroke-width="1.3"
              stroke-dasharray="8 5"
              opacity="0.55"
            />
          </template>

          <template v-else-if="variant === 'planarization'">
            <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" fill="#d8edf8" opacity="0.88" />
          </template>

          <template v-else-if="variant === 'color_filter'">
            <rect x="0" y="0" :width="plot.size" :height="plot.size" fill="#d8edf8" opacity="0.72" />
            <rect
              v-for="cell in colorFilterCells"
              :key="'cf-base-' + cell.id"
              :x="rectX(cell.base.x0)"
              :y="rectY(cell.base.y1)"
              :width="rectW(cell.base.x0, cell.base.x1)"
              :height="rectH(cell.base.y0, cell.base.y1)"
              :fill="cell.fill"
              opacity="0.82"
            />
            <rect
              v-for="cell in colorFilterCells"
              :key="'cf-top-' + cell.id"
              :x="rectX(cell.top.x0)"
              :y="rectY(cell.top.y1)"
              :width="rectW(cell.top.x0, cell.top.x1)"
              :height="rectH(cell.top.y0, cell.top.y1)"
              :fill="cell.fill"
              class="cf-top"
            />
            <rect
              v-for="grid in metalGridRects"
              :key="'metal-' + grid.id"
              :x="rectX(grid.x0)"
              :y="rectY(grid.y1)"
              :width="rectW(grid.x0, grid.x1)"
              :height="rectH(grid.y0, grid.y1)"
              class="metal-fill"
              fill="#4b5563"
              opacity="0.96"
            />
          </template>

          <template v-else-if="variant === 'barl'">
            <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" :fill="`url(#${filmPatternId})`" opacity="0.92" />
          </template>

          <template v-else>
            <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" fill="#cfd4dc" opacity="0.92" />
            <rect
              v-for="dti in dtiRects"
              :key="'dti-' + dti.id"
              :x="rectX(dti.x0)"
              :y="rectY(dti.y1)"
              :width="rectW(dti.x0, dti.x1)"
              :height="rectH(dti.y0, dti.y1)"
              class="dti-fill"
              fill="#dff3ff"
              stroke="#7db9d8"
              stroke-width="0.6"
              opacity="0.95"
            />
            <rect
              v-for="pd in photodiodes"
              :key="'pd-' + pd.id"
              :x="rectX(pd.x0)"
              :y="rectY(pd.y1)"
              :width="rectW(pd.x0, pd.x1)"
              :height="rectH(pd.y0, pd.y1)"
              rx="5"
              class="pd-fill"
              fill="#b85c5c"
              stroke="#8a3f3f"
              stroke-width="1"
              opacity="0.68"
            />
          </template>

          <path v-if="showPixelReference" :d="pixelBoundaryPath" class="pixel-reference" />
        </g>

        <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" class="domain-outline" />

        <text
          v-for="cell in visibleLabels"
          :key="'label-' + cell.id"
          :x="unitX(cell.cx)"
          :y="unitY(cell.cy) + 5"
          text-anchor="middle"
          class="cell-label"
          font-size="18"
          font-weight="760"
        >
          {{ cell.key }}
        </text>

        <template v-if="variant === 'pixel'">
          <circle cx="0" :cy="plot.size" r="4" class="origin-dot" />
          <text x="-5" :y="plot.size + 17" text-anchor="end" class="tiny" font-size="11" font-weight="620">origin</text>
          <line x1="0" :y1="plot.size + 26" :x2="plot.cell" :y2="plot.size + 26" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text :x="plot.cell / 2" :y="plot.size + 43" text-anchor="middle" class="label" font-size="12" font-weight="650">pitch</text>
          <line x1="-22" :y1="plot.size" x2="-22" :y2="plot.cell" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text x="-36" :y="plot.cell + 35" text-anchor="middle" class="label rotate-label" font-size="12" font-weight="650">pitch</text>
        </template>

        <template v-if="variant === 'microlens'">
          <line :x1="unitX(0.5)" :y1="unitY(0.5)" :x2="unitX(0.5 + illustrativeCraShift)" :y2="unitY(0.5)" class="shift" stroke="#f59e0b" stroke-width="1.7" stroke-dasharray="4 4" fill="none" :marker-end="arrowRef" />
          <text :x="unitX(0.5 + illustrativeCraShift) + 5" :y="unitY(0.5) - 6" class="tiny" font-size="11" font-weight="620">CRA shift</text>
          <line :x1="unitX(0.5)" :y1="unitY(0.5)" :x2="unitX(0.5 + mlRadiusX)" :y2="unitY(0.5)" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text :x="unitX(0.5 + mlRadiusX / 2)" :y="unitY(0.5) - 10" text-anchor="middle" class="tiny" font-size="11" font-weight="620">radius</text>
        </template>

        <template v-if="variant === 'color_filter'">
          <line :x1="unitX(1 - gridWidth / 2)" :y1="unitY(0.22)" :x2="unitX(1 + gridWidth / 2)" :y2="unitY(0.22)" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text :x="unitX(1)" :y="unitY(0.22) - 7" text-anchor="middle" class="tiny" font-size="11" font-weight="620">grid.width</text>
        </template>

        <template v-if="variant === 'silicon'">
          <line :x1="unitX(0.5 - pdSize / 2)" :y1="unitY(0.5) + pdSizePx / 2 + 11" :x2="unitX(0.5 + pdSize / 2)" :y2="unitY(0.5) + pdSizePx / 2 + 11" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text :x="unitX(0.5)" :y="unitY(0.5) + pdSizePx / 2 + 26" text-anchor="middle" class="tiny" font-size="11" font-weight="620">PD size</text>
          <line :x1="unitX(1 - dtiWidth / 2)" :y1="unitY(1.75)" :x2="unitX(1 + dtiWidth / 2)" :y2="unitY(1.75)" class="dim" stroke="#2563eb" stroke-width="1.7" fill="none" :marker-end="arrowRef" />
          <text :x="unitX(1)" :y="unitY(1.75) - 7" text-anchor="middle" class="tiny" font-size="11" font-weight="620">DTI width</text>
        </template>
      </g>

      <g class="callouts" transform="translate(278 47)">
        <text x="0" y="0" class="callout-title" font-size="14" font-weight="760">{{ meta.calloutTitle }}</text>
        <text
          v-for="(line, i) in meta.lines"
          :key="line"
          x="0"
          :y="29 + i * 25"
          class="callout-line"
          font-size="13"
          font-weight="540"
        >
          {{ line }}
        </text>
      </g>
    </svg>
  </figure>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { bayerCells2x2, pixelStackDefaults } from '../composables/pixelStackDefaults'

type Variant = 'pixel' | 'air' | 'microlens' | 'planarization' | 'color_filter' | 'barl' | 'silicon'
type Rect = { id: string; x0: number; x1: number; y0: number; y1: number }

const props = withDefaults(defineProps<{ variant?: Variant }>(), {
  variant: 'pixel',
})

const { t } = useLocale()

const variant = computed(() => props.variant)
const idSuffix = computed(() => `top-${variant.value}`)
const arrowId = computed(() => `sectionArrow-${idSuffix.value}`)
const lensGradId = computed(() => `sectionLensGrad-${idSuffix.value}`)
const filmPatternId = computed(() => `sectionFilmPattern-${idSuffix.value}`)
const clipId = computed(() => `sectionClip-${idSuffix.value}`)
const arrowRef = computed(() => `url(#${arrowId.value})`)

const defaults = pixelStackDefaults
const plot = { x: 56, y: 38, size: 172, cell: 86 }
const unitSize = defaults.pitch * defaults.unitCell[1]
const scale = plot.size / unitSize

const gridWidth = defaults.colorFilter.grid.width
const dtiWidth = defaults.silicon.dti.width
const pdSize = defaults.silicon.photodiode.sizeX
const pdSizePx = pdSize * scale
const mlRadiusX = defaults.microlens.radiusX
const mlRadiusY = defaults.microlens.radiusY
const mlN = defaults.microlens.profileN
const illustrativeCraShift = 0.16
const colorFilterGridThickness = defaults.colorFilter.grid.thickness

const channelSpecs = defaults.colorFilter.channels
const cells = bayerCells2x2.map((cell) => ({
  ...cell,
  fill: channelSpecs[cell.key].sectionFill,
}))

const visibleLabels = computed(() => {
  if (!['pixel', 'color_filter'].includes(variant.value)) return []
  return cells
})

const showPixelReference = computed(() => true)
const pixelBoundaryPath = `M ${plot.cell} 0 V ${plot.size} M 0 ${plot.cell} H ${plot.size}`

const microlenses = cells.map((cell) => ({
  id: cell.id,
  cx: cell.cx,
  cy: cell.cy,
}))

const photodiodes = cells.map((cell): Rect => ({
  id: cell.id,
  x0: cell.cx - pdSize / 2,
  x1: cell.cx + pdSize / 2,
  y0: cell.cy - pdSize / 2,
  y1: cell.cy + pdSize / 2,
}))

const metalGridRects = boundaryRects(gridWidth, 'metal')
const dtiRects = boundaryRects(dtiWidth, 'dti')

const colorFilterCells = cells.map((cell) => {
  const baseInset = gridWidth / 2
  const topInset = baseInset + cfTopInset(cell.key)
  return {
    ...cell,
    base: {
      x0: cell.x0 + baseInset,
      x1: cell.x1 - baseInset,
      y0: cell.y0 + baseInset,
      y1: cell.y1 - baseInset,
    },
    top: {
      x0: cell.x0 + topInset,
      x1: cell.x1 - topInset,
      y0: cell.y0 + topInset,
      y1: cell.y1 - topInset,
    },
  }
})

function boundaryRects(width: number, prefix: string): Rect[] {
  const half = width / 2
  const rects: Rect[] = []
  for (const x of [0, 1, 2]) {
    rects.push({
      id: `${prefix}-v-${x}`,
      x0: Math.max(0, x - half),
      x1: Math.min(unitSize, x + half),
      y0: 0,
      y1: unitSize,
    })
  }
  for (const y of [0, 1, 2]) {
    rects.push({
      id: `${prefix}-h-${y}`,
      x0: 0,
      x1: unitSize,
      y0: Math.max(0, y - half),
      y1: Math.min(unitSize, y + half),
    })
  }
  return rects
}

function cfTopInset(channel: string): number {
  const spec = channelSpecs[channel]
  const protrusion = Math.max(0, spec.thickness - colorFilterGridThickness)
  if (protrusion <= 0 || spec.contactAngle >= 89.999) return 0
  const theta = (Math.PI / 180) * Math.max(1, Math.min(89.999, spec.contactAngle))
  return protrusion / Math.tan(theta)
}

function unitX(x: number) {
  return x * scale
}

function unitY(y: number) {
  return plot.size - y * scale
}

function rectX(x0: number) {
  return unitX(x0)
}

function rectY(y1: number) {
  return unitY(y1)
}

function rectW(x0: number, x1: number) {
  return Math.max(0, (x1 - x0) * scale)
}

function rectH(y0: number, y1: number) {
  return Math.max(0, (y1 - y0) * scale)
}

function superellipsePath(cx: number, cy: number, rx: number, ry: number, n: number) {
  const pts: string[] = []
  const count = 88
  for (let i = 0; i < count; i++) {
    const theta = (2 * Math.PI * i) / count
    const cosT = Math.cos(theta)
    const sinT = Math.sin(theta)
    const x = cx + rx * Math.sign(cosT) * Math.pow(Math.abs(cosT), 2 / n)
    const y = cy + ry * Math.sign(sinT) * Math.pow(Math.abs(sinT), 2 / n)
    pts.push(`${i === 0 ? 'M' : 'L'} ${unitX(x).toFixed(2)} ${unitY(y).toFixed(2)}`)
  }
  return `${pts.join(' ')} Z`
}

const meta = computed(() => {
  const entries: Record<Variant, { title: string; note: string; calloutTitle: string; lines: string[]; aria: string }> = {
    pixel: {
      title: t('XY top view: unit cell', 'XY top view: unit cell'),
      note: t('pitch, unit_cell, bayer_map', 'pitch, unit_cell, bayer_map'),
      calloutTitle: t('Top-level scope', '최상위 범위'),
      lines: [
        t(`${defaults.unitCell[0]} x ${defaults.unitCell[1]} periodic tile, ${defaults.pitch.toFixed(1)} um pitch`, `${defaults.unitCell[0]} x ${defaults.unitCell[1]} periodic tile, ${defaults.pitch.toFixed(1)} um pitch`),
        t('origin is the lower-left corner', 'origin은 좌측 하단 모서리'),
        t('row index maps to increasing y', 'row index는 +y 방향으로 증가'),
      ],
      aria: t('XY top view of pitch, unit cell, and Bayer map', 'pitch, unit cell, Bayer map의 XY top view'),
    },
    air: {
      title: t('XY top view: air', 'XY top view: air'),
      note: t('uniform incident medium', '균일한 입사 매질'),
      calloutTitle: t('What changes here', '여기서 바뀌는 것'),
      lines: [
        t('no lateral material pattern', '횡방향 재료 패턴 없음'),
        t('same aperture as the unit cell', 'unit cell 전체 aperture'),
        t('thickness is a z-only parameter', 'thickness는 z 방향 값'),
      ],
      aria: t('XY top view of the air layer', 'air 레이어의 XY top view'),
    },
    microlens: {
      title: t('XY top view: microlens', 'XY top view: microlens'),
      note: t('radius, gap, sharing, CRA shift', 'radius, gap, sharing, CRA shift'),
      calloutTitle: t('Footprint controls', 'footprint 제어'),
      lines: [
        t(`solid footprints use radius_x/y = ${mlRadiusX.toFixed(2)} um`, `실선 footprint는 radius_x/y = ${mlRadiusX.toFixed(2)} um`),
        t('dashed footprints show optional CRA shift', '점선은 optional CRA shift'),
        t('outer dashed outline shows sharing > 1', '외곽 점선은 sharing > 1 예'),
      ],
      aria: t('XY top view of microlens footprint controls', 'microlens footprint 제어의 XY top view'),
    },
    planarization: {
      title: t('XY top view: planarization', 'XY top view: planarization'),
      note: t('uniform spacer between lens and CFA', '렌즈와 CFA 사이의 균일 spacer'),
      calloutTitle: t('Why it matters', '중요한 이유'),
      lines: [
        t('no default x-y pattern', '기본 x-y 패턴 없음'),
        t('fills the full simulation tile', '시뮬레이션 tile 전체를 채움'),
        t('thickness controls focus depth', 'thickness는 초점 깊이 제어'),
      ],
      aria: t('XY top view of planarization as a uniform spacer', '균일 spacer인 planarization의 XY top view'),
    },
    color_filter: {
      title: t('XY top view: color filter', 'XY top view: color filter'),
      note: t('Bayer cells, metal grid, tapered top', 'Bayer 셀, metal grid, taper top'),
      calloutTitle: t('Pattern controls', '패턴 제어'),
      lines: [
        t('metal grid exists at all pixel boundaries', 'metal grid는 모든 pixel boundary'),
        t('solid fill is the footprint at grid top', 'solid fill은 grid top footprint'),
        t('inner outline is the tapered top footprint', 'inner outline은 taper top footprint'),
      ],
      aria: t('XY top view of color filter and metal grid', 'color filter와 metal grid의 XY top view'),
    },
    barl: {
      title: t('XY top view: BARL', 'XY top view: BARL'),
      note: t('full-area thin-film stack', '전체 면적 박막 stack'),
      calloutTitle: t('Layer-list scope', 'layer list 범위'),
      lines: [
        t('no lateral pattern by default', '기본 횡방향 패턴 없음'),
        t('pixel lines are reference only', 'pixel line은 reference only'),
        t('optimize as a thin-film recipe', '박막 recipe로 최적화'),
      ],
      aria: t('XY top view of BARL full-area films', 'BARL 전체 면적 박막의 XY top view'),
    },
    silicon: {
      title: t('XY top view: silicon', 'XY top view: silicon'),
      note: t('photodiode footprint and DTI grid', 'photodiode footprint와 DTI grid'),
      calloutTitle: t('Collection controls', '수광부 제어'),
      lines: [
        t(`PD windows are ${pdSize.toFixed(1)} x ${pdSize.toFixed(1)} um`, `PD window는 ${pdSize.toFixed(1)} x ${pdSize.toFixed(1)} um`),
        t('DTI lines include periodic outer edges', 'DTI line은 주기 외곽 edge 포함'),
        t('position offsets from pixel center', 'position은 pixel center 기준'),
      ],
      aria: t('XY top view of photodiode and DTI controls', 'photodiode와 DTI 제어의 XY top view'),
    },
  }
  return entries[variant.value]
})
</script>

<style scoped>
.section-view {
  margin: 16px 0 20px;
  max-width: 760px;
}

.section-view-head {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  align-items: baseline;
  margin-bottom: 8px;
}

.section-view-head strong {
  color: var(--vp-c-text-1);
  font-size: 14px;
}

.section-view-head span {
  color: var(--vp-c-text-2);
  font-size: 13px;
}

.section-svg {
  display: block;
  width: 100%;
  height: auto;
}

.canvas {
  fill: #f8fafc;
  stroke: #d8dee9;
}

:root.dark .canvas {
  fill: #1a1f2e;
  stroke: #3a4060;
}

.domain {
  fill: #ffffff;
  stroke: #94a3b8;
  stroke-width: 1.2;
}

:root.dark .domain {
  fill: #242938;
  stroke: #4a5580;
}

.domain-outline {
  fill: none;
  stroke: #334155;
  stroke-width: 1.2;
}

:root.dark .domain-outline {
  stroke: #8899bb;
}

.pixel-reference {
  stroke: #64748b;
  stroke-width: 1;
  fill: none;
  stroke-dasharray: 5 5;
  opacity: 0.55;
}

:root.dark .pixel-reference {
  stroke: #7a8aaa;
}

.lens-footprint {
  stroke: #2f9bb8;
  stroke-width: 1.5;
  opacity: 0.92;
}

.shifted-lens {
  fill: none;
  stroke: #f59e0b;
  stroke-width: 1.3;
  stroke-dasharray: 5 4;
  opacity: 0.6;
}

.shared-lens {
  fill: none;
  stroke: #2563eb;
  stroke-width: 1.3;
  stroke-dasharray: 8 5;
  opacity: 0.55;
}

.metal-fill {
  fill: #4b5563;
  opacity: 0.96;
}

.cf-top {
  fill-opacity: 0.18;
  stroke: #111827;
  stroke-width: 1.5;
  stroke-dasharray: 5 4;
}

:root.dark .cf-top {
  stroke: #c8d0e0;
}

.dti-fill {
  fill: #dff3ff;
  stroke: #7db9d8;
  stroke-width: 0.6;
  opacity: 0.95;
}

.pd-fill {
  fill: #b85c5c;
  stroke: #8a3f3f;
  stroke-width: 1;
  opacity: 0.68;
}

.dim,
.shift {
  stroke: currentColor;
  stroke-width: 1.7;
  fill: none;
}

.dim {
  color: #2563eb;
}

.shift {
  color: #f59e0b;
  stroke-dasharray: 4 4;
}

.origin-dot {
  fill: #2563eb;
}

.label,
.tiny,
.cell-label,
.callout-title,
.callout-line {
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-base);
}

.label {
  font-size: 12px;
  font-weight: 650;
}

.tiny {
  font-size: 11px;
  font-weight: 620;
}

.rotate-label {
  transform-box: fill-box;
  transform-origin: center;
  transform: rotate(-90deg);
}

.cell-label {
  font-size: 18px;
  font-weight: 760;
  fill: rgba(17, 24, 39, 0.85);
  paint-order: stroke;
  stroke: rgba(255, 255, 255, 0.8);
  stroke-width: 3px;
}

:root.dark .cell-label {
  fill: rgba(230, 235, 245, 0.92);
  stroke: rgba(20, 25, 40, 0.75);
}

.callout-title {
  font-size: 14px;
  font-weight: 760;
}

.callout-line {
  fill: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 540;
}

@media (max-width: 640px) {
  .section-view-head {
    display: block;
  }

  .section-view-head span {
    display: block;
    margin-top: 2px;
  }

  .callout-line {
    font-size: 12px;
  }
}
</style>
