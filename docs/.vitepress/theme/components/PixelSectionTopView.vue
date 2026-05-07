<template>
  <figure class="section-view" :aria-label="meta.aria">
    <div class="section-view-head">
      <strong>{{ meta.title }}</strong>
      <span>{{ meta.note }}</span>
    </div>

    <svg viewBox="0 0 620 238" class="section-svg" role="img">
      <defs>
        <marker :id="arrowId" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
        </marker>
        <marker :id="lightArrowId" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#f59e0b" />
        </marker>
        <radialGradient :id="lensGradId" cx="50%" cy="50%" r="60%">
          <stop offset="0%" stop-color="#e8f7ff" />
          <stop offset="100%" stop-color="#7bc7e7" />
        </radialGradient>
        <pattern :id="filmPatternId" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="8" height="8" fill="#fef3c7" />
          <rect width="3" height="8" fill="#fde68a" opacity="0.75" />
        </pattern>
      </defs>

      <rect x="0" y="0" width="620" height="238" rx="8" class="canvas" />

      <g :transform="`translate(${plot.x} ${plot.y})`">
        <rect
          x="0"
          y="0"
          :width="plot.size"
          :height="plot.size"
          rx="4"
          class="domain"
        />

        <template v-if="variant === 'pixel'">
          <rect
            v-for="cell in cells"
            :key="'pixel-' + cell.id"
            :x="cellX(cell.col)"
            :y="cellY(cell.row)"
            :width="plot.cell"
            :height="plot.cell"
            :fill="cell.fill"
            opacity="0.52"
          />
          <path :d="gridPath" class="thin-grid" />
          <circle cx="0" :cy="plot.size" r="4" class="origin-dot" />
          <text x="-4" :y="plot.size + 19" text-anchor="end" class="tiny">origin</text>
          <line x1="0" :y1="plot.size + 25" :x2="plot.cell" :y2="plot.size + 25" class="dim" :marker-end="arrowRef" />
          <text :x="plot.cell / 2" :y="plot.size + 42" text-anchor="middle" class="label">pitch</text>
          <line x1="-20" :y1="plot.size" x2="-20" :y2="plot.cell" class="dim" :marker-end="arrowRef" />
          <text x="-33" :y="plot.cell + 36" text-anchor="middle" class="label rotate-label">pitch</text>
        </template>

        <template v-else-if="variant === 'air'">
          <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="5" fill="#e8f2ff" opacity="0.82" />
          <path :d="gridPath" class="thin-grid" />
          <line
            v-for="x in [42, 82, 122]"
            :key="'ray-' + x"
            :x1="x"
            y1="30"
            :x2="x"
            y2="87"
            class="light-ray"
            :marker-end="lightArrowRef"
          />
          <circle cx="82" cy="104" r="44" fill="none" stroke="#60a5fa" stroke-dasharray="4 4" opacity="0.75" />
        </template>

        <template v-else-if="variant === 'microlens'">
          <path :d="gridPath" class="thin-grid" />
          <ellipse
            v-for="cell in cells"
            :key="'lens-' + cell.id"
            :cx="cellX(cell.col) + plot.cell / 2 + (cell.key === 'R' ? 7 : 0)"
            :cy="cellY(cell.row) + plot.cell / 2"
            rx="32"
            ry="32"
            :fill="`url(#${lensGradId})`"
            stroke="#3498b8"
            stroke-width="1.5"
          />
          <line :x1="plot.cell / 2" :y1="plot.cell / 2" :x2="plot.cell / 2 + 39" :y2="plot.cell / 2" class="shift" :marker-end="arrowRef" />
          <text :x="plot.cell / 2 + 42" :y="plot.cell / 2 - 5" class="tiny">shift</text>
          <line x1="50" y1="42" x2="82" y2="42" class="dim" :marker-end="arrowRef" />
          <text x="67" y="35" text-anchor="middle" class="tiny">radius</text>
        </template>

        <template v-else-if="variant === 'planarization'">
          <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" fill="#d8edf8" opacity="0.86" />
          <path :d="gridPath" class="thin-grid muted" />
          <circle
            v-for="cell in cells"
            :key="'focus-' + cell.id"
            :cx="cellX(cell.col) + plot.cell / 2"
            :cy="cellY(cell.row) + plot.cell / 2"
            r="15"
            fill="#38bdf8"
            opacity="0.22"
          />
          <path d="M22 82 C52 50, 112 50, 142 82" fill="none" stroke="#0ea5e9" stroke-width="1.7" stroke-dasharray="5 4" />
        </template>

        <template v-else-if="variant === 'color_filter'">
          <rect x="0" y="0" :width="plot.size" :height="plot.size" fill="#4b5563" opacity="0.28" />
          <rect
            v-for="cell in cells"
            :key="'cf-' + cell.id"
            :x="cellX(cell.col) + cfInset"
            :y="cellY(cell.row) + cfInset"
            :width="plot.cell - cfInset * 2"
            :height="plot.cell - cfInset * 2"
            rx="9"
            :fill="cell.fill"
            opacity="0.82"
          />
          <path :d="gridPath" class="metal-grid" />
          <text x="82" y="87" text-anchor="middle" class="tiny dark">grid.width</text>
        </template>

        <template v-else-if="variant === 'barl'">
          <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" :fill="`url(#${filmPatternId})`" opacity="0.9" />
          <path :d="gridPath" class="thin-grid muted" />
          <line x1="28" y1="118" x2="136" y2="118" class="dim" :marker-end="arrowRef" />
          <text x="82" y="109" text-anchor="middle" class="label">full aperture film</text>
        </template>

        <template v-else>
          <rect x="0" y="0" :width="plot.size" :height="plot.size" rx="4" fill="#cfd4dc" opacity="0.88" />
          <path :d="gridPath" class="dti-grid" />
          <rect
            v-for="cell in cells"
            :key="'pd-' + cell.id"
            :x="cellX(cell.col) + 24"
            :y="cellY(cell.row) + 24"
            width="34"
            height="34"
            rx="5"
            :fill="cell.fill"
            opacity="0.62"
          />
          <text x="82" y="88" text-anchor="middle" class="tiny dark">DTI</text>
          <line x1="24" y1="60" x2="58" y2="60" class="dim" :marker-end="arrowRef" />
          <text x="42" y="54" text-anchor="middle" class="tiny">PD size</text>
        </template>

        <text
          v-for="cell in visibleLabels"
          :key="'label-' + cell.id"
          :x="cellX(cell.col) + plot.cell / 2"
          :y="cellY(cell.row) + plot.cell / 2 + 5"
          text-anchor="middle"
          class="cell-label"
        >
          {{ cell.key }}
        </text>
      </g>

      <g class="callouts" transform="translate(258 43)">
        <text x="0" y="0" class="callout-title">{{ meta.calloutTitle }}</text>
        <text
          v-for="(line, i) in meta.lines"
          :key="line"
          x="0"
          :y="27 + i * 25"
          class="callout-line"
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

type Variant = 'pixel' | 'air' | 'microlens' | 'planarization' | 'color_filter' | 'barl' | 'silicon'

const props = withDefaults(defineProps<{ variant?: Variant }>(), {
  variant: 'pixel',
})

const { t } = useLocale()

const variant = computed(() => props.variant)
const idSuffix = computed(() => `top-${variant.value}`)
const arrowId = computed(() => `sectionArrow-${idSuffix.value}`)
const lightArrowId = computed(() => `sectionLightArrow-${idSuffix.value}`)
const lensGradId = computed(() => `sectionLensGrad-${idSuffix.value}`)
const filmPatternId = computed(() => `sectionFilmPattern-${idSuffix.value}`)
const arrowRef = computed(() => `url(#${arrowId.value})`)
const lightArrowRef = computed(() => `url(#${lightArrowId.value})`)

const plot = { x: 54, y: 33, size: 164, cell: 82 }
const cfInset = 9

const cells = [
  { id: 'r0c0', key: 'R', row: 0, col: 0, fill: '#f87171' },
  { id: 'r0c1', key: 'G', row: 0, col: 1, fill: '#4ade80' },
  { id: 'r1c0', key: 'G', row: 1, col: 0, fill: '#4ade80' },
  { id: 'r1c1', key: 'B', row: 1, col: 1, fill: '#60a5fa' },
]

const visibleLabels = computed(() => {
  if (['air', 'planarization', 'barl'].includes(variant.value)) return []
  return cells
})

const gridPath = `M ${plot.cell} 0 V ${plot.size} M 0 ${plot.cell} H ${plot.size}`

function cellX(col: number) {
  return col * plot.cell
}

function cellY(row: number) {
  return row * plot.cell
}

const meta = computed(() => {
  const entries: Record<Variant, { title: string; note: string; calloutTitle: string; lines: string[]; aria: string }> = {
    pixel: {
      title: t('XY top view: unit cell', 'XY top view: unit cell'),
      note: t('pitch, unit_cell, bayer_map', 'pitch, unit_cell, bayer_map'),
      calloutTitle: t('Top-level scope', '최상위 범위'),
      lines: [
        t('pitch sets one pixel period', 'pitch는 한 픽셀 주기'),
        t('unit_cell repeats in x-y', 'unit_cell은 x-y 반복 단위'),
        t('bayer_map assigns channels', 'bayer_map은 채널 배치'),
      ],
      aria: t('XY top view of pitch, unit cell, and Bayer map', 'pitch, unit cell, Bayer map의 XY top view'),
    },
    air: {
      title: t('XY top view: air', 'XY top view: air'),
      note: t('uniform incident medium', '균일한 입사 매질'),
      calloutTitle: t('What changes here', '여기서 바뀌는 것'),
      lines: [
        t('no lateral pattern', '횡방향 패턴 없음'),
        t('thickness changes propagation distance', 'thickness는 전파 거리'),
        t('material is usually air', 'material은 보통 air'),
      ],
      aria: t('XY top view of the air layer', 'air 레이어의 XY top view'),
    },
    microlens: {
      title: t('XY top view: microlens', 'XY top view: microlens'),
      note: t('radius, gap, sharing, CRA shift', 'radius, gap, sharing, CRA shift'),
      calloutTitle: t('Footprint controls', 'footprint 제어'),
      lines: [
        t('radius_x / radius_y set aperture', 'radius_x/y는 렌즈 footprint'),
        t('sharing merges N x N pixels', 'sharing은 N x N 픽셀 공유'),
        t('shift moves lens centers', 'shift는 렌즈 중심 이동'),
      ],
      aria: t('XY top view of microlens footprint controls', 'microlens footprint 제어의 XY top view'),
    },
    planarization: {
      title: t('XY top view: planarization', 'XY top view: planarization'),
      note: t('uniform spacer between lens and CFA', '렌즈와 CFA 사이의 균일 spacer'),
      calloutTitle: t('Why it matters', '중요한 이유'),
      lines: [
        t('no x-y pattern by default', '기본적으로 x-y 패턴 없음'),
        t('thickness shifts the focus depth', 'thickness는 초점 깊이 이동'),
        t('material changes refraction', 'material은 굴절 경로 변경'),
      ],
      aria: t('XY top view of planarization as a uniform spacer', '균일 spacer인 planarization의 XY top view'),
    },
    color_filter: {
      title: t('XY top view: color filter', 'XY top view: color filter'),
      note: t('Bayer cells, metal grid, rounded corners', 'Bayer 셀, metal grid, 둥근 corner'),
      calloutTitle: t('Pattern controls', '패턴 제어'),
      lines: [
        t('red / green / blue set material and height', 'red/green/blue는 재료와 높이'),
        t('grid.width isolates pixels', 'grid.width는 픽셀 격리'),
        t('corner_radius rounds CF cells', 'corner_radius는 CF corner'),
      ],
      aria: t('XY top view of color filter and metal grid', 'color filter와 metal grid의 XY top view'),
    },
    barl: {
      title: t('XY top view: BARL', 'XY top view: BARL'),
      note: t('full-area thin-film stack', '전체 면적 박막 stack'),
      calloutTitle: t('Layer-list scope', 'layer list 범위'),
      lines: [
        t('each entry covers the full aperture', '각 layer는 전체 aperture'),
        t('thickness is a z parameter', 'thickness는 z 방향 값'),
        t('optimize as a thin-film recipe', '박막 recipe로 최적화'),
      ],
      aria: t('XY top view of BARL full-area films', 'BARL 전체 면적 박막의 XY top view'),
    },
    silicon: {
      title: t('XY top view: silicon', 'XY top view: silicon'),
      note: t('photodiode footprint and DTI grid', 'photodiode footprint와 DTI grid'),
      calloutTitle: t('Collection controls', '수광부 제어'),
      lines: [
        t('photodiode.size sets collection area', 'photodiode.size는 수광 면적'),
        t('dti.width sets boundary isolation', 'dti.width는 경계 격리'),
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
  max-width: 720px;
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
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-divider);
}

.domain {
  fill: var(--vp-c-bg);
  stroke: var(--vp-c-text-3);
  stroke-width: 1.2;
}

.thin-grid {
  stroke: var(--vp-c-text-3);
  stroke-width: 1;
  fill: none;
}

.thin-grid.muted {
  opacity: 0.42;
  stroke-dasharray: 4 4;
}

.metal-grid {
  stroke: #4b5563;
  stroke-width: 9;
  fill: none;
  stroke-linecap: square;
}

.dti-grid {
  stroke: #e0f2fe;
  stroke-width: 10;
  fill: none;
}

.dim,
.shift {
  color: var(--vp-c-brand-1);
  stroke: currentColor;
  stroke-width: 1.7;
  fill: none;
}

.shift {
  stroke-dasharray: 4 4;
}

.light-ray {
  stroke: #f59e0b;
  stroke-width: 2.2;
  stroke-linecap: round;
}

.origin-dot {
  fill: var(--vp-c-brand-1);
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

.tiny.dark {
  fill: #111827;
}

.rotate-label {
  transform-box: fill-box;
  transform-origin: center;
  transform: rotate(-90deg);
}

.cell-label {
  font-size: 18px;
  font-weight: 760;
  fill: rgba(17, 24, 39, 0.78);
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
