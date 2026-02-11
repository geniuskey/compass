<template>
  <div class="module-arch-container">
    <h4>{{ t('Module Architecture Diagram', '모듈 아키텍처 다이어그램') }}</h4>
    <p class="component-description">
      {{ t(
        'Click on any module to see its description and key classes. Hover to highlight dependency arrows.',
        '모듈을 클릭하면 설명과 주요 클래스를 볼 수 있습니다. 마우스를 올리면 의존성 화살표가 강조됩니다.'
      ) }}
    </p>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="arch-svg">
        <defs>
          <marker id="archArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--vp-c-text-3)" />
          </marker>
          <marker id="archArrowHl" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--vp-c-brand-1)" />
          </marker>
        </defs>

        <!-- Arrows (rendered before blocks so blocks appear on top) -->
        <line
          v-for="(arrow, idx) in arrows"
          :key="'arrow-' + idx"
          :x1="arrow.x1"
          :y1="arrow.y1"
          :x2="arrow.x2"
          :y2="arrow.y2"
          :stroke="isArrowHighlighted(arrow) ? 'var(--vp-c-brand-1)' : 'var(--vp-c-text-3)'"
          :stroke-width="isArrowHighlighted(arrow) ? 2.5 : 1.5"
          :stroke-dasharray="isArrowHighlighted(arrow) ? 'none' : '4,3'"
          :marker-end="isArrowHighlighted(arrow) ? 'url(#archArrowHl)' : 'url(#archArrow)'"
          :opacity="hoveredModule && !isArrowHighlighted(arrow) ? 0.25 : 1"
          class="arrow-line"
        />

        <!-- Module blocks -->
        <g
          v-for="mod in modules"
          :key="mod.id"
          :transform="`translate(${mod.x}, ${mod.y})`"
          class="module-block"
          @mouseenter="hoveredModule = mod.id"
          @mouseleave="hoveredModule = null"
          @click="toggleSelected(mod.id)"
          style="cursor: pointer"
        >
          <rect
            x="0"
            y="0"
            :width="blockW"
            :height="blockH"
            :fill="mod.color"
            :stroke="selectedModule === mod.id ? 'var(--vp-c-brand-1)' : hoveredModule === mod.id ? 'var(--vp-c-brand-1)' : 'var(--vp-c-divider)'"
            :stroke-width="selectedModule === mod.id ? 2.5 : hoveredModule === mod.id ? 2 : 1"
            rx="8"
            ry="8"
          />
          <text
            :x="blockW / 2"
            :y="blockH / 2 - 4"
            text-anchor="middle"
            dominant-baseline="central"
            class="block-label"
          >{{ mod.label }}</text>
          <text
            :x="blockW / 2"
            :y="blockH / 2 + 12"
            text-anchor="middle"
            dominant-baseline="central"
            class="block-sub"
          >{{ mod.file }}</text>
        </g>
      </svg>
    </div>

    <!-- Detail panel -->
    <transition name="detail-fade">
      <div v-if="selectedModule" class="detail-panel">
        <div class="detail-header">
          <strong>{{ selectedDetail.label }}</strong>
          <button class="detail-close" @click="selectedModule = null">&times;</button>
        </div>
        <p class="detail-desc">{{ selectedDetail.description }}</p>
        <div class="detail-classes">
          <span class="detail-classes-label">{{ t('Key API', '주요 API') }}:</span>
          <code v-for="cls in selectedDetail.classes" :key="cls" class="detail-class-tag">{{ cls }}</code>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

const svgW = 600
const svgH = 300
const blockW = 150
const blockH = 60

const hoveredModule = ref(null)
const selectedModule = ref(null)

function toggleSelected(id) {
  selectedModule.value = selectedModule.value === id ? null : id
}

// Module positions: top row and bottom row
const modules = computed(() => [
  {
    id: 'config',
    label: 'CompassConfig',
    file: 'config_schema.py',
    x: 20,
    y: 30,
    color: 'var(--mod-config)',
    description: t('Validates YAML configuration', 'YAML 설정 검증'),
    classes: ['config_schema.py', 'PixelConfig', 'SolverConfig', 'SourceConfig'],
  },
  {
    id: 'material',
    label: 'MaterialDB',
    file: 'database.py',
    x: 225,
    y: 30,
    color: 'var(--mod-material)',
    description: t('Optical property registry', '광학 특성 레지스트리'),
    classes: ['get_nk()', 'get_epsilon()', 'register_constant()'],
  },
  {
    id: 'pixel',
    label: 'PixelStack',
    file: 'pixel_stack.py',
    x: 430,
    y: 30,
    color: 'var(--mod-pixel)',
    description: t('Solver-agnostic pixel structure', '솔버 독립적 픽셀 구조'),
    classes: ['get_layer_slices()', 'get_permittivity_grid()'],
  },
  {
    id: 'solver',
    label: 'SolverBase',
    file: 'base.py',
    x: 20,
    y: 200,
    color: 'var(--mod-solver)',
    description: t('EM solver interface', 'EM 솔버 인터페이스'),
    classes: ['setup_geometry()', 'run()', 'SolverFactory'],
  },
  {
    id: 'result',
    label: 'SimulationResult',
    file: 'types.py',
    x: 225,
    y: 200,
    color: 'var(--mod-result)',
    description: t('Standardized output', '표준화된 출력'),
    classes: ['qe_per_pixel', 'fields', 'R/T/A'],
  },
  {
    id: 'analysis',
    label: 'Analysis',
    file: 'analysis/',
    x: 430,
    y: 200,
    color: 'var(--mod-analysis)',
    description: t('Post-processing', '후처리'),
    classes: ['QECalculator', 'EnergyBalance', 'SolverComparison'],
  },
])

// Arrow definitions: from module center-right/bottom to module center-left/top
const arrowDefs = [
  { from: 'config', to: 'material' },
  { from: 'config', to: 'pixel' },
  { from: 'material', to: 'pixel' },
  { from: 'pixel', to: 'solver' },
  { from: 'config', to: 'solver' },
  { from: 'solver', to: 'result' },
  { from: 'result', to: 'analysis' },
]

function getModuleById(id) {
  return modules.value.find(m => m.id === id)
}

// Compute arrow start/end points with proper anchoring
const arrows = computed(() => {
  return arrowDefs.map(def => {
    const from = getModuleById(def.from)
    const to = getModuleById(def.to)
    if (!from || !to) return null

    const fromCx = from.x + blockW / 2
    const fromCy = from.y + blockH / 2
    const toCx = to.x + blockW / 2
    const toCy = to.y + blockH / 2

    let x1, y1, x2, y2

    // Determine best anchor points
    const dx = toCx - fromCx
    const dy = toCy - fromCy

    if (Math.abs(dx) > Math.abs(dy)) {
      // Horizontal-dominant: use left/right sides
      if (dx > 0) {
        x1 = from.x + blockW
        y1 = fromCy
        x2 = to.x
        y2 = toCy
      } else {
        x1 = from.x
        y1 = fromCy
        x2 = to.x + blockW
        y2 = toCy
      }
    } else {
      // Vertical-dominant: use top/bottom sides
      if (dy > 0) {
        x1 = fromCx
        y1 = from.y + blockH
        x2 = toCx
        y2 = to.y
      } else {
        x1 = fromCx
        y1 = from.y
        x2 = toCx
        y2 = to.y + blockH
      }
    }

    return { from: def.from, to: def.to, x1, y1, x2, y2 }
  }).filter(Boolean)
})

function isArrowHighlighted(arrow) {
  if (!hoveredModule.value) return false
  return arrow.from === hoveredModule.value || arrow.to === hoveredModule.value
}

const selectedDetail = computed(() => {
  const mod = modules.value.find(m => m.id === selectedModule.value)
  if (!mod) return { label: '', description: '', classes: [] }
  return {
    label: mod.label,
    description: mod.description,
    classes: mod.classes,
  }
})
</script>

<style scoped>
.module-arch-container {
  --mod-config: #e8f4fd;
  --mod-material: #fef3e2;
  --mod-pixel: #e8fbe8;
  --mod-solver: #f3e8fd;
  --mod-result: #fde8e8;
  --mod-analysis: #e8f0fe;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

:root.dark .module-arch-container {
  --mod-config: #1a2e3d;
  --mod-material: #3d2e1a;
  --mod-pixel: #1a3d1a;
  --mod-solver: #2e1a3d;
  --mod-result: #3d1a1a;
  --mod-analysis: #1a2440;
}

.module-arch-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.svg-wrapper {
  margin-top: 8px;
}

.arch-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}

.arrow-line {
  transition: stroke 0.15s, stroke-width 0.15s, opacity 0.15s;
}

.module-block rect {
  transition: stroke 0.15s, stroke-width 0.15s;
}

.block-label {
  font-size: 12px;
  font-weight: 600;
  fill: var(--vp-c-text-1);
  pointer-events: none;
}

.block-sub {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  pointer-events: none;
}

.detail-panel {
  margin-top: 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 6px;
  padding: 12px 16px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.detail-header strong {
  font-size: 1em;
  color: var(--vp-c-text-1);
}

.detail-close {
  background: none;
  border: none;
  font-size: 1.3em;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
}

.detail-close:hover {
  color: var(--vp-c-text-1);
}

.detail-desc {
  margin: 6px 0 10px 0;
  color: var(--vp-c-text-2);
  font-size: 0.88em;
}

.detail-classes {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.detail-classes-label {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.detail-class-tag {
  font-size: 0.78em;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 2px 6px;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-brand-1);
}

.detail-fade-enter-active,
.detail-fade-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}

.detail-fade-enter-from,
.detail-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}
</style>
