<template>
  <div class="arch-overview-container">
    <div class="arch-flow">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="arch-flow-svg">
        <defs>
          <marker id="archFlowArrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--vp-c-text-3)" />
          </marker>
          <marker id="archFlowArrowHl" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--vp-c-brand-1)" />
          </marker>
        </defs>

        <!-- Connection arrows between steps -->
        <line
          v-for="(arrow, idx) in arrows"
          :key="'arrow-' + idx"
          :x1="arrow.x1" :y1="arrow.y1"
          :x2="arrow.x2" :y2="arrow.y2"
          :stroke="hoveredStep !== null && (hoveredStep === idx || hoveredStep === idx + 1) ? 'var(--vp-c-brand-1)' : 'var(--vp-c-text-3)'"
          :stroke-width="hoveredStep !== null && (hoveredStep === idx || hoveredStep === idx + 1) ? 2.5 : 1.5"
          :marker-end="hoveredStep !== null && (hoveredStep === idx || hoveredStep === idx + 1) ? 'url(#archFlowArrowHl)' : 'url(#archFlowArrow)'"
          class="flow-arrow"
        />

        <!-- Steps -->
        <g
          v-for="(step, idx) in steps"
          :key="step.id"
          :transform="`translate(${step.x}, ${step.y})`"
          @mouseenter="hoveredStep = idx"
          @mouseleave="hoveredStep = null"
          @click="toggleStep(idx)"
          style="cursor: pointer"
        >
          <!-- Main box -->
          <rect
            x="0" y="0"
            :width="boxW" :height="boxH"
            :fill="step.color"
            :stroke="selectedStep === idx ? 'var(--vp-c-brand-1)' : hoveredStep === idx ? 'var(--vp-c-brand-1)' : 'var(--vp-c-divider)'"
            :stroke-width="selectedStep === idx ? 2.5 : hoveredStep === idx ? 2 : 1"
            rx="10" ry="10"
            class="step-box"
          />

          <!-- Step number circle -->
          <circle
            :cx="boxW / 2" cy="-10"
            r="12"
            :fill="step.accentColor"
            stroke="var(--vp-c-bg)" stroke-width="2"
          />
          <text
            :x="boxW / 2" y="-6"
            text-anchor="middle" dominant-baseline="central"
            class="step-number"
          >{{ idx + 1 }}</text>

          <!-- Step title -->
          <text
            :x="boxW / 2" :y="boxH / 2 - 6"
            text-anchor="middle" dominant-baseline="central"
            class="step-title"
          >{{ step.title }}</text>

          <!-- Step subtitle -->
          <text
            :x="boxW / 2" :y="boxH / 2 + 10"
            text-anchor="middle" dominant-baseline="central"
            class="step-subtitle"
          >{{ step.file }}</text>

          <!-- Sub-items below the box -->
          <g v-for="(item, iIdx) in step.items" :key="item">
            <rect
              :x="(boxW - subItemW) / 2"
              :y="boxH + 10 + iIdx * (subItemH + 4)"
              :width="subItemW" :height="subItemH"
              fill="var(--vp-c-bg)"
              stroke="var(--vp-c-divider)"
              stroke-width="0.8"
              rx="4"
            />
            <text
              :x="boxW / 2"
              :y="boxH + 10 + iIdx * (subItemH + 4) + subItemH / 2 + 1"
              text-anchor="middle" dominant-baseline="central"
              class="sub-item-label"
            >{{ item }}</text>
          </g>
        </g>
      </svg>
    </div>

    <!-- Detail panel -->
    <transition name="arch-detail-fade">
      <div v-if="selectedStep !== null" class="arch-detail-panel">
        <div class="arch-detail-header">
          <strong>{{ steps[selectedStep].title }}</strong>
          <button class="arch-detail-close" @click="selectedStep = null">&times;</button>
        </div>
        <p class="arch-detail-desc">{{ steps[selectedStep].description }}</p>
        <div class="arch-detail-items">
          <span class="arch-detail-items-label">{{ t('Key modules:', '주요 모듈:') }}</span>
          <code v-for="m in steps[selectedStep].modules" :key="m" class="arch-detail-tag">{{ m }}</code>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const hoveredStep = ref(null)
const selectedStep = ref(null)

function toggleStep(idx) {
  selectedStep.value = selectedStep.value === idx ? null : idx
}

// Layout
const boxW = 120
const boxH = 60
const subItemW = 100
const subItemH = 22
const gapX = 30
const startX = 20
const startY = 30
const svgH = 210

const steps = computed(() => [
  {
    id: 'config',
    title: t('Config', '설정'),
    file: 'YAML',
    color: 'var(--arch-config)',
    accentColor: 'var(--vp-c-brand-1)',
    x: startX,
    y: startY,
    items: [
      t('Pixel stack', '픽셀 스택'),
      t('Source', '광원'),
      t('Solver params', '솔버 파라미터'),
    ],
    description: t(
      'A single YAML file defines the complete simulation: pixel geometry, material properties, illumination conditions, and solver configuration. Validated by Pydantic schemas.',
      '하나의 YAML 파일로 전체 시뮬레이션을 정의합니다: 픽셀 기하학, 재료 특성, 조명 조건, 솔버 설정. Pydantic 스키마로 검증됩니다.'
    ),
    modules: ['config_schema.py', 'PixelConfig', 'SolverConfig'],
  },
  {
    id: 'geometry',
    title: t('Geometry', '기하학'),
    file: 'PixelStack',
    color: 'var(--arch-geometry)',
    accentColor: '#27ae60',
    x: startX + (boxW + gapX) * 1,
    y: startY,
    items: [
      t('Microlens', '마이크로렌즈'),
      t('Color filter', '컬러 필터'),
      t('Silicon + DTI', '실리콘 + DTI'),
    ],
    description: t(
      'The PixelStack converts config into a solver-agnostic 3D structure. Handles microlens superellipse profiles, Bayer patterns, BARL layers, and DTI trenches.',
      'PixelStack은 설정을 솔버 독립적인 3D 구조로 변환합니다. 마이크로렌즈 초타원 프로파일, 베이어 패턴, BARL 층, DTI 트렌치를 처리합니다.'
    ),
    modules: ['pixel_stack.py', 'builder.py', 'MaterialDB'],
  },
  {
    id: 'solver',
    title: t('Solver', '솔버'),
    file: 'SolverBase',
    color: 'var(--arch-solver)',
    accentColor: '#8e44ad',
    x: startX + (boxW + gapX) * 2,
    y: startY,
    items: ['RCWA', 'FDTD', 'TMM'],
    description: t(
      'The SolverBase ABC dispatches to concrete backends (torcwa, grcwa, meent, fmmax, flaport, Meep, fdtdz). Each solver computes fields and diffraction efficiencies with S-matrix stability.',
      'SolverBase ABC가 구체적인 백엔드(torcwa, grcwa, meent, fmmax, flaport, Meep, fdtdz)로 디스패치합니다. 각 솔버는 S-행렬 안정성으로 필드와 회절 효율을 계산합니다.'
    ),
    modules: ['SolverBase', 'SolverFactory', 'stability.py'],
  },
  {
    id: 'analysis',
    title: t('Analysis', '분석'),
    file: 'analysis/',
    color: 'var(--arch-analysis)',
    accentColor: '#e67e22',
    x: startX + (boxW + gapX) * 3,
    y: startY,
    items: [
      t('QE calculator', 'QE 계산'),
      t('Energy balance', '에너지 균형'),
      t('Comparison', '비교'),
    ],
    description: t(
      'Post-processing modules compute quantum efficiency per pixel, validate energy conservation (R+T+A=1), and compare results across solvers with statistical metrics.',
      '후처리 모듈이 픽셀별 양자 효율을 계산하고, 에너지 보존(R+T+A=1)을 검증하며, 통계적 지표로 솔버 간 결과를 비교합니다.'
    ),
    modules: ['QECalculator', 'EnergyBalance', 'SolverComparison'],
  },
  {
    id: 'results',
    title: t('Results', '결과'),
    file: 'HDF5 / Plot',
    color: 'var(--arch-results)',
    accentColor: '#2980b9',
    x: startX + (boxW + gapX) * 4,
    y: startY,
    items: [
      t('QE spectra', 'QE 스펙트럼'),
      t('Field maps', '필드 맵'),
      t('Crosstalk', '크로스토크'),
    ],
    description: t(
      'Results are exported as HDF5 datasets and visualized with built-in plotting (matplotlib) and 3D viewers (PyVista). Structure, field, and QE plots available out of the box.',
      '결과는 HDF5 데이터셋으로 내보내지고 내장 플로팅(matplotlib)과 3D 뷰어(PyVista)로 시각화됩니다. 구조, 필드, QE 플롯을 즉시 사용할 수 있습니다.'
    ),
    modules: ['hdf5_handler.py', 'qe_plot.py', 'field_plot_2d.py'],
  },
])

const svgW = computed(() => startX * 2 + (boxW + gapX) * 4 + boxW)

// Arrows between consecutive steps
const arrows = computed(() => {
  const result = []
  for (let i = 0; i < steps.value.length - 1; i++) {
    const from = steps.value[i]
    const to = steps.value[i + 1]
    result.push({
      x1: from.x + boxW,
      y1: startY + boxH / 2,
      x2: to.x,
      y2: startY + boxH / 2,
    })
  }
  return result
})
</script>

<style scoped>
.arch-overview-container {
  --arch-config: #e8f4fd;
  --arch-geometry: #e8fbe8;
  --arch-solver: #f3e8fd;
  --arch-analysis: #fef3e2;
  --arch-results: #e8f0fe;

  margin: 24px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 20px;
  background: var(--vp-c-bg-soft);
  overflow-x: auto;
}

:root.dark .arch-overview-container {
  --arch-config: #1a2e3d;
  --arch-geometry: #1a3d1a;
  --arch-solver: #2e1a3d;
  --arch-analysis: #3d2e1a;
  --arch-results: #1a2440;
}

.arch-flow {
  overflow-x: auto;
  padding-bottom: 4px;
}

.arch-flow-svg {
  width: 100%;
  min-width: 680px;
  max-width: 780px;
  display: block;
  margin: 0 auto;
}

.flow-arrow {
  transition: stroke 0.2s, stroke-width 0.2s;
}

.step-box {
  transition: stroke 0.2s, stroke-width 0.2s;
}

.step-number {
  font-size: 11px;
  font-weight: 700;
  fill: white;
}

.step-title {
  font-size: 12px;
  font-weight: 700;
  fill: var(--vp-c-text-1);
  pointer-events: none;
}

.step-subtitle {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  pointer-events: none;
}

.sub-item-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  pointer-events: none;
}

/* Detail panel */
.arch-detail-panel {
  margin-top: 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 6px;
  padding: 14px 18px;
}

.arch-detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.arch-detail-header strong {
  font-size: 1em;
  color: var(--vp-c-text-1);
}

.arch-detail-close {
  background: none;
  border: none;
  font-size: 1.3em;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
}

.arch-detail-close:hover {
  color: var(--vp-c-text-1);
}

.arch-detail-desc {
  margin: 8px 0 12px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
  line-height: 1.6;
}

.arch-detail-items {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.arch-detail-items-label {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.arch-detail-tag {
  font-size: 0.78em;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 2px 8px;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-brand-1);
}

.arch-detail-fade-enter-active,
.arch-detail-fade-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}

.arch-detail-fade-enter-from,
.arch-detail-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}
</style>
