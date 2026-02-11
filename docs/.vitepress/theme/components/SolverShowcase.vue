<template>
  <div class="solver-showcase-container">
    <!-- Solver type groups -->
    <div v-for="group in solverGroups" :key="group.type" class="solver-group">
      <div class="group-header">
        <span class="group-badge" :style="{ background: group.color }">{{ group.type }}</span>
        <span class="group-count">{{ group.solvers.length }} {{ t('backends', '백엔드') }}</span>
      </div>

      <div class="solver-badges">
        <button
          v-for="solver in group.solvers"
          :key="solver.name"
          class="solver-badge"
          :class="{ active: selectedSolver === solver.name }"
          @click="toggleSolver(solver.name)"
        >
          <span class="solver-name">{{ solver.name }}</span>
          <span v-if="solver.gpu" class="gpu-tag">GPU</span>
          <span v-if="solver.diffable" class="diff-tag">AD</span>
        </button>
      </div>
    </div>

    <!-- Detail panel -->
    <transition name="solver-detail-fade">
      <div v-if="selectedSolver && selectedDetail" class="solver-detail-panel">
        <div class="solver-detail-header">
          <div class="solver-detail-title-row">
            <strong>{{ selectedDetail.name }}</strong>
            <span class="solver-type-badge" :style="{ background: selectedDetail.groupColor }">
              {{ selectedDetail.type }}
            </span>
            <span v-if="selectedDetail.gpu" class="gpu-tag-lg">GPU</span>
            <span v-if="selectedDetail.diffable" class="diff-tag-lg">{{ t('Differentiable', '미분 가능') }}</span>
          </div>
          <button class="solver-detail-close" @click="selectedSolver = null">&times;</button>
        </div>
        <p class="solver-detail-desc">{{ selectedDetail.description }}</p>
        <div class="solver-detail-meta">
          <div class="meta-item">
            <span class="meta-label">{{ t('Framework:', '프레임워크:') }}</span>
            <code class="meta-value">{{ selectedDetail.framework }}</code>
          </div>
          <div class="meta-item">
            <span class="meta-label">{{ t('Method:', '방법:') }}</span>
            <span class="meta-value">{{ selectedDetail.method }}</span>
          </div>
          <div v-if="selectedDetail.link" class="meta-item">
            <span class="meta-label">{{ t('Library:', '라이브러리:') }}</span>
            <code class="meta-value">{{ selectedDetail.library }}</code>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()
const selectedSolver = ref(null)

function toggleSolver(name) {
  selectedSolver.value = selectedSolver.value === name ? null : name
}

const solvers = computed(() => [
  {
    name: 'torcwa',
    type: 'RCWA',
    gpu: true,
    diffable: true,
    framework: 'PyTorch',
    method: 'S-matrix RCWA',
    library: 'torcwa',
    groupColor: '#3451b2',
    description: t(
      'PyTorch-based RCWA solver with full GPU acceleration and autograd support. Ideal for inverse design workflows and large-scale parameter sweeps.',
      'PyTorch 기반 RCWA 솔버로 완전한 GPU 가속과 autograd를 지원합니다. 역설계 워크플로와 대규모 파라미터 스윕에 적합합니다.'
    ),
  },
  {
    name: 'grcwa',
    type: 'RCWA',
    gpu: true,
    diffable: true,
    framework: 'JAX',
    method: 'S-matrix RCWA',
    library: 'grcwa',
    groupColor: '#3451b2',
    description: t(
      'JAX-based RCWA with JIT compilation and automatic differentiation. Excellent for gradient-based optimization with XLA compilation.',
      'JAX 기반 RCWA로 JIT 컴파일과 자동 미분을 제공합니다. XLA 컴파일을 통한 기울기 기반 최적화에 탁월합니다.'
    ),
  },
  {
    name: 'meent',
    type: 'RCWA',
    gpu: true,
    diffable: true,
    framework: 'PyTorch / NumPy',
    method: 'S-matrix RCWA',
    library: 'meent',
    groupColor: '#3451b2',
    description: t(
      'Flexible RCWA implementation supporting both NumPy (CPU) and PyTorch (GPU) backends. Features enhanced tangential field formulation for metallic gratings.',
      'NumPy(CPU)와 PyTorch(GPU) 백엔드를 모두 지원하는 유연한 RCWA 구현입니다. 금속 격자를 위한 향상된 접선 필드 공식을 제공합니다.'
    ),
  },
  {
    name: 'fmmax',
    type: 'RCWA',
    gpu: true,
    diffable: true,
    framework: 'JAX',
    method: 'FMM / S-matrix',
    library: 'fmmax',
    groupColor: '#3451b2',
    description: t(
      'JAX-based Fourier Modal Method (FMM) solver. High-performance implementation with automatic vectorization and differentiation via JAX transformations.',
      'JAX 기반 푸리에 모달 방법(FMM) 솔버입니다. JAX 변환을 통한 자동 벡터화와 미분으로 고성능 구현을 제공합니다.'
    ),
  },
  {
    name: 'Meep',
    type: 'FDTD',
    gpu: false,
    diffable: true,
    framework: 'C++ / Python',
    method: '3D FDTD',
    library: 'meep',
    groupColor: '#8e44ad',
    description: t(
      'MIT\'s Meep is a mature, full-featured FDTD solver. Supports subpixel averaging, dispersive materials, and adjoint-based gradient computation for inverse design.',
      'MIT의 Meep은 성숙하고 기능이 풍부한 FDTD 솔버입니다. 서브픽셀 평균화, 분산 재료, 역설계를 위한 수반(adjoint) 기반 기울기 계산을 지원합니다.'
    ),
  },
  {
    name: 'flaport',
    type: 'FDTD',
    gpu: true,
    diffable: true,
    framework: 'PyTorch',
    method: '2.5D FDTD',
    library: 'fdtd (flaport)',
    groupColor: '#8e44ad',
    description: t(
      'Lightweight PyTorch-based FDTD with GPU support and automatic differentiation. Well-suited for rapid prototyping and differentiable photonic design.',
      'GPU 지원과 자동 미분을 갖춘 경량 PyTorch 기반 FDTD입니다. 빠른 프로토타이핑과 미분 가능한 광자 설계에 적합합니다.'
    ),
  },
  {
    name: 'fdtdz',
    type: 'FDTD',
    gpu: true,
    diffable: true,
    framework: 'JAX',
    method: '2D FDTD (z-invariant)',
    library: 'fdtdz',
    groupColor: '#8e44ad',
    description: t(
      'JAX-based 2D FDTD for z-invariant structures. Extremely fast for problems reducible to 2D cross-sections, with full JAX differentiation support.',
      'z-불변 구조를 위한 JAX 기반 2D FDTD입니다. 2D 단면으로 축소 가능한 문제에 매우 빠르며, 완전한 JAX 미분을 지원합니다.'
    ),
  },
  {
    name: 'TMM',
    type: 'TMM',
    gpu: false,
    diffable: false,
    framework: 'NumPy',
    method: 'Transfer Matrix Method',
    library: 'compass.solvers.tmm',
    groupColor: '#e67e22',
    description: t(
      'Built-in Transfer Matrix Method for planar multilayer stacks. Ultra-fast analytical solution for 1D problems -- useful for BARL optimization and quick sanity checks.',
      '평면 다층 스택을 위한 내장 전달 행렬 방법입니다. 1D 문제에 대한 초고속 해석적 솔루션으로, BARL 최적화와 빠른 유효성 검사에 유용합니다.'
    ),
  },
])

const solverGroups = computed(() => {
  const groups = [
    {
      type: 'RCWA',
      color: '#3451b2',
      solvers: solvers.value.filter(s => s.type === 'RCWA'),
    },
    {
      type: 'FDTD',
      color: '#8e44ad',
      solvers: solvers.value.filter(s => s.type === 'FDTD'),
    },
    {
      type: 'TMM',
      color: '#e67e22',
      solvers: solvers.value.filter(s => s.type === 'TMM'),
    },
  ]
  return groups
})

const selectedDetail = computed(() => {
  if (!selectedSolver.value) return null
  return solvers.value.find(s => s.name === selectedSolver.value) || null
})
</script>

<style scoped>
.solver-showcase-container {
  margin: 24px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  background: var(--vp-c-bg-soft);
}

.solver-group {
  margin-bottom: 20px;
}

.solver-group:last-child {
  margin-bottom: 0;
}

.group-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.group-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 12px;
  font-size: 0.8em;
  font-weight: 700;
  color: white;
  letter-spacing: 0.5px;
}

.group-count {
  font-size: 0.82em;
  color: var(--vp-c-text-3);
}

.solver-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.solver-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
  font-family: inherit;
  color: var(--vp-c-text-1);
}

.solver-badge:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 2px 8px rgba(52, 81, 178, 0.08);
  transform: translateY(-1px);
}

.solver-badge.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  box-shadow: 0 2px 12px rgba(52, 81, 178, 0.12);
}

:root.dark .solver-badge:hover {
  box-shadow: 0 2px 8px rgba(107, 141, 229, 0.1);
}

:root.dark .solver-badge.active {
  box-shadow: 0 2px 12px rgba(107, 141, 229, 0.15);
}

.solver-name {
  font-size: 0.92em;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}

.gpu-tag,
.diff-tag {
  font-size: 0.68em;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 4px;
  letter-spacing: 0.3px;
}

.gpu-tag {
  background: #27ae60;
  color: white;
}

.diff-tag {
  background: #2980b9;
  color: white;
}

/* Detail panel */
.solver-detail-panel {
  margin-top: 20px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 8px;
  padding: 16px 20px;
}

.solver-detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.solver-detail-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.solver-detail-title-row strong {
  font-size: 1.1em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}

.solver-type-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 8px;
  font-size: 0.72em;
  font-weight: 700;
  color: white;
}

.gpu-tag-lg {
  font-size: 0.72em;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 8px;
  background: #27ae60;
  color: white;
}

.diff-tag-lg {
  font-size: 0.72em;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 8px;
  background: #2980b9;
  color: white;
}

.solver-detail-close {
  background: none;
  border: none;
  font-size: 1.4em;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
}

.solver-detail-close:hover {
  color: var(--vp-c-text-1);
}

.solver-detail-desc {
  margin: 10px 0 14px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
  line-height: 1.6;
}

.solver-detail-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.meta-label {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.meta-value {
  font-size: 0.82em;
  color: var(--vp-c-text-1);
}

code.meta-value {
  font-family: var(--vp-font-family-mono);
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  padding: 1px 6px;
  color: var(--vp-c-brand-1);
}

.solver-detail-fade-enter-active,
.solver-detail-fade-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}

.solver-detail-fade-enter-from,
.solver-detail-fade-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

@media (max-width: 640px) {
  .solver-showcase-container {
    padding: 16px;
  }
  .solver-detail-meta {
    flex-direction: column;
    gap: 8px;
  }
}
</style>
