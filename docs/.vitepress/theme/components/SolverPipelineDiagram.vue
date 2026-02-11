<template>
  <div class="solver-pipeline-container">
    <h4>{{ t('Solver Pipeline: Abstract Methods', '솔버 파이프라인: 추상 메서드') }}</h4>
    <p class="component-description">
      {{ t('Click on each step to see the method signature and data types. Toggle between RCWA and FDTD paths.', '각 단계를 클릭하여 메서드 시그니처와 데이터 타입을 확인하세요. RCWA와 FDTD 경로를 전환할 수 있습니다.') }}
    </p>

    <!-- Solver type toggle -->
    <div class="toggle-row">
      <button
        class="toggle-pill"
        :class="{ active: solverType === 'rcwa' }"
        @click="solverType = 'rcwa'"
      >RCWA</button>
      <button
        class="toggle-pill"
        :class="{ active: solverType === 'fdtd' }"
        @click="solverType = 'fdtd'"
      >FDTD</button>
    </div>

    <!-- Horizontal flow diagram -->
    <div class="flow-row">
      <div
        v-for="(step, idx) in steps"
        :key="step.id"
        class="flow-block"
        :class="{ active: selectedStep === step.id }"
        @click="selectedStep = selectedStep === step.id ? null : step.id"
      >
        <div class="flow-label">{{ step.label }}</div>
        <div class="flow-sub">{{ step.sub }}</div>
        <div v-if="idx < steps.length - 1" class="flow-arrow">&rarr;</div>
      </div>
    </div>

    <!-- Step detail panel -->
    <div v-if="selectedStep" class="step-detail">
      <div class="detail-signature">
        <code>{{ stepDetails[selectedStep].signature }}</code>
      </div>
      <div class="detail-grid">
        <div class="detail-item">
          <span class="detail-label">{{ t('Input', '입력') }}:</span>
          <span class="detail-value">{{ stepDetails[selectedStep].input }}</span>
        </div>
        <div class="detail-item">
          <span class="detail-label">{{ t('Output', '출력') }}:</span>
          <span class="detail-value">{{ stepDetails[selectedStep].output }}</span>
        </div>
        <div class="detail-item detail-desc">
          <span class="detail-label">{{ t('Description', '설명') }}:</span>
          <span class="detail-value">{{ stepDetails[selectedStep].description }}</span>
        </div>
      </div>
    </div>

    <!-- SolverFactory section -->
    <div class="factory-section">
      <div class="factory-box">
        <code>SolverFactory.create(name, config, device)</code>
      </div>
      <div class="backends-row">
        <div
          v-for="backend in backends"
          :key="backend.name"
          class="backend-card"
          :class="{ highlighted: backend.type === solverType }"
        >
          <div class="backend-name">{{ backend.name }}</div>
          <div class="backend-type">{{ backend.typeLabel }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

const solverType = ref('rcwa')
const selectedStep = ref(null)

const steps = computed(() => [
  {
    id: 'setup_geometry',
    label: 'setup_geometry(pixel_stack)',
    sub: solverType.value === 'rcwa'
      ? t('calls pixel_stack.get_layer_slices()', 'pixel_stack.get_layer_slices() 호출')
      : t('calls pixel_stack.get_permittivity_grid()', 'pixel_stack.get_permittivity_grid() 호출'),
  },
  {
    id: 'setup_source',
    label: 'setup_source(source_config)',
    sub: solverType.value === 'rcwa'
      ? t('2D eps grids per layer', '레이어별 2D 유전율 그리드')
      : t('3D voxel grid', '3D 복셀 그리드'),
  },
  {
    id: 'run',
    label: 'run()',
    sub: t('SimulationResult', 'SimulationResult'),
  },
  {
    id: 'get_field',
    label: 'get_field_distribution(...)',
    sub: t('component, plane, position', 'component, plane, position'),
  },
])

const stepDetails = computed(() => ({
  setup_geometry: {
    signature: 'def setup_geometry(self, pixel_stack: PixelStack) -> None',
    input: 'PixelStack',
    output: 'None',
    description: t(
      'Converts the solver-agnostic PixelStack into the internal representation required by the specific backend. RCWA backends call get_layer_slices() to obtain 2D permittivity grids per layer. FDTD backends call get_permittivity_grid() to obtain a full 3D voxel grid.',
      '솔버에 독립적인 PixelStack을 특정 백엔드에 필요한 내부 표현으로 변환합니다. RCWA 백엔드는 get_layer_slices()를 호출하여 레이어별 2D 유전율 그리드를 얻습니다. FDTD 백엔드는 get_permittivity_grid()를 호출하여 전체 3D 복셀 그리드를 얻습니다.'
    ),
  },
  setup_source: {
    signature: 'def setup_source(self, source_config: dict) -> None',
    input: 'dict',
    output: 'None',
    description: t(
      'Configures the electromagnetic source parameters including wavelength, incidence angle, and polarization state for the simulation.',
      '시뮬레이션을 위한 파장, 입사각, 편광 상태를 포함한 전자기파 소스 파라미터를 구성합니다.'
    ),
  },
  run: {
    signature: 'def run(self) -> SimulationResult',
    input: t('None (uses configured geometry and source)', 'None (구성된 구조와 소스 사용)'),
    output: 'SimulationResult',
    description: t(
      'Executes the electromagnetic simulation and returns a SimulationResult containing reflection, transmission, absorption coefficients, and optional field data.',
      '전자기 시뮬레이션을 실행하고 반사, 투과, 흡수 계수 및 선택적 필드 데이터를 포함하는 SimulationResult를 반환합니다.'
    ),
  },
  get_field: {
    signature: 'def get_field_distribution(self, component: str, plane: str, position: float) -> np.ndarray',
    input: 'component: str, plane: str, position: float',
    output: 'np.ndarray',
    description: t(
      'Extracts a 2D field slice from the simulation results. Component specifies the field (Ex, Ey, Ez, Hx, Hy, Hz), plane specifies the cross-section orientation (xy, xz, yz), and position specifies the coordinate along the normal axis.',
      '시뮬레이션 결과에서 2D 필드 슬라이스를 추출합니다. component는 필드(Ex, Ey, Ez, Hx, Hy, Hz)를, plane은 단면 방향(xy, xz, yz)을, position은 법선축 좌표를 지정합니다.'
    ),
  },
}))

const backends = [
  { name: 'torcwa', type: 'rcwa', typeLabel: 'RCWA' },
  { name: 'grcwa', type: 'rcwa', typeLabel: 'RCWA' },
  { name: 'meent', type: 'rcwa', typeLabel: 'RCWA' },
  { name: 'fdtd_flaport', type: 'fdtd', typeLabel: 'FDTD' },
]
</script>

<style scoped>
.solver-pipeline-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.solver-pipeline-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

/* Toggle pills */
.toggle-row {
  display: flex;
  gap: 0;
  margin-bottom: 16px;
}
.toggle-pill {
  padding: 6px 20px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.toggle-pill:first-child {
  border-radius: 6px 0 0 6px;
}
.toggle-pill:last-child {
  border-radius: 0 6px 6px 0;
  border-left: none;
}
.toggle-pill.active {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #fff;
}

/* Flow diagram */
.flow-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.flow-block {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  cursor: pointer;
  transition: all 0.15s;
  min-width: 80px;
  margin-right: 20px;
}
.flow-block:last-child {
  margin-right: 0;
}
.flow-block:hover,
.flow-block.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.flow-label {
  font-size: 0.72em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  text-align: center;
}
.flow-sub {
  font-size: 0.65em;
  color: var(--vp-c-text-2);
  margin-top: 2px;
  text-align: center;
}
.flow-arrow {
  position: absolute;
  right: -14px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--vp-c-text-2);
  font-size: 1.1em;
  z-index: 1;
}

/* Step detail panel */
.step-detail {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 16px;
}
.detail-signature {
  margin-bottom: 10px;
}
.detail-signature code {
  font-size: 0.82em;
  background: var(--vp-c-bg-soft);
  padding: 4px 8px;
  border-radius: 4px;
  color: var(--vp-c-brand-1);
  display: inline-block;
  word-break: break-all;
}
.detail-grid {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.detail-item {
  font-size: 0.82em;
  display: flex;
  gap: 6px;
}
.detail-desc {
  flex-direction: column;
  gap: 2px;
}
.detail-label {
  font-weight: 600;
  color: var(--vp-c-text-1);
  white-space: nowrap;
}
.detail-value {
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.95em;
}
.detail-desc .detail-value {
  font-family: inherit;
}

/* SolverFactory section */
.factory-section {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--vp-c-divider);
}
.factory-box {
  text-align: center;
  margin-bottom: 12px;
}
.factory-box code {
  font-size: 0.85em;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  padding: 6px 14px;
  border-radius: 6px;
  color: var(--vp-c-text-1);
  display: inline-block;
}
.backends-row {
  display: flex;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
}
.backend-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 8px 14px;
  background: var(--vp-c-bg);
  text-align: center;
  transition: all 0.15s;
  min-width: 90px;
}
.backend-card.highlighted {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.backend-name {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.backend-type {
  font-size: 0.7em;
  color: var(--vp-c-text-2);
  margin-top: 2px;
}
</style>
