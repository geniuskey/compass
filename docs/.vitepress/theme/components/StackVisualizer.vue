<template>
  <div class="stack-container">
    <h4>{{ t('Interactive BSI Pixel Stack Cross-Section', '인터랙티브 BSI 픽셀 스택 단면') }}</h4>
    <p class="component-description">
      {{ t(
        'Click on any layer to view its material properties and role in the pixel stack.',
        '스택 다이어그램에서 레이어를 클릭하면 해당 재료 특성과 픽셀 스택에서의 역할을 확인할 수 있습니다.'
      ) }}
    </p>

    <div class="stack-layout">
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="stack-svg">
          <!-- Incident light arrow -->
          <defs>
            <marker id="lightArrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#f39c12" />
            </marker>
            <!-- Microlens gradient -->
            <radialGradient id="lensGrad" cx="50%" cy="100%" r="80%">
              <stop offset="0%" stop-color="#d4e6f1" stop-opacity="0.9" />
              <stop offset="100%" stop-color="#aed6f1" stop-opacity="0.6" />
            </radialGradient>
          </defs>

          <line x1="180" y1="5" x2="180" y2="35" stroke="#f39c12" stroke-width="2" marker-end="url(#lightArrow)" />
          <text x="200" y="15" class="light-label">{{ t('Incident light', '입사광') }}</text>

          <!-- Layers -->
          <template v-for="(layer, idx) in layers" :key="layer.id">
            <g
              :class="['layer-group', { selected: selectedId === layer.id }]"
              @click="selectLayer(layer.id)"
              style="cursor: pointer"
            >
              <!-- Layer rectangle (air extends through microlens space; microlens rect hidden) -->
              <rect
                :x="layerX"
                :y="layerPositions[idx].y"
                :width="layerW"
                :height="layer.id === 'air' ? layerPositions[0].h + layerPositions[1].h : layerPositions[idx].h"
                :fill="layer.id === 'microlens' ? 'transparent' : layer.color"
                :stroke="layer.id === 'microlens' ? 'none' : (selectedId === layer.id ? 'var(--vp-c-brand-1)' : '#555')"
                :stroke-width="selectedId === layer.id ? 2.5 : 1"
                rx="2"
                :opacity="layer.opacity || 1"
              />

              <!-- Microlens dome (half-ellipse: base at bottom, dome curves upward) -->
              <path
                v-if="layer.id === 'microlens'"
                :d="`M ${layerX + 5} ${layerPositions[idx].y + layerPositions[idx].h} A ${layerW / 2 - 5} ${layerPositions[idx].h * 0.85} 0 0 1 ${layerX + layerW - 5} ${layerPositions[idx].y + layerPositions[idx].h} Z`"
                fill="url(#lensGrad)"
                stroke="#7fb3d3"
                stroke-width="1"
                style="pointer-events: none"
              />

              <!-- DTI trenches in silicon -->
              <template v-if="layer.id === 'silicon'">
                <rect
                  :x="layerX + 5"
                  :y="layerPositions[idx].y + 2"
                  width="12"
                  :height="layerPositions[idx].h - 4"
                  fill="#aed6f1"
                  opacity="0.8"
                  rx="1"
                  style="pointer-events: none"
                />
                <rect
                  :x="layerX + layerW - 17"
                  :y="layerPositions[idx].y + 2"
                  width="12"
                  :height="layerPositions[idx].h - 4"
                  fill="#aed6f1"
                  opacity="0.8"
                  rx="1"
                  style="pointer-events: none"
                />
                <text
                  :x="layerX + 11"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="dti-label"
                >DTI</text>
                <text
                  :x="layerX + layerW - 11"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="dti-label"
                >DTI</text>
              </template>

              <!-- Layer name (center) -->
              <text
                :x="180"
                :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 4"
                text-anchor="middle"
                :class="['layer-name', { 'dark-text': layer.lightText }]"
              >{{ layer.name }}</text>

              <!-- Thickness label (right side) -->
              <line
                :x1="layerX + layerW + 8"
                :y1="layerPositions[idx].y"
                :x2="layerX + layerW + 8"
                :y2="layerPositions[idx].y + layerPositions[idx].h"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.8"
              />
              <line
                :x1="layerX + layerW + 5"
                :y1="layerPositions[idx].y"
                :x2="layerX + layerW + 11"
                :y2="layerPositions[idx].y"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.8"
              />
              <line
                :x1="layerX + layerW + 5"
                :y1="layerPositions[idx].y + layerPositions[idx].h"
                :x2="layerX + layerW + 11"
                :y2="layerPositions[idx].y + layerPositions[idx].h"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.8"
              />
              <text
                :x="layerX + layerW + 15"
                :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                text-anchor="start"
                class="thickness-label"
              >{{ layer.thickness }}</text>
            </g>
          </template>
        </svg>
      </div>

      <div class="details-panel">
        <div v-if="selectedLayer" class="detail-card">
          <h5>{{ selectedLayer.name }}</h5>
          <table class="props-table">
            <tr>
              <td class="prop-key">{{ t('Material', '재료') }}</td>
              <td class="prop-val">{{ selectedLayer.material }}</td>
            </tr>
            <tr>
              <td class="prop-key">{{ t('Typical thickness', '일반적인 두께') }}</td>
              <td class="prop-val">{{ selectedLayer.thickness }}</td>
            </tr>
            <tr>
              <td class="prop-key">{{ t('Refractive index', '굴절률') }} (n)</td>
              <td class="prop-val">{{ selectedLayer.n }}</td>
            </tr>
            <tr v-if="selectedLayer.k">
              <td class="prop-key">{{ t('Extinction coeff.', '소광 계수') }} (k)</td>
              <td class="prop-val">{{ selectedLayer.k }}</td>
            </tr>
            <tr>
              <td class="prop-key">{{ t('Function', '기능') }}</td>
              <td class="prop-val">{{ t(selectedLayer.role, selectedLayer.roleKo) }}</td>
            </tr>
          </table>
          <p class="detail-desc">{{ t(selectedLayer.description, selectedLayer.descriptionKo) }}</p>
        </div>
        <div v-else class="detail-placeholder">
          {{ t(
            'Click a layer in the stack diagram to see its properties.',
            '스택 다이어그램에서 레이어를 클릭하면 속성을 확인할 수 있습니다.'
          ) }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const svgW = 340
const svgH = 420
const layerX = 80
const layerW = 200
const startY = 42

const layers = [
  {
    id: 'air',
    name: 'Air',
    color: '#e8f4fd',
    thickness: '(semi-infinite)',
    material: 'Air / vacuum',
    n: '1.000',
    k: null,
    role: 'Incident medium',
    roleKo: '입사 매질',
    description: 'The medium above the pixel. For on-chip applications this is air (n=1). In some configurations an encapsulant (n~1.5) may be present. Defines the reference for incidence angle and Fresnel reflection at the top surface.',
    descriptionKo: '픽셀 위의 매질입니다. 온칩 응용에서는 공기(n=1)입니다. 일부 구성에서는 봉지재(n~1.5)가 존재할 수 있습니다. 입사각과 상부 표면에서의 프레넬 반사에 대한 기준을 정의합니다.',
    heightFrac: 0.06,
    lightText: false,
  },
  {
    id: 'microlens',
    name: 'Microlens',
    color: '#aed6f1',
    opacity: 0.7,
    thickness: '0.4 - 0.8 um',
    material: 'Polymer (styrene-based)',
    n: '1.50 - 1.58',
    k: null,
    role: 'Light focusing',
    roleKo: '집광',
    description: 'Curved polymer structure that focuses incoming light toward the center of the pixel photodiode. Modeled in COMPASS as a superellipse profile. The lens center can be shifted to compensate for Chief Ray Angle (CRA) at the sensor edge.',
    descriptionKo: '입사광을 픽셀 포토다이오드 중심으로 집중시키는 곡면 폴리머 구조입니다. COMPASS에서 초타원 프로파일로 모델링됩니다. 센서 가장자리에서 주광선 각도(CRA)를 보정하기 위해 렌즈 중심을 이동할 수 있습니다.',
    heightFrac: 0.10,
    lightText: false,
  },
  {
    id: 'planarization',
    name: 'Planarization',
    color: '#d5dbdb',
    thickness: '0.3 - 1.0 um',
    material: 'SiO2',
    n: '1.46',
    k: null,
    role: 'Gap fill / planarization',
    roleKo: '갭 충전 / 평탄화',
    description: 'A uniform dielectric layer (usually SiO2) that fills the space between the microlens and color filter array. Provides a flat surface for the color filter deposition and acts as a spacer that controls the microlens focal distance.',
    descriptionKo: '마이크로렌즈와 컬러 필터 배열 사이 공간을 채우는 균일한 유전체층(보통 SiO2)입니다. 컬러 필터 증착을 위한 평탄한 표면을 제공하며, 마이크로렌즈 초점 거리를 제어하는 스페이서 역할을 합니다.',
    heightFrac: 0.10,
    lightText: false,
  },
  {
    id: 'colorfilter',
    name: 'Color Filter',
    color: '#27ae60',
    thickness: '0.4 - 0.8 um',
    material: 'Organic dye (R/G/B)',
    n: '1.55 - 1.70',
    k: '0.0 - 0.5 (passband/stopband)',
    role: 'Wavelength selection',
    roleKo: '파장 선택',
    description: 'Organic dye-based filter arranged in a Bayer RGGB pattern. Each sub-pixel absorbs unwanted wavelengths while transmitting its target color. The metal grid (typically tungsten, 40-80 nm wide) between color filter elements provides optical isolation to reduce crosstalk.',
    descriptionKo: '베이어 RGGB 패턴으로 배열된 유기 염료 기반 필터입니다. 각 서브 픽셀은 불필요한 파장을 흡수하고 목표 색상만 투과시킵니다. 컬러 필터 요소 사이의 금속 격자(일반적으로 텅스텐, 40-80 nm 폭)가 크로스토크를 줄이기 위한 광학 차단 역할을 합니다.',
    heightFrac: 0.12,
    lightText: false,
  },
  {
    id: 'barl',
    name: 'BARL',
    color: '#8e44ad',
    thickness: '0.05 - 0.12 um',
    material: 'SiO2 / HfO2 / Si3N4 stack',
    n: '1.46 - 2.05 (varies by sub-layer)',
    k: '~0 (dielectric)',
    role: 'Anti-reflection',
    roleKo: '반사 방지',
    description: 'Bottom Anti-Reflection Layer: a multi-layer dielectric stack (typically 2-5 sub-layers of SiO2, HfO2, Si3N4) designed to minimize reflection at the color-filter-to-silicon interface. Without BARL, the large index mismatch (n~1.55 to n~4.0) causes ~20-30% reflection loss.',
    descriptionKo: '하부 반사 방지층: 컬러 필터-실리콘 계면에서의 반사를 최소화하기 위해 설계된 다층 유전체 스택(일반적으로 SiO2, HfO2, Si3N4의 2-5개 하위층)입니다. BARL이 없으면 큰 굴절률 불일치(n~1.55에서 n~4.0)로 인해 약 20-30%의 반사 손실이 발생합니다.',
    heightFrac: 0.05,
    lightText: true,
  },
  {
    id: 'silicon',
    name: 'Silicon',
    color: '#5d6d7e',
    thickness: '2.0 - 4.0 um',
    material: 'Crystalline Si',
    n: '3.5 - 4.3 (wavelength dependent)',
    k: '0.003 - 2.2 (wavelength dependent)',
    role: 'Photon absorption & charge generation',
    roleKo: '광자 흡수 및 전하 생성',
    description: 'The active absorbing layer where photons generate electron-hole pairs. Deep Trench Isolation (DTI) trenches filled with SiO2 (n~1.46) optically isolate adjacent pixels via total internal reflection. The photodiode occupies a defined region within the silicon volume. COMPASS integrates absorbed power within the photodiode bounding box to compute QE.',
    descriptionKo: '광자가 전자-정공 쌍을 생성하는 활성 흡수층입니다. SiO2(n~1.46)로 채워진 Deep Trench Isolation(DTI) 트렌치가 전반사를 통해 인접 픽셀을 광학적으로 차단합니다. 포토다이오드는 실리콘 내 정의된 영역을 차지합니다. COMPASS는 포토다이오드 경계 상자 내에서 흡수된 전력을 적분하여 QE를 계산합니다.',
    heightFrac: 0.38,
    lightText: true,
  },
  {
    id: 'substrate',
    name: 'Substrate / Metal',
    color: '#2c3e50',
    thickness: '(semi-infinite)',
    material: 'Si substrate + metal wiring (Cu/W)',
    n: 'N/A (absorbing boundary)',
    k: null,
    role: 'Mechanical support & interconnect',
    roleKo: '기계적 지지 및 배선',
    description: 'In a BSI sensor, the metal wiring and transistors are on this side (opposite to light entry). Acts as a reflecting/absorbing boundary in the simulation. Any photons reaching this layer are either absorbed by metal or reflected back into the silicon. COMPASS typically models this as a perfectly matched layer (PML) or fixed boundary condition.',
    descriptionKo: 'BSI 센서에서 금속 배선과 트랜지스터는 이 쪽(빛 입사 반대편)에 위치합니다. 시뮬레이션에서 반사/흡수 경계 역할을 합니다. 이 층에 도달하는 광자는 금속에 흡수되거나 실리콘으로 다시 반사됩니다. COMPASS는 일반적으로 이를 완전 정합층(PML) 또는 고정 경계 조건으로 모델링합니다.',
    heightFrac: 0.10,
    lightText: true,
  },
]

// Compute vertical positions for each layer
const totalFrac = layers.reduce((s, l) => s + l.heightFrac, 0)
const availableH = svgH - startY - 20

const layerPositions = computed(() => {
  const positions = []
  let currentY = startY
  for (const layer of layers) {
    const h = (layer.heightFrac / totalFrac) * availableH
    positions.push({ y: currentY, h })
    currentY += h
  }
  return positions
})

const selectedId = ref(null)

function selectLayer(id) {
  selectedId.value = selectedId.value === id ? null : id
}

const selectedLayer = computed(() => {
  if (!selectedId.value) return null
  return layers.find((l) => l.id === selectedId.value) || null
})
</script>

<style scoped>
.stack-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.stack-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.stack-layout {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.svg-wrapper {
  flex: 0 0 auto;
}
.stack-svg {
  width: 340px;
  max-width: 100%;
}
.details-panel {
  flex: 1;
  min-width: 240px;
}
.layer-group:hover rect {
  filter: brightness(1.1);
}
.layer-group.selected rect {
  filter: brightness(1.05);
}
.layer-name {
  font-size: 11px;
  fill: #2c3e50;
  font-weight: 600;
  pointer-events: none;
}
.layer-name.dark-text {
  fill: #ecf0f1;
}
.thickness-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.dti-label {
  font-size: 7px;
  fill: #2c3e50;
  font-weight: 600;
  pointer-events: none;
}
.light-label {
  font-size: 10px;
  fill: #f39c12;
}
.detail-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 18px;
}
.detail-card h5 {
  margin: 0 0 12px 0;
  font-size: 1.05em;
  color: var(--vp-c-brand-1);
}
.props-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 12px;
}
.props-table td {
  padding: 5px 8px;
  font-size: 0.88em;
  border-bottom: 1px solid var(--vp-c-divider);
}
.prop-key {
  color: var(--vp-c-text-2);
  white-space: nowrap;
  width: 40%;
}
.prop-val {
  font-family: var(--vp-font-family-mono);
  font-size: 0.85em;
  color: var(--vp-c-text-1);
}
.detail-desc {
  font-size: 0.88em;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin: 0;
}
.detail-placeholder {
  background: var(--vp-c-bg);
  border: 1px dashed var(--vp-c-divider);
  border-radius: 10px;
  padding: 30px 20px;
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 0.9em;
}
</style>
