<template>
  <div class="pixel-anatomy-container">
    <h4>{{ t('BSI Pixel Cross-Section Anatomy', 'BSI 픽셀 단면 구조') }}</h4>
    <p class="component-description">
      {{ t(
        'Click on any layer to highlight it and view a detailed description below. An animated light ray traces the optical path.',
        '레이어를 클릭하면 강조 표시되고 아래에 상세 설명이 나타납니다. 애니메이션 광선이 광학 경로를 추적합니다.'
      ) }}
    </p>

    <div class="anatomy-layout">
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="anatomy-svg">
          <defs>
            <!-- Light arrow marker -->
            <marker id="paLightArrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#f5c842" />
            </marker>
            <!-- Microlens gradients -->
            <radialGradient id="paLensGradL" cx="50%" cy="100%" r="80%">
              <stop offset="0%" stop-color="#d8c4e8" stop-opacity="0.95" />
              <stop offset="100%" stop-color="#c3a6db" stop-opacity="0.6" />
            </radialGradient>
            <radialGradient id="paLensGradR" cx="50%" cy="100%" r="80%">
              <stop offset="0%" stop-color="#d8c4e8" stop-opacity="0.95" />
              <stop offset="100%" stop-color="#c3a6db" stop-opacity="0.6" />
            </radialGradient>
          </defs>

          <!-- Incoming light arrow -->
          <line x1="200" y1="4" x2="200" y2="30" stroke="#f5c842" stroke-width="2.5" marker-end="url(#paLightArrow)" />
          <line x1="260" y1="4" x2="250" y2="30" stroke="#f5c842" stroke-width="1.8" marker-end="url(#paLightArrow)" opacity="0.6" />
          <line x1="140" y1="4" x2="150" y2="30" stroke="#f5c842" stroke-width="1.8" marker-end="url(#paLightArrow)" opacity="0.6" />
          <text x="305" y="14" class="light-label">{{ t('Incident light (-z)', '입사광 (-z)') }}</text>

          <!-- Animated light ray through stack -->
          <line
            class="light-ray"
            x1="200" y1="34"
            x2="200" :y2="svgH - 20"
            stroke="#f5c842"
            stroke-width="1.5"
            stroke-dasharray="6,4"
            opacity="0.7"
          />

          <!-- Layer groups -->
          <template v-for="(layer, idx) in layers" :key="layer.id">
            <g
              :class="['pa-layer-group', { selected: selectedId === layer.id }]"
              @click="selectLayer(layer.id)"
              @mouseenter="hoveredId = layer.id"
              @mouseleave="hoveredId = null"
              style="cursor: pointer"
            >
              <!-- Main layer rectangle(s) -->
              <!-- Air layer — extends through microlens space (air surrounds the domes) -->
              <rect
                v-if="layer.id === 'air'"
                :x="leftPixelX"
                :y="layerPositions[idx].y"
                :width="totalPixelW"
                :height="layerPositions[0].h + layerPositions[1].h"
                :fill="layer.color"
                :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#aaa'"
                :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                :opacity="isHighlighted(layer.id) ? 1 : 0.7"
                rx="1"
              />

              <!-- Microlens — two half-ellipse domes side by side -->
              <template v-if="layer.id === 'microlens'">
                <!-- Transparent click target -->
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="totalPixelW"
                  :height="layerPositions[idx].h"
                  fill="transparent"
                />
                <!-- Left dome (half-ellipse: base at bottom, dome curves upward) -->
                <path
                  :d="`M ${leftPixelX + 4} ${layerPositions[idx].y + layerPositions[idx].h} A ${pixelW / 2 - 4} ${layerPositions[idx].h * 0.85} 0 0 1 ${leftPixelX + pixelW - 4} ${layerPositions[idx].y + layerPositions[idx].h} Z`"
                  fill="url(#paLensGradL)"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#a88bc7'"
                  stroke-width="1"
                  style="pointer-events: none"
                />
                <!-- Right dome (half-ellipse: base at bottom, dome curves upward) -->
                <path
                  :d="`M ${leftPixelX + pixelW + gapW + 4} ${layerPositions[idx].y + layerPositions[idx].h} A ${pixelW / 2 - 4} ${layerPositions[idx].h * 0.85} 0 0 1 ${leftPixelX + pixelW + gapW + pixelW - 4} ${layerPositions[idx].y + layerPositions[idx].h} Z`"
                  fill="url(#paLensGradR)"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#a88bc7'"
                  stroke-width="1"
                  style="pointer-events: none"
                />
              </template>

              <!-- Planarization -->
              <rect
                v-if="layer.id === 'planarization'"
                :x="leftPixelX"
                :y="layerPositions[idx].y"
                :width="totalPixelW"
                :height="layerPositions[idx].h"
                :fill="layer.color"
                :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#aaa'"
                :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                :opacity="isHighlighted(layer.id) ? 1 : 0.8"
                rx="1"
              />

              <!-- Color Filter — two side by side (green + red) with metal grid between -->
              <template v-if="layer.id === 'colorfilter'">
                <!-- Green pixel -->
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="pixelW"
                  :height="layerPositions[idx].h"
                  fill="#27ae60"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#555'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  rx="1"
                />
                <text
                  :x="leftPixelX + pixelW / 2"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="cf-label"
                >G</text>
                <!-- Red pixel -->
                <rect
                  :x="leftPixelX + pixelW + gapW"
                  :y="layerPositions[idx].y"
                  :width="pixelW"
                  :height="layerPositions[idx].h"
                  fill="#c0392b"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#555'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  rx="1"
                />
                <text
                  :x="leftPixelX + pixelW + gapW + pixelW / 2"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="cf-label"
                >R</text>
              </template>

              <!-- Metal Grid — vertical lines between color filters -->
              <template v-if="layer.id === 'metalgrid'">
                <!-- Span full width at this layer's y position -->
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="totalPixelW"
                  :height="layerPositions[idx].h"
                  fill="transparent"
                  style="pointer-events: all"
                />
                <!-- Left boundary -->
                <rect
                  :x="leftPixelX - 2"
                  :y="layerPositions[idx].y"
                  width="6"
                  :height="layerPositions[idx].h"
                  :fill="layer.color"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : 'none'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0"
                />
                <!-- Center grid -->
                <rect
                  :x="leftPixelX + pixelW - 1"
                  :y="layerPositions[idx].y"
                  :width="gapW + 2"
                  :height="layerPositions[idx].h"
                  :fill="layer.color"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : 'none'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0"
                />
                <!-- Right boundary -->
                <rect
                  :x="leftPixelX + totalPixelW - 4"
                  :y="layerPositions[idx].y"
                  width="6"
                  :height="layerPositions[idx].h"
                  :fill="layer.color"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : 'none'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0"
                />
              </template>

              <!-- BARL — alternating thin layers -->
              <template v-if="layer.id === 'barl'">
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="totalPixelW"
                  :height="layerPositions[idx].h"
                  :fill="layer.color"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#888'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.8"
                  rx="1"
                />
                <!-- Alternating sub-layers -->
                <rect
                  v-for="j in 3"
                  :key="'barl-' + j"
                  :x="leftPixelX + 1"
                  :y="layerPositions[idx].y + (j - 1) * (layerPositions[idx].h / 3)"
                  :width="totalPixelW - 2"
                  :height="layerPositions[idx].h / 6"
                  fill="#e8f0fe"
                  opacity="0.5"
                  style="pointer-events: none"
                />
              </template>

              <!-- Silicon — thick layer with photodiode region -->
              <template v-if="layer.id === 'silicon'">
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="totalPixelW"
                  :height="layerPositions[idx].h"
                  :fill="layer.color"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#555'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.9"
                  rx="1"
                />
              </template>

              <!-- Photodiode — lighter region inside silicon -->
              <template v-if="layer.id === 'photodiode'">
                <!-- Left photodiode -->
                <rect
                  :x="leftPixelX + 20"
                  :y="layerPositions[idx].y"
                  :width="pixelW - 40"
                  :height="layerPositions[idx].h"
                  fill="#b85c5c"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#884444'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 1"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.8"
                  rx="3"
                  stroke-dasharray="4,2"
                />
                <text
                  :x="leftPixelX + pixelW / 2"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="pd-label"
                >PD</text>
                <!-- Right photodiode -->
                <rect
                  :x="leftPixelX + pixelW + gapW + 20"
                  :y="layerPositions[idx].y"
                  :width="pixelW - 40"
                  :height="layerPositions[idx].h"
                  fill="#b85c5c"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#884444'"
                  :stroke-width="isHighlighted(layer.id) ? 2.5 : 1"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.8"
                  rx="3"
                  stroke-dasharray="4,2"
                />
                <text
                  :x="leftPixelX + pixelW + gapW + pixelW / 2"
                  :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                  text-anchor="middle"
                  class="pd-label"
                >PD</text>
              </template>

              <!-- DTI — white vertical lines at pixel boundaries inside silicon -->
              <template v-if="layer.id === 'dti'">
                <!-- Hit area for click -->
                <rect
                  :x="leftPixelX"
                  :y="layerPositions[idx].y"
                  :width="totalPixelW"
                  :height="layerPositions[idx].h"
                  fill="transparent"
                  style="pointer-events: all"
                />
                <!-- Left boundary DTI -->
                <rect
                  :x="leftPixelX + 2"
                  :y="layerPositions[idx].y"
                  width="8"
                  :height="layerPositions[idx].h"
                  fill="#e8e8e8"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#bbb'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  rx="1"
                />
                <!-- Center DTI -->
                <rect
                  :x="leftPixelX + pixelW - 2"
                  :y="layerPositions[idx].y"
                  :width="gapW + 4"
                  :height="layerPositions[idx].h"
                  fill="#e8e8e8"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#bbb'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  rx="1"
                />
                <!-- Right boundary DTI -->
                <rect
                  :x="leftPixelX + totalPixelW - 10"
                  :y="layerPositions[idx].y"
                  width="8"
                  :height="layerPositions[idx].h"
                  fill="#e8e8e8"
                  :stroke="isHighlighted(layer.id) ? 'var(--vp-c-brand-1)' : '#bbb'"
                  :stroke-width="isHighlighted(layer.id) ? 1.5 : 0.5"
                  :opacity="isHighlighted(layer.id) ? 1 : 0.85"
                  rx="1"
                />
              </template>

              <!-- Layer label on left side -->
              <text
                v-if="layer.id !== 'photodiode' && layer.id !== 'dti'"
                :x="leftPixelX - 8"
                :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                text-anchor="end"
                :class="['layer-label', { highlighted: isHighlighted(layer.id) }]"
              >{{ t(layer.name, layer.nameKo) }}</text>
              <text
                v-if="layer.id === 'photodiode' || layer.id === 'dti'"
                :x="leftPixelX + totalPixelW + 8"
                :y="layerPositions[idx].y + layerPositions[idx].h / 2 + 3"
                text-anchor="start"
                :class="['layer-label', { highlighted: isHighlighted(layer.id) }]"
              >{{ t(layer.name, layer.nameKo) }}</text>
            </g>
          </template>

          <!-- z-axis direction label -->
          <text :x="svgW - 16" :y="svgH / 2" text-anchor="middle" class="axis-label" transform="rotate(90, 484, 200)">z</text>
          <line :x1="svgW - 10" y1="50" :x2="svgW - 10" :y2="svgH - 30" stroke="var(--vp-c-text-3)" stroke-width="0.8" marker-end="url(#paLightArrow)" opacity="0.4" />
        </svg>
      </div>

      <!-- Info card panel -->
      <div class="info-panel">
        <div v-if="selectedLayer" class="info-card">
          <h5>{{ t(selectedLayer.name, selectedLayer.nameKo) }}</h5>
          <p class="info-purpose">{{ t(selectedLayer.purpose, selectedLayer.purposeKo) }}</p>
          <table class="info-table">
            <tr>
              <td class="info-key">{{ t('Typical thickness', '일반적인 두께') }}</td>
              <td class="info-val">{{ selectedLayer.thickness }}</td>
            </tr>
            <tr>
              <td class="info-key">{{ t('Material', '재료') }}</td>
              <td class="info-val">{{ t(selectedLayer.material, selectedLayer.materialKo) }}</td>
            </tr>
          </table>
          <p class="info-desc">{{ t(selectedLayer.description, selectedLayer.descriptionKo) }}</p>
        </div>
        <div v-else class="info-placeholder">
          {{ t(
            'Click on a layer in the diagram to learn about its role in the pixel stack.',
            '다이어그램에서 레이어를 클릭하여 픽셀 스택에서의 역할을 알아보세요.'
          ) }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const svgW = 500
const svgH = 420
const leftPixelX = 120
const pixelW = 120
const gapW = 16
const totalPixelW = pixelW * 2 + gapW
const startY = 36

interface Layer {
  id: string
  name: string
  nameKo: string
  color: string
  heightFrac: number
  purpose: string
  purposeKo: string
  thickness: string
  material: string
  materialKo: string
  description: string
  descriptionKo: string
}

const layers: Layer[] = [
  {
    id: 'air',
    name: 'Air',
    nameKo: '공기',
    color: '#d6eaf8',
    heightFrac: 0.05,
    purpose: 'Light enters from above',
    purposeKo: '빛이 위에서 들어옵니다',
    thickness: 'Semi-infinite',
    material: 'Air (n = 1.0)',
    materialKo: '공기 (n = 1.0)',
    description: 'The medium above the pixel. Light travels in the -z direction from air into the pixel stack. The refractive index of air (n=1) defines the reference for Fresnel reflection at the top surface.',
    descriptionKo: '픽셀 위의 매질입니다. 빛은 공기에서 픽셀 스택으로 -z 방향으로 이동합니다. 공기의 굴절률(n=1)이 상부 표면에서의 프레넬 반사 기준을 정의합니다.',
  },
  {
    id: 'microlens',
    name: 'Microlens',
    nameKo: '마이크로렌즈',
    color: '#d2b4de',
    heightFrac: 0.10,
    purpose: 'Focuses light into the photodiode',
    purposeKo: '빛을 포토다이오드로 집중시킵니다',
    thickness: '0.4 - 0.8 um',
    material: 'Polymer (styrene-based, n ~ 1.5)',
    materialKo: '폴리머 (스티렌 기반, n ~ 1.5)',
    description: 'A curved dome of transparent polymer that acts as a tiny lens. It focuses incoming light toward the center of the photodiode, improving light collection efficiency. In COMPASS this is modeled as a superellipse profile.',
    descriptionKo: '작은 렌즈 역할을 하는 투명 폴리머의 곡면 돔입니다. 입사광을 포토다이오드 중심으로 집중시켜 집광 효율을 향상시킵니다. COMPASS에서는 초타원 프로파일로 모델링됩니다.',
  },
  {
    id: 'planarization',
    name: 'Planarization',
    nameKo: '평탄화층',
    color: '#d5d8dc',
    heightFrac: 0.06,
    purpose: 'Smooths surface between lens and filter',
    purposeKo: '렌즈와 필터 사이의 표면을 평탄화합니다',
    thickness: '0.3 - 1.0 um',
    material: 'SiO2 (n ~ 1.46)',
    materialKo: 'SiO2 (n ~ 1.46)',
    description: 'A uniform dielectric layer (usually SiO2) that fills the gap between the microlens and color filter. It provides a flat surface for color filter deposition and acts as a spacer controlling the microlens focal distance.',
    descriptionKo: '마이크로렌즈와 컬러 필터 사이의 간격을 채우는 균일한 유전체층(보통 SiO2)입니다. 컬러 필터 증착을 위한 평탄한 표면을 제공하며 마이크로렌즈 초점 거리를 제어하는 스페이서 역할을 합니다.',
  },
  {
    id: 'colorfilter',
    name: 'Color Filter',
    nameKo: '컬러 필터',
    color: '#27ae60',
    heightFrac: 0.10,
    purpose: 'Selects wavelength range for this pixel',
    purposeKo: '이 픽셀의 파장 범위를 선택합니다',
    thickness: '0.4 - 0.8 um',
    material: 'Organic dye (R/G/B)',
    materialKo: '유기 염료 (R/G/B)',
    description: 'Organic dye-based filter that selectively absorbs unwanted wavelengths while transmitting the target color. Arranged in a Bayer RGGB pattern across the sensor. The green pixel shown here transmits ~500-580 nm; the red transmits ~590-700 nm.',
    descriptionKo: '불필요한 파장을 선택적으로 흡수하고 목표 색상을 투과하는 유기 염료 기반 필터입니다. 센서 전체에 베이어 RGGB 패턴으로 배열됩니다. 여기 표시된 녹색 픽셀은 ~500-580 nm을, 적색은 ~590-700 nm을 투과합니다.',
  },
  {
    id: 'metalgrid',
    name: 'Metal Grid',
    nameKo: '금속 격자',
    color: '#555555',
    heightFrac: 0.04,
    purpose: 'Prevents optical crosstalk between color filters',
    purposeKo: '컬러 필터 간의 광학적 크로스토크를 방지합니다',
    thickness: '40 - 80 nm wide',
    material: 'Tungsten (W)',
    materialKo: '텅스텐 (W)',
    description: 'Thin vertical walls of tungsten between adjacent color filter elements. These opaque barriers prevent light that enters through one color filter from leaking sideways into a neighboring pixel with a different color, reducing color crosstalk.',
    descriptionKo: '인접한 컬러 필터 요소 사이의 얇은 텅스텐 수직 벽입니다. 이 불투명 장벽은 하나의 컬러 필터를 통해 들어온 빛이 다른 색상의 인접 픽셀로 옆으로 누출되는 것을 방지하여 색상 크로스토크를 줄입니다.',
  },
  {
    id: 'barl',
    name: 'BARL',
    nameKo: 'BARL (반사방지층)',
    color: '#5b8cc4',
    heightFrac: 0.05,
    purpose: 'Anti-reflection coating reduces reflection loss',
    purposeKo: '반사 방지 코팅이 반사 손실을 줄입니다',
    thickness: '0.05 - 0.12 um',
    material: 'SiO2 / HfO2 / Si3N4 stack',
    materialKo: 'SiO2 / HfO2 / Si3N4 스택',
    description: 'Bottom Anti-Reflection Layer: a multi-layer dielectric stack (typically 2-5 sub-layers) designed to minimize reflection at the color-filter-to-silicon interface. Without BARL, the large index mismatch (n~1.55 to n~4.0) causes 20-30% reflection loss.',
    descriptionKo: '하부 반사 방지층: 컬러 필터-실리콘 계면에서의 반사를 최소화하기 위해 설계된 다층 유전체 스택(일반적으로 2-5개 하위층)입니다. BARL이 없으면 큰 굴절률 불일치(n~1.55에서 n~4.0)로 인해 20-30%의 반사 손실이 발생합니다.',
  },
  {
    id: 'silicon',
    name: 'Silicon',
    nameKo: '실리콘',
    color: '#6b2c2c',
    heightFrac: 0.35,
    purpose: 'Absorbs photons, generates electrons',
    purposeKo: '광자를 흡수하고 전자를 생성합니다',
    thickness: '2.0 - 4.0 um',
    material: 'Crystalline Si (n ~ 4.08 at 550 nm)',
    materialKo: '결정질 Si (550 nm에서 n ~ 4.08)',
    description: 'The active absorbing layer where photons generate electron-hole pairs. Silicon has a high refractive index and wavelength-dependent absorption: blue light is absorbed within ~0.2 um, green within ~1 um, and red requires ~3 um. This is the thickest layer in a BSI pixel.',
    descriptionKo: '광자가 전자-정공 쌍을 생성하는 활성 흡수층입니다. 실리콘은 높은 굴절률과 파장에 따른 흡수 특성을 가집니다: 청색광은 ~0.2 um 이내에서, 녹색광은 ~1 um 이내에서, 적색광은 ~3 um이 필요합니다. BSI 픽셀에서 가장 두꺼운 층입니다.',
  },
  {
    id: 'photodiode',
    name: 'Photodiode',
    nameKo: '포토다이오드',
    color: '#b85c5c',
    heightFrac: 0.15,
    purpose: 'Active detection region where charge is collected',
    purposeKo: '전하가 수집되는 능동 감지 영역',
    thickness: '1.0 - 2.5 um',
    material: 'Doped Si (p-n junction)',
    materialKo: '도핑된 Si (p-n 접합)',
    description: 'The p-n junction region within the silicon where photo-generated electrons are collected by the electric field. COMPASS integrates absorbed optical power within this bounding box to calculate Quantum Efficiency (QE). Its position and size determine the signal collection volume.',
    descriptionKo: '광생성 전자가 전기장에 의해 수집되는 실리콘 내 p-n 접합 영역입니다. COMPASS는 이 경계 상자 내에서 흡수된 광학 전력을 적분하여 양자 효율(QE)을 계산합니다. 위치와 크기가 신호 수집 체적을 결정합니다.',
  },
  {
    id: 'dti',
    name: 'DTI',
    nameKo: 'DTI (깊은 트렌치 격리)',
    color: '#e8e8e8',
    heightFrac: 0.10,
    purpose: 'Deep trench isolation walls between pixels',
    purposeKo: '픽셀 사이의 깊은 트렌치 격리 벽',
    thickness: '~0.1 um wide, full Si depth',
    material: 'SiO2-filled trench (n ~ 1.46)',
    materialKo: 'SiO2 충전 트렌치 (n ~ 1.46)',
    description: 'Narrow trenches etched through the full depth of the silicon and filled with SiO2. The large refractive index contrast between Si (n~4) and SiO2 (n~1.46) creates total internal reflection, optically isolating adjacent pixels and significantly reducing optical crosstalk.',
    descriptionKo: '실리콘 전체 깊이를 관통하여 식각되고 SiO2로 채워진 좁은 트렌치입니다. Si(n~4)와 SiO2(n~1.46) 사이의 큰 굴절률 대비가 전반사를 생성하여 인접 픽셀을 광학적으로 격리하고 광학적 크로스토크를 크게 줄입니다.',
  },
]

const hoveredId = ref<string | null>(null)
const selectedId = ref<string | null>(null)

function selectLayer(id: string) {
  selectedId.value = selectedId.value === id ? null : id
}

function isHighlighted(id: string): boolean {
  return selectedId.value === id || hoveredId.value === id
}

const selectedLayer = computed(() => {
  if (!selectedId.value) return null
  return layers.find((l) => l.id === selectedId.value) || null
})

// Compute vertical positions
// Photodiode and DTI overlap with silicon — they share the silicon space
const mainLayers = layers.filter(l => l.id !== 'photodiode' && l.id !== 'dti')
const totalFrac = mainLayers.reduce((s, l) => s + l.heightFrac, 0)
const availableH = svgH - startY - 16

const layerPositions = computed(() => {
  const positions: { y: number; h: number }[] = []
  let currentY = startY

  // Find silicon position for photodiode and DTI overlay
  let siliconY = 0
  let siliconH = 0

  for (const layer of layers) {
    if (layer.id === 'photodiode' || layer.id === 'dti') {
      // These will be placed relative to silicon
      positions.push({ y: 0, h: 0 }) // placeholder
      continue
    }
    const h = (layer.heightFrac / totalFrac) * availableH
    if (layer.id === 'silicon') {
      siliconY = currentY
      siliconH = h
    }
    positions.push({ y: currentY, h })
    currentY += h
  }

  // Now place photodiode and DTI within silicon region
  for (let i = 0; i < layers.length; i++) {
    if (layers[i].id === 'photodiode') {
      // Photodiode occupies middle portion of silicon
      positions[i] = {
        y: siliconY + siliconH * 0.15,
        h: siliconH * 0.55,
      }
    }
    if (layers[i].id === 'dti') {
      // DTI spans full silicon depth
      positions[i] = {
        y: siliconY + 2,
        h: siliconH - 4,
      }
    }
  }

  return positions
})
</script>

<style scoped>
.pixel-anatomy-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.pixel-anatomy-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.anatomy-layout {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.svg-wrapper {
  flex: 0 0 auto;
  overflow-x: auto;
}
.anatomy-svg {
  width: 500px;
  max-width: 100%;
  display: block;
}
.info-panel {
  flex: 1;
  min-width: 240px;
}
.light-label {
  font-size: 10px;
  fill: #f5c842;
  font-weight: 600;
}
.light-ray {
  animation: pa-dash-move 2s linear infinite;
}
@keyframes pa-dash-move {
  0% { stroke-dashoffset: 0; }
  100% { stroke-dashoffset: -20; }
}
.pa-layer-group:hover {
  filter: brightness(1.08);
}
.pa-layer-group.selected {
  filter: brightness(1.04);
}
.layer-label {
  font-size: 9.5px;
  fill: var(--vp-c-text-2);
  transition: fill 0.2s;
}
.layer-label.highlighted {
  fill: var(--vp-c-brand-1);
  font-weight: 700;
}
.cf-label {
  font-size: 12px;
  fill: #fff;
  font-weight: 700;
  pointer-events: none;
}
.pd-label {
  font-size: 9px;
  fill: #f0d0d0;
  font-weight: 600;
  pointer-events: none;
}
.axis-label {
  font-size: 11px;
  fill: var(--vp-c-text-3);
  font-style: italic;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 18px;
}
.info-card h5 {
  margin: 0 0 8px 0;
  font-size: 1.05em;
  color: var(--vp-c-brand-1);
}
.info-purpose {
  font-size: 0.92em;
  color: var(--vp-c-text-1);
  font-weight: 500;
  margin: 0 0 12px 0;
  padding: 8px 12px;
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
  border-left: 3px solid var(--vp-c-brand-1);
}
.info-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 12px;
}
.info-table td {
  padding: 5px 8px;
  font-size: 0.88em;
  border-bottom: 1px solid var(--vp-c-divider);
}
.info-key {
  color: var(--vp-c-text-2);
  white-space: nowrap;
  width: 42%;
}
.info-val {
  font-family: var(--vp-font-family-mono);
  font-size: 0.85em;
  color: var(--vp-c-text-1);
}
.info-desc {
  font-size: 0.88em;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin: 0;
}
.info-placeholder {
  background: var(--vp-c-bg);
  border: 1px dashed var(--vp-c-divider);
  border-radius: 10px;
  padding: 30px 20px;
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 0.9em;
}
@media (max-width: 700px) {
  .anatomy-layout {
    flex-direction: column;
  }
  .anatomy-svg {
    width: 100%;
  }
}
</style>
