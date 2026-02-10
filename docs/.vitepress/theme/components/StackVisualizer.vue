<template>
  <div class="stack-container">
    <h4>Interactive BSI Pixel Stack Cross-Section</h4>
    <p class="component-description">
      Click on any layer to view its material properties and role in the pixel stack.
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
          <text x="200" y="15" class="light-label">Incident light</text>

          <!-- Layers -->
          <template v-for="(layer, idx) in layers" :key="layer.id">
            <g
              :class="['layer-group', { selected: selectedId === layer.id }]"
              @click="selectLayer(layer.id)"
              style="cursor: pointer"
            >
              <!-- Layer rectangle -->
              <rect
                :x="layerX"
                :y="layerPositions[idx].y"
                :width="layerW"
                :height="layerPositions[idx].h"
                :fill="layer.color"
                :stroke="selectedId === layer.id ? 'var(--vp-c-brand-1)' : '#555'"
                :stroke-width="selectedId === layer.id ? 2.5 : 1"
                rx="2"
                :opacity="layer.opacity || 1"
              />

              <!-- Microlens curved top -->
              <ellipse
                v-if="layer.id === 'microlens'"
                :cx="180"
                :cy="layerPositions[idx].y"
                :rx="layerW / 2 - 5"
                :ry="layerPositions[idx].h * 0.6"
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
              <td class="prop-key">Material</td>
              <td class="prop-val">{{ selectedLayer.material }}</td>
            </tr>
            <tr>
              <td class="prop-key">Typical thickness</td>
              <td class="prop-val">{{ selectedLayer.thickness }}</td>
            </tr>
            <tr>
              <td class="prop-key">Refractive index (n)</td>
              <td class="prop-val">{{ selectedLayer.n }}</td>
            </tr>
            <tr v-if="selectedLayer.k">
              <td class="prop-key">Extinction coeff. (k)</td>
              <td class="prop-val">{{ selectedLayer.k }}</td>
            </tr>
            <tr>
              <td class="prop-key">Function</td>
              <td class="prop-val">{{ selectedLayer.role }}</td>
            </tr>
          </table>
          <p class="detail-desc">{{ selectedLayer.description }}</p>
        </div>
        <div v-else class="detail-placeholder">
          Click a layer in the stack diagram to see its properties.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

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
    description: 'The medium above the pixel. For on-chip applications this is air (n=1). In some configurations an encapsulant (n~1.5) may be present. Defines the reference for incidence angle and Fresnel reflection at the top surface.',
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
    description: 'Curved polymer structure that focuses incoming light toward the center of the pixel photodiode. Modeled in COMPASS as a superellipse profile. The lens center can be shifted to compensate for Chief Ray Angle (CRA) at the sensor edge.',
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
    description: 'A uniform dielectric layer (usually SiO2) that fills the space between the microlens and color filter array. Provides a flat surface for the color filter deposition and acts as a spacer that controls the microlens focal distance.',
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
    description: 'Organic dye-based filter arranged in a Bayer RGGB pattern. Each sub-pixel absorbs unwanted wavelengths while transmitting its target color. The metal grid (typically tungsten, 40-80 nm wide) between color filter elements provides optical isolation to reduce crosstalk.',
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
    description: 'Bottom Anti-Reflection Layer: a multi-layer dielectric stack (typically 2-5 sub-layers of SiO2, HfO2, Si3N4) designed to minimize reflection at the color-filter-to-silicon interface. Without BARL, the large index mismatch (n~1.55 to n~4.0) causes ~20-30% reflection loss.',
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
    description: 'The active absorbing layer where photons generate electron-hole pairs. Deep Trench Isolation (DTI) trenches filled with SiO2 (n~1.46) optically isolate adjacent pixels via total internal reflection. The photodiode occupies a defined region within the silicon volume. COMPASS integrates absorbed power within the photodiode bounding box to compute QE.',
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
    description: 'In a BSI sensor, the metal wiring and transistors are on this side (opposite to light entry). Acts as a reflecting/absorbing boundary in the simulation. Any photons reaching this layer are either absorbed by metal or reflected back into the silicon. COMPASS typically models this as a perfectly matched layer (PML) or fixed boundary condition.',
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
