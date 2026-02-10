<template>
  <div class="bayer-container">
    <h4>Interactive Bayer Pattern Viewer</h4>
    <p class="component-description">
      Explore different color filter array (CFA) patterns. Click a pixel to see its details.
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label for="bayer-pattern-select">Pattern:</label>
        <select id="bayer-pattern-select" v-model="selectedPattern" class="pattern-select">
          <option v-for="p in patternOptions" :key="p.value" :value="p.value">{{ p.label }}</option>
        </select>
      </div>
      <div v-if="selectedPixel" class="pixel-info-badge">
        <span class="pixel-color-dot" :style="{ backgroundColor: selectedPixel.displayColor }"></span>
        <span><strong>{{ selectedPixel.color }}</strong> pixel at ({{ selectedPixel.row }}, {{ selectedPixel.col }})</span>
      </div>
    </div>

    <div class="viewer-layout">
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${svgSize} ${svgSize}`" class="bayer-svg">
          <!-- Grid cells -->
          <template v-for="(cell, idx) in gridCells" :key="idx">
            <rect
              :x="cell.x"
              :y="cell.y"
              :width="cellSize"
              :height="cellSize"
              :fill="cell.fill"
              :stroke="selectedPixel && selectedPixel.row === cell.row && selectedPixel.col === cell.col ? 'var(--vp-c-brand-1)' : '#555'"
              :stroke-width="selectedPixel && selectedPixel.row === cell.row && selectedPixel.col === cell.col ? 2.5 : 0.8"
              style="cursor: pointer"
              @click="selectPixel(cell)"
            />
            <!-- Pixel index label -->
            <text
              :x="cell.x + cellSize / 2"
              :y="cell.y + cellSize / 2 - 4"
              text-anchor="middle"
              class="cell-label"
              :fill="cell.textColor"
            >{{ cell.colorLetter }}</text>
            <text
              :x="cell.x + cellSize / 2"
              :y="cell.y + cellSize / 2 + 10"
              text-anchor="middle"
              class="cell-index"
              :fill="cell.textColor"
            >({{ cell.row }},{{ cell.col }})</text>
          </template>

          <!-- Unit cell outline -->
          <rect
            :x="padding"
            :y="padding"
            :width="unitCellPixelSize"
            :height="unitCellPixelSize"
            fill="none"
            stroke="var(--vp-c-text-1)"
            stroke-width="2"
            stroke-dasharray="6,3"
          />
          <text
            :x="padding + unitCellPixelSize + 6"
            :y="padding + 14"
            class="unit-cell-label"
          >Unit cell</text>
        </svg>
      </div>

      <div class="pattern-info">
        <div class="info-card">
          <div class="info-label">Pattern</div>
          <div class="info-value">{{ currentPatternInfo.name }}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Unit cell</div>
          <div class="info-value">{{ currentPatternInfo.unitCell }}</div>
        </div>
        <div class="info-card">
          <div class="info-label">Green ratio</div>
          <div class="info-value">{{ currentPatternInfo.greenRatio }}</div>
        </div>
        <div class="info-card description-card">
          <div class="info-label">Description</div>
          <div class="info-desc">{{ currentPatternInfo.description }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const padding = 20
const svgSize = 320

const patternOptions = [
  { value: 'RGGB', label: 'RGGB (standard Bayer)' },
  { value: 'GRBG', label: 'GRBG' },
  { value: 'BGGR', label: 'BGGR' },
  { value: 'GBRG', label: 'GBRG' },
  { value: 'QuadBayer', label: 'Quad-Bayer (2x2 OCL)' },
]

const selectedPattern = ref('RGGB')
const selectedPixelState = ref(null)

const patterns = {
  RGGB: {
    grid: [['R','G'],['G','B']],
    name: 'RGGB',
    unitCell: '2x2',
    greenRatio: '50%',
    description: 'Most common Bayer pattern. Two green pixels per unit cell provide higher luminance resolution, mimicking human vision sensitivity.',
  },
  GRBG: {
    grid: [['G','R'],['B','G']],
    name: 'GRBG',
    unitCell: '2x2',
    greenRatio: '50%',
    description: 'Rotated Bayer pattern with green in top-left. Functionally equivalent to RGGB after demosaicing, but shifts the phase of the color filter by one pixel.',
  },
  BGGR: {
    grid: [['B','G'],['G','R']],
    name: 'BGGR',
    unitCell: '2x2',
    greenRatio: '50%',
    description: 'Blue-first Bayer pattern. Used by some sensor manufacturers. The demosaicing algorithm must account for the different starting position.',
  },
  GBRG: {
    grid: [['G','B'],['R','G']],
    name: 'GBRG',
    unitCell: '2x2',
    greenRatio: '50%',
    description: 'Green-Blue first row variant. All four standard Bayer variants are related by row/column shifts and produce equivalent image quality.',
  },
  QuadBayer: {
    grid: [
      ['R','R','G','G'],
      ['R','R','G','G'],
      ['G','G','B','B'],
      ['G','G','B','B'],
    ],
    name: 'Quad-Bayer',
    unitCell: '4x4',
    greenRatio: '50%',
    description: 'Each 2x2 sub-block shares the same color filter. Enables 2x2 binning for high-sensitivity mode and full-resolution readout for high-detail mode. Used in modern smartphone sensors.',
  },
}

const colorMap = {
  R: { fill: '#e74c3c', displayColor: '#e74c3c', textColor: '#fff', name: 'Red' },
  G: { fill: '#27ae60', displayColor: '#27ae60', textColor: '#fff', name: 'Green' },
  B: { fill: '#3498db', displayColor: '#3498db', textColor: '#fff', name: 'Blue' },
}

const currentPatternInfo = computed(() => patterns[selectedPattern.value])

const gridSize = computed(() => {
  const g = patterns[selectedPattern.value].grid
  return g.length
})

const cellSize = computed(() => {
  return (svgSize - padding * 2) / (gridSize.value * 2)
})

const unitCellPixelSize = computed(() => {
  return cellSize.value * gridSize.value
})

const gridCells = computed(() => {
  const g = patterns[selectedPattern.value].grid
  const size = g.length
  const cells = []
  // Show 2x2 tiling of the unit cell
  for (let tr = 0; tr < 2; tr++) {
    for (let tc = 0; tc < 2; tc++) {
      for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
          const color = g[r][c]
          const absRow = tr * size + r
          const absCol = tc * size + c
          cells.push({
            x: padding + absCol * cellSize.value,
            y: padding + absRow * cellSize.value,
            row: absRow,
            col: absCol,
            color: colorMap[color].name,
            colorLetter: color,
            fill: colorMap[color].fill,
            textColor: colorMap[color].textColor,
            displayColor: colorMap[color].displayColor,
          })
        }
      }
    }
  }
  return cells
})

const selectedPixel = computed(() => selectedPixelState.value)

function selectPixel(cell) {
  if (selectedPixelState.value && selectedPixelState.value.row === cell.row && selectedPixelState.value.col === cell.col) {
    selectedPixelState.value = null
  } else {
    selectedPixelState.value = cell
  }
}
</script>

<style scoped>
.bayer-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.bayer-container h4 {
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
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.9em;
  font-weight: 600;
}
.pattern-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}
.pixel-info-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 0.85em;
}
.pixel-color-dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 1px solid #555;
}
.viewer-layout {
  display: flex;
  gap: 24px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.svg-wrapper {
  flex: 0 0 auto;
}
.bayer-svg {
  width: 320px;
  max-width: 100%;
}
.cell-label {
  font-size: 14px;
  font-weight: 700;
  pointer-events: none;
}
.cell-index {
  font-size: 8px;
  pointer-events: none;
  opacity: 0.85;
}
.unit-cell-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.pattern-info {
  flex: 1;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
}
.info-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 2px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  font-size: 0.95em;
}
.info-desc {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}
.description-card {
  margin-top: 4px;
}
</style>
