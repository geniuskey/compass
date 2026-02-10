<template>
  <div class="pixel-stack-container">
    <h4>{{ t('Interactive Pixel Stack Builder', '인터랙티브 픽셀 스택 빌더') }}</h4>
    <p class="component-description">
      {{ t('Adjust the thickness of each layer in a BSI pixel cross-section. The visualization shows the vertical stack with refractive indices and a scale bar.', 'BSI 픽셀 단면의 각 층 두께를 조정하세요. 시각화는 굴절률과 스케일 바가 포함된 수직 스택을 보여줍니다.') }}
    </p>

    <div class="layout-row">
      <div class="controls-panel">
        <div class="slider-group">
          <label>{{ t('Microlens height', '마이크로렌즈 높이') }}: <strong>{{ microlens.toFixed(2) }} um</strong></label>
          <input type="range" min="0.2" max="1.0" step="0.01" v-model.number="microlens" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Planarization', '평탄화층') }}: <strong>{{ planarization.toFixed(2) }} um</strong></label>
          <input type="range" min="0.1" max="0.5" step="0.01" v-model.number="planarization" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Color filter', '컬러 필터') }}: <strong>{{ colorFilter.toFixed(2) }} um</strong></label>
          <input type="range" min="0.3" max="1.0" step="0.01" v-model.number="colorFilter" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('BARL total', 'BARL 전체') }}: <strong>{{ barl.toFixed(3) }} um</strong></label>
          <input type="range" min="0.02" max="0.2" step="0.005" v-model.number="barl" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Silicon', '실리콘') }}: <strong>{{ silicon.toFixed(1) }} um</strong></label>
          <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="silicon" class="ctrl-range" />
        </div>

        <div class="toggle-group">
          <label class="toggle-label">
            <input type="checkbox" v-model="showDTI" />
            {{ t('Show DTI trenches', 'DTI 트렌치 표시') }}
          </label>
        </div>
        <div class="toggle-group">
          <label class="toggle-label">
            <input type="checkbox" v-model="showGrid" />
            {{ t('Show metal grid', '메탈 그리드 표시') }}
          </label>
        </div>

        <div class="info-card total-card">
          <span class="info-label">{{ t('Total stack height', '전체 스택 높이') }}:</span>
          <span class="info-value">{{ totalHeight.toFixed(2) }} um</span>
        </div>
      </div>

      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="stack-svg">
          <!-- Scale bar on left -->
          <line :x1="scaleBarX" :y1="stackTop" :x2="scaleBarX" :y2="stackTop + stackPxH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <template v-for="tick in scaleTicks" :key="tick.um">
            <line :x1="scaleBarX - 4" :y1="tick.y" :x2="scaleBarX + 4" :y2="tick.y" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <text :x="scaleBarX - 8" :y="tick.y + 3" text-anchor="end" class="scale-label">{{ tick.label }}</text>
          </template>

          <!-- Layer rectangles -->
          <template v-for="(layer, idx) in layers" :key="idx">
            <rect
              :x="layerX"
              :y="layer.y"
              :width="layerW"
              :height="Math.max(layer.h, 1)"
              :fill="layer.color"
              stroke="var(--vp-c-divider)"
              stroke-width="0.5"
            />
            <!-- Layer label (centered) -->
            <text
              v-if="layer.h > 8"
              :x="layerX + layerW / 2"
              :y="layer.y + layer.h / 2 + 4"
              text-anchor="middle"
              class="layer-label"
            >{{ layer.name }}</text>
            <!-- Thickness on left inside -->
            <text
              v-if="layer.h > 12"
              :x="layerX + 4"
              :y="layer.y + layer.h / 2 + 4"
              class="layer-thickness"
            >{{ layer.thickness }} um</text>
            <!-- Refractive index on right -->
            <text
              :x="layerX + layerW + 6"
              :y="layer.y + layer.h / 2 + 4"
              class="layer-n"
            >n={{ layer.n }}</text>
          </template>

          <!-- DTI trenches (in silicon layer) -->
          <template v-if="showDTI">
            <rect
              :x="layerX"
              :y="siliconLayer.y"
              :width="8"
              :height="siliconLayer.h"
              fill="#b0c4de"
              opacity="0.7"
              stroke="#6b8cae"
              stroke-width="0.5"
            />
            <rect
              :x="layerX + layerW - 8"
              :y="siliconLayer.y"
              :width="8"
              :height="siliconLayer.h"
              fill="#b0c4de"
              opacity="0.7"
              stroke="#6b8cae"
              stroke-width="0.5"
            />
            <text
              :x="layerX + layerW - 4"
              :y="siliconLayer.y + 12"
              text-anchor="end"
              class="dti-label"
            >DTI</text>
          </template>

          <!-- Metal grid (in color filter layer) -->
          <template v-if="showGrid">
            <rect
              :x="layerX"
              :y="cfLayer.y"
              :width="5"
              :height="cfLayer.h"
              fill="#707070"
              opacity="0.8"
            />
            <rect
              :x="layerX + layerW - 5"
              :y="cfLayer.y"
              :width="5"
              :height="cfLayer.h"
              fill="#707070"
              opacity="0.8"
            />
          </template>

          <!-- Microlens curved top -->
          <path
            :d="microlensArc"
            fill="none"
            stroke="#9b59b6"
            stroke-width="1.5"
          />
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

const microlens = ref(0.6)
const planarization = ref(0.3)
const colorFilter = ref(0.6)
const barl = ref(0.08)
const silicon = ref(3.0)
const showDTI = ref(true)
const showGrid = ref(true)

const totalHeight = computed(() => microlens.value + planarization.value + colorFilter.value + barl.value + silicon.value)

const svgW = 320
const svgH = 400
const stackTop = 30
const stackMaxH = 340
const scaleBarX = 42
const layerX = 80
const layerW = 140

const stackPxH = computed(() => stackMaxH)

// Scale: map totalHeight to stackPxH pixels
function umToPx(um) {
  return (um / totalHeight.value) * stackPxH.value
}

const layerDefs = computed(() => [
  { name: t('Microlens', '마이크로렌즈'), thickness: microlens.value, color: '#dda0dd', n: '1.56' },
  { name: t('Planarization', '평탄화층'), thickness: planarization.value, color: '#add8e6', n: '1.46' },
  { name: t('Color Filter', '컬러 필터'), thickness: colorFilter.value, color: '#90ee90', n: '1.55' },
  { name: 'BARL', thickness: barl.value, color: '#fffacd', n: '1.8' },
  { name: t('Silicon', '실리콘'), thickness: silicon.value, color: '#c0c0c0', n: '3.5' },
])

const layers = computed(() => {
  let yPos = stackTop
  return layerDefs.value.map(def => {
    const h = umToPx(def.thickness)
    const layer = {
      name: def.name,
      thickness: def.thickness < 0.1 ? def.thickness.toFixed(3) : def.thickness.toFixed(2),
      color: def.color,
      n: def.n,
      y: yPos,
      h,
    }
    yPos += h
    return layer
  })
})

const siliconLayer = computed(() => layers.value[4])
const cfLayer = computed(() => layers.value[2])

// Scale bar ticks
const scaleTicks = computed(() => {
  const ticks = []
  const step = totalHeight.value <= 3 ? 0.5 : 1.0
  for (let um = 0; um <= totalHeight.value + 0.01; um += step) {
    ticks.push({
      um,
      y: stackTop + umToPx(um),
      label: um.toFixed(1),
    })
  }
  return ticks
})

// Microlens arc
const microlensArc = computed(() => {
  const ly = layers.value[0]
  const cx = layerX + layerW / 2
  const topY = ly.y
  const botY = ly.y + ly.h
  const halfW = layerW / 2
  return `M ${layerX} ${botY} Q ${cx} ${topY - ly.h * 0.3} ${layerX + layerW} ${botY}`
})
</script>

<style scoped>
.pixel-stack-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.pixel-stack-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.layout-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.controls-panel {
  flex: 1;
  min-width: 200px;
}
.slider-group {
  margin-bottom: 10px;
}
.slider-group label {
  display: block;
  margin-bottom: 3px;
  font-size: 0.85em;
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.toggle-group {
  margin-bottom: 8px;
}
.toggle-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85em;
  cursor: pointer;
}
.toggle-label input {
  cursor: pointer;
}
.total-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
  margin-top: 12px;
}
.info-label {
  color: var(--vp-c-text-2);
  margin-right: 4px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.svg-wrapper {
  flex: 0 0 auto;
}
.stack-svg {
  width: 320px;
  max-width: 100%;
}
.scale-label {
  font-size: 8px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.layer-label {
  font-size: 10px;
  fill: #333;
  font-weight: 600;
  pointer-events: none;
}
.layer-thickness {
  font-size: 7px;
  fill: #555;
  font-family: var(--vp-font-family-mono);
  pointer-events: none;
}
.layer-n {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.dti-label {
  font-size: 7px;
  fill: #4a6fa5;
  font-weight: 600;
}
</style>
