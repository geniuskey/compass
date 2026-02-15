<template>
  <div class="prnu-container">
    <h4>{{ t('PRNU / DSNU Visualizer', 'PRNU / DSNU 시각화') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize Photo Response Non-Uniformity (PRNU) and Dark Signal Non-Uniformity (DSNU) as fixed pattern noise on a 2D pixel array.',
        '2D 픽셀 배열에서 광응답 비균일성(PRNU)과 암신호 비균일성(DSNU)을 고정 패턴 노이즈로 시각화합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>PRNU: <strong>{{ prnu.toFixed(1) }}%</strong></label>
        <input type="range" min="0" max="5" step="0.1" v-model.number="prnu" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>DSNU: <strong>{{ dsnu }} DN</strong></label>
        <input type="range" min="0" max="50" step="1" v-model.number="dsnu" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Signal Level', '신호 레벨') }}: <strong>{{ signalLevel }} DN</strong></label>
        <input type="range" min="100" max="4000" step="50" v-model.number="signalLevel" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('View', '보기') }}</label>
        <div class="view-toggle">
          <button :class="['toggle-btn', { active: viewMode === 'fpn' }]" @click="viewMode = 'fpn'">FPN</button>
          <button :class="['toggle-btn', { active: viewMode === 'prnu' }]" @click="viewMode = 'prnu'">PRNU</button>
          <button :class="['toggle-btn', { active: viewMode === 'dsnu' }]" @click="viewMode = 'dsnu'">DSNU</button>
        </div>
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('FPN RMS', 'FPN RMS') }}</div>
        <div class="result-value highlight">{{ fpnRms.toFixed(2) }} DN</div>
      </div>
      <div class="result-card">
        <div class="result-label">PRNU RMS</div>
        <div class="result-value">{{ prnuRms.toFixed(2) }} DN</div>
      </div>
      <div class="result-card">
        <div class="result-label">DSNU RMS</div>
        <div class="result-value">{{ dsnu.toFixed(1) }} DN</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Peak-to-Peak', '피크투피크') }}</div>
        <div class="result-value">{{ p2p.toFixed(1) }} DN</div>
      </div>
    </div>

    <div class="prnu-visuals">
      <div class="heatmap-section">
        <h5>{{ t('Pixel Array Heatmap', '픽셀 어레이 히트맵') }} ({{ GRID }}x{{ GRID }})</h5>
        <div class="pixel-grid">
          <div v-for="(row, ri) in pixelGrid" :key="ri" class="pixel-row">
            <div v-for="(val, ci) in row" :key="ci" class="pixel-cell"
              :style="{ background: cellColor(val) }">
            </div>
          </div>
        </div>
      </div>

      <div class="hist-section">
        <h5>{{ t('Distribution', '분포') }}</h5>
        <svg :viewBox="`0 0 ${HW} ${HH}`" class="hist-svg">
          <line :x1="hPad.left" :y1="hPad.top" :x2="hPad.left" :y2="hPad.top + hPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="hPad.left" :y1="hPad.top + hPlotH" :x2="hPad.left + hPlotW" :y2="hPad.top + hPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <rect v-for="(bin, i) in histBins" :key="'hb'+i"
            :x="hPad.left + i * binW" :y="hPad.top + hPlotH - (bin / histMax) * hPlotH"
            :width="Math.max(1, binW - 1)" :height="(bin / histMax) * hPlotH"
            fill="var(--vp-c-brand-1)" opacity="0.7" />
          <text :x="hPad.left" :y="hPad.top + hPlotH + 12" class="tick-label">{{ histRange[0].toFixed(0) }}</text>
          <text :x="hPad.left + hPlotW" :y="hPad.top + hPlotH + 12" text-anchor="end" class="tick-label">{{ histRange[1].toFixed(0) }}</text>
          <text :x="hPad.left + hPlotW / 2" :y="hPad.top + hPlotH + 24" text-anchor="middle" class="axis-title">DN</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const prnu = ref(1.0)
const dsnu = ref(5)
const signalLevel = ref(1000)
const viewMode = ref<'fpn' | 'prnu' | 'dsnu'>('fpn')

function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = a + 0x6D2B79F5 | 0
    let t = Math.imul(a ^ a >>> 15, 1 | a)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function gaussRng(rng: () => number): number {
  return Math.sqrt(-2 * Math.log(rng() + 1e-10)) * Math.cos(2 * Math.PI * rng())
}

const GRID = 32
const prnuMap = computed(() => {
  const rng = mulberry32(123)
  const map: number[][] = []
  for (let r = 0; r < GRID; r++) {
    const row: number[] = []
    for (let c = 0; c < GRID; c++) row.push(gaussRng(rng))
    map.push(row)
  }
  return map
})

const dsnuMap = computed(() => {
  const rng = mulberry32(456)
  const map: number[][] = []
  for (let r = 0; r < GRID; r++) {
    const row: number[] = []
    for (let c = 0; c < GRID; c++) row.push(gaussRng(rng))
    map.push(row)
  }
  return map
})

const prnuRms = computed(() => signalLevel.value * prnu.value / 100)
const fpnRms = computed(() => Math.sqrt(prnuRms.value ** 2 + dsnu.value ** 2))

const pixelGrid = computed(() => {
  const s = signalLevel.value
  const pSigma = prnu.value / 100
  const dSigma = dsnu.value
  return prnuMap.value.map((row, ri) =>
    row.map((pVal, ci) => {
      if (viewMode.value === 'prnu') return s * pSigma * pVal
      if (viewMode.value === 'dsnu') return dSigma * dsnuMap.value[ri][ci]
      return s + s * pSigma * pVal + dSigma * dsnuMap.value[ri][ci]
    })
  )
})

const flatValues = computed(() => pixelGrid.value.flat())
const gridMin = computed(() => Math.min(...flatValues.value))
const gridMax = computed(() => Math.max(...flatValues.value))
const p2p = computed(() => gridMax.value - gridMin.value)

function cellColor(val: number): string {
  const lo = gridMin.value, hi = gridMax.value
  const norm = hi > lo ? (val - lo) / (hi - lo) : 0.5
  const v = Math.round(norm * 255)
  return `rgb(${v},${v},${v})`
}

const BINS = 30
const histRange = computed<[number, number]>(() => [gridMin.value, gridMax.value])
const histBins = computed(() => {
  const bins = new Array(BINS).fill(0)
  const lo = histRange.value[0], range = histRange.value[1] - lo
  if (range === 0) return bins
  for (const v of flatValues.value) {
    const idx = Math.min(BINS - 1, Math.floor(((v - lo) / range) * BINS))
    bins[idx]++
  }
  return bins
})
const histMax = computed(() => Math.max(1, ...histBins.value))

const HW = 260, HH = 160
const hPad = { top: 8, right: 8, bottom: 28, left: 24 }
const hPlotW = HW - hPad.left - hPad.right
const hPlotH = HH - hPad.top - hPad.bottom
const binW = computed(() => hPlotW / BINS)
</script>

<style scoped>
.prnu-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.prnu-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.prnu-container h5 { margin: 8px 0 6px 0; font-size: 0.9em; color: var(--vp-c-text-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.view-toggle { display: flex; gap: 0; border: 1px solid var(--vp-c-divider); border-radius: 8px; overflow: hidden; }
.toggle-btn { padding: 5px 12px; font-size: 0.82em; font-weight: 500; border: none; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; }
.toggle-btn.active { background: var(--vp-c-brand-1); color: #fff; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 10px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.prnu-visuals { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
.heatmap-section { flex: 0 0 auto; }
.hist-section { flex: 1; min-width: 240px; }
.pixel-grid { display: grid; grid-template-columns: repeat(32, 1fr); gap: 0; width: 224px; height: 224px; border: 1px solid var(--vp-c-divider); border-radius: 4px; overflow: hidden; }
.pixel-row { display: contents; }
.pixel-cell { aspect-ratio: 1; }
.hist-svg { width: 100%; max-width: 260px; display: block; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
