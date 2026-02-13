<template>
  <div class="barl-optimizer-container">
    <h4>{{ t('Thin Film Stack Designer', '박막 스택 설계기') }}</h4>
    <p class="component-description">
      {{ t(
        'Design multi-layer thin film coatings with arbitrary materials, order, and thickness. Add/remove/reorder layers freely. Supports 20+ materials.',
        '임의의 재질, 순서, 두께로 다층 박막 코팅을 설계합니다. 레이어를 자유롭게 추가/제거/재배치할 수 있으며 20개 이상의 재질을 지원합니다.'
      ) }}
    </p>

    <!-- Incident / Substrate media -->
    <div class="media-row">
      <div class="media-group">
        <label>{{ t('Incident medium:', '입사 매질:') }}</label>
        <select v-model="incidentMat" class="bo-select">
          <option v-for="k in substrateMats" :key="'inc'+k" :value="k">{{ matName(k) }}</option>
        </select>
      </div>
      <div class="media-group">
        <label>{{ t('Substrate:', '기판:') }}</label>
        <select v-model="substrateMat" class="bo-select">
          <option v-for="k in substrateMats" :key="'sub'+k" :value="k">{{ matName(k) }}</option>
        </select>
      </div>
    </div>

    <!-- Preset + target + optimize -->
    <div class="controls-row">
      <div class="select-group">
        <label>{{ t('Preset:', '프리셋:') }}</label>
        <select v-model="selectedPreset" class="bo-select" @change="applyPreset">
          <option value="custom">{{ t('Custom', '사용자 정의') }}</option>
          <option value="barl4">{{ t('BARL 4-layer (default)', 'BARL 4층 (기본)') }}</option>
          <option value="ar2">{{ t('2-layer AR (MgF₂/TiO₂)', '2층 AR (MgF₂/TiO₂)') }}</option>
          <option value="broadband">{{ t('Broadband AR (4-layer)', '광대역 AR (4층)') }}</option>
          <option value="hr_blue">{{ t('HR Blue mirror', 'HR 블루 미러') }}</option>
          <option value="nir_cut">{{ t('NIR-cut filter', 'NIR 차단 필터') }}</option>
        </select>
      </div>
      <div class="select-group">
        <label>{{ t('Target:', '목표 대역:') }}</label>
        <select v-model="targetBand" class="bo-select">
          <option value="blue">Blue (430-470nm)</option>
          <option value="green">Green (510-560nm)</option>
          <option value="red">Red (590-640nm)</option>
          <option value="visible">Visible (430-640nm)</option>
          <option value="full">Full (380-780nm)</option>
        </select>
      </div>
      <button class="optimize-btn" :disabled="optimizing || layers.length === 0" @click="runOptimize">
        <template v-if="optimizing">{{ t('Optimizing...', '최적화 중...') }}</template>
        <template v-else>{{ t('Auto Optimize', '자동 최적화') }}</template>
      </button>
    </div>

    <!-- Layer list -->
    <div class="layers-section">
      <div class="layers-header">
        <span class="layers-title">{{ t('Layers', '레이어') }} ({{ layers.length }})</span>
        <span class="layers-subtitle">{{ t('top → bottom (light direction)', '상단 → 하단 (빛 진행 방향)') }}</span>
        <button class="add-btn" @click="addLayer">+ {{ t('Add layer', '레이어 추가') }}</button>
      </div>
      <div v-if="layers.length === 0" class="empty-layers">
        {{ t('No layers. Click "Add layer" to start.', '레이어가 없습니다. "레이어 추가"를 클릭하세요.') }}
      </div>
      <div class="layer-list">
        <div v-for="(layer, idx) in layers" :key="layer.id" class="layer-row">
          <span class="layer-num">{{ idx + 1 }}</span>
          <span class="layer-swatch" :style="{ background: matColor(layer.material) }"></span>
          <select v-model="layer.material" class="bo-select layer-mat-select" @change="markCustom">
            <option v-for="k in coatingMats" :key="k" :value="k">{{ matName(k) }}</option>
          </select>
          <div class="thickness-ctrl">
            <input
              type="range" min="1" max="200" step="1"
              :value="layer.thicknessNm"
              @input="setThickness(idx, $event)"
              class="bo-range"
            />
            <input
              type="number" min="0" max="500" step="1"
              v-model.number="layer.thicknessNm"
              class="thickness-input"
              @input="markCustom"
            />
            <span class="unit">nm</span>
          </div>
          <div class="layer-actions">
            <button class="icon-btn" @click="moveLayer(idx, -1)" :disabled="idx === 0" :title="t('Move up','위로')">&#x25B2;</button>
            <button class="icon-btn" @click="moveLayer(idx, 1)" :disabled="idx === layers.length - 1" :title="t('Move down','아래로')">&#x25BC;</button>
            <button class="icon-btn remove-btn" @click="removeLayer(idx)" :title="t('Remove','제거')">&#x2715;</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Info cards -->
    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Avg R in band:', '대역 평균 R:') }}</span>
        <span class="info-value">{{ avgRInBand.toFixed(2) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Avg T in band:', '대역 평균 T:') }}</span>
        <span class="info-value">{{ avgTInBand.toFixed(2) }}%</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Min R:', '최소 R:') }}</span>
        <span class="info-value">{{ minR.value.toFixed(2) }}% @ {{ minR.wl }} nm</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Total thickness:', '총 두께:') }}</span>
        <span class="info-value">{{ totalThickness }} nm</span>
      </div>
    </div>

    <!-- Stack bar visualization -->
    <div class="stack-viz-section" v-if="layers.length > 0">
      <span class="stack-viz-label">{{ t('Stack cross-section:', '스택 단면:') }}</span>
      <div class="stack-bar-wrap">
        <span class="stack-medium-label">{{ matName(incidentMat) }}</span>
        <div class="stack-bar">
          <div
            v-for="(layer, idx) in layers" :key="'bar' + layer.id"
            class="stack-segment"
            :style="{
              width: totalThickness > 0 ? (layer.thicknessNm / totalThickness * 100) + '%' : (100 / layers.length) + '%',
              background: matColor(layer.material),
            }"
          >
            <span v-if="layer.thicknessNm / totalThickness > 0.08" class="stack-segment-label">{{ layer.thicknessNm }}</span>
          </div>
        </div>
        <span class="stack-medium-label">{{ matName(substrateMat) }}</span>
      </div>
      <div class="stack-legend">
        <span v-for="mat in uniqueMaterials" :key="'leg'+mat" class="stack-legend-item">
          <span class="stack-legend-swatch" :style="{ background: matColor(mat) }"></span>
          {{ matName(mat) }}
        </span>
      </div>
    </div>

    <!-- SVG Chart: R and T -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="bo-svg"
        @mousemove="onMouseMove"
        @mouseleave="hoverIdx = -1"
      >
        <defs>
          <linearGradient id="boVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>
        <rect :x="pad.left" :y="pad.top + plotH + 2" :width="plotW" height="8" fill="url(#boVisSpectrum)" rx="2" />

        <!-- Target band shading -->
        <rect
          :x="xScale(bandRange[0])" :y="pad.top"
          :width="xScale(bandRange[1]) - xScale(bandRange[0])" :height="plotH"
          fill="var(--vp-c-brand-1)" opacity="0.07"
        />

        <!-- Grid -->
        <line v-for="tick in yTicks" :key="'yg'+tick"
          :x1="pad.left" :y1="yScale(tick)" :x2="pad.left+plotW" :y2="yScale(tick)"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        <line v-for="tick in xTicks" :key="'xg'+tick"
          :x1="xScale(tick)" :y1="pad.top" :x2="xScale(tick)" :y2="pad.top+plotH"
          stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

        <!-- Axes -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top+plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top+plotH" :x2="pad.left+plotW" :y2="pad.top+plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <!-- Y labels -->
        <text v-for="tick in yTicks" :key="'yl'+tick" :x="pad.left-6" :y="yScale(tick)+3" text-anchor="end" class="axis-label">{{ tick }}%</text>
        <!-- X labels -->
        <text v-for="tick in xTicks" :key="'xl'+tick" :x="xScale(tick)" :y="pad.top+plotH+24" text-anchor="middle" class="axis-label">{{ tick }}</text>

        <!-- Axis titles -->
        <text :x="pad.left+plotW/2" :y="svgH-2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)','파장 (nm)') }}</text>
        <text :x="14" :y="pad.top+plotH/2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 14, ${pad.top+plotH/2})`">%</text>

        <!-- Reflectance fill + curve -->
        <path :d="reflFillPath" fill="#3498db" opacity="0.12" />
        <path :d="reflPath" fill="none" stroke="#3498db" stroke-width="2" />

        <!-- Transmittance fill + curve -->
        <path :d="transFillPath" fill="#e67e22" opacity="0.12" />
        <path :d="transPath" fill="none" stroke="#e67e22" stroke-width="2" />

        <!-- Absorption fill + curve -->
        <path :d="absFillPath" fill="#95a5a6" opacity="0.08" />
        <path :d="absPath" fill="none" stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="5,3" />

        <!-- Legend -->
        <rect :x="pad.left+12" :y="pad.top+6" width="170" height="50" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.6" opacity="0.92" />
        <line :x1="pad.left+20" :y1="pad.top+18" :x2="pad.left+38" :y2="pad.top+18" stroke="#3498db" stroke-width="2.5" />
        <text :x="pad.left+44" :y="pad.top+22" class="legend-text">{{ t('Reflectance (R)', '반사율 (R)') }}</text>
        <line :x1="pad.left+20" :y1="pad.top+32" :x2="pad.left+38" :y2="pad.top+32" stroke="#e67e22" stroke-width="2.5" />
        <text :x="pad.left+44" :y="pad.top+36" class="legend-text">{{ t('Transmittance (T)', '투과율 (T)') }}</text>
        <line :x1="pad.left+20" :y1="pad.top+46" :x2="pad.left+38" :y2="pad.top+46" stroke="#95a5a6" stroke-width="1.5" stroke-dasharray="5,3" />
        <text :x="pad.left+44" :y="pad.top+50" class="legend-text">{{ t('Absorption (A)', '흡수율 (A)') }}</text>

        <!-- Hover -->
        <template v-if="hoverIdx >= 0">
          <line :x1="xScale(hoverWl)" :y1="pad.top" :x2="xScale(hoverWl)" :y2="pad.top+plotH"
            stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverR)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
          <circle :cx="xScale(hoverWl)" :cy="yScale(hoverT)" r="4" fill="#e67e22" stroke="#fff" stroke-width="1" />
          <rect :x="tooltipX" :y="pad.top+4" width="140" height="48" rx="4"
            fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="tooltipX+8" :y="pad.top+17" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX+8" :y="pad.top+30" class="tooltip-text" fill="#3498db">R: {{ hoverR.toFixed(2) }}%</text>
          <text :x="tooltipX+8" :y="pad.top+43" class="tooltip-text" fill="#e67e22">T: {{ hoverT.toFixed(2) }}%</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive } from 'vue'
import { useLocale } from '../composables/useLocale'
import {
  tmmSpectrum, wlRange, MATERIALS, COATING_MATERIALS, SUBSTRATE_MATERIALS,
  type TmmLayer, type TmmResult
} from '../composables/tmm'

const { t } = useLocale()

// --- Material helpers ---
const coatingMats = COATING_MATERIALS
const substrateMats = SUBSTRATE_MATERIALS

function matName(key: string): string {
  return MATERIALS[key]?.name ?? key
}

const MAT_COLORS: Record<string, string> = {
  sio2:'#7fb3d8', si3n4:'#2aa198', hfo2:'#6c71c4', tio2:'#d35400',
  al2o3:'#8e44ad', mgf2:'#bdc3c7', ta2o5:'#e74c3c', zro2:'#16a085',
  nb2o5:'#c0392b', zns:'#f1c40f', znse:'#e67e22', caf2:'#ecf0f1',
  ito:'#1abc9c', aln:'#3498db', sic:'#2c3e50', ge:'#7f8c8d',
  polymer:'#dda0dd', tungsten:'#555555', glass:'#a9cce3',
}
function matColor(key: string): string {
  return MAT_COLORS[key] || '#999'
}

// --- Layer state ---
interface LayerEntry { id: number; material: string; thicknessNm: number }
let nextId = 0

const layers = reactive<LayerEntry[]>([
  { id: nextId++, material: 'sio2', thicknessNm: 10 },
  { id: nextId++, material: 'hfo2', thicknessNm: 25 },
  { id: nextId++, material: 'sio2', thicknessNm: 15 },
  { id: nextId++, material: 'si3n4', thicknessNm: 30 },
])

const incidentMat = ref('air')
const substrateMat = ref('silicon')
const selectedPreset = ref('barl4')
const targetBand = ref<string>('visible')
const optimizing = ref(false)

function markCustom() { selectedPreset.value = 'custom' }

function addLayer() {
  layers.push({ id: nextId++, material: 'sio2', thicknessNm: 20 })
  markCustom()
}

function removeLayer(idx: number) {
  layers.splice(idx, 1)
  markCustom()
}

function moveLayer(idx: number, dir: number) {
  const target = idx + dir
  if (target < 0 || target >= layers.length) return
  const tmp = layers[idx]
  layers[idx] = layers[target]
  layers[target] = tmp
  markCustom()
}

function setThickness(idx: number, event: Event) {
  layers[idx].thicknessNm = parseInt((event.target as HTMLInputElement).value)
  markCustom()
}

// --- Presets ---
function applyPreset() {
  const p = selectedPreset.value
  if (p === 'custom') return
  incidentMat.value = 'air'
  substrateMat.value = 'silicon'
  layers.length = 0
  const presets: Record<string, { inc?: string; sub?: string; stack: [string, number][] }> = {
    barl4: { stack: [['sio2',10],['hfo2',25],['sio2',15],['si3n4',30]] },
    ar2: { stack: [['mgf2',95],['tio2',15]] },
    broadband: { stack: [['mgf2',90],['al2o3',30],['zro2',15],['mgf2',45]] },
    hr_blue: { stack: [['tio2',55],['sio2',90],['tio2',55],['sio2',90],['tio2',55],['sio2',90]] },
    nir_cut: { sub:'glass', stack: [['tio2',45],['sio2',80],['tio2',90],['sio2',80],['tio2',45]] },
  }
  const def = presets[p]
  if (!def) return
  if (def.inc) incidentMat.value = def.inc
  if (def.sub) substrateMat.value = def.sub
  for (const [mat, th] of def.stack) {
    layers.push({ id: nextId++, material: mat, thicknessNm: th })
  }
}

const totalThickness = computed(() => layers.reduce((s, l) => s + l.thicknessNm, 0))
const uniqueMaterials = computed(() => [...new Set(layers.map(l => l.material))])

// --- Band ranges ---
const bandRanges: Record<string, [number, number]> = {
  blue: [430, 470], green: [510, 560], red: [590, 640], visible: [430, 640], full: [380, 780],
}
const bandRange = computed(() => bandRanges[targetBand.value])

// --- SVG layout ---
const svgW = 620
const svgH = 320
const pad = { left: 52, right: 20, top: 20, bottom: 40 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom
const wlMin = 380, wlMax = 780
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]
const yTicks = [0, 20, 40, 60, 80, 100]
const yMax = 100

function xScale(wl: number): number { return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW }
function yScale(v: number): number { return pad.top + plotH - (v / yMax) * plotH }

// --- TMM computation ---
const wavelengths = computed(() => wlRange(0.38, 0.78, 0.005))
const wavelengthsNm = computed(() => wavelengths.value.map(w => Math.round(w * 1000)))

function buildStack(): TmmLayer[] {
  return layers.map(l => ({ material: l.material, thickness: l.thicknessNm / 1000 }))
}

const spectra = computed((): TmmResult[] => {
  if (layers.length === 0) return wavelengths.value.map(() => ({ R: 0, T: 1, A: 0, layerA: [] }))
  return tmmSpectrum(buildStack(), incidentMat.value, substrateMat.value, wavelengths.value, 0, 'avg')
})

const reflectances = computed(() => spectra.value.map(r => r.R * 100))
const transmittances = computed(() => spectra.value.map(r => r.T * 100))
const absorptions = computed(() => spectra.value.map(r => r.A * 100))

// --- Info cards ---
function avgInBand(data: number[]): number {
  const [lo, hi] = bandRange.value
  let sum = 0, cnt = 0
  for (let i = 0; i < wavelengthsNm.value.length; i++) {
    if (wavelengthsNm.value[i] >= lo && wavelengthsNm.value[i] <= hi) { sum += data[i]; cnt++ }
  }
  return cnt > 0 ? sum / cnt : 0
}
const avgRInBand = computed(() => avgInBand(reflectances.value))
const avgTInBand = computed(() => avgInBand(transmittances.value))

const minR = computed(() => {
  let minVal = Infinity, minWl = 550
  for (let i = 0; i < reflectances.value.length; i++) {
    if (reflectances.value[i] < minVal) { minVal = reflectances.value[i]; minWl = wavelengthsNm.value[i] }
  }
  return { value: minVal === Infinity ? 0 : minVal, wl: minWl }
})

// --- SVG paths ---
function makePath(data: number[]): string {
  return data.map((v, i) => {
    const x = xScale(wavelengthsNm.value[i]).toFixed(1)
    const y = yScale(Math.min(Math.max(v, 0), yMax)).toFixed(1)
    return (i === 0 ? 'M' : 'L') + x + ',' + y
  }).join(' ')
}
function makeFillPath(data: number[]): string {
  const line = makePath(data)
  const x0 = xScale(wavelengthsNm.value[0]).toFixed(1)
  const xN = xScale(wavelengthsNm.value[wavelengthsNm.value.length - 1]).toFixed(1)
  const yBase = yScale(0).toFixed(1)
  return line + ` L${xN},${yBase} L${x0},${yBase} Z`
}
const reflPath = computed(() => makePath(reflectances.value))
const reflFillPath = computed(() => makeFillPath(reflectances.value))
const transPath = computed(() => makePath(transmittances.value))
const transFillPath = computed(() => makeFillPath(transmittances.value))
const absPath = computed(() => makePath(absorptions.value))
const absFillPath = computed(() => makeFillPath(absorptions.value))

// --- Hover ---
const hoverIdx = ref(-1)
const hoverWl = computed(() => hoverIdx.value >= 0 ? wavelengthsNm.value[hoverIdx.value] : 0)
const hoverR = computed(() => hoverIdx.value >= 0 ? reflectances.value[hoverIdx.value] : 0)
const hoverT = computed(() => hoverIdx.value >= 0 ? transmittances.value[hoverIdx.value] : 0)
const tooltipX = computed(() => {
  if (hoverIdx.value < 0) return 0
  const x = xScale(hoverWl.value)
  return x + 150 > svgW - pad.right ? x - 150 : x + 10
})

function onMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mouseX = (event.clientX - rect.left) * (svgW / rect.width)
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const targetUm = wl / 1000
    let bestIdx = 0, bestDist = Infinity
    for (let i = 0; i < wavelengths.value.length; i++) {
      const d = Math.abs(wavelengths.value[i] - targetUm)
      if (d < bestDist) { bestDist = d; bestIdx = i }
    }
    hoverIdx.value = bestIdx
  } else {
    hoverIdx.value = -1
  }
}

// --- Auto Optimize (minimize avg R in band) ---
function computeAvgR(thickArr: number[]): number {
  const stack: TmmLayer[] = thickArr.map((th, i) => ({
    material: layers[i].material, thickness: th / 1000,
  }))
  const results = tmmSpectrum(stack, incidentMat.value, substrateMat.value, wavelengths.value, 0, 'avg')
  const [lo, hi] = bandRange.value
  let sum = 0, cnt = 0
  for (let i = 0; i < wavelengthsNm.value.length; i++) {
    if (wavelengthsNm.value[i] >= lo && wavelengthsNm.value[i] <= hi) { sum += results[i].R; cnt++ }
  }
  return cnt > 0 ? sum / cnt : 1
}

async function runOptimize() {
  if (layers.length === 0) return
  optimizing.value = true
  await new Promise(r => requestAnimationFrame(r))

  const n = layers.length
  const maxPerLayer = 200

  // Phase 1: coarse (step = max(3, total_combinations < 500000))
  let step = 5
  let totalCombs = 1
  for (let i = 0; i < n; i++) totalCombs *= Math.floor(maxPerLayer / step) + 1
  while (totalCombs > 300000 && step < 20) { step += 2; totalCombs = 1; for (let i = 0; i < n; i++) totalCombs *= Math.floor(maxPerLayer / step) + 1 }

  const coarseVals: number[][] = []
  for (let i = 0; i < n; i++) {
    const vals: number[] = []
    for (let v = 1; v <= maxPerLayer; v += step) vals.push(v)
    coarseVals.push(vals)
  }

  let bestR = Infinity
  let best = layers.map(l => l.thicknessNm)

  // Recursive grid search
  function searchCoarse(idx: number, current: number[]) {
    if (idx === n) {
      const r = computeAvgR(current)
      if (r < bestR) { bestR = r; best = [...current] }
      return
    }
    for (const v of coarseVals[idx]) {
      current[idx] = v
      searchCoarse(idx + 1, current)
    }
  }
  searchCoarse(0, new Array(n))

  await new Promise(r => requestAnimationFrame(r))

  // Phase 2: fine search (step 1nm, +/- step around best)
  const fineRange = step
  function searchFine(idx: number, current: number[]) {
    if (idx === n) {
      const r = computeAvgR(current)
      if (r < bestR) { bestR = r; best = [...current] }
      return
    }
    const lo = Math.max(1, best[idx] - fineRange)
    const hi = Math.min(maxPerLayer, best[idx] + fineRange)
    for (let v = lo; v <= hi; v++) {
      current[idx] = v
      searchFine(idx + 1, current)
    }
  }
  searchFine(0, new Array(n))

  for (let i = 0; i < n; i++) layers[i].thicknessNm = best[i]
  markCustom()
  optimizing.value = false
}

// --- Spectrum gradient ---
function wavelengthToCSS(wl: number): string {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) { r = -(wl-440)/60; b = 1 }
  else if (wl >= 440 && wl < 490) { g = (wl-440)/50; b = 1 }
  else if (wl >= 490 && wl < 510) { g = 1; b = -(wl-510)/20 }
  else if (wl >= 510 && wl < 580) { r = (wl-510)/70; g = 1 }
  else if (wl >= 580 && wl < 645) { r = 1; g = -(wl-645)/65 }
  else if (wl >= 645 && wl <= 780) { r = 1 }
  let f = 1.0
  if (wl >= 380 && wl < 420) f = 0.3 + 0.7*(wl-380)/40
  else if (wl >= 700 && wl <= 780) f = 0.3 + 0.7*(780-wl)/80
  r = Math.round(255*Math.pow(r*f,0.8))
  g = Math.round(255*Math.pow(g*f,0.8))
  b = Math.round(255*Math.pow(b*f,0.8))
  return `rgb(${r},${g},${b})`
}

const spectrumStops = computed(() => {
  const stops: { offset: string; color: string }[] = []
  for (let wl = wlMin; wl <= wlMax; wl += 20) {
    stops.push({ offset: ((wl-wlMin)/(wlMax-wlMin)*100)+'%', color: wavelengthToCSS(wl) })
  }
  return stops
})
</script>

<style scoped>
.barl-optimizer-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.barl-optimizer-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.media-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.media-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.media-group label {
  font-size: 0.85em;
  font-weight: 600;
  white-space: nowrap;
}
.controls-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 14px;
}
.select-group {
  display: flex;
  align-items: center;
  gap: 6px;
}
.select-group label {
  font-size: 0.85em;
  font-weight: 600;
  white-space: nowrap;
}
.bo-select {
  padding: 5px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.85em;
}
.optimize-btn {
  padding: 7px 18px;
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.85em;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}
.optimize-btn:hover:not(:disabled) { opacity: 0.85; }
.optimize-btn:disabled { opacity: 0.6; cursor: not-allowed; }

/* Layers */
.layers-section {
  margin-bottom: 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  padding: 10px 12px;
}
.layers-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.layers-title {
  font-size: 0.9em;
  font-weight: 700;
}
.layers-subtitle {
  font-size: 0.78em;
  color: var(--vp-c-text-3);
}
.add-btn {
  margin-left: auto;
  padding: 4px 12px;
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: transparent;
  color: var(--vp-c-brand-1);
  font-size: 0.82em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
}
.add-btn:hover { background: var(--vp-c-brand-1); color: #fff; }
.empty-layers {
  text-align: center;
  padding: 20px;
  color: var(--vp-c-text-3);
  font-size: 0.85em;
}
.layer-list {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.layer-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  flex-wrap: wrap;
}
.layer-num {
  font-size: 0.75em;
  font-weight: 700;
  color: var(--vp-c-text-3);
  min-width: 16px;
  text-align: center;
}
.layer-swatch {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  flex-shrink: 0;
}
.layer-mat-select {
  width: 120px;
  flex-shrink: 0;
}
.thickness-ctrl {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  min-width: 160px;
}
.bo-range {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 5px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.bo-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
}
.bo-range::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
}
.thickness-input {
  width: 52px;
  padding: 3px 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.82em;
  font-family: var(--vp-font-family-mono);
  text-align: right;
}
.thickness-input::-webkit-inner-spin-button,
.thickness-input::-webkit-outer-spin-button { opacity: 1; }
.unit {
  font-size: 0.78em;
  color: var(--vp-c-text-3);
}
.layer-actions {
  display: flex;
  gap: 3px;
  flex-shrink: 0;
}
.icon-btn {
  width: 24px;
  height: 24px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.7em;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}
.icon-btn:hover:not(:disabled) { background: var(--vp-c-bg-soft); }
.icon-btn:disabled { opacity: 0.3; cursor: default; }
.remove-btn:hover:not(:disabled) { background: #e74c3c22; color: #e74c3c; border-color: #e74c3c55; }

/* Info cards */
.info-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 7px 10px;
  font-size: 0.82em;
}
.info-label { color: var(--vp-c-text-2); margin-right: 4px; }
.info-value { font-weight: 600; font-family: var(--vp-font-family-mono); }

/* Stack viz */
.stack-viz-section { margin-bottom: 12px; }
.stack-viz-label { font-size: 0.82em; font-weight: 600; display: block; margin-bottom: 4px; }
.stack-bar-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
.stack-medium-label {
  font-size: 0.72em;
  color: var(--vp-c-text-3);
  white-space: nowrap;
}
.stack-bar {
  display: flex;
  height: 22px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  flex: 1;
}
.stack-segment {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 2px;
  transition: width 0.2s;
}
.stack-segment-label {
  font-size: 0.65em;
  font-weight: 600;
  color: #000;
  text-shadow: 0 0 2px rgba(255,255,255,0.6);
}
.stack-legend {
  display: flex;
  gap: 10px;
  margin-top: 4px;
  flex-wrap: wrap;
}
.stack-legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75em;
  color: var(--vp-c-text-2);
}
.stack-legend-swatch {
  width: 9px;
  height: 9px;
  border-radius: 2px;
  flex-shrink: 0;
}

/* Chart */
.svg-wrapper { margin-top: 6px; }
.bo-svg { width: 100%; max-width: 620px; display: block; margin: 0 auto; }
.axis-label { font-size: 9px; fill: var(--vp-c-text-2); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
.tooltip-text { font-size: 9px; fill: var(--vp-c-text-1); font-family: var(--vp-font-family-mono); }
.legend-text { font-size: 9px; fill: var(--vp-c-text-1); }

@media (max-width: 640px) {
  .layer-mat-select { width: 100px; }
  .thickness-ctrl { min-width: 120px; }
  .media-row, .controls-row { flex-direction: column; gap: 8px; }
}
</style>
