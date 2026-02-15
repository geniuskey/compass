<template>
  <div class="ls-container">
    <h4>{{ t('Lens Shading Simulator', '렌즈 쉐이딩 시뮬레이터') }}</h4>
    <p class="component-description">
      {{ t(
        'Simulate relative illumination across the sensor based on CRA, pixel design, and cos⁴ fall-off. Visualize per-channel color shading.',
        'CRA, 픽셀 설계, cos⁴ 감쇄를 기반으로 센서 전면의 상대 조도를 시뮬레이션합니다. 채널별 색 쉐이딩을 시각화합니다.'
      ) }}
    </p>

    <div class="controls-grid">
      <div class="slider-group">
        <label>{{ t('Max CRA (edge)', '최대 CRA (에지)') }}: <strong>{{ maxCRA }}&deg;</strong></label>
        <input type="range" min="5" max="40" step="1" v-model.number="maxCRA" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Pixel Pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} &mu;m</strong></label>
        <input type="range" min="0.5" max="3" step="0.05" v-model.number="pitch" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('ML Shift Ratio', 'ML 시프트 비') }}: <strong>{{ mlShift.toFixed(2) }}</strong></label>
        <input type="range" min="0" max="1" step="0.05" v-model.number="mlShift" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Channel', '채널') }}</label>
        <div class="channel-toggle">
          <button v-for="ch in channels" :key="ch.key"
            :class="['ch-btn', { active: channel === ch.key }]"
            :style="{ '--ch-color': ch.color }"
            @click="channel = ch.key"
          >{{ ch.label }}</button>
        </div>
      </div>
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Center RI', '중심 RI') }}</div>
        <div class="result-value highlight">100%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Corner RI', '코너 RI') }}</div>
        <div class="result-value">{{ cornerRI.toFixed(1) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Edge RI', '에지 RI') }}</div>
        <div class="result-value">{{ edgeRI.toFixed(1) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('RI Loss (stops)', 'RI 손실 (스톱)') }}</div>
        <div class="result-value">{{ cornerLoss.toFixed(2) }} EV</div>
      </div>
    </div>

    <div class="ls-visuals">
      <!-- 2D shading map -->
      <div class="shade-map">
        <h5>{{ t('2D Shading Map', '2D 쉐이딩 맵') }}</h5>
        <div class="shade-grid">
          <div v-for="(row, ri) in shadingGrid" :key="ri" class="shade-row">
            <div v-for="(val, ci) in row" :key="ci" class="shade-cell"
              :style="{ background: shadeCellColor(val) }">
            </div>
          </div>
        </div>
        <div class="shade-bar">
          <span>{{ cornerRI.toFixed(0) }}%</span>
          <div class="shade-gradient"></div>
          <span>100%</span>
        </div>
      </div>

      <!-- 1D profile -->
      <div class="profile-chart">
        <h5>{{ t('Radial Profile', '반경 프로파일') }}</h5>
        <svg :viewBox="`0 0 ${PW} ${PH}`" class="ls-svg">
          <line v-for="tick in pYTicks" :key="'pyg'+tick"
            :x1="pPad.left" :y1="pyScale(tick)" :x2="pPad.left + pPlotW" :y2="pyScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
          <line :x1="pPad.left" :y1="pPad.top" :x2="pPad.left" :y2="pPad.top + pPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="pPad.left" :y1="pPad.top + pPlotH" :x2="pPad.left + pPlotW" :y2="pPad.top + pPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <text v-for="tick in pYTicks" :key="'pyl'+tick"
            :x="pPad.left - 4" :y="pyScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick }}%</text>
          <text v-for="tick in pXTicks" :key="'pxl'+tick"
            :x="pxScale(tick)" :y="pPad.top + pPlotH + 12" text-anchor="middle" class="tick-label">{{ tick.toFixed(1) }}</text>
          <path :d="profilePathR" fill="none" stroke="#e74c3c" stroke-width="1.5" :opacity="channel === 'all' || channel === 'r' ? 0.8 : 0.2" />
          <path :d="profilePathG" fill="none" stroke="#27ae60" stroke-width="1.5" :opacity="channel === 'all' || channel === 'g' ? 0.8 : 0.2" />
          <path :d="profilePathB" fill="none" stroke="#3498db" stroke-width="1.5" :opacity="channel === 'all' || channel === 'b' ? 0.8 : 0.2" />
          <text :x="pPad.left + pPlotW / 2" :y="pPad.top + pPlotH + 26" text-anchor="middle" class="axis-title">{{ t('Relative Position (r/r_max)', '상대 위치 (r/r_max)') }}</text>
          <text :x="6" :y="pPad.top + pPlotH / 2" text-anchor="middle" class="axis-title"
            :transform="`rotate(-90, 6, ${pPad.top + pPlotH / 2})`">RI (%)</text>
        </svg>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const maxCRA = ref(25)
const pitch = ref(1.0)
const mlShift = ref(0.7)
const channel = ref<'all' | 'r' | 'g' | 'b'>('all')

const channels = [
  { key: 'all' as const, label: 'All', color: 'var(--vp-c-brand-1)' },
  { key: 'r' as const, label: 'R', color: '#e74c3c' },
  { key: 'g' as const, label: 'G', color: '#27ae60' },
  { key: 'b' as const, label: 'B', color: '#3498db' },
]

function riAtRadius(rNorm: number, chOffset: number): number {
  const cra = maxCRA.value * rNorm * (Math.PI / 180)
  const cos4 = Math.cos(cra) ** 4
  const mismatch = (1 - mlShift.value) * rNorm
  const mlCoupling = Math.exp(-2 * (mismatch * maxCRA.value / 15) ** 2)
  const cfShift = 1 - chOffset * rNorm * 0.02
  return cos4 * mlCoupling * Math.max(0.5, cfShift) * 100
}

function riR(r: number): number { return riAtRadius(r, -1) }
function riG(r: number): number { return riAtRadius(r, 0) }
function riB(r: number): number { return riAtRadius(r, 1.5) }

const cornerRI = computed(() => riG(1))
const edgeRI = computed(() => riG(0.707))
const cornerLoss = computed(() => -Math.log2(cornerRI.value / 100))

const GRID = 21
const shadingGrid = computed(() => {
  const grid: number[][] = []
  const fn = channel.value === 'r' ? riR : channel.value === 'b' ? riB : riG
  for (let y = 0; y < GRID; y++) {
    const row: number[] = []
    for (let x = 0; x < GRID; x++) {
      const nx = (x - GRID / 2) / (GRID / 2)
      const ny = (y - GRID / 2) / (GRID / 2)
      const r = Math.min(1, Math.sqrt(nx ** 2 + ny ** 2))
      row.push(fn(r))
    }
    grid.push(row)
  }
  return grid
})

function shadeCellColor(val: number): string {
  const minRI = cornerRI.value
  const t = Math.max(0, Math.min(1, (val - minRI) / (100 - minRI)))
  if (channel.value === 'r') return `rgba(231,76,60,${0.3 + t * 0.7})`
  if (channel.value === 'b') return `rgba(52,152,219,${0.3 + t * 0.7})`
  const g = Math.round(80 + t * 175)
  return `rgb(${g},${g},${g})`
}

const PW = 300, PH = 200
const pPad = { top: 12, right: 8, bottom: 32, left: 40 }
const pPlotW = PW - pPad.left - pPad.right
const pPlotH = PH - pPad.top - pPad.bottom

const pYMin = computed(() => Math.floor(Math.min(riR(1), riB(1), riG(1)) / 10) * 10)
const pYTicks = computed(() => {
  const ticks: number[] = []
  for (let v = pYMin.value; v <= 100; v += 10) ticks.push(v)
  return ticks
})
const pXTicks = [0, 0.25, 0.5, 0.75, 1.0]

function pxScale(r: number): number { return pPad.left + r * pPlotW }
function pyScale(ri: number): number { return pPad.top + pPlotH - ((ri - pYMin.value) / (100 - pYMin.value)) * pPlotH }

function profilePath(fn: (r: number) => number): string {
  let d = ''
  for (let i = 0; i <= 100; i++) {
    const r = i / 100
    const ri = fn(r)
    d += `${i === 0 ? 'M' : 'L'}${pxScale(r).toFixed(1)},${pyScale(ri).toFixed(1)}`
  }
  return d
}

const profilePathR = computed(() => profilePath(riR))
const profilePathG = computed(() => profilePath(riG))
const profilePathB = computed(() => profilePath(riB))
</script>

<style scoped>
.ls-container { border: 1px solid var(--vp-c-divider); border-radius: 12px; padding: 24px; margin: 24px 0; background: var(--vp-c-bg-soft); }
.ls-container h4 { margin: 0 0 4px 0; font-size: 1.1em; color: var(--vp-c-brand-1); }
.ls-container h5 { margin: 12px 0 6px 0; font-size: 0.9em; color: var(--vp-c-text-1); }
.component-description { margin: 0 0 16px 0; color: var(--vp-c-text-2); font-size: 0.9em; }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 16px; }
.slider-group label { display: block; margin-bottom: 4px; font-size: 0.85em; }
.ctrl-range { width: 100%; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: var(--vp-c-divider); outline: none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--vp-c-brand-1); cursor: pointer; }
.channel-toggle { display: flex; gap: 0; border: 1px solid var(--vp-c-divider); border-radius: 8px; overflow: hidden; }
.ch-btn { padding: 5px 12px; font-size: 0.82em; font-weight: 500; border: none; background: var(--vp-c-bg); color: var(--vp-c-text-2); cursor: pointer; }
.ch-btn.active { background: var(--ch-color); color: #fff; }
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.result-card { background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider); border-radius: 8px; padding: 12px; text-align: center; }
.result-label { font-size: 0.8em; color: var(--vp-c-text-2); margin-bottom: 4px; }
.result-value { font-weight: 600; font-family: var(--vp-font-family-mono); }
.result-value.highlight { color: var(--vp-c-brand-1); }
.ls-visuals { display: flex; gap: 20px; flex-wrap: wrap; }
.shade-map { flex: 0 0 auto; }
.profile-chart { flex: 1; min-width: 280px; }
.shade-grid { display: grid; grid-template-columns: repeat(21, 1fr); gap: 1px; width: 210px; height: 210px; border: 1px solid var(--vp-c-divider); border-radius: 8px; overflow: hidden; }
.shade-row { display: contents; }
.shade-cell { aspect-ratio: 1; }
.shade-bar { display: flex; align-items: center; gap: 6px; margin-top: 4px; font-size: 0.75em; color: var(--vp-c-text-3); }
.shade-gradient { flex: 1; height: 8px; border-radius: 4px; background: linear-gradient(to right, #555, #fff); }
.ls-svg { width: 100%; max-width: 300px; display: block; }
.tick-label { font-size: 9px; fill: var(--vp-c-text-3); }
.axis-title { font-size: 10px; fill: var(--vp-c-text-2); font-weight: 600; }
</style>
