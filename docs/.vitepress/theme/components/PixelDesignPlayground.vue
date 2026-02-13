<template>
  <div class="pixel-playground">
    <h4>{{ t('Pixel Design Playground', '픽셀 설계 플레이그라운드') }}</h4>
    <p class="component-description">
      {{ t(
        'Configure all pixel parameters and explore multi-panel results: QE spectra, layer stack visualization, and energy budget analysis powered by TMM.',
        '모든 픽셀 파라미터를 구성하고 TMM 기반의 다중 패널 결과(QE 스펙트럼, 레이어 스택 시각화, 에너지 버짓 분석)를 탐색합니다.'
      ) }}
    </p>

    <div class="playground-layout">
      <!-- Left Panel: Controls -->
      <div class="controls-panel">
        <!-- Preset selector -->
        <div class="ctrl-section">
          <label class="section-label">{{ t('Preset', '프리셋') }}</label>
          <select v-model="preset" class="preset-select" @change="applyPreset">
            <option value="bsi_1um">BSI 1um ({{ t('default', '기본') }})</option>
            <option value="bsi_08um">BSI 0.8um ({{ t('thin Si', '얇은 Si') }})</option>
            <option value="high_qe">{{ t('High QE (thick Si)', '고 QE (두꺼운 Si)') }}</option>
            <option value="custom">{{ t('Custom', '사용자 정의') }}</option>
          </select>
        </div>

        <!-- Pixel Parameters -->
        <div class="ctrl-section">
          <div class="section-header">{{ t('Pixel Parameters', '픽셀 파라미터') }}</div>
          <div class="slider-group">
            <label>{{ t('Pixel pitch:', '픽셀 피치:') }} <strong>{{ pitch.toFixed(1) }} um</strong></label>
            <input type="range" min="0.5" max="2.0" step="0.1" v-model.number="pitch" class="ctrl-range" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Silicon thickness:', '실리콘 두께:') }} <strong>{{ siThickness.toFixed(1) }} um</strong></label>
            <input type="range" min="1.0" max="5.0" step="0.1" v-model.number="siThickness" class="ctrl-range" @input="markCustom" />
          </div>
        </div>

        <!-- BARL Parameters (collapsible) -->
        <div class="ctrl-section">
          <div class="section-header collapsible" @click="barlOpen = !barlOpen">
            {{ t('BARL Parameters', 'BARL 파라미터') }}
            <span class="toggle-icon">{{ barlOpen ? '\u25B2' : '\u25BC' }}</span>
          </div>
          <div v-show="barlOpen" class="collapsible-body">
            <div class="slider-group">
              <label>SiO2 #1: <strong>{{ barl1 }} nm</strong></label>
              <input type="range" min="0" max="50" step="1" v-model.number="barl1" class="ctrl-range" @input="markCustom" />
            </div>
            <div class="slider-group">
              <label>HfO2: <strong>{{ barl2 }} nm</strong></label>
              <input type="range" min="0" max="50" step="1" v-model.number="barl2" class="ctrl-range" @input="markCustom" />
            </div>
            <div class="slider-group">
              <label>SiO2 #3: <strong>{{ barl3 }} nm</strong></label>
              <input type="range" min="0" max="50" step="1" v-model.number="barl3" class="ctrl-range" @input="markCustom" />
            </div>
            <div class="slider-group">
              <label>Si3N4: <strong>{{ barl4 }} nm</strong></label>
              <input type="range" min="0" max="80" step="1" v-model.number="barl4" class="ctrl-range" @input="markCustom" />
            </div>
          </div>
        </div>

        <!-- Color Filter -->
        <div class="ctrl-section">
          <div class="section-header">{{ t('Color Filter', '컬러 필터') }}</div>
          <div class="slider-group">
            <label>{{ t('CF thickness:', 'CF 두께:') }} <strong>{{ cfThickness.toFixed(2) }} um</strong></label>
            <input type="range" min="0.2" max="1.0" step="0.05" v-model.number="cfThickness" class="ctrl-range" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Channel:', '채널:') }}</label>
            <div class="pol-btns">
              <button :class="['pol-btn', { active: cfChannel === 'red' }]" @click="cfChannel = 'red'" style="color:#e74c3c">R</button>
              <button :class="['pol-btn', { active: cfChannel === 'green' }]" @click="cfChannel = 'green'" style="color:#27ae60">G</button>
              <button :class="['pol-btn', { active: cfChannel === 'blue' }]" @click="cfChannel = 'blue'" style="color:#3498db">B</button>
            </div>
          </div>
        </div>

        <!-- Top Layers (collapsible) -->
        <div class="ctrl-section">
          <div class="section-header collapsible" @click="topOpen = !topOpen">
            {{ t('Top Layers', '상부 레이어') }}
            <span class="toggle-icon">{{ topOpen ? '\u25B2' : '\u25BC' }}</span>
          </div>
          <div v-show="topOpen" class="collapsible-body">
            <div class="slider-group">
              <label>{{ t('Planarization:', '평탄화:') }} <strong>{{ planThickness.toFixed(2) }} um</strong></label>
              <input type="range" min="0.1" max="0.5" step="0.05" v-model.number="planThickness" class="ctrl-range" @input="markCustom" />
            </div>
            <div class="slider-group">
              <label>{{ t('Microlens:', '마이크로렌즈:') }} <strong>{{ mlThickness.toFixed(2) }} um</strong></label>
              <input type="range" min="0.2" max="1.0" step="0.05" v-model.number="mlThickness" class="ctrl-range" @input="markCustom" />
            </div>
          </div>
        </div>

        <!-- Simulation Settings -->
        <div class="ctrl-section">
          <div class="section-header">{{ t('Simulation Settings', '시뮬레이션 설정') }}</div>
          <div class="slider-group">
            <label>{{ t('Angle of incidence:', '입사각:') }} <strong>{{ angle }}deg</strong></label>
            <input type="range" min="0" max="30" step="1" v-model.number="angle" class="ctrl-range" />
          </div>
          <div class="slider-group">
            <label>{{ t('Polarization:', '편광:') }}</label>
            <div class="pol-btns">
              <button :class="['pol-btn', { active: pol === 'avg' }]" @click="pol = 'avg'">{{ t('Unpolarized', '비편광') }}</button>
              <button :class="['pol-btn', { active: pol === 's' }]" @click="pol = 's'">s</button>
              <button :class="['pol-btn', { active: pol === 'p' }]" @click="pol = 'p'">p</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel: Results -->
      <div class="results-panel">
        <!-- Tab buttons -->
        <div class="tab-row">
          <button :class="['tab-btn', { active: activeTab === 'qe' }]" @click="activeTab = 'qe'">
            {{ t('QE Spectrum', 'QE 스펙트럼') }}
          </button>
          <button :class="['tab-btn', { active: activeTab === 'stack' }]" @click="activeTab = 'stack'">
            {{ t('Layer Stack', '레이어 스택') }}
          </button>
          <button :class="['tab-btn', { active: activeTab === 'energy' }]" @click="activeTab = 'energy'">
            {{ t('Energy Budget', '에너지 버짓') }}
          </button>
        </div>

        <!-- Tab 1: QE Spectrum -->
        <div v-show="activeTab === 'qe'" class="tab-content">
          <svg :viewBox="`0 0 ${qeW} ${qeH}`" class="result-svg" @mousemove="onQeHover" @mouseleave="qeHoverWl = null">
            <defs>
              <linearGradient id="pdpVisSpec" x1="0" y1="0" x2="1" y2="0">
                <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
              </linearGradient>
            </defs>
            <rect :x="qePad.left" :y="qePad.top + qePlotH + 2" :width="qePlotW" height="8" fill="url(#pdpVisSpec)" rx="2" />
            <!-- Grid -->
            <line v-for="tick in qeYTicks" :key="'qeyg'+tick" :x1="qePad.left" :y1="qeYScale(tick)" :x2="qePad.left+qePlotW" :y2="qeYScale(tick)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in qeXTicks" :key="'qexg'+tick" :x1="qeXScale(tick)" :y1="qePad.top" :x2="qeXScale(tick)" :y2="qePad.top+qePlotH" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <!-- Axes -->
            <line :x1="qePad.left" :y1="qePad.top" :x2="qePad.left" :y2="qePad.top+qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="qePad.left" :y1="qePad.top+qePlotH" :x2="qePad.left+qePlotW" :y2="qePad.top+qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <!-- Y labels -->
            <text v-for="tick in qeYTicks" :key="'qeyl'+tick" :x="qePad.left-6" :y="qeYScale(tick)+3" text-anchor="end" class="axis-label">{{ tick }}%</text>
            <!-- X labels -->
            <text v-for="tick in qeXTicks" :key="'qexl'+tick" :x="qeXScale(tick)" :y="qePad.top+qePlotH+22" text-anchor="middle" class="axis-label">{{ tick }}</text>
            <!-- Axis titles -->
            <text :x="qePad.left+qePlotW/2" :y="qeH-2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
            <text :x="12" :y="qePad.top+qePlotH/2" text-anchor="middle" class="axis-title" :transform="`rotate(-90,12,${qePad.top+qePlotH/2})`">QE (%)</text>
            <!-- Filled areas -->
            <path :d="blueArea" fill="#3498db" opacity="0.08" />
            <path :d="greenArea" fill="#27ae60" opacity="0.08" />
            <path :d="redArea" fill="#e74c3c" opacity="0.08" />
            <!-- Curves -->
            <path :d="bluePath" fill="none" stroke="#3498db" stroke-width="2" opacity="0.9" />
            <path :d="greenPath" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.9" />
            <path :d="redPath" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.9" />
            <!-- Hover -->
            <template v-if="qeHoverWl !== null">
              <line :x1="qeXScale(qeHoverWl)" :y1="qePad.top" :x2="qeXScale(qeHoverWl)" :y2="qePad.top+qePlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
              <circle :cx="qeXScale(qeHoverWl)" :cy="qeYScale(qeHoverVals.b)" r="4" fill="#3498db" stroke="#fff" stroke-width="1" />
              <circle :cx="qeXScale(qeHoverWl)" :cy="qeYScale(qeHoverVals.g)" r="4" fill="#27ae60" stroke="#fff" stroke-width="1" />
              <circle :cx="qeXScale(qeHoverWl)" :cy="qeYScale(qeHoverVals.r)" r="4" fill="#e74c3c" stroke="#fff" stroke-width="1" />
              <rect :x="qeTooltipX" :y="qePad.top+4" width="120" height="58" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
              <text :x="qeTooltipX+8" :y="qePad.top+18" class="tooltip-text">{{ qeHoverWl }} nm</text>
              <text :x="qeTooltipX+8" :y="qePad.top+32" class="tooltip-text" fill="#3498db">B: {{ qeHoverVals.b.toFixed(1) }}%</text>
              <text :x="qeTooltipX+8" :y="qePad.top+44" class="tooltip-text" fill="#27ae60">G: {{ qeHoverVals.g.toFixed(1) }}%</text>
              <text :x="qeTooltipX+8" :y="qePad.top+56" class="tooltip-text" fill="#e74c3c">R: {{ qeHoverVals.r.toFixed(1) }}%</text>
            </template>
            <!-- Legend -->
            <line :x1="qePad.left+qePlotW-90" :y1="qePad.top+12" :x2="qePad.left+qePlotW-72" :y2="qePad.top+12" stroke="#3498db" stroke-width="2" />
            <text :x="qePad.left+qePlotW-68" :y="qePad.top+16" class="legend-label">{{ t('Blue', '파랑') }}</text>
            <line :x1="qePad.left+qePlotW-90" :y1="qePad.top+26" :x2="qePad.left+qePlotW-72" :y2="qePad.top+26" stroke="#27ae60" stroke-width="2" />
            <text :x="qePad.left+qePlotW-68" :y="qePad.top+30" class="legend-label">{{ t('Green', '초록') }}</text>
            <line :x1="qePad.left+qePlotW-90" :y1="qePad.top+40" :x2="qePad.left+qePlotW-72" :y2="qePad.top+40" stroke="#e74c3c" stroke-width="2" />
            <text :x="qePad.left+qePlotW-68" :y="qePad.top+44" class="legend-label">{{ t('Red', '빨강') }}</text>
          </svg>
        </div>

        <!-- Tab 2: Layer Stack -->
        <div v-show="activeTab === 'stack'" class="tab-content">
          <svg :viewBox="`0 0 ${stackW} ${stackH}`" class="result-svg">
            <defs>
              <marker id="pdpArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L10,5 L0,10 Z" fill="var(--vp-c-text-2)" />
              </marker>
            </defs>
            <!-- Light arrow at top -->
            <line :x1="stackPad + stackBarW / 2" y1="4" :x2="stackPad + stackBarW / 2" :y2="(stackVis[0]?.y ?? 30) - 2" stroke="var(--vp-c-text-2)" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#pdpArrow)" />
            <text :x="stackPad + stackBarW / 2 + 16" y="16" class="axis-title">{{ t('Light', '빛') }}</text>
            <!-- Layer rectangles -->
            <template v-for="(layer, idx) in stackVis" :key="'sl'+idx">
              <rect :x="stackPad" :y="layer.y" :width="stackBarW" :height="Math.max(layer.h, 1)" :fill="layer.color" :stroke="layer.borderColor" stroke-width="0.5" rx="2" />
              <text :x="stackPad + stackBarW + 10" :y="layer.y + Math.max(layer.h, 1) / 2 + 4" class="stack-label">{{ layer.label }}</text>
              <text :x="stackPad + stackBarW + 10" :y="layer.y + Math.max(layer.h, 1) / 2 + 16" class="stack-thickness">{{ layer.thicknessLabel }}</text>
            </template>
            <!-- Total thickness -->
            <text :x="stackPad + stackBarW / 2" :y="stackH - 6" text-anchor="middle" class="axis-title">
              {{ t('Total:', '합계:') }} {{ totalStackThickness.toFixed(2) }} um
            </text>
          </svg>
        </div>

        <!-- Tab 3: Energy Budget -->
        <div v-show="activeTab === 'energy'" class="tab-content">
          <svg :viewBox="`0 0 ${ebW} ${ebH}`" class="result-svg">
            <template v-for="(bar, idx) in energyBars" :key="'eb'+idx">
              <!-- Channel label -->
              <text :x="ebPad.left - 8" :y="bar.y + bar.h / 2 + 4" text-anchor="end" class="axis-label" :fill="bar.color">{{ bar.name }}</text>
              <!-- Reflection segment -->
              <rect :x="ebPad.left" :y="bar.y" :width="Math.max(bar.rW, 0)" :height="bar.h" fill="#95a5a6" rx="1" />
              <!-- Other losses segment -->
              <rect :x="ebPad.left + bar.rW" :y="bar.y" :width="Math.max(bar.lossW, 0)" :height="bar.h" fill="#e67e22" rx="1" />
              <!-- QE (silicon absorption) segment -->
              <rect :x="ebPad.left + bar.rW + bar.lossW" :y="bar.y" :width="Math.max(bar.qeW, 0)" :height="bar.h" :fill="bar.color" rx="1" />
              <!-- Transmission segment -->
              <rect :x="ebPad.left + bar.rW + bar.lossW + bar.qeW" :y="bar.y" :width="Math.max(bar.tW, 0)" :height="bar.h" fill="#d5dbdb" rx="1" />
              <!-- QE percentage label inside bar -->
              <text v-if="bar.qeW > 30" :x="ebPad.left + bar.rW + bar.lossW + bar.qeW / 2" :y="bar.y + bar.h / 2 + 4" text-anchor="middle" class="bar-label">{{ bar.qePct }}%</text>
            </template>
            <!-- X-axis ticks -->
            <line v-for="pct in [0, 25, 50, 75, 100]" :key="'ebt'+pct" :x1="ebPad.left + pct / 100 * ebPlotW" :y1="ebPad.top + ebPlotH" :x2="ebPad.left + pct / 100 * ebPlotW" :y2="ebPad.top + ebPlotH + 4" stroke="var(--vp-c-text-2)" stroke-width="0.5" />
            <text v-for="pct in [0, 25, 50, 75, 100]" :key="'ebl'+pct" :x="ebPad.left + pct / 100 * ebPlotW" :y="ebPad.top + ebPlotH + 16" text-anchor="middle" class="axis-label">{{ pct }}%</text>
            <!-- Legend -->
            <rect :x="ebPad.left" :y="ebH - 20" width="10" height="10" fill="#95a5a6" rx="1" />
            <text :x="ebPad.left + 14" :y="ebH - 11" class="legend-label">{{ t('Reflection', '반사') }}</text>
            <rect :x="ebPad.left + 80" :y="ebH - 20" width="10" height="10" fill="#e67e22" rx="1" />
            <text :x="ebPad.left + 94" :y="ebH - 11" class="legend-label">{{ t('Other loss', '기타 손실') }}</text>
            <rect :x="ebPad.left + 170" :y="ebH - 20" width="10" height="10" fill="#27ae60" rx="1" />
            <text :x="ebPad.left + 184" :y="ebH - 11" class="legend-label">{{ t('QE (Si abs.)', 'QE (Si 흡수)') }}</text>
            <rect :x="ebPad.left + 275" :y="ebH - 20" width="10" height="10" fill="#d5dbdb" rx="1" />
            <text :x="ebPad.left + 289" :y="ebH - 11" class="legend-label">{{ t('Transmission', '투과') }}</text>
          </svg>
        </div>
      </div>
    </div>

    <!-- Bottom: Key Metrics Summary -->
    <div class="metrics-row">
      <div class="metric-card" style="border-top: 3px solid #e74c3c;">
        <div class="metric-label">{{ t('Peak QE R', '피크 QE R') }}</div>
        <div class="metric-value">{{ peakQeR.toFixed(1) }}%</div>
      </div>
      <div class="metric-card" style="border-top: 3px solid #27ae60;">
        <div class="metric-label">{{ t('Peak QE G', '피크 QE G') }}</div>
        <div class="metric-value">{{ peakQeG.toFixed(1) }}%</div>
      </div>
      <div class="metric-card" style="border-top: 3px solid #3498db;">
        <div class="metric-label">{{ t('Peak QE B', '피크 QE B') }}</div>
        <div class="metric-value">{{ peakQeB.toFixed(1) }}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ t('Avg QE (vis)', '평균 QE (가시)') }}</div>
        <div class="metric-value">{{ avgQeVis.toFixed(1) }}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ t('Avg Reflectance', '평균 반사율') }}</div>
        <div class="metric-value">{{ avgReflectance.toFixed(1) }}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">{{ t('Stack Height', '스택 높이') }}</div>
        <div class="metric-value">{{ totalStackThickness.toFixed(2) }} um</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { tmmCalc, wlRange, MATERIALS, type TmmLayer, type TmmResult } from '../composables/tmm'

const { t } = useLocale()

// ── Control state ────────────────────────────────────────────────────────────
const preset = ref('bsi_1um')
const pitch = ref(1.0)
const siThickness = ref(3.0)
const barl1 = ref(10)   // SiO2 layer 1 (nm)
const barl2 = ref(25)   // HfO2 (nm)
const barl3 = ref(15)   // SiO2 layer 3 (nm)
const barl4 = ref(30)   // Si3N4 (nm)
const cfThickness = ref(0.6)
const planThickness = ref(0.3)
const mlThickness = ref(0.6)
const cfChannel = ref<'red' | 'green' | 'blue'>('green')
const angle = ref(0)
const pol = ref<'s' | 'p' | 'avg'>('avg')
const barlOpen = ref(false)
const topOpen = ref(false)
const activeTab = ref<'qe' | 'stack' | 'energy'>('qe')

const presetDefs: Record<string, {
  pitch: number; si: number; barl: number[]; cf: number; plan: number; ml: number
}> = {
  bsi_1um:  { pitch: 1.0, si: 3.0, barl: [10, 25, 15, 30], cf: 0.6,  plan: 0.3,  ml: 0.6 },
  bsi_08um: { pitch: 0.8, si: 2.5, barl: [8, 20, 12, 25],  cf: 0.5,  plan: 0.25, ml: 0.5 },
  high_qe:  { pitch: 1.0, si: 4.0, barl: [15, 30, 20, 40], cf: 0.6,  plan: 0.3,  ml: 0.6 },
}

function applyPreset() {
  const p = presetDefs[preset.value]
  if (!p) return
  pitch.value = p.pitch
  siThickness.value = p.si
  barl1.value = p.barl[0]
  barl2.value = p.barl[1]
  barl3.value = p.barl[2]
  barl4.value = p.barl[3]
  cfThickness.value = p.cf
  planThickness.value = p.plan
  mlThickness.value = p.ml
}

function markCustom() {
  preset.value = 'custom'
}

const cfMatKey = computed(() => 'cf_' + cfChannel.value)

// ── Stack builder ────────────────────────────────────────────────────────────
function buildStack(cfMat: string): TmmLayer[] {
  return [
    { material: 'polymer', thickness: mlThickness.value },
    { material: 'sio2', thickness: planThickness.value },
    { material: cfMat, thickness: cfThickness.value },
    { material: 'si3n4', thickness: barl4.value / 1000 },
    { material: 'sio2', thickness: barl3.value / 1000 },
    { material: 'hfo2', thickness: barl2.value / 1000 },
    { material: 'sio2', thickness: barl1.value / 1000 },
    { material: 'silicon', thickness: siThickness.value },
  ]
}

const SI_IDX = 7

// ── TMM spectrum computation ─────────────────────────────────────────────────
const wavelengths = computed(() => wlRange(0.38, 0.78, 0.005))

interface ChannelSpectrum {
  qe: number[]
  R: number[]
  T: number[]
  A: number[]
}

function computeChannel(cfMat: string): ChannelSpectrum {
  const stack = buildStack(cfMat)
  const wls = wavelengths.value
  const p = pol.value
  const a = angle.value
  const qe: number[] = []
  const R: number[] = []
  const T: number[] = []
  const A: number[] = []
  for (const wl of wls) {
    const res = tmmCalc(stack, 'air', 'sio2', wl, a, p)
    qe.push(res.layerA[SI_IDX] * 100)
    R.push(res.R * 100)
    T.push(res.T * 100)
    A.push(res.A * 100)
  }
  return { qe, R, T, A }
}

const redSpec = computed(() => computeChannel('cf_red'))
const greenSpec = computed(() => computeChannel('cf_green'))
const blueSpec = computed(() => computeChannel('cf_blue'))

// ── QE Spectrum Chart ────────────────────────────────────────────────────────
const qeW = 520
const qeH = 280
const qePad = { top: 20, right: 20, bottom: 35, left: 50 }
const qePlotW = qeW - qePad.left - qePad.right
const qePlotH = qeH - qePad.top - qePad.bottom
const qeYMax = 80
const qeYTicks = [0, 20, 40, 60, 80]
const qeXTicks = [400, 450, 500, 550, 600, 650, 700, 750]

function qeXScale(wlNm: number): number {
  return qePad.left + ((wlNm - 380) / (780 - 380)) * qePlotW
}
function qeYScale(v: number): number {
  return qePad.top + qePlotH - (v / qeYMax) * qePlotH
}

function buildPath(data: number[]): string {
  const wls = wavelengths.value
  return data.map((v, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${qeXScale(wls[i] * 1000).toFixed(1)},${qeYScale(v).toFixed(1)}`
  }).join(' ')
}

function buildArea(data: number[]): string {
  const wls = wavelengths.value
  const line = buildPath(data)
  const lastNm = wls[wls.length - 1] * 1000
  const firstNm = wls[0] * 1000
  return `${line} L${qeXScale(lastNm).toFixed(1)},${qeYScale(0).toFixed(1)} L${qeXScale(firstNm).toFixed(1)},${qeYScale(0).toFixed(1)} Z`
}

const redPath = computed(() => buildPath(redSpec.value.qe))
const greenPath = computed(() => buildPath(greenSpec.value.qe))
const bluePath = computed(() => buildPath(blueSpec.value.qe))
const redArea = computed(() => buildArea(redSpec.value.qe))
const greenArea = computed(() => buildArea(greenSpec.value.qe))
const blueArea = computed(() => buildArea(blueSpec.value.qe))

// QE hover tooltip
const qeHoverWl = ref<number | null>(null)
const qeHoverVals = ref({ r: 0, g: 0, b: 0 })

function onQeHover(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = qeW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wlNm = 380 + ((mouseX - qePad.left) / qePlotW) * (780 - 380)
  if (wlNm >= 380 && wlNm <= 780) {
    const snapped = Math.round(wlNm)
    qeHoverWl.value = snapped
    const wls = wavelengths.value
    const idx = Math.min(Math.max(Math.round((snapped / 1000 - 0.38) / 0.005), 0), wls.length - 1)
    qeHoverVals.value = {
      r: redSpec.value.qe[idx] ?? 0,
      g: greenSpec.value.qe[idx] ?? 0,
      b: blueSpec.value.qe[idx] ?? 0,
    }
  } else {
    qeHoverWl.value = null
  }
}

const qeTooltipX = computed(() => {
  if (qeHoverWl.value === null) return 0
  const x = qeXScale(qeHoverWl.value)
  return x + 130 > qeW - qePad.right ? x - 130 : x + 10
})

// ── Layer Stack Visualization ────────────────────────────────────────────────
const stackW = 520
const stackH = 350
const stackPad = 60
const stackBarW = 180

interface StackLayerVis {
  label: string
  thicknessLabel: string
  color: string
  borderColor: string
  y: number
  h: number
}

const layerColors: Record<string, { fill: string; border: string }> = {
  polymer: { fill: '#dda0dd', border: '#b370b3' },
  sio2:    { fill: '#7fb3d8', border: '#5a9bc5' },
  cf_red:  { fill: '#e74c3c', border: '#c0392b' },
  cf_green:{ fill: '#27ae60', border: '#1e8449' },
  cf_blue: { fill: '#3498db', border: '#2980b9' },
  hfo2:    { fill: '#6c71c4', border: '#585cb0' },
  si3n4:   { fill: '#2aa198', border: '#1e7b73' },
  silicon: { fill: '#5d6d7e', border: '#4a5a6a' },
}

const totalStackThickness = computed(() => {
  const stack = buildStack(cfMatKey.value)
  return stack.reduce((sum, l) => sum + l.thickness, 0)
})

const stackVis = computed((): StackLayerVis[] => {
  const stack = buildStack(cfMatKey.value)
  const totalUm = totalStackThickness.value
  if (totalUm <= 0) return []
  const availH = stackH - 60
  const minPx = 14

  // Proportional heights with minimum enforcement
  const rawH = stack.map(l => (l.thickness / totalUm) * availH)
  const smallSum = rawH.reduce((s, h) => s + (h < minPx ? minPx - h : 0), 0)
  const bigSum = rawH.reduce((s, h) => s + (h >= minPx ? h : 0), 0)
  const scale = bigSum > 0 ? Math.max(0, 1 - smallSum / bigSum) : 1
  const heights = rawH.map(h => h < minPx ? minPx : h * scale)

  const cfDisplayNames: Record<string, string[]> = {
    red: ['Red', '빨강'],
    green: ['Green', '초록'],
    blue: ['Blue', '파랑'],
  }
  const cfDisplay = cfDisplayNames[cfChannel.value] || ['Green', '초록']

  const nameMap: Record<string, string[]> = {
    polymer: ['Microlens (Polymer)', '마이크로렌즈 (Polymer)'],
    [cfMatKey.value]: [`Color Filter (${cfDisplay[0]})`, `컬러 필터 (${cfDisplay[1]})`],
    si3n4: ['Si\u2083N\u2084 (BARL)', 'Si\u2083N\u2084 (BARL)'],
    hfo2: ['HfO\u2082 (BARL)', 'HfO\u2082 (BARL)'],
    silicon: ['Silicon', '실리콘'],
  }

  const layers: StackLayerVis[] = []
  let y = 28
  for (let i = 0; i < stack.length; i++) {
    const l = stack[i]
    const h = heights[i]
    const c = layerColors[l.material] || { fill: '#bbb', border: '#999' }
    const nm = l.thickness * 1000
    const thLabel = l.thickness >= 0.1 ? `${l.thickness.toFixed(2)} um` : `${nm.toFixed(0)} nm`

    let label: string
    if (l.material === 'sio2') {
      if (i === 1) label = t('Planarization (SiO\u2082)', '\uD3C9\uD0C4\uD654 (SiO\u2082)')
      else if (i === 4) label = 'SiO\u2082 (BARL #3)'
      else if (i === 6) label = 'SiO\u2082 (BARL #1)'
      else label = 'SiO\u2082'
    } else {
      const names = nameMap[l.material]
      label = names ? t(names[0], names[1]) : (MATERIALS[l.material]?.name || l.material)
    }

    layers.push({ label, thicknessLabel: thLabel, color: c.fill, borderColor: c.border, y, h })
    y += h + 1
  }
  return layers
})

// ── Energy Budget ────────────────────────────────────────────────────────────
const ebW = 520
const ebH = 200
const ebPad = { top: 20, right: 20, bottom: 35, left: 50 }
const ebPlotW = ebW - ebPad.left - ebPad.right
const ebPlotH = ebH - ebPad.top - ebPad.bottom

interface EnergyBar {
  name: string
  color: string
  y: number
  h: number
  rW: number
  lossW: number
  qeW: number
  tW: number
  qePct: string
}

const energyBars = computed((): EnergyBar[] => {
  const barH = Math.min(40, (ebPlotH - 20) / 3)
  const gap = (ebPlotH - barH * 3) / 2
  const channels: { name: string; color: string; wl: number; cfMat: string }[] = [
    { name: t('Red', '빨강'),   color: '#e74c3c', wl: 0.62, cfMat: 'cf_red' },
    { name: t('Green', '초록'), color: '#27ae60', wl: 0.53, cfMat: 'cf_green' },
    { name: t('Blue', '파랑'),  color: '#3498db', wl: 0.45, cfMat: 'cf_blue' },
  ]

  return channels.map((ch, idx) => {
    const stack = buildStack(ch.cfMat)
    const res = tmmCalc(stack, 'air', 'sio2', ch.wl, angle.value, pol.value)
    const R = res.R
    const Tr = res.T
    const siQe = res.layerA[SI_IDX]
    const otherLoss = Math.max(0, 1 - R - Tr - siQe)
    return {
      name: ch.name,
      color: ch.color,
      y: ebPad.top + idx * (barH + gap),
      h: barH,
      rW: R * ebPlotW,
      lossW: otherLoss * ebPlotW,
      qeW: siQe * ebPlotW,
      tW: Tr * ebPlotW,
      qePct: (siQe * 100).toFixed(1),
    }
  })
})

// ── Key Metrics ──────────────────────────────────────────────────────────────
const peakQeR = computed(() => Math.max(0, ...redSpec.value.qe))
const peakQeG = computed(() => Math.max(0, ...greenSpec.value.qe))
const peakQeB = computed(() => Math.max(0, ...blueSpec.value.qe))

const avgQeVis = computed(() => {
  const all = [...redSpec.value.qe, ...greenSpec.value.qe, ...blueSpec.value.qe]
  return all.length > 0 ? all.reduce((a, b) => a + b, 0) / all.length : 0
})

const avgReflectance = computed(() => {
  const rArr = greenSpec.value.R
  return rArr.length > 0 ? rArr.reduce((a, b) => a + b, 0) / rArr.length : 0
})

// ── Visible spectrum gradient stops ──────────────────────────────────────────
function wavelengthToCSS(wl: number): string {
  let r = 0, g = 0, b = 0
  if (wl >= 380 && wl < 440) { r = -(wl - 440) / 60; b = 1 }
  else if (wl >= 440 && wl < 490) { g = (wl - 440) / 50; b = 1 }
  else if (wl >= 490 && wl < 510) { g = 1; b = -(wl - 510) / 20 }
  else if (wl >= 510 && wl < 580) { r = (wl - 510) / 70; g = 1 }
  else if (wl >= 580 && wl < 645) { r = 1; g = -(wl - 645) / 65 }
  else if (wl >= 645 && wl <= 780) { r = 1 }
  let f = 1.0
  if (wl >= 380 && wl < 420) f = 0.3 + 0.7 * (wl - 380) / 40
  else if (wl >= 700 && wl <= 780) f = 0.3 + 0.7 * (780 - wl) / 80
  r = Math.round(255 * Math.pow(r * f, 0.8))
  g = Math.round(255 * Math.pow(g * f, 0.8))
  b = Math.round(255 * Math.pow(b * f, 0.8))
  return `rgb(${r},${g},${b})`
}

const spectrumStops = computed(() => {
  const stops: { offset: string; color: string }[] = []
  for (let wl = 380; wl <= 780; wl += 20) {
    stops.push({
      offset: ((wl - 380) / (780 - 380) * 100) + '%',
      color: wavelengthToCSS(wl),
    })
  }
  return stops
})
</script>

<style scoped>
.pixel-playground {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.pixel-playground h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

/* Two-column layout: controls left, results right */
.playground-layout {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 20px;
  margin-bottom: 16px;
}
@media (max-width: 768px) {
  .playground-layout {
    grid-template-columns: 1fr;
  }
}

/* ── Controls panel ── */
.controls-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ctrl-section {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 12px;
}
.section-label {
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
  display: block;
}
.section-header {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  margin-bottom: 8px;
}
.section-header.collapsible {
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  user-select: none;
  margin-bottom: 0;
}
.section-header.collapsible:hover {
  opacity: 0.8;
}
.toggle-icon {
  font-size: 0.7em;
  color: var(--vp-c-text-3);
}
.collapsible-body {
  margin-top: 8px;
}
.preset-select {
  width: 100%;
  padding: 6px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.85em;
}
.slider-group {
  margin-bottom: 6px;
}
.slider-group:last-child {
  margin-bottom: 0;
}
.slider-group label {
  display: block;
  margin-bottom: 2px;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 5px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.pol-btns {
  display: flex;
  gap: 4px;
  margin-top: 4px;
}
.pol-btn {
  flex: 1;
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.78em;
  cursor: pointer;
  transition: all 0.15s;
}
.pol-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}

/* ── Results panel ── */
.results-panel {
  min-width: 0;
}
.tab-row {
  display: flex;
  gap: 4px;
  margin-bottom: 12px;
}
.tab-btn {
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  cursor: pointer;
  font-size: 13px;
  color: var(--vp-c-text-2);
  transition: all 0.15s;
}
.tab-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
.tab-content {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
}
.result-svg {
  width: 100%;
  max-width: 520px;
  display: block;
  margin: 0 auto;
}

/* ── SVG text classes ── */
.axis-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.legend-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}
.stack-label {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-weight: 600;
}
.stack-thickness {
  font-size: 8px;
  fill: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}
.bar-label {
  font-size: 10px;
  fill: #fff;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}

/* ── Bottom metrics row ── */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 8px;
}
.metric-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}
.metric-label {
  font-size: 0.75em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.metric-value {
  font-weight: 600;
  font-size: 1.05em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-1);
}
</style>
