<template>
  <div class="fdti-simulator">
    <h4>{{ t('FDTI / BDTI CMOS Pixel Optical Simulator', 'FDTI / BDTI CMOS 픽셀 광학 시뮬레이터') }}</h4>
    <p class="component-description">
      {{ t(
        'Build a virtual deep trench isolation pixel and inspect the approximate optical field, collected signal, and crosstalk trend.',
        '가상의 Deep Trench Isolation 픽셀을 만들고 근사 광학장, 수집 신호, 크로스토크 추세를 확인합니다.'
      ) }}
    </p>

    <div class="fdti-layout">
      <div class="controls-panel">
        <div class="ctrl-section">
          <label class="section-label">{{ t('Preset', '프리셋') }}</label>
          <select v-model="preset" class="preset-select" @change="applyPreset">
            <option value="balanced">{{ t('Balanced FDTI', '균형형 FDTI') }}</option>
            <option value="deep">{{ t('Full-depth FDTI', '관통형 FDTI') }}</option>
            <option value="shallow">{{ t('Shallow BDTI', '얕은 BDTI') }}</option>
            <option value="oblique">{{ t('BDTI oblique CRA test', 'BDTI 경사 CRA 테스트') }}</option>
            <option value="custom">{{ t('Custom', '사용자 정의') }}</option>
          </select>
        </div>

        <div class="ctrl-section">
          <div class="section-header">{{ t('Pixel Geometry', '픽셀 구조') }}</div>
          <div class="slider-group">
            <label>{{ t('Pixel pitch', '픽셀 피치') }} <strong>{{ pitch.toFixed(2) }} um</strong></label>
            <input v-model.number="pitch" class="ctrl-range" type="range" min="0.7" max="1.6" step="0.05" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Silicon thickness', '실리콘 두께') }} <strong>{{ siliconThickness.toFixed(1) }} um</strong></label>
            <input v-model.number="siliconThickness" class="ctrl-range" type="range" min="2.0" max="5.0" step="0.1" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Microlens height', '마이크로렌즈 높이') }} <strong>{{ lensHeight.toFixed(2) }} um</strong></label>
            <input v-model.number="lensHeight" class="ctrl-range" type="range" min="0.25" max="0.85" step="0.02" @input="markCustom" />
          </div>
        </div>

        <div class="ctrl-section">
          <div class="section-header">{{ t('Isolation', '절연 구조') }}</div>
          <div class="slider-group">
            <label>{{ t('Trench type', '트렌치 방식') }}</label>
            <div class="segmented-row">
              <button :class="['seg-btn', { active: isolationMode === 'fdti' }]" @click="setIsolationMode('fdti')">FDTI</button>
              <button :class="['seg-btn', { active: isolationMode === 'bdti' }]" @click="setIsolationMode('bdti')">BDTI</button>
            </div>
          </div>
          <div class="slider-group">
            <label>
              {{ t('Trench depth', '트렌치 깊이') }}
              <strong v-if="isolationMode === 'fdti'">{{ t('full silicon', '실리콘 관통') }} {{ siliconThickness.toFixed(2) }} um</strong>
              <strong v-else>{{ fdtiDepth.toFixed(2) }} um</strong>
            </label>
            <input
              v-if="isolationMode === 'bdti'"
              v-model.number="fdtiDepth"
              class="ctrl-range"
              type="range"
              min="0"
              :max="siliconThickness * 0.95"
              step="0.05"
              @input="markCustom"
            />
            <div v-else class="locked-depth">{{ t('FDTI is drawn as a full-depth trench through silicon.', 'FDTI는 실리콘 전체를 관통하는 트렌치로 표시됩니다.') }}</div>
          </div>
          <div class="slider-group">
            <label>{{ t('Trench width', '트렌치 폭') }} <strong>{{ trenchWidth.toFixed(2) }} um</strong></label>
            <input v-model.number="trenchWidth" class="ctrl-range" type="range" min="0.04" max="0.24" step="0.01" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Fill / liner', '충전 / 라이너') }}</label>
            <div class="segmented-row">
              <button :class="['seg-btn', { active: liner === 'oxide' }]" @click="setLiner('oxide')">SiO2</button>
              <button :class="['seg-btn', { active: liner === 'nitride' }]" @click="setLiner('nitride')">SiN</button>
              <button :class="['seg-btn', { active: liner === 'metal' }]" @click="setLiner('metal')">Metal</button>
            </div>
          </div>
        </div>

        <div class="ctrl-section">
          <div class="section-header">{{ t('Illumination', '조명 조건') }}</div>
          <div class="slider-group">
            <label>
              {{ t('Wavelength', '파장') }} <strong>{{ wavelengthNm }} nm</strong>
              <span class="color-dot" :style="{ backgroundColor: wavelengthToCSS(wavelengthNm) }"></span>
            </label>
            <input v-model.number="wavelengthNm" class="ctrl-range" type="range" min="420" max="700" step="10" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Chief ray angle', '주광선 각도') }} <strong>{{ craDeg }} deg</strong></label>
            <input v-model.number="craDeg" class="ctrl-range" type="range" min="-24" max="24" step="1" @input="markCustom" />
          </div>
          <div class="slider-group">
            <label>{{ t('Lens offset', '렌즈 오프셋') }} <strong>{{ lensOffset.toFixed(2) }} um</strong></label>
            <input v-model.number="lensOffset" class="ctrl-range" type="range" min="-0.25" max="0.25" step="0.01" @input="markCustom" />
          </div>
        </div>
      </div>

      <div class="visual-panel">
        <div class="view-grid">
          <div class="panel-block">
            <div class="panel-title">{{ t('Virtual DTI structure', '가상 DTI 구조') }}</div>
            <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="structure-svg">
              <defs>
                <linearGradient id="fdtiLensGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#f7c7e8" stop-opacity="0.75" />
                  <stop offset="100%" stop-color="#a875d6" stop-opacity="0.45" />
                </linearGradient>
                <marker id="fdtiArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 Z" fill="#f59e0b" opacity="0.8" />
                </marker>
              </defs>

              <rect x="0" y="0" :width="svgW" :height="svgH" fill="var(--vp-c-bg)" />

              <template v-for="layer in layerRects" :key="layer.key">
                <rect :x="layer.x" :y="layer.y" :width="layer.w" :height="layer.h" :fill="layer.color" :opacity="layer.opacity" />
                <text :x="layer.x + 8" :y="layer.y + Math.min(layer.h / 2 + 4, layer.h - 3)" class="svg-label" :fill="layer.textColor">
                  {{ layer.label }}
                </text>
              </template>

              <path v-for="dome in lensDomes" :key="dome" :d="dome" fill="url(#fdtiLensGradient)" stroke="#b26bd2" stroke-width="1.2" />

              <rect
                v-for="trench in trenchRects"
                :key="trench.key"
                :x="trench.x"
                :y="trench.y"
                :width="trench.w"
                :height="trench.h"
                :fill="linerStyle.fill"
                :stroke="linerStyle.stroke"
                stroke-width="1"
                opacity="0.92"
                rx="1"
              />

              <rect
                v-for="pd in photodiodeRects"
                :key="pd.key"
                :x="pd.x"
                :y="pd.y"
                :width="pd.w"
                :height="pd.h"
                fill="none"
                stroke="#facc15"
                stroke-width="1.4"
                stroke-dasharray="4 3"
                rx="3"
              />

              <text v-for="pd in photodiodeRects" :key="pd.key + '-label'" :x="pd.x + pd.w / 2" :y="pd.y + pd.h / 2 + 4" text-anchor="middle" class="tiny-label" fill="#facc15">
                {{ pd.label }}
              </text>

              <line
                v-for="boundary in pixelBoundaryLines"
                :key="'boundary-' + boundary"
                :x1="boundary"
                :y1="structureY(siliconTop)"
                :x2="boundary"
                :y2="structureY(totalHeight)"
                stroke="var(--vp-c-text-3)"
                stroke-width="0.7"
                stroke-dasharray="4 4"
                opacity="0.8"
              />

              <path
                v-for="ray in structureRays"
                :key="ray.key"
                :d="ray.path"
                :stroke="ray.color"
                fill="none"
                stroke-width="1.4"
                stroke-linecap="round"
                opacity="0.72"
                marker-end="url(#fdtiArrow)"
              />

              <text :x="structureX(0)" :y="svgH - 8" text-anchor="middle" class="axis-label">{{ t('center pixel', '중앙 픽셀') }}</text>
              <text :x="svgW - 12" :y="18" text-anchor="end" class="axis-label">{{ linerStyle.label }}</text>
            </svg>
          </div>

          <div class="panel-block">
            <div class="panel-title">{{ t('Approximate optical field', '근사 광학장') }}</div>
            <div class="field-wrapper">
              <canvas ref="fieldCanvas" class="field-canvas"></canvas>
            </div>
          </div>
        </div>

        <div class="metrics-row">
          <div class="metric-card" style="border-top-color:#22c55e">
            <div class="metric-label">{{ t('Central signal', '중앙 신호') }}</div>
            <div class="metric-value">{{ currentMetrics.centerPct.toFixed(1) }}%</div>
          </div>
          <div class="metric-card" style="border-top-color:#ef4444">
            <div class="metric-label">{{ t('Neighbor crosstalk', '인접 픽셀 크로스토크') }}</div>
            <div class="metric-value">{{ currentMetrics.crosstalkPct.toFixed(1) }}%</div>
          </div>
          <div class="metric-card" style="border-top-color:#0ea5e9">
            <div class="metric-label">{{ t('DTI suppression', 'DTI 억제량') }}</div>
            <div class="metric-value">{{ dtiGainDb.toFixed(1) }} dB</div>
          </div>
          <div class="metric-card" style="border-top-color:#a855f7">
            <div class="metric-label">{{ t('Optical loss', '광학 손실') }}</div>
            <div class="metric-value">{{ currentMetrics.lossPct.toFixed(1) }}%</div>
          </div>
        </div>

        <div class="qa-panel">
          <div class="panel-title">{{ t('Visual QA scenarios', '시각적 QA 시나리오') }}</div>
          <div class="qa-grid">
            <div v-for="item in qaScenarios" :key="item.key" class="qa-card">
              <div class="qa-header">
                <span>{{ item.label }}</span>
                <span :class="['qa-badge', item.pass ? 'pass' : 'warn']">{{ item.pass ? 'PASS' : 'CHECK' }}</span>
              </div>
              <svg viewBox="0 0 180 54" class="qa-mini">
                <rect x="0" y="0" width="180" height="54" rx="6" fill="var(--vp-c-bg)" />
                <rect x="16" y="10" :width="item.centerWidth" height="12" fill="#22c55e" rx="2" />
                <rect x="16" y="32" :width="item.crossWidth" height="12" fill="#ef4444" rx="2" />
                <line x1="108" y1="4" x2="108" y2="50" stroke="var(--vp-c-divider)" stroke-dasharray="3 3" />
                <text x="12" y="20" text-anchor="end" class="qa-axis">S</text>
                <text x="12" y="42" text-anchor="end" class="qa-axis">XT</text>
                <text x="172" y="20" text-anchor="end" class="qa-axis">{{ item.centerText }}</text>
                <text x="172" y="42" text-anchor="end" class="qa-axis">{{ item.crossText }}</text>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>

    <p class="model-note">
      {{ t(
        'Model note: this browser view is a deterministic paraxial/ray approximation for design review, not a replacement for RCWA or FDTD sign-off.',
        '모델 참고: 이 브라우저 뷰는 설계 검토용 결정론적 근축/광선 근사이며 RCWA 또는 FDTD 사인오프를 대체하지 않습니다.'
      ) }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watchEffect } from 'vue'
import { useLocale } from '../composables/useLocale'

type IsolationMode = 'fdti' | 'bdti'
type LinerKey = 'oxide' | 'nitride' | 'metal'

interface SimParams {
  isolationMode: IsolationMode
  pitch: number
  siliconThickness: number
  lensHeight: number
  fdtiDepth: number
  trenchWidth: number
  liner: LinerKey
  wavelengthNm: number
  craDeg: number
  lensOffset: number
}

interface Metrics {
  centerPct: number
  crosstalkPct: number
  lossPct: number
  leftPct: number
  rightPct: number
  rawCenter: number
  rawLeft: number
  rawRight: number
}

const { t } = useLocale()

const preset = ref('balanced')
const pitch = ref(1.0)
const siliconThickness = ref(3.2)
const lensHeight = ref(0.52)
const isolationMode = ref<IsolationMode>('fdti')
const fdtiDepth = ref(2.25)
const trenchWidth = ref(0.12)
const liner = ref<LinerKey>('oxide')
const wavelengthNm = ref(550)
const craDeg = ref(7)
const lensOffset = ref(0.04)

const cfThickness = 0.55
const frontOxideThickness = 0.28

const presets: Record<string, SimParams> = {
  balanced: {
    isolationMode: 'fdti',
    pitch: 1.0,
    siliconThickness: 3.2,
    lensHeight: 0.52,
    fdtiDepth: 2.25,
    trenchWidth: 0.12,
    liner: 'oxide',
    wavelengthNm: 550,
    craDeg: 7,
    lensOffset: 0.04,
  },
  deep: {
    isolationMode: 'fdti',
    pitch: 0.9,
    siliconThickness: 3.4,
    lensHeight: 0.58,
    fdtiDepth: 3.05,
    trenchWidth: 0.14,
    liner: 'nitride',
    wavelengthNm: 530,
    craDeg: 10,
    lensOffset: 0.06,
  },
  shallow: {
    isolationMode: 'bdti',
    pitch: 1.0,
    siliconThickness: 3.0,
    lensHeight: 0.46,
    fdtiDepth: 0.85,
    trenchWidth: 0.08,
    liner: 'oxide',
    wavelengthNm: 620,
    craDeg: 14,
    lensOffset: 0.02,
  },
  oblique: {
    isolationMode: 'bdti',
    pitch: 0.85,
    siliconThickness: 2.8,
    lensHeight: 0.5,
    fdtiDepth: 2.1,
    trenchWidth: 0.12,
    liner: 'metal',
    wavelengthNm: 450,
    craDeg: 20,
    lensOffset: 0.12,
  },
}

function applyPreset() {
  const next = presets[preset.value]
  if (!next) return
  isolationMode.value = next.isolationMode
  pitch.value = next.pitch
  siliconThickness.value = next.siliconThickness
  lensHeight.value = next.lensHeight
  fdtiDepth.value = next.fdtiDepth
  trenchWidth.value = next.trenchWidth
  liner.value = next.liner
  wavelengthNm.value = next.wavelengthNm
  craDeg.value = next.craDeg
  lensOffset.value = next.lensOffset
}

function markCustom() {
  preset.value = 'custom'
}

function setLiner(value: LinerKey) {
  liner.value = value
  markCustom()
}

function setIsolationMode(value: IsolationMode) {
  isolationMode.value = value
  markCustom()
}

const params = computed<SimParams>(() => ({
  isolationMode: isolationMode.value,
  pitch: pitch.value,
  siliconThickness: siliconThickness.value,
  lensHeight: lensHeight.value,
  fdtiDepth: isolationMode.value === 'fdti'
    ? siliconThickness.value
    : Math.min(fdtiDepth.value, siliconThickness.value * 0.95),
  trenchWidth: trenchWidth.value,
  liner: liner.value,
  wavelengthNm: wavelengthNm.value,
  craDeg: craDeg.value,
  lensOffset: lensOffset.value,
}))

const linerDefs: Record<LinerKey, { label: string; koLabel: string; fill: string; stroke: string; barrier: number; shadow: number }> = {
  oxide: {
    label: 'SiO2 filled trench',
    koLabel: 'SiO2 충전 트렌치',
    fill: '#dbeafe',
    stroke: '#60a5fa',
    barrier: 1.0,
    shadow: 0.1,
  },
  nitride: {
    label: 'SiN liner trench',
    koLabel: 'SiN 라이너 트렌치',
    fill: '#ccfbf1',
    stroke: '#14b8a6',
    barrier: 1.22,
    shadow: 0.16,
  },
  metal: {
    label: 'Reflective metal liner',
    koLabel: '반사성 메탈 라이너',
    fill: '#cbd5e1',
    stroke: '#64748b',
    barrier: 1.65,
    shadow: 0.32,
  },
}

const linerStyle = computed(() => {
  const def = linerDefs[params.value.liner]
  return { ...def, label: t(def.label, def.koLabel) }
})

function siliconTopFor(p: SimParams): number {
  return p.lensHeight + cfThickness + frontOxideThickness
}

function totalHeightFor(p: SimParams): number {
  return siliconTopFor(p) + p.siliconThickness
}

const siliconTop = computed(() => siliconTopFor(params.value))
const totalHeight = computed(() => totalHeightFor(params.value))

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function lerp(a: number, b: number, tValue: number): number {
  return a + (b - a) * clamp(tValue, 0, 1)
}

function smootherstep(tValue: number): number {
  const x = clamp(tValue, 0, 1)
  return x * x * x * (x * (x * 6 - 15) + 10)
}

function effectiveTrenchDepthFor(p: SimParams): number {
  return p.isolationMode === 'fdti' ? p.siliconThickness : p.fdtiDepth
}

function trenchStartFor(p: SimParams): number {
  const depth = effectiveTrenchDepthFor(p)
  if (depth <= 0) return siliconTopFor(p)
  return siliconTopFor(p)
}

function trenchEndFor(p: SimParams): number {
  const depth = effectiveTrenchDepthFor(p)
  return p.isolationMode === 'fdti'
    ? totalHeightFor(p)
    : Math.min(totalHeightFor(p), siliconTopFor(p) + depth)
}

function pdTopFor(p: SimParams): number {
  return siliconTopFor(p) + p.siliconThickness * 0.28
}

function pdBottomFor(p: SimParams): number {
  return siliconTopFor(p) + p.siliconThickness * 0.9
}

function pixelIndexFor(x: number, p: SimParams): number {
  return Math.floor((x + p.pitch / 2) / p.pitch)
}

function crossesTrench(x1: number, x2: number, boundary: number): boolean {
  return Math.min(x1, x2) < boundary && Math.max(x1, x2) > boundary
}

function barrierTransmission(x: number, z: number, beamCenter: number, p: SimParams): number {
  const siTop = siliconTopFor(p)
  const depth = effectiveTrenchDepthFor(p)
  if (z < siTop || depth <= 0) return 1

  let transmission = 1
  const boundaries = [-p.pitch / 2, p.pitch / 2]
  const def = linerDefs[p.liner]
  const trenchStart = trenchStartFor(p)
  const trenchEnd = trenchEndFor(p)

  for (const boundary of boundaries) {
    if (!crossesTrench(x, beamCenter, boundary)) continue

    const widthFactor = p.trenchWidth / 0.12
    const depthFactor = depth / p.siliconThickness
    const wavelengthFactor = 550 / p.wavelengthNm
    const strength = def.barrier * widthFactor * wavelengthFactor

    if (z >= trenchStart && z <= trenchEnd) {
      transmission *= Math.exp(-2.45 * strength)
    } else if (p.isolationMode === 'bdti') {
      transmission *= Math.exp(-0.55 * strength * depthFactor)
    } else {
      transmission *= Math.exp(-1.25 * strength * depthFactor)
    }
  }

  return clamp(transmission, 0.015, 1)
}

function insideTrench(x: number, z: number, p: SimParams): boolean {
  if (z < trenchStartFor(p) || z > trenchEndFor(p)) return false
  const half = p.trenchWidth / 2
  return Math.abs(x - p.pitch / 2) < half || Math.abs(x + p.pitch / 2) < half
}

function opticalFieldAt(x: number, z: number, p: SimParams): number {
  const siTop = siliconTopFor(p)
  const total = totalHeightFor(p)
  const focusZ = siTop + p.siliconThickness * 0.5
  const craRad = (p.craDeg * Math.PI) / 180
  const travel = Math.max(0, z - p.lensHeight * 0.35)
  const beamCenter = p.lensOffset + Math.tan(craRad) * travel * 0.22
  const topSigma = p.pitch * 0.42
  const focusSigma = p.pitch * (0.105 + (p.wavelengthNm - 420) * 0.00012 + Math.abs(p.craDeg) * 0.001)
  const bottomSigma = p.pitch * 0.33
  const sigma = z <= focusZ
    ? lerp(topSigma, focusSigma, smootherstep((z - p.lensHeight * 0.25) / (focusZ - p.lensHeight * 0.25)))
    : lerp(focusSigma, bottomSigma, smootherstep((z - focusZ) / Math.max(0.1, total - focusZ)))

  const gaussian = Math.exp(-((x - beamCenter) ** 2) / (2 * sigma * sigma))
  const zInSi = Math.max(0, z - siTop)
  const absorption = z < siTop ? 0.88 : Math.exp(-0.16 * zInSi) * (1.06 + 0.22 * Math.exp(-((z - focusZ) ** 2) / 0.55))
  const fringe = 1 + 0.08 * Math.sin((z * 2 * Math.PI * 3.6) / (p.wavelengthNm / 1000) + x * 8.5)
  let value = gaussian * absorption * fringe

  if (z >= siTop) {
    value *= barrierTransmission(x, z, beamCenter, p)
  }

  if (insideTrench(x, z, p)) {
    value *= p.liner === 'metal' ? 0.08 : 0.18
  }

  return clamp(value, 0, 1.35)
}

function integratePixel(pixelIndex: number, p: SimParams): number {
  const center = pixelIndex * p.pitch
  const halfActive = p.pitch * 0.36
  const xMin = center - halfActive
  const xMax = center + halfActive
  const zMin = pdTopFor(p)
  const zMax = pdBottomFor(p)
  const nx = 30
  const nz = 34
  let sum = 0

  for (let iz = 0; iz < nz; iz++) {
    const z = zMin + ((iz + 0.5) / nz) * (zMax - zMin)
    const depthWeight = 1.1 - 0.2 * ((z - zMin) / Math.max(0.01, zMax - zMin))
    for (let ix = 0; ix < nx; ix++) {
      const x = xMin + ((ix + 0.5) / nx) * (xMax - xMin)
      sum += opticalFieldAt(x, z, p) * depthWeight
    }
  }

  return sum / (nx * nz)
}

function simulateCoreMetrics(p: SimParams): Metrics {
  const left = integratePixel(-1, p)
  const center = integratePixel(0, p)
  const right = integratePixel(1, p)
  const collected = left + center + right
  const def = linerDefs[p.liner]
  const trenchShadow = (effectiveTrenchDepthFor(p) / p.siliconThickness) * (p.trenchWidth / p.pitch) * def.shadow
  const surfaceReflection = 0.11 + Math.abs(p.craDeg) * 0.0015 + (p.wavelengthNm < 470 ? 0.025 : 0)
  const opticalLoss = collected * (surfaceReflection + trenchShadow + 0.08)
  const denominator = Math.max(1e-6, collected + opticalLoss)

  return {
    centerPct: (center / denominator) * 100,
    crosstalkPct: ((left + right) / Math.max(1e-6, collected)) * 100,
    lossPct: (opticalLoss / denominator) * 100,
    leftPct: (left / Math.max(1e-6, collected)) * 100,
    rightPct: (right / Math.max(1e-6, collected)) * 100,
    rawCenter: center,
    rawLeft: left,
    rawRight: right,
  }
}

const currentMetrics = computed(() => simulateCoreMetrics(params.value))

const noFdtiMetrics = computed(() => simulateCoreMetrics({
  ...params.value,
  isolationMode: 'bdti',
  fdtiDepth: 0,
  trenchWidth: 0.04,
}))

const dtiGainDb = computed(() => {
  return 10 * Math.log10((noFdtiMetrics.value.crosstalkPct + 0.2) / (currentMetrics.value.crosstalkPct + 0.2))
})

const svgW = 620
const svgH = 410
const sPad = { left: 38, right: 24, top: 22, bottom: 32 }

function structureX(x: number): number {
  const p = params.value
  const xMin = -1.5 * p.pitch
  const xMax = 1.5 * p.pitch
  return sPad.left + ((x - xMin) / (xMax - xMin)) * (svgW - sPad.left - sPad.right)
}

function structureY(z: number): number {
  return sPad.top + (z / totalHeight.value) * (svgH - sPad.top - sPad.bottom)
}

const layerRects = computed(() => {
  const p = params.value
  const x = structureX(-1.5 * p.pitch)
  const w = structureX(1.5 * p.pitch) - x
  const siTop = siliconTopFor(p)

  return [
    {
      key: 'cf',
      label: t('Color filter', '컬러 필터'),
      x,
      y: structureY(p.lensHeight),
      w,
      h: structureY(p.lensHeight + cfThickness) - structureY(p.lensHeight),
      color: '#22c55e',
      opacity: 0.24,
      textColor: '#15803d',
    },
    {
      key: 'oxide',
      label: t('Front oxide / gate stack', '전면 산화막 / 게이트 스택'),
      x,
      y: structureY(p.lensHeight + cfThickness),
      w,
      h: structureY(siTop) - structureY(p.lensHeight + cfThickness),
      color: '#93c5fd',
      opacity: 0.24,
      textColor: '#2563eb',
    },
    {
      key: 'si',
      label: 'Silicon',
      x,
      y: structureY(siTop),
      w,
      h: structureY(totalHeightFor(p)) - structureY(siTop),
      color: '#64748b',
      opacity: 0.5,
      textColor: '#f8fafc',
    },
  ]
})

const lensDomes = computed(() => {
  const p = params.value
  const paths: string[] = []
  for (let idx = -1; idx <= 1; idx++) {
    const cx = idx * p.pitch + p.lensOffset
    const half = p.pitch * 0.45
    const baseZ = p.lensHeight
    let d = `M ${structureX(cx - half).toFixed(1)} ${structureY(baseZ).toFixed(1)}`
    for (let i = 0; i <= 32; i++) {
      const u = -1 + (2 * i) / 32
      const x = cx + u * half
      const dome = p.lensHeight * (1 - u * u) * 0.88
      const y = structureY(baseZ - dome)
      d += ` L ${structureX(x).toFixed(1)} ${y.toFixed(1)}`
    }
    d += ` L ${structureX(cx + half).toFixed(1)} ${structureY(baseZ).toFixed(1)} Z`
    paths.push(d)
  }
  return paths
})

const trenchRects = computed(() => {
  const p = params.value
  const y = structureY(trenchStartFor(p))
  const h = structureY(trenchEndFor(p)) - y
  return [-p.pitch / 2, p.pitch / 2].map((boundary, index) => ({
    key: `trench-${index}`,
    x: structureX(boundary - p.trenchWidth / 2),
    y,
    w: structureX(boundary + p.trenchWidth / 2) - structureX(boundary - p.trenchWidth / 2),
    h,
  }))
})

const photodiodeRects = computed(() => {
  const p = params.value
  const y = structureY(pdTopFor(p))
  const h = structureY(pdBottomFor(p)) - y
  return [-1, 0, 1].map((idx) => {
    const center = idx * p.pitch
    const half = p.pitch * 0.34
    return {
      key: `pd-${idx}`,
      label: idx === 0 ? 'PD' : t('NBR', '인접'),
      x: structureX(center - half),
      y,
      w: structureX(center + half) - structureX(center - half),
      h,
    }
  })
})

const pixelBoundaryLines = computed(() => [
  structureX(-params.value.pitch / 2),
  structureX(params.value.pitch / 2),
])

const structureRays = computed(() => {
  const p = params.value
  const rays: Array<{ key: string; path: string; color: string }> = []
  const top = 0.08
  const focusZ = siliconTopFor(p) + p.siliconThickness * 0.5
  const craRad = (p.craDeg * Math.PI) / 180
  for (let i = 0; i < 9; i++) {
    const u = -0.42 + (0.84 * i) / 8
    const startX = u * p.pitch
    const midZ = p.lensHeight * 0.95
    const midX = startX + p.lensOffset * 0.35 + Math.tan(craRad) * midZ * 0.12
    const endX = p.lensOffset + Math.tan(craRad) * focusZ * 0.22 + u * p.pitch * 0.18
    const targetIdx = pixelIndexFor(endX, p)
    rays.push({
      key: `ray-${i}`,
      color: targetIdx === 0 ? '#22c55e' : '#ef4444',
      path: `M ${structureX(startX).toFixed(1)} ${structureY(top).toFixed(1)} Q ${structureX(midX).toFixed(1)} ${structureY(midZ).toFixed(1)} ${structureX(endX).toFixed(1)} ${structureY(focusZ).toFixed(1)}`,
    })
  }
  return rays
})

const fieldCanvas = ref<HTMLCanvasElement | null>(null)

function fieldColor(value: number): [number, number, number] {
  const tValue = clamp(value, 0, 1)
  if (tValue < 0.25) {
    const s = tValue / 0.25
    return [Math.round(18 + s * 10), Math.round(24 + s * 52), Math.round(44 + s * 120)]
  }
  if (tValue < 0.55) {
    const s = (tValue - 0.25) / 0.3
    return [Math.round(28 + s * 8), Math.round(76 + s * 140), Math.round(164 + s * 36)]
  }
  if (tValue < 0.8) {
    const s = (tValue - 0.55) / 0.25
    return [Math.round(36 + s * 214), Math.round(216 + s * 18), Math.round(200 - s * 120)]
  }
  const s = (tValue - 0.8) / 0.2
  return [250, Math.round(234 + s * 21), Math.round(80 + s * 155)]
}

function drawFieldCanvas() {
  const canvas = fieldCanvas.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const cssW = 440
  const cssH = 320
  const dpr = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1
  canvas.style.width = `${cssW}px`
  canvas.style.height = `${cssH}px`
  canvas.width = Math.round(cssW * dpr)
  canvas.height = Math.round(cssH * dpr)
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const p = params.value
  const xMin = -1.5 * p.pitch
  const xMax = 1.5 * p.pitch
  const zMax = totalHeightFor(p)
  const img = ctx.createImageData(cssW, cssH)

  for (let py = 0; py < cssH; py++) {
    const z = (py / (cssH - 1)) * zMax
    for (let px = 0; px < cssW; px++) {
      const x = xMin + (px / (cssW - 1)) * (xMax - xMin)
      const intensity = opticalFieldAt(x, z, p)
      const logI = Math.log10(1 + 9 * intensity)
      const [r, g, b] = fieldColor(logI)
      const idx = (py * cssW + px) * 4
      img.data[idx] = r
      img.data[idx + 1] = g
      img.data[idx + 2] = b
      img.data[idx + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)

  function cx(x: number): number {
    return ((x - xMin) / (xMax - xMin)) * cssW
  }
  function cy(z: number): number {
    return (z / zMax) * cssH
  }

  ctx.save()
  ctx.lineWidth = 1
  ctx.strokeStyle = 'rgba(255,255,255,0.38)'
  ctx.setLineDash([4, 4])
  for (const boundary of [-p.pitch / 2, p.pitch / 2]) {
    ctx.beginPath()
    ctx.moveTo(cx(boundary), cy(siliconTopFor(p)))
    ctx.lineTo(cx(boundary), cssH)
    ctx.stroke()
  }
  ctx.setLineDash([])

  ctx.fillStyle = p.liner === 'metal' ? 'rgba(148,163,184,0.65)' : 'rgba(191,219,254,0.55)'
  for (const boundary of [-p.pitch / 2, p.pitch / 2]) {
    ctx.fillRect(cx(boundary - p.trenchWidth / 2), cy(trenchStartFor(p)), cx(boundary + p.trenchWidth / 2) - cx(boundary - p.trenchWidth / 2), cy(trenchEndFor(p)) - cy(trenchStartFor(p)))
  }

  ctx.strokeStyle = 'rgba(250,204,21,0.9)'
  ctx.lineWidth = 1.2
  ctx.setLineDash([4, 3])
  for (const idx of [-1, 0, 1]) {
    const center = idx * p.pitch
    const half = p.pitch * 0.34
    ctx.strokeRect(cx(center - half), cy(pdTopFor(p)), cx(center + half) - cx(center - half), cy(pdBottomFor(p)) - cy(pdTopFor(p)))
  }
  ctx.setLineDash([])

  ctx.fillStyle = 'rgba(255,255,255,0.84)'
  ctx.font = '11px Inter, system-ui, sans-serif'
  ctx.fillText('I(x,z)', 12, 18)
  ctx.fillText(`${currentMetrics.value.crosstalkPct.toFixed(1)}% XT`, cssW - 70, 18)
  ctx.restore()
}

onMounted(() => {
  drawFieldCanvas()
})

watchEffect(() => {
  params.value
  currentMetrics.value
  drawFieldCanvas()
})

const qaScenarios = computed(() => {
  const p = params.value
  const scenarios = [
    {
      key: 'current',
      label: t('Current design', '현재 설계'),
      metrics: currentMetrics.value,
      targetCenter: 48,
      targetCross: 14,
    },
    {
      key: 'no-fdti',
      label: t('No DTI control', 'DTI 없음 대조군'),
      metrics: noFdtiMetrics.value,
      targetCenter: 42,
      targetCross: 35,
    },
    {
      key: 'shallow',
      label: t('Shallow BDTI', '얕은 BDTI'),
      metrics: simulateCoreMetrics({ ...p, isolationMode: 'bdti', fdtiDepth: Math.min(0.75, p.siliconThickness * 0.4), trenchWidth: Math.max(0.06, p.trenchWidth * 0.7) }),
      targetCenter: 42,
      targetCross: 24,
    },
    {
      key: 'deep-cra',
      label: t('Full FDTI + oblique', '관통 FDTI + 경사 입사'),
      metrics: simulateCoreMetrics({ ...p, isolationMode: 'fdti', fdtiDepth: p.siliconThickness, craDeg: clamp(p.craDeg + 8, -24, 24), lensOffset: clamp(p.lensOffset + 0.05, -0.25, 0.25) }),
      targetCenter: 40,
      targetCross: 18,
    },
  ]

  return scenarios.map((scenario) => {
    const centerWidth = clamp((scenario.metrics.centerPct / 70) * 138, 4, 138)
    const crossWidth = clamp((scenario.metrics.crosstalkPct / 40) * 138, 4, 138)
    const pass = scenario.metrics.centerPct >= scenario.targetCenter && scenario.metrics.crosstalkPct <= scenario.targetCross
    return {
      ...scenario,
      pass,
      centerWidth,
      crossWidth,
      centerText: `${scenario.metrics.centerPct.toFixed(0)}%`,
      crossText: `${scenario.metrics.crosstalkPct.toFixed(0)}%`,
    }
  })
})

function wavelengthToCSS(wl: number): string {
  let r = 0
  let g = 0
  let b = 0
  if (wl >= 380 && wl < 440) { r = -(wl - 440) / 60; b = 1 }
  else if (wl >= 440 && wl < 490) { g = (wl - 440) / 50; b = 1 }
  else if (wl >= 490 && wl < 510) { g = 1; b = -(wl - 510) / 20 }
  else if (wl >= 510 && wl < 580) { r = (wl - 510) / 70; g = 1 }
  else if (wl >= 580 && wl < 645) { r = 1; g = -(wl - 645) / 65 }
  else if (wl >= 645 && wl <= 780) { r = 1 }
  let factor = 1
  if (wl >= 380 && wl < 420) factor = 0.3 + 0.7 * (wl - 380) / 40
  else if (wl >= 700 && wl <= 780) factor = 0.3 + 0.7 * (780 - wl) / 80
  return `rgb(${Math.round(255 * Math.pow(r * factor, 0.8))}, ${Math.round(255 * Math.pow(g * factor, 0.8))}, ${Math.round(255 * Math.pow(b * factor, 0.8))})`
}
</script>

<style scoped>
.fdti-simulator {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.fdti-simulator h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description,
.model-note {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.model-note {
  margin: 14px 0 0;
  font-size: 0.82em;
}

.fdti-layout {
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 18px;
}

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
  display: block;
  margin-bottom: 4px;
  font-size: 0.82em;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

.section-header {
  margin-bottom: 8px;
  color: var(--vp-c-brand-1);
  font-size: 0.85em;
  font-weight: 700;
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
  margin-bottom: 8px;
}

.slider-group:last-child {
  margin-bottom: 0;
}

.slider-group label {
  display: block;
  margin-bottom: 3px;
  color: var(--vp-c-text-2);
  font-size: 0.8em;
}

.ctrl-range {
  width: 100%;
  height: 5px;
  border-radius: 3px;
  outline: none;
  background: var(--vp-c-divider);
  appearance: none;
}

.ctrl-range::-webkit-slider-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.22);
  cursor: pointer;
  appearance: none;
}

.ctrl-range::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border: 0;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.22);
  cursor: pointer;
}

.segmented-row {
  display: flex;
  gap: 4px;
}

.seg-btn {
  flex: 1;
  padding: 5px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 5px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.78em;
  cursor: pointer;
}

.seg-btn.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
  color: #fff;
}

.locked-depth {
  padding: 6px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 5px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 0.76em;
  line-height: 1.35;
}

.color-dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin-left: 4px;
  border: 1px solid #888;
  border-radius: 50%;
  vertical-align: middle;
}

.visual-panel {
  min-width: 0;
}

.view-grid {
  display: grid;
  grid-template-columns: minmax(280px, 1.05fr) minmax(280px, 0.95fr);
  gap: 12px;
}

.panel-block,
.qa-panel {
  min-width: 0;
  padding: 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
}

.panel-title {
  margin-bottom: 8px;
  color: var(--vp-c-text-2);
  font-size: 0.83em;
  font-weight: 700;
}

.structure-svg {
  display: block;
  width: 100%;
  max-width: 620px;
  margin: 0 auto;
}

.svg-label {
  font-size: 9px;
  font-weight: 700;
}

.tiny-label,
.axis-label,
.qa-axis {
  font-size: 8px;
  font-weight: 700;
  fill: var(--vp-c-text-2);
}

.field-wrapper {
  display: flex;
  justify-content: center;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: #111827;
}

.field-canvas {
  display: block;
  max-width: 100%;
}

.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(110px, 1fr));
  gap: 8px;
  margin-top: 10px;
}

.metric-card {
  border: 1px solid var(--vp-c-divider);
  border-top: 3px solid var(--vp-c-brand-1);
  border-radius: 8px;
  padding: 9px 10px;
  background: var(--vp-c-bg);
  text-align: center;
}

.metric-label {
  margin-bottom: 3px;
  color: var(--vp-c-text-2);
  font-size: 0.74em;
}

.metric-value {
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 1.02em;
  font-weight: 700;
}

.qa-panel {
  margin-top: 10px;
}

.qa-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 8px;
}

.qa-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px;
  background: var(--vp-c-bg-soft);
}

.qa-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  margin-bottom: 6px;
  color: var(--vp-c-text-1);
  font-size: 0.76em;
  font-weight: 700;
}

.qa-badge {
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 0.68em;
  font-family: var(--vp-font-family-mono);
}

.qa-badge.pass {
  background: rgba(34, 197, 94, 0.14);
  color: #16a34a;
}

.qa-badge.warn {
  background: rgba(245, 158, 11, 0.16);
  color: #d97706;
}

.qa-mini {
  display: block;
  width: 100%;
}

@media (max-width: 980px) {
  .fdti-layout,
  .view-grid {
    grid-template-columns: 1fr;
  }

  .metrics-row,
  .qa-grid {
    grid-template-columns: repeat(2, minmax(120px, 1fr));
  }
}

@media (max-width: 560px) {
  .metrics-row,
  .qa-grid {
    grid-template-columns: 1fr;
  }
}
</style>
