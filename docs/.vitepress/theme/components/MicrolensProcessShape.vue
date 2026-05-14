<template>
  <div class="mlp-container">
    <h4>{{ t('Microlens Process Shape Predictor', '마이크로렌즈 공정 형상 예측기') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how layout gap, reflow budget, and etch-transfer settings can move a CIS microlens toward a final gap, height, and profile.',
        '레이아웃 gap, reflow budget, etch-transfer 조건이 CIS 마이크로렌즈의 최종 gap, 높이, profile에 어떤 방향으로 작용하는지 살펴봅니다.'
      ) }}
    </p>

    <div class="control-section">
      <div class="control-heading">{{ t('Layout and resist', '레이아웃 및 resist') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Pixel pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} um</strong></label>
          <input type="range" min="0.6" max="3.0" step="0.02" v-model.number="pitch" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Mask island width', '마스크 island 폭') }}: <strong>{{ maskWidth.toFixed(2) }} um</strong></label>
          <input type="range" :min="minMaskWidth" :max="maxMaskWidth" step="0.01" v-model.number="maskWidth" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Resist thickness', 'Resist 두께') }}: <strong>{{ resistThickness.toFixed(2) }} um</strong></label>
          <input type="range" min="0.12" max="1.20" step="0.01" v-model.number="resistThickness" class="ctrl-range" />
        </div>
      </div>
      <div class="controls-row">
        <div class="slider-group compact">
          <label>{{ t('Aperture shape', '개구 형상') }}</label>
          <select v-model="apertureShape" class="ctrl-select">
            <option value="circular">{{ t('Circular', '원형') }}</option>
            <option value="rounded-square">{{ t('Rounded square', '라운드 사각') }}</option>
            <option value="square">{{ t('Square-like', '사각형 근사') }}</option>
          </select>
        </div>
      </div>
    </div>

    <div class="control-section">
      <div class="control-heading">{{ t('Thermal reflow', 'Thermal reflow') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Reflow temperature', 'Reflow 온도') }}: <strong>{{ reflowTemp }} C</strong></label>
          <input type="range" min="125" max="220" step="1" v-model.number="reflowTemp" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Reflow time', 'Reflow 시간') }}: <strong>{{ reflowTime }} s</strong></label>
          <input type="range" min="20" max="300" step="5" v-model.number="reflowTime" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Lens index', '렌즈 굴절률') }}: <strong>{{ lensIndex.toFixed(2) }}</strong></label>
          <input type="range" min="1.40" max="1.75" step="0.01" v-model.number="lensIndex" class="ctrl-range" />
        </div>
      </div>
    </div>

    <div class="control-section">
      <div class="control-heading">{{ t('Etch transfer', 'Etch transfer') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Mask thickness', 'Mask 두께') }}: <strong>{{ maskThickness.toFixed(2) }} um</strong></label>
          <input type="range" min="0.10" max="1.00" step="0.01" v-model.number="maskThickness" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Polymerizing gas', 'Polymerizing gas') }}: <strong>{{ polymerGas }}%</strong></label>
          <input type="range" min="0" max="100" step="1" v-model.number="polymerGas" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Etch time', 'Etch 시간') }}: <strong>{{ etchTime }} s</strong></label>
          <input type="range" min="0" max="140" step="1" v-model.number="etchTime" class="ctrl-range" />
        </div>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card">
        <span>{{ t('Initial gap', '초기 gap') }}</span>
        <strong>{{ initialGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('After reflow', 'Reflow 후 gap') }}</span>
        <strong>{{ reflowGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card accent">
        <span>{{ t('Final gap', '최종 gap') }}</span>
        <strong>{{ finalGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card accent">
        <span>{{ t('Final height', '최종 높이') }}</span>
        <strong>{{ finalHeight.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('ROC at vertex', 'Vertex ROC') }}</span>
        <strong>{{ roc.toFixed(2) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('f-number', 'f-number') }}</span>
        <strong>f/{{ fNumber.toFixed(2) }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Fill factor', 'Fill factor') }}</span>
        <strong>{{ fillFactor.toFixed(1) }}%</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Height retention', '높이 보존율') }}</span>
        <strong>{{ heightRetention.toFixed(0) }}%</strong>
      </div>
    </div>

    <div class="status-row">
      <span v-for="flag in processFlags" :key="flag.text" :class="['status-pill', flag.tone]">
        {{ flag.text }}
      </span>
    </div>

    <div class="tab-row">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        :class="['tab-btn', { active: viewMode === tab.key }]"
        :aria-pressed="viewMode === tab.key"
        @click="viewMode = tab.key"
      >
        {{ t(tab.en, tab.ko) }}
      </button>
    </div>

    <div class="plot-panel">
      <svg
        v-if="viewMode === 'section'"
        :viewBox="`0 0 ${sectionW} ${sectionH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Microlens process cross-section', '마이크로렌즈 공정 단면')"
      >
        <rect x="0" y="0" :width="sectionW" :height="sectionH" fill="var(--vp-c-bg)" />
        <line
          v-for="tick in sectionXTicks"
          :key="'sxg-' + tick"
          :x1="sectionXScale(tick)"
          :y1="sectionPad.top"
          :x2="sectionXScale(tick)"
          :y2="sectionH - sectionPad.bottom"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line
          v-for="tick in sectionYTicks"
          :key="'syg-' + tick"
          :x1="sectionPad.left"
          :y1="sectionYScale(tick)"
          :x2="sectionW - sectionPad.right"
          :y2="sectionYScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line :x1="sectionPad.left" :y1="sectionYScale(0)" :x2="sectionW - sectionPad.right" :y2="sectionYScale(0)" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="sectionPad.left" :y1="sectionPad.top" :x2="sectionPad.left" :y2="sectionH - sectionPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <template v-for="center in lensCenters" :key="'rect-' + center">
          <rect
            :x="sectionXScale(center - maskWidth / 2)"
            :y="sectionYScale(resistThickness)"
            :width="Math.max(1, sectionXScale(center + maskWidth / 2) - sectionXScale(center - maskWidth / 2))"
            :height="sectionYScale(0) - sectionYScale(resistThickness)"
            fill="#9b59b6"
            fill-opacity="0.08"
            stroke="#9b59b6"
            stroke-width="1"
            stroke-dasharray="4,3"
          />
        </template>

        <path
          v-for="profile in reflowProfiles"
          :key="'reflow-' + profile"
          :d="profile"
          fill="none"
          stroke="#e67e22"
          stroke-width="1.5"
          stroke-dasharray="5,4"
          opacity="0.9"
        />
        <path
          v-for="profile in finalProfiles"
          :key="'final-' + profile"
          :d="profile"
          fill="#3498db"
          fill-opacity="0.20"
          stroke="#1f78b4"
          stroke-width="2"
        />

        <line
          :x1="sectionXScale(-finalGap / 2)"
          :y1="sectionYScale(-0.035)"
          :x2="sectionXScale(finalGap / 2)"
          :y2="sectionYScale(-0.035)"
          stroke="#c0392b"
          stroke-width="2"
          :stroke-dasharray="finalGap <= 0.002 ? '2,2' : 'none'"
        />
        <text :x="sectionXScale(0)" :y="sectionYScale(-0.08)" text-anchor="middle" class="plot-label" fill="#c0392b">
          {{ t('final gap', '최종 gap') }}
        </text>

        <line :x1="sectionXScale(0)" :y1="sectionYScale(0)" :x2="sectionXScale(0)" :y2="sectionYScale(finalHeight)" stroke="#27ae60" stroke-width="1.4" stroke-dasharray="4,3" />
        <text :x="sectionXScale(0) + 6" :y="sectionYScale(finalHeight / 2)" class="plot-label" fill="#27ae60">
          h={{ finalHeight.toFixed(2) }} um
        </text>

        <line x1="438" y1="22" x2="462" y2="22" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="468" y="26" class="legend-label">{{ t('litho resist island', 'litho resist island') }}</text>
        <line x1="438" y1="40" x2="462" y2="40" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="5,4" />
        <text x="468" y="44" class="legend-label">{{ t('after reflow', 'reflow 후') }}</text>
        <line x1="438" y1="58" x2="462" y2="58" stroke="#1f78b4" stroke-width="2" />
        <text x="468" y="62" class="legend-label">{{ t('after etch transfer', 'etch transfer 후') }}</text>

        <text v-for="tick in sectionXTicks" :key="'sxl-' + tick" :x="sectionXScale(tick)" :y="sectionH - 12" text-anchor="middle" class="axis-label">
          {{ tick.toFixed(1) }}
        </text>
        <text v-for="tick in sectionYTicks" :key="'syl-' + tick" :x="sectionPad.left - 8" :y="sectionYScale(tick) + 3" text-anchor="end" class="axis-label">
          {{ tick.toFixed(1) }}
        </text>
        <text :x="(sectionPad.left + sectionW - sectionPad.right) / 2" :y="sectionH - 2" text-anchor="middle" class="axis-label">x (um)</text>
        <text x="13" :y="(sectionPad.top + sectionH - sectionPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 13, ${(sectionPad.top + sectionH - sectionPad.bottom) / 2})`">z (um)</text>
      </svg>

      <svg
        v-if="viewMode === 'surface'"
        :viewBox="`0 0 ${surfaceW} ${surfaceH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Microlens final surface wireframe', '마이크로렌즈 최종 표면 wireframe')"
      >
        <rect x="0" y="0" :width="surfaceW" :height="surfaceH" fill="var(--vp-c-bg)" />
        <polygon :points="cellBasePolygon" fill="#7f8c8d" opacity="0.08" stroke="var(--vp-c-divider)" stroke-width="1" />
        <path
          v-for="line in surfaceLines"
          :key="line.key"
          :d="line.d"
          fill="none"
          :stroke="line.stroke"
          :stroke-width="line.major ? 1.5 : 0.8"
          :opacity="line.major ? 0.95 : 0.65"
        />
        <path v-for="edge in surfaceFootprintEdges" :key="'edge-' + edge" :d="edge" fill="none" stroke="#34495e" stroke-width="1.2" opacity="0.8" />
        <text x="24" y="28" class="surface-title">{{ t('Final 3D footprint and profile', '최종 3D footprint/profile') }}</text>
        <text x="24" y="48" class="surface-note">
          {{ t('Superellipse footprint; height field uses the predicted final height and profile exponent.', 'Superellipse footprint와 예측 height/profile exponent를 사용합니다.') }}
        </text>
        <g transform="translate(24, 298)">
          <rect x="0" y="0" width="155" height="34" rx="5" fill="var(--vp-c-bg-soft)" stroke="var(--vp-c-divider)" />
          <text x="10" y="14" class="legend-label">{{ t('width', '폭') }} {{ finalWidth.toFixed(3) }} um</text>
          <text x="10" y="28" class="legend-label">{{ t('profile exponent', 'profile exponent') }} {{ profilePower.toFixed(2) }}</text>
        </g>
      </svg>

      <svg
        v-if="viewMode === 'process'"
        :viewBox="`0 0 ${processW} ${processH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Etch response curves', 'Etch response curve')"
      >
        <rect x="0" y="0" :width="processW" :height="processH" fill="var(--vp-c-bg)" />
        <line
          v-for="tick in processXTicks"
          :key="'pxg-' + tick"
          :x1="processXScale(tick)"
          :y1="processPad.top"
          :x2="processXScale(tick)"
          :y2="processH - processPad.bottom"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line
          v-for="tick in processGapTicks"
          :key="'pyg-' + tick"
          :x1="processPad.left"
          :y1="processGapYScale(tick)"
          :x2="processW - processPad.right"
          :y2="processGapYScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line :x1="processPad.left" :y1="processPad.top" :x2="processPad.left" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="processPad.left" :y1="processH - processPad.bottom" :x2="processW - processPad.right" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="processW - processPad.right" :y1="processPad.top" :x2="processW - processPad.right" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <path :d="gapCurvePath" fill="none" stroke="#c0392b" stroke-width="2.4" />
        <path :d="heightCurvePath" fill="none" stroke="#27ae60" stroke-width="2.4" />
        <line :x1="currentProcessPoint.x" :y1="processPad.top" :x2="currentProcessPoint.x" :y2="processH - processPad.bottom" stroke="var(--vp-c-brand-1)" stroke-width="1.3" stroke-dasharray="5,4" />
        <circle :cx="currentProcessPoint.x" :cy="currentProcessPoint.gapY" r="4.5" fill="#c0392b" stroke="#fff" stroke-width="1.2" />
        <circle :cx="currentProcessPoint.x" :cy="currentProcessPoint.heightY" r="4.5" fill="#27ae60" stroke="#fff" stroke-width="1.2" />

        <line x1="88" y1="30" x2="118" y2="30" stroke="#c0392b" stroke-width="2.4" />
        <text x="124" y="34" class="legend-label">{{ t('Final gap (left axis)', '최종 gap (좌축)') }}</text>
        <line x1="88" y1="49" x2="118" y2="49" stroke="#27ae60" stroke-width="2.4" />
        <text x="124" y="53" class="legend-label">{{ t('Height retention (right axis)', '높이 보존율 (우축)') }}</text>

        <text v-for="tick in processXTicks" :key="'pxl-' + tick" :x="processXScale(tick)" :y="processH - 12" text-anchor="middle" class="axis-label">{{ tick }}</text>
        <text v-for="tick in processGapTicks" :key="'pyl-' + tick" :x="processPad.left - 8" :y="processGapYScale(tick) + 3" text-anchor="end" class="axis-label">{{ tick.toFixed(2) }}</text>
        <text v-for="tick in processRetentionTicks" :key="'prl-' + tick" :x="processW - processPad.right + 8" :y="processRetentionYScale(tick) + 3" class="axis-label">{{ tick }}%</text>
        <text :x="(processPad.left + processW - processPad.right) / 2" :y="processH - 2" text-anchor="middle" class="axis-label">{{ t('Etch time (s)', 'Etch 시간 (s)') }}</text>
        <text x="13" :y="(processPad.top + processH - processPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 13, ${(processPad.top + processH - processPad.bottom) / 2})`">{{ t('Gap (um)', 'Gap (um)') }}</text>
      </svg>
    </div>

    <div class="formula-box">
      <strong>{{ t('Model note', '모델 메모') }}:</strong>
      {{ t(
        'This is a calibrated-by-user surrogate, not a foundry recipe. It combines volume-conserving reflow, parabolic/superellipse caps, and DOE-inspired etch trends: more etch time closes gap; polymerization mainly preserves height; mask thickness changes transfer robustness.',
        '이 모델은 foundry recipe가 아니라 사용자가 보정해 쓰는 surrogate입니다. Volume-conserving reflow, parabolic/superellipse cap, DOE식 etch 경향을 결합합니다. Etch time은 gap closure를 키우고, polymerization은 주로 height 보존에, mask thickness는 transfer robustness에 영향을 준다고 둡니다.'
      ) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

type ApertureShape = 'circular' | 'rounded-square' | 'square'
type ViewMode = 'section' | 'surface' | 'process'

const pitch = ref(1.10)
const maskWidth = ref(0.88)
const resistThickness = ref(0.42)
const apertureShape = ref<ApertureShape>('rounded-square')
const reflowTemp = ref(170)
const reflowTime = ref(90)
const lensIndex = ref(1.55)
const maskThickness = ref(0.45)
const polymerGas = ref(55)
const etchTime = ref(55)
const viewMode = ref<ViewMode>('section')

const tabs = [
  { key: 'section' as const, en: 'Cross-section', ko: '단면' },
  { key: 'surface' as const, en: '3D surface', ko: '3D 표면' },
  { key: 'process' as const, en: 'Etch response', ko: 'Etch 응답' },
]

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const minMaskWidth = computed(() => Math.max(0.2, pitch.value * 0.35))
const maxMaskWidth = computed(() => Math.max(minMaskWidth.value + 0.02, pitch.value * 0.98))

watch([minMaskWidth, maxMaskWidth], () => {
  maskWidth.value = clamp(maskWidth.value, minMaskWidth.value, maxMaskWidth.value)
}, { immediate: true })

const boundedMaskWidth = computed(() => clamp(maskWidth.value, minMaskWidth.value, maxMaskWidth.value))
const initialGap = computed(() => Math.max(0, pitch.value - boundedMaskWidth.value))
const thermalDose = computed(() => {
  const tempNorm = clamp((reflowTemp.value - 125) / 80, 0, 1.35)
  const timeNorm = clamp(Math.log1p(reflowTime.value / 30) / Math.log1p(300 / 30), 0, 1.1)
  return clamp(0.62 * tempNorm + 0.38 * timeNorm, 0, 1.25)
})

const shapeExponent = computed(() => {
  if (apertureShape.value === 'circular') return 2
  if (apertureShape.value === 'rounded-square') return 4
  return 8
})

const areaFactor = computed(() => {
  if (apertureShape.value === 'circular') return Math.PI / 4
  if (apertureShape.value === 'rounded-square') return 0.91
  return 0.98
})

const reflowSpread = computed(() => {
  const thicknessTerm = 0.10 * resistThickness.value
  const gapPull = 0.20 * initialGap.value
  return thermalDose.value * (0.018 + thicknessTerm + gapPull)
})

const reflowWidth = computed(() => Math.min(pitch.value * 1.04, boundedMaskWidth.value + 2 * reflowSpread.value))
const reflowGap = computed(() => Math.max(0, pitch.value - reflowWidth.value))
const reflowHeight = computed(() => {
  const r0 = boundedMaskWidth.value / 2
  const r1 = reflowWidth.value / 2
  const volumeRetention = clamp(0.96 - 0.08 * thermalDose.value, 0.84, 0.98)
  return clamp((2 * resistThickness.value * r0 * r0 * volumeRetention) / Math.max(r1 * r1, 0.01), 0.035, 2.0)
})

const polyNorm = computed(() => polymerGas.value / 100)
const maskRobustness = computed(() => clamp(0.72 + maskThickness.value / 0.75, 0.72, 1.70))

function etchTransferAt(timeSeconds: number) {
  const poly = polyNorm.value
  const mask = maskRobustness.value
  const lateralRate = (0.0018 + 0.0012 * poly + 0.00045 * thermalDose.value) * (0.84 + 0.16 * mask)
  const closure = timeSeconds * lateralRate
  const gap = Math.max(0, reflowGap.value - 2 * closure)
  const flattenRate = 0.0030 * (1 - 0.62 * poly) * (1.12 - 0.16 * mask)
  const lossFraction = clamp(timeSeconds * flattenRate, 0, 0.58)
  const transferredHeight = Math.max(0.025, reflowHeight.value * (1 - lossFraction))
  return { gap, closure, lossFraction, height: transferredHeight }
}

const etchState = computed(() => etchTransferAt(etchTime.value))
const finalGap = computed(() => etchState.value.gap)
const finalWidth = computed(() => Math.max(0.05, pitch.value - finalGap.value))
const finalHeight = computed(() => etchState.value.height)
const heightRetention = computed(() => clamp((finalHeight.value / Math.max(reflowHeight.value, 0.001)) * 100, 0, 120))
const profilePower = computed(() => clamp(2.0 + 0.42 * thermalDose.value + 1.2 * etchState.value.lossFraction - 0.22 * polyNorm.value, 1.7, 4.2))
const halfFinalWidth = computed(() => finalWidth.value / 2)
const halfReflowWidth = computed(() => reflowWidth.value / 2)
const roc = computed(() => {
  const a = halfFinalWidth.value
  const h = finalHeight.value
  return (a * a + h * h) / Math.max(2 * h, 0.001)
})
const focalLength = computed(() => roc.value / Math.max(lensIndex.value - 1, 0.05))
const fNumber = computed(() => focalLength.value / Math.max(finalWidth.value, 0.05))
const fillFactor = computed(() => clamp((areaFactor.value * finalWidth.value * finalWidth.value / (pitch.value * pitch.value)) * 100, 0, 100))

const processFlags = computed(() => {
  const flags: { text: string; tone: string }[] = []
  if (finalGap.value <= 0.015) flags.push({ text: t('zero-gap candidate', 'zero-gap 후보'), tone: 'good' })
  else if (finalGap.value <= 0.06) flags.push({ text: t('near zero-space', 'zero-space 근접'), tone: 'good' })
  else flags.push({ text: t('visible lens gap', '렌즈 gap 잔존'), tone: 'warn' })

  if (reflowGap.value <= 0.002 && initialGap.value > 0.01) flags.push({ text: t('merger risk during reflow', 'reflow 중 merger 위험'), tone: 'risk' })
  if (etchState.value.lossFraction > 0.32) flags.push({ text: t('height loss risk', 'height loss 위험'), tone: 'risk' })
  if (fillFactor.value > 92) flags.push({ text: t('high fill factor', '높은 fill factor'), tone: 'good' })
  if (fNumber.value < 0.9 || fNumber.value > 3.8) flags.push({ text: t('check optical focus', 'optical focus 확인 필요'), tone: 'warn' })
  return flags
})

function lensZ1D(x: number, halfWidth: number, height: number, power: number) {
  const u = Math.abs(x) / Math.max(halfWidth, 0.001)
  if (u >= 1) return 0
  return height * Math.pow(1 - Math.pow(u, power), 1.0)
}

function lensZ2D(x: number, y: number, halfWidth: number, height: number, profile: number) {
  const n = shapeExponent.value
  const r = Math.pow(
    Math.pow(Math.abs(x) / Math.max(halfWidth, 0.001), n) +
      Math.pow(Math.abs(y) / Math.max(halfWidth, 0.001), n),
    1 / n,
  )
  if (r >= 1) return 0
  return height * Math.pow(1 - Math.pow(r, profile), 1.0)
}

// Cross-section plot
const sectionW = 640
const sectionH = 330
const sectionPad = { left: 52, right: 22, top: 22, bottom: 42 }
const lensCenters = computed(() => [-pitch.value, 0, pitch.value])
const sectionXMin = computed(() => -1.55 * pitch.value)
const sectionXMax = computed(() => 1.55 * pitch.value)
const sectionYMax = computed(() => Math.max(reflowHeight.value, finalHeight.value, resistThickness.value) * 1.22 + 0.08)
const sectionYMin = computed(() => -0.12)
const sectionXTicks = computed(() => {
  return [-1.5, -1, -0.5, 0, 0.5, 1, 1.5].map(v => v * pitch.value).filter(v => v >= sectionXMin.value && v <= sectionXMax.value)
})
const sectionYTicks = computed(() => {
  const max = sectionYMax.value
  return [0, max * 0.25, max * 0.5, max * 0.75, max].map(v => Number(v.toFixed(2)))
})

function sectionXScale(x: number) {
  const plotW = sectionW - sectionPad.left - sectionPad.right
  return sectionPad.left + ((x - sectionXMin.value) / (sectionXMax.value - sectionXMin.value)) * plotW
}

function sectionYScale(y: number) {
  const plotH = sectionH - sectionPad.top - sectionPad.bottom
  return sectionPad.top + (1 - (y - sectionYMin.value) / (sectionYMax.value - sectionYMin.value)) * plotH
}

function buildSectionProfile(center: number, width: number, height: number, power: number, fill: boolean) {
  const half = width / 2
  const points: string[] = []
  for (let i = 0; i <= 72; i += 1) {
    const xLocal = -half + (2 * half * i) / 72
    const z = lensZ1D(xLocal, half, height, power)
    points.push(`${i === 0 ? 'M' : 'L'} ${sectionXScale(center + xLocal).toFixed(2)} ${sectionYScale(z).toFixed(2)}`)
  }
  if (fill) {
    points.push(`L ${sectionXScale(center + half).toFixed(2)} ${sectionYScale(0).toFixed(2)}`)
    points.push(`L ${sectionXScale(center - half).toFixed(2)} ${sectionYScale(0).toFixed(2)} Z`)
  }
  return points.join(' ')
}

const reflowProfiles = computed(() => lensCenters.value.map(center => buildSectionProfile(center, reflowWidth.value, reflowHeight.value, 2.0, false)))
const finalProfiles = computed(() => lensCenters.value.map(center => buildSectionProfile(center, finalWidth.value, finalHeight.value, profilePower.value, true)))

// 3D wireframe
const surfaceW = 640
const surfaceH = 360
const surfaceCx = 322
const surfaceCy = 214
const surfaceScale = computed(() => 170 / Math.max(pitch.value, 0.4))
const surfaceZScale = computed(() => 120 / Math.max(finalHeight.value, 0.08))

function projectSurface(x: number, y: number, z: number) {
  const sx = surfaceScale.value
  return {
    x: surfaceCx + (x - y) * sx * 0.82,
    y: surfaceCy + (x + y) * sx * 0.38 - z * surfaceZScale.value,
  }
}

function buildSurfaceLine(points: { x: number; y: number; z: number }[]) {
  return points.map((p, i) => {
    const projected = projectSurface(p.x, p.y, p.z)
    return `${i === 0 ? 'M' : 'L'} ${projected.x.toFixed(2)} ${projected.y.toFixed(2)}`
  }).join(' ')
}

const surfaceLines = computed(() => {
  const lines: { key: string; d: string; stroke: string; major: boolean }[] = []
  const a = halfFinalWidth.value
  const steps = 16
  const samples = 42
  for (let row = 0; row <= steps; row += 1) {
    const y = -a + (2 * a * row) / steps
    const pts: { x: number; y: number; z: number }[] = []
    for (let i = 0; i <= samples; i += 1) {
      const x = -a + (2 * a * i) / samples
      const z = lensZ2D(x, y, a, finalHeight.value, profilePower.value)
      if (z > 0 || Math.abs(Math.abs(x) - a) < 1e-6) pts.push({ x, y, z })
    }
    if (pts.length > 1) {
      const major = row === 0 || row === steps || row === steps / 2
      lines.push({ key: `x-${row}`, d: buildSurfaceLine(pts), stroke: major ? '#1f78b4' : '#3498db', major })
    }
  }
  for (let col = 0; col <= steps; col += 1) {
    const x = -a + (2 * a * col) / steps
    const pts: { x: number; y: number; z: number }[] = []
    for (let i = 0; i <= samples; i += 1) {
      const y = -a + (2 * a * i) / samples
      const z = lensZ2D(x, y, a, finalHeight.value, profilePower.value)
      if (z > 0 || Math.abs(Math.abs(y) - a) < 1e-6) pts.push({ x, y, z })
    }
    if (pts.length > 1) {
      const major = col === 0 || col === steps || col === steps / 2
      lines.push({ key: `y-${col}`, d: buildSurfaceLine(pts), stroke: major ? '#8e44ad' : '#9b59b6', major })
    }
  }
  return lines
})

const surfaceFootprintEdges = computed(() => {
  const a = halfFinalWidth.value
  const edgePts: { x: number; y: number; z: number }[] = []
  const n = shapeExponent.value
  for (let i = 0; i <= 96; i += 1) {
    const theta = (2 * Math.PI * i) / 96
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    const denom = Math.pow(Math.pow(Math.abs(c), n) + Math.pow(Math.abs(s), n), 1 / n)
    edgePts.push({ x: (a * c) / denom, y: (a * s) / denom, z: 0 })
  }
  return [buildSurfaceLine(edgePts)]
})

const cellBasePolygon = computed(() => {
  const a = pitch.value / 2
  return [
    projectSurface(-a, -a, 0),
    projectSurface(a, -a, 0),
    projectSurface(a, a, 0),
    projectSurface(-a, a, 0),
  ].map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')
})

// Etch response plot
const processW = 640
const processH = 330
const processPad = { left: 58, right: 62, top: 24, bottom: 42 }
const processMaxTime = 140
const processXTicks = [0, 20, 40, 60, 80, 100, 120, 140]
const processGapTicks = computed(() => {
  const maxGap = Math.max(reflowGap.value, initialGap.value * 0.55, 0.08)
  return [0, maxGap * 0.25, maxGap * 0.5, maxGap * 0.75, maxGap].map(v => Number(v.toFixed(2)))
})
const processRetentionTicks = [0, 25, 50, 75, 100]
const processGapMax = computed(() => processGapTicks.value[processGapTicks.value.length - 1])

function processXScale(x: number) {
  const plotW = processW - processPad.left - processPad.right
  return processPad.left + (x / processMaxTime) * plotW
}

function processGapYScale(gap: number) {
  const plotH = processH - processPad.top - processPad.bottom
  return processPad.top + (1 - gap / Math.max(processGapMax.value, 0.01)) * plotH
}

function processRetentionYScale(retentionPct: number) {
  const plotH = processH - processPad.top - processPad.bottom
  return processPad.top + (1 - retentionPct / 100) * plotH
}

const processSamples = computed(() => {
  const rows: { time: number; gap: number; retention: number }[] = []
  for (let time = 0; time <= processMaxTime; time += 2) {
    const state = etchTransferAt(time)
    rows.push({
      time,
      gap: state.gap,
      retention: clamp((state.height / Math.max(reflowHeight.value, 0.001)) * 100, 0, 110),
    })
  }
  return rows
})

const gapCurvePath = computed(() => processSamples.value.map((p, i) => `${i === 0 ? 'M' : 'L'} ${processXScale(p.time).toFixed(2)} ${processGapYScale(p.gap).toFixed(2)}`).join(' '))
const heightCurvePath = computed(() => processSamples.value.map((p, i) => `${i === 0 ? 'M' : 'L'} ${processXScale(p.time).toFixed(2)} ${processRetentionYScale(Math.min(100, p.retention)).toFixed(2)}`).join(' '))
const currentProcessPoint = computed(() => ({
  x: processXScale(etchTime.value),
  gapY: processGapYScale(finalGap.value),
  heightY: processRetentionYScale(Math.min(100, heightRetention.value)),
}))

</script>

<style scoped>
.mlp-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.4rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.mlp-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.control-section {
  padding: 12px 0 2px;
  border-top: 1px solid var(--vp-c-divider);
}

.control-section:first-of-type {
  border-top: 0;
  padding-top: 0;
}

.control-heading {
  margin-bottom: 9px;
  font-size: 0.82em;
  font-weight: 700;
  color: var(--vp-c-text-1);
  text-transform: uppercase;
  letter-spacing: 0;
}

.controls-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.slider-group {
  flex: 1;
  min-width: 156px;
}

.slider-group.compact {
  max-width: 230px;
}

.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
  color: var(--vp-c-text-1);
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
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.ctrl-select {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}

.metric-card {
  padding: 9px 10px;
  min-height: 58px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.metric-card span {
  display: block;
  color: var(--vp-c-text-2);
  font-size: 0.78em;
  line-height: 1.25;
}

.metric-card strong {
  display: block;
  margin-top: 4px;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 1.02em;
}

.metric-card.accent {
  border-top: 3px solid var(--vp-c-brand-1);
}

.status-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.status-pill {
  padding: 4px 9px;
  border-radius: 999px;
  font-size: 0.78em;
  font-weight: 600;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
}

.status-pill.good {
  color: #1f7a4d;
  border-color: rgba(39, 174, 96, 0.35);
  background: rgba(39, 174, 96, 0.09);
}

.status-pill.warn {
  color: #a85f00;
  border-color: rgba(230, 126, 34, 0.35);
  background: rgba(230, 126, 34, 0.10);
}

.status-pill.risk {
  color: #a93226;
  border-color: rgba(192, 57, 43, 0.35);
  background: rgba(192, 57, 43, 0.10);
}

.tab-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.tab-btn {
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s, background 0.2s;
}

.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.tab-btn.active {
  color: #fff;
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
}

.plot-panel {
  display: flex;
  justify-content: center;
  padding: 8px 0 2px;
}

.main-svg {
  width: 100%;
  max-width: 680px;
  min-height: 260px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.axis-label,
.legend-label,
.plot-label,
.surface-note {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}

.legend-label {
  font-weight: 600;
}

.surface-title {
  font-size: 14px;
  font-weight: 700;
  fill: var(--vp-c-text-1);
}

.surface-note {
  font-size: 10px;
}

.formula-box {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.82em;
  line-height: 1.55;
}

.formula-box strong {
  color: var(--vp-c-text-1);
}

@media (max-width: 760px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .mlp-container {
    padding: 1rem;
  }

  .controls-row {
    flex-direction: column;
    gap: 8px;
  }

  .slider-group.compact {
    max-width: none;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }
}
</style>
