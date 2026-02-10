<template>
  <div class="emwave-container">
    <h4>{{ t('Electromagnetic Wave Propagation', '전자기파 전파') }}</h4>
    <p class="component-description">
      {{ t('Animated EM wave showing perpendicular E and H fields. Adjust absorption to see exponential decay in an absorbing medium.', '수직인 E장과 H장을 보여주는 전자기파 애니메이션입니다. 흡수를 조절하여 흡수 매질에서의 지수적 감쇠를 확인하세요.') }}
    </p>

    <div class="controls-row">
      <button class="play-btn" @click="togglePlay">
        {{ isPlaying ? t('Pause', '일시정지') : t('Play', '재생') }}
      </button>
      <div class="slider-group">
        <label>{{ t('Wavelength', '파장') }}: <strong>{{ wavelength }} nm</strong></label>
        <input type="range" min="400" max="700" step="10" v-model.number="wavelength" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Absorption', '흡수') }} k: <strong>{{ kValue.toFixed(2) }}</strong></label>
        <input type="range" min="0" max="1.0" step="0.01" v-model.number="kValue" class="ctrl-range" />
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="emwave-svg">
        <!-- Background regions -->
        <!-- Transparent medium (left) -->
        <rect :x="padL" :y="0" :width="interfaceX - padL" :height="H" fill="#e8f4fd" opacity="0.25" />
        <!-- Absorbing medium (right) -->
        <rect :x="interfaceX" :y="0" :width="padL + plotW - interfaceX" :height="H" fill="#fde8e8" :opacity="kValue > 0 ? 0.3 : 0.1" />

        <!-- Interface line -->
        <line
          v-if="kValue > 0"
          :x1="interfaceX"
          y1="10"
          :x2="interfaceX"
          :y2="H - 10"
          stroke="var(--vp-c-text-3)"
          stroke-width="2"
          stroke-dasharray="6,4"
        />
        <text
          v-if="kValue > 0"
          :x="interfaceX"
          :y="H - 4"
          text-anchor="middle"
          class="interface-label"
        >{{ t('Interface', '계면') }}</text>

        <!-- Propagation axis -->
        <line :x1="padL" :y1="midY" :x2="padL + plotW" :y2="midY" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="3,3" />

        <!-- E-field wave (blue, vertical oscillation) -->
        <path :d="eFieldPath" fill="none" stroke="#3498db" stroke-width="2.5" />

        <!-- H-field wave (red, shown as dashed to imply perpendicular plane) -->
        <path :d="hFieldPath" fill="none" stroke="#e74c3c" stroke-width="2" stroke-dasharray="6,3" />

        <!-- Propagation arrow (k vector) -->
        <line :x1="padL + 10" :y1="midY - 60" :x2="padL + 50" :y2="midY - 60" stroke="#333" stroke-width="2" marker-end="url(#emArrowK)" />
        <text :x="padL + 55" :y="midY - 56" class="vec-label">k</text>

        <!-- Wavelength bracket -->
        <line :x1="wlBracketX1" :y1="midY + 55" :x2="wlBracketX2" :y2="midY + 55" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="wlBracketX1" :y1="midY + 50" :x2="wlBracketX1" :y2="midY + 60" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="wlBracketX2" :y1="midY + 50" :x2="wlBracketX2" :y2="midY + 60" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <text :x="(wlBracketX1 + wlBracketX2) / 2" :y="midY + 72" text-anchor="middle" class="lambda-label">&lambda;</text>

        <!-- Labels -->
        <text :x="padL + 15" :y="midY - 42" class="field-label e-label">E</text>
        <text :x="padL + 15" :y="midY + 38" class="field-label h-label">H</text>

        <!-- Medium labels -->
        <text v-if="kValue > 0" :x="interfaceX - 20" y="20" text-anchor="end" class="medium-text">n = 1.0</text>
        <text v-if="kValue > 0" :x="interfaceX + 10" y="20" text-anchor="start" class="medium-text">k = {{ kValue.toFixed(2) }}</text>

        <defs>
          <marker id="emArrowK" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#333" />
          </marker>
        </defs>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const W = 560
const H = 240
const padL = 30
const plotW = 500
const midY = H / 2
const amplitude = 45

const wavelength = ref(550)
const kValue = ref(0.0)
const isPlaying = ref(true)
const phase = ref(0)
let animFrame = null

const interfaceX = computed(() => padL + plotW * 0.4)

// Wavelength in SVG units (pixels for one period)
const wlPx = computed(() => {
  // Map 400-700nm to a visual period of 60-110px
  return 60 + ((wavelength.value - 400) / 300) * 50
})

const wlBracketX1 = computed(() => padL + 70)
const wlBracketX2 = computed(() => padL + 70 + wlPx.value)

function buildWavePath(phaseOffset, amplitudeSign) {
  const steps = 300
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const x = padL + (i / steps) * plotW
    const distFromStart = (i / steps) * plotW
    const spatialPhase = (distFromStart / wlPx.value) * 2 * Math.PI

    let env = 1.0
    if (kValue.value > 0 && x > interfaceX.value) {
      const distInMedium = x - interfaceX.value
      // Absorption: amplitude decays as exp(-alpha * d), where alpha ~ k * (2pi/lambda_px)
      const alphaVisual = kValue.value * (2 * Math.PI / wlPx.value) * 2.5
      env = Math.exp(-alphaVisual * distInMedium)
    }

    const val = env * amplitude * amplitudeSign * Math.sin(spatialPhase - phase.value + phaseOffset)
    const y = midY - val
    d += i === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`
  }
  return d
}

const eFieldPath = computed(() => buildWavePath(0, 1))
const hFieldPath = computed(() => buildWavePath(0, -0.7))

function animate() {
  phase.value += 0.06
  if (phase.value > 200 * Math.PI) phase.value -= 200 * Math.PI
  animFrame = requestAnimationFrame(animate)
}

function togglePlay() {
  isPlaying.value = !isPlaying.value
  if (isPlaying.value) {
    animFrame = requestAnimationFrame(animate)
  } else if (animFrame) {
    cancelAnimationFrame(animFrame)
    animFrame = null
  }
}

onMounted(() => {
  if (isPlaying.value) {
    animFrame = requestAnimationFrame(animate)
  }
})

onUnmounted(() => {
  if (animFrame) {
    cancelAnimationFrame(animFrame)
  }
})
</script>

<style scoped>
.emwave-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.emwave-container h4 {
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
  gap: 16px;
  flex-wrap: wrap;
  align-items: flex-end;
  margin-bottom: 16px;
}
.play-btn {
  padding: 8px 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.9em;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}
.play-btn:hover {
  opacity: 0.85;
}
.slider-group {
  flex: 1;
  min-width: 150px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
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
.svg-wrapper {
  margin-top: 8px;
}
.emwave-svg {
  width: 100%;
  max-width: 620px;
  display: block;
  margin: 0 auto;
}
.interface-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.vec-label {
  font-size: 12px;
  fill: #333;
  font-weight: 700;
  font-style: italic;
}
.field-label {
  font-size: 14px;
  font-weight: 700;
  font-style: italic;
}
.e-label {
  fill: #3498db;
}
.h-label {
  fill: #e74c3c;
}
.lambda-label {
  font-size: 12px;
  fill: var(--vp-c-text-2);
  font-style: italic;
}
.medium-text {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
</style>
