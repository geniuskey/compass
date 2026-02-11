<template>
  <div class="photon-journey-container">
    <h4>{{ t('Photon Journey Through a Pixel', '픽셀을 통과하는 광자의 여정') }}</h4>
    <p class="component-description">
      {{ t(
        'Watch what happens to 100 photons as they pass through a BSI pixel stack. See how many become signal vs. losses.',
        '100개의 광자가 BSI 픽셀 스택을 통과할 때 어떤 일이 일어나는지 관찰하세요. 신호가 되는 광자와 손실되는 광자의 수를 확인하세요.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="control-group">
        <label>{{ t('Wavelength', '파장') }}</label>
        <div class="wavelength-btns">
          <button
            v-for="wl in wavelengthOptions"
            :key="wl.value"
            :class="['wl-btn', { active: selectedWavelength === wl.value }]"
            :style="{ borderColor: wl.color }"
            @click="selectWavelength(wl.value)"
          >
            {{ t(wl.labelEn, wl.labelKo) }}
          </button>
        </div>
      </div>
      <div class="control-group">
        <label>{{ t('Speed', '속도') }}</label>
        <div class="speed-btns">
          <button
            v-for="sp in speedOptions"
            :key="sp.value"
            :class="['speed-btn', { active: speed === sp.value }]"
            @click="speed = sp.value"
          >{{ sp.label }}</button>
        </div>
      </div>
    </div>

    <div class="controls-row">
      <button class="ctrl-btn" @click="togglePlay">
        {{ isPlaying ? t('Pause', '일시정지') : t('Play', '재생') }}
      </button>
      <button class="ctrl-btn" @click="stepBackward" :disabled="currentStep <= 0 || isPlaying">
        {{ t('Prev', '이전') }}
      </button>
      <button class="ctrl-btn" @click="stepForward" :disabled="currentStep >= totalSteps || isPlaying">
        {{ t('Next', '다음') }}
      </button>
      <button class="ctrl-btn reset-btn" @click="resetAnimation">
        {{ t('Reset', '리셋') }}
      </button>
      <span class="step-indicator">
        {{ t('Step', '단계') }} {{ currentStep }} / {{ totalSteps }}
      </span>
    </div>

    <!-- Step description -->
    <div class="step-desc" v-if="currentStepInfo">
      <strong>{{ t(currentStepInfo.titleEn, currentStepInfo.titleKo) }}</strong>
      <span> &mdash; {{ t(currentStepInfo.descEn, currentStepInfo.descKo) }}</span>
    </div>

    <!-- SVG animation area -->
    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="journey-svg">
        <!-- Background: simplified pixel cross-section outlines -->
        <rect x="40" :y="layerY.microlens" :width="svgW - 80" height="30" fill="none" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="3,2" rx="2" />
        <text :x="svgW - 36" :y="layerY.microlens + 18" text-anchor="start" class="bg-label">{{ t('Lens', '렌즈') }}</text>

        <rect x="40" :y="layerY.colorfilter" :width="svgW - 80" height="40" :fill="filterBgColor" fill-opacity="0.15" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="3,2" rx="2" />
        <text :x="svgW - 36" :y="layerY.colorfilter + 24" text-anchor="start" class="bg-label">{{ t('Filter', '필터') }}</text>

        <rect x="40" :y="layerY.barl" :width="svgW - 80" height="20" fill="none" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="3,2" rx="2" />
        <text :x="svgW - 36" :y="layerY.barl + 14" text-anchor="start" class="bg-label">BARL</text>

        <rect x="40" :y="layerY.silicon" :width="svgW - 80" height="120" fill="#6b2c2c" fill-opacity="0.1" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="3,2" rx="2" />
        <text :x="svgW - 36" :y="layerY.silicon + 60" text-anchor="start" class="bg-label">{{ t('Silicon', '실리콘') }}</text>

        <!-- Photodiode region outline -->
        <rect x="100" :y="layerY.silicon + 30" width="200" height="60" fill="none" stroke="#b85c5c" stroke-width="1" stroke-dasharray="4,3" rx="4" />
        <text x="200" :y="layerY.silicon + 64" text-anchor="middle" class="pd-outline-label">{{ t('Photodiode', '포토다이오드') }}</text>

        <!-- Pixel boundary / neighbor -->
        <line x1="340" :y1="layerY.silicon" x2="340" :y2="layerY.silicon + 120" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="5,3" />
        <text x="365" :y="layerY.silicon + 60" text-anchor="middle" class="bg-label">{{ t('Neighbor', '인접') }}</text>

        <!-- Photon dots -->
        <circle
          v-for="p in photons"
          :key="p.id"
          :cx="p.x"
          :cy="p.y"
          :r="3"
          :fill="p.color"
          :opacity="p.opacity"
          :class="{ 'photon-dot': true }"
        />
      </svg>
    </div>

    <!-- Photon budget counters -->
    <div class="budget-row">
      <div class="budget-item">
        <span class="budget-dot" style="background: #999"></span>
        {{ t('Reflected', '반사') }}: <strong>{{ counts.reflected }}</strong>
      </div>
      <div class="budget-item">
        <span class="budget-dot" style="background: #e74c3c"></span>
        {{ t('Filtered', '필터 흡수') }}: <strong>{{ counts.filtered }}</strong>
      </div>
      <div class="budget-item">
        <span class="budget-dot" style="background: #8e44ad"></span>
        {{ t('Absorbed (Si)', '흡수 (Si)') }}: <strong>{{ counts.absorbedSi }}</strong>
      </div>
      <div class="budget-item">
        <span class="budget-dot" style="background: #2ecc71"></span>
        {{ t('Detected', '검출') }}: <strong>{{ counts.detected }}</strong>
      </div>
      <div class="budget-item">
        <span class="budget-dot" style="background: #f39c12"></span>
        {{ t('Crosstalk', '크로스토크') }}: <strong>{{ counts.crosstalk }}</strong>
      </div>
      <div class="budget-item">
        <span class="budget-dot" style="background: #3498db"></span>
        {{ t('Passed through', '투과') }}: <strong>{{ counts.passthrough }}</strong>
      </div>
    </div>

    <!-- QE result -->
    <div class="qe-result" v-if="currentStep >= totalSteps">
      {{ t('Quantum Efficiency', '양자 효율') }}: <strong>{{ counts.detected }} / 100 = {{ counts.detected }}% QE</strong>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive, onMounted, onUnmounted, watch } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const svgW = 440
const svgH = 380

const layerY = {
  start: 20,
  microlens: 50,
  colorfilter: 100,
  barl: 160,
  silicon: 200,
}

interface Photon {
  id: number
  x: number
  y: number
  color: string
  opacity: number
  state: 'active' | 'reflected' | 'filtered' | 'absorbed_si' | 'detected' | 'crosstalk' | 'passthrough'
  targetY: number
  targetX: number
  speed: number
}

interface WavelengthConfig {
  reflectedAtLens: number
  filteredOut: number
  reflectedAtBarl: number
  absorbedInSi: number
  detected: number
  crosstalk: number
  passthrough: number
  photonColor: string
  filterBg: string
}

const wavelengthConfigs: Record<number, WavelengthConfig> = {
  450: {
    reflectedAtLens: 4,
    filteredOut: 30,
    reflectedAtBarl: 1,
    absorbedInSi: 8,
    detected: 50,
    crosstalk: 3,
    passthrough: 4,
    photonColor: '#4488ff',
    filterBg: '#4488ff',
  },
  550: {
    reflectedAtLens: 4,
    filteredOut: 8,
    reflectedAtBarl: 1,
    absorbedInSi: 10,
    detected: 68,
    crosstalk: 4,
    passthrough: 5,
    photonColor: '#44cc44',
    filterBg: '#27ae60',
  },
  650: {
    reflectedAtLens: 4,
    filteredOut: 12,
    reflectedAtBarl: 1,
    absorbedInSi: 6,
    detected: 55,
    crosstalk: 6,
    passthrough: 16,
    photonColor: '#ee4444',
    filterBg: '#c0392b',
  },
}

const wavelengthOptions = [
  { value: 450, labelEn: 'Blue 450nm', labelKo: '청색 450nm', color: '#4488ff' },
  { value: 550, labelEn: 'Green 550nm', labelKo: '녹색 550nm', color: '#44cc44' },
  { value: 650, labelEn: 'Red 650nm', labelKo: '적색 650nm', color: '#ee4444' },
]

const speedOptions = [
  { value: 0.5, label: '0.5x' },
  { value: 1, label: '1x' },
  { value: 2, label: '2x' },
]

const selectedWavelength = ref(550)
const speed = ref(1)
const isPlaying = ref(false)
const currentStep = ref(0)
const totalSteps = 7

const config = computed(() => wavelengthConfigs[selectedWavelength.value])
const filterBgColor = computed(() => config.value.filterBg)

interface StepInfo {
  titleEn: string
  titleKo: string
  descEn: string
  descKo: string
}

const steps: StepInfo[] = [
  {
    titleEn: 'Step 1: Arrival',
    titleKo: '1단계: 도착',
    descEn: '100 photons arrive at the microlens surface.',
    descKo: '100개의 광자가 마이크로렌즈 표면에 도착합니다.',
  },
  {
    titleEn: 'Step 2: Microlens reflection',
    titleKo: '2단계: 마이크로렌즈 반사',
    descEn: 'A few photons reflect off the lens surface and are lost.',
    descKo: '일부 광자가 렌즈 표면에서 반사되어 손실됩니다.',
  },
  {
    titleEn: 'Step 3: Color filter',
    titleKo: '3단계: 컬러 필터',
    descEn: 'Wrong-color photons are absorbed by the dye filter.',
    descKo: '잘못된 색상의 광자가 염료 필터에 의해 흡수됩니다.',
  },
  {
    titleEn: 'Step 4: BARL',
    titleKo: '4단계: BARL',
    descEn: 'Anti-reflection coating minimizes reflection at the silicon interface.',
    descKo: '반사 방지 코팅이 실리콘 계면에서의 반사를 최소화합니다.',
  },
  {
    titleEn: 'Step 5: Silicon absorption',
    titleKo: '5단계: 실리콘 흡수',
    descEn: 'Photons are absorbed at varying depths, generating electrons.',
    descKo: '광자가 다양한 깊이에서 흡수되어 전자를 생성합니다.',
  },
  {
    titleEn: 'Step 6: Detection & crosstalk',
    titleKo: '6단계: 검출 및 크로스토크',
    descEn: 'Photons in the photodiode become signal. Some leak to neighbors.',
    descKo: '포토다이오드의 광자가 신호가 됩니다. 일부는 인접 픽셀로 누출됩니다.',
  },
  {
    titleEn: 'Step 7: Final score',
    titleKo: '7단계: 최종 결과',
    descEn: 'The photon budget determines the Quantum Efficiency (QE).',
    descKo: '광자 수지가 양자 효율(QE)을 결정합니다.',
  },
]

const currentStepInfo = computed(() => {
  if (currentStep.value <= 0 || currentStep.value > totalSteps) return null
  return steps[currentStep.value - 1]
})

// Photon state
const photons = reactive<Photon[]>([])
const counts = reactive({
  reflected: 0,
  filtered: 0,
  absorbedSi: 0,
  detected: 0,
  crosstalk: 0,
  passthrough: 0,
})

function initPhotons() {
  photons.length = 0
  const c = config.value
  for (let i = 0; i < 100; i++) {
    photons.push({
      id: i,
      x: 60 + (i % 20) * 16 + (Math.random() - 0.5) * 6,
      y: layerY.start + Math.random() * 20,
      color: c.photonColor,
      opacity: 1,
      state: 'active',
      targetY: layerY.start + Math.random() * 20,
      targetX: 0,
      speed: 0.8 + Math.random() * 0.4,
    })
  }
  counts.reflected = 0
  counts.filtered = 0
  counts.absorbedSi = 0
  counts.detected = 0
  counts.crosstalk = 0
  counts.passthrough = 0
}

function selectWavelength(wl: number) {
  selectedWavelength.value = wl
  resetAnimation()
}

function resetAnimation() {
  isPlaying.value = false
  currentStep.value = 0
  initPhotons()
}

// Assign photon fates based on config
function assignFates() {
  const c = config.value
  let idx = 0
  const fates: string[] = []
  for (let i = 0; i < c.reflectedAtLens; i++) fates.push('reflected_lens')
  for (let i = 0; i < c.filteredOut; i++) fates.push('filtered')
  for (let i = 0; i < c.reflectedAtBarl; i++) fates.push('reflected_barl')
  for (let i = 0; i < c.absorbedInSi; i++) fates.push('absorbed_si')
  for (let i = 0; i < c.detected; i++) fates.push('detected')
  for (let i = 0; i < c.crosstalk; i++) fates.push('crosstalk')
  for (let i = 0; i < c.passthrough; i++) fates.push('passthrough')
  // Pad if needed
  while (fates.length < 100) fates.push('detected')

  // Shuffle fates
  for (let i = fates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[fates[i], fates[j]] = [fates[j], fates[i]]
  }

  return fates
}

let fates: string[] = []

function executeStep(step: number) {
  if (step === 1) {
    fates = assignFates()
    // Move all photons to just above microlens
    for (let i = 0; i < photons.length; i++) {
      photons[i].targetY = layerY.microlens - 2
      photons[i].x = 60 + (i % 20) * 16 + (Math.random() - 0.5) * 4
    }
  } else if (step === 2) {
    // Reflected at lens: bounce upward
    for (let i = 0; i < photons.length; i++) {
      if (fates[i] === 'reflected_lens') {
        photons[i].targetY = 5
        photons[i].color = '#999'
        photons[i].opacity = 0.4
        photons[i].state = 'reflected'
        counts.reflected++
      } else {
        photons[i].targetY = layerY.colorfilter - 2
      }
    }
  } else if (step === 3) {
    // Filtered out
    for (let i = 0; i < photons.length; i++) {
      if (photons[i].state !== 'active') continue
      if (fates[i] === 'filtered') {
        photons[i].color = '#e74c3c'
        photons[i].opacity = 0.3
        photons[i].state = 'filtered'
        photons[i].targetY = layerY.colorfilter + 20
        counts.filtered++
      } else {
        photons[i].targetY = layerY.barl - 2
      }
    }
  } else if (step === 4) {
    // BARL reflection
    for (let i = 0; i < photons.length; i++) {
      if (photons[i].state !== 'active') continue
      if (fates[i] === 'reflected_barl') {
        photons[i].targetY = 5
        photons[i].color = '#999'
        photons[i].opacity = 0.4
        photons[i].state = 'reflected'
        counts.reflected++
      } else {
        photons[i].targetY = layerY.silicon + 5
      }
    }
  } else if (step === 5) {
    // Silicon absorption: photons stop at varying depths
    for (let i = 0; i < photons.length; i++) {
      if (photons[i].state !== 'active') continue
      if (fates[i] === 'absorbed_si') {
        // Absorbed outside photodiode
        const depth = 10 + Math.random() * 100
        photons[i].targetY = layerY.silicon + depth
        photons[i].color = '#8e44ad'
        photons[i].opacity = 0.5
        photons[i].state = 'absorbed_si'
        counts.absorbedSi++
      } else if (fates[i] === 'detected') {
        const depth = 30 + Math.random() * 55
        photons[i].targetY = layerY.silicon + depth
        photons[i].x = 110 + Math.random() * 180
      } else if (fates[i] === 'crosstalk') {
        const depth = 30 + Math.random() * 80
        photons[i].targetY = layerY.silicon + depth
      } else if (fates[i] === 'passthrough') {
        photons[i].targetY = layerY.silicon + 115
      }
    }
  } else if (step === 6) {
    // Detection and crosstalk
    for (let i = 0; i < photons.length; i++) {
      if (photons[i].state !== 'active') continue
      if (fates[i] === 'detected') {
        photons[i].color = '#2ecc71'
        photons[i].opacity = 1
        photons[i].state = 'detected'
        counts.detected++
      } else if (fates[i] === 'crosstalk') {
        photons[i].targetX = 350 + Math.random() * 40
        photons[i].color = '#f39c12'
        photons[i].opacity = 0.8
        photons[i].state = 'crosstalk'
        counts.crosstalk++
      } else if (fates[i] === 'passthrough') {
        photons[i].color = '#3498db'
        photons[i].opacity = 0.4
        photons[i].state = 'passthrough'
        counts.passthrough++
      }
    }
  }
  // Step 7 is just the summary, no movement
}

// Animation loop
let animFrame: number | null = null
let lastTime = 0

function animatePhotons(timestamp: number) {
  if (!lastTime) lastTime = timestamp
  const dt = (timestamp - lastTime) / 1000
  lastTime = timestamp

  const moveSpeed = 120 * speed.value

  let allSettled = true
  for (const p of photons) {
    // Move toward targetY
    if (Math.abs(p.y - p.targetY) > 1) {
      allSettled = false
      const dir = p.targetY > p.y ? 1 : -1
      p.y += dir * moveSpeed * p.speed * dt
      if (dir > 0 && p.y > p.targetY) p.y = p.targetY
      if (dir < 0 && p.y < p.targetY) p.y = p.targetY
    }
    // Move toward targetX if set (crosstalk)
    if (p.targetX && Math.abs(p.x - p.targetX) > 1) {
      allSettled = false
      const dir = p.targetX > p.x ? 1 : -1
      p.x += dir * moveSpeed * 0.6 * dt
      if (dir > 0 && p.x > p.targetX) p.x = p.targetX
      if (dir < 0 && p.x < p.targetX) p.x = p.targetX
    }
  }

  if (allSettled && isPlaying.value) {
    // Move to next step after a pause
    if (currentStep.value < totalSteps) {
      setTimeout(() => {
        if (isPlaying.value) {
          currentStep.value++
          executeStep(currentStep.value)
        }
      }, 600 / speed.value)
    } else {
      isPlaying.value = false
    }
  }

  if (isPlaying.value || !allSettled) {
    animFrame = requestAnimationFrame(animatePhotons)
  } else {
    animFrame = null
  }
}

function startAnimation() {
  if (animFrame) return
  lastTime = 0
  animFrame = requestAnimationFrame(animatePhotons)
}

function togglePlay() {
  if (isPlaying.value) {
    isPlaying.value = false
  } else {
    isPlaying.value = true
    if (currentStep.value >= totalSteps) {
      resetAnimation()
      currentStep.value = 0
    }
    currentStep.value++
    executeStep(currentStep.value)
    startAnimation()
  }
}

function stepForward() {
  if (currentStep.value >= totalSteps) return
  currentStep.value++
  executeStep(currentStep.value)
  // Animate photons to new positions
  if (!animFrame) {
    lastTime = 0
    animFrame = requestAnimationFrame(animatePhotons)
  }
}

function stepBackward() {
  if (currentStep.value <= 0) return
  // Reset and replay up to previous step
  const targetStep = currentStep.value - 1
  initPhotons()
  currentStep.value = 0
  for (let s = 1; s <= targetStep; s++) {
    currentStep.value = s
    executeStep(s)
    // Instantly move photons to targets
    for (const p of photons) {
      p.y = p.targetY
      if (p.targetX) p.x = p.targetX
    }
  }
}

onMounted(() => {
  initPhotons()
})

onUnmounted(() => {
  if (animFrame) {
    cancelAnimationFrame(animFrame)
    animFrame = null
  }
})

watch(selectedWavelength, () => {
  resetAnimation()
})
</script>

<style scoped>
.photon-journey-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.photon-journey-container h4 {
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
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 12px;
}
.control-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.control-group label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  font-weight: 600;
}
.wavelength-btns, .speed-btns {
  display: flex;
  gap: 6px;
}
.wl-btn {
  padding: 5px 10px;
  border: 2px solid #888;
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.82em;
  cursor: pointer;
  transition: all 0.2s;
}
.wl-btn.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  font-weight: 600;
}
.wl-btn:hover {
  opacity: 0.85;
}
.speed-btn {
  padding: 4px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.82em;
  cursor: pointer;
}
.speed-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  font-weight: 600;
}
.ctrl-btn {
  padding: 7px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-brand-1);
  color: #fff;
  font-size: 0.88em;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}
.ctrl-btn:hover {
  opacity: 0.85;
}
.ctrl-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.reset-btn {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-divider);
}
.step-indicator {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}
.step-desc {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: 0.9em;
  color: var(--vp-c-text-1);
}
.step-desc strong {
  color: var(--vp-c-brand-1);
}
.svg-wrapper {
  margin: 8px 0;
  overflow-x: auto;
}
.journey-svg {
  width: 100%;
  max-width: 500px;
  display: block;
  margin: 0 auto;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}
.bg-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.pd-outline-label {
  font-size: 8px;
  fill: #b85c5c;
}
.photon-dot {
  transition: opacity 0.3s;
}
.budget-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 12px;
  padding: 10px 14px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
}
.budget-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.82em;
  color: var(--vp-c-text-1);
}
.budget-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.qe-result {
  margin-top: 12px;
  padding: 12px 16px;
  background: var(--vp-c-brand-soft);
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 8px;
  font-size: 1em;
  text-align: center;
  color: var(--vp-c-text-1);
}
.qe-result strong {
  color: var(--vp-c-brand-1);
  font-size: 1.1em;
}
@media (max-width: 600px) {
  .controls-row {
    gap: 8px;
  }
  .wl-btn, .ctrl-btn {
    padding: 4px 8px;
    font-size: 0.78em;
  }
  .budget-row {
    gap: 6px;
  }
  .budget-item {
    font-size: 0.75em;
  }
}
</style>
