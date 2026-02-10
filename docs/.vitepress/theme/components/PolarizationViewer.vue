<template>
  <div class="polarization-container">
    <h4>{{ t('Polarization State Viewer', '편광 상태 뷰어') }}</h4>
    <p class="component-description">
      {{ t(
        'Visualize different polarization states of light with animated E-field vector propagation in a 3D perspective view.',
        '3D 투시도에서 애니메이션 전기장 벡터 전파를 통해 다양한 편광 상태를 시각화합니다.'
      ) }}
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label for="pol-type">{{ t('Polarization:', '편광:') }}</label>
        <select id="pol-type" v-model="polType" class="pol-select">
          <option v-for="p in polOptions" :key="p.key" :value="p.key">{{ t(p.label, p.labelKo) }}</option>
        </select>
      </div>
      <button class="play-btn" @click="togglePlay">
        {{ isPlaying ? t('Pause', '일시정지') : t('Play', '재생') }}
      </button>
      <div class="slider-group">
        <label>{{ t('Speed:', '속도:') }} <strong>{{ speed.toFixed(1) }}x</strong></label>
        <input type="range" min="0.2" max="3.0" step="0.1" v-model.number="speed" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">{{ t('Polarization Type:', '편광 유형:') }}</span>
        <span class="info-value">{{ t(currentInfo.type, currentInfo.typeKo) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">Ex:</span>
        <span class="info-value">{{ currentInfo.ex }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">Ey:</span>
        <span class="info-value">{{ currentInfo.ey }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Jones vector:', 'Jones 벡터:') }}</span>
        <span class="info-value jones">{{ currentInfo.jones }}</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${W} ${H}`" class="pol-svg">
        <!-- Background -->
        <rect x="0" y="0" :width="W" :height="H" fill="none" />

        <!-- Propagation axis (k vector) -->
        <line
          :x1="projX(0, 0, -axisLen)"
          :y1="projY(0, 0, -axisLen)"
          :x2="projX(0, 0, axisLen)"
          :y2="projY(0, 0, axisLen)"
          stroke="var(--vp-c-text-3)"
          stroke-width="1"
          stroke-dasharray="4,3"
        />
        <line
          :x1="projX(0, 0, axisLen - 20)"
          :y1="projY(0, 0, axisLen - 20)"
          :x2="projX(0, 0, axisLen)"
          :y2="projY(0, 0, axisLen)"
          stroke="var(--vp-c-text-1)"
          stroke-width="2"
          marker-end="url(#polArrowK)"
        />
        <text
          :x="projX(0, 0, axisLen) + 8"
          :y="projY(0, 0, axisLen) + 4"
          class="vec-label"
        >k</text>

        <!-- Reference axes at origin -->
        <!-- Vertical axis (y) -->
        <line
          :x1="projX(0, -35, 0)"
          :y1="projY(0, -35, 0)"
          :x2="projX(0, 35, 0)"
          :y2="projY(0, 35, 0)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="2,2"
          opacity="0.5"
        />
        <!-- Horizontal axis (x) -->
        <line
          :x1="projX(-35, 0, 0)"
          :y1="projY(-35, 0, 0)"
          :x2="projX(35, 0, 0)"
          :y2="projY(35, 0, 0)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="2,2"
          opacity="0.5"
        />

        <!-- Wave trail (E-field path) -->
        <path :d="wavePath" fill="none" stroke="#3498db" stroke-width="2" opacity="0.7" />

        <!-- E-field vectors along propagation -->
        <template v-for="(vec, i) in fieldVectors" :key="'fv' + i">
          <line
            :x1="projX(0, 0, vec.z)"
            :y1="projY(0, 0, vec.z)"
            :x2="projX(vec.ex, vec.ey, vec.z)"
            :y2="projY(vec.ex, vec.ey, vec.z)"
            stroke="#e74c3c"
            :stroke-width="1.2"
            :opacity="vec.opacity"
          />
        </template>

        <!-- Current E-field vector (tip) -->
        <line
          :x1="projX(0, 0, tipZ)"
          :y1="projY(0, 0, tipZ)"
          :x2="projX(tipEx, tipEy, tipZ)"
          :y2="projY(tipEx, tipEy, tipZ)"
          stroke="#e74c3c"
          stroke-width="2.5"
          marker-end="url(#polArrowE)"
        />
        <circle
          :cx="projX(tipEx, tipEy, tipZ)"
          :cy="projY(tipEx, tipEy, tipZ)"
          r="3"
          fill="#e74c3c"
        />

        <!-- Labels -->
        <text :x="projX(0, 42, 0)" :y="projY(0, 42, 0)" class="axis-text">y (Ey)</text>
        <text :x="projX(42, 0, 0)" :y="projY(42, 0, 0)" class="axis-text">x (Ex)</text>

        <defs>
          <marker id="polArrowK" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="var(--vp-c-text-1)" />
          </marker>
          <marker id="polArrowE" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
            <polygon points="0 0, 7 2.5, 0 5" fill="#e74c3c" />
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

const W = 520
const H = 300
const cx = W / 2
const cy = H / 2
const amp = 40
const axisLen = 160

const polOptions = [
  { key: 'te', label: 'TE (s-polarization)', labelKo: 'TE (s-편광)' },
  { key: 'tm', label: 'TM (p-polarization)', labelKo: 'TM (p-편광)' },
  { key: 'rcp', label: 'Circular (RCP)', labelKo: '원형 편광 (RCP)' },
  { key: 'lcp', label: 'Circular (LCP)', labelKo: '원형 편광 (LCP)' },
  { key: 'elliptical', label: 'Elliptical', labelKo: '타원 편광' },
  { key: 'unpolarized', label: 'Unpolarized', labelKo: '비편광' },
]

const polType = ref('rcp')
const isPlaying = ref(true)
const speed = ref(1.0)
const phase = ref(0)
const unpolarizedAngle = ref(0)
let animFrame = null
let unpolarizedTimer = 0

const polInfo = {
  te: { type: 'TE (s-pol)', typeKo: 'TE (s-편광)', ex: '0', ey: 'E0 cos(wt-kz)', jones: '[0, 1]' },
  tm: { type: 'TM (p-pol)', typeKo: 'TM (p-편광)', ex: 'E0 cos(wt-kz)', ey: '0', jones: '[1, 0]' },
  rcp: { type: 'Right Circular', typeKo: '우원형 편광', ex: 'E0 cos(wt-kz)', ey: 'E0 sin(wt-kz)', jones: '[1, -i] / sqrt(2)' },
  lcp: { type: 'Left Circular', typeKo: '좌원형 편광', ex: 'E0 cos(wt-kz)', ey: '-E0 sin(wt-kz)', jones: '[1, i] / sqrt(2)' },
  elliptical: { type: 'Elliptical', typeKo: '타원 편광', ex: 'E0 cos(wt-kz)', ey: '0.5 E0 sin(wt-kz)', jones: '[1, -0.5i]' },
  unpolarized: { type: 'Unpolarized', typeKo: '비편광', ex: 'Random', ey: 'Random', jones: 'N/A (mixed state)' },
}

const currentInfo = computed(() => polInfo[polType.value])

// 3D to 2D projection (simple isometric-like)
function projX(x, y, z) {
  return cx + x + z * 0.6
}
function projY(x, y, z) {
  return cy - y + z * 0.25
}

function getField(zPos, phaseVal) {
  const kz = (zPos / 80) * 2 * Math.PI
  const wt = phaseVal
  const t = wt - kz

  switch (polType.value) {
    case 'te':
      return { ex: 0, ey: amp * Math.cos(t) }
    case 'tm':
      return { ex: amp * Math.cos(t), ey: 0 }
    case 'rcp':
      return { ex: amp * Math.cos(t), ey: amp * Math.sin(t) }
    case 'lcp':
      return { ex: amp * Math.cos(t), ey: -amp * Math.sin(t) }
    case 'elliptical':
      return { ex: amp * Math.cos(t), ey: 0.5 * amp * Math.sin(t) }
    case 'unpolarized': {
      const angle = unpolarizedAngle.value
      return {
        ex: amp * Math.cos(t) * Math.cos(angle),
        ey: amp * Math.cos(t) * Math.sin(angle),
      }
    }
    default:
      return { ex: 0, ey: 0 }
  }
}

const tipZ = 80
const tipEx = computed(() => getField(tipZ, phase.value).ex)
const tipEy = computed(() => getField(tipZ, phase.value).ey)

const fieldVectors = computed(() => {
  const vecs = []
  const numVecs = 16
  for (let i = 0; i < numVecs; i++) {
    const z = -axisLen + 10 + (i / (numVecs - 1)) * (axisLen * 2 - 20)
    const f = getField(z, phase.value)
    const distFromTip = Math.abs(z - tipZ)
    const opacity = Math.max(0.1, 1 - distFromTip / (axisLen * 1.5))
    vecs.push({ z, ex: f.ex, ey: f.ey, opacity })
  }
  return vecs
})

const wavePath = computed(() => {
  let d = ''
  const steps = 120
  for (let i = 0; i <= steps; i++) {
    const z = -axisLen + 10 + (i / steps) * (axisLen * 2 - 20)
    const f = getField(z, phase.value)
    const px = projX(f.ex, f.ey, z)
    const py = projY(f.ex, f.ey, z)
    d += i === 0 ? `M ${px.toFixed(1)} ${py.toFixed(1)}` : ` L ${px.toFixed(1)} ${py.toFixed(1)}`
  }
  return d
})

function animate() {
  phase.value += 0.04 * speed.value
  if (phase.value > 200 * Math.PI) phase.value -= 200 * Math.PI

  if (polType.value === 'unpolarized') {
    unpolarizedTimer += 1
    if (unpolarizedTimer > 12) {
      unpolarizedAngle.value = Math.random() * 2 * Math.PI
      unpolarizedTimer = 0
    }
  }

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
.polarization-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.polarization-container h4 {
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
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.9em;
  font-weight: 600;
}
.pol-select {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
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
.info-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85em;
}
.info-label {
  color: var(--vp-c-text-2);
  margin-right: 4px;
}
.info-value {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.info-value.jones {
  font-size: 0.85em;
}
.svg-wrapper {
  margin-top: 8px;
}
.pol-svg {
  width: 100%;
  max-width: 580px;
  display: block;
  margin: 0 auto;
}
.vec-label {
  font-size: 14px;
  fill: var(--vp-c-text-1);
  font-weight: 700;
  font-style: italic;
}
.axis-text {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
</style>
