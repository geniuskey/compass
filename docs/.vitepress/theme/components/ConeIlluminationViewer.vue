<template>
  <div class="cone-illum-container">
    <h4>{{ t('Interactive Cone Illumination Viewer', '인터랙티브 콘 조명 뷰어') }}</h4>
    <p class="component-description">
      {{ t('Visualize how Chief Ray Angle (CRA) and cone half-angle affect pixel illumination. The microlens shifts to compensate for oblique incidence.', 'CRA(주광선 각도)와 콘 반각이 픽셀 조명에 미치는 영향을 시각화합니다. 마이크로렌즈가 경사 입사를 보상하기 위해 이동합니다.') }}
    </p>

    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('CRA Angle', 'CRA 각도') }}: <strong>{{ cra.toFixed(1) }}&deg;</strong>
        </label>
        <input type="range" min="0" max="30" step="0.5" v-model.number="cra" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Cone half-angle', '콘 반각') }}: <strong>{{ halfAngle.toFixed(1) }}&deg;</strong>
        </label>
        <input type="range" min="5" max="30" step="0.5" v-model.number="halfAngle" class="ctrl-range" />
      </div>
    </div>

    <div class="info-row">
      <div class="info-card">
        <span class="info-label">CRA:</span>
        <span class="info-value">{{ cra.toFixed(1) }}&deg;</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Microlens shift', '마이크로렌즈 시프트') }}:</span>
        <span class="info-value">{{ mlShift.toFixed(3) }} um</span>
      </div>
      <div class="info-card">
        <span class="info-label">f/#:</span>
        <span class="info-value">{{ fNumber.toFixed(1) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('Solid angle', '입체각') }}:</span>
        <span class="info-value">{{ solidAngle.toFixed(3) }} sr</span>
      </div>
    </div>

    <div class="svg-wrapper">
      <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="cone-svg">
        <!-- Background regions -->
        <rect x="0" y="0" :width="svgW" :height="mlY" fill="#f0f8ff" opacity="0.3" />
        <rect x="0" :y="mlY" :width="svgW" :height="pdY - mlY" fill="#e8f0e8" opacity="0.3" />
        <rect x="0" :y="pdY" :width="svgW" :height="svgH - pdY" fill="#e8e8ee" opacity="0.3" />

        <!-- Pixel boundaries -->
        <line :x1="pixelLeft" :y1="mlY" :x2="pixelLeft" :y2="svgH - 10" stroke="var(--vp-c-divider)" stroke-width="1" stroke-dasharray="3,2" />
        <line :x1="pixelRight" :y1="mlY" :x2="pixelRight" :y2="svgH - 10" stroke="var(--vp-c-divider)" stroke-width="1" stroke-dasharray="3,2" />

        <!-- Microlens (curved) -->
        <path
          :d="microlensPath"
          fill="#dda0dd"
          fill-opacity="0.5"
          stroke="#9b59b6"
          stroke-width="1.5"
        />
        <text :x="mlCenterX" :y="mlY - 8" text-anchor="middle" class="region-label">{{ t('Microlens', '마이크로렌즈') }}</text>

        <!-- Microlens shift arrow -->
        <template v-if="cra > 1">
          <line
            :x1="pixelCenterX"
            :y1="mlY + 12"
            :x2="mlCenterX"
            :y2="mlY + 12"
            stroke="#9b59b6"
            stroke-width="1.5"
            marker-end="url(#coneArrowPurple)"
          />
          <text :x="(pixelCenterX + mlCenterX) / 2" :y="mlY + 24" text-anchor="middle" class="shift-label">{{ t('shift', '시프트') }}</text>
        </template>

        <!-- Stack layers (simplified) -->
        <rect :x="pixelLeft" :y="mlY + mlH" :width="pixelW" :height="planH" fill="#add8e6" opacity="0.4" stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text :x="pixelCenterX" :y="mlY + mlH + planH / 2 + 3" text-anchor="middle" class="small-label">{{ t('Planarization', '평탄화층') }}</text>

        <rect :x="pixelLeft" :y="cfY" :width="pixelW" :height="cfH" fill="#90ee90" opacity="0.4" stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text :x="pixelCenterX" :y="cfY + cfH / 2 + 3" text-anchor="middle" class="small-label">{{ t('Color Filter', '컬러 필터') }}</text>

        <!-- Photodiode region -->
        <rect :x="pdLeft" :y="pdY" :width="pdW" :height="pdH" fill="#f5deb3" opacity="0.5" stroke="#d2691e" stroke-width="1.2" rx="2" />
        <text :x="pixelCenterX" :y="pdY + pdH / 2 + 4" text-anchor="middle" class="region-label">{{ t('Photodiode', '포토다이오드') }}</text>

        <!-- Light rays (cone) -->
        <template v-for="(ray, idx) in rays" :key="idx">
          <line
            :x1="ray.x1"
            :y1="ray.y1"
            :x2="ray.x2"
            :y2="ray.y2"
            :stroke="ray.color"
            :stroke-width="ray.isChief ? 2 : 1"
            :opacity="ray.isChief ? 0.9 : 0.45"
            :stroke-dasharray="ray.isChief ? 'none' : '4,2'"
          />
          <!-- Refracted continuation into stack -->
          <line
            :x1="ray.x2"
            :y1="ray.y2"
            :x2="ray.xBot"
            :y2="ray.yBot"
            :stroke="ray.color"
            :stroke-width="ray.isChief ? 1.5 : 0.8"
            :opacity="ray.isChief ? 0.7 : 0.3"
          />
        </template>

        <!-- Focus spot indicator -->
        <ellipse
          :cx="focusX"
          :cy="pdY + 6"
          :rx="focusSpotR"
          ry="3"
          fill="#f39c12"
          opacity="0.6"
        />
        <text :x="focusX" :y="pdY - 4" text-anchor="middle" class="focus-label">{{ t('Focus spot', '초점 위치') }}</text>

        <!-- Arrow markers -->
        <defs>
          <marker id="coneArrowPurple" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#9b59b6" />
          </marker>
        </defs>

        <!-- Labels -->
        <text x="10" y="16" class="region-label">{{ t('Air', '공기') }}</text>
        <text :x="svgW - 10" :y="svgH - 8" text-anchor="end" class="region-label">{{ t('Silicon', '실리콘') }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

const cra = ref(10)
const halfAngle = ref(14.5)

const svgW = 400
const svgH = 340

// Layout constants
const mlY = 60
const mlH = 30
const planH = 25
const cfY = mlY + mlH + planH
const cfH = 30
const pdY = cfY + cfH + 15
const pdH = 50
const pixelW = 140
const pixelCenterX = svgW / 2
const pixelLeft = pixelCenterX - pixelW / 2
const pixelRight = pixelCenterX + pixelW / 2
const pdW = pixelW * 0.7
const pdLeft = pixelCenterX - pdW / 2

// Stack height in um (for shift calculation)
const stackHeight = 2.5 // um from microlens to photodiode

const craRad = computed(() => cra.value * Math.PI / 180)
const halfAngleRad = computed(() => halfAngle.value * Math.PI / 180)

// Microlens shift = stackHeight * tan(CRA) -- in um
const mlShift = computed(() => stackHeight * Math.tan(craRad.value))

// Microlens shift in SVG pixels (map um to pixels)
// pixel pitch ~ 1.0um ~ pixelW px
const mlShiftPx = computed(() => mlShift.value * (pixelW / 1.0))
const mlCenterX = computed(() => pixelCenterX - mlShiftPx.value)

// Microlens path (arc centered at mlCenterX)
const microlensPath = computed(() => {
  const cx = mlCenterX.value
  const halfW = pixelW / 2 - 2
  const left = cx - halfW
  const right = cx + halfW
  const top = mlY
  const bot = mlY + mlH
  return `M ${left} ${bot} Q ${cx} ${top - 5} ${right} ${bot} Z`
})

// f/# equivalent
const fNumber = computed(() => {
  const sinHA = Math.sin(halfAngleRad.value)
  return sinHA > 0 ? 1 / (2 * sinHA) : 99
})

// Solid angle
const solidAngle = computed(() => {
  return 2 * Math.PI * (1 - Math.cos(halfAngleRad.value))
})

// Generate rays within the cone
const numRays = 7
const rays = computed(() => {
  const result = []
  for (let i = 0; i < numRays; i++) {
    const frac = numRays > 1 ? (i / (numRays - 1)) * 2 - 1 : 0 // -1 to 1
    const angle = craRad.value + frac * halfAngleRad.value
    const isChief = Math.abs(frac) < 0.01

    // Ray starts above the SVG, arrives at microlens
    const entryX = mlCenterX.value
    const entryY = mlY + mlH * 0.5
    const rayLength = 80
    const x1 = entryX - rayLength * Math.sin(angle)
    const y1 = entryY - rayLength * Math.cos(angle)

    // Refracted ray after microlens (Snell-like bending toward center)
    const nAir = 1.0
    const nLens = 1.56
    const sinRefracted = (nAir / nLens) * Math.sin(angle)
    const refractedAngle = Math.abs(sinRefracted) < 1 ? Math.asin(sinRefracted) : angle * 0.6

    const botLength = pdY + 6 - entryY
    const xBot = entryX + botLength * Math.sin(refractedAngle)
    const yBot = pdY + 6

    // Color based on position in cone
    const hue = 30 + 20 * Math.abs(frac)
    const color = isChief ? '#f39c12' : `hsl(${hue}, 80%, 55%)`

    result.push({ x1, y1, x2: entryX, y2: entryY, xBot, yBot, isChief, color })
  }
  return result
})

// Focus spot position
const focusX = computed(() => {
  const nLens = 1.56
  const sinRef = (1.0 / nLens) * Math.sin(craRad.value)
  const refAngle = Math.abs(sinRef) < 1 ? Math.asin(sinRef) : craRad.value * 0.6
  const dist = pdY + 6 - (mlY + mlH * 0.5)
  return mlCenterX.value + dist * Math.sin(refAngle)
})

const focusSpotR = computed(() => {
  return 4 + halfAngle.value * 0.3
})
</script>

<style scoped>
.cone-illum-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.cone-illum-container h4 {
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
  margin-bottom: 16px;
}
.slider-group {
  flex: 1;
  min-width: 160px;
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
.svg-wrapper {
  display: flex;
  justify-content: center;
}
.cone-svg {
  width: 100%;
  max-width: 420px;
}
.region-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.small-label {
  font-size: 8px;
  fill: var(--vp-c-text-2);
}
.shift-label {
  font-size: 8px;
  fill: #9b59b6;
  font-weight: 600;
}
.focus-label {
  font-size: 8px;
  fill: #f39c12;
  font-weight: 600;
}
</style>
