<template>
  <div class="fresnel-container">
    <h4>{{ t('Interactive Fresnel Calculator', '인터랙티브 프레넬 계산기') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how reflection and transmission depend on refractive indices and incidence angle.',
        '반사와 투과가 굴절률과 입사각에 따라 어떻게 달라지는지 살펴보세요.'
      ) }}
    </p>

    <div class="inputs-row">
      <div class="input-group">
        <label for="n1-input">n<sub>1</sub> ({{ t('incident medium', '입사 매질') }})</label>
        <input id="n1-input" type="number" v-model.number="n1" min="1.0" max="5.0" step="0.01" />
        <span class="preset-hint">Air=1.0, Glass=1.5, Water=1.33</span>
      </div>
      <div class="input-group">
        <label for="n2-input">n<sub>2</sub> ({{ t('transmitted medium', '투과 매질') }})</label>
        <input id="n2-input" type="number" v-model.number="n2" min="1.0" max="5.0" step="0.01" />
        <span class="preset-hint">SiO<sub>2</sub>=1.46, Si<sub>3</sub>N<sub>4</sub>=2.0, Si=3.5</span>
      </div>
    </div>

    <div class="slider-section">
      <label for="angle-input">
        {{ t('Angle of incidence', '입사각') }}: <strong>{{ angle.toFixed(1) }}&deg;</strong>
      </label>
      <input
        id="angle-input"
        type="range"
        min="0"
        max="89.5"
        step="0.5"
        v-model.number="angle"
        class="angle-range"
      />
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">R<sub>s</sub> (TE)</div>
        <div class="result-value">{{ (Rs * 100).toFixed(2) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">R<sub>p</sub> (TM)</div>
        <div class="result-value">{{ (Rp * 100).toFixed(2) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">R<sub>avg</sub> ({{ t('unpolarized', '비편광') }})</div>
        <div class="result-value highlight">{{ (Ravg * 100).toFixed(2) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Brewster Angle', '브루스터 각') }}</div>
        <div class="result-value brewster">{{ brewsterAngle.toFixed(2) }}&deg;</div>
      </div>
    </div>

    <div v-if="hasTIR" class="tir-notice">
      {{ t('Critical angle', '임계각') }}: {{ criticalAngle.toFixed(2) }}&deg;
      <span v-if="angle >= criticalAngle">&mdash; <strong>{{ t('Total Internal Reflection!', '전반사!') }}</strong></span>
    </div>

    <div class="chart-section">
      <svg viewBox="0 0 460 280" class="fresnel-svg">
        <!-- Axes -->
        <line x1="50" y1="20" x2="50" y2="240" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line x1="50" y1="240" x2="440" y2="240" stroke="var(--vp-c-text-3)" stroke-width="1" />

        <!-- Y-axis label -->
        <text x="15" y="130" text-anchor="middle" transform="rotate(-90, 15, 130)" class="axis-label">{{ t('Reflectance', '반사율') }}</text>

        <!-- Y-axis ticks -->
        <template v-for="tick in [0, 0.25, 0.5, 0.75, 1.0]" :key="tick">
          <line :x1="46" :y1="240 - tick * 220" :x2="50" :y2="240 - tick * 220" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="42" :y="244 - tick * 220" text-anchor="end" class="tick-label">{{ (tick * 100).toFixed(0) }}%</text>
          <line x1="50" :y1="240 - tick * 220" x2="440" :y2="240 - tick * 220" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
        </template>

        <!-- X-axis label -->
        <text x="245" y="272" text-anchor="middle" class="axis-label">{{ t('Angle of incidence', '입사각') }} (&deg;)</text>

        <!-- X-axis ticks -->
        <template v-for="a in [0, 15, 30, 45, 60, 75, 90]" :key="a">
          <line :x1="50 + (a / 90) * 390" y1="240" :x2="50 + (a / 90) * 390" y2="244" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <text :x="50 + (a / 90) * 390" y="256" text-anchor="middle" class="tick-label">{{ a }}</text>
        </template>

        <!-- Rs curve -->
        <path :d="rsPath" fill="none" stroke="#e74c3c" stroke-width="2" />
        <!-- Rp curve -->
        <path :d="rpPath" fill="none" stroke="#3498db" stroke-width="2" />
        <!-- Ravg curve -->
        <path :d="ravgPath" fill="none" stroke="#2ecc71" stroke-width="2" stroke-dasharray="5,3" />

        <!-- Brewster angle marker -->
        <line
          :x1="50 + (brewsterAngle / 90) * 390"
          y1="20"
          :x2="50 + (brewsterAngle / 90) * 390"
          y2="240"
          stroke="#f39c12"
          stroke-width="1.5"
          stroke-dasharray="4,3"
        />
        <text
          :x="50 + (brewsterAngle / 90) * 390"
          y="16"
          text-anchor="middle"
          class="brewster-label"
        >&theta;<tspan font-size="7" dy="2">B</tspan></text>

        <!-- Critical angle marker if applicable -->
        <line
          v-if="hasTIR"
          :x1="50 + (criticalAngle / 90) * 390"
          y1="20"
          :x2="50 + (criticalAngle / 90) * 390"
          y2="240"
          stroke="#9b59b6"
          stroke-width="1.5"
          stroke-dasharray="4,3"
        />
        <text
          v-if="hasTIR"
          :x="50 + (criticalAngle / 90) * 390"
          y="16"
          text-anchor="middle"
          class="critical-label"
        >&theta;<tspan font-size="7" dy="2">c</tspan></text>

        <!-- Current angle marker -->
        <circle
          :cx="50 + (angle / 90) * 390"
          :cy="240 - Rs * 220"
          r="4"
          fill="#e74c3c"
        />
        <circle
          :cx="50 + (angle / 90) * 390"
          :cy="240 - Rp * 220"
          r="4"
          fill="#3498db"
        />

        <!-- Legend -->
        <line x1="300" y1="30" x2="320" y2="30" stroke="#e74c3c" stroke-width="2" />
        <text x="325" y="34" class="legend-label">R<tspan font-size="8" dy="2">s</tspan><tspan dy="-2"> (TE)</tspan></text>
        <line x1="300" y1="48" x2="320" y2="48" stroke="#3498db" stroke-width="2" />
        <text x="325" y="52" class="legend-label">R<tspan font-size="8" dy="2">p</tspan><tspan dy="-2"> (TM)</tspan></text>
        <line x1="300" y1="66" x2="320" y2="66" stroke="#2ecc71" stroke-width="2" stroke-dasharray="5,3" />
        <text x="325" y="70" class="legend-label">R<tspan font-size="8" dy="2">avg</tspan></text>
      </svg>
    </div>

    <!-- Ray diagram -->
    <div class="ray-section">
      <svg viewBox="0 0 300 180" class="ray-svg">
        <!-- Interface line -->
        <line x1="0" y1="90" x2="300" y2="90" stroke="var(--vp-c-text-3)" stroke-width="1.5" />

        <!-- Medium labels -->
        <rect x="5" y="5" width="50" height="20" rx="3" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" />
        <text x="30" y="18" text-anchor="middle" class="medium-label">n<tspan font-size="8" dy="2">1</tspan><tspan dy="-2"> = {{ n1 }}</tspan></text>
        <rect x="5" y="155" width="50" height="20" rx="3" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" />
        <text x="30" y="168" text-anchor="middle" class="medium-label">n<tspan font-size="8" dy="2">2</tspan><tspan dy="-2"> = {{ n2 }}</tspan></text>

        <!-- Normal line -->
        <line x1="150" y1="10" x2="150" y2="170" stroke="var(--vp-c-text-3)" stroke-width="0.5" stroke-dasharray="4,4" />

        <!-- Incident ray -->
        <line
          :x1="150 - 75 * Math.sin(angleRad)"
          :y1="90 - 75 * Math.cos(angleRad)"
          x2="150"
          y2="90"
          stroke="#f39c12"
          stroke-width="2.5"
          marker-end="url(#arrowOrange)"
        />

        <!-- Reflected ray -->
        <line
          x1="150"
          y1="90"
          :x2="150 + 75 * Math.sin(angleRad)"
          :y2="90 - 75 * Math.cos(angleRad)"
          stroke="#e74c3c"
          :stroke-width="1 + Ravg * 2"
          :opacity="0.4 + Ravg * 0.6"
          marker-end="url(#arrowRed)"
        />

        <!-- Transmitted ray (only if not TIR) -->
        <line
          v-if="!isTIR"
          x1="150"
          y1="90"
          :x2="150 + 75 * Math.sin(transmittedAngleRad)"
          :y2="90 + 75 * Math.cos(transmittedAngleRad)"
          stroke="#3498db"
          :stroke-width="1 + (1 - Ravg) * 2"
          :opacity="0.4 + (1 - Ravg) * 0.6"
          marker-end="url(#arrowBlue)"
        />

        <!-- Arrow markers -->
        <defs>
          <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#f39c12" />
          </marker>
          <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#e74c3c" />
          </marker>
          <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#3498db" />
          </marker>
        </defs>

        <!-- Angle arcs -->
        <path
          :d="incidentArc"
          fill="none"
          stroke="#f39c12"
          stroke-width="1"
        />
        <text
          :x="150 - 30 * Math.sin(angleRad / 2)"
          :y="90 - 30 * Math.cos(angleRad / 2)"
          text-anchor="middle"
          class="angle-text"
        >&theta;<tspan font-size="7" dy="2">i</tspan></text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const n1 = ref(1.0)
const n2 = ref(1.5)
const angle = ref(0)

const angleRad = computed(() => (angle.value * Math.PI) / 180)

const brewsterAngle = computed(() => {
  return (Math.atan(n2.value / n1.value) * 180) / Math.PI
})

const hasTIR = computed(() => n1.value > n2.value)

const criticalAngle = computed(() => {
  if (!hasTIR.value) return 90
  return (Math.asin(n2.value / n1.value) * 180) / Math.PI
})

const isTIR = computed(() => {
  return hasTIR.value && angle.value >= criticalAngle.value
})

function fresnelCalc(angleDeg) {
  const thetaI = (angleDeg * Math.PI) / 180
  const cosI = Math.cos(thetaI)
  const sinI = Math.sin(thetaI)

  const sinT2 = ((n1.value * sinI) / n2.value) ** 2
  if (sinT2 > 1) {
    // Total internal reflection
    return { rs: 1, rp: 1, ravg: 1 }
  }

  const cosT = Math.sqrt(1 - sinT2)
  const rs = ((n1.value * cosI - n2.value * cosT) / (n1.value * cosI + n2.value * cosT)) ** 2
  const rp = ((n2.value * cosI - n1.value * cosT) / (n2.value * cosI + n1.value * cosT)) ** 2

  return { rs, rp, ravg: (rs + rp) / 2 }
}

const Rs = computed(() => fresnelCalc(angle.value).rs)
const Rp = computed(() => fresnelCalc(angle.value).rp)
const Ravg = computed(() => fresnelCalc(angle.value).ravg)

const transmittedAngleRad = computed(() => {
  const sinT = (n1.value * Math.sin(angleRad.value)) / n2.value
  if (Math.abs(sinT) > 1) return 0
  return Math.asin(sinT)
})

function buildCurvePath(getFn) {
  const steps = 180
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const a = (i / steps) * 89.5
    const val = getFn(a)
    const x = 50 + (a / 90) * 390
    const y = 240 - Math.min(val, 1) * 220
    d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return d
}

const rsPath = computed(() => buildCurvePath((a) => fresnelCalc(a).rs))
const rpPath = computed(() => buildCurvePath((a) => fresnelCalc(a).rp))
const ravgPath = computed(() => buildCurvePath((a) => fresnelCalc(a).ravg))

const incidentArc = computed(() => {
  const r = 25
  const startX = 150
  const startY = 90 - r
  const endX = 150 - r * Math.sin(angleRad.value)
  const endY = 90 - r * Math.cos(angleRad.value)
  const largeArc = angle.value > 180 ? 1 : 0
  return `M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 0 ${endX} ${endY}`
})
</script>

<style scoped>
.fresnel-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.fresnel-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.inputs-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.input-group {
  flex: 1;
  min-width: 180px;
}
.input-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.9em;
  font-weight: 600;
}
.input-group input[type="number"] {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 1em;
  font-family: var(--vp-font-family-mono);
}
.preset-hint {
  display: block;
  font-size: 0.75em;
  color: var(--vp-c-text-3);
  margin-top: 3px;
}
.slider-section {
  margin-bottom: 16px;
}
.slider-section label {
  display: block;
  margin-bottom: 6px;
  font-size: 0.95em;
}
.angle-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.angle-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.angle-range::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.05em;
  font-family: var(--vp-font-family-mono);
}
.result-value.highlight {
  color: var(--vp-c-brand-1);
}
.result-value.brewster {
  color: #f39c12;
}
.tir-notice {
  background: #f8d7da;
  color: #842029;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 0.9em;
  margin-bottom: 16px;
}
.dark .tir-notice {
  background: #4a1c21;
  color: #f5c6cb;
}
.chart-section {
  margin-bottom: 16px;
}
.fresnel-svg {
  width: 100%;
  max-width: 520px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 11px;
  fill: var(--vp-c-text-2);
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.legend-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.brewster-label {
  font-size: 10px;
  fill: #f39c12;
  font-weight: 600;
}
.critical-label {
  font-size: 10px;
  fill: #9b59b6;
  font-weight: 600;
}
.ray-section {
  margin-top: 8px;
}
.ray-svg {
  width: 100%;
  max-width: 350px;
  display: block;
  margin: 0 auto;
}
.medium-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.angle-text {
  font-size: 10px;
  fill: #f39c12;
}
</style>
