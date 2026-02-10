<template>
  <div class="snell-container">
    <h4>{{ t("Interactive Snell's Law Visualizer", '대화형 스넬의 법칙 시각화') }}</h4>
    <p class="component-description">
      {{ t('Adjust refractive indices and incidence angle to explore refraction, reflection, and total internal reflection.', '굴절률과 입사각을 조절하여 굴절, 반사 및 전반사를 탐색해 보세요.') }}
    </p>

    <div class="inputs-row">
      <div class="input-group">
        <label for="snell-n1">n<sub>1</sub> ({{ t('incident', '입사') }})</label>
        <input id="snell-n1" type="number" v-model.number="n1" min="1.0" max="5.0" step="0.01" />
        <span class="preset-hint">Air=1.0, Water=1.33, Glass=1.5</span>
      </div>
      <div class="input-group">
        <label for="snell-n2">n<sub>2</sub> ({{ t('transmitted', '투과') }})</label>
        <input id="snell-n2" type="number" v-model.number="n2" min="1.0" max="5.0" step="0.01" />
        <span class="preset-hint">SiO2=1.46, Si3N4=2.0, Si=3.5</span>
      </div>
    </div>

    <div class="slider-section">
      <label for="snell-angle">
        {{ t('Incident angle', '입사각') }}: <strong>{{ thetaI.toFixed(1) }}&deg;</strong>
      </label>
      <input
        id="snell-angle"
        type="range"
        min="0"
        max="89"
        step="0.5"
        v-model.number="thetaI"
        class="angle-range"
      />
    </div>

    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">&theta;<sub>i</sub> ({{ t('incident', '입사') }})</div>
        <div class="result-value">{{ thetaI.toFixed(1) }}&deg;</div>
      </div>
      <div class="result-card">
        <div class="result-label">&theta;<sub>r</sub> ({{ t('refracted', '굴절') }})</div>
        <div class="result-value" :class="{ tir: isTIR }">
          {{ isTIR ? 'TIR' : thetaR.toFixed(1) + '\u00B0' }}
        </div>
      </div>
      <div class="result-card" v-if="hasCritical">
        <div class="result-label">&theta;<sub>c</sub> ({{ t('critical', '임계') }})</div>
        <div class="result-value critical">{{ criticalAngle.toFixed(1) }}&deg;</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t("Snell's law", '스넬의 법칙') }}</div>
        <div class="result-value formula-val">n<sub>1</sub> sin&theta;<sub>i</sub> = n<sub>2</sub> sin&theta;<sub>r</sub></div>
      </div>
    </div>

    <div v-if="isTIR" class="tir-notice">
      {{ t('Total Internal Reflection — all light is reflected at the interface.', '전반사 — 모든 빛이 계면에서 반사됩니다.') }}
    </div>

    <div class="svg-wrapper">
      <svg viewBox="0 0 400 300" class="snell-svg">
        <!-- Medium 1 (top) -->
        <rect x="0" y="0" width="400" height="150" fill="#e8f4fd" opacity="0.5" />
        <!-- Medium 2 (bottom) -->
        <rect x="0" y="150" width="400" height="150" fill="#fde8e8" opacity="0.4" />

        <!-- Interface line -->
        <line x1="0" y1="150" x2="400" y2="150" stroke="var(--vp-c-text-3)" stroke-width="2" />

        <!-- Normal line (vertical dashed) -->
        <line x1="200" y1="15" x2="200" y2="285" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="5,4" />
        <text x="208" y="25" class="normal-label">{{ t('Normal', '법선') }}</text>

        <!-- Medium labels -->
        <text x="15" y="30" class="medium-label">n<tspan font-size="9" dy="2">1</tspan><tspan dy="-2"> = {{ n1 }}</tspan></text>
        <text x="15" y="280" class="medium-label">n<tspan font-size="9" dy="2">2</tspan><tspan dy="-2"> = {{ n2 }}</tspan></text>

        <!-- Incident ray -->
        <line
          :x1="200 - rayLen * Math.sin(thetaIRad)"
          :y1="150 - rayLen * Math.cos(thetaIRad)"
          x2="200"
          y2="150"
          stroke="#f39c12"
          stroke-width="2.5"
          marker-end="url(#snellArrowOrange)"
        />

        <!-- Reflected ray -->
        <line
          x1="200"
          y1="150"
          :x2="200 + rayLen * Math.sin(thetaIRad)"
          :y2="150 - rayLen * Math.cos(thetaIRad)"
          stroke="#e74c3c"
          :stroke-width="isTIR ? 2.5 : 1.5"
          :opacity="isTIR ? 1 : 0.6"
          marker-end="url(#snellArrowRed)"
        />

        <!-- Refracted ray (not shown during TIR) -->
        <line
          v-if="!isTIR"
          x1="200"
          y1="150"
          :x2="200 + rayLen * Math.sin(thetaRRad)"
          :y2="150 + rayLen * Math.cos(thetaRRad)"
          stroke="#3498db"
          stroke-width="2.5"
          marker-end="url(#snellArrowBlue)"
        />

        <!-- Incident angle arc -->
        <path
          v-if="thetaI > 2"
          :d="arcPath(200, 150, 35, 0, -thetaI, true)"
          fill="none"
          stroke="#f39c12"
          stroke-width="1.2"
        />
        <text
          v-if="thetaI > 5"
          :x="200 - 45 * Math.sin(thetaIRad / 2)"
          :y="150 - 45 * Math.cos(thetaIRad / 2)"
          text-anchor="middle"
          class="angle-label incident-label"
        >&theta;<tspan font-size="7" dy="2">i</tspan></text>

        <!-- Reflected angle arc -->
        <path
          v-if="thetaI > 2"
          :d="arcPath(200, 150, 30, 0, thetaI, true)"
          fill="none"
          stroke="#e74c3c"
          stroke-width="1"
          stroke-dasharray="3,2"
        />

        <!-- Refracted angle arc -->
        <path
          v-if="!isTIR && thetaR > 2"
          :d="arcPath(200, 150, 35, 180, 180 + thetaR, false)"
          fill="none"
          stroke="#3498db"
          stroke-width="1.2"
        />
        <text
          v-if="!isTIR && thetaR > 5"
          :x="200 + 45 * Math.sin(thetaRRad / 2)"
          :y="150 + 45 * Math.cos(thetaRRad / 2)"
          text-anchor="middle"
          class="angle-label refracted-label"
        >&theta;<tspan font-size="7" dy="2">r</tspan></text>

        <!-- TIR label -->
        <text
          v-if="isTIR"
          x="200"
          y="200"
          text-anchor="middle"
          class="tir-label"
        >{{ t('Total Internal Reflection', '전반사') }}</text>

        <!-- Arrow markers -->
        <defs>
          <marker id="snellArrowOrange" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#f39c12" />
          </marker>
          <marker id="snellArrowRed" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#e74c3c" />
          </marker>
          <marker id="snellArrowBlue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#3498db" />
          </marker>
        </defs>

        <!-- Legend -->
        <line x1="290" y1="18" x2="310" y2="18" stroke="#f39c12" stroke-width="2" />
        <text x="315" y="22" class="legend-label">{{ t('Incident', '입사') }}</text>
        <line x1="290" y1="33" x2="310" y2="33" stroke="#e74c3c" stroke-width="2" />
        <text x="315" y="37" class="legend-label">{{ t('Reflected', '반사') }}</text>
        <line x1="290" y1="48" x2="310" y2="48" stroke="#3498db" stroke-width="2" />
        <text x="315" y="52" class="legend-label">{{ t('Refracted', '굴절') }}</text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const n1 = ref(1.5)
const n2 = ref(1.0)
const thetaI = ref(30)
const rayLen = 120

const thetaIRad = computed(() => (thetaI.value * Math.PI) / 180)

const hasCritical = computed(() => n1.value > n2.value)

const criticalAngle = computed(() => {
  if (!hasCritical.value) return 90
  return (Math.asin(n2.value / n1.value) * 180) / Math.PI
})

const isTIR = computed(() => {
  return hasCritical.value && thetaI.value >= criticalAngle.value
})

const thetaR = computed(() => {
  const sinR = (n1.value * Math.sin(thetaIRad.value)) / n2.value
  if (Math.abs(sinR) > 1) return 90
  return (Math.asin(sinR) * 180) / Math.PI
})

const thetaRRad = computed(() => (thetaR.value * Math.PI) / 180)

function arcPath(cx, cy, r, startAngleDeg, endAngleDeg, aboveInterface) {
  // For above interface: angles measured from vertical (normal), going clockwise for positive
  // For below interface: angles measured from downward normal
  const toRad = (d) => (d * Math.PI) / 180

  let sx, sy, ex, ey

  if (aboveInterface) {
    // startAngleDeg and endAngleDeg are angles from the upward normal
    // Negative = left of normal, positive = right of normal
    const sRad = toRad(startAngleDeg)
    const eRad = toRad(endAngleDeg)
    sx = cx + r * Math.sin(sRad)
    sy = cy - r * Math.cos(sRad)
    ex = cx + r * Math.sin(eRad)
    ey = cy - r * Math.cos(eRad)
  } else {
    // Below interface
    const sRad = toRad(startAngleDeg)
    const eRad = toRad(endAngleDeg)
    sx = cx + r * Math.cos(sRad)
    sy = cy - r * Math.sin(sRad)
    ex = cx + r * Math.cos(eRad)
    ey = cy - r * Math.sin(eRad)
  }

  const largeArc = Math.abs(endAngleDeg - startAngleDeg) > 180 ? 1 : 0
  const sweep = endAngleDeg > startAngleDeg ? 1 : 0
  return `M ${sx.toFixed(1)} ${sy.toFixed(1)} A ${r} ${r} 0 ${largeArc} ${sweep} ${ex.toFixed(1)} ${ey.toFixed(1)}`
}
</script>

<style scoped>
.snell-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.snell-container h4 {
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
  min-width: 160px;
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
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}
.result-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1em;
  font-family: var(--vp-font-family-mono);
}
.result-value.tir {
  color: #e74c3c;
}
.result-value.critical {
  color: #9b59b6;
}
.formula-val {
  font-size: 0.82em;
}
.tir-notice {
  background: #f8d7da;
  color: #842029;
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 0.9em;
  margin-bottom: 16px;
  text-align: center;
}
.dark .tir-notice {
  background: #4a1c21;
  color: #f5c6cb;
}
.svg-wrapper {
  margin-top: 8px;
}
.snell-svg {
  width: 100%;
  max-width: 460px;
  display: block;
  margin: 0 auto;
}
.medium-label {
  font-size: 12px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.normal-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.angle-label {
  font-size: 10px;
  font-weight: 600;
}
.incident-label {
  fill: #f39c12;
}
.refracted-label {
  fill: #3498db;
}
.tir-label {
  font-size: 14px;
  fill: #e74c3c;
  font-weight: 700;
}
.legend-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
</style>
