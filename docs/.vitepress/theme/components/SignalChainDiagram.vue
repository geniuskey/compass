<template>
  <div class="signal-chain-container">
    <h4>{{ t('Interactive Signal Chain Diagram', '인터랙티브 신호 체인 다이어그램') }}</h4>
    <p class="component-description">
      {{ t('Trace the spectrum through each stage of the signal chain. Select illuminant and scene to see how the spectrum transforms at each stage.', '신호 체인의 각 단계를 통해 스펙트럼을 추적합니다. 광원과 장면을 선택하여 각 단계에서 스펙트럼이 어떻게 변환되는지 확인하세요.') }}
    </p>

    <div class="controls-row">
      <div class="select-group">
        <label>{{ t('Select Illuminant', '광원 선택') }}:</label>
        <select v-model="selectedIlluminant" class="ctrl-select">
          <option value="d65">CIE D65 ({{ t('Daylight', '주광') }})</option>
          <option value="a">CIE A ({{ t('Incandescent', '백열등') }})</option>
          <option value="led">LED White (5000K)</option>
        </select>
      </div>
      <div class="select-group">
        <label>{{ t('Select Scene', '장면 선택') }}:</label>
        <select v-model="selectedScene" class="ctrl-select">
          <option value="grey18">{{ t('18% Grey', '18% 그레이') }}</option>
          <option value="red">{{ t('Red Patch', '빨간색 패치') }}</option>
          <option value="green">{{ t('Green Patch', '초록색 패치') }}</option>
          <option value="blue">{{ t('Blue Patch', '파란색 패치') }}</option>
        </select>
      </div>
    </div>

    <!-- Flow diagram -->
    <div class="flow-row">
      <div
        v-for="(stage, idx) in stages"
        :key="stage.id"
        class="flow-block"
        :class="{ active: hoveredStage === stage.id }"
        @mouseenter="hoveredStage = stage.id"
        @mouseleave="hoveredStage = null"
      >
        <div class="flow-icon">{{ stage.icon }}</div>
        <div class="flow-label">{{ stage.label }}</div>
        <div class="flow-sub">{{ stage.sub }}</div>
        <div v-if="idx < stages.length - 1" class="flow-arrow">&rarr;</div>
      </div>
    </div>

    <!-- Stage detail tooltip -->
    <div v-if="hoveredStage" class="stage-detail">
      <strong>{{ stageDetail.title }}</strong>: {{ stageDetail.description }}
    </div>

    <!-- Info cards -->
    <div class="info-row">
      <div class="info-card" style="border-left: 3px solid #e74c3c;">
        <span class="info-label">{{ t('Red signal', '빨간색 신호') }}:</span>
        <span class="info-value">{{ signalR.toFixed(2) }}</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #27ae60;">
        <span class="info-label">{{ t('Green signal', '초록색 신호') }}:</span>
        <span class="info-value">{{ signalG.toFixed(2) }}</span>
      </div>
      <div class="info-card" style="border-left: 3px solid #3498db;">
        <span class="info-label">{{ t('Blue signal', '파란색 신호') }}:</span>
        <span class="info-value">{{ signalB.toFixed(2) }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('R/G Ratio', 'R/G 비율') }}:</span>
        <span class="info-value">{{ rgRatio }}</span>
      </div>
      <div class="info-card">
        <span class="info-label">{{ t('B/G Ratio', 'B/G 비율') }}:</span>
        <span class="info-value">{{ bgRatio }}</span>
      </div>
    </div>

    <!-- Spectrum chart -->
    <div class="svg-wrapper">
      <svg
        :viewBox="`0 0 ${svgW} ${svgH}`"
        class="chain-svg"
        @mousemove="onMouseMove"
        @mouseleave="onMouseLeave"
      >
        <defs>
          <linearGradient id="scVisSpectrum" x1="0" y1="0" x2="1" y2="0">
            <stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" />
          </linearGradient>
        </defs>

        <!-- Visible spectrum bar -->
        <rect
          :x="pad.left"
          :y="pad.top + plotH + 2"
          :width="plotW"
          height="8"
          fill="url(#scVisSpectrum)"
          rx="2"
        />

        <!-- Grid lines -->
        <line
          v-for="tick in yTicks"
          :key="'yg' + tick"
          :x1="pad.left"
          :y1="yScale(tick)"
          :x2="pad.left + plotW"
          :y2="yScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />
        <line
          v-for="tick in xTicks"
          :key="'xg' + tick"
          :x1="xScale(tick)"
          :y1="pad.top"
          :x2="xScale(tick)"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-divider)"
          stroke-width="0.5"
          stroke-dasharray="3,3"
        />

        <!-- Axes -->
        <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <!-- Y-axis labels -->
        <text
          v-for="tick in yTicks"
          :key="'yl' + tick"
          :x="pad.left - 6"
          :y="yScale(tick) + 3"
          text-anchor="end"
          class="axis-label"
        >{{ tick.toFixed(1) }}</text>

        <!-- X-axis labels -->
        <text
          v-for="tick in xTicks"
          :key="'xl' + tick"
          :x="xScale(tick)"
          :y="pad.top + plotH + 24"
          text-anchor="middle"
          class="axis-label"
        >{{ tick }}</text>

        <!-- Axis titles -->
        <text :x="pad.left + plotW / 2" :y="svgH - 2" text-anchor="middle" class="axis-title">Wavelength (nm)</text>
        <text :x="12" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 12, ${pad.top + plotH / 2})`">Relative Intensity</text>

        <!-- Stage 1: Source spectrum -->
        <path :d="sourcePath" fill="none" stroke="#8e44ad" stroke-width="1.5" opacity="0.5" />

        <!-- Stage 2: After scene -->
        <path :d="afterScenePath" fill="none" stroke="#e67e22" stroke-width="1.5" opacity="0.5" />

        <!-- Stage 3: After lens -->
        <path :d="afterLensPath" fill="none" stroke="#2980b9" stroke-width="1.5" opacity="0.5" />

        <!-- Stage 4: After IR filter (main curve, emphasized) -->
        <path :d="afterIRPath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" opacity="0.9" />
        <path :d="afterIRArea" fill="var(--vp-c-brand-1)" opacity="0.06" />

        <!-- Hover crosshair -->
        <template v-if="hoverWl !== null">
          <line :x1="xScale(hoverWl)" :y1="pad.top" :x2="xScale(hoverWl)" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
          <rect :x="tooltipX" :y="pad.top + 4" width="130" height="58" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
          <text :x="tooltipX + 6" :y="pad.top + 16" class="tooltip-text">{{ hoverWl }} nm</text>
          <text :x="tooltipX + 6" :y="pad.top + 28" class="tooltip-text" fill="#8e44ad">Source: {{ hoverSource.toFixed(3) }}</text>
          <text :x="tooltipX + 6" :y="pad.top + 40" class="tooltip-text" fill="#e67e22">+Scene: {{ hoverScene.toFixed(3) }}</text>
          <text :x="tooltipX + 6" :y="pad.top + 52" class="tooltip-text" fill="var(--vp-c-brand-1)">+IR: {{ hoverIR.toFixed(3) }}</text>
        </template>

        <!-- Legend -->
        <g :transform="`translate(${pad.left + plotW - 110}, ${pad.top + 8})`">
          <line x1="0" y1="6" x2="14" y2="6" stroke="#8e44ad" stroke-width="1.5" opacity="0.5" />
          <text x="18" y="10" class="legend-label">Source</text>
          <line x1="0" y1="20" x2="14" y2="20" stroke="#e67e22" stroke-width="1.5" opacity="0.5" />
          <text x="18" y="24" class="legend-label">+Scene</text>
          <line x1="0" y1="34" x2="14" y2="34" stroke="#2980b9" stroke-width="1.5" opacity="0.5" />
          <text x="18" y="38" class="legend-label">+Lens</text>
          <line x1="0" y1="48" x2="14" y2="48" stroke="var(--vp-c-brand-1)" stroke-width="2.5" />
          <text x="18" y="52" class="legend-label">+IR Filter</text>
        </g>
      </svg>
    </div>

    <!-- RGB Bar chart -->
    <div class="bar-section">
      <h5>{{ t('Per-Channel Signal (integrated with QE)', '채널별 신호 (QE 적분)') }}</h5>
      <div class="bar-chart">
        <div class="bar-group">
          <div class="bar-wrapper">
            <div class="bar red-bar" :style="{ height: barH(signalR) + 'px' }"></div>
          </div>
          <span class="bar-label">R</span>
          <span class="bar-value">{{ signalR.toFixed(2) }}</span>
        </div>
        <div class="bar-group">
          <div class="bar-wrapper">
            <div class="bar green-bar" :style="{ height: barH(signalG) + 'px' }"></div>
          </div>
          <span class="bar-label">G</span>
          <span class="bar-value">{{ signalG.toFixed(2) }}</span>
        </div>
        <div class="bar-group">
          <div class="bar-wrapper">
            <div class="bar blue-bar" :style="{ height: barH(signalB) + 'px' }"></div>
          </div>
          <span class="bar-label">B</span>
          <span class="bar-value">{{ signalB.toFixed(2) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
const { t } = useLocale()

const selectedIlluminant = ref('d65')
const selectedScene = ref('grey18')
const hoveredStage = ref(null)

const svgW = 560
const svgH = 300
const pad = { top: 20, right: 20, bottom: 40, left: 55 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

const wlMin = 380
const wlMax = 780
const wlStep = 5
const xTicks = [400, 450, 500, 550, 600, 650, 700, 750]
const yTicks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

function xScale(wl) {
  return pad.left + ((wl - wlMin) / (wlMax - wlMin)) * plotW
}

function yScale(val) {
  return pad.top + plotH - (val / 1.0) * plotH
}

// Flow stages
const stages = computed(() => [
  { id: 'source', icon: '\u2600', label: t('Light Source', '광원'), sub: 'L(\u03BB)' },
  { id: 'scene', icon: '\uD83C\uDFA8', label: t('Scene', '장면'), sub: 'R(\u03BB)' },
  { id: 'lens', icon: '\uD83D\uDD0D', label: t('Lens', '렌즈'), sub: 'T_lens(\u03BB)' },
  { id: 'ir', icon: '\uD83D\uDEE1', label: t('IR Filter', 'IR 필터'), sub: 'T_IR(\u03BB)' },
  { id: 'sensor', icon: '\u25A3', label: t('Sensor', '센서'), sub: 'QE(\u03BB)' },
  { id: 'signal', icon: '\u26A1', label: t('Signal', '신호'), sub: 'N_e' },
])

const stageDetails = computed(() => ({
  source: { title: t('Light Source', '광원'), description: t('Spectral power distribution of the illuminant. Different sources emit different spectral shapes.', '광원의 분광 출력 분포입니다. 광원에 따라 서로 다른 스펙트럼 형태를 방출합니다.') },
  scene: { title: t('Scene Reflectance', '장면 반사율'), description: t('The fraction of light reflected at each wavelength. Colored objects selectively reflect certain wavelengths.', '각 파장에서 반사되는 빛의 비율입니다. 색상이 있는 물체는 특정 파장을 선택적으로 반사합니다.') },
  lens: { title: t('Camera Lens', '카메라 렌즈'), description: t('Multi-element lens transmittance. Typically 85-95% across the visible spectrum for a well-designed module.', '다중 소자 렌즈 투과율입니다. 잘 설계된 모듈의 경우 가시광 스펙트럼 전역에서 일반적으로 85-95%입니다.') },
  ir: { title: t('IR Cut Filter', 'IR 차단 필터'), description: t('Blocks near-infrared light (>650nm) to prevent color distortion. Sharp cutoff with ~30nm transition.', '색 왜곡을 방지하기 위해 근적외선 빛(>650nm)을 차단합니다. ~30nm 전이 영역에서 급격히 차단합니다.') },
  sensor: { title: t('Image Sensor', '이미지 센서'), description: t('Pixel quantum efficiency determines how many photons are converted to electrons per color channel.', '픽셀 양자 효율은 각 색상 채널에서 얼마나 많은 광자가 전자로 변환되는지를 결정합니다.') },
  signal: { title: t('Electrical Signal', '전기 신호'), description: t('Final photoelectron count per channel, ready for analog-to-digital conversion.', '채널별 최종 광전자 수로, 아날로그-디지털 변환 준비가 완료된 상태입니다.') },
}))

const stageDetail = computed(() => {
  return stageDetails.value[hoveredStage.value] || { title: '', description: '' }
})

// Planck blackbody (relative)
function planck(wlNm, T) {
  const lam = wlNm * 1e-9
  const h = 6.626e-34; const c = 2.998e8; const kB = 1.381e-23
  return (2 * h * c * c / Math.pow(lam, 5)) / (Math.exp((h * c) / (lam * kB * T)) - 1)
}

// Illuminant SPDs
function getSourceSpd(wlNm) {
  if (selectedIlluminant.value === 'd65') return cieD65(wlNm)
  if (selectedIlluminant.value === 'a') return cieA(wlNm)
  return ledWhite(wlNm)
}

function cieD65(wlNm) {
  const d65Data = [
    [380, 0.50], [390, 0.56], [400, 0.83], [410, 0.91], [420, 0.93],
    [430, 0.87], [440, 1.05], [450, 1.17], [460, 1.18], [470, 1.15],
    [480, 1.14], [490, 1.08], [500, 1.09], [510, 1.09], [520, 1.07],
    [530, 1.07], [540, 1.04], [550, 1.04], [560, 1.00], [570, 0.96],
    [580, 0.96], [590, 0.89], [600, 0.90], [610, 0.88], [620, 0.84],
    [630, 0.83], [640, 0.78], [650, 0.74], [660, 0.69], [670, 0.70],
    [680, 0.64], [690, 0.55], [700, 0.57], [710, 0.52], [720, 0.49],
    [730, 0.46], [740, 0.47], [750, 0.44], [760, 0.34], [770, 0.38],
    [780, 0.35],
  ]
  for (let i = 0; i < d65Data.length - 1; i++) {
    if (wlNm >= d65Data[i][0] && wlNm <= d65Data[i + 1][0]) {
      const t = (wlNm - d65Data[i][0]) / (d65Data[i + 1][0] - d65Data[i][0])
      return d65Data[i][1] * (1 - t) + d65Data[i + 1][1] * t
    }
  }
  if (wlNm <= 380) return 0.50
  return 0.35
}

function cieA(wlNm) {
  const val = planck(wlNm, 2856)
  const ref560 = planck(560, 2856)
  return val / ref560
}

function ledWhite(wlNm) {
  const blue = 0.75 * Math.exp(-0.5 * Math.pow((wlNm - 450) / 12, 2))
  const phosphor = 0.95 * Math.exp(-0.5 * Math.pow((wlNm - 570) / 55, 2))
  return blue + phosphor
}

// Scene reflectance
function getSceneRefl(wlNm) {
  if (selectedScene.value === 'grey18') return 0.18
  if (selectedScene.value === 'red') {
    return 0.05 + 0.35 * Math.exp(-0.5 * Math.pow((wlNm - 620) / 35, 2))
  }
  if (selectedScene.value === 'green') {
    return 0.05 + 0.30 * Math.exp(-0.5 * Math.pow((wlNm - 540) / 30, 2))
  }
  if (selectedScene.value === 'blue') {
    return 0.05 + 0.20 * Math.exp(-0.5 * Math.pow((wlNm - 460) / 25, 2))
  }
  return 0.18
}

// Lens transmittance (relatively flat)
function lensT(wlNm) {
  // Slight dropoff at blue edge
  if (wlNm < 420) return 0.82 + 0.08 * ((wlNm - 380) / 40)
  return 0.90
}

// IR filter
function irFilter(wlNm) {
  return 1.0 / (1.0 + Math.exp((wlNm - 650) / 10))
}

// QE for R, G, B channels (simplified)
function qeR(wlNm) {
  return 0.55 * Math.exp(-0.5 * Math.pow((wlNm - 610) / 35, 2))
}
function qeG(wlNm) {
  return 0.70 * Math.exp(-0.5 * Math.pow((wlNm - 530) / 35, 2))
}
function qeB(wlNm) {
  return 0.60 * Math.exp(-0.5 * Math.pow((wlNm - 450) / 30, 2))
}

// Compute spectra at each stage
function computeSpectra() {
  const pts = []
  let maxSource = 0

  for (let wl = wlMin; wl <= wlMax; wl += wlStep) {
    const src = getSourceSpd(wl)
    if (src > maxSource) maxSource = src
    pts.push({ wl, source: src })
  }

  // Normalize source to 1
  for (const p of pts) {
    p.source /= maxSource
    p.afterScene = p.source * getSceneRefl(p.wl)
    p.afterLens = p.afterScene * lensT(p.wl)
    p.afterIR = p.afterLens * irFilter(p.wl)
  }

  // Normalize all stages to the same max (source peak = 1)
  return pts
}

const spectra = computed(() => computeSpectra())

// Compute integrated signals
const signalR = computed(() => {
  let sum = 0
  for (const p of spectra.value) {
    sum += p.afterIR * qeR(p.wl) * wlStep
  }
  return sum / 100 // Scale for display
})
const signalG = computed(() => {
  let sum = 0
  for (const p of spectra.value) {
    sum += p.afterIR * qeG(p.wl) * wlStep
  }
  return sum / 100
})
const signalB = computed(() => {
  let sum = 0
  for (const p of spectra.value) {
    sum += p.afterIR * qeB(p.wl) * wlStep
  }
  return sum / 100
})

const rgRatio = computed(() => {
  if (signalG.value === 0) return '--'
  return (signalR.value / signalG.value).toFixed(3)
})
const bgRatio = computed(() => {
  if (signalG.value === 0) return '--'
  return (signalB.value / signalG.value).toFixed(3)
})

function pathFromPoints(pts, key) {
  return pts.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L'
    return `${cmd}${xScale(p.wl).toFixed(1)},${yScale(p[key]).toFixed(1)}`
  }).join(' ')
}

function areaFromPoints(pts, key) {
  const line = pathFromPoints(pts, key)
  const last = pts[pts.length - 1]
  const first = pts[0]
  return line + ` L${xScale(last.wl).toFixed(1)},${yScale(0).toFixed(1)} L${xScale(first.wl).toFixed(1)},${yScale(0).toFixed(1)} Z`
}

const sourcePath = computed(() => pathFromPoints(spectra.value, 'source'))
const afterScenePath = computed(() => pathFromPoints(spectra.value, 'afterScene'))
const afterLensPath = computed(() => pathFromPoints(spectra.value, 'afterLens'))
const afterIRPath = computed(() => pathFromPoints(spectra.value, 'afterIR'))
const afterIRArea = computed(() => areaFromPoints(spectra.value, 'afterIR'))

// Bar chart helper
const maxSignal = computed(() => Math.max(signalR.value, signalG.value, signalB.value, 0.01))
function barH(val) {
  return Math.max(2, (val / maxSignal.value) * 80)
}

// Hover
const hoverWl = ref(null)
const hoverSource = ref(0)
const hoverScene = ref(0)
const hoverIR = ref(0)

function onMouseMove(event) {
  const svg = event.currentTarget
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const wl = wlMin + ((mouseX - pad.left) / plotW) * (wlMax - wlMin)
  if (wl >= wlMin && wl <= wlMax) {
    const snapped = Math.round(wl / wlStep) * wlStep
    hoverWl.value = snapped
    const match = spectra.value.find(p => p.wl === snapped)
    if (match) {
      hoverSource.value = match.source
      hoverScene.value = match.afterScene
      hoverIR.value = match.afterIR
    }
  } else {
    hoverWl.value = null
  }
}

function onMouseLeave() {
  hoverWl.value = null
}

const tooltipX = computed(() => {
  if (hoverWl.value === null) return 0
  const x = xScale(hoverWl.value)
  return x + 140 > svgW - pad.right ? x - 140 : x + 10
})

// Visible spectrum gradient
function wavelengthToCSS(wl) {
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
  return `rgb(${r}, ${g}, ${b})`
}

const spectrumStops = computed(() => {
  const stops = []
  for (let wl = wlMin; wl <= wlMax; wl += 20) {
    stops.push({
      offset: ((wl - wlMin) / (wlMax - wlMin) * 100) + '%',
      color: wavelengthToCSS(wl),
    })
  }
  return stops
})
</script>

<style scoped>
.signal-chain-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.signal-chain-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.signal-chain-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-2);
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
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-group label {
  font-size: 0.85em;
  font-weight: 600;
}
.ctrl-select {
  padding: 4px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.85em;
  cursor: pointer;
}
.flow-row {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.flow-block {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  cursor: pointer;
  transition: all 0.15s;
  min-width: 70px;
}
.flow-block:hover,
.flow-block.active {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}
.flow-icon {
  font-size: 1.3em;
  margin-bottom: 2px;
}
.flow-label {
  font-size: 0.72em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.flow-sub {
  font-size: 0.65em;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}
.flow-arrow {
  position: absolute;
  right: -12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--vp-c-text-3);
  font-size: 1.1em;
  z-index: 1;
}
.stage-detail {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 0.82em;
  color: var(--vp-c-text-2);
  margin-bottom: 12px;
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
  margin-top: 8px;
}
.chain-svg {
  width: 100%;
  max-width: 560px;
  display: block;
  margin: 0 auto;
}
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
.bar-section {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--vp-c-divider);
}
.bar-chart {
  display: flex;
  justify-content: center;
  gap: 24px;
  align-items: flex-end;
  height: 120px;
}
.bar-group {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}
.bar-wrapper {
  display: flex;
  align-items: flex-end;
  height: 80px;
}
.bar {
  width: 36px;
  border-radius: 4px 4px 0 0;
  transition: height 0.3s ease;
}
.red-bar { background: #e74c3c; }
.green-bar { background: #27ae60; }
.blue-bar { background: #3498db; }
.bar-label {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.bar-value {
  font-size: 0.75em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-2);
}
</style>
