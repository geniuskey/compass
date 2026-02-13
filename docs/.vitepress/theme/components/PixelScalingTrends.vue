<template>
  <div class="scaling-container">
    <h4>{{ t('Pixel Scaling Trends', '픽셀 스케일링 트렌드') }}</h4>
    <p class="component-description">
      {{ t(
        'Explore how key sensor metrics scale with pixel pitch. Compare theoretical scaling laws against published sensor data.',
        '주요 센서 지표가 픽셀 피치에 따라 어떻게 변하는지 살펴보세요. 이론적 스케일링 법칙과 공개된 센서 데이터를 비교합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="radio-group">
        <span class="radio-label">{{ t('Technology Node', '공정 노드') }}:</span>
        <label v-for="node in techNodes" :key="node.name" class="radio-item">
          <input type="radio" :value="node.name" v-model="selectedNode" />
          {{ node.name }}
        </label>
      </div>
      <div class="slider-group">
        <label>
          {{ t('f-number', 'f값') }}: <strong>f/{{ fNumber.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.4" max="5.6" step="0.1" v-model.number="fNumber" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Wavelength', '파장') }}: <strong>{{ wavelengthNm }} nm</strong>
        </label>
        <input type="range" min="450" max="650" step="10" v-model.number="wavelengthNm" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('My Pitch', '내 피치') }}: <strong>{{ myPitch.toFixed(2) }} &mu;m</strong>
        </label>
        <input type="range" min="0.5" max="3.0" step="0.05" v-model.number="myPitch" class="ctrl-range pitch-range" />
      </div>
    </div>

    <!-- 2x2 mini charts -->
    <div class="charts-grid">
      <!-- Chart 1: FWC vs pitch -->
      <div class="mini-chart">
        <h5>{{ t('Full Well Capacity', '풀웰 용량') }} (ke⁻)</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${miniW} ${miniH}`"
            class="mini-svg"
            @mousemove="(e) => onMiniMouseMove(e, 0)"
            @mouseleave="syncHoverPitch = null"
          >
            <!-- Grid -->
            <line v-for="tick in fwcYTicks" :key="'fwcyg'+tick"
              :x1="miniPad.left" :y1="fwcYScale(tick)"
              :x2="miniPad.left + miniPlotW" :y2="fwcYScale(tick)"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in xTicks" :key="'fwcxg'+tick"
              :x1="miniXScale(tick)" :y1="miniPad.top"
              :x2="miniXScale(tick)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

            <!-- Axes -->
            <line :x1="miniPad.left" :y1="miniPad.top" :x2="miniPad.left" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="miniPad.left" :y1="miniPad.top + miniPlotH" :x2="miniPad.left + miniPlotW" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- Tick labels -->
            <template v-for="tick in xTicks" :key="'fwcxl'+tick">
              <text :x="miniXScale(tick)" :y="miniPad.top + miniPlotH + 12" text-anchor="middle" class="tick-label">
                {{ tick.toFixed(1) }}
              </text>
            </template>
            <template v-for="tick in fwcYTicks" :key="'fwcyl'+tick">
              <text :x="miniPad.left - 5" :y="fwcYScale(tick) + 3" text-anchor="end" class="tick-label">
                {{ tick }}
              </text>
            </template>

            <!-- Technology node curves -->
            <path v-for="node in techNodes" :key="'fwc-'+node.name"
              :d="buildFwcPath(node.kFwc)"
              fill="none"
              :stroke="node.name === selectedNode ? '#3498db' : 'var(--vp-c-divider)'"
              :stroke-width="node.name === selectedNode ? 2 : 1"
              :stroke-dasharray="node.name === selectedNode ? 'none' : '4,3'"
              :opacity="node.name === selectedNode ? 1 : 0.5"
            />

            <!-- Node labels at right edge -->
            <template v-for="node in techNodes" :key="'fwcl-'+node.name">
              <text
                :x="miniPad.left + miniPlotW + 3"
                :y="fwcYScale(node.kFwc * pitchMax * pitchMax / 1000) + 3"
                class="node-label"
                :fill="node.name === selectedNode ? '#3498db' : 'var(--vp-c-text-3)'"
              >{{ node.name }}</text>
            </template>

            <!-- Reference data points -->
            <template v-for="sensor in sensors" :key="'fwcd-'+sensor.name">
              <circle
                v-if="sensor.pitch >= pitchMin && sensor.pitch <= pitchMax"
                :cx="miniXScale(sensor.pitch)"
                :cy="fwcYScale(sensor.fwc / 1000)"
                r="3.5" fill="#e74c3c" stroke="#fff" stroke-width="1"
              />
              <text
                v-if="sensor.pitch >= pitchMin && sensor.pitch <= pitchMax"
                :x="miniXScale(sensor.pitch) + 5"
                :y="fwcYScale(sensor.fwc / 1000) - 5"
                class="sensor-label"
              >{{ sensor.name }}</text>
            </template>

            <!-- My pitch marker -->
            <line
              :x1="miniXScale(myPitch)" :y1="miniPad.top"
              :x2="miniXScale(myPitch)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" />

            <!-- Sync hover crosshair -->
            <template v-if="syncHoverPitch !== null">
              <line
                :x1="miniXScale(syncHoverPitch)" :y1="miniPad.top"
                :x2="miniXScale(syncHoverPitch)" :y2="miniPad.top + miniPlotH"
                stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="3,2" />
              <circle
                :cx="miniXScale(syncHoverPitch)"
                :cy="fwcYScale(currentNode.kFwc * syncHoverPitch * syncHoverPitch / 1000)"
                r="3" fill="#3498db" stroke="#fff" stroke-width="1" />
            </template>

            <!-- Axis label -->
            <text :x="miniPad.left + miniPlotW / 2" :y="miniH - 1" text-anchor="middle" class="axis-title">
              {{ t('Pitch (\u00B5m)', '\uD53C\uCE58 (\u00B5m)') }}
            </text>
          </svg>
        </div>
      </div>

      <!-- Chart 2: Max SNR vs pitch -->
      <div class="mini-chart">
        <h5>{{ t('Max SNR', '최대 SNR') }} (dB)</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${miniW} ${miniH}`"
            class="mini-svg"
            @mousemove="(e) => onMiniMouseMove(e, 1)"
            @mouseleave="syncHoverPitch = null"
          >
            <!-- Grid -->
            <line v-for="tick in snrYTicks" :key="'snryg'+tick"
              :x1="miniPad.left" :y1="snrYScale(tick)"
              :x2="miniPad.left + miniPlotW" :y2="snrYScale(tick)"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in xTicks" :key="'snrxg'+tick"
              :x1="miniXScale(tick)" :y1="miniPad.top"
              :x2="miniXScale(tick)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

            <!-- Axes -->
            <line :x1="miniPad.left" :y1="miniPad.top" :x2="miniPad.left" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="miniPad.left" :y1="miniPad.top + miniPlotH" :x2="miniPad.left + miniPlotW" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- Tick labels -->
            <template v-for="tick in xTicks" :key="'snrxl'+tick">
              <text :x="miniXScale(tick)" :y="miniPad.top + miniPlotH + 12" text-anchor="middle" class="tick-label">
                {{ tick.toFixed(1) }}
              </text>
            </template>
            <template v-for="tick in snrYTicks" :key="'snryl'+tick">
              <text :x="miniPad.left - 5" :y="snrYScale(tick) + 3" text-anchor="end" class="tick-label">
                {{ tick }}
              </text>
            </template>

            <!-- SNR curve -->
            <path :d="snrPath" fill="none" stroke="#27ae60" stroke-width="2" />

            <!-- Reference data points -->
            <template v-for="sensor in sensors" :key="'snrd-'+sensor.name">
              <circle
                v-if="sensor.pitch >= pitchMin && sensor.pitch <= pitchMax"
                :cx="miniXScale(sensor.pitch)"
                :cy="snrYScale(sensor.snr)"
                r="3.5" fill="#e74c3c" stroke="#fff" stroke-width="1"
              />
              <text
                v-if="sensor.pitch >= pitchMin && sensor.pitch <= pitchMax"
                :x="miniXScale(sensor.pitch) + 5"
                :y="snrYScale(sensor.snr) - 5"
                class="sensor-label"
              >{{ sensor.name }}</text>
            </template>

            <!-- My pitch marker -->
            <line
              :x1="miniXScale(myPitch)" :y1="miniPad.top"
              :x2="miniXScale(myPitch)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" />

            <!-- Sync hover crosshair -->
            <template v-if="syncHoverPitch !== null">
              <line
                :x1="miniXScale(syncHoverPitch)" :y1="miniPad.top"
                :x2="miniXScale(syncHoverPitch)" :y2="miniPad.top + miniPlotH"
                stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="3,2" />
              <circle
                :cx="miniXScale(syncHoverPitch)"
                :cy="snrYScale(calcSnr(syncHoverPitch))"
                r="3" fill="#27ae60" stroke="#fff" stroke-width="1" />
            </template>

            <!-- Axis label -->
            <text :x="miniPad.left + miniPlotW / 2" :y="miniH - 1" text-anchor="middle" class="axis-title">
              {{ t('Pitch (\u00B5m)', '\uD53C\uCE58 (\u00B5m)') }}
            </text>
          </svg>
        </div>
      </div>

      <!-- Chart 3: Nyquist vs pitch -->
      <div class="mini-chart">
        <h5>{{ t('Nyquist Frequency', '나이퀴스트 주파수') }} (lp/mm)</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${miniW} ${miniH}`"
            class="mini-svg"
            @mousemove="(e) => onMiniMouseMove(e, 2)"
            @mouseleave="syncHoverPitch = null"
          >
            <!-- Grid -->
            <line v-for="tick in nyqYTicks" :key="'nyqyg'+tick"
              :x1="miniPad.left" :y1="nyqYScale(tick)"
              :x2="miniPad.left + miniPlotW" :y2="nyqYScale(tick)"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in xTicks" :key="'nyqxg'+tick"
              :x1="miniXScale(tick)" :y1="miniPad.top"
              :x2="miniXScale(tick)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

            <!-- Axes -->
            <line :x1="miniPad.left" :y1="miniPad.top" :x2="miniPad.left" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="miniPad.left" :y1="miniPad.top + miniPlotH" :x2="miniPad.left + miniPlotW" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- Tick labels -->
            <template v-for="tick in xTicks" :key="'nyqxl'+tick">
              <text :x="miniXScale(tick)" :y="miniPad.top + miniPlotH + 12" text-anchor="middle" class="tick-label">
                {{ tick.toFixed(1) }}
              </text>
            </template>
            <template v-for="tick in nyqYTicks" :key="'nyqyl'+tick">
              <text :x="miniPad.left - 5" :y="nyqYScale(tick) + 3" text-anchor="end" class="tick-label">
                {{ tick }}
              </text>
            </template>

            <!-- Nyquist curve -->
            <path :d="nyquistPath" fill="none" stroke="#e67e22" stroke-width="2" />

            <!-- My pitch marker -->
            <line
              :x1="miniXScale(myPitch)" :y1="miniPad.top"
              :x2="miniXScale(myPitch)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" />

            <!-- Sync hover crosshair -->
            <template v-if="syncHoverPitch !== null">
              <line
                :x1="miniXScale(syncHoverPitch)" :y1="miniPad.top"
                :x2="miniXScale(syncHoverPitch)" :y2="miniPad.top + miniPlotH"
                stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="3,2" />
              <circle
                :cx="miniXScale(syncHoverPitch)"
                :cy="nyqYScale(calcNyquist(syncHoverPitch))"
                r="3" fill="#e67e22" stroke="#fff" stroke-width="1" />
            </template>

            <!-- Axis label -->
            <text :x="miniPad.left + miniPlotW / 2" :y="miniH - 1" text-anchor="middle" class="axis-title">
              {{ t('Pitch (\u00B5m)', '\uD53C\uCE58 (\u00B5m)') }}
            </text>
          </svg>
        </div>
      </div>

      <!-- Chart 4: Diffraction QE vs pitch -->
      <div class="mini-chart">
        <h5>{{ t('Diffraction QE', '회절 QE') }} (%)</h5>
        <div class="svg-wrapper">
          <svg
            :viewBox="`0 0 ${miniW} ${miniH}`"
            class="mini-svg"
            @mousemove="(e) => onMiniMouseMove(e, 3)"
            @mouseleave="syncHoverPitch = null"
          >
            <!-- Grid -->
            <line v-for="tick in qeYTicks" :key="'qeyg'+tick"
              :x1="miniPad.left" :y1="qeYScale(tick)"
              :x2="miniPad.left + miniPlotW" :y2="qeYScale(tick)"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in xTicks" :key="'qexg'+tick"
              :x1="miniXScale(tick)" :y1="miniPad.top"
              :x2="miniXScale(tick)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />

            <!-- Axes -->
            <line :x1="miniPad.left" :y1="miniPad.top" :x2="miniPad.left" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="miniPad.left" :y1="miniPad.top + miniPlotH" :x2="miniPad.left + miniPlotW" :y2="miniPad.top + miniPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />

            <!-- Tick labels -->
            <template v-for="tick in xTicks" :key="'qexl'+tick">
              <text :x="miniXScale(tick)" :y="miniPad.top + miniPlotH + 12" text-anchor="middle" class="tick-label">
                {{ tick.toFixed(1) }}
              </text>
            </template>
            <template v-for="tick in qeYTicks" :key="'qeyl'+tick">
              <text :x="miniPad.left - 5" :y="qeYScale(tick) + 3" text-anchor="end" class="tick-label">
                {{ tick }}
              </text>
            </template>

            <!-- QE curve -->
            <path :d="qePath" fill="none" stroke="#8e44ad" stroke-width="2" />

            <!-- My pitch marker -->
            <line
              :x1="miniXScale(myPitch)" :y1="miniPad.top"
              :x2="miniXScale(myPitch)" :y2="miniPad.top + miniPlotH"
              stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" />

            <!-- Sync hover crosshair -->
            <template v-if="syncHoverPitch !== null">
              <line
                :x1="miniXScale(syncHoverPitch)" :y1="miniPad.top"
                :x2="miniXScale(syncHoverPitch)" :y2="miniPad.top + miniPlotH"
                stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="3,2" />
              <circle
                :cx="miniXScale(syncHoverPitch)"
                :cy="qeYScale(calcDiffQePercent(syncHoverPitch))"
                r="3" fill="#8e44ad" stroke="#fff" stroke-width="1" />
            </template>

            <!-- Axis label -->
            <text :x="miniPad.left + miniPlotW / 2" :y="miniH - 1" text-anchor="middle" class="axis-title">
              {{ t('Pitch (\u00B5m)', '\uD53C\uCE58 (\u00B5m)') }}
            </text>
          </svg>
        </div>
      </div>
    </div>

    <!-- Info cards -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('FWC', '풀웰 용량') }}</div>
        <div class="result-value" style="color: #3498db;">{{ myFwc.toFixed(0) }} e⁻</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Max SNR', '최대 SNR') }}</div>
        <div class="result-value" style="color: #27ae60;">{{ mySnr.toFixed(1) }} dB</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Nyquist', '나이퀴스트') }}</div>
        <div class="result-value" style="color: #e67e22;">{{ myNyquist.toFixed(0) }} lp/mm</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Diff. QE', '회절 QE') }}</div>
        <div class="result-value" style="color: #8e44ad;">{{ myDiffQe.toFixed(1) }}%</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Dark Current @60\u00B0C', '\uC554\uC804\uB958 @60\u00B0C') }}</div>
        <div class="result-value">{{ myDarkCurrent.toFixed(1) }} e⁻/s</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// ---- Technology nodes ----
interface TechNode {
  name: string
  kFwc: number // e-/um²
}

const techNodes: TechNode[] = [
  { name: '65nm', kFwc: 8000 },
  { name: '40nm', kFwc: 10000 },
  { name: '22nm', kFwc: 12000 },
]

// ---- Reference sensors ----
interface SensorRef {
  name: string
  pitch: number
  fwc: number
  snr: number
}

const sensors: SensorRef[] = [
  { name: 'IMX586', pitch: 0.8, fwc: 6000, snr: 37.8 },
  { name: 'IMX766', pitch: 1.0, fwc: 10000, snr: 40 },
  { name: 'IMX890', pitch: 1.22, fwc: 14000, snr: 41.5 },
  { name: 'IMX989', pitch: 1.6, fwc: 22000, snr: 43.4 },
  { name: 'IMX283', pitch: 2.74, fwc: 51000, snr: 47 },
]

// ---- Controls ----
const selectedNode = ref('40nm')
const fNumber = ref(2.8)
const wavelengthNm = ref(550)
const myPitch = ref(1.0)

const currentNode = computed(() => techNodes.find(n => n.name === selectedNode.value) || techNodes[1])
const wavelengthUm = computed(() => wavelengthNm.value / 1000)

// ---- Physics functions ----
const kDark = 5 // e-/s/um² at 60C

function calcFwc(pitch: number): number {
  return currentNode.value.kFwc * pitch * pitch
}

function calcSnr(pitch: number): number {
  const fwc = calcFwc(pitch)
  return 10 * Math.log10(Math.max(1, fwc))
}

function calcNyquist(pitch: number): number {
  return 1 / (2 * pitch) * 1000 // lp/mm
}

function calcDiffQe(pitch: number): number {
  const airyRadius = 1.22 * wavelengthUm.value * fNumber.value
  return Math.min(1, (pitch / airyRadius) ** 2)
}

function calcDiffQePercent(pitch: number): number {
  return calcDiffQe(pitch) * 100
}

function calcDarkCurrent(pitch: number): number {
  return kDark * pitch * pitch
}

// ---- Computed metrics for "my pitch" ----
const myFwc = computed(() => calcFwc(myPitch.value))
const mySnr = computed(() => calcSnr(myPitch.value))
const myNyquist = computed(() => calcNyquist(myPitch.value))
const myDiffQe = computed(() => calcDiffQePercent(myPitch.value))
const myDarkCurrent = computed(() => calcDarkCurrent(myPitch.value))

// ---- Chart dimensions ----
const miniW = 300
const miniH = 180
const miniPad = { top: 12, right: 35, bottom: 28, left: 42 }
const miniPlotW = miniW - miniPad.left - miniPad.right
const miniPlotH = miniH - miniPad.top - miniPad.bottom

const pitchMin = 0.5
const pitchMax = 3.0

const xTicks = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

function miniXScale(pitch: number): number {
  return miniPad.left + ((pitch - pitchMin) / (pitchMax - pitchMin)) * miniPlotW
}

// ---- FWC chart (0-100 ke-) ----
const fwcYMin = 0
const fwcYMax = 110
const fwcYTicks = [0, 20, 40, 60, 80, 100]

function fwcYScale(ke: number): number {
  return miniPad.top + miniPlotH - ((ke - fwcYMin) / (fwcYMax - fwcYMin)) * miniPlotH
}

function buildFwcPath(kFwc: number): string {
  const steps = 100
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const p = pitchMin + (i / steps) * (pitchMax - pitchMin)
    const fwcKe = kFwc * p * p / 1000
    const clampedY = Math.max(fwcYMin, Math.min(fwcYMax, fwcKe))
    const x = miniXScale(p)
    const y = fwcYScale(clampedY)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

// ---- SNR chart (20-50 dB) ----
const snrYMin = 25
const snrYMax = 50
const snrYTicks = [25, 30, 35, 40, 45, 50]

function snrYScale(db: number): number {
  return miniPad.top + miniPlotH - ((db - snrYMin) / (snrYMax - snrYMin)) * miniPlotH
}

const snrPath = computed(() => {
  const steps = 100
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const p = pitchMin + (i / steps) * (pitchMax - pitchMin)
    const snr = calcSnr(p)
    const clampedY = Math.max(snrYMin, Math.min(snrYMax, snr))
    const x = miniXScale(p)
    const y = snrYScale(clampedY)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
})

// ---- Nyquist chart (150-1000 lp/mm) ----
const nyqYMin = 100
const nyqYMax = 1050
const nyqYTicks = [200, 400, 600, 800, 1000]

function nyqYScale(lpmm: number): number {
  return miniPad.top + miniPlotH - ((lpmm - nyqYMin) / (nyqYMax - nyqYMin)) * miniPlotH
}

const nyquistPath = computed(() => {
  const steps = 100
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const p = pitchMin + (i / steps) * (pitchMax - pitchMin)
    const nyq = calcNyquist(p)
    const clampedY = Math.max(nyqYMin, Math.min(nyqYMax, nyq))
    const x = miniXScale(p)
    const y = nyqYScale(clampedY)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
})

// ---- QE chart (0-100%) ----
const qeYMin = 0
const qeYMax = 105
const qeYTicks = [0, 20, 40, 60, 80, 100]

function qeYScale(pct: number): number {
  return miniPad.top + miniPlotH - ((pct - qeYMin) / (qeYMax - qeYMin)) * miniPlotH
}

const qePath = computed(() => {
  const steps = 100
  let d = ''
  for (let i = 0; i <= steps; i++) {
    const p = pitchMin + (i / steps) * (pitchMax - pitchMin)
    const qe = calcDiffQePercent(p)
    const clampedY = Math.max(qeYMin, Math.min(qeYMax, qe))
    const x = miniXScale(p)
    const y = qeYScale(clampedY)
    d += i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : ` L${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
})

// ---- Synchronized hover ----
const syncHoverPitch = ref<number | null>(null)

function onMiniMouseMove(event: MouseEvent, _chartIndex: number) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = miniW / rect.width
  const mouseX = (event.clientX - rect.left) * scaleX
  const p = pitchMin + ((mouseX - miniPad.left) / miniPlotW) * (pitchMax - pitchMin)
  if (p >= pitchMin && p <= pitchMax) {
    syncHoverPitch.value = p
  } else {
    syncHoverPitch.value = null
  }
}
</script>

<style scoped>
.scaling-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.scaling-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.scaling-container h5 {
  margin: 0 0 6px 0;
  font-size: 0.9em;
  color: var(--vp-c-text-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
  align-items: flex-end;
}
.radio-group {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  min-width: 200px;
  padding-bottom: 4px;
}
.radio-label {
  font-size: 0.85em;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.radio-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.85em;
  cursor: pointer;
}
.radio-item input[type="radio"] {
  accent-color: var(--vp-c-brand-1);
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
.pitch-range::-webkit-slider-thumb {
  background: var(--vp-c-brand-1);
}
.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 16px;
}
@media (max-width: 640px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}
.mini-chart {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
}
.svg-wrapper {
  margin-top: 4px;
}
.mini-svg {
  width: 100%;
  display: block;
  margin: 0 auto;
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
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
  font-size: 1.0em;
  font-family: var(--vp-font-family-mono);
}
.tick-label {
  font-size: 8px;
  fill: var(--vp-c-text-3);
}
.axis-title {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.node-label {
  font-size: 7.5px;
  font-weight: 600;
}
.sensor-label {
  font-size: 7px;
  fill: #e74c3c;
  font-weight: 600;
}
</style>
