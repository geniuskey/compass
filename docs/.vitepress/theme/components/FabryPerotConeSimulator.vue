<template>
  <div class="fpcs-container">
    <h4>{{ t('Fabry-Perot Cone Illumination Simulator', '파브리-페로 원뿔 조명 시뮬레이터') }}</h4>
    <p class="component-description">
      {{ t(
        'Reproduces the central result of Goossens et al. (2018), Appl. Opt. 57(26):7539. A single-cavity Fabry-Perot filter on an image sensor is illuminated by a focused cone of light; the simulator integrates the Airy transmittance over the cone defined by the chief-ray angle and the F-number, and compares against the plane-wave result.',
        'Goossens et al. (2018), Appl. Opt. 57(26):7539의 핵심 결과를 재현합니다. 이미지 센서 위 단일 공진 Fabry-Perot 필터에 초점 광선이 입사할 때, 주광선 각도(CRA)와 F 넘버로 정의되는 원뿔에 대해 Airy 투과율을 적분하고 평면파 결과와 비교합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Design center', '설계 중심파장') }} &lambda;<sub>0</sub>:
          <strong>{{ lambda0.toFixed(0) }} nm</strong>
        </label>
        <input type="range" min="400" max="900" step="5" v-model.number="lambda0" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Effective index', '유효 굴절률') }} n<sub>eff</sub>:
          <strong>{{ nEff.toFixed(2) }}</strong>
        </label>
        <input type="range" min="1.30" max="2.60" step="0.01" v-model.number="nEff" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Mirror reflectance', '거울 반사율') }} R:
          <strong>{{ (mirrorR * 100).toFixed(0) }}%</strong>
        </label>
        <input type="range" min="0.50" max="0.98" step="0.01" v-model.number="mirrorR" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Chief ray angle', '주광선 각도') }} CRA:
          <strong>{{ craDeg.toFixed(1) }}&deg;</strong>
        </label>
        <input type="range" min="0" max="30" step="0.5" v-model.number="craDeg" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('F-number', 'F 넘버') }}:
          <strong>F/{{ fNumber.toFixed(1) }}</strong>
        </label>
        <input type="range" min="1.0" max="11.0" step="0.1" v-model.number="fNumber" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Cone samples', '원뿔 샘플 수') }}:
          <strong>{{ nSamples }}</strong>
        </label>
        <input type="range" min="7" max="201" step="2" v-model.number="nSamples" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>{{ t('Sampling pattern', '샘플링 패턴') }}</label>
        <select v-model="samplingMode" class="ctrl-select">
          <option value="rings">{{ t('Concentric rings', '동심 ring') }}</option>
          <option value="fibonacci">{{ t('Fibonacci spiral', '피보나치 나선') }}</option>
          <option value="grid">{{ t('Polar grid', '극좌표 격자') }}</option>
        </select>
      </div>
      <div class="slider-group">
        <label>{{ t('Pupil weighting', '동공 가중치') }}</label>
        <select v-model="weightingMode" class="ctrl-select">
          <option value="uniform">{{ t('Uniform', '균일') }}</option>
          <option value="cosine">cos &theta;<sub>local</sub></option>
          <option value="cos4">cos&#8308; &theta;<sub>local</sub></option>
          <option value="gaussian">{{ t('Gaussian', '가우시안') }}</option>
        </select>
      </div>
    </div>

    <!-- Top view of cone samples -->
    <div class="topview-section">
      <h5 class="panel-title">{{ t('Cone samples (top view, sensor frame)', '원뿔 샘플 (위에서 본 모습, 센서 좌표계)') }}</h5>
      <svg
        :viewBox="`0 0 ${tvW} ${tvH}`"
        class="topview-svg"
        role="img"
        :aria-label="t('Fabry-Perot cone integration sample points in sensor-frame direction cosine space', '센서 좌표계 방향코사인 공간의 Fabry-Perot 콘 적분 샘플 포인트')"
      >
        <!-- Concentric reference circles at fixed polar angles -->
        <template v-for="ang in tvRefAngles" :key="'rc' + ang">
          <circle
            :cx="tvCx" :cy="tvCy" :r="tvAngleToR(ang)"
            fill="none" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <text
            :x="tvCx + tvAngleToR(ang) + 3" :y="tvCy + 3"
            class="tick-label"
          >{{ ang }}&deg;</text>
        </template>
        <!-- Axes -->
        <line :x1="tvCx - tvR" :y1="tvCy" :x2="tvCx + tvR" :y2="tvCy"
              stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <line :x1="tvCx" :y1="tvCy - tvR" :x2="tvCx" :y2="tvCy + tvR"
              stroke="var(--vp-c-divider)" stroke-width="0.5" />
        <text :x="tvCx + tvR + 12" :y="tvCy + 4" text-anchor="middle" class="tick-label">x</text>
        <text :x="tvCx - 6" :y="tvCy - tvR - 4" text-anchor="middle" class="tick-label">y</text>

        <!-- Cone outline (projected ellipse) -->
        <path :d="coneOutlinePath" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7" />

        <!-- Ring arcs (only for ring sampling: connect points on the same pupil ring) -->
        <path v-for="(arc, i) in ringArcPaths" :key="'arc' + i"
              :d="arc" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="0.8" opacity="0.28" />

        <!-- Chief ray marker -->
        <circle :cx="chiefScreen.x" :cy="chiefScreen.y" r="4"
                fill="none" stroke="var(--vp-c-brand-1)" stroke-width="1.5" />
        <line :x1="chiefScreen.x - 5" :y1="chiefScreen.y" :x2="chiefScreen.x + 5" :y2="chiefScreen.y"
              stroke="var(--vp-c-brand-1)" stroke-width="1" />
        <line :x1="chiefScreen.x" :y1="chiefScreen.y - 5" :x2="chiefScreen.x" :y2="chiefScreen.y + 5"
              stroke="var(--vp-c-brand-1)" stroke-width="1" />

        <!-- Sample points -->
        <circle v-for="(p, i) in samplePoints" :key="'p' + i"
                :cx="p.x" :cy="p.y" :r="p.r"
                data-visual-id="fp-cone-sample"
                :fill="p.color" :opacity="p.opacity" stroke="var(--vp-c-bg)" stroke-width="0.5" />

        <!-- Weight colorbar -->
        <g :transform="`translate(${tvW - 36}, ${tvH - 158})`">
          <defs>
            <linearGradient :id="colorbarGradientId" x1="0" y1="1" x2="0" y2="0">
              <stop v-for="(s, i) in colorbarStops" :key="'stop' + i"
                    :offset="(i / (colorbarStops.length - 1) * 100).toFixed(1) + '%'"
                    :stop-color="s" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="10" height="120" :fill="`url(#${colorbarGradientId})`"
                stroke="var(--vp-c-divider)" stroke-width="0.5" />
          <text x="14" y="6" class="tick-label">w<sub>max</sub></text>
          <text x="14" y="64" class="tick-label">w</text>
          <text x="14" y="122" class="tick-label">0</text>
          <text x="-2" y="-6" class="legend-label">{{ t('weight', '가중치') }}</text>
        </g>

        <!-- Legend -->
        <g :transform="`translate(10, ${tvH - 56})`">
          <rect x="-6" y="-6" width="148" height="56" rx="4"
                fill="var(--vp-c-bg)" opacity="0.85" stroke="var(--vp-c-divider)" />
          <line x1="2" y1="6" x2="14" y2="6" stroke="var(--vp-c-brand-1)" stroke-width="1.5" stroke-dasharray="5,3" />
          <text x="18" y="9" class="legend-label">{{ t('cone outline', '원뿔 외곽') }}</text>
          <line x1="2" y1="22" x2="14" y2="22" stroke="var(--vp-c-brand-1)" stroke-width="0.8" opacity="0.6" />
          <text x="18" y="25" class="legend-label">{{ t('pupil ring (constant r)', '동공 ring (등반경)') }}</text>
          <circle cx="8" cy="38" r="3" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="1.5" />
          <text x="18" y="41" class="legend-label">{{ t('chief ray', '주광선') }}</text>
        </g>
      </svg>
      <div class="topview-caption">
        {{ t(
          'Each dot is one plane-wave sample fed into the cone integration. Axes are direction cosines (sin&theta; cos&phi;, sin&theta; sin&phi;) in the sensor frame; thin arcs connect samples on the same pupil-ring (equal r_pupil) and show how concentric pupil rings tilt into small circles on the sphere when CRA is nonzero. The colorbar maps the pupil weighting w(&theta;_local); switch the weighting selector to see how uniform vs cos&theta; vs cos&#8308;&theta; vs Gaussian weighting redistributes the integrand.',
          '각 점은 원뿔 적분에 들어가는 평면파 샘플입니다. 축은 센서 좌표계의 방향코사인 (sin&theta; cos&phi;, sin&theta; sin&phi;)이며, 가는 호는 동일 동공 반경(equal r_pupil)을 가지는 샘플들을 잇습니다 — CRA가 0이 아니면 동심 동공 ring이 구면 위에서 기울어진 소원(small circle)이 되어 호처럼 보입니다. 컬러바는 동공 가중치 w(&theta;_local)을 표현하며, 가중치 셀렉터를 바꾸면 uniform / cos&theta; / cos&#8308;&theta; / Gaussian이 적분에 어떻게 다르게 기여하는지 비교할 수 있습니다.'
        ) }}
      </div>
    </div>

    <!-- Spectrum chart -->
    <div class="chart-section">
      <svg
        :viewBox="`0 0 ${chartW} ${chartH}`"
        class="chart-svg"
        role="img"
        :aria-label="t('Fabry-Perot transmittance spectrum comparing normal incidence, plane wave at CRA, and cone integration', '수직 입사, CRA 평면파, 콘 적분을 비교하는 Fabry-Perot 투과율 스펙트럼')"
      >
        <!-- Grid -->
        <template v-for="pct in [0, 25, 50, 75, 100]" :key="'gy' + pct">
          <line
            :x1="padL" :y1="pctToY(pct)"
            :x2="padL + plotW" :y2="pctToY(pct)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <text :x="padL - 6" :y="pctToY(pct) + 3" text-anchor="end" class="tick-label">{{ pct }}%</text>
        </template>

        <!-- X-axis ticks -->
        <template v-for="(wl, i) in xTicks" :key="'gx' + i">
          <line
            :x1="lambdaToX(wl)" :y1="padT + plotH"
            :x2="lambdaToX(wl)" :y2="padT + plotH + 4"
            stroke="var(--vp-c-text-3)" stroke-width="1"
          />
          <text :x="lambdaToX(wl)" :y="padT + plotH + 16" text-anchor="middle" class="tick-label">{{ wl.toFixed(0) }}</text>
        </template>

        <!-- Axes -->
        <line :x1="padL" :y1="padT" :x2="padL" :y2="padT + plotH" stroke="var(--vp-c-text-3)" stroke-width="1" />
        <line :x1="padL" :y1="padT + plotH" :x2="padL + plotW" :y2="padT + plotH" stroke="var(--vp-c-text-3)" stroke-width="1" />

        <!-- Axis labels -->
        <text
          x="14" :y="padT + plotH / 2" text-anchor="middle"
          :transform="`rotate(-90, 14, ${padT + plotH / 2})`" class="axis-label"
        >{{ t('Transmittance', '투과율') }} T</text>
        <text :x="padL + plotW / 2" :y="chartH - 4" text-anchor="middle" class="axis-label">
          {{ t('Wavelength (nm)', '파장 (nm)') }}
        </text>

        <!-- Normal-incidence reference -->
        <path :d="pathNormal" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="3,3" />
        <!-- Plane wave at CRA -->
        <path :d="pathPlaneWave" fill="none" stroke="#3498db" stroke-width="2" />
        <!-- Cone-integrated -->
        <path data-visual-id="fp-cone-spectrum" :d="pathCone" fill="none" stroke="#e74c3c" stroke-width="2.5" />

        <!-- Peak markers -->
        <template v-if="planePeak">
          <line
            :x1="lambdaToX(planePeak.lambda)" :y1="pctToY(planePeak.T * 100)"
            :x2="lambdaToX(planePeak.lambda)" :y2="padT + plotH"
            stroke="#3498db" stroke-width="1" stroke-dasharray="2,2" opacity="0.6"
          />
        </template>
        <template v-if="conePeak">
          <line
            :x1="lambdaToX(conePeak.lambda)" :y1="pctToY(conePeak.T * 100)"
            :x2="lambdaToX(conePeak.lambda)" :y2="padT + plotH"
            stroke="#e74c3c" stroke-width="1" stroke-dasharray="2,2" opacity="0.6"
          />
        </template>

        <!-- Legend -->
        <g :transform="`translate(${padL + plotW - 165}, ${padT + 8})`">
          <rect x="-6" y="-6" width="165" height="58" rx="4" fill="var(--vp-c-bg)" opacity="0.85" stroke="var(--vp-c-divider)" />
          <line x1="0" y1="6" x2="20" y2="6" stroke="var(--vp-c-text-3)" stroke-width="1" stroke-dasharray="3,3" />
          <text x="26" y="9" class="legend-label">{{ t('Normal incidence', '수직 입사') }}</text>
          <line x1="0" y1="22" x2="20" y2="22" stroke="#3498db" stroke-width="2" />
          <text x="26" y="25" class="legend-label">{{ t('Plane wave at CRA', 'CRA 평면파') }}</text>
          <line x1="0" y1="38" x2="20" y2="38" stroke="#e74c3c" stroke-width="2.5" />
          <text x="26" y="41" class="legend-label">{{ t('Cone-integrated', '원뿔 적분') }}</text>
        </g>
      </svg>
    </div>

    <!-- Metrics cards -->
    <div class="info-cards">
      <div class="info-card">
        <div class="info-label">{{ t('Peak shift vs &lambda;&#8320;', '피크 이동 (vs &lambda;&#8320;)') }}</div>
        <div class="info-value highlight">&Delta;&lambda; = {{ conePeak ? (conePeak.lambda - lambda0).toFixed(2) : '--' }} nm</div>
        <div class="info-sub">{{ t('plane-wave', '평면파') }}: {{ planePeak ? (planePeak.lambda - lambda0).toFixed(2) : '--' }} nm</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('FWHM (cone)', '반치폭 (원뿔)') }}</div>
        <div class="info-value">{{ coneFWHM ? coneFWHM.toFixed(2) + ' nm' : '--' }}</div>
        <div class="info-sub">{{ t('plane-wave', '평면파') }}: {{ planeFWHM ? planeFWHM.toFixed(2) + ' nm' : '--' }} nm</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Half-cone angle', '반원뿔 각도') }}</div>
        <div class="info-value">{{ halfConeDeg.toFixed(2) }}&deg;</div>
        <div class="info-sub">arcsin(1/2F)</div>
      </div>
      <div class="info-card">
        <div class="info-label">{{ t('Peak transmittance', '피크 투과율') }}</div>
        <div class="info-value">{{ conePeak ? (conePeak.T * 100).toFixed(1) + '%' : '--' }}</div>
        <div class="info-sub">{{ t('plane-wave', '평면파') }}: {{ planePeak ? (planePeak.T * 100).toFixed(1) + '%' : '--' }}</div>
      </div>
    </div>

    <details class="theory-block">
      <summary>{{ t('Model details', '모델 설명') }}</summary>
      <div class="theory-body">
        <p>
          {{ t(
            'The Airy transmittance of a single-cavity Fabry-Perot filter is',
            '단일 공진 Fabry-Perot 필터의 Airy 투과율은 다음과 같습니다'
          ) }}
        </p>
        <p class="formula">
          T(&lambda;, &theta;) = 1 / [1 + F &middot; sin&sup2;(&delta;(&lambda;, &theta;) / 2)],
          {{ t('with', '여기서') }}
          &delta; = (4&pi; n<sub>eff</sub> d cos&theta;<sub>int</sub>) / &lambda;,
          F = 4R / (1 &minus; R)&sup2;,
          n<sub>eff</sub> sin&theta;<sub>int</sub> = sin&theta;.
        </p>
        <p>
          {{ t(
            'The cavity thickness d is fixed by the design condition (first-order peak at &lambda;&#8320; at normal incidence): d = &lambda;&#8320; / (2 n_eff). For each cone sample, the local ray direction is rotated by the chief-ray angle around the y-axis to obtain the actual incidence angle &theta; on the filter; the cone is sampled on a Fibonacci spiral over the half-cone &theta;_h = arcsin(1 / 2F) and weighted by cos&theta; (aplanatic pupil).',
            '공진층 두께 d는 수직 입사에서 &lambda;&#8320;가 1차 피크에 오도록 d = &lambda;&#8320; / (2 n_eff)로 고정됩니다. 각 원뿔 샘플에서 주광선 각도만큼 y-축으로 회전시켜 필터 법선에 대한 실제 입사각 &theta;를 구하고, 반원뿔 &theta;_h = arcsin(1 / 2F) 위에서 피보나치 나선으로 샘플링한 후 cos&theta; 가중치(aplanatic pupil)로 평균합니다.'
          ) }}
        </p>
        <p>
          {{ t(
            'The cone-integrated transmittance is then',
            '원뿔 적분 투과율은 다음과 같습니다'
          ) }}
        </p>
        <p class="formula">
          T<sub>cone</sub>(&lambda;) = &sum;<sub>i</sub> w<sub>i</sub> T(&lambda;, &theta;<sub>i</sub>) / &sum;<sub>i</sub> w<sub>i</sub>.
        </p>
        <p>
          {{ t(
            'This reproduces the two effects emphasized in Goossens et al. (2018): a blue-shift of the centroid (because the average cos&theta;_int over the cone is less than the chief-ray value) and a broadening of the transmission peak (because each sample peaks at a different wavelength).',
            '이 두 가지 결과 — centroid의 blue-shift(원뿔 평균 cos&theta;_int가 주광선 값보다 작기 때문)와 투과 피크의 broadening(각 샘플이 서로 다른 파장에서 피크를 가지기 때문) — 가 Goossens et al. (2018)에서 강조한 핵심 효과입니다.'
          ) }}
        </p>
      </div>
    </details>

    <p class="ref-line">
      {{ t('Reference', '참고문헌') }}: T. Goossens et al.,
      <em>{{ t('Finite aperture correction for spectral cameras with integrated thin-film Fabry-Perot filters', '집적 박막 Fabry-Perot 필터 분광 카메라의 유한 조리개 보정') }}</em>,
      Appl. Opt. <strong>57</strong>(26):7539 (2018).
      DOI:
      <a href="https://doi.org/10.1364/AO.57.007539" target="_blank" rel="noopener">10.1364/AO.57.007539</a>.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

// Controls
const lambda0 = ref(550)
const nEff = ref(1.80)
const mirrorR = ref(0.90)
const craDeg = ref(15.0)
const fNumber = ref(2.0)
const nSamples = ref(61)
const samplingMode = ref<'rings' | 'fibonacci' | 'grid'>('rings')
const weightingMode = ref<'uniform' | 'cosine' | 'cos4' | 'gaussian'>('cos4')

const colorbarGradientId = 'fpcs-cb-' + Math.random().toString(36).slice(2, 9)

// Viridis-like colormap (sampled 7 stops, dark purple -> teal -> yellow)
const colormap = [
  '#440154', '#46337e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725',
]
function colorAt(t: number): string {
  // t in [0, 1]
  const c = Math.max(0, Math.min(1, t))
  const idx = c * (colormap.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(colormap.length - 1, lo + 1)
  const f = idx - lo
  // Interpolate hex colors
  const a = colormap[lo]
  const b = colormap[hi]
  const ar = parseInt(a.slice(1, 3), 16)
  const ag = parseInt(a.slice(3, 5), 16)
  const ab = parseInt(a.slice(5, 7), 16)
  const br = parseInt(b.slice(1, 3), 16)
  const bg = parseInt(b.slice(3, 5), 16)
  const bb = parseInt(b.slice(5, 7), 16)
  const r = Math.round(ar + (br - ar) * f)
  const g = Math.round(ag + (bg - ag) * f)
  const bl = Math.round(ab + (bb - ab) * f)
  return `rgb(${r},${g},${bl})`
}
const colorbarStops = computed(() => {
  const stops: string[] = []
  for (let i = 0; i <= 12; i++) stops.push(colorAt(i / 12))
  return stops
})

// Chart geometry
const chartW = 720
const chartH = 320
const padL = 48
const padR = 18
const padT = 14
const padB = 36
const plotW = chartW - padL - padR
const plotH = chartH - padT - padB

// Wavelength window: centered on lambda0 with width tied to expected shift+FWHM
const windowHalfWidth = computed(() => {
  // span roughly 4x FWHM_normal plus the maximum expected shift
  const F = 4 * mirrorR.value / Math.pow(1 - mirrorR.value, 2)
  const finesse = Math.PI * Math.sqrt(F) / 2
  const fwhmNormal = lambda0.value / finesse
  // Approx max angle inside filter
  const sinThetaMax = Math.sin((craDeg.value * Math.PI) / 180) + 1 / (2 * fNumber.value)
  const sinIntMax = Math.min(0.95, sinThetaMax / nEff.value)
  const cosIntMin = Math.sqrt(Math.max(1e-6, 1 - sinIntMax * sinIntMax))
  const maxShift = lambda0.value * (1 - cosIntMin)
  return Math.max(8 * fwhmNormal, 1.4 * maxShift + 4 * fwhmNormal, 6)
})

const wlMin = computed(() => lambda0.value - windowHalfWidth.value)
const wlMax = computed(() => lambda0.value + 0.35 * windowHalfWidth.value)

const halfConeDeg = computed(() => (Math.asin(1 / (2 * fNumber.value)) * 180) / Math.PI)

// Cone samples in pupil coordinates -> sensor-frame direction (dx, dy)
// Each sample also carries its theta_local (radial pupil coordinate) and a ring index
// (for ring-pattern sampling; -1 otherwise) so we can draw arcs connecting same-ring samples.
type Sample = {
  theta: number
  weight: number
  dx: number
  dy: number
  thetaLocal: number
  ring: number
  ringSize: number
}

function pupilWeight(thetaLocal: number, thetaH: number): number {
  switch (weightingMode.value) {
    case 'uniform':
      return 1
    case 'cosine':
      return Math.cos(thetaLocal)
    case 'cos4': {
      const c = Math.cos(thetaLocal)
      return c * c * c * c
    }
    case 'gaussian': {
      const sigma = thetaH / 1.5
      return Math.exp(-(thetaLocal * thetaLocal) / (2 * sigma * sigma))
    }
  }
}

const coneSamples = computed<Sample[]>(() => {
  const samples: Sample[] = []
  const N = nSamples.value
  const thetaH = (halfConeDeg.value * Math.PI) / 180
  const craRad = (craDeg.value * Math.PI) / 180
  const cosA = Math.cos(craRad)
  const sinA = Math.sin(craRad)

  const pushSample = (thetaLocal: number, phi: number, ring: number, ringSize: number) => {
    const sinL = Math.sin(thetaLocal)
    const cosL = Math.cos(thetaLocal)
    const dx = sinL * Math.cos(phi)
    const dy = sinL * Math.sin(phi)
    const dz = cosL
    const dxs = cosA * dx + sinA * dz
    const dys = dy
    const dzs = -sinA * dx + cosA * dz
    const cosTheta = Math.max(-1, Math.min(1, dzs))
    const theta = Math.acos(cosTheta)
    const w = Math.max(0, pupilWeight(thetaLocal, thetaH))
    samples.push({ theta, weight: w, dx: dxs, dy: dys, thetaLocal, ring, ringSize })
  }

  if (samplingMode.value === 'fibonacci') {
    const goldenAngle = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < N; i++) {
      const r = N === 1 ? 0 : Math.sqrt((i + 0.5) / N)
      const phi = i * goldenAngle
      pushSample(r * thetaH, phi, -1, 0)
    }
  } else if (samplingMode.value === 'rings') {
    // 1 center sample at theta_local = 0 + nRings concentric outer rings with
    // equal-area centroid radii. Sample count per ring is proportional to ring
    // index so the total comes out close to N. The center sample avoids the
    // asymmetric "single off-center point" that proportional scaling produces
    // for the innermost ring.
    pushSample(0, 0, 0, 1)
    const remaining = N - 1
    if (remaining > 0) {
      const nRings = Math.max(1, Math.round(Math.sqrt(N / Math.PI)))
      const totalWeight = (nRings * (nRings + 1)) / 2
      let assigned = 0
      for (let k = 1; k <= nRings; k++) {
        let m = Math.max(1, Math.round((k * remaining) / totalWeight))
        if (k === nRings) m = Math.max(1, remaining - assigned)
        assigned += m
        const rNorm = Math.sqrt((k - 0.5) / nRings) // equal-area annulus centroid
        const thetaLocal = rNorm * thetaH
        const phi0 = (k * Math.PI) / Math.max(1, nRings)
        for (let j = 0; j < m; j++) {
          const phi = phi0 + (2 * Math.PI * j) / m
          pushSample(thetaLocal, phi, k, m)
        }
      }
    }
  } else {
    // Polar grid: structured (n_theta x n_phi)
    const nTheta = Math.max(1, Math.round(Math.sqrt(N)))
    const nPhi = Math.max(1, Math.round(N / nTheta))
    for (let i = 0; i < nTheta; i++) {
      const rNorm = (i + 0.5) / nTheta
      const thetaLocal = rNorm * thetaH
      for (let j = 0; j < nPhi; j++) {
        const phi = (2 * Math.PI * j) / nPhi
        pushSample(thetaLocal, phi, i, nPhi)
      }
    }
  }

  // Normalize weights
  const wSum = samples.reduce((s, x) => s + x.weight, 0) || 1
  for (const s of samples) s.weight /= wSum
  return samples
})

// Build smooth arcs connecting samples on the same ring (for visualization)
const ringArcPaths = computed<string[]>(() => {
  if (samplingMode.value !== 'rings') return []
  const samples = coneSamples.value
  const thetaH = (halfConeDeg.value * Math.PI) / 180
  const craRad = (craDeg.value * Math.PI) / 180
  const cosA = Math.cos(craRad)
  const sinA = Math.sin(craRad)
  // Collect unique ring indices and their theta_local
  const ringMap = new Map<number, number>()
  for (const s of samples) {
    if (s.ring >= 0 && !ringMap.has(s.ring)) ringMap.set(s.ring, s.thetaLocal)
  }
  const paths: string[] = []
  for (const [, thetaLocal] of ringMap) {
    if (thetaLocal <= 1e-4) continue
    // Trace full ring (phi 0..2pi) at fixed thetaLocal, transformed to screen
    let d = ''
    const Nphi = 96
    const sinL = Math.sin(thetaLocal)
    const cosL = Math.cos(thetaLocal)
    for (let i = 0; i <= Nphi; i++) {
      const phi = (2 * Math.PI * i) / Nphi
      const dx = sinL * Math.cos(phi)
      const dy = sinL * Math.sin(phi)
      const dz = cosL
      const dxs = cosA * dx + sinA * dz
      const dys = dy
      const x = tvCx + (dxs / tvRangeSin.value) * tvR
      const y = tvCy - (dys / tvRangeSin.value) * tvR
      d += (i === 0 ? 'M ' : ' L ') + x.toFixed(2) + ' ' + y.toFixed(2)
    }
    paths.push(d)
  }
  return paths
})

// Top-view geometry (sensor-frame direction cosines)
const tvW = 320
const tvH = 320
const tvCx = tvW / 2
const tvCy = tvH / 2
const tvR = 138 // outer radius in px

// Half-angle range that the top view should accommodate (sin theta units)
const tvRangeSin = computed(() => {
  const craRad = (craDeg.value * Math.PI) / 180
  const thetaH = (halfConeDeg.value * Math.PI) / 180
  // Outer envelope: chief ray sin + half-cone sin, with small margin
  const v = Math.sin(craRad) + Math.sin(thetaH)
  return Math.max(0.05, v * 1.15)
})

function tvSinToR(sinVal: number): number {
  return (Math.abs(sinVal) / tvRangeSin.value) * tvR
}

function tvAngleToR(angDeg: number): number {
  return tvSinToR(Math.sin((angDeg * Math.PI) / 180))
}

// Reference angle rings: pick a sensible step from the range
const tvRefAngles = computed(() => {
  const maxDeg = (Math.asin(tvRangeSin.value) * 180) / Math.PI
  const step = maxDeg > 30 ? 10 : maxDeg > 15 ? 5 : maxDeg > 6 ? 2 : 1
  const angles: number[] = []
  for (let a = step; a <= maxDeg + 0.001; a += step) angles.push(a)
  return angles
})

// Chief ray projected onto the (x, y) sensor plane
const chiefScreen = computed(() => {
  const craRad = (craDeg.value * Math.PI) / 180
  const x = tvCx + tvSinToR(Math.sin(craRad))
  const y = tvCy // chief ray rotates around y -> dys = 0
  return { x, y }
})

// Outline of the cone in the sensor frame: parameterize phi_local in [0, 2pi)
// and trace (dxs, dys) at theta_local = theta_h
const coneOutlinePath = computed(() => {
  const thetaH = (halfConeDeg.value * Math.PI) / 180
  const craRad = (craDeg.value * Math.PI) / 180
  const cosA = Math.cos(craRad)
  const sinA = Math.sin(craRad)
  const sinL = Math.sin(thetaH)
  const cosL = Math.cos(thetaH)
  let d = ''
  const N = 96
  for (let i = 0; i <= N; i++) {
    const phi = (2 * Math.PI * i) / N
    const dx = sinL * Math.cos(phi)
    const dy = sinL * Math.sin(phi)
    const dz = cosL
    const dxs = cosA * dx + sinA * dz
    const dys = dy
    const x = tvCx + (dxs / tvRangeSin.value) * tvR
    const y = tvCy - (dys / tvRangeSin.value) * tvR
    d += (i === 0 ? 'M ' : ' L ') + x.toFixed(2) + ' ' + y.toFixed(2)
  }
  return d + ' Z'
})

// Sample point screen positions + visual encoding by weight
const samplePoints = computed(() => {
  const samples = coneSamples.value
  if (samples.length === 0) return []
  let wMax = 0
  for (const s of samples) if (s.weight > wMax) wMax = s.weight
  wMax = wMax || 1
  return samples.map(s => {
    const x = tvCx + (s.dx / tvRangeSin.value) * tvR
    const y = tvCy - (s.dy / tvRangeSin.value) * tvR
    const rel = s.weight / wMax
    // Size 1.8-4.6 px to give a clearly visible weight cue
    const r = 1.8 + 2.8 * rel
    const opacity = 0.85
    // Viridis colormap by relative weight
    const color = colorAt(rel)
    return { x, y, r, opacity, color }
  })
})

// Airy transmittance at given wavelength and angle (radians)
function airyT(wl: number, theta: number): number {
  const sinT = Math.sin(theta)
  const sinInt = sinT / nEff.value
  if (Math.abs(sinInt) >= 1) return 0
  const cosInt = Math.sqrt(1 - sinInt * sinInt)
  // Cavity thickness chosen so 4*pi*n_eff*d / lambda0 = 2*pi  -> d = lambda0 / (2*n_eff)
  const d = lambda0.value / (2 * nEff.value)
  const delta = (4 * Math.PI * nEff.value * d * cosInt) / wl
  const sinHalf = Math.sin(delta / 2)
  const F = 4 * mirrorR.value / Math.pow(1 - mirrorR.value, 2)
  return 1 / (1 + F * sinHalf * sinHalf)
}

// Sampled spectra
const NWL = 401
const wavelengths = computed(() => {
  const arr: number[] = []
  const a = wlMin.value
  const b = wlMax.value
  for (let i = 0; i < NWL; i++) arr.push(a + ((b - a) * i) / (NWL - 1))
  return arr
})

const spectrumNormal = computed(() => wavelengths.value.map(wl => airyT(wl, 0)))

const craRad = computed(() => (craDeg.value * Math.PI) / 180)
const spectrumPlaneWave = computed(() => wavelengths.value.map(wl => airyT(wl, craRad.value)))

const spectrumCone = computed(() => {
  const samples = coneSamples.value
  return wavelengths.value.map(wl => {
    let s = 0
    for (const sam of samples) s += sam.weight * airyT(wl, sam.theta)
    return s
  })
})

// Peak + FWHM helpers (parabolic refinement at the discrete maximum)
function peakAndFWHM(ys: number[]): { lambda: number; T: number; fwhm: number | null } | null {
  const wls = wavelengths.value
  if (ys.length === 0) return null
  let iMax = 0
  for (let i = 1; i < ys.length; i++) if (ys[i] > ys[iMax]) iMax = i
  // Parabolic interp at iMax
  let lambdaPeak = wls[iMax]
  let tPeak = ys[iMax]
  if (iMax > 0 && iMax < ys.length - 1) {
    const y0 = ys[iMax - 1], y1 = ys[iMax], y2 = ys[iMax + 1]
    const denom = y0 - 2 * y1 + y2
    if (Math.abs(denom) > 1e-12) {
      const dx = (0.5 * (y0 - y2)) / denom
      lambdaPeak = wls[iMax] + dx * (wls[1] - wls[0])
      tPeak = y1 - 0.25 * (y0 - y2) * dx
    }
  }
  // FWHM: find points where ys crosses tPeak/2 either side of iMax
  const half = tPeak / 2
  let lLo: number | null = null
  let lHi: number | null = null
  for (let i = iMax; i > 0; i--) {
    if (ys[i] >= half && ys[i - 1] < half) {
      const frac = (half - ys[i - 1]) / (ys[i] - ys[i - 1])
      lLo = wls[i - 1] + frac * (wls[i] - wls[i - 1])
      break
    }
  }
  for (let i = iMax; i < ys.length - 1; i++) {
    if (ys[i] >= half && ys[i + 1] < half) {
      const frac = (ys[i] - half) / (ys[i] - ys[i + 1])
      lHi = wls[i] + frac * (wls[i + 1] - wls[i])
      break
    }
  }
  const fwhm = lLo !== null && lHi !== null ? lHi - lLo : null
  return { lambda: lambdaPeak, T: tPeak, fwhm }
}

const planePeak = computed(() => peakAndFWHM(spectrumPlaneWave.value))
const conePeak = computed(() => peakAndFWHM(spectrumCone.value))
const planeFWHM = computed(() => planePeak.value?.fwhm ?? null)
const coneFWHM = computed(() => conePeak.value?.fwhm ?? null)

// Coordinate mappers
function lambdaToX(wl: number): number {
  return padL + ((wl - wlMin.value) / (wlMax.value - wlMin.value)) * plotW
}
function pctToY(pct: number): number {
  return padT + plotH - (pct / 100) * plotH
}

function buildPath(ys: number[]): string {
  const wls = wavelengths.value
  let path = ''
  for (let i = 0; i < ys.length; i++) {
    const x = lambdaToX(wls[i]).toFixed(2)
    const y = pctToY(ys[i] * 100).toFixed(2)
    path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`
  }
  return path
}

const pathNormal = computed(() => buildPath(spectrumNormal.value))
const pathPlaneWave = computed(() => buildPath(spectrumPlaneWave.value))
const pathCone = computed(() => buildPath(spectrumCone.value))

const xTicks = computed(() => {
  const a = wlMin.value
  const b = wlMax.value
  const span = b - a
  const step = span > 200 ? 50 : span > 80 ? 20 : span > 30 ? 10 : 5
  const first = Math.ceil(a / step) * step
  const ticks: number[] = []
  for (let v = first; v <= b; v += step) ticks.push(v)
  return ticks
})
</script>

<style scoped>
.fpcs-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.fpcs-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
  line-height: 1.5;
}
.controls-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-bottom: 16px;
}
.slider-group label {
  display: block;
  font-size: 0.85em;
  margin-bottom: 4px;
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
.ctrl-select {
  width: 100%;
  padding: 4px 6px;
  font-size: 0.85em;
  border-radius: 4px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
}
.topview-section {
  margin: 4px 0 16px;
  text-align: center;
}
.panel-title {
  margin: 0 0 6px;
  font-size: 0.92em;
  color: var(--vp-c-text-2);
  font-weight: 600;
}
.topview-svg {
  width: 100%;
  max-width: 320px;
  display: block;
  margin: 0 auto;
}
.topview-caption {
  max-width: 560px;
  margin: 6px auto 0;
  font-size: 0.8em;
  color: var(--vp-c-text-3);
  line-height: 1.5;
}
.chart-section {
  margin: 8px 0 16px;
}
.chart-svg {
  width: 100%;
  max-width: 720px;
  display: block;
  margin: 0 auto;
}
.axis-label {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.legend-label {
  font-size: 9.5px;
  fill: var(--vp-c-text-2);
}
.info-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.info-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}
.info-label {
  font-size: 0.78em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.info-value {
  font-weight: 600;
  font-size: 1em;
  font-family: var(--vp-font-family-mono);
}
.info-value.highlight {
  color: var(--vp-c-brand-1);
}
.info-sub {
  font-size: 0.75em;
  color: var(--vp-c-text-3);
  margin-top: 3px;
  font-family: var(--vp-font-family-mono);
}
.theory-block {
  margin: 8px 0 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
}
.theory-block summary {
  cursor: pointer;
  padding: 8px 12px;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
  user-select: none;
}
.theory-body {
  padding: 4px 14px 12px;
  font-size: 0.88em;
  line-height: 1.55;
  color: var(--vp-c-text-2);
}
.theory-body p {
  margin: 6px 0;
}
.theory-body .formula {
  font-family: var(--vp-font-family-mono);
  background: var(--vp-c-bg-soft);
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 0.92em;
}
.ref-line {
  font-size: 0.8em;
  color: var(--vp-c-text-3);
  margin: 0;
}
.ref-line a {
  color: var(--vp-c-brand-1);
}
</style>
