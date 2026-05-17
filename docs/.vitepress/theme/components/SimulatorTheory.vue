<template>
  <section v-if="entry" ref="theoryRoot" class="sim-theory tex2jax_process">
    <div class="sim-theory-eyebrow">{{ t('Physics Notes', '물리 수식과 이론') }}</div>
    <h2>{{ pick(entry.title) }}</h2>

    <div v-if="entry.intuition" class="sim-theory-intuition">
      <div class="intuition-eyebrow">{{ t('Plain-English Intuition', '쉽게 이해하기') }}</div>
      <p>{{ pick(entry.intuition) }}</p>
    </div>

    <p class="sim-theory-summary">{{ pick(entry.summary) }}</p>

    <div
      v-if="entry.assumptions?.length || entry.outputs?.length || entry.validationExamples?.length"
      class="sim-theory-standard-grid"
    >
      <div v-if="entry.assumptions?.length" class="sim-theory-card">
        <h3>{{ t('Assumptions', '모델 가정') }}</h3>
        <ul>
          <li v-for="item in entry.assumptions" :key="pick(item)">{{ pick(item) }}</li>
        </ul>
      </div>

      <div v-if="entry.outputs?.length" class="sim-theory-card">
        <h3>{{ t('Outputs', '출력값') }}</h3>
        <ul>
          <li v-for="item in entry.outputs" :key="pick(item)">{{ pick(item) }}</li>
        </ul>
      </div>

      <div v-if="entry.validationExamples?.length" class="sim-theory-card validation-card">
        <h3>{{ t('Validation Example', '검증 예제') }}</h3>
        <ul>
          <li v-for="item in entry.validationExamples" :key="pick(item)">{{ pick(item) }}</li>
        </ul>
      </div>
    </div>

    <div class="sim-theory-grid">
      <div class="sim-theory-card">
        <h3>{{ t('Core Equations', '핵심 수식') }}</h3>
        <div v-for="formula in entry.formulas" :key="formula.equation" class="formula-row">
          <div class="formula-label">{{ pick(formula.label) }}</div>
          <div class="formula-equation">
            $${{ formula.equation }}$$
          </div>
          
          <div v-if="formula.variables?.length" class="formula-variables">
            <ul>
              <li v-for="v in formula.variables" :key="v.symbol">
                <span class="var-symbol">\({{ v.symbol }}\)</span>: {{ pick(v.description) }}
              </li>
            </ul>
          </div>

          <p class="formula-note">{{ pick(formula.note) }}</p>
        </div>
      </div>

      <div class="sim-theory-card">
        <h3>{{ t('Model Interpretation', '모델 해석') }}</h3>
        <ul>
          <li v-for="item in entry.concepts" :key="pick(item)">{{ pick(item) }}</li>
        </ul>
      </div>
    </div>

    <div v-if="entry.sections?.length" class="sim-theory-detail-grid">
      <div v-for="section in entry.sections" :key="pick(section.title)" class="sim-theory-card">
        <h3>{{ pick(section.title) }}</h3>
        <ul>
          <li v-for="item in section.items" :key="pick(item)">{{ pick(item) }}</li>
        </ul>
      </div>
    </div>

    <div class="sim-theory-card references-card">
      <h3>{{ t('References', '관련 논문 및 레퍼런스') }}</h3>
      <ul>
        <li v-for="ref in entry.references" :key="ref.label">
          <a v-if="ref.href" :href="ref.href" target="_blank" rel="noreferrer">{{ ref.label }}</a>
          <span v-else>{{ ref.label }}</span>
          <span v-if="ref.note"> — {{ pick(ref.note) }}</span>
        </li>
      </ul>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, watch, nextTick, ref } from 'vue'
import { useLocale } from '../composables/useLocale'

const props = defineProps<{ slug: string }>()
const { isKo, t } = useLocale()
const theoryRoot = ref<HTMLElement | null>(null)

function pick(value: Localized) {
  return isKo.value ? value.ko : value.en
}

const waitForMathJax = async (timeoutMs = 8000) => {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const mj = (window as any).MathJax
    if (mj?.startup?.promise && typeof mj.typesetPromise === 'function') {
      await mj.startup.promise
      return mj
    }
    await new Promise((resolve) => setTimeout(resolve, 50))
  }
  return null
}

const typesetMath = async () => {
  if (typeof window === 'undefined') return
  await nextTick()
  const mj = await waitForMathJax()
  if (mj) {
    try {
      const target = theoryRoot.value
      if (target) {
        mj.typesetClear?.([target])
        await mj.typesetPromise([target])
      } else {
        await mj.typesetPromise()
      }
    } catch {
      /* ignore typeset errors */
    }
  }
}

onMounted(typesetMath)
watch([() => props.slug, isKo], typesetMath)

type Localized = { en: string; ko: string }

interface Variable {
  symbol: string
  description: Localized
}

interface Formula {
  label: Localized
  equation: string
  variables?: Variable[]
  note: Localized
}

interface Reference {
  label: string
  href?: string
  note?: Localized
}

interface TheorySection {
  title: Localized
  items: Localized[]
}

interface TheoryEntry {
  title: Localized
  summary: Localized
  intuition?: Localized
  assumptions?: Localized[]
  outputs?: Localized[]
  validationExamples?: Localized[]
  formulas: Formula[]
  concepts: Localized[]
  sections?: TheorySection[]
  references: Reference[]
}

const refs = {
  catrysse2002: {
    label: 'Catrysse & Wandell, "Optical efficiency of image sensor pixels", JOSA A, 2002',
    href: 'https://doi.org/10.1364/JOSAA.19.001610',
  },
  agranov2003: {
    label: 'Agranov, Berezin & Tsai, "Crosstalk and microlens study in a color CMOS image sensor", IEEE TED, 2003',
    href: 'https://doi.org/10.1109/TED.2002.806473',
  },
  green2008: {
    label: 'Green, "Self-consistent optical parameters of intrinsic silicon at 300 K", Solar Energy Materials and Solar Cells, 2008',
    href: 'https://doi.org/10.1016/j.solmat.2008.06.009',
  },
  macleod: {
    label: 'Macleod, Thin-Film Optical Filters, 5th ed., CRC Press, 2017',
    href: 'https://www.routledge.com/Thin-Film-Optical-Filters/Macleod/p/book/9781138198241',
  },
  bornWolf: {
    label: 'Born & Wolf, Principles of Optics, 7th ed., Cambridge University Press, 1999',
    href: 'https://doi.org/10.1017/CBO9781139644181',
  },
  goossens2018: {
    label: 'Goossens et al., "Finite aperture correction for spectral cameras with integrated thin-film Fabry-Perot filters", Applied Optics, 2018',
    href: 'https://doi.org/10.1364/AO.57.007539',
  },
  hwang2023: {
    label: 'Hwang & Kim, "A Numerical Method of Aligning the Optical Stacks for All Pixels", Sensors, 2023',
    href: 'https://doi.org/10.3390/s23020702',
  },
  yokogawa2017: {
    label: 'Yokogawa et al., "IR sensitivity enhancement of CMOS Image Sensor with diffractive light trapping pixels", Scientific Reports, 2017',
    href: 'https://doi.org/10.1038/s41598-017-04200-y',
  },
  han2020: {
    label: 'Han, Chiou & Lin, "Deep trench isolation and inverted pyramid array structures...", Sensors, 2020',
    href: 'https://doi.org/10.3390/s20113062',
  },
  ristoiu2020: {
    label: 'Ristoiu et al., "A DOE study of plasma etched microlens shape for CMOS image sensors", SPIE, 2020',
    href: 'https://doi.org/10.1117/12.2551857',
  },
  baillie2004: {
    label: 'Baillie & Gendler, "Zero-space microlenses for CMOS image sensors", SPIE, 2004',
    href: 'https://doi.org/10.1117/12.533453',
  },
  jin2011: {
    label: 'Jin, Liu & Yang, "Design, characterization and evaluation of high performance 2.8 um pitch zero space microlens", Optics Communications, 2011',
    href: 'https://doi.org/10.1016/j.optcom.2010.11.073',
  },
  tan2020: {
    label: 'Tan, Goh & Kim, "Microfabrication of Microlens by Timed-Development-and-Thermal-Reflow (TDTR) Process for Projection Lithography", Micromachines, 2020',
    href: 'https://doi.org/10.3390/mi11030277',
  },
  emva: {
    label: 'EMVA 1288 Standard, Release 4.0',
    href: 'https://www.emva.org/standards-technology/emva-1288/',
  },
  iso12233: {
    label: 'ISO 12233:2024, Digital cameras - Resolution and spatial frequency responses',
    href: 'https://www.iso.org/standard/88626.html',
  },
  nistCie: {
    label: 'NIST, "CIE Fundamentals for Color Measurements"',
    href: 'https://www.nist.gov/publications/cie-fundamentals-color-measurements-0',
  },
  mccamy1992: {
    label: 'McCamy, "Correlated Color Temperature as an Explicit Function of Chromaticity Coordinates", Color Research & Application, 1992',
    href: 'https://doi.org/10.1002/col.5080170211',
  },
  blockstein2010: {
    label: 'Blockstein & Yadid-Pecht, "Crosstalk quantification, analysis, and trends in CMOS image sensors", Applied Optics, 2010',
    href: 'https://doi.org/10.1364/AO.49.004483',
  },
}

const theoryEntries: Record<string, TheoryEntry> = {
  'tmm-qe': {
    title: { en: 'Transfer-Matrix QE Model', ko: '전달 행렬 기반 QE 모델' },
    summary: {
      en: 'The calculator treats the BSI pixel stack as a coherent one-dimensional thin-film system: air, polymer microlens proxy, planarization, color filter, BARL, and silicon. It computes wavelength- and angle-dependent reflection, transmission, and layer absorption; the plotted QE is the silicon-layer absorption proxy.',
      ko: '이 계산기는 BSI 픽셀 스택을 air, polymer microlens proxy, planarization, color filter, BARL, silicon으로 이어지는 coherent 1차원 박막계로 취급합니다. 파장과 입사각에 따른 반사, 투과, 레이어별 흡수를 계산하고, 표시되는 QE는 silicon layer absorption proxy입니다.',
    },
    intuition: {
      en: 'Think of the pixel stack as many transparent plates. Each interface reflects a small wave; each layer adds phase delay and absorption. The transfer matrix keeps the amplitude and phase of all forward and backward waves coherent, so a thin BARL layer can increase QE by canceling reflection at one wavelength while hurting another wavelength.',
      ko: '픽셀 스택을 여러 장의 투명한 판으로 생각하면 됩니다. 각 계면은 작은 반사파를 만들고, 각 레이어는 위상 지연과 흡수를 더합니다. 전달 행렬은 전진파와 후진파의 진폭 및 위상을 coherent하게 추적하므로, 얇은 BARL 레이어가 한 파장에서는 반사를 상쇄해 QE를 올리면서 다른 파장에서는 QE를 낮출 수도 있습니다.',
    },
    assumptions: [
      { en: 'Every layer is laterally infinite, planar, homogeneous inside the layer, and coherent over the simulated optical path.', ko: '모든 레이어는 횡방향으로 무한하고, 평탄하며, 레이어 내부가 균질하고, 시뮬레이션 광경로에서 coherent하다고 가정합니다.' },
      { en: 'The plotted QE is silicon optical absorption, not collected charge after carrier transport.', ko: '표시 QE는 silicon optical absorption이며, carrier transport 후 수집된 전하가 아닙니다.' },
      { en: 'Microlens, metal grid, DTI, and color-filter relief are collapsed into planar effective films.', ko: 'Microlens, metal grid, DTI, color-filter relief는 평면 effective film으로 축약됩니다.' },
    ],
    outputs: [
      { en: 'Wavelength-dependent $R(\\lambda)$, $T(\\lambda)$, parasitic layer absorption, and silicon absorption by color channel.', ko: '파장별 $R(\\lambda)$, $T(\\lambda)$, 기생 레이어 흡수, 색 채널별 silicon absorption을 출력합니다.' },
      { en: 'Angle and polarization trends for a flat stack, useful before switching to RCWA/FDTD for lateral geometry.', ko: '평탄 스택의 입사각/편광 경향을 보여주며, lateral geometry를 RCWA/FDTD로 계산하기 전 선별에 유용합니다.' },
    ],
    validationExamples: [
      { en: 'With all coating layers removed, compare the normal-incidence air/Si reflection against $R=|(1-n_{Si})/(1+n_{Si})|^2$ at the selected wavelength.', ko: '코팅 레이어를 모두 제거한 조건에서는 선택 파장에서 normal-incidence air/Si 반사율을 $R=|(1-n_{Si})/(1+n_{Si})|^2$와 비교합니다.' },
      { en: 'For a passive stack, verify $R+T+\\sum_j A_j\\approx1$ before interpreting QE peaks.', ko: '수동 스택에서는 QE peak를 해석하기 전에 $R+T+\\sum_j A_j\\approx1$인지 확인합니다.' },
    ],
    formulas: [
      {
        label: { en: 'Complex refractive index', ko: '복소 굴절률' },
        equation: '\\tilde{n}_j(\\lambda)=n_j(\\lambda)+i k_j(\\lambda)',
        variables: [
          { symbol: '\\tilde{n}_j', description: { en: 'Complex refractive index of layer $j$', ko: '레이어 $j$의 복소 굴절률' } },
          { symbol: 'n_j', description: { en: 'Real refractive index controlling phase velocity', ko: '위상 속도를 결정하는 실수 굴절률' } },
          { symbol: 'k_j', description: { en: 'Extinction coefficient controlling absorption', ko: '흡수를 결정하는 소광 계수' } },
          { symbol: '\\lambda', description: { en: 'Vacuum wavelength', ko: '진공 파장' } },
        ],
        note: { en: 'The browser model uses tabulated, Sellmeier, Cauchy, or constant material data depending on the layer.', ko: '브라우저 모델은 레이어에 따라 tabulated, Sellmeier, Cauchy, constant material data를 사용합니다.' },
      },
      {
        label: { en: 'Snell relation in each film', ko: '각 박막의 스넬 관계' },
        equation: '\\tilde{n}_0\\sin\\theta_0=\\tilde{n}_j\\sin\\theta_j, \\quad \\cos\\theta_j=\\sqrt{1-\\left(\\frac{\\tilde{n}_0\\sin\\theta_0}{\\tilde{n}_j}\\right)^2}',
        variables: [
          { symbol: '\\theta_0', description: { en: 'External angle of incidence selected in the simulator', ko: '시뮬레이터에서 선택한 외부 입사각' } },
          { symbol: '\\theta_j', description: { en: 'Internal propagation angle in layer $j$', ko: '레이어 $j$ 내부 전파각' } },
          { symbol: '\\tilde{n}_0', description: { en: 'Incident medium index, air in this simulator', ko: '입사 매질 굴절률; 이 시뮬레이터에서는 air' } },
        ],
        note: { en: 'At oblique incidence, every layer gets a different optical path length and different s/p polarization response.', ko: '경사 입사에서는 각 레이어의 광경로와 s/p 편광 응답이 달라집니다.' },
      },
      {
        label: { en: 'Layer phase thickness', ko: '레이어 위상 두께' },
        equation: '\\delta_j = \\frac{2\\pi}{\\lambda}\\tilde{n}_j d_j\\cos\\theta_j',
        variables: [
          { symbol: '\\delta_j', description: { en: 'Complex phase thickness of layer $j$', ko: '레이어 $j$의 복소 위상 두께' } },
          { symbol: 'd_j', description: { en: 'Physical thickness of layer $j$', ko: '레이어 $j$의 물리적 두께' } },
          { symbol: '\\cos\\theta_j', description: { en: 'Obliquity factor inside the layer', ko: '레이어 내부 경사 입사 보정 인자' } },
        ],
        note: { en: 'Interference fringes shift when layer thickness, wavelength, or angle changes because this phase term changes.', ko: '레이어 두께, 파장, 입사각이 바뀌면 이 위상 항이 변하기 때문에 간섭 fringe가 이동합니다.' },
      },
      {
        label: { en: 'Characteristic matrix', ko: '특성 행렬' },
        equation: 'M_j=\\begin{bmatrix}\\cos\\delta_j & -i\\sin\\delta_j/\\eta_j \\\\ -i\\eta_j\\sin\\delta_j & \\cos\\delta_j\\end{bmatrix}, \\quad M=\\prod_j M_j',
        variables: [
          { symbol: 'M_j', description: { en: 'Transfer matrix of layer $j$', ko: '레이어 $j$의 전달 행렬' } },
          { symbol: 'M', description: { en: 'Total stack transfer matrix', ko: '전체 스택 전달 행렬' } },
          { symbol: '\\eta_j', description: { en: 'Optical admittance; $\\eta_j=\\tilde{n}_j\\cos\\theta_j$ for s polarization and $\\eta_j=\\tilde{n}_j/\\cos\\theta_j$ for p polarization in this implementation', ko: '광학 어드미턴스; 이 구현에서는 s 편광에서 $\\eta_j=\\tilde{n}_j\\cos\\theta_j$, p 편광에서 $\\eta_j=\\tilde{n}_j/\\cos\\theta_j$' } },
        ],
        note: { en: 'The total matrix maps the boundary fields at the incident side to those at the substrate side.', ko: '전체 행렬은 입사측 경계장의 전기장/자기장을 기판측 경계장과 연결합니다.' },
      },
      {
        label: { en: 'Reflection and transmission amplitudes', ko: '반사 및 투과 진폭' },
        equation: 'r=\\frac{\\eta_0M_{00}+\\eta_0\\eta_sM_{01}-M_{10}-\\eta_sM_{11}}{\\eta_0M_{00}+\\eta_0\\eta_sM_{01}+M_{10}+\\eta_sM_{11}}, \\quad t=\\frac{2\\eta_0}{\\eta_0M_{00}+\\eta_0\\eta_sM_{01}+M_{10}+\\eta_sM_{11}}',
        variables: [
          { symbol: 'r,t', description: { en: 'Complex reflection and transmission amplitudes', ko: '복소 반사 및 투과 진폭' } },
          { symbol: '\\eta_0', description: { en: 'Incident-side optical admittance', ko: '입사측 광학 어드미턴스' } },
          { symbol: '\\eta_s', description: { en: 'Substrate-side optical admittance', ko: '기판측 광학 어드미턴스' } },
          { symbol: 'M_{mn}', description: { en: 'Elements of the total transfer matrix', ko: '전체 전달 행렬의 원소' } },
        ],
        note: { en: 'The implementation averages separate s and p calculations for unpolarized light.', ko: '비편광 조건에서는 s와 p 계산 결과를 평균합니다.' },
      },
      {
        label: { en: 'Power balance and layer absorption', ko: '파워 보존 및 레이어 흡수' },
        equation: 'R=|r|^2, \\quad T=\\frac{\\operatorname{Re}(\\eta_s)}{\\operatorname{Re}(\\eta_0)}|t|^2, \\quad A_j=P_{\\text{top},j}-P_{\\text{bot},j}',
        variables: [
          { symbol: 'R,T', description: { en: 'Reflected and transmitted power fractions', ko: '반사 및 투과 파워 비율' } },
          { symbol: 'A_j', description: { en: 'Absorbed power fraction assigned to layer $j$', ko: '레이어 $j$에 배정된 흡수 파워 비율' } },
          { symbol: 'P_{\\text{top},j}, P_{\\text{bot},j}', description: { en: 'Normalized Poynting flux at the top and bottom of layer $j$', ko: '레이어 $j$ 상단 및 하단의 정규화된 포인팅 플럭스' } },
        ],
        note: { en: 'For a consistent passive stack, $R+T+\\sum_j A_j\\approx1$; numerical clipping only prevents tiny negative absorption artifacts.', ko: '수동 스택에서는 $R+T+\\sum_j A_j\\approx1$이어야 하며, 수치 clipping은 작은 음수 흡수 artifact를 막기 위한 것입니다.' },
      },
      {
        label: { en: 'Displayed QE proxy', ko: '표시 QE proxy' },
        equation: 'QE_{c,\\text{opt}}(\\lambda)=100\\,A_{\\text{Si},c}(\\lambda)',
        variables: [
          { symbol: 'QE_{c,\\text{opt}}', description: { en: 'Displayed optical QE proxy for color channel $c$', ko: '색 채널 $c$에 대해 표시되는 광학 QE proxy' } },
          { symbol: 'A_{\\text{Si},c}', description: { en: 'Absorption fraction in the silicon layer of the selected color-filter stack', ko: '선택한 컬러 필터 스택의 silicon layer 흡수율' } },
          { symbol: 'c', description: { en: 'Color channel: red, green, or blue', ko: '색 채널: red, green, blue' } },
        ],
        note: { en: 'This is an optical upper bound: carrier collection efficiency, recombination, electrical conversion gain, and pixel aperture effects are outside the model.', ko: '이는 광학적 상한입니다. 전하 수집 효율, 재결합, 전기적 conversion gain, pixel aperture effect는 모델 밖입니다.' },
      },
    ],
    concepts: [
      { en: 'Best for flat BSI stack trends, BARL thickness tuning, color-filter absorption intuition, and angle/polarization sensitivity screening.', ko: '평탄 BSI 스택 경향, BARL 두께 조정, 컬러 필터 흡수 직관, 입사각/편광 민감도 screening에 적합합니다.' },
      { en: 'The color-filter curves in this browser tool are compact pedagogical spectra, not proprietary process-verified pigment data.', ko: '이 브라우저 도구의 color-filter curve는 교육용 compact spectrum이며, proprietary process-verified pigment data가 아닙니다.' },
      { en: 'The polymer microlens layer is treated as a planar film, so focusing, lens shift, fill factor, and CRA-dependent spot displacement are not solved.', ko: 'Polymer microlens layer는 평면 박막으로 처리되므로 focusing, lens shift, fill factor, CRA-dependent spot displacement는 풀지 않습니다.' },
    ],
    sections: [
      {
        title: { en: 'How To Read The Plot', ko: '그래프 해석 방법' },
        items: [
          { en: 'A QE peak moving with silicon thickness usually indicates interference and absorption-depth tradeoff, not a direct change in electrical quantum yield.', ko: '실리콘 두께에 따라 QE peak가 이동하면 보통 전기적 quantum yield 변화가 아니라 간섭과 absorption-depth tradeoff를 의미합니다.' },
          { en: 'A BARL layer that improves green may reduce blue or red because the phase-cancellation condition is wavelength dependent.', ko: 'Green을 개선하는 BARL layer가 blue 또는 red를 낮출 수 있는데, phase-cancellation 조건이 파장 의존적이기 때문입니다.' },
          { en: 'Angle sweeps should be interpreted as planar-stack angular response; they are not a full chief-ray-angle pixel model.', ko: '입사각 sweep은 planar-stack angular response로 해석해야 하며, 전체 chief-ray-angle pixel model은 아닙니다.' },
        ],
      },
      {
        title: { en: 'Calibration Checklist', ko: '보정 체크리스트' },
        items: [
          { en: 'Replace simplified $n,k$ data with process-specific ellipsometry for BARL, color filter, polymer, and silicon if quantitative accuracy matters.', ko: '정량 정확도가 필요하면 BARL, color filter, polymer, silicon의 단순화된 $n,k$ 데이터를 공정별 ellipsometry 데이터로 교체해야 합니다.' },
          { en: 'Compare $R(\\lambda)$ and $T(\\lambda)$ against wafer optical metrology before trusting silicon absorption trends.', ko: 'Silicon absorption 경향을 신뢰하기 전에 wafer optical metrology의 $R(\\lambda)$ 및 $T(\\lambda)$와 비교해야 합니다.' },
          { en: 'Use RCWA/FDTD when the stack has lateral structures: metal grids, DTI, color-filter relief, sub-wavelength texture, or microlens curvature.', ko: 'Metal grid, DTI, color-filter relief, sub-wavelength texture, microlens curvature 같은 lateral structure가 있으면 RCWA/FDTD를 사용해야 합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'No diffraction, scattering, lateral crosstalk, finite pixel aperture, or microlens focusing is included.', ko: '회절, 산란, lateral crosstalk, finite pixel aperture, microlens focusing은 포함하지 않습니다.' },
          { en: 'Silicon absorption is not the same as collected charge; depletion depth, recombination, and carrier collection probability are omitted.', ko: 'Silicon absorption은 collected charge와 동일하지 않습니다. Depletion depth, recombination, carrier collection probability가 빠져 있습니다.' },
          { en: 'The calculation assumes laterally infinite coherent films, so roughness, thickness non-uniformity, and incoherent thick-layer effects are approximated away.', ko: '계산은 횡방향 무한 coherent film을 가정하므로 roughness, thickness non-uniformity, incoherent thick-layer effect는 근사적으로 제외됩니다.' },
        ],
      },
    ],
    references: [
      {
        ...refs.macleod,
        note: { en: 'Reference for thin-film characteristic matrices, optical admittance, and multilayer coating interpretation.', ko: '박막 characteristic matrix, optical admittance, multilayer coating 해석의 기준 레퍼런스입니다.' },
      },
      {
        ...refs.green2008,
        note: { en: 'Useful source for wavelength-dependent intrinsic silicon optical constants.', ko: '파장 의존 intrinsic silicon optical constant에 유용한 레퍼런스입니다.' },
      },
      {
        ...refs.catrysse2002,
        note: { en: 'Connects optical efficiency modeling to image-sensor pixel stacks and QE interpretation.', ko: 'Optical efficiency modeling을 image-sensor pixel stack 및 QE 해석과 연결합니다.' },
      },
      {
        ...refs.bornWolf,
        note: { en: 'Background for electromagnetic boundary conditions, polarization, and coherent optics.', ko: '전자기 경계조건, 편광, coherent optics의 배경 이론입니다.' },
      },
    ],
  },
  'barl-optimizer': {
    title: { en: 'Thin-Film Anti-Reflection Theory', ko: '박막 반사 방지 이론' },
    summary: {
      en: 'The BARL optimizer searches thin-film thicknesses that reduce reflection and parasitic loss before photons reach silicon. It is a constrained multilayer interference problem: each film changes phase, admittance, absorption, and therefore the color-channel QE tradeoff.',
      ko: 'BARL optimizer는 광자가 silicon에 도달하기 전에 생기는 반사와 기생 손실을 줄이는 박막 두께를 탐색합니다. 이는 제한 조건이 있는 multilayer interference 문제입니다. 각 박막은 위상, 어드미턴스, 흡수율을 바꾸고 결국 색 채널 QE tradeoff를 만듭니다.',
    },
    intuition: {
      en: 'Anti-reflection layers work like timing control for echoes. A wave reflected from the top of a film can meet a wave reflected from the bottom of the film with opposite phase, canceling the return wave. But the cancellation only holds over a limited wavelength and angle range, so a CIS BARL stack is always a compromise between blue, green, red, angle, process limits, and absorption.',
      ko: '반사 방지 박막은 빛의 echo timing을 조절하는 장치와 비슷합니다. 박막 윗면에서 반사된 파동과 아랫면에서 반사된 파동이 반대 위상으로 만나면 되돌아가는 파동이 상쇄됩니다. 하지만 이 상쇄는 제한된 파장/각도 범위에서만 성립하므로, CIS BARL 스택은 항상 blue, green, red, angle, 공정 한계, 흡수 사이의 절충입니다.',
    },
    formulas: [
      {
        label: { en: 'Bare-interface reflection', ko: 'Bare 계면 반사' },
        equation: 'r_{0s}=\\frac{n_0-n_s}{n_0+n_s}, \\quad R_{0s}=|r_{0s}|^2',
        variables: [
          { symbol: 'r_{0s}', description: { en: 'Complex amplitude reflection coefficient between incident medium and substrate', ko: '입사 매질과 기판 사이의 복소 진폭 반사 계수' } },
          { symbol: 'R_{0s}', description: { en: 'Bare-interface reflectance', ko: 'Bare 계면 반사율' } },
          { symbol: 'n_0,n_s', description: { en: 'Incident and substrate refractive indices', ko: '입사 매질 및 기판 굴절률' } },
        ],
        note: { en: 'The optimizer is useful because polymer/oxide-to-silicon index contrast creates a large reflection penalty without a matching layer.', ko: 'Polymer/oxide-to-silicon 굴절률 차이가 크면 matching layer 없이 큰 반사 손실이 생기므로 optimizer가 유용합니다.' },
      },
      {
        label: { en: 'Quarter-wave phase target', ko: '1/4파장 위상 조건' },
        equation: '\\delta_1=\\frac{2\\pi n_1d_1}{\\lambda_0}\\approx\\frac{\\pi}{2}, \\quad d_1\\approx\\frac{\\lambda_0}{4n_1}',
        variables: [
          { symbol: '\\delta_1', description: { en: 'Phase thickness of a single anti-reflection layer', ko: '단일 반사 방지층의 위상 두께' } },
          { symbol: 'd_1', description: { en: 'Physical thickness of the layer', ko: '레이어의 물리적 두께' } },
          { symbol: 'n_1', description: { en: 'Refractive index of the anti-reflection layer', ko: '반사 방지층 굴절률' } },
          { symbol: '\\lambda_0', description: { en: 'Design wavelength', ko: '설계 중심 파장' } },
        ],
        note: { en: 'Quarter-wave thickness is a seed, not the final answer, because CIS stacks are absorbing, broadband, and angle dependent.', ko: '1/4파장 두께는 초기값일 뿐입니다. CIS 스택은 흡수성, 광대역, 각도 의존성을 갖기 때문입니다.' },
      },
      {
        label: { en: 'Single-layer cancellation condition', ko: '단일층 상쇄 조건' },
        equation: 'r_{01}+r_{1s}e^{2i\\delta_1}\\approx0, \\quad n_1\\approx\\sqrt{n_0n_s}\\;\\text{(lossless normal-incidence limit)}',
        variables: [
          { symbol: 'r_{01},r_{1s}', description: { en: 'Fresnel reflection amplitudes at the two interfaces of the layer', ko: '박막 양쪽 계면의 프레넬 반사 진폭' } },
          { symbol: 'e^{2i\\delta_1}', description: { en: 'Round-trip phase factor inside the layer', ko: '박막 내부 왕복 위상 인자' } },
          { symbol: '\\sqrt{n_0n_s}', description: { en: 'Ideal matching index for a lossless single layer', ko: '무손실 단일층에서의 이상적 matching index' } },
        ],
        note: { en: 'Real BARL material choices rarely hit the ideal index, so multiple layers and numerical search are used.', ko: '실제 BARL 재료는 이상적 굴절률과 잘 맞지 않는 경우가 많아 다층 구조와 수치 탐색을 사용합니다.' },
      },
      {
        label: { en: 'Oblique-incidence admittance', ko: '경사 입사 어드미턴스' },
        equation: '\\eta_j^{(s)}=\\tilde{n}_j\\cos\\theta_j, \\quad \\eta_j^{(p)}=\\frac{\\tilde{n}_j}{\\cos\\theta_j}',
        variables: [
          { symbol: '\\eta_j^{(s)},\\eta_j^{(p)}', description: { en: 'Layer optical admittance for s and p polarization', ko: 's/p 편광에 대한 레이어 광학 어드미턴스' } },
          { symbol: '\\tilde{n}_j', description: { en: 'Complex refractive index of layer $j$', ko: '레이어 $j$의 복소 굴절률' } },
          { symbol: '\\theta_j', description: { en: 'Internal propagation angle', ko: '레이어 내부 전파각' } },
        ],
        note: { en: 'A coating optimized at normal incidence may fail at high CRA because s and p admittances diverge.', ko: '수직 입사에서 최적화된 코팅도 큰 CRA에서는 s/p 어드미턴스가 갈라져 성능이 나빠질 수 있습니다.' },
      },
      {
        label: { en: 'Optimization objective', ko: '최적화 목적 함수' },
        equation: '\\mathcal{L}=\\sum_{c\\in\\{R,G,B\\}}w_c\\left\\langle R_c(\\lambda,\\theta)+\\gamma A_{\\text{parasitic},c}(\\lambda,\\theta)\\right\\rangle_{\\lambda\\in\\Omega_c}',
        variables: [
          { symbol: '\\mathcal{L}', description: { en: 'Weighted loss minimized by the optimizer', ko: 'Optimizer가 최소화하는 가중 손실 함수' } },
          { symbol: 'w_c', description: { en: 'Channel weight for color $c$', ko: '색 채널 $c$의 가중치' } },
          { symbol: '\\Omega_c', description: { en: 'Wavelength band of interest for channel $c$', ko: '색 채널 $c$의 관심 파장 대역' } },
          { symbol: '\\gamma', description: { en: 'Penalty weight for parasitic non-silicon absorption', ko: '비실리콘 기생 흡수 penalty 가중치' } },
        ],
        note: { en: 'In practice, the best stack minimizes reflection without moving too much power into lossy BARL or color-filter absorption.', ko: '실제로 좋은 스택은 반사를 줄이되 BARL 또는 color-filter 기생 흡수로 파워를 너무 많이 보내지 않아야 합니다.' },
      },
      {
        label: { en: 'QE improvement budget', ko: 'QE 개선 예산' },
        equation: '\\Delta A_{\\text{Si}}\\approx-\\Delta R-\\Delta T_{\\text{escape}}-\\Delta A_{\\text{parasitic}}',
        variables: [
          { symbol: '\\Delta A_{\\text{Si}}', description: { en: 'Change in useful silicon absorption', ko: '유효 실리콘 흡수율 변화' } },
          { symbol: '\\Delta R', description: { en: 'Change in reflected power', ko: '반사 파워 변화' } },
          { symbol: '\\Delta T_{\\text{escape}}', description: { en: 'Change in power transmitted past the active silicon region', ko: '활성 실리콘 영역을 지나 빠져나가는 파워 변화' } },
          { symbol: '\\Delta A_{\\text{parasitic}}', description: { en: 'Change in absorption outside the photodiode silicon', ko: '포토다이오드 실리콘 외부 흡수 변화' } },
        ],
        note: { en: 'A lower reflectance curve is only valuable if the saved photons are redirected into silicon absorption.', ko: '반사율 감소는 절약된 광자가 실리콘 흡수로 이동할 때만 QE 개선이 됩니다.' },
      },
    ],
    concepts: [
      { en: 'The optimum depends on incident medium, silicon optical constants, color-filter absorption, angle, polarization, and allowed process materials.', ko: '최적점은 입사 매질, silicon optical constant, color-filter absorption, angle, polarization, 허용 공정 재료에 따라 달라집니다.' },
      { en: 'A stack optimized for green peak QE can hurt blue, red, or off-axis response because phase cancellation is narrowband.', ko: 'Phase cancellation은 narrowband이므로 green peak QE에 맞춘 스택이 blue, red, off-axis response를 악화시킬 수 있습니다.' },
      { en: 'BARL optimization should be judged by silicon absorption and total color-channel balance, not reflectance alone.', ko: 'BARL 최적화는 reflectance만이 아니라 silicon absorption과 전체 색 채널 balance로 판단해야 합니다.' },
    ],
    sections: [
      {
        title: { en: 'Tuning Workflow', ko: '튜닝 절차' },
        items: [
          { en: 'Start from quarter-wave thickness near the target band, then sweep thickness around that seed because real stacks are absorbing and multilayered.', ko: '목표 대역 주변의 quarter-wave 두께에서 시작한 뒤, 실제 스택은 흡수성 다층계이므로 그 주변 두께를 sweep합니다.' },
          { en: 'Check $R$, $T$, parasitic absorption, and $A_{\\text{Si}}$ together; a reflectance minimum alone can be misleading.', ko: '$R$, $T$, 기생 흡수, $A_{\\text{Si}}$를 함께 확인해야 합니다. Reflectance minimum만 보면 오해할 수 있습니다.' },
          { en: 'Re-run the candidate at oblique incidence and both polarizations before treating it as a camera-edge solution.', ko: '후보 구조를 camera edge solution으로 보기 전에 경사 입사 및 양 편광에서 다시 확인해야 합니다.' },
        ],
      },
      {
        title: { en: 'Process Constraints', ko: '공정 제약' },
        items: [
          { en: 'Allowed materials, minimum thickness, etch selectivity, stress, thermal budget, and contamination rules usually restrict the mathematical optimum.', ko: '허용 재료, 최소 두께, etch selectivity, stress, thermal budget, contamination rule이 수학적 최적점을 제한합니다.' },
          { en: 'The same BARL stack may behave differently under different color-filter refractive-index and absorption spectra.', ko: '동일한 BARL 스택도 color-filter 굴절률과 흡수 spectrum에 따라 다르게 동작할 수 있습니다.' },
          { en: 'Thickness tolerance should be checked because narrow interference minima can be fragile to wafer non-uniformity.', ko: '좁은 간섭 minimum은 wafer non-uniformity에 취약할 수 있으므로 두께 tolerance를 확인해야 합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The optimizer is a planar-film model; it does not include microlens focusing, metal-grid diffraction, DTI, roughness scattering, or color-filter relief.', ko: 'Optimizer는 planar-film 모델이므로 microlens focusing, metal-grid diffraction, DTI, roughness scattering, color-filter relief를 포함하지 않습니다.' },
          { en: 'It treats optical constants as known inputs; process drift in $n,k$ can move the optimum.', ko: 'Optical constant를 알려진 입력으로 취급하지만, 공정 drift로 $n,k$가 바뀌면 최적점도 이동합니다.' },
          { en: 'For sub-wavelength lateral features, validate BARL candidates with RCWA/FDTD after TMM screening.', ko: '서브파장 lateral feature가 있으면 TMM screening 이후 RCWA/FDTD로 BARL 후보를 검증해야 합니다.' },
        ],
      },
    ],
    references: [
      {
        ...refs.macleod,
        note: { en: 'Thin-film coating design, characteristic matrices, and anti-reflection stack interpretation.', ko: '박막 코팅 설계, characteristic matrix, anti-reflection stack 해석 레퍼런스입니다.' },
      },
      {
        ...refs.bornWolf,
        note: { en: 'Background for Fresnel coefficients, polarization, and coherent interference.', ko: 'Fresnel coefficient, polarization, coherent interference의 배경 이론입니다.' },
      },
      {
        ...refs.green2008,
        note: { en: 'Silicon optical constants used when judging whether saved photons are absorbed usefully.', ko: '절약된 광자가 유효하게 흡수되는지 판단할 때 필요한 silicon optical constant 레퍼런스입니다.' },
      },
    ],
  },
  'energy-budget': {
    title: { en: 'Photon Energy Accounting', ko: '광자 에너지 예산' },
    summary: {
      en: 'The energy-budget view decomposes incident optical power into reflection, escape transmission, useful silicon absorption, and parasitic absorption in non-silicon layers. It is a diagnostic layer on top of the TMM result: it explains why QE changes, not just that QE changes.',
      ko: 'Energy-budget view는 입사 광파워를 반사, escape transmission, 유효 silicon absorption, 비실리콘 레이어의 기생 흡수로 분해합니다. 이는 TMM 결과 위의 진단 레이어로, QE가 변했다는 사실뿐 아니라 왜 변했는지를 설명합니다.',
    },
    intuition: {
      en: 'Think of every incoming photon as a coin. Some coins bounce off the stack, some pass through the active silicon without being used, some are spent heating filters or coatings, and only the coins absorbed in the photodiode silicon can become signal. The budget tells you which bucket is stealing photons.',
      ko: '입사 광자 하나하나를 동전으로 생각하면 됩니다. 일부 동전은 스택에서 반사되고, 일부는 활성 실리콘을 지나쳐 빠져나가며, 일부는 필터나 코팅을 데우는 데 쓰이고, 포토다이오드 실리콘에서 흡수된 동전만 신호가 될 수 있습니다. 에너지 예산은 어떤 bucket이 광자를 빼앗는지 보여줍니다.',
    },
    formulas: [
      {
        label: { en: 'Incident power normalization', ko: '입사 파워 정규화' },
        equation: 'P_{\\text{inc}}(\\lambda)=1',
        variables: [
          { symbol: 'P_{\\text{inc}}', description: { en: 'Incident optical power at a wavelength, normalized to unity', ko: '해당 파장의 입사 광파워; 1로 정규화' } },
          { symbol: '\\lambda', description: { en: 'Wavelength at which the stack is evaluated', ko: '스택을 평가하는 파장' } },
        ],
        note: { en: 'All bars in the budget are fractions of the same incident power, so channels and wavelengths can be compared directly.', ko: '예산 막대는 모두 같은 입사 파워에 대한 비율이므로 채널과 파장을 직접 비교할 수 있습니다.' },
      },
      {
        label: { en: 'Power conservation residual', ko: '파워 보존 잔차' },
        equation: '\\varepsilon = 1-R-T-\\sum_jA_j',
        variables: [
          { symbol: '\\varepsilon', description: { en: 'Numerical energy-balance residual', ko: '수치적 에너지 보존 잔차' } },
          { symbol: 'R,T', description: { en: 'Reflected and transmitted power fractions', ko: '반사 및 투과 파워 비율' } },
          { symbol: 'A_j', description: { en: 'Absorption fraction in layer $j$', ko: '레이어 $j$의 흡수 파워 비율' } },
        ],
        note: { en: 'A large residual means the optical model, interpolation, or layer accounting should be inspected before interpreting QE.', ko: '잔차가 크면 QE 해석 전에 광학 모델, 보간, 레이어 accounting을 점검해야 합니다.' },
      },
      {
        label: { en: 'Layer absorption from flux difference', ko: '플럭스 차이 기반 레이어 흡수' },
        equation: 'A_j=P_{\\text{top},j}-P_{\\text{bot},j}, \\quad P=\\frac{\\operatorname{Re}(EH^*)}{P_{\\text{inc}}}',
        variables: [
          { symbol: 'P_{\\text{top},j},P_{\\text{bot},j}', description: { en: 'Normalized Poynting flux at the top and bottom of layer $j$', ko: '레이어 $j$ 상단 및 하단의 정규화된 포인팅 플럭스' } },
          { symbol: 'E,H', description: { en: 'Electric and magnetic field amplitudes recovered from the transfer matrix', ko: '전달 행렬에서 복원한 전기장 및 자기장 진폭' } },
          { symbol: 'H^*', description: { en: 'Complex conjugate of magnetic field amplitude', ko: '자기장 진폭의 복소 켤레' } },
        ],
        note: { en: 'This assigns absorption to physical layers instead of only reporting total stack loss.', ko: '이 방식은 전체 스택 손실만 보는 대신 흡수를 물리 레이어별로 배정합니다.' },
      },
      {
        label: { en: 'Useful and parasitic buckets', ko: '유효/기생 흡수 bucket' },
        equation: 'A_{\\text{use}}=A_{\\text{Si}}, \\quad A_{\\text{parasitic}}=\\sum_{j\\notin\\text{Si}}A_j',
        variables: [
          { symbol: 'A_{\\text{use}}', description: { en: 'Absorption that can potentially contribute to signal', ko: '신호에 기여할 수 있는 잠재적 흡수' } },
          { symbol: 'A_{\\text{parasitic}}', description: { en: 'Absorption in filters, coatings, metal, or other non-photodiode layers', ko: '필터, 코팅, 금속, 기타 비포토다이오드 레이어의 흡수' } },
          { symbol: 'A_{\\text{Si}}', description: { en: 'Absorption in the silicon photodiode layer', ko: '실리콘 포토다이오드 레이어 흡수' } },
        ],
        note: { en: 'A design that reduces reflection but increases parasitic absorption may not improve sensor signal.', ko: '반사를 줄였더라도 기생 흡수가 늘면 센서 신호는 개선되지 않을 수 있습니다.' },
      },
      {
        label: { en: 'Optical QE upper bound', ko: '광학 QE 상한' },
        equation: 'QE_{\\text{ext}}(\\lambda)=\\eta_{\\text{cc}}(\\lambda)A_{\\text{Si}}(\\lambda), \\quad QE_{\\text{ext}}\\le A_{\\text{Si}}',
        variables: [
          { symbol: 'QE_{\\text{ext}}', description: { en: 'External quantum efficiency after carrier collection effects', ko: '전하 수집 효과까지 포함한 외부 양자 효율' } },
          { symbol: '\\eta_{\\text{cc}}', description: { en: 'Carrier-collection probability for generated electron-hole pairs', ko: '생성된 전자-정공쌍의 전하 수집 확률' } },
        ],
        note: { en: 'The budget reports optical absorption; electrical QE can only be lower unless carrier collection is ideal.', ko: '이 예산은 광학 흡수를 보고합니다. 전하 수집이 완전하지 않으면 전기적 QE는 이보다 낮습니다.' },
      },
      {
        label: { en: 'Design sensitivity', ko: '설계 민감도' },
        equation: 'S_{x_k}(\\lambda)=\\frac{\\partial A_{\\text{Si}}(\\lambda)}{\\partial x_k}',
        variables: [
          { symbol: 'S_{x_k}', description: { en: 'Sensitivity of silicon absorption to design variable $x_k$', ko: '설계 변수 $x_k$에 대한 silicon absorption 민감도' } },
          { symbol: 'x_k', description: { en: 'Layer thickness, material index, angle, or wavelength-dependent filter parameter', ko: '레이어 두께, 재료 굴절률, 입사각, 또는 파장 의존 필터 파라미터' } },
        ],
        note: { en: 'Sensitivity helps identify whether a QE loss is controlled by BARL thickness, silicon depth, filter absorption, or angle.', ko: '민감도는 QE 손실이 BARL 두께, silicon depth, filter absorption, angle 중 무엇에 지배되는지 식별하는 데 유용합니다.' },
      },
    ],
    concepts: [
      { en: 'Reflection-limited loss points to coating or BARL work; parasitic absorption points to material or thickness choices; escape transmission points to silicon thickness or light trapping.', ko: 'Reflection-limited loss는 coating/BARL 문제를, parasitic absorption은 재료/두께 선택 문제를, escape transmission은 silicon thickness 또는 light trapping 문제를 가리킵니다.' },
      { en: 'Energy accounting is the bridge between a spectral curve and an engineering action.', ko: 'Energy accounting은 spectral curve와 실제 engineering action을 연결하는 다리입니다.' },
      { en: 'The budget is most useful when compared across a before/after design change rather than read as a standalone number.', ko: '예산은 단독 숫자로 읽기보다 설계 변경 전후를 비교할 때 가장 유용합니다.' },
    ],
    sections: [
      {
        title: { en: 'Diagnosis Workflow', ko: '진단 절차' },
        items: [
          { en: 'First check $\\varepsilon$ to confirm power conservation, then inspect whether loss sits in $R$, $T$, $A_{\\text{parasitic}}$, or missing $A_{\\text{Si}}$.', ko: '먼저 $\\varepsilon$로 파워 보존을 확인한 뒤, 손실이 $R$, $T$, $A_{\\text{parasitic}}$, 부족한 $A_{\\text{Si}}$ 중 어디에 있는지 봅니다.' },
          { en: 'If $R$ dominates, tune BARL/coating; if $A_{\\text{parasitic}}$ dominates, revisit color-filter or metal absorption; if $T$ dominates, adjust silicon thickness or backside trapping.', ko: '$R$이 지배적이면 BARL/coating을, $A_{\\text{parasitic}}$가 지배적이면 color-filter/metal absorption을, $T$가 지배적이면 silicon thickness 또는 backside trapping을 조정합니다.' },
          { en: 'Compare the same budget for R/G/B because a fix for one channel can move loss into another channel.', ko: '한 채널 개선이 다른 채널 손실로 이동할 수 있으므로 R/G/B 예산을 함께 비교합니다.' },
        ],
      },
      {
        title: { en: 'Calibration Targets', ko: '보정 대상' },
        items: [
          { en: 'Use measured reflectance/transmittance spectra to anchor $R$ and $T$ before trusting layer absorption allocation.', ko: '레이어 흡수 배정을 신뢰하기 전에 실측 reflectance/transmittance spectrum으로 $R$ 및 $T$를 맞춥니다.' },
          { en: 'Replace generic $n,k$ with process-specific ellipsometry for color filters, BARL layers, silicon, and passivation.', ko: 'Color filter, BARL layer, silicon, passivation에는 generic $n,k$ 대신 공정별 ellipsometry 데이터를 사용합니다.' },
          { en: 'Validate $A_{\\text{Si}}$ against measured QE only after estimating carrier collection probability.', ko: '전하 수집 확률을 추정한 뒤에야 $A_{\\text{Si}}$를 실측 QE와 비교해야 합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The budget does not include lateral redistribution: photons absorbed in silicon outside the photodiode ROI may become crosstalk.', ko: '이 예산은 lateral redistribution을 포함하지 않습니다. 포토다이오드 ROI 밖 실리콘에서 흡수된 광자는 crosstalk가 될 수 있습니다.' },
          { en: 'It does not model electrical collection, diffusion, recombination, or depletion-region geometry.', ko: '전기적 collection, diffusion, recombination, depletion-region geometry는 모델링하지 않습니다.' },
          { en: 'Structured pixels need RCWA/FDTD field maps to decide where within silicon absorption occurs.', ko: '구조화된 픽셀에서는 실리콘 내부 흡수 위치를 판단하기 위해 RCWA/FDTD field map이 필요합니다.' },
        ],
      },
    ],
    references: [
      {
        ...refs.catrysse2002,
        note: { en: 'Image-sensor optical efficiency framing and connection between absorption and pixel-level efficiency.', ko: 'Image-sensor optical efficiency와 absorption/pixel-level efficiency 연결을 다룹니다.' },
      },
      {
        ...refs.green2008,
        note: { en: 'Silicon optical constants for wavelength-dependent absorption-depth and escape-loss analysis.', ko: '파장 의존 absorption-depth 및 escape-loss 분석에 필요한 silicon optical constant 레퍼런스입니다.' },
      },
      {
        ...refs.macleod,
        note: { en: 'Thin-film energy balance and multilayer optical power accounting background.', ko: '박막 에너지 보존 및 multilayer optical power accounting 배경 이론입니다.' },
      },
    ],
  },
  'angular-response': {
    title: { en: 'Chief-Ray Angle and Polarization Response', ko: '주광선각 및 편광 응답' },
    summary: {
      en: 'Angular response explains how QE changes when light enters a planar pixel stack away from normal incidence. It combines Snell refraction, s/p Fresnel behavior, TMM phase changes, longer absorption path length, and normalization against the on-axis response.',
      ko: 'Angular response는 빛이 수직 입사가 아니라 기울어진 각도로 평면 픽셀 스택에 들어올 때 QE가 어떻게 변하는지를 설명합니다. 스넬 굴절, s/p 프레넬 거동, TMM 위상 변화, 길어진 흡수 경로, on-axis 응답 대비 정규화를 함께 다룹니다.',
    },
    intuition: {
      en: 'A tilted ray does not simply travel through the same stack sideways. It refracts at each interface, sees different effective thicknesses, and splits into s and p polarization responses. At the edge of a camera field, that planar-film angular rolloff is only one part of the story; microlens shift and finite-cone averaging decide whether the focused spot still lands on the photodiode.',
      ko: '기울어진 광선은 단순히 같은 스택을 옆으로 지나가는 것이 아닙니다. 각 계면에서 굴절되고, 다른 유효 두께를 통과하며, s/p 편광 응답으로 갈라집니다. 카메라 주변부에서는 이 planar-film angular rolloff가 전체 이야기의 일부일 뿐이고, microlens shift와 finite-cone averaging이 초점 spot이 여전히 포토다이오드에 닿는지를 결정합니다.',
    },
    formulas: [
      {
        label: { en: 'Layer refraction', ko: '레이어 굴절' },
        equation: '\\tilde{n}_0\\sin\\theta_0=\\tilde{n}_j\\sin\\theta_j',
        variables: [
          { symbol: '\\tilde{n}_0,\\tilde{n}_j', description: { en: 'Complex refractive indices of incident medium and layer $j$', ko: '입사 매질 및 레이어 $j$의 복소 굴절률' } },
          { symbol: '\\theta_0', description: { en: 'External incidence angle or chief-ray angle in the planar-stack model', ko: '평면 스택 모델의 외부 입사각 또는 chief-ray angle' } },
          { symbol: '\\theta_j', description: { en: 'Internal angle inside layer $j$', ko: '레이어 $j$ 내부 각도' } },
        ],
        note: { en: 'High-index silicon strongly reduces the internal ray angle, but upper polymer/filter layers can still see meaningful obliquity.', ko: '고굴절률 silicon은 내부 광선 각도를 크게 줄이지만, 상부 polymer/filter layer에서는 여전히 경사 입사 효과가 중요할 수 있습니다.' },
      },
      {
        label: { en: 'Polarized Fresnel amplitudes', ko: '편광별 프레넬 진폭' },
        equation: 'r_s=\\frac{n_i\\cos\\theta_i-n_t\\cos\\theta_t}{n_i\\cos\\theta_i+n_t\\cos\\theta_t}, \\quad r_p=\\frac{n_t\\cos\\theta_i-n_i\\cos\\theta_t}{n_t\\cos\\theta_i+n_i\\cos\\theta_t}',
        variables: [
          { symbol: 'r_s,r_p', description: { en: 'Reflection amplitudes for s and p polarization', ko: 's/p 편광 반사 진폭' } },
          { symbol: 'n_i,n_t', description: { en: 'Incident and transmitted side refractive indices', ko: '입사측 및 투과측 굴절률' } },
          { symbol: '\\theta_i,\\theta_t', description: { en: 'Incident and transmitted angles at an interface', ko: '계면의 입사각 및 투과각' } },
        ],
        note: { en: 's and p responses can diverge strongly at high angles, especially near Brewster-like conditions.', ko: '큰 입사각에서는 특히 Brewster-like 조건 근처에서 s/p 응답이 크게 갈라질 수 있습니다.' },
      },
      {
        label: { en: 'TMM admittance at angle', ko: '각도 의존 TMM 어드미턴스' },
        equation: '\\eta_j^{(s)}=\\tilde{n}_j\\cos\\theta_j, \\quad \\eta_j^{(p)}=\\frac{\\tilde{n}_j}{\\cos\\theta_j}',
        variables: [
          { symbol: '\\eta_j^{(s)},\\eta_j^{(p)}', description: { en: 'Optical admittance of layer $j$ for s and p polarization', ko: 's/p 편광에서 레이어 $j$의 광학 어드미턴스' } },
          { symbol: '\\tilde{n}_j', description: { en: 'Complex refractive index', ko: '복소 굴절률' } },
          { symbol: '\\cos\\theta_j', description: { en: 'Obliquity factor inside the layer', ko: '레이어 내부 경사 인자' } },
        ],
        note: { en: 'The same physical stack has different effective boundary conditions for the two polarization states.', ko: '같은 물리 스택도 두 편광 상태에서는 서로 다른 유효 경계조건을 갖습니다.' },
      },
      {
        label: { en: 'Angular phase thickness', ko: '각도 의존 위상 두께' },
        equation: '\\delta_j(\\theta_0)=\\frac{2\\pi}{\\lambda}\\tilde{n}_jd_j\\cos\\theta_j',
        variables: [
          { symbol: '\\delta_j(\\theta_0)', description: { en: 'Phase thickness induced by external incidence angle $\\theta_0$', ko: '외부 입사각 $\\theta_0$가 유도하는 위상 두께' } },
          { symbol: 'd_j', description: { en: 'Layer physical thickness', ko: '레이어 물리 두께' } },
          { symbol: '\\lambda', description: { en: 'Vacuum wavelength', ko: '진공 파장' } },
        ],
        note: { en: 'A phase shift with angle can move BARL minima and color-filter interference features across wavelength.', ko: '각도에 따른 위상 변화는 BARL minimum 및 color-filter 간섭 feature를 파장축에서 이동시킬 수 있습니다.' },
      },
      {
        label: { en: 'Relative angular response', ko: '상대 각도 응답' },
        equation: 'AR_{p/s}(\\theta,\\lambda)=\\frac{QE_{p/s}(\\theta,\\lambda)}{QE_{p/s}(0,\\lambda)}, \\quad AR_{\\text{unpol}}=\\frac{AR_s+AR_p}{2}',
        variables: [
          { symbol: 'AR_{p/s}', description: { en: 'Normalized angular response for each polarization', ko: '각 편광의 정규화된 각도 응답' } },
          { symbol: 'QE_{p/s}', description: { en: 'Optical QE proxy for s or p polarization', ko: 's 또는 p 편광의 광학 QE proxy' } },
          { symbol: 'AR_{\\text{unpol}}', description: { en: 'Unpolarized angular-response estimate', ko: '비편광 각도 응답 추정치' } },
        ],
        note: { en: 'Normalization removes absolute QE scale and highlights angular rolloff or angular gain.', ko: '정규화는 절대 QE scale을 제거하고 각도별 rolloff 또는 gain을 드러냅니다.' },
      },
      {
        label: { en: 'Finite-cone average', ko: '유한 cone 평균' },
        equation: '\\overline{AR}(\\lambda)=\\frac{\\int_{\\Omega}AR(\\theta,\\phi,\\lambda)W(\\theta,\\phi)d\\Omega}{\\int_{\\Omega}W(\\theta,\\phi)d\\Omega}',
        variables: [
          { symbol: '\\Omega', description: { en: 'Angular cone set by lens f-number and chief-ray direction', ko: '렌즈 f-number 및 chief-ray direction이 정하는 각도 cone' } },
          { symbol: 'W(\\theta,\\phi)', description: { en: 'Angular weighting function of the optical system', ko: '광학계의 각도 가중 함수' } },
          { symbol: '\\phi', description: { en: 'Azimuthal angle inside the cone', ko: 'Cone 내부 방위각' } },
        ],
        note: { en: 'A real camera pixel receives a cone of rays, not a single plane wave, so edge-pixel behavior needs cone averaging.', ko: '실제 카메라 픽셀은 단일 평면파가 아니라 ray cone을 받으므로 주변부 픽셀 거동에는 cone 평균이 필요합니다.' },
      },
    ],
    concepts: [
      { en: 'Planar angular response is a stack property; full CRA response is a pixel property that also includes microlens focusing and lateral collection.', ko: 'Planar angular response는 스택 특성이고, 전체 CRA response는 microlens focusing 및 lateral collection까지 포함하는 픽셀 특성입니다.' },
      { en: 's/p separation is not optional at high angle because coating minima and Fresnel loss can split by polarization.', ko: '큰 입사각에서는 coating minimum과 Fresnel loss가 편광별로 갈라질 수 있으므로 s/p 분리는 필수입니다.' },
      { en: 'A good edge-pixel design usually needs stack tuning, microlens shift, and finite-cone validation together.', ko: '좋은 edge-pixel 설계에는 보통 stack tuning, microlens shift, finite-cone validation이 함께 필요합니다.' },
    ],
    sections: [
      {
        title: { en: 'How To Read The Plot', ko: '그래프 해석 방법' },
        items: [
          { en: 'If both s and p fall together, the loss is likely path-length or absorption driven; if they split, Fresnel/admittance effects are important.', ko: 's와 p가 함께 떨어지면 path-length 또는 absorption 영향일 가능성이 크고, 서로 갈라지면 Fresnel/admittance 효과가 중요합니다.' },
          { en: 'A response above 1 at some angle can occur when interference minima shift in a favorable direction; it is not automatically an error.', ko: '특정 각도에서 응답이 1보다 커질 수 있는데, 간섭 minimum이 유리한 방향으로 이동한 결과일 수 있어 자동으로 오류는 아닙니다.' },
          { en: 'Compare wavelength slices, not only broadband averages, because angular color shading is channel dependent.', ko: 'Angular color shading은 채널 의존적이므로 broadband average만 보지 말고 파장별 slice를 비교해야 합니다.' },
        ],
      },
      {
        title: { en: 'CRA Design Implications', ko: 'CRA 설계 의미' },
        items: [
          { en: 'At the sensor edge, the chief ray may enter at high angle even when the lens cone has many nearby angles.', ko: '센서 edge에서는 lens cone 안의 여러 각도와 함께 chief ray 자체가 큰 각도로 들어올 수 있습니다.' },
          { en: 'Microlens shift compensates the lateral focus displacement, while stack tuning controls planar-film throughput at that angle.', ko: 'Microlens shift는 lateral focus displacement를 보정하고, stack tuning은 그 각도에서의 planar-film throughput을 제어합니다.' },
          { en: 'Use this angular page before expensive ray/FDTD studies to identify angle bands and wavelengths at risk.', ko: '비싼 ray/FDTD 연구 전에 이 angular 페이지로 위험한 angle band와 wavelength를 식별합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The model does not trace the focused spot to a finite photodiode or model microlens shift.', ko: '이 모델은 초점 spot이 유한한 포토다이오드에 닿는지 추적하거나 microlens shift를 모델링하지 않습니다.' },
          { en: 'It ignores lateral diffraction, metal-grid shadowing, color-filter relief, and DTI-induced waveguide effects.', ko: 'Lateral diffraction, metal-grid shadowing, color-filter relief, DTI-induced waveguide effect를 무시합니다.' },
          { en: 'For finite aperture filters or Fabry-Perot stacks, integrate the full angular cone rather than relying on a single angle.', ko: 'Finite aperture filter 또는 Fabry-Perot stack에서는 단일 각도 대신 전체 angular cone을 적분해야 합니다.' },
        ],
      },
    ],
    references: [
      {
        ...refs.hwang2023,
        note: { en: 'Practical CIS context for aligning optical stacks under pixel-dependent chief-ray angle.', ko: 'Pixel-dependent chief-ray angle에서 optical stack alignment를 다루는 CIS 실무 맥락입니다.' },
      },
      {
        ...refs.goossens2018,
        note: { en: 'Finite-aperture angular averaging reference for spectral cameras with thin-film filters.', ko: 'Thin-film filter를 갖는 spectral camera의 finite-aperture angular averaging 레퍼런스입니다.' },
      },
      {
        ...refs.macleod,
        note: { en: 'Thin-film angular and polarization response theory.', ko: '박막의 각도 및 편광 응답 이론 레퍼런스입니다.' },
      },
    ],
  },
  'snr-calculator': {
    title: { en: 'Pixel Signal-to-Noise Model', ko: '픽셀 신호 대 잡음 모델' },
    summary: {
      en: 'SNR follows from the signal charge divided by the root-sum-square of shot, dark, read, and fixed-pattern noise terms.',
      ko: 'SNR은 신호 전하를 샷 노이즈, 암전류 노이즈, 읽기 노이즈, 고정 패턴 노이즈의 제곱합 제곱근으로 나눈 값입니다.',
    },
    intuition: {
      en: 'Imagine trying to hear someone whisper in a noisy room — what matters is the whisper compared to the background noise, not the whisper alone. In a pixel, the signal is the electrons created by light, while the background noise comes from random photon arrivals, thermal dark current, and the readout circuit. SNR tells you how clearly the image rises above the noise floor.',
      ko: '시끄러운 방에서 누군가의 속삭임을 들으려 한다고 상상해 보세요. 중요한 것은 속삭임 자체가 아니라 배경 소음과 비교한 크기입니다. 픽셀에서는 빛으로 생성된 전자가 신호이고, 광자 도착의 무작위성, 열에 의한 암전류, 회로의 읽기 노이즈가 배경 소음에 해당합니다. SNR은 이미지가 노이즈 바닥 위로 얼마나 또렷이 떠오르는지를 알려줍니다.',
    },
    formulas: [
      {
        label: { en: 'Photoelectrons', ko: '광전자 수' },
        equation: 'S = \\Phi_{\\text{ph}} \\cdot A_{\\text{px}} \\cdot t_{\\text{exp}} \\cdot QE',
        variables: [
          { symbol: 'S', description: { en: 'Collected signal (electrons)', ko: '수집된 신호 전하량 (전자 수)' } },
          { symbol: '\\Phi_{\\text{ph}}', description: { en: 'Incident photon flux (photons/area/time)', ko: '입사 광자 플럭스' } },
          { symbol: 'A_{\\text{px}}', description: { en: 'Pixel area', ko: '픽셀 면적' } },
          { symbol: 't_{\\text{exp}}', description: { en: 'Integration time', ko: '노출 시간' } },
        ],
        note: { en: 'Signal scales with photon flux, pixel area, exposure time, and quantum efficiency.', ko: '신호는 광자 플럭스, 픽셀 면적, 노출 시간, 양자 효율에 비례합니다.' },
      },
      {
        label: { en: 'Noise variance', ko: '노이즈 분산' },
        equation: '\\sigma^2 = S + D \\cdot t_{\\text{exp}} + \\sigma_{\\text{read}}^2 + (PRNU \\cdot S)^2',
        variables: [
          { symbol: '\\sigma^2', description: { en: 'Total noise variance (electrons squared)', ko: '전체 노이즈 분산' } },
          { symbol: 'D', description: { en: 'Dark current (electrons/s)', ko: '암전류' } },
          { symbol: '\\sigma_{\\text{read}}', description: { en: 'RMS read noise (electrons)', ko: '읽기 노이즈' } },
          { symbol: 'PRNU', description: { en: 'Photo-response non-uniformity factor', ko: 'PRNU 계수' } },
        ],
        note: { en: 'Shot and dark noise are Poisson terms; read and PRNU terms are added in variance.', ko: '샷/암전류 노이즈는 포아송 항이고, 읽기 노이즈와 PRNU는 분산으로 더합니다.' },
      },
      {
        label: { en: 'SNR', ko: 'SNR' },
        equation: 'SNR_{\\text{dB}} = 20 \\log_{10} \\left( \\frac{S}{\\sigma} \\right)',
        variables: [
          { symbol: '\\sigma', description: { en: 'Total noise (RMS electrons)', ko: '전체 노이즈 표준편차' } },
        ],
        note: { en: 'The dB form is useful for comparing operating points across illumination levels.', ko: 'dB 표현은 조도별 동작점을 비교할 때 유용합니다.' },
      },
    ],
    concepts: [
      { en: 'At low signal, read noise dominates; at high signal, photon shot noise dominates.', ko: '저신호에서는 읽기 노이즈가, 고신호에서는 광자 샷 노이즈가 지배적입니다.' },
      { en: 'Full well capacity clips signal before the formula can continue to improve.', ko: '풀웰 용량에 도달하면 신호가 포화되어 SNR 개선이 멈춥니다.' },
      { en: 'PRNU is signal-proportional, so it matters most in bright regions.', ko: 'PRNU는 신호에 비례하므로 밝은 영역에서 중요해집니다.' },
    ],
    sections: [
      {
        title: { en: 'Regime Map', ko: '동작 영역 지도' },
        items: [
          { en: 'Read-noise limited: when $S \\ll \\sigma_{\\text{read}}^2$, increasing exposure or QE gives nearly linear SNR improvement.', ko: 'Read-noise 제한: $S \\ll \\sigma_{\\text{read}}^2$이면 노출 시간이나 QE 증가가 SNR을 거의 선형적으로 개선합니다.' },
          { en: 'Shot-noise limited: when $S$ dominates the variance, $SNR\\approx\\sqrt{S}$, so doubling signal improves SNR only by $\\sqrt{2}$.', ko: 'Shot-noise 제한: $S$가 분산을 지배하면 $SNR\\approx\\sqrt{S}$이므로 신호를 두 배로 늘려도 SNR은 $\\sqrt{2}$배만 개선됩니다.' },
          { en: 'PRNU limited: at bright levels, $(PRNU\\cdot S)^2$ can cap SNR even when read noise is excellent.', ko: 'PRNU 제한: 밝은 영역에서는 읽기 노이즈가 좋아도 $(PRNU\\cdot S)^2$ 항이 SNR 상한을 만들 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Calibration Inputs', ko: '보정 입력' },
        items: [
          { en: 'Estimate conversion gain and read noise from photon-transfer data before comparing electron-domain SNR.', ko: '전자 단위 SNR을 비교하기 전에 photon-transfer data에서 conversion gain과 read noise를 추정해야 합니다.' },
          { en: 'Use dark frames at temperature to calibrate $D$; dark current can change exponentially with temperature.', ko: '온도별 dark frame으로 $D$를 보정해야 합니다. Dark current는 온도에 따라 지수적으로 변할 수 있습니다.' },
          { en: 'Measure PRNU from flat fields after removing DSNU and illumination shading.', ko: 'DSNU와 illumination shading을 제거한 flat field에서 PRNU를 측정합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The formula assumes independent noise terms; row noise, column noise, flicker noise, and ADC nonlinearity may violate this.', ko: '이 식은 노이즈 항이 독립이라고 가정합니다. Row noise, column noise, flicker noise, ADC 비선형성은 이를 깨뜨릴 수 있습니다.' },
          { en: 'Spatial processing, demosaic, denoise, and tone mapping can change perceived image SNR after sensor readout.', ko: 'Demosaic, denoise, tone mapping 같은 후처리는 센서 readout 이후 체감 이미지 SNR을 바꿀 수 있습니다.' },
          { en: 'Full-well, saturation, blooming, and dual-conversion-gain switching need a piecewise model beyond this compact equation.', ko: 'Full-well, saturation, blooming, dual-conversion-gain switching은 이 compact equation을 넘어선 piecewise 모델이 필요합니다.' },
        ],
      },
    ],
    references: [refs.emva, refs.catrysse2002],
  },
  'color-filter': {
    title: { en: 'Spectral Color Separation', ko: '분광 색 분리 모델' },
    summary: {
      en: 'Color-filter design maps spectral transmittance into camera RGB sensitivities, CIE tristimulus values, chromaticity, crosstalk, and color-correction conditioning.',
      ko: '컬러 필터 설계는 분광 투과율을 카메라 RGB 감도, CIE 삼자극값, 색도, 크로스토크, 색 보정 조건수로 변환합니다.',
    },
    intuition: {
      en: 'A colour filter is like coloured sunglasses for one pixel — a red filter mostly lets red light through and blocks the rest. By giving each pixel a different filter (R, G, or B), the sensor learns colour. The trade-off is balance: narrow filters give purer colour but waste photons, while wide filters keep brightness but mix colour channels together.',
      ko: '컬러 필터는 픽셀 하나에 씌운 색깔 선글라스와 비슷합니다. 빨간 필터는 빨간빛만 잘 통과시키고 나머지는 차단하죠. 픽셀마다 다른 필터(R/G/B)를 씌우면 센서가 색을 인식합니다. 핵심은 균형입니다. 좁은 필터는 색이 깨끗하지만 광량을 잃고, 넓은 필터는 밝지만 채널 간 색이 섞입니다.',
    },
    formulas: [
      {
        label: { en: 'Channel response', ko: '채널 응답' },
        equation: 'S_c(\\lambda) = T_c(\\lambda) \\cdot QE_{\\text{stack}}(\\lambda)',
        variables: [
          { symbol: 'S_c', description: { en: 'Spectral response of channel $c$', ko: '채널 $c$의 분광 응답' } },
          { symbol: 'T_c', description: { en: 'Filter transmittance', ko: '필터 투과율' } },
        ],
        note: { en: 'The sensor channel is the filter transmission multiplied by the optical stack QE.', ko: '센서 채널 응답은 필터 투과율과 광학 스택 QE의 곱입니다.' },
      },
      {
        label: { en: 'CIE tristimulus integration', ko: 'CIE 삼자극 적분' },
        equation: 'X = \\int S(\\lambda) \\bar{x}(\\lambda) \\,d\\lambda',
        variables: [
          { symbol: 'X, Y, Z', description: { en: 'Tristimulus values', ko: '삼자극값' } },
          { symbol: '\\bar{x}, \\bar{y}, \\bar{z}', description: { en: 'CIE color matching functions', ko: 'CIE 색 매칭 함수' } },
        ],
        note: { en: 'The same integration is used for Y and Z with ybar and zbar.', ko: 'Y와 Z도 각각 ybar, zbar로 같은 방식으로 적분합니다.' },
      },
      {
        label: { en: 'Chromaticity', ko: '색도 좌표' },
        equation: 'x = \\frac{X}{X+Y+Z}, \\quad y = \\frac{Y}{X+Y+Z}',
        variables: [
          { symbol: 'x, y', description: { en: 'Normalized chromaticity coordinates', ko: '색도 좌표' } },
        ],
        note: { en: 'The gamut triangle comes from the R, G, and B chromaticity points.', ko: '색역 삼각형은 R, G, B 색도점으로부터 만들어집니다.' },
      },
    ],
    concepts: [
      { en: 'Narrow filters improve color separation but reduce photon throughput.', ko: '좁은 필터는 색 분리를 개선하지만 광자 처리량을 낮춥니다.' },
      { en: 'Large spectral overlap increases crosstalk and makes the CCM less well conditioned.', ko: '분광 겹침이 크면 크로스토크가 증가하고 CCM 조건이 나빠집니다.' },
      { en: 'The IR-cut setting is critical because silicon remains sensitive beyond the visible band.', ko: '실리콘은 가시광 이후에도 민감하므로 IR 차단 설정이 중요합니다.' },
    ],
    sections: [
      {
        title: { en: 'Design Tradeoff', ko: '설계 tradeoff' },
        items: [
          { en: 'Photon efficiency favors broad $T_c(\\lambda)$; color separation favors narrow and well-spaced filters.', ko: '광자 효율은 넓은 $T_c(\\lambda)$를 선호하고, 색 분리는 좁고 잘 분리된 필터를 선호합니다.' },
          { en: 'Channel overlap raises off-diagonal camera response terms, increasing the burden on the color-correction matrix.', ko: '채널 overlap은 camera response의 off-diagonal 항을 키워 color-correction matrix 부담을 늘립니다.' },
          { en: 'The best filter set is illuminant dependent; daylight, tungsten, LED, and NIR leakage can rank designs differently.', ko: '최적 필터 세트는 조명 의존적입니다. Daylight, tungsten, LED, NIR leakage에서 설계 순위가 달라질 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Validation Targets', ko: '검증 대상' },
        items: [
          { en: 'Measure filter transmittance and full stack QE separately; a color error can come from pigment absorption or from the optical stack.', ko: 'Filter transmittance와 full stack QE를 분리 측정해야 합니다. 색 오차는 pigment absorption 또는 optical stack 양쪽에서 올 수 있습니다.' },
          { en: 'Validate chromaticities with a known illuminant and CIE observer before fitting a CCM.', ko: 'CCM fitting 전에 알려진 illuminant와 CIE observer 기준으로 chromaticity를 검증합니다.' },
          { en: 'Check IR-cut assumptions because a small NIR leak can dominate red-channel response for silicon sensors.', ko: 'Silicon sensor에서는 작은 NIR leak도 red-channel response를 지배할 수 있으므로 IR-cut 가정을 확인합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'This spectral model does not include color-filter topography, scattering, pigment granularity, or angle-dependent spectral shift.', ko: '이 분광 모델은 color-filter topography, scattering, pigment granularity, angle-dependent spectral shift를 포함하지 않습니다.' },
          { en: 'Human color metrics do not directly predict demosaiced image artifacts such as zippering or false color.', ko: 'Human color metric은 demosaic 이후 zippering이나 false color 같은 이미지 artifact를 직접 예측하지 않습니다.' },
          { en: 'For submicron pixels, filter relief and metal-grid geometry can require RCWA/FDTD rather than a scalar spectrum.', ko: 'Submicron pixel에서는 filter relief와 metal-grid geometry 때문에 scalar spectrum이 아니라 RCWA/FDTD가 필요할 수 있습니다.' },
        ],
      },
    ],
    references: [refs.nistCie, refs.mccamy1992, refs.catrysse2002],
  },
  'pixel-playground': {
    title: { en: 'Coupled Pixel Stack Model', ko: '결합 픽셀 스택 모델' },
    summary: {
      en: 'The playground combines pixel pitch, color-filter thickness, BARL layers, silicon thickness, and incidence angle into one simplified stack-level optical response.',
      ko: '플레이그라운드는 픽셀 피치, 컬러 필터 두께, BARL 레이어, 실리콘 두께, 입사각을 하나의 단순화된 스택 광학 응답으로 결합합니다.',
    },
    intuition: {
      en: 'This is a what-if sandbox for pixel design. Drag a slider — pixel pitch, filter thickness, silicon depth — and watch several metrics move at once because everything in a pixel is coupled. It is a quick first-pass tool to build intuition before you spend hours on a full RCWA or FDTD simulation.',
      ko: '픽셀 설계를 위한 “만약에” 실험실입니다. 픽셀 피치, 필터 두께, 실리콘 두께 같은 슬라이더를 움직이면 픽셀의 모든 요소가 서로 얽혀 있기 때문에 여러 지표가 동시에 변하는 모습을 볼 수 있습니다. 본격적인 RCWA/FDTD 시뮬레이션에 시간을 쓰기 전, 빠르게 감을 잡기 위한 1차 도구입니다.',
    },
    formulas: [
      {
        label: { en: 'Stack response', ko: '스택 응답' },
        equation: 'QE_c(\\lambda) \\approx T_{\\text{CF},c}(\\lambda) \\cdot A_{\\text{Si}}(\\lambda; \\text{stack})',
        variables: [
          { symbol: 'QE_c', description: { en: 'Quantum efficiency for color $c$', ko: '색상 $c$의 양자 효율' } },
          { symbol: 'T_{\\text{CF},c}', description: { en: 'Color filter transmittance', ko: '컬러 필터 투과율' } },
          { symbol: 'A_{\\text{Si}}', description: { en: 'Absorbance in the silicon photodiode', ko: '실리콘 포토다이오드의 흡수율' } },
        ],
        note: { en: 'The color channel response is approximated by filter transmission times silicon absorption.', ko: '색 채널 응답은 필터 투과율과 실리콘 흡수율의 곱으로 근사합니다.' },
      },
      {
        label: { en: 'Pixel area scaling', ko: '픽셀 면적 스케일링' },
        equation: 'A_{\\text{px}} = p^2',
        variables: [
          { symbol: 'A_{\\text{px}}', description: { en: 'Geometric pixel area', ko: '픽셀의 기하학적 면적' } },
          { symbol: 'p', description: { en: 'Pixel pitch', ko: '픽셀 피치' } },
        ],
        note: { en: 'Area changes photon count and full-well trends even if spectral QE is unchanged.', ko: '분광 QE가 같아도 면적은 광자 수와 풀웰 경향을 바꿉니다.' },
      },
      {
        label: { en: 'Energy budget', ko: '에너지 예산' },
        equation: 'R + T + \\sum A_{\\text{layer}} = 1',
        variables: [
          { symbol: 'R, T', description: { en: 'Reflectance and transmittance', ko: '반사율 및 투과율' } },
          { symbol: 'A_{\\text{layer}}', description: { en: 'Per-layer absorbance', ko: '레이어별 흡수율' } },
        ],
        note: { en: 'The same conservation check is used to interpret stack losses.', ko: '스택 손실 해석에는 같은 에너지 보존 검증을 사용합니다.' },
      },
    ],
    concepts: [
      { en: 'Changing one slider can shift multiple metrics because optical stacks are coupled.', ko: '광학 스택은 결합되어 있으므로 하나의 슬라이더가 여러 지표를 동시에 바꿀 수 있습니다.' },
      { en: 'Use this for design-space triage before running RCWA or FDTD on a detailed geometry.', ko: '상세 형상 RCWA/FDTD 전에 설계 공간을 좁히는 용도로 쓰세요.' },
      { en: 'The model omits lateral field maps, carrier transport, and process variation.', ko: '이 모델은 횡방향 필드맵, 전하 수송, 공정 변동을 포함하지 않습니다.' },
    ],
    sections: [
      {
        title: { en: 'Coupled Knobs', ko: '결합된 설계 변수' },
        items: [
          { en: 'Reducing pitch lowers photon count through $A_{\\text{px}}=p^2$ even when optical QE stays constant.', ko: 'Pitch를 줄이면 optical QE가 같아도 $A_{\\text{px}}=p^2$ 때문에 광자 수가 감소합니다.' },
          { en: 'Changing color-filter thickness shifts both spectral separation and parasitic absorption.', ko: 'Color-filter 두께 변화는 spectral separation과 parasitic absorption을 동시에 움직입니다.' },
          { en: 'Silicon thickness improves long-wavelength absorption but can increase crosstalk or carrier-collection burden in a real pixel.', ko: 'Silicon 두께 증가는 장파장 흡수를 개선하지만 실제 픽셀에서는 crosstalk 또는 carrier collection 부담을 키울 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Use As A Triage Tool', ko: 'Triage 도구로 쓰는 방법' },
        items: [
          { en: 'Use large metric changes to identify promising regions, then move those candidates into TMM, ray tracing, RCWA, or FDTD.', ko: '큰 지표 변화를 이용해 유망 영역을 찾은 뒤, 후보를 TMM, ray tracing, RCWA, FDTD로 넘깁니다.' },
          { en: 'Compare relative trends rather than absolute numbers because the simplified browser model uses compact material spectra.', ko: '간이 브라우저 모델은 compact material spectrum을 쓰므로 절대값보다 상대 경향을 비교합니다.' },
          { en: 'Treat any optimum at a slider boundary as a sign that the explored design range is too narrow.', ko: '최적점이 slider 경계에 있으면 탐색 설계 범위가 너무 좁다는 신호로 봅니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The playground does not solve lateral diffraction, microlens focusing, DTI confinement, or charge diffusion.', ko: 'Playground는 lateral diffraction, microlens focusing, DTI confinement, charge diffusion을 풀지 않습니다.' },
          { en: 'Process variation, thickness tolerance, and material dispersion uncertainty are not sampled statistically.', ko: '공정 변동, 두께 tolerance, material dispersion uncertainty를 통계적으로 샘플링하지 않습니다.' },
          { en: 'Use it to choose simulations, not to sign off a pixel stack.', ko: 'Pixel stack sign-off가 아니라 어떤 시뮬레이션을 돌릴지 고르는 용도로 사용합니다.' },
        ],
      },
    ],
    references: [refs.catrysse2002, refs.green2008, refs.macleod],
  },
  'si-absorption': {
    title: { en: 'Beer-Lambert Absorption in Silicon', ko: '실리콘 내 Beer-Lambert 흡수' },
    summary: {
      en: 'Silicon absorption depends strongly on wavelength. Blue light is absorbed near the surface, while red and NIR photons penetrate much deeper.',
      ko: '실리콘 흡수는 파장에 매우 강하게 의존합니다. 청색광은 표면 근처에서 흡수되고, 적색 및 NIR 광자는 훨씬 깊게 침투합니다.',
    },
    intuition: {
      en: 'Silicon swallows different colours at different depths. Blue light is absorbed within the first few hundred nanometres near the surface, while red and near-infrared photons sneak much deeper before being absorbed — sometimes deeper than the photodiode reaches. That is why thin BSI pixels often lose near-IR sensitivity.',
      ko: '실리콘은 색에 따라 흡수 깊이가 크게 다릅니다. 청색광은 표면에서 수백 nm 안에 거의 다 흡수되지만, 적색과 NIR은 훨씬 깊은 곳까지 침투해서야 흡수됩니다. 때로는 포토다이오드가 닿지 않는 깊이까지 가버려서, 얇은 BSI 픽셀에서 NIR 감도가 떨어지는 이유가 됩니다.',
    },
    formulas: [
      {
        label: { en: 'Absorption coefficient', ko: '흡수 계수' },
        equation: '\\alpha(\\lambda) = \\frac{4\\pi k(\\lambda)}{\\lambda}',
        variables: [
          { symbol: '\\alpha', description: { en: 'Absorption coefficient ($1/\\mu\\text{m}$)', ko: '흡수 계수' } },
          { symbol: 'k', description: { en: 'Extinction coefficient', ko: '소광 계수' } },
          { symbol: '\\lambda', description: { en: 'Incident wavelength', ko: '입사 파장' } },
        ],
        note: { en: 'The extinction coefficient k is converted to an absorption coefficient.', ko: '소광 계수 k를 흡수 계수로 변환합니다.' },
      },
      {
        label: { en: 'Intensity decay', ko: '강도 감쇠' },
        equation: 'I(z, \\lambda) = I_0 e^{-\\alpha(\\lambda)z}',
        variables: [
          { symbol: 'I(z)', description: { en: 'Intensity at depth $z$', ko: '깊이 $z$에서의 빛의 강도' } },
          { symbol: 'I_0', description: { en: 'Intensity at the surface ($z=0$)', ko: '표면에서의 강도' } },
        ],
        note: { en: 'This is the Beer-Lambert law for a uniform absorbing medium.', ko: '균일 흡수 매질에 대한 Beer-Lambert 법칙입니다.' },
      },
      {
        label: { en: 'Absorbed fraction', ko: '흡수율' },
        equation: 'A(d, \\lambda) = 1 - e^{-\\alpha(\\lambda)d}',
        variables: [
          { symbol: 'A(d)', description: { en: 'Absorbed fraction within thickness $d$', ko: '두께 $d$에서의 흡수율' } },
          { symbol: 'd', description: { en: 'Silicon physical thickness', ko: '실리콘 물리적 두께' } },
        ],
        note: { en: 'This ignores front-surface reflection and interference.', ko: '이 식은 표면 반사와 간섭을 무시한 근사입니다.' },
      },
    ],
    concepts: [
      { en: 'Thin BSI pixels may collect visible light well but lose NIR photons.', ko: '얇은 BSI 픽셀은 가시광은 잘 수집해도 NIR 광자를 놓칠 수 있습니다.' },
      { en: 'DTI, backside texture, and reflectors can increase the effective path length.', ko: 'DTI, 후면 텍스처, 반사 구조는 유효 광경로를 늘릴 수 있습니다.' },
      { en: 'Carrier collection depth is a separate electrical problem not included here.', ko: '전하 수집 깊이는 별도의 전기적 문제이며 여기에는 포함하지 않습니다.' },
    ],
    sections: [
      {
        title: { en: 'Wavelength Regimes', ko: '파장별 영역' },
        items: [
          { en: 'Blue light has large $\\alpha$, so it is absorbed near the surface and is sensitive to surface passivation and shallow collection.', ko: 'Blue light는 $\\alpha$가 커서 표면 근처에서 흡수되며 surface passivation 및 shallow collection에 민감합니다.' },
          { en: 'Red and NIR light have smaller $\\alpha$, so a thin silicon layer can transmit photons without useful absorption.', ko: 'Red 및 NIR은 $\\alpha$가 작아 얇은 silicon layer에서는 유효 흡수 없이 투과될 수 있습니다.' },
          { en: 'The absorption length $1/\\alpha$ is a useful scale, but 63% absorption at one length is not the same as high QE.', ko: '흡수 길이 $1/\\alpha$는 유용한 scale이지만, 한 흡수 길이에서 63% 흡수된다는 것이 높은 QE와 동일하진 않습니다.' },
        ],
      },
      {
        title: { en: 'Device Implications', ko: '소자 설계 의미' },
        items: [
          { en: 'Thicker silicon improves red/NIR absorption but may worsen crosstalk if carriers are generated far from the intended photodiode.', ko: '두꺼운 silicon은 red/NIR 흡수를 개선하지만 의도한 포토다이오드에서 먼 곳에 carrier가 생성되면 crosstalk를 악화시킬 수 있습니다.' },
          { en: 'Backside reflectors or light trapping can convert escape transmission into another absorption opportunity.', ko: 'Backside reflector 또는 light trapping은 escape transmission을 추가 흡수 기회로 바꿀 수 있습니다.' },
          { en: 'Collection probability $\\eta_{\\text{cc}}(z)$ should be multiplied with the absorption profile for electrical QE.', ko: '전기적 QE를 위해서는 흡수 profile에 collection probability $\\eta_{\\text{cc}}(z)$를 곱해야 합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'Beer-Lambert absorption ignores interference from upper films and coherent standing waves inside the stack.', ko: 'Beer-Lambert 흡수는 상부 박막 간섭과 스택 내부 coherent standing wave를 무시합니다.' },
          { en: 'The model does not include doping, temperature, strain, or free-carrier absorption changes in silicon.', ko: '모델은 silicon의 doping, temperature, strain, free-carrier absorption 변화를 포함하지 않습니다.' },
          { en: 'It predicts where photons are absorbed, not whether generated carriers are collected.', ko: '이 모델은 광자가 어디서 흡수되는지를 예측할 뿐, 생성 carrier가 수집되는지는 예측하지 않습니다.' },
        ],
      },
    ],
    references: [refs.green2008, refs.yokogawa2017, refs.han2020],
  },
  'microlens-raytrace': {
    title: { en: 'Geometric Microlens Ray Tracing', ko: '기하광학 마이크로렌즈 광선 추적' },
    summary: {
      en: 'This simulator traces rays through a smooth microlens surface using local surface normals and Snell refraction.',
      ko: '이 시뮬레이터는 매끄러운 마이크로렌즈 표면의 국소 법선과 스넬 굴절을 사용해 광선을 추적합니다.',
    },
    intuition: {
      en: 'A microlens sits on top of each pixel like a tiny magnifying glass, focusing incoming light onto the photodiode. This tool draws individual light rays through that lens — bending each one at the surface using basic Snell refraction — and counts how many hit the active area. It is the same idea as ray-tracing a camera lens, just at the micron scale.',
      ko: '마이크로렌즈은 각 픽셀 위에 놓인 작은 돋보기처럼, 들어오는 빛을 포토다이오드로 집광합니다. 이 도구는 그 렌즈를 통과하는 광선 하나하나를 그려, 표면마다 스넬의 법칙으로 굴절시키고 활성 영역에 몇 개가 닿는지를 셉니다. 마이크론 단위로 축소된 카메라 렌즈 광선 추적이라 생각하면 됩니다.',
    },
    formulas: [
      {
        label: { en: 'Superellipse lens profile', ko: '초타원 렌즈 프로파일' },
        equation: 'z(r) = h \\left[ 1 - \\left( \\frac{r}{R} \\right)^n \\right]^{1/n}',
        variables: [
          { symbol: 'z(r)', description: { en: 'Height at radius $r$', ko: '반경 $r$에서의 높이' } },
          { symbol: 'h', description: { en: 'Lens vertex height (sag)', ko: '렌즈 꼭짓점 높이 (sag)' } },
          { symbol: 'R', description: { en: 'Lens aperture radius', ko: '렌즈 개구 반경' } },
          { symbol: 'n', description: { en: 'Superellipse exponent', ko: '초타원 지수' } },
        ],
        note: { en: 'The exponent controls whether the lens is rounded, flat-topped, or steep-edged.', ko: '지수는 렌즈가 둥근지, 평탄한지, 가장자리가 가파른지를 조절합니다.' },
      },
      {
        label: { en: 'Snell refraction', ko: '스넬 굴절' },
        equation: 'n_1 \\sin(\\theta_1) = n_2 \\sin(\\theta_2)',
        variables: [
          { symbol: 'n_1, n_2', description: { en: 'Refractive indices across the interface', ko: '경계면 양단의 굴절률' } },
          { symbol: '\\theta_1, \\theta_2', description: { en: 'Angles relative to the surface normal', ko: '표면 법선 기준의 각도' } },
        ],
        note: { en: 'Refraction is evaluated at the local surface normal, not at the global vertical axis.', ko: '굴절은 전역 수직축이 아니라 국소 표면 법선 기준으로 계산합니다.' },
      },
      {
        label: { en: 'Spot efficiency proxy', ko: '스폿 효율 근사' },
        equation: '\\eta_{\\text{coll}} \\approx \\frac{N_{\\text{hits}}}{N_{\\text{total}}}',
        variables: [
          { symbol: '\\eta_{\\text{coll}}', description: { en: 'Geometric collection efficiency', ko: '기하학적 집광 효율' } },
          { symbol: 'N_{\\text{hits}}', description: { en: 'Number of rays hitting the photodiode', ko: '포토다이오드에 닿은 광선 수' } },
        ],
        note: { en: 'Ray-count efficiency is intuitive but not a wave-optical QE calculation.', ko: '광선 개수 기반 효율은 직관적이지만 파동광학 QE 계산은 아닙니다.' },
      },
    ],
    concepts: [
      { en: 'Microlens shift is needed when chief rays arrive at high CRA.', ko: '주광선각이 큰 경우 마이크로렌즈 시프트가 필요합니다.' },
      { en: 'Geometric ray tracing breaks down when pixel pitch approaches the wavelength.', ko: '픽셀 피치가 파장에 가까워지면 기하광학 추적은 한계가 있습니다.' },
      { en: 'Use RCWA/FDTD for diffraction, interference, and sub-wavelength metal-grid effects.', ko: '회절, 간섭, 서브파장 금속 그리드 효과는 RCWA/FDTD가 필요합니다.' },
    ],
    sections: [
      {
        title: { en: 'Ray-Trace Workflow', ko: '광선 추적 절차' },
        items: [
          { en: 'Define lens sag and local surface normal, apply Snell law at the curved interface, then propagate rays to the photodiode plane.', ko: 'Lens sag와 local surface normal을 정의하고 곡면 계면에서 Snell law를 적용한 뒤 포토다이오드 평면까지 ray를 전파합니다.' },
          { en: 'Collection is usually scored by hit fraction or weighted flux inside the photodiode aperture.', ko: '집광은 보통 포토다이오드 aperture 안에 들어온 hit fraction 또는 weighted flux로 평가합니다.' },
          { en: 'Sweep CRA and lens shift together; good on-axis focusing does not guarantee edge-pixel collection.', ko: 'CRA와 lens shift를 함께 sweep해야 합니다. On-axis focusing이 좋아도 edge-pixel collection이 보장되지는 않습니다.' },
        ],
      },
      {
        title: { en: 'Geometry Sensitivities', ko: '형상 민감도' },
        items: [
          { en: 'Increasing sag usually strengthens focusing but can also increase aberration or shift focus above/below the photodiode.', ko: 'Sag 증가는 보통 focusing을 강화하지만 aberration을 늘리거나 초점을 포토다이오드 위/아래로 이동시킬 수 있습니다.' },
          { en: 'A smaller aperture gap improves optical fill factor, but process merger and surface slope become limiting factors.', ko: '작은 aperture gap은 optical fill factor를 높이지만 공정 merger와 surface slope가 제한 요인이 됩니다.' },
          { en: 'Index contrast between lens and surrounding medium controls bending strength through $n_1\\sin\\theta_1=n_2\\sin\\theta_2$.', ko: '렌즈와 주변 매질의 굴절률 대비는 $n_1\\sin\\theta_1=n_2\\sin\\theta_2$를 통해 굴절 세기를 결정합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'Ray tracing ignores diffraction, interference, polarization, and finite-wavelength scattering.', ko: 'Ray tracing은 회절, 간섭, 편광, 유한 파장 산란을 무시합니다.' },
          { en: 'It does not model metal-grid shadowing, DTI waveguiding, or color-filter relief unless those surfaces are explicitly included.', ko: '명시적으로 표면을 포함하지 않는 한 metal-grid shadowing, DTI waveguiding, color-filter relief를 모델링하지 않습니다.' },
          { en: 'For pixels near the wavelength scale, validate ray trends with electromagnetic solvers.', ko: '픽셀이 파장 scale에 가까우면 ray trend를 전자기 solver로 검증해야 합니다.' },
        ],
      },
    ],
    references: [refs.agranov2003, refs.hwang2023, refs.catrysse2002],
  },
  'microlens-process-shape': {
    title: { en: 'Layout-to-Etch Microlens Shape Model', ko: '레이아웃-식각 마이크로렌즈 형상 모델' },
    summary: {
      en: 'This page treats microlens formation as a three-stage surrogate: lithographic resist volume, volume-conserving thermal reflow, and plasma etch transfer that closes residual gap while eroding height. The goal is not a foundry recipe, but a transparent model that can later be fitted to AFM/SEM metrology.',
      ko: '이 페이지는 마이크로렌즈 형성을 lithography resist 체적, 체적 보존형 thermal reflow, 그리고 잔류 gap을 닫는 동시에 높이를 손실시키는 plasma etch transfer의 3단계 surrogate로 봅니다. 목표는 foundry recipe가 아니라, 나중에 AFM/SEM 계측으로 보정할 수 있는 투명한 모델입니다.',
    },
    intuition: {
      en: 'Start with a printed resist island. Heating lets surface tension round it into a lens cap, so a wider cap usually means a lower cap if the same material volume is spread out. Plasma etch then transfers that cap into the target layer: longer etch can remove the valley between lenses, but it can also flatten the lens. The useful process window is therefore a compromise between zero gap, enough height, acceptable curvature, and no lens merger.',
      ko: '먼저 포토레지스트 island가 인쇄됩니다. 가열하면 표면장력이 이를 둥근 렌즈 cap으로 만들고, 같은 재료 체적이 더 넓게 퍼질수록 cap 높이는 낮아지는 경향이 있습니다. 그 다음 plasma etch가 이 cap을 목표층으로 전사합니다. Etch 시간이 길면 렌즈 사이 valley를 없앨 수 있지만 동시에 렌즈가 납작해질 수 있습니다. 따라서 유효 공정 창은 zero gap, 충분한 높이, 적절한 곡률, lens merger 회피 사이의 절충입니다.',
    },
    assumptions: [
      { en: 'The resist island is represented by pitch, mask width, thickness, and a footprint-shape exponent; local corner rounding and stochastic CD variation are not solved.', ko: 'Resist island는 pitch, mask width, thickness, footprint-shape exponent로 표현하며, local corner rounding과 stochastic CD variation은 직접 풀지 않습니다.' },
      { en: 'Reflow is volume constrained and monotonic with a normalized temperature/time budget; the coefficients are user-calibrated surrogate parameters.', ko: 'Reflow는 체적 제약을 받으며 정규화된 온도/시간 budget에 대해 단조적으로 변한다고 두고, 계수는 사용자가 보정하는 surrogate parameter입니다.' },
      { en: 'Etch transfer is split into lateral gap closure and vertical height loss, following the DOE variables exposed by the CIS plasma-etch literature.', ko: 'Etch transfer는 CIS plasma-etch 문헌의 DOE 변수에 맞춰 lateral gap closure와 vertical height loss로 분리합니다.' },
    ],
    outputs: [
      { en: 'Initial gap, post-reflow gap, final gap, final height, fill factor, vertex radius, f-number proxy, profile exponent, and zero-gap etch-time estimate.', ko: '초기 gap, reflow 후 gap, 최종 gap, 최종 높이, fill factor, vertex radius, f-number 근사, profile exponent, zero-gap etch-time 추정값을 출력합니다.' },
      { en: 'Cross-section, final 3D surface wireframe, and etch-time response curves for gap and height retention.', ko: '단면, 최종 3D surface wireframe, gap 및 height retention의 etch-time response curve를 보여줍니다.' },
    ],
    validationExamples: [
      { en: 'At fixed reflow settings, increasing etch time should reduce $g_f$ while reducing height retention; if metrology shows the opposite, refit the lateral and vertical etch gains separately.', ko: 'Reflow 조건을 고정하면 etch time 증가에 따라 $g_f$는 감소하고 height retention은 낮아져야 합니다. 계측이 반대 경향이면 lateral/vertical etch gain을 따로 재피팅해야 합니다.' },
      { en: 'At fixed etch settings, a larger reflow spread gain should reduce residual gap but also lower height through volume conservation.', ko: 'Etch 조건을 고정하면 reflow spread gain 증가는 residual gap을 줄이지만 체적 보존 때문에 height를 낮춰야 합니다.' },
    ],
    formulas: [
      {
        label: { en: 'Lithographic starting gap', ko: 'Lithography 시작 gap' },
        equation: 'g_0 = \\max(0, p - w_m)',
        variables: [
          { symbol: 'g_0', description: { en: 'Initial space between neighboring resist islands', ko: '인접 resist island 사이의 초기 간격' } },
          { symbol: 'p', description: { en: 'Pixel pitch or microlens array pitch', ko: '픽셀 pitch 또는 마이크로렌즈 어레이 pitch' } },
          { symbol: 'w_m', description: { en: 'Lithographic mask island width after clamping to allowed pitch limits', ko: '허용 pitch 범위로 제한된 lithographic mask island 폭' } },
        ],
        note: { en: 'A small layout gap helps optical fill factor but increases the chance that resist islands touch or merge during reflow.', ko: '작은 layout gap은 광학 fill factor에는 유리하지만 reflow 중 resist island가 닿거나 합쳐질 위험을 키웁니다.' },
      },
      {
        label: { en: 'Normalized reflow budget', ko: '정규화된 reflow budget' },
        equation: 'B_r = c_T \\hat{T} + c_t \\hat{t}, \\quad \\hat{T}=\\frac{T_r-T_{\\min}}{T_{\\max}-T_{\\min}}, \\quad \\hat{t}=\\frac{\\log(1+t_r/t_0)}{\\log(1+t_{\\max}/t_0)}',
        variables: [
          { symbol: 'B_r', description: { en: 'Dimensionless reflow budget used by the simulator', ko: '시뮬레이터가 사용하는 무차원 reflow budget' } },
          { symbol: 'T_r', description: { en: 'Reflow temperature', ko: 'Reflow 온도' } },
          { symbol: 't_r', description: { en: 'Reflow time', ko: 'Reflow 시간' } },
          { symbol: 'c_T, c_t', description: { en: 'Temperature and time weights; defaults encode a directional surrogate, not universal kinetics', ko: '온도 및 시간 가중치; 기본값은 범용 kinetics가 아니라 방향성 surrogate입니다.' } },
        ],
        note: { en: 'The logarithmic time term makes early reflow changes stronger than late soak changes, matching the qualitative behavior of many resist reflow processes.', ko: '로그 시간 항은 초기 reflow 변화가 장시간 soak 변화보다 크도록 하며, 많은 resist reflow 공정의 정성적 거동과 맞습니다.' },
      },
      {
        label: { en: 'Reflow spread and residual gap', ko: 'Reflow spread 및 잔류 gap' },
        equation: '\\Delta_r = G_r B_r(k_0+k_h h_0+k_g g_0), \\quad w_r=\\min(1.04p, w_m+2\\Delta_r), \\quad g_r=\\max(0,p-w_r)',
        variables: [
          { symbol: '\\Delta_r', description: { en: 'Lateral spread per side during reflow', ko: 'Reflow 중 한쪽 방향 lateral spread' } },
          { symbol: 'G_r', description: { en: 'User calibration multiplier for reflow spread gain', ko: 'Reflow spread gain에 대한 사용자 보정 multiplier' } },
          { symbol: 'h_0', description: { en: 'Initial resist thickness', ko: '초기 resist 두께' } },
          { symbol: 'w_r', description: { en: 'Post-reflow lens footprint width', ko: 'Reflow 후 렌즈 footprint 폭' } },
          { symbol: 'g_r', description: { en: 'Gap after reflow but before etch transfer', ko: 'Reflow 후, etch transfer 전 gap' } },
        ],
        note: { en: 'The constants are intentionally exposed as surrogate coefficients: a real process should replace them with DOE-fitted values.', ko: '상수들은 의도적으로 surrogate coefficient로 둔 것입니다. 실제 공정에서는 DOE fitting 값으로 치환해야 합니다.' },
      },
      {
        label: { en: 'Volume-constrained reflow height', ko: '체적 제약 기반 reflow 높이' },
        equation: 'V_0 \\approx A_m h_0, \\quad V_r \\approx C_{\\text{cap}} A_r h_r, \\quad h_r \\approx \\frac{G_v\\eta_v A_m h_0}{C_{\\text{cap}}A_r}',
        variables: [
          { symbol: 'V_0, V_r', description: { en: 'Resist volume before and after reflow', ko: 'Reflow 전후 resist 체적' } },
          { symbol: 'A_m, A_r', description: { en: 'Mask island area and reflowed lens footprint area', ko: 'Mask island 면적 및 reflow 후 렌즈 footprint 면적' } },
          { symbol: 'C_{\\text{cap}}', description: { en: 'Shape factor for the cap profile; a parabolic cap has a different value from a spherical cap', ko: 'Cap profile의 형상 계수; 포물면 cap과 구면 cap은 다른 값을 가집니다.' } },
          { symbol: 'G_v', description: { en: 'User calibration multiplier for effective volume retention', ko: '유효 체적 보존에 대한 사용자 보정 multiplier' } },
          { symbol: '\\eta_v', description: { en: 'Effective volume retention after reflow', ko: 'Reflow 후 유효 체적 보존율' } },
          { symbol: 'h_r', description: { en: 'Reflowed lens height before etch transfer', ko: 'Etch transfer 전 reflow 렌즈 높이' } },
        ],
        note: { en: 'This is the key physical constraint: if the footprint grows faster than volume, lens height must drop.', ko: '핵심 물리 제약입니다. Footprint가 체적보다 빠르게 커지면 렌즈 높이는 낮아져야 합니다.' },
      },
      {
        label: { en: 'Etch-transfer gap closure and height loss', ko: 'Etch-transfer gap closure 및 height loss' },
        equation: 'g_f=\\max(0,g_r-2G_{\\ell}v_{\\ell,0}t_e), \\quad h_f=h_r(1-L_e), \\quad L_e=\\operatorname{clip}(G_z v_{z,0} t_e,0,L_{\\max}), \\quad t_{zg}=\\frac{g_r}{2G_{\\ell}v_{\\ell,0}}',
        variables: [
          { symbol: 'g_f', description: { en: 'Final gap after etch transfer', ko: 'Etch transfer 후 최종 gap' } },
          { symbol: 'h_f', description: { en: 'Final transferred lens height', ko: '전사 후 최종 렌즈 높이' } },
          { symbol: 'G_{\\ell},G_z', description: { en: 'User calibration multipliers for lateral etch closure and vertical height loss', ko: 'Lateral etch closure 및 vertical height loss에 대한 사용자 보정 multiplier' } },
          { symbol: 'v_{\\ell,0},v_{z,0}', description: { en: 'Base lateral closure and vertical flattening rates from the normalized surrogate', ko: '정규화 surrogate의 기본 lateral closure 및 vertical flattening rate' } },
          { symbol: 't_e', description: { en: 'Etch time', ko: 'Etch 시간' } },
          { symbol: 't_{zg}', description: { en: 'Estimated etch time needed to reach zero gap before clipping', ko: 'Clipping 전 zero gap 도달에 필요한 추정 etch 시간' } },
        ],
        note: { en: 'In this surrogate, polymerizing gas increases lateral gap closure and reduces height loss; mask thickness changes transfer robustness; the four gains expose the DOE fitting knobs.', ko: '이 surrogate에서 polymerizing gas는 lateral gap closure를 키우고 height loss를 줄이며, mask thickness는 transfer robustness를 바꿉니다. 네 개의 gain은 DOE fitting knob입니다.' },
      },
      {
        label: { en: 'Final 3D surface profile', ko: '최종 3D 표면 profile' },
        equation: '\\rho=\\left[\\left|\\frac{x}{a}\\right|^n+\\left|\\frac{y}{a}\\right|^n\\right]^{1/n}, \\quad z(x,y)=h_f\\max(0,1-\\rho^q)',
        variables: [
          { symbol: '\\rho', description: { en: 'Normalized superellipse radius', ko: '정규화된 superellipse 반경' } },
          { symbol: 'a', description: { en: 'Final half-width or aperture radius of the lens footprint', ko: '최종 lens footprint의 반폭 또는 개구 반경' } },
          { symbol: 'n', description: { en: 'Footprint exponent: 2 is circular, larger values approach rounded-square or square footprints', ko: 'Footprint 지수: 2는 원형, 더 큰 값은 rounded-square 또는 square footprint에 가까워집니다.' } },
          { symbol: 'q', description: { en: 'Profile exponent controlling edge steepness and cap flatness', ko: '가장자리 기울기와 cap flatness를 제어하는 profile 지수' } },
        ],
        note: { en: 'This links the 2D layout aperture to the 3D height field used in the wireframe view.', ko: '2D layout aperture를 wireframe view에서 쓰는 3D height field와 연결합니다.' },
      },
      {
        label: { en: 'Curvature, focal length, and f-number proxy', ko: '곡률, 초점거리, f-number 근사' },
        equation: 'R_{\\text{vtx}}\\approx\\frac{a^2+h_f^2}{2h_f}, \\quad f\\approx\\frac{R_{\\text{vtx}}}{n_{\\ell}-1}, \\quad N\\approx\\frac{f}{2a}',
        variables: [
          { symbol: 'R_{\\text{vtx}}', description: { en: 'Vertex radius of curvature estimated from a spherical-cap approximation', ko: '구면 cap 근사로 추정한 꼭짓점 곡률 반경' } },
          { symbol: 'f', description: { en: 'Thin-lens focal length proxy', ko: 'Thin-lens 초점거리 근사값' } },
          { symbol: 'n_{\\ell}', description: { en: 'Microlens refractive index', ko: '마이크로렌즈 굴절률' } },
          { symbol: 'N', description: { en: 'Microlens f-number proxy', ko: '마이크로렌즈 f-number 근사값' } },
        ],
        note: { en: 'The optical numbers are screening metrics only; diffraction, finite stack thickness, and CRA shift require ray tracing or EM simulation.', ko: '광학 수치는 screening metric입니다. 회절, 유한 stack 두께, CRA shift는 ray tracing 또는 EM simulation이 필요합니다.' },
      },
    ],
    concepts: [
      { en: 'The model separates layout-limited gap, reflow-limited spread/height, and etch-limited transfer loss so each failure mode is visible.', ko: '모델은 layout-limited gap, reflow-limited spread/height, etch-limited transfer loss를 분리해 각 failure mode가 보이도록 합니다.' },
      { en: 'Zero gap is not automatically good: over-reflow can merge lenses, while over-etch can erase sag and weaken focusing.', ko: 'Zero gap이 항상 좋은 것은 아닙니다. 과도한 reflow는 lens merger를 만들 수 있고, 과도한 etch는 sag를 지워 집광을 약화시킬 수 있습니다.' },
      { en: 'The browser result should be interpreted as a process-window map before calibration, not as a released process recipe.', ko: '브라우저 결과는 calibration 전에는 release된 공정 recipe가 아니라 process-window map으로 해석해야 합니다.' },
    ],
    sections: [
      {
        title: { en: 'How To Calibrate This Surrogate', ko: '이 surrogate를 보정하는 방법' },
        items: [
          { en: 'Measure $g_f$, $h_f$, footprint width, and profile exponent from AFM/SEM across a DOE grid of mask thickness, polymerizing gas flow, and etch time.', ko: 'Mask thickness, polymerizing gas flow, etch time DOE grid에서 AFM/SEM으로 $g_f$, $h_f$, footprint width, profile exponent를 측정합니다.' },
          { en: 'Fit the reflow coefficients first using pre-etch or short-etch samples, then fit $v_{\\ell}$ and $v_z$ from etch-time sweeps.', ko: '먼저 pre-etch 또는 short-etch sample로 reflow coefficient를 맞추고, etch-time sweep으로 $v_{\\ell}$ 및 $v_z$를 fitting합니다.' },
          { en: 'Validate the optical proxy by comparing predicted $R_{\\text{vtx}}$, $N$, and fill factor against ray-trace collection or silicon-level QE/crosstalk data.', ko: '예측된 $R_{\\text{vtx}}$, $N$, fill factor를 ray-trace collection 또는 silicon-level QE/crosstalk data와 비교해 optical proxy를 검증합니다.' },
        ],
      },
      {
        title: { en: 'What The References Contribute', ko: '레퍼런스가 제공하는 역할' },
        items: [
          { en: 'Ristoiu et al. motivates the DOE variables used here: mask thickness, polymerizing gas, and etch time as drivers of final microlens gap and height.', ko: 'Ristoiu et al.은 여기서 쓰는 DOE 변수, 즉 mask thickness, polymerizing gas, etch time이 최종 microlens gap/height를 움직인다는 근거를 제공합니다.' },
          { en: 'Baillie and Gendler frame the zero-space problem: residual space reduces optical fill factor, but insufficient lithographic space can cause reflow merger.', ko: 'Baillie and Gendler는 zero-space 문제를 정의합니다. Residual space는 optical fill factor를 낮추지만, lithographic space가 너무 작으면 reflow merger가 발생할 수 있습니다.' },
          { en: 'Jin, Liu, and Yang connect zero-space microlens geometry to AFM characterization and sensor-level sensitivity/crosstalk tests.', ko: 'Jin, Liu, Yang은 zero-space microlens geometry를 AFM characterization 및 sensor-level sensitivity/crosstalk test와 연결합니다.' },
          { en: 'Tan, Goh, and Kim support the aperture-geometry and regression view of thermal-reflow microlens fabrication.', ko: 'Tan, Goh, Kim은 thermal-reflow microlens fabrication에서 aperture geometry와 regression 기반 모델링 관점을 뒷받침합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The model does not solve surface-tension fluid dynamics, resist viscosity, contact-angle pinning, or plasma sheath chemistry.', ko: '이 모델은 표면장력 유체역학, resist viscosity, contact-angle pinning, plasma sheath chemistry를 직접 풀지 않습니다.' },
          { en: 'It does not include color-filter topography, neighboring-lens wetting asymmetry, wafer-edge non-uniformity, or lens shift for CRA compensation.', ko: 'Color-filter topography, 인접 lens wetting asymmetry, wafer-edge non-uniformity, CRA 보정용 lens shift는 포함하지 않습니다.' },
          { en: 'Use this page to choose DOE regions; use measured profiles plus optical simulation before making device-performance claims.', ko: '이 페이지는 DOE 영역 선택에 사용하고, device-performance 주장은 실측 profile과 optical simulation을 함께 사용해야 합니다.' },
        ],
      },
    ],
    references: [
      {
        ...refs.ristoiu2020,
        note: { en: 'Closest direct CIS DOE reference for plasma etch transfer of reflowed microlenses.', ko: 'Reflowed microlens의 plasma etch transfer를 다룬 가장 직접적인 CIS DOE 레퍼런스입니다.' },
      },
      {
        ...refs.baillie2004,
        note: { en: 'Defines the zero-space layout/fabrication motivation and merger-risk tradeoff.', ko: 'Zero-space layout/fabrication 동기와 merger-risk tradeoff를 설명합니다.' },
      },
      {
        ...refs.jin2011,
        note: { en: 'Connects zero-space geometry to AFM and silicon-level sensitivity/crosstalk measurements.', ko: 'Zero-space geometry를 AFM 및 silicon-level sensitivity/crosstalk 계측과 연결합니다.' },
      },
      {
        ...refs.tan2020,
        note: { en: 'Useful for aperture-geometry and regression-based thermal-reflow profile thinking.', ko: 'Aperture geometry 및 regression 기반 thermal-reflow profile 사고에 유용합니다.' },
      },
    ],
  },
  'mla-array': {
    title: { en: 'Microlens Array Surface Geometry', ko: '마이크로렌즈 어레이 표면 형상' },
    summary: {
      en: 'The MLA visualizer extends a single lens profile into periodic arrays and shows how pitch, curvature, asymmetry, and ray direction change array behavior.',
      ko: 'MLA 시각화는 단일 렌즈 프로파일을 주기 어레이로 확장하고 pitch, 곡률, 비대칭성, 광선 방향이 어레이 거동을 어떻게 바꾸는지 보여줍니다.',
    },
    intuition: {
      en: 'A real sensor has not one microlens but a whole grid of them — one per pixel, like an array of tiny bumps on a glass sheet. This visualiser stitches a single lens profile into that grid so you can see how pitch, asymmetry, and curvature interact across the array. Gaps between lenses or overlapping skirts both reduce optical fill factor.',
      ko: '실제 센서는 단일 렌즈가 아니라 픽셀마다 하나씩 격자로 배열된 마이크로렌즈 어레이를 가집니다. 유리판 위에 작은 볼록들이 격자로 깔린 모습을 상상해 보세요. 이 시각화는 단일 렌즈 형상을 그 격자에 이어 붙여 pitch, 비대칭, 곡률이 어레이에서 어떻게 결합되는지를 보여줍니다. 렌즈 사이의 틈이나 가장자리 겹침은 모두 광학 fill factor를 떨어뜨립니다.',
    },
    formulas: [
      {
        label: { en: 'Anisotropic normalized radius', ko: '비등방 정규화 반경' },
        equation: '\\rho = \\left( \\left| \\frac{x}{R_x} \\right|^n + \\left| \\frac{y}{R_y} \\right|^n \\right)^{1/n}',
        variables: [
          { symbol: '\\rho', description: { en: 'Normalized radial coordinate', ko: '정규화된 반경 좌표 $\\rho$' } },
          { symbol: 'R_x, R_y', description: { en: 'Lateral radii of the lens', ko: '가로/세로 반경 $R_x, R_y$' } },
        ],
        note: { en: 'Different Rx and Ry produce astigmatic or rectangular-pixel lens footprints.', ko: 'Rx와 Ry가 다르면 비점수차 또는 직사각형 픽셀용 렌즈 footprint가 됩니다.' },
      },
      {
        label: { en: 'Surface sag', ko: '표면 sag' },
        equation: 'z(x, y) = h \\cdot \\max(0, 1 - \\rho^\\alpha)',
        variables: [
          { symbol: 'z', description: { en: 'Surface height', ko: '표면 높이 $z$' } },
          { symbol: '\\alpha', description: { en: 'Curvature power factor', ko: '곡률 지수 $\\alpha$' } },
        ],
        note: { en: 'Height h and curvature alpha set focusing strength and edge steepness.', ko: '높이 h와 곡률 alpha는 집광 세기와 가장자리 기울기를 결정합니다.' },
      },
      {
        label: { en: 'Array pitch', ko: '어레이 피치' },
        equation: '(x, y)_{m,n} = (m P_x, n P_y)',
        variables: [
          { symbol: 'P_x, P_y', description: { en: 'Unit cell spacing in X and Y', ko: 'X 및 Y 방향의 격자 간격 $P_x, P_y$' } },
          { symbol: 'm, n', description: { en: 'Integer array indices', ko: '격자 인덱스 $m, n$' } },
        ],
        note: { en: 'Array spacing controls gap, overlap risk, and fill factor.', ko: '어레이 간격은 gap, overlap 위험, fill factor를 결정합니다.' },
      },
    ],
    concepts: [
      { en: 'Array-level gap and asymmetry matter as much as single-lens curvature.', ko: '어레이 수준의 gap과 비대칭성은 단일 렌즈 곡률만큼 중요합니다.' },
      { en: 'A 3D surface view is geometric evidence, not a wave-optical simulation result.', ko: '3D 표면 뷰는 기하학적 근거이지 파동광학 시뮬레이션 결과는 아닙니다.' },
      { en: 'Use this with the process-shape tool when connecting layout to final reflow geometry.', ko: '레이아웃과 최종 리플로우 형상을 연결할 때 공정 형상 도구와 함께 사용하세요.' },
    ],
    sections: [
      {
        title: { en: 'Array Geometry Checks', ko: '어레이 형상 체크' },
        items: [
          { en: 'Check pitch consistency first: array centers follow $(x,y)_{m,n}=(mP_x,nP_y)$, so anisotropic pitch directly changes gap and fill factor.', ko: '먼저 pitch consistency를 확인합니다. 어레이 중심은 $(x,y)_{m,n}=(mP_x,nP_y)$를 따르므로 비등방 pitch는 gap과 fill factor를 직접 바꿉니다.' },
          { en: 'Compare footprint exponent and profile exponent separately; one controls plan-view shape and the other controls vertical sag.', ko: 'Footprint exponent와 profile exponent를 분리해 비교합니다. 하나는 평면 형상, 다른 하나는 vertical sag를 제어합니다.' },
          { en: 'Look for edge overlap or residual valley regions because both can signal process or optical fill-factor risk.', ko: '가장자리 overlap 또는 residual valley 영역을 찾습니다. 둘 다 공정 또는 optical fill-factor 위험 신호일 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Connection To Process', ko: '공정과의 연결' },
        items: [
          { en: 'Lithographic island shape, reflow spread, and etch transfer determine the final footprint more than the ideal lens equation alone.', ko: 'Lithographic island shape, reflow spread, etch transfer가 이상적인 lens equation보다 최종 footprint를 더 크게 좌우합니다.' },
          { en: 'A zero-space design must be checked against merger risk, not just final gap.', ko: 'Zero-space 설계는 final gap뿐 아니라 merger risk도 함께 확인해야 합니다.' },
          { en: 'AFM/SEM profile samples can calibrate $R_x$, $R_y$, $h$, and exponent values used by this surface model.', ko: 'AFM/SEM profile sample로 이 표면 모델의 $R_x$, $R_y$, $h$, exponent 값을 보정할 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The 3D surface is geometric only; it does not compute optical focus, diffraction, or collection efficiency.', ko: '3D 표면은 기하 형상일 뿐 optical focus, diffraction, collection efficiency를 계산하지 않습니다.' },
          { en: 'It assumes identical repeated lenses and does not include wafer-level process gradients or local defects.', ko: '동일 렌즈 반복을 가정하며 wafer-level process gradient 또는 local defect를 포함하지 않습니다.' },
          { en: 'Use ray tracing or FDTD after this view when the shape must be linked to QE or crosstalk.', ko: '형상을 QE 또는 crosstalk와 연결해야 할 때는 이 view 이후 ray tracing 또는 FDTD를 사용해야 합니다.' },
        ],
      },
    ],
    references: [refs.agranov2003, refs.baillie2004, refs.ristoiu2020],
  },
  'fdti-pixel': {
    title: { en: 'DTI Optical Confinement and Crosstalk', ko: 'DTI 광 구속 및 크로스토크' },
    summary: {
      en: 'Deep trench isolation reduces optical crosstalk by redirecting or blocking lateral light leakage between neighboring pixels.',
      ko: 'Deep trench isolation은 인접 픽셀 사이로 새는 횡방향 빛을 반사하거나 차단해 광학 크로스토크를 줄입니다.',
    },
    intuition: {
      en: 'Picture each pixel as a small room. Without walls, light leaking sideways from one room shows up as colour bleeding in the neighbour — that is optical crosstalk. DTI (Deep Trench Isolation) builds tiny vertical walls between pixels, reflecting or blocking that sideways light so each pixel mostly sees its own photons.',
      ko: '각 픽셀을 작은 방이라고 생각해 보세요. 벽이 없으면 한 방에서 옆방으로 빛이 새어 나가 색이 번지는 현상(광학 크로스토크)이 생깁니다. DTI(Deep Trench Isolation)는 픽셀 사이에 수직 벽을 세워 옆으로 새는 빛을 반사하거나 차단함으로써, 각 픽셀이 자기 광자를 위주로 받도록 만듭니다.',
    },
    formulas: [
      {
        label: { en: 'Critical angle', ko: '임계각' },
        equation: '\\theta_c = \\arcsin\\left( \\frac{n_{\\text{trench}}}{n_{\\text{Si}}} \\right)',
        variables: [
          { symbol: '\\theta_c', description: { en: 'Angle for Total Internal Reflection (TIR)', ko: '전반사 임계각 $\\theta_c$' } },
          { symbol: 'n_{\\text{trench}}', description: { en: 'Refractive index of trench filling', ko: '트렌치 충전재의 굴절률 $n_{\\text{trench}}$' } },
        ],
        note: { en: 'A low-index trench can confine silicon-guided rays by total internal reflection.', ko: '저굴절률 트렌치는 전반사를 통해 실리콘 내 광선을 가둘 수 있습니다.' },
      },
      {
        label: { en: 'Optical crosstalk proxy', ko: '광학 크로스토크 근사' },
        equation: 'XT = \\frac{P_{\\text{neighbor}}}{P_{\\text{center}} + P_{\\text{neighbor}}}',
        variables: [
          { symbol: 'P', description: { en: 'Optical power in pixel', ko: '픽셀의 광파워 $P$' } },
        ],
        note: { en: 'The simulator reports a structural optical trend, not carrier diffusion crosstalk.', ko: '시뮬레이터는 구조적 광학 경향을 보여주며 carrier diffusion 크로스토크는 포함하지 않습니다.' },
      },
      {
        label: { en: 'Absorption path benefit', ko: '흡수 경로 이득' },
        equation: 'A = 1 - e^{-\\alpha L_{\\text{eff}}}',
        variables: [
          { symbol: 'L_{\\text{eff}}', description: { en: 'Effective absorption path length', ko: '유효 흡수 경로 길이 $L_{\\text{eff}}$' } },
        ],
        note: { en: 'DTI and backside structures can increase effective path length in silicon.', ko: 'DTI와 후면 구조는 실리콘 내 유효 경로 길이를 늘릴 수 있습니다.' },
      },
    ],
    concepts: [
      { en: 'FDTI and BDTI differ in how much of the silicon depth is isolated.', ko: 'FDTI와 BDTI는 실리콘 깊이 중 어느 범위를 격리하는지가 다릅니다.' },
      { en: 'Metal-filled trenches improve shielding but introduce absorption and process complexity.', ko: '금속 충전 트렌치는 차광을 개선하지만 흡수와 공정 복잡도를 늘립니다.' },
      { en: 'Electrical isolation and optical isolation are related but not identical metrics.', ko: '전기적 격리와 광학적 격리는 관련되지만 동일한 지표가 아닙니다.' },
    ],
    sections: [
      {
        title: { en: 'Optical Confinement Logic', ko: '광 구속 논리' },
        items: [
          { en: 'DTI works by interrupting lateral propagation paths before photons reach a neighboring photodiode.', ko: 'DTI는 광자가 이웃 포토다이오드에 도달하기 전 lateral propagation path를 끊는 방식으로 동작합니다.' },
          { en: 'Total internal reflection is possible when rays in silicon meet a lower-index trench beyond $\\theta_c=\\arcsin(n_{\\text{trench}}/n_{\\text{Si}})$.', ko: 'Silicon 내 ray가 낮은 굴절률 trench를 $\\theta_c=\\arcsin(n_{\\text{trench}}/n_{\\text{Si}})$보다 큰 각도로 만나면 전반사가 가능합니다.' },
          { en: 'Absorbing or metal-filled trenches can block leakage but may also introduce parasitic absorption and process complexity.', ko: '흡수성 또는 금속 충전 trench는 leakage를 막지만 parasitic absorption과 공정 복잡도를 늘릴 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Design Checks', ko: '설계 체크' },
        items: [
          { en: 'Compare center-pixel absorption and neighbor-pixel absorption; crosstalk is a ratio, not an absolute photon count alone.', ko: 'Center-pixel absorption과 neighbor-pixel absorption을 비교합니다. Crosstalk는 절대 광자 수만이 아니라 비율입니다.' },
          { en: 'Sweep trench depth and width because shallow trenches may miss long red/NIR absorption paths.', ko: '얕은 trench는 긴 red/NIR 흡수 경로를 놓칠 수 있으므로 trench depth와 width를 sweep합니다.' },
          { en: 'Check color dependence: blue may be surface-limited while red/NIR is more sensitive to deep lateral paths.', ko: '색 의존성을 확인합니다. Blue는 표면 제한적이고 red/NIR은 깊은 lateral path에 더 민감할 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'Simple DTI metrics do not include waveguide modes, corner diffraction, roughness, or polarization-dependent trench response.', ko: '간단한 DTI 지표는 waveguide mode, corner diffraction, roughness, polarization-dependent trench response를 포함하지 않습니다.' },
          { en: 'Electrical isolation and optical isolation are related but not identical; carrier diffusion can still cause crosstalk.', ko: '전기적 isolation과 광학 isolation은 관련 있지만 동일하지 않습니다. Carrier diffusion도 crosstalk를 만들 수 있습니다.' },
          { en: 'Final validation should use 3D EM fields and charge-collection modeling for advanced pixels.', ko: '고급 픽셀의 최종 검증에는 3D EM field와 charge-collection modeling이 필요합니다.' },
        ],
      },
    ],
    references: [refs.han2020, refs.yokogawa2017, refs.blockstein2010],
  },
  'fabry-perot': {
    title: { en: 'Single-Layer Interference', ko: '단층 박막 간섭' },
    summary: {
      en: 'Fabry-Perot behavior comes from repeated internal reflections whose phase can add constructively or destructively.',
      ko: '파브리-페로 거동은 박막 내부에서 반복 반사된 빛의 위상이 보강 또는 상쇄 간섭을 일으키면서 발생합니다.',
    },
    intuition: {
      en: 'A single thin transparent film acts as a tiny resonator: light bounces back and forth between its two surfaces, and depending on the thickness, certain wavelengths come out strong (transmission peaks) while others are reflected. This is the same effect that paints rainbow colours on soap films and oily puddles, and it is the basic building block of nearly every thin-film optical filter.',
      ko: '하나의 얇은 투명 박막은 작은 공진기처럼 동작합니다. 빛이 두 표면 사이를 왕복하면서, 두께에 따라 특정 파장은 강하게 통과하고(투과 피크) 다른 파장은 반사됩니다. 비누 막이나 길 위의 기름띠가 무지개색을 띠게 만드는 원리와 같으며, 거의 모든 박막 광학 필터의 기본 구성 요소이기도 합니다.',
    },
    formulas: [
      {
        label: { en: 'Round-trip phase', ko: '왕복 위상' },
        equation: '\\delta = \\frac{4\\pi n d \\cos(\\theta)}{\\lambda}',
        variables: [
          { symbol: '\\delta', description: { en: 'Phase change for one round trip', ko: '1회 왕복 시의 위상 변화 $\\delta$' } },
          { symbol: 'n', description: { en: 'Refractive index of the film', ko: '박막의 굴절률 $n$' } },
          { symbol: 'd', description: { en: 'Physical thickness', ko: '물리적 두께 $d$' } },
        ],
        note: { en: 'A small thickness or angle change can move the film between reflection peaks and valleys.', ko: '두께나 각도의 작은 변화만으로도 반사 피크와 밸리 사이를 이동할 수 있습니다.' },
      },
      {
        label: { en: 'Constructive condition', ko: '보강 조건' },
        equation: '2nd \\cos(\\theta) = m\\lambda',
        variables: [
          { symbol: 'm', description: { en: 'Integer order of interference', ko: '간섭 차수 $m$ (정수)' } },
        ],
        note: { en: 'Integer round-trip phase gives resonant transmission or reflection depending on the stack.', ko: '왕복 위상이 정수배가 되면 스택에 따라 공진 투과 또는 반사가 나타납니다.' },
      },
      {
        label: { en: 'Quarter-wave AR', ko: '1/4파장 AR' },
        equation: 'd = \\frac{\\lambda_0}{4n}',
        variables: [
          { symbol: '\\lambda_0', description: { en: 'Reference design wavelength', ko: '기준 설계 파장 $\\lambda_0$' } },
        ],
        note: { en: 'The common AR seed thickness creates destructive interference for front-surface reflection.', ko: '일반적인 AR 초기 두께는 전면 반사를 상쇄 간섭시키도록 잡습니다.' },
      },
    ],
    concepts: [
      { en: 'A single layer is the simplest case of the full TMM stack.', ko: '단층 박막은 전체 TMM 스택의 가장 단순한 경우입니다.' },
      { en: 'Angle and spectral bandwidth broaden or shift interference features.', ko: '입사각과 분광 대역폭은 간섭 특징을 넓히거나 이동시킵니다.' },
      { en: 'Real color filters and BARL stacks require multiple materials and absorbing films.', ko: '실제 컬러 필터와 BARL 스택은 여러 재료와 흡수성 박막을 포함합니다.' },
    ],
    sections: [
      {
        title: { en: 'Resonance Reading', ko: '공진 해석' },
        items: [
          { en: 'A peak or valley appears when the round-trip phase $\\delta$ lines up reflected waves constructively or destructively.', ko: '왕복 위상 $\\delta$가 반사파를 보강 또는 상쇄하도록 맞을 때 peak 또는 valley가 나타납니다.' },
          { en: 'Increasing thickness $d$ shifts the same interference order toward longer wavelength.', ko: '두께 $d$를 늘리면 동일 간섭 차수가 더 긴 파장 쪽으로 이동합니다.' },
          { en: 'Increasing angle reduces $\\cos\\theta$ and therefore shifts the resonance condition.', ko: '입사각이 커지면 $\\cos\\theta$가 줄어 공진 조건이 이동합니다.' },
        ],
      },
      {
        title: { en: 'Connection To TMM', ko: 'TMM과의 연결' },
        items: [
          { en: 'The single-film equation is the one-layer limit of the characteristic-matrix method.', ko: '단일 박막 식은 characteristic-matrix method의 one-layer limit입니다.' },
          { en: 'Absorbing films require complex $\\tilde{n}=n+ik$, so resonance strength and loss must be interpreted together.', ko: '흡수성 박막은 복소 $\\tilde{n}=n+ik$가 필요하므로 resonance strength와 loss를 함께 해석해야 합니다.' },
          { en: 'Real BARL or color-filter stacks superpose multiple such resonances across many interfaces.', ko: '실제 BARL 또는 color-filter stack은 여러 계면의 이러한 공진을 중첩합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The compact equations omit finite aperture averaging, roughness scattering, and lateral patterning.', ko: 'Compact equation은 finite aperture averaging, roughness scattering, lateral patterning을 생략합니다.' },
          { en: 'A single layer cannot model broadband color filters or multi-material anti-reflection coatings.', ko: '단일층은 broadband color filter 또는 multi-material anti-reflection coating을 모델링할 수 없습니다.' },
          { en: 'For high angles, s/p polarization split should be treated explicitly.', ko: '큰 입사각에서는 s/p 편광 분리를 명시적으로 다뤄야 합니다.' },
        ],
      },
    ],
    references: [refs.macleod, refs.bornWolf, refs.goossens2018],
  },
  'diffraction-psf': {
    title: { en: 'Airy Pattern and Pixel Sampling', ko: '에어리 패턴과 픽셀 샘플링' },
    summary: {
      en: 'A circular aperture forms an Airy diffraction pattern, which spreads energy across multiple pixels as f-number or wavelength increases.',
      ko: '원형 개구는 에어리 회절 패턴을 만들며, f-number 또는 파장이 증가하면 에너지가 여러 픽셀로 퍼집니다.',
    },
    intuition: {
      en: 'Even a perfect lens cannot focus light to a single mathematical point — diffraction smears it into a bright central disk surrounded by faint rings (the Airy pattern). The smaller the aperture or the longer the wavelength, the bigger the smear. When that smear grows larger than one pixel, neighbouring pixels start sharing the same point of light and resolution suffers.',
      ko: '이상적인 렌즈조차 빛을 수학적인 한 점에 모을 수는 없습니다. 회절 때문에 중심부의 밝은 원반과 그 주위의 희미한 고리(에어리 패턴)로 퍼지죠. 개구가 작거나 파장이 길수록 더 크게 퍼집니다. 이 퍼짐이 픽셀 하나보다 커지면 이웃 픽셀들이 같은 점의 빛을 나눠 받게 되어 해상도가 떨어집니다.',
    },
    formulas: [
      {
        label: { en: 'Airy intensity', ko: '에어리 강도' },
        equation: 'I(r) = I_0 \\left[ \\frac{2J_1(\\frac{\\pi r}{\\lambda N})}{\\frac{\\pi r}{\\lambda N}} \\right]^2',
        variables: [
          { symbol: 'r', description: { en: 'Radial distance from center', ko: '중심으로부터의 반경 $r$' } },
          { symbol: 'J_1', description: { en: 'First-order Bessel function', ko: '1차 베셀 함수 $J_1$' } },
          { symbol: 'N', description: { en: 'Lens f-number', ko: '렌즈 조립 f-number $N$' } },
        ],
        note: { en: 'J1 is the first-order Bessel function and N is the f-number.', ko: 'J1은 1차 베셀 함수이고 N은 f-number입니다.' },
      },
      {
        label: { en: 'First dark ring', ko: '첫 암환 반경' },
        equation: 'r_1 = 1.22 \\lambda N',
        variables: [
          { symbol: 'r_1', description: { en: 'Radius of first Airy zero', ko: '첫 번째 에어리 0점 반경 $r_1$' } },
        ],
        note: { en: 'This gives the common diffraction-limited spot radius estimate.', ko: '회절 한계 스폿 반경의 흔한 추정식입니다.' },
      },
      {
        label: { en: 'Encircled energy', ko: '포위 에너지' },
        equation: 'EE(r) = \\int_0^r 2\\pi \\rho I(\\rho) \\,d\\rho',
        variables: [
          { symbol: 'EE(r)', description: { en: 'Energy inside radius $r$', ko: '반경 $r$ 내의 에너지 비율 $EE(r)$' } },
        ],
        note: { en: 'Pixel collection depends on how much PSF energy falls within the active aperture.', ko: '픽셀 수집은 PSF 에너지 중 얼마가 활성 개구 안에 들어오는지에 좌우됩니다.' },
      },
    ],
    concepts: [
      { en: 'Diffraction becomes a first-order limit when pixel pitch approaches the wavelength.', ko: '픽셀 피치가 파장에 가까워지면 회절이 1차 제한 요인이 됩니다.' },
      { en: 'The PSF viewer assumes an ideal circular aperture and ignores aberrations.', ko: '이 PSF 뷰어는 이상적인 원형 개구를 가정하며 수차는 무시합니다.' },
      { en: 'For real camera resolution, combine diffraction, pixel aperture MTF, lens aberration, and demosaicing.', ko: '실제 카메라 해상도는 회절, 픽셀 개구 MTF, 렌즈 수차, 디모자이킹을 함께 봐야 합니다.' },
    ],
    sections: [
      {
        title: { en: 'Sampling Interpretation', ko: '샘플링 해석' },
        items: [
          { en: 'Compare $r_1=1.22\\lambda N$ with pixel pitch; once the central lobe spans multiple pixels, point detail is shared.', ko: '$r_1=1.22\\lambda N$을 pixel pitch와 비교합니다. 중심 lobe가 여러 픽셀에 걸치면 점 디테일이 공유됩니다.' },
          { en: 'Encircled energy is often more useful than peak intensity because a pixel collects finite area, not a point.', ko: '픽셀은 점이 아니라 유한 면적을 수집하므로 peak intensity보다 encircled energy가 더 유용할 때가 많습니다.' },
          { en: 'Longer wavelength and larger f-number both broaden the PSF through the same product $\\lambda N$.', ko: '긴 파장과 큰 f-number는 같은 곱 $\\lambda N$을 통해 PSF를 넓힙니다.' },
        ],
      },
      {
        title: { en: 'Resolution Workflow', ko: '해상도 분석 절차' },
        items: [
          { en: 'Use the PSF to judge point spread, then use MTF to judge sinusoidal contrast transfer.', ko: 'PSF로 point spread를 판단하고, MTF로 sinusoidal contrast transfer를 판단합니다.' },
          { en: 'Compare the Airy diameter with the pixel pitch and with the Nyquist frequency of the sampling grid.', ko: 'Airy diameter를 pixel pitch 및 sampling grid의 Nyquist frequency와 비교합니다.' },
          { en: 'For color sensors, evaluate blue, green, and red separately because $\\lambda$ changes the diffraction blur.', ko: 'Color sensor에서는 $\\lambda$가 diffraction blur를 바꾸므로 blue, green, red를 따로 평가합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The ideal Airy PSF omits lens aberration, defocus, cover-glass effects, and microlens aperture truncation.', ko: '이상적인 Airy PSF는 lens aberration, defocus, cover-glass effect, microlens aperture truncation을 생략합니다.' },
          { en: 'It does not include pixel cross-section geometry, metal shading, or charge diffusion.', ko: 'Pixel cross-section geometry, metal shading, charge diffusion을 포함하지 않습니다.' },
          { en: 'System resolution also depends on demosaic, sharpening, motion blur, and measurement target processing.', ko: '시스템 해상도는 demosaic, sharpening, motion blur, measurement target processing에도 의존합니다.' },
        ],
      },
    ],
    references: [refs.bornWolf, refs.iso12233],
  },
  'mtf-analyzer': {
    title: { en: 'Modulation Transfer Function', ko: '변조 전달 함수' },
    summary: {
      en: 'MTF describes how contrast is transferred as a function of spatial frequency. The analyzer combines pixel aperture and diffraction limits.',
      ko: 'MTF는 공간 주파수에 따라 대비가 얼마나 전달되는지를 나타냅니다. 이 분석기는 픽셀 개구와 회절 한계를 결합합니다.',
    },
    intuition: {
      en: 'MTF asks a simple question: how much contrast survives as image detail gets finer? Big bold patterns transfer almost perfectly, but as you zoom into stripe patterns close to the pixel size, contrast falls away. This analyser combines two limits — finite pixel size acting as a low-pass filter, and lens diffraction blur — to predict where the detail finally vanishes.',
      ko: 'MTF는 간단한 질문에 답합니다. “이미지의 세부 패턴이 얼마나 또렷이 전달되는가?” 크고 굵은 무늬는 거의 그대로 전달되지만, 픽셀 크기에 가까운 줄무늬로 갈수록 대비가 사라집니다. 이 분석기는 두 가지 한계 — 픽셀 크기가 만드는 저역통과 효과와 렌즈 회절 흐림 — 을 결합해 디테일이 어디서 사라지는지를 예측합니다.',
    },
    formulas: [
      {
        label: { en: 'Pixel aperture MTF', ko: '픽셀 개구 MTF' },
        equation: 'MTF_{\\text{px}}(f) = \\left| \\text{sinc}(\\pi f \\cdot p \\cdot FF) \\right|',
        variables: [
          { symbol: 'f', description: { en: 'Spatial frequency', ko: '공간 주파수 $f$' } },
          { symbol: 'p', description: { en: 'Pixel pitch', ko: '픽셀 피치 $p$' } },
          { symbol: 'FF', description: { en: 'Fill factor', ko: '필 팩터 $FF$' } },
        ],
        note: { en: 'A rectangular aperture low-pass filters the sampled image.', ko: '직사각형 개구는 샘플링된 이미지를 저역통과 필터링합니다.' },
      },
      {
        label: { en: 'Diffraction cutoff', ko: '회절 cutoff' },
        equation: 'f_c = \\frac{1}{\\lambda N}',
        variables: [
          { symbol: 'f_c', description: { en: 'Cutoff frequency', ko: '차단 주파수 $f_c$' } },
        ],
        note: { en: 'No ideal incoherent diffraction-limited contrast remains beyond the cutoff.', ko: 'cutoff 이후에는 이상적인 비간섭 회절 제한 대비가 남지 않습니다.' },
      },
      {
        label: { en: 'Nyquist frequency', ko: '나이퀴스트 주파수' },
        equation: 'f_N = \\frac{1}{2p}',
        variables: [
          { symbol: 'f_N', description: { en: 'Nyquist sampling limit', ko: '나이퀴스트 샘플링 한계 $f_N$' } },
        ],
        note: { en: 'Sampling above Nyquist aliases into lower spatial frequencies.', ko: '나이퀴스트를 넘는 샘플링 성분은 낮은 주파수로 aliasing됩니다.' },
      },
    ],
    concepts: [
      { en: 'Pixel pitch controls sampling; f-number and wavelength control diffraction blur.', ko: '픽셀 피치는 샘플링을, f-number와 파장은 회절 blur를 결정합니다.' },
      { en: 'The combined MTF is a product only under simplified linear, shift-invariant assumptions.', ko: '결합 MTF를 곱으로 보는 것은 선형/시불변 근사 아래에서만 단순화된 표현입니다.' },
      { en: 'ISO 12233/SFR measurements are system-level tests, not just pixel physics.', ko: 'ISO 12233/SFR 측정은 픽셀 물리만이 아니라 시스템 레벨 테스트입니다.' },
    ],
    sections: [
      {
        title: { en: 'Frequency Landmarks', ko: '주파수 기준점' },
        items: [
          { en: 'The Nyquist limit $f_N=1/(2p)$ tells where sampled detail begins to alias.', ko: 'Nyquist limit $f_N=1/(2p)$는 샘플링된 디테일이 aliasing되기 시작하는 지점을 알려줍니다.' },
          { en: 'The diffraction cutoff $f_c=1/(\\lambda N)$ tells the optical passband limit for an ideal aperture.', ko: 'Diffraction cutoff $f_c=1/(\\lambda N)$는 이상적인 aperture의 optical passband 한계를 나타냅니다.' },
          { en: 'The practical resolution limit is whichever term suppresses contrast first: pixel aperture, diffraction, lens aberration, or processing.', ko: '실용 해상도 한계는 pixel aperture, diffraction, lens aberration, processing 중 먼저 contrast를 억제하는 항입니다.' },
        ],
      },
      {
        title: { en: 'Measurement Notes', ko: '측정 참고' },
        items: [
          { en: 'Slanted-edge SFR estimates system MTF from an image, including optics, sensor sampling, demosaic, and sharpening.', ko: 'Slanted-edge SFR은 optics, sensor sampling, demosaic, sharpening이 포함된 이미지에서 system MTF를 추정합니다.' },
          { en: 'MTF50 is a convenient scalar, but full-curve shape matters for aliasing and texture rendering.', ko: 'MTF50은 편리한 스칼라이지만 aliasing과 texture rendering에는 전체 곡선 형태가 중요합니다.' },
          { en: 'Compare luminance and chroma MTF separately for Bayer sensors because demosaic changes each channel differently.', ko: 'Bayer sensor에서는 demosaic가 채널별로 다르게 작용하므로 luminance와 chroma MTF를 분리 비교합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The compact model assumes linear shift-invariant behavior; real image pipelines may be nonlinear and adaptive.', ko: 'Compact model은 linear shift-invariant 동작을 가정하지만 실제 image pipeline은 비선형/적응형일 수 있습니다.' },
          { en: 'It does not include motion blur, focus error, lens field curvature, or rolling-shutter effects.', ko: 'Motion blur, focus error, lens field curvature, rolling-shutter effect는 포함하지 않습니다.' },
          { en: 'Pixel optical crosstalk and charge diffusion can reduce high-frequency contrast beyond aperture MTF.', ko: 'Pixel optical crosstalk와 charge diffusion은 aperture MTF 이상으로 고주파 contrast를 낮출 수 있습니다.' },
        ],
      },
    ],
    references: [refs.iso12233, refs.bornWolf],
  },
  'pixel-scaling': {
    title: { en: 'Pixel Pitch Scaling Laws', ko: '픽셀 피치 스케일링 법칙' },
    summary: {
      en: 'Shrinking pixels changes photon count, full well capacity, crosstalk, diffraction sensitivity, and read-noise requirements together.',
      ko: '픽셀 축소는 광자 수, 풀웰 용량, 크로스토크, 회절 민감도, 읽기 노이즈 요구조건을 동시에 바꿉니다.',
    },
    intuition: {
      en: 'Making a pixel smaller is not just shrinking — it changes everything at once. Smaller pixels collect fewer photons (lower SNR), hold less charge (lower full-well), are hit harder by diffraction blur, and demand much stronger read-noise reduction. This tool lays out those trade-offs side by side so you can see what scales how.',
      ko: '픽셀을 작게 만드는 것은 단순한 축소가 아닙니다. 모든 특성이 동시에 바뀝니다. 작은 픽셀은 광자를 적게 모으고(SNR 저하), 전하 용량이 작아지고(풀웰 저하), 회절 흐림의 영향을 더 받으며, 읽기 노이즈도 훨씬 더 줄여야 합니다. 이 도구는 그 트레이드오프를 한눈에 비교해 어떤 항목이 어떻게 스케일링되는지 보여줍니다.',
    },
    formulas: [
      {
        label: { en: 'Area scaling', ko: '면적 스케일링' },
        equation: 'A_{\\text{px}} = p^2',
        variables: [
          { symbol: 'A_{\\text{px}}', description: { en: 'Pixel area', ko: '픽셀 면적 $A_{\\text{px}}$' } },
        ],
        note: { en: 'Photon capture and approximate capacitance both scale with pixel area.', ko: '광자 수집과 대략적인 커패시턴스는 모두 픽셀 면적에 비례합니다.' },
      },
      {
        label: { en: 'Shot-noise limit', ko: '샷 노이즈 한계' },
        equation: 'SNR_{\\text{shot}} = \\sqrt{N_e}',
        variables: [
          { symbol: 'N_e', description: { en: 'Number of electrons', ko: '전자 수 $N_e$' } },
        ],
        note: { en: 'If the collected electron count falls by 4x, shot-limited SNR falls by 2x.', ko: '수집 전자 수가 4배 줄면 샷 노이즈 한계 SNR은 2배 줄어듭니다.' },
      },
      {
        label: { en: 'Diffraction ratio', ko: '회절 비율' },
        equation: '\\frac{D_{\\text{Airy}}}{p} \\approx \\frac{2.44 \\lambda N}{p}',
        variables: [
          { symbol: 'D_{\\text{Airy}}', description: { en: 'Airy disk diameter', ko: '에어리 원반 직경 $D_{\\text{Airy}}$' } },
        ],
        note: { en: 'This ratio rises as pixels shrink, increasing optical sharing between pixels.', ko: '픽셀이 작아질수록 이 비율이 커져 픽셀 간 광학 공유가 증가합니다.' },
      },
    ],
    concepts: [
      { en: 'Modern sub-micron pixels rely on BSI, DTI, microlens shift, and computational binning.', ko: '현대 서브마이크론 픽셀은 BSI, DTI, 마이크로렌즈 시프트, 계산적 binning에 의존합니다.' },
      { en: 'Pitch scaling is not a single-variable problem because optics and electronics scale differently.', ko: '광학과 전자 회로는 서로 다르게 스케일링되므로 픽셀 축소는 단일 변수 문제가 아닙니다.' },
      { en: 'Commercial trend lines should be treated as context, not as a process design rule.', ko: '상용 센서 추세선은 공정 설계 규칙이 아니라 맥락 정보로 봐야 합니다.' },
    ],
    sections: [
      {
        title: { en: 'What Scales With Pitch', ko: 'Pitch와 함께 스케일되는 항목' },
        items: [
          { en: 'Photon count falls roughly with $p^2$ for the same illuminance and exposure.', ko: '동일 조도와 노출에서 photon count는 대략 $p^2$에 비례해 감소합니다.' },
          { en: 'Diffraction pressure rises through $D_{\\text{Airy}}/p$, so the same lens f-number becomes harder for smaller pixels.', ko: '$D_{\\text{Airy}}/p$가 커지므로 같은 lens f-number도 작은 픽셀에서 더 부담스러워집니다.' },
          { en: 'Read-noise requirements become stricter because fewer electrons are available before SNR collapses.', ko: 'SNR이 무너지기 전 사용할 수 있는 전자 수가 줄어들기 때문에 read-noise 요구조건이 더 엄격해집니다.' },
        ],
      },
      {
        title: { en: 'Mitigation Stack', ko: '완화 기술 stack' },
        items: [
          { en: 'BSI improves optical access; DTI limits crosstalk; microlens shift recovers CRA response; binning recovers photon statistics.', ko: 'BSI는 optical access를 개선하고, DTI는 crosstalk를 제한하며, microlens shift는 CRA response를 회복하고, binning은 photon statistics를 회복합니다.' },
          { en: 'None of these fixes is free: each adds process complexity, optical side effects, or signal-processing assumptions.', ko: '이 보정들은 모두 공짜가 아닙니다. 각각 공정 복잡도, 광학 부작용, 신호처리 가정을 추가합니다.' },
          { en: 'Scaling should be evaluated at the camera-system level, not only as a pixel-layout shrink.', ko: '스케일링은 pixel-layout shrink만이 아니라 camera-system level에서 평가해야 합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'Simple scaling laws ignore detailed capacitance, source-follower noise, CFA geometry, and wafer process differences.', ko: '단순 scaling law는 상세 capacitance, source-follower noise, CFA geometry, wafer process difference를 무시합니다.' },
          { en: 'They do not predict HDR behavior, dual conversion gain, or computational multi-frame recovery.', ko: 'HDR 동작, dual conversion gain, computational multi-frame recovery를 예측하지 않습니다.' },
          { en: 'Treat trend curves as first-order pressure indicators, not final performance forecasts.', ko: 'Trend curve는 최종 성능 예측이 아니라 1차 pressure indicator로 봐야 합니다.' },
        ],
      },
    ],
    references: [refs.catrysse2002, refs.agranov2003, refs.blockstein2010],
  },
  'color-accuracy': {
    title: { en: 'Color Correction and Delta E', ko: '색 보정과 Delta E' },
    summary: {
      en: 'The analyzer maps simulated camera RGB responses to reference colorimetry with a color correction matrix, then measures perceptual error.',
      ko: '이 분석기는 시뮬레이션된 카메라 RGB 응답을 색 보정 행렬로 기준 색도계 값에 매핑하고 지각 색차를 측정합니다.',
    },
    intuition: {
      en: 'A camera does not see colour exactly like a human eye, so its raw RGB values must be translated to match what we would expect — that translation is the colour correction matrix (CCM). Delta E then measures how close the final colours are to the reference, scaled to match human perception (a Delta E of 1 is roughly the smallest difference an eye can spot). Better filters and more light make the translation easier and more accurate.',
      ko: '카메라는 사람의 눈과 똑같이 색을 보지는 않습니다. 그래서 raw RGB 값을 우리가 인식하는 색에 맞게 “번역”해야 하는데, 이것이 색 보정 행렬(CCM)입니다. Delta E는 번역된 최종 색이 기준 색과 얼마나 가까운지를 사람의 지각 감각에 맞춰 측정합니다(Delta E ≈ 1은 사람이 겨우 구분할 수 있는 최소 색차). 좋은 필터와 충분한 광량은 이 번역을 더 쉽고 정확하게 만듭니다.',
    },
    formulas: [
      {
        label: { en: 'Least-squares CCM', ko: '최소제곱 CCM' },
        equation: 'M = \\text{argmin}_M \\left\\| \\mathbf{R}_{\\text{cam}} M - \\mathbf{X}_{\\text{ref}} \\right\\|_2^2',
        variables: [
          { symbol: 'M', description: { en: '3x3 CCM matrix', ko: '3x3 CCM 행렬 $M$' } },
          { symbol: '\\mathbf{R}_{\\text{cam}}', description: { en: 'Camera response matrix', ko: '카메라 응답 행렬 $\\mathbf{R}_{\\text{cam}}$' } },
        ],
        note: { en: 'The CCM is fitted to map camera sensor responses to reference XYZ values.', ko: 'CCM은 카메라 센서 응답을 기준 XYZ 값에 맞추도록 피팅됩니다.' },
      },
      {
        label: { en: 'CIELAB error', ko: 'CIELAB 색차' },
        equation: '\\Delta E_{ab}^* = \\sqrt{(\\Delta L^*)^2 + (\\Delta a^*)^2 + (\\Delta b^*)^2}',
        variables: [
          { symbol: '\\Delta E_{ab}^*', description: { en: 'Total color difference', ko: '전체 색차 $\\Delta E_{ab}^*$' } },
        ],
        note: { en: 'Delta E summarizes color mismatch after conversion to a perceptual color space.', ko: 'Delta E는 지각 색공간에서의 색 불일치를 요약합니다.' },
      },
      {
        label: { en: 'Noise-perturbed RGB', ko: '노이즈 포함 RGB' },
        equation: 'R_{\\text{meas}} = R_{\\text{ideal}} + n_{\\text{shot}} + n_{\\text{read}}',
        variables: [
          { symbol: 'n', description: { en: 'Noise terms', ko: '노이즈 항 $n$' } },
        ],
        note: { en: 'Color accuracy degrades when low-light noise perturbs channel ratios.', ko: '저조도 노이즈가 채널 비율을 흔들면 색 정확도가 나빠집니다.' },
      },
    ],
    concepts: [
      { en: 'Good color filters are not only narrow; they must produce well-conditioned RGB bases.', ko: '좋은 컬러 필터는 단순히 좁은 필터가 아니라 조건이 좋은 RGB 기저를 만들어야 합니다.' },
      { en: 'A CCM can correct linear color mixing but cannot recover missing spectral information.', ko: 'CCM은 선형 색 혼합은 보정할 수 있지만 사라진 분광 정보는 복원할 수 없습니다.' },
      { en: 'Illuminant choice changes both reference colors and camera channel balance.', ko: '광원 선택은 기준 색과 카메라 채널 밸런스를 모두 바꿉니다.' },
    ],
    sections: [
      {
        title: { en: 'CCM Conditioning', ko: 'CCM 조건성' },
        items: [
          { en: 'If RGB channel spectra are too similar, $\\mathbf{R}_{\\text{cam}}$ becomes ill-conditioned and noise is amplified by the CCM.', ko: 'RGB 채널 spectrum이 너무 비슷하면 $\\mathbf{R}_{\\text{cam}}$이 ill-conditioned가 되어 CCM이 노이즈를 증폭합니다.' },
          { en: 'A low mean $\\Delta E$ can hide large outliers, so inspect worst patch and hue-specific errors.', ko: '낮은 평균 $\\Delta E$가 큰 outlier를 숨길 수 있으므로 worst patch와 hue별 error를 봐야 합니다.' },
          { en: 'Better spectral separation often trades off with lower throughput and higher low-light color noise.', ko: '더 좋은 spectral separation은 종종 낮은 throughput 및 저조도 color noise 증가와 tradeoff입니다.' },
        ],
      },
      {
        title: { en: 'Measurement Workflow', ko: '측정 절차' },
        items: [
          { en: 'Capture color targets under known illuminants, subtract black level, linearize response, then fit the CCM.', ko: '알려진 illuminant에서 color target을 촬영하고 black level 제거, response linearization 이후 CCM을 fitting합니다.' },
          { en: 'Evaluate under multiple illuminants because a CCM fitted to one light source can fail under another.', ko: '한 광원에 맞춘 CCM은 다른 광원에서 실패할 수 있으므로 여러 illuminant에서 평가합니다.' },
          { en: 'Separate spectral design errors from pipeline errors by testing raw linear data before tone mapping.', ko: 'Tone mapping 전 raw linear data를 테스트해 spectral design error와 pipeline error를 분리합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The model does not include demosaic, white-balance adaptation, local tone mapping, or color appearance modeling.', ko: 'Demosaic, white-balance adaptation, local tone mapping, color appearance modeling은 포함하지 않습니다.' },
          { en: 'It assumes a linear CCM; real cameras often use nonlinear hue/saturation transforms after the matrix.', ko: 'Linear CCM을 가정하지만 실제 카메라는 matrix 이후 비선형 hue/saturation transform을 자주 사용합니다.' },
          { en: 'Sensor noise, flare, lens shading, and IR leakage can all masquerade as color-filter design problems.', ko: 'Sensor noise, flare, lens shading, IR leakage가 모두 color-filter design 문제처럼 보일 수 있습니다.' },
        ],
      },
    ],
    references: [refs.nistCie, refs.mccamy1992],
  },
  'dark-current': {
    title: { en: 'Thermal Generation and Dark Noise', ko: '열 생성과 암전류 노이즈' },
    summary: {
      en: 'Dark current is thermally generated charge accumulated during exposure. It rises rapidly with temperature and contributes shot noise.',
      ko: '암전류는 노출 중 열적으로 생성되어 축적되는 전하입니다. 온도에 따라 빠르게 증가하며 샷 노이즈를 더합니다.',
    },
    intuition: {
      en: 'Even in total darkness, silicon pixels slowly accumulate electrons just from heat — like a faint hiss on a turned-off radio. The hotter the sensor and the longer the exposure, the more of these dark electrons pile up, adding their own random noise on top of any real signal. This is especially painful for long-exposure or astrophotography use cases.',
      ko: '완전히 어두운 곳에서도 실리콘 픽셀에는 열에 의해 전자가 천천히 쌓입니다. 꺼놓은 라디오에서 들리는 희미한 잡음 같은 거죠. 센서가 뜨겁고 노출 시간이 길수록 이 “암전자”가 더 많이 쌓이고, 실제 신호 위에 자체적인 무작위 노이즈를 더합니다. 장노출이나 천체 촬영에서 특히 큰 문제가 됩니다.',
    },
    formulas: [
      {
        label: { en: 'Arrhenius trend', ko: '아레니우스 경향' },
        equation: 'I_{\\text{dark}}(T) \\propto T^3 e^{-\\frac{E_a}{kT}}',
        variables: [
          { symbol: 'I_{\\text{dark}}', description: { en: 'Dark current generation rate', ko: '암전류 생성률 $I_{\\text{dark}}$' } },
          { symbol: 'E_a', description: { en: 'Activation energy', ko: '활성화 에너지 $E_a$' } },
          { symbol: 'k', description: { en: 'Boltzmann constant', ko: '볼츠만 상수 $k$' } },
        ],
        note: { en: 'Activation energy controls how steeply dark current rises with temperature.', ko: '활성화 에너지는 온도 상승에 따른 암전류 증가 기울기를 결정합니다.' },
      },
      {
        label: { en: 'Dark signal', ko: '암신호' },
        equation: 'D = I_{\\text{dark}} \\cdot t_{\\text{exp}}',
        variables: [
          { symbol: 'D', description: { en: 'Accumulated dark charge', ko: '축적된 암전하 $D$' } },
          { symbol: 't_{\\text{exp}}', description: { en: 'Integration time', ko: '노출 시간 $t_{\\text{exp}}$' } },
        ],
        note: { en: 'Longer exposure accumulates more dark electrons.', ko: '노출 시간이 길수록 더 많은 암전자가 축적됩니다.' },
      },
      {
        label: { en: 'Dark shot noise', ko: '암전류 샷 노이즈' },
        equation: '\\sigma_{\\text{dark}} = \\sqrt{D}',
        variables: [
          { symbol: '\\sigma_{\\text{dark}}', description: { en: 'RMS noise due to dark current', ko: '암전류에 의한 노이즈 표준편차 $\\sigma_{\\text{dark}}$' } },
        ],
        note: { en: 'Thermal generation is a Poisson process in this simplified model.', ko: '이 단순 모델에서 열 생성은 포아송 과정으로 봅니다.' },
      },
    ],
    concepts: [
      { en: 'Cooling, pinned photodiodes, and better isolation reduce dark-current impact.', ko: '냉각, pinned photodiode, 개선된 격리는 암전류 영향을 줄입니다.' },
      { en: 'Hot pixels are spatial outliers and are not captured by a single average current.', ko: '핫픽셀은 공간적 이상치이므로 단일 평균 암전류만으로는 포착되지 않습니다.' },
      { en: 'Dark noise matters most in long exposure and low-light operation.', ko: '암전류 노이즈는 장노출 및 저조도 동작에서 가장 중요합니다.' },
    ],
    sections: [
      {
        title: { en: 'Temperature Scaling', ko: '온도 스케일링' },
        items: [
          { en: 'The Arrhenius term means small temperature changes can dominate long-exposure noise.', ko: 'Arrhenius 항 때문에 작은 온도 변화도 장노출 노이즈를 지배할 수 있습니다.' },
          { en: 'Activation energy $E_a$ depends on the dominant generation mechanism, so one slope may not fit all temperatures.', ko: '활성화 에너지 $E_a$는 지배적인 생성 메커니즘에 따라 달라지므로 하나의 기울기가 모든 온도에 맞지 않을 수 있습니다.' },
          { en: 'Cooling reduces both mean dark signal $D$ and dark shot noise $\\sqrt{D}$.', ko: '냉각은 평균 dark signal $D$와 dark shot noise $\\sqrt{D}$를 함께 줄입니다.' },
        ],
      },
      {
        title: { en: 'Measurement Notes', ko: '측정 참고' },
        items: [
          { en: 'Measure dark frames at multiple exposure times to separate fixed offset from time-dependent dark current.', ko: '여러 노출 시간의 dark frame을 측정해 fixed offset과 시간 의존 dark current를 분리합니다.' },
          { en: 'Use temperature-stabilized acquisition because dark current can drift during a measurement run.', ko: '측정 중 dark current가 drift할 수 있으므로 온도 안정화된 acquisition을 사용합니다.' },
          { en: 'Track hot pixels separately from median dark current; they drive image defects and calibration tables.', ko: 'Hot pixel은 median dark current와 별도로 추적합니다. 이미지 결함과 보정 테이블을 지배할 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The compact model does not distinguish diffusion current, depletion current, surface generation, and trap-assisted tunneling.', ko: 'Compact model은 diffusion current, depletion current, surface generation, trap-assisted tunneling을 구분하지 않습니다.' },
          { en: 'It does not model random telegraph signal, hot-pixel distributions, or radiation/aging effects.', ko: 'Random telegraph signal, hot-pixel distribution, radiation/aging effect를 모델링하지 않습니다.' },
          { en: 'Dark-frame correction residuals depend on temperature matching and calibration cadence.', ko: 'Dark-frame correction 잔차는 temperature matching과 calibration cadence에 의존합니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
  'photon-transfer-curve': {
    title: { en: 'Photon Transfer Curve', ko: '광자 전달 곡선' },
    summary: {
      en: 'PTC uses the relationship between mean signal and variance to estimate conversion gain, read noise, full well, and PRNU.',
      ko: 'PTC는 평균 신호와 분산의 관계를 이용해 conversion gain, 읽기 노이즈, 풀웰, PRNU를 추정합니다.',
    },
    intuition: {
      en: 'If you measure how noisy a pixel is at many different brightness levels and plot it on log-log axes, three distinct regions appear: a flat read-noise floor (in the dark), a 1/2-slope photon shot-noise region (mid tones), and an upturning PRNU region (bright). This shape is so consistent across silicon pixels that engineers use it as a fingerprint to reverse-engineer the sensor gain, read noise, and full-well from real measurements.',
      ko: '다양한 밝기에서 픽셀의 노이즈를 측정해 로그-로그로 그리면, 세 구간이 뚜렷이 나타나는 곡선이 됩니다. 어두운 영역의 평평한 읽기 노이즈 바닥, 중간 영역의 기울기 1/2 광자 샷 노이즈, 밝은 영역의 위로 휘는 PRNU. 이 형태가 실리콘 픽셀에서 매우 일관되기 때문에 엔지니어들은 실제 측정값에서 센서의 gain, 읽기 노이즈, 풀웰을 지문처럼 역산할 수 있습니다.',
    },
    formulas: [
      {
        label: { en: 'Shot-noise region', ko: '샷 노이즈 영역' },
        equation: '\\sigma_{\\text{DN}}^2 \\approx \\frac{\\mu_{\\text{DN}}}{K}',
        variables: [
          { symbol: '\\sigma_{\\text{DN}}^2', description: { en: 'Variance in digital numbers', ko: '디지털 값의 분산' } },
          { symbol: 'K', description: { en: 'System gain ($e^-/\\text{DN}$)', ko: '시스템 이득 $K$' } },
        ],
        note: { en: 'The slope in the shot-noise region gives conversion gain K in e-/DN.', ko: '샷 노이즈 영역의 기울기에서 e-/DN 단위의 conversion gain K를 얻습니다.' },
      },
      {
        label: { en: 'Read noise', ko: '읽기 노이즈' },
        equation: '\\sigma_{\\text{read}, e^-} = K \\cdot \\sigma_{\\text{read}, \\text{DN}}',
        variables: [
          { symbol: '\\sigma_{\\text{read}}', description: { en: 'Readout noise', ko: '읽기 노이즈' } },
        ],
        note: { en: 'Dark-frame variance near zero signal estimates read noise.', ko: '0 신호 근처 dark-frame 분산으로 읽기 노이즈를 추정합니다.' },
      },
      {
        label: { en: 'PRNU region', ko: 'PRNU 영역' },
        equation: '\\sigma_{\\text{total}}^2 \\approx \\sigma_{\\text{shot}}^2 + (PRNU \\cdot S)^2',
        variables: [
          { symbol: 'PRNU', description: { en: 'PRNU coefficient', ko: 'PRNU 계수' } },
        ],
        note: { en: 'At high signal, multiplicative non-uniformity bends the curve upward.', ko: '고신호에서는 곱셈성 비균일성이 곡선을 위쪽으로 휘게 합니다.' },
      },
    ],
    concepts: [
      { en: 'The log-log PTC separates read-noise, shot-noise, and saturation regions.', ko: '로그-로그 PTC는 읽기 노이즈, 샷 노이즈, 포화 영역을 구분합니다.' },
      { en: 'Linearity and flat-field quality affect gain extraction.', ko: '선형성과 flat-field 품질은 gain 추정에 영향을 줍니다.' },
      { en: 'PTC is a measurement method; the simulator shows idealized trends.', ko: 'PTC는 측정 방법이며, 시뮬레이터는 이상화된 경향을 보여줍니다.' },
    ],
    sections: [
      {
        title: { en: 'Curve Regions', ko: '곡선 영역' },
        items: [
          { en: 'At low signal, variance flattens near the read-noise floor.', ko: '저신호에서는 분산이 read-noise floor 근처에서 평평해집니다.' },
          { en: 'In the shot-noise region, variance grows linearly with mean signal, enabling conversion-gain extraction.', ko: 'Shot-noise 영역에서는 분산이 평균 신호와 선형으로 증가하므로 conversion gain을 추출할 수 있습니다.' },
          { en: 'Near saturation, clipping and nonlinearity break the simple variance relationship.', ko: '포화 근처에서는 clipping과 비선형성이 단순한 분산 관계를 깨뜨립니다.' },
        ],
      },
      {
        title: { en: 'Acquisition Checklist', ko: '취득 체크리스트' },
        items: [
          { en: 'Use pairs of flat frames at each exposure so temporal noise can be separated from spatial non-uniformity.', ko: '각 노출에서 flat frame pair를 사용해 temporal noise와 spatial non-uniformity를 분리합니다.' },
          { en: 'Subtract black level and verify linear exposure spacing before fitting gain.', ko: 'Gain fitting 전에 black level을 빼고 노출 간격의 선형성을 확인합니다.' },
          { en: 'Avoid saturated and strongly non-linear points when estimating the shot-noise slope.', ko: 'Shot-noise slope를 추정할 때 saturated 및 강한 비선형 point는 제외합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The idealized PTC does not include ADC quantization structure, row/column noise, or temporal drift.', ko: '이상화된 PTC는 ADC quantization structure, row/column noise, temporal drift를 포함하지 않습니다.' },
          { en: 'PRNU and DSNU extraction depends on flat-field uniformity and dark-frame stability.', ko: 'PRNU/DSNU 추출은 flat-field uniformity와 dark-frame stability에 의존합니다.' },
          { en: 'Dual conversion gain or HDR sensors require segmented PTC interpretation.', ko: 'Dual conversion gain 또는 HDR sensor는 구간별 PTC 해석이 필요합니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
  'dynamic-range': {
    title: { en: 'Dynamic Range and Saturation', ko: '다이나믹 레인지와 포화' },
    summary: {
      en: 'Dynamic range compares the largest usable signal to the smallest distinguishable signal, usually limited by full well and noise floor.',
      ko: '다이나믹 레인지는 최대 사용 가능 신호와 구분 가능한 최소 신호의 비이며, 보통 풀웰과 노이즈 플로어가 제한합니다.',
    },
    intuition: {
      en: 'Dynamic range is the gap between the brightest highlight a pixel can record before it saturates and the faintest detail it can pull out of the noise. Wide DR means you can see both bright clouds and dark shadows in the same shot. Lifting the ceiling (larger full-well) helps highlights; lowering the floor (less read noise, less dark current) helps shadows.',
      ko: '다이나믹 레인지는 픽셀이 포화되기 직전의 가장 밝은 신호와 노이즈에서 겨우 구분되는 가장 어두운 신호 사이의 폭입니다. 넓을수록 한 장면 안에서 밝은 구름과 어두운 그림자를 모두 살릴 수 있죠. 천장(풀웰)을 높이면 하이라이트가, 바닥(읽기 노이즈·암전류)을 낮추면 섀도가 좋아집니다.',
    },
    formulas: [
      {
        label: { en: 'Single-exposure DR', ko: '단일 노출 DR' },
        equation: 'DR_{\\text{dB}} = 20 \\log_{10} \\left( \\frac{FWC}{\\sigma_{\\text{floor}}} \\right)',
        variables: [
          { symbol: 'DR', description: { en: 'Dynamic Range', ko: '다이나믹 레인지 $DR$' } },
          { symbol: 'FWC', description: { en: 'Full Well Capacity', ko: '풀웰 용량 $FWC$' } },
        ],
        note: { en: 'The noise floor often includes read noise plus dark noise.', ko: '노이즈 플로어에는 보통 읽기 노이즈와 암전류 노이즈가 포함됩니다.' },
      },
      {
        label: { en: 'Saturation', ko: '포화' },
        equation: 'S_{\\text{clip}} = \\min(S, FWC)',
        variables: [
          { symbol: 'S', description: { en: 'Input photoelectron signal', ko: '입사 광전자 신호 $S$' } },
        ],
        note: { en: 'No linear signal information remains above full well.', ko: '풀웰을 넘으면 선형 신호 정보가 남지 않습니다.' },
      },
      {
        label: { en: 'HDR exposure span', ko: 'HDR 노출 범위' },
        equation: 'DR_{\\text{HDR}} \\approx 20 \\log_{10} \\left( \\frac{S_{\\text{max, long}}}{S_{\\text{min, short}}} \\right)',
        variables: [
          { symbol: 'S_{\\text{max, long}}', description: { en: 'Max signal in long frame', ko: '장노출 프레임의 최대 신호' } },
          { symbol: 'S_{\\text{min, short}}', description: { en: 'Min signal in short frame', ko: '단노출 프레임의 최소 신호' } },
        ],
        note: { en: 'Multi-exposure HDR extends range but introduces motion and merge constraints.', ko: '다중 노출 HDR은 범위를 넓히지만 motion 및 병합 제약을 만듭니다.' },
      },
    ],
    concepts: [
      { en: 'Increasing FWC helps highlights; reducing read noise helps shadows.', ko: 'FWC 증가는 하이라이트에, 읽기 노이즈 감소는 섀도에 유리합니다.' },
      { en: 'Dark current raises the floor for long exposures and high temperatures.', ko: '암전류는 장노출과 고온에서 바닥 노이즈를 올립니다.' },
      { en: 'Published DR values depend on measurement definitions and SNR thresholds.', ko: '공개 DR 값은 측정 정의와 SNR 임계값에 따라 달라집니다.' },
    ],
    sections: [
      {
        title: { en: 'Floor And Ceiling', ko: '바닥과 천장' },
        items: [
          { en: 'The ceiling is set by full well, ADC range, or nonlinearity, whichever occurs first.', ko: '천장은 full well, ADC range, nonlinearity 중 먼저 도달하는 항목이 결정합니다.' },
          { en: 'The floor is set by read noise, dark noise, quantization noise, and the chosen SNR threshold.', ko: '바닥은 read noise, dark noise, quantization noise, 선택한 SNR threshold가 결정합니다.' },
          { en: 'A DR number without a threshold definition is not directly comparable across cameras.', ko: 'Threshold 정의가 없는 DR 숫자는 카메라 간 직접 비교가 어렵습니다.' },
        ],
      },
      {
        title: { en: 'HDR Caveats', ko: 'HDR 주의점' },
        items: [
          { en: 'Multi-exposure HDR expands range but adds motion artifacts and merge nonlinearity.', ko: 'Multi-exposure HDR은 range를 넓히지만 motion artifact와 merge nonlinearity를 추가합니다.' },
          { en: 'Dual conversion gain changes the effective read-noise/full-well tradeoff across the response curve.', ko: 'Dual conversion gain은 response curve 내 read-noise/full-well tradeoff를 바꿉니다.' },
          { en: 'Highlight recovery is limited by what was not clipped in at least one exposure or gain state.', ko: 'Highlight recovery는 최소 하나의 exposure 또는 gain state에서 clipping되지 않은 정보로 제한됩니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'This compact DR model does not include tone mapping, local contrast, flare, or black-sun artifacts.', ko: '이 compact DR 모델은 tone mapping, local contrast, flare, black-sun artifact를 포함하지 않습니다.' },
          { en: 'It treats saturation as a hard clip, while real pixels can show blooming, compression, and color-channel clipping.', ko: '포화를 hard clip으로 취급하지만 실제 픽셀은 blooming, compression, color-channel clipping을 보일 수 있습니다.' },
          { en: 'Scene dynamic range after ISP processing can differ from sensor-domain DR.', ko: 'ISP 처리 후 scene dynamic range는 sensor-domain DR과 다를 수 있습니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
  'emva1288': {
    title: { en: 'EMVA 1288 Characterization Metrics', ko: 'EMVA 1288 특성화 지표' },
    summary: {
      en: 'The dashboard groups sensor metrics using the EMVA 1288 measurement vocabulary: sensitivity, noise, saturation, dynamic range, and SNR.',
      ko: '이 대시보드는 EMVA 1288 측정 용어에 따라 감도, 노이즈, 포화, 다이나믹 레인지, SNR 지표를 묶어 보여줍니다.',
    },
    intuition: {
      en: 'EMVA 1288 is the industry standard sensor report card — a fixed set of measurements every manufacturer can run, so cameras from different vendors can be compared apples-to-apples. This dashboard groups those metrics (sensitivity, noise, full-well, dynamic range, SNR) in one place using EMVA vocabulary, so you can read every sensor with the same yardstick.',
      ko: 'EMVA 1288은 업계 표준 “센서 성적표”입니다. 모든 제조사가 동일한 방식으로 시험하기 때문에 서로 다른 회사의 카메라끼리 사과 대 사과로 비교할 수 있죠. 이 대시보드는 EMVA 용어에 맞춰 감도, 노이즈, 풀웰, 다이나믹 레인지, SNR을 한 화면에 묶어 보여주므로, 모든 센서를 같은 잣대로 읽을 수 있습니다.',
    },
    formulas: [
      {
        label: { en: 'Absolute sensitivity threshold', ko: '절대 감도 임계값' },
        equation: '\\mu_{p, \\text{min}} \\quad (\\text{at } SNR=1)',
        variables: [
          { symbol: '\\mu_{p, \\text{min}}', description: { en: 'Minimum detectable photon count', ko: '최소 검출 가능 광자 수' } },
        ],
        note: { en: 'This estimates the photon count needed for signal to equal noise.', ko: '신호가 노이즈와 같아지는 데 필요한 광자 수를 추정합니다.' },
      },
      {
        label: { en: 'Saturation capacity', ko: '포화 용량' },
        equation: 'S_{\\text{sat}} \\approx FWC',
        variables: [
          { symbol: 'S_{\\text{sat}}', description: { en: 'Saturation signal', ko: '포화 신호' } },
        ],
        note: { en: 'Full well is the charge capacity before clipping or nonlinearity dominates.', ko: '풀웰은 clipping 또는 비선형성이 지배하기 전 전하 용량입니다.' },
      },
      {
        label: { en: 'SNR', ko: 'SNR' },
        equation: 'SNR = \\frac{\\mu_e}{\\sqrt{\\sigma_d^2 + \\sigma_q^2 + \\mu_e}}',
        variables: [
          { symbol: '\\mu_e', description: { en: 'Mean photoelectron count', ko: '평균 광전자 수 $\\mu_e$' } },
          { symbol: '\\sigma_d', description: { en: 'Dark noise std dev', ko: '암전류 노이즈 표준편차 $\\sigma_d$' } },
        ],
        note: { en: 'The simplified dashboard includes dark/read-like noise and shot noise terms.', ko: '단순화된 대시보드는 dark/read 계열 노이즈와 샷 노이즈 항을 포함합니다.' },
      },
    ],
    concepts: [
      { en: 'EMVA 1288 is a measurement standard, not a device physics simulator.', ko: 'EMVA 1288은 측정 표준이지 소자 물리 시뮬레이터가 아닙니다.' },
      { en: 'Use it to compare camera-level metrics under consistent definitions.', ko: '일관된 정의 아래 카메라 수준 지표를 비교하는 데 사용하세요.' },
      { en: 'Real EMVA testing requires calibrated illumination and image acquisition procedures.', ko: '실제 EMVA 시험에는 보정된 조명과 영상 취득 절차가 필요합니다.' },
    ],
    sections: [
      {
        title: { en: 'Metric Families', ko: '지표군' },
        items: [
          { en: 'Sensitivity metrics describe how many photons are needed for a usable signal.', ko: 'Sensitivity 지표는 사용 가능한 신호를 얻는 데 필요한 photon 수를 설명합니다.' },
          { en: 'Noise metrics separate temporal noise, dark noise, and fixed-pattern contributions.', ko: 'Noise 지표는 temporal noise, dark noise, fixed-pattern contribution을 분리합니다.' },
          { en: 'Saturation and dynamic range metrics connect full-well capacity to the noise floor.', ko: 'Saturation 및 dynamic range 지표는 full-well capacity를 noise floor와 연결합니다.' },
        ],
      },
      {
        title: { en: 'Testing Discipline', ko: '시험 절차 discipline' },
        items: [
          { en: 'Calibrated irradiance, exposure control, black-level handling, and linear raw data are required for meaningful results.', ko: '의미 있는 결과에는 calibrated irradiance, exposure control, black-level handling, linear raw data가 필요합니다.' },
          { en: 'Camera settings such as gain, bit depth, and image processing must be reported with the measurement.', ko: 'Gain, bit depth, image processing 같은 camera setting을 측정값과 함께 보고해야 합니다.' },
          { en: 'Repeatability matters: temperature and illumination stability can dominate small differences.', ko: '반복성이 중요합니다. 온도와 조명 안정성이 작은 차이를 지배할 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The dashboard summarizes EMVA vocabulary; it does not emulate a full laboratory acquisition pipeline.', ko: '대시보드는 EMVA 용어를 요약하지만 완전한 실험실 acquisition pipeline을 모사하지 않습니다.' },
          { en: 'It does not model spatial artifacts, image-processing defaults, or application-specific perception metrics.', ko: 'Spatial artifact, image-processing default, application-specific perception metric을 모델링하지 않습니다.' },
          { en: 'Use official EMVA procedures for reportable camera characterization.', ko: '보고 가능한 camera characterization에는 공식 EMVA 절차를 사용해야 합니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
  'lens-shading': {
    title: { en: 'Relative Illumination and CRA Shading', ko: '상대 조도와 CRA 쉐이딩' },
    summary: {
      en: 'Lens shading combines optical cos-fourth falloff, chief-ray angle mismatch, microlens offset, and color-channel angular response.',
      ko: '렌즈 쉐이딩은 cos^4 상대 조도 저하, 주광선각 mismatch, 마이크로렌즈 offset, 색 채널별 각도 응답이 결합된 결과입니다.',
    },
    intuition: {
      en: 'Even with a perfect lens, the corners of every image come out darker than the centre — it is geometry, not a defect. Light arrives at edge pixels at a steep angle while the pixel optics are tuned for straight-on rays, so corner response drops. Modern sensors compensate by shifting microlenses inward at the edges, and image processing fills in the rest.',
      ko: '완벽한 렌즈를 써도 모든 이미지의 모서리는 중심보다 어둡게 나옵니다. 결함이 아니라 기하학적인 이유 때문이죠. 모서리 픽셀에는 빛이 비스듬히 들어오는데 픽셀 광학은 수직 입사에 최적화되어 있어 응답이 떨어집니다. 현대 센서는 모서리 쪽 마이크로렌즈를 안쪽으로 시프트해 보정하고, 나머지는 이미지 처리로 채웁니다.',
    },
    formulas: [
      {
        label: { en: 'Cos-fourth falloff', ko: 'cos^4 감쇠' },
        equation: 'RI(\\theta) \\approx \\cos^4(\\theta)',
        variables: [
          { symbol: 'RI', description: { en: 'Relative Illumination', ko: '상대 조도 $RI$' } },
          { symbol: '\\theta', description: { en: 'Chief Ray Angle (CRA)', ko: '주광선각(CRA) $\\theta$' } },
        ],
        note: { en: 'This is a classical first-order model for image-plane relative illumination.', ko: '이미지 평면 상대 조도의 고전적 1차 모델입니다.' },
      },
      {
        label: { en: 'Microlens shift', ko: '마이크로렌즈 시프트' },
        equation: '\\text{shift} \\approx h_{\\text{stack}} \\cdot \\tan(CRA_{\\text{eff}})',
        variables: [
          { symbol: 'h_{\\text{stack}}', description: { en: 'Optical stack height', ko: '광학 스택 높이 $h_{\\text{stack}}$' } },
        ],
        note: { en: 'Refraction in the stack usually makes CRA_eff smaller than the air-side CRA.', ko: '스택 내 굴절 때문에 CRA_eff는 보통 공기 중 CRA보다 작습니다.' },
      },
      {
        label: { en: 'Color shading', ko: '색 쉐이딩' },
        equation: 'G_c(r) = \\frac{S_c(0)}{S_c(r)}',
        variables: [
          { symbol: 'G_c(r)', description: { en: 'Spatial gain correction', ko: '공간적 이득 보정 계수 $G_c(r)$' } },
        ],
        note: { en: 'Per-channel lens-shading correction equalizes radial response.', ko: '채널별 렌즈 쉐이딩 보정은 반경별 응답을 맞춥니다.' },
      },
    ],
    concepts: [
      { en: 'Color shading appears when R/G/B angular responses are not identical.', ko: 'R/G/B 각도 응답이 다르면 색 쉐이딩이 나타납니다.' },
      { en: 'Microlens shift improves edge response but can reduce center tolerance if overdone.', ko: '마이크로렌즈 시프트는 주변부 응답을 개선하지만 과하면 중심부 허용도를 낮출 수 있습니다.' },
      { en: 'Production correction tables combine optical design and factory calibration.', ko: '양산 보정 테이블은 광학 설계와 공장 보정을 함께 반영합니다.' },
    ],
    sections: [
      {
        title: { en: 'Shading Sources', ko: 'Shading 원인' },
        items: [
          { en: 'Lens geometry creates classical $\\cos^4\\theta$ falloff before pixel optics are considered.', ko: 'Lens geometry는 pixel optics를 고려하기 전부터 고전적인 $\\cos^4\\theta$ falloff를 만듭니다.' },
          { en: 'Pixel stack angular response adds color-dependent rolloff when R/G/B filters respond differently to CRA.', ko: 'R/G/B 필터가 CRA에 다르게 반응하면 pixel stack angular response가 color-dependent rolloff를 추가합니다.' },
          { en: 'Microlens shift corrects lateral focus displacement but must match stack height and refracted CRA.', ko: 'Microlens shift는 lateral focus displacement를 보정하지만 stack height와 refracted CRA에 맞아야 합니다.' },
        ],
      },
      {
        title: { en: 'Correction Workflow', ko: '보정 절차' },
        items: [
          { en: 'Measure flat fields per channel, estimate radial or 2D gain maps $G_c(r)$, and apply correction before color processing.', ko: '채널별 flat field를 측정하고 radial 또는 2D gain map $G_c(r)$를 추정한 뒤 color processing 전에 보정합니다.' },
          { en: 'Separate optical shading from illumination nonuniformity and dust by using calibrated fixtures.', ko: 'Calibrated fixture를 사용해 optical shading을 illumination nonuniformity 및 dust와 분리합니다.' },
          { en: 'Validate correction at multiple focus distances and apertures if the lens system changes CRA distribution.', ko: 'Lens system이 CRA distribution을 바꾸면 여러 focus distance와 aperture에서 보정을 검증합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The model does not solve full lens chief-ray distribution, pupil aberration, or sensor cover-glass refraction.', ko: 'Full lens chief-ray distribution, pupil aberration, sensor cover-glass refraction을 풀지 않습니다.' },
          { en: 'Correction tables can amplify corner noise because they multiply weak signals.', ko: 'Correction table은 약한 신호를 곱해 키우므로 corner noise를 증폭할 수 있습니다.' },
          { en: 'Real products also compensate manufacturing variation, module tilt, and color-temperature dependence.', ko: '실제 제품은 manufacturing variation, module tilt, color-temperature dependence도 보정합니다.' },
        ],
      },
    ],
    references: [refs.hwang2023, refs.agranov2003],
  },
  'prnu-visualizer': {
    title: { en: 'Fixed-Pattern Non-Uniformity', ko: '고정 패턴 비균일성' },
    summary: {
      en: 'PRNU and DSNU model pixel-to-pixel response variation that remains spatially fixed across frames.',
      ko: 'PRNU와 DSNU는 프레임 사이에서 공간적으로 고정되어 남는 픽셀 간 응답 변동을 모델링합니다.',
    },
    intuition: {
      en: 'No two pixels are exactly identical — tiny manufacturing variations make some slightly more sensitive than others. PRNU is the fingerprint of those per-pixel sensitivity differences (visible in bright frames), and DSNU is the equivalent pattern in the dark baseline. Unlike random noise, these patterns are fixed in place from frame to frame, so they can largely be calibrated out.',
      ko: '두 픽셀이 완전히 똑같지는 않습니다. 미세한 공정 변동 때문에 어떤 픽셀은 다른 픽셀보다 약간 더 민감합니다. PRNU는 이러한 픽셀별 감도 차이의 “지문”(밝은 프레임에서 보임)이고, DSNU는 어두운 기준점에서의 같은 패턴입니다. 무작위 노이즈와 달리 프레임마다 동일한 위치에 고정되어 나타나므로 보정으로 대부분 제거할 수 있습니다.',
    },
    formulas: [
      {
        label: { en: 'Pixel signal model', ko: '픽셀 신호 모델' },
        equation: 'Y_{ij} = (1 + g_{ij})S + d_{ij} + n_{ij}',
        variables: [
          { symbol: 'g_{ij}', description: { en: 'Local gain deviation (PRNU)', ko: '국소 이득 편차 (PRNU) $g_{ij}$' } },
          { symbol: 'd_{ij}', description: { en: 'Local dark offset (DSNU)', ko: '국소 암전류 오프셋 (DSNU) $d_{ij}$' } },
        ],
        note: { en: 'g_ij is PRNU and d_ij is DSNU/dark offset.', ko: 'g_ij는 PRNU, d_ij는 DSNU 또는 dark offset입니다.' },
      },
      {
        label: { en: 'PRNU', ko: 'PRNU' },
        equation: 'PRNU = \\frac{\\sigma_{\\text{flat}}}{\\mu_{\\text{flat}}}',
        variables: [
          { symbol: '\\sigma_{\\text{flat}}', description: { en: 'Std dev of flat field', ko: '평탄 필드의 표준편차 $\\sigma_{\\text{flat}}$' } },
        ],
        note: { en: 'PRNU is measured from illuminated flat-field frames after offset correction.', ko: 'PRNU는 offset 보정 후 조명 flat-field 프레임에서 측정합니다.' },
      },
      {
        label: { en: 'DSNU', ko: 'DSNU' },
        equation: 'DSNU = \\sigma_{\\text{dark}}',
        variables: [
          { symbol: '\\sigma_{\\text{dark}}', description: { en: 'Std dev in the dark', ko: '암흑 상태의 표준편차 $\\sigma_{\\text{dark}}$' } },
        ],
        note: { en: 'DSNU describes spatial variation of dark signal.', ko: 'DSNU는 암신호의 공간적 변동을 나타냅니다.' },
      },
    ],
    concepts: [
      { en: 'PRNU grows with signal; DSNU is present even in darkness.', ko: 'PRNU는 신호와 함께 커지고 DSNU는 어두운 상태에서도 존재합니다.' },
      { en: 'Fixed-pattern noise can often be calibrated, but residuals remain after temperature and aging changes.', ko: '고정 패턴 노이즈는 보정 가능하지만 온도와 노화 변화 후 잔차가 남을 수 있습니다.' },
      { en: 'The visualizer uses synthetic spatial fields, not measured wafer maps.', ko: '이 시각화는 측정 wafer map이 아니라 합성 공간장을 사용합니다.' },
    ],
    sections: [
      {
        title: { en: 'Pattern Components', ko: '패턴 구성 요소' },
        items: [
          { en: 'PRNU is multiplicative, so its visible amplitude grows with signal level.', ko: 'PRNU는 곱셈성이므로 보이는 진폭이 신호 레벨과 함께 증가합니다.' },
          { en: 'DSNU is additive, so it is most visible in dark or low-signal frames.', ko: 'DSNU는 덧셈성이므로 dark 또는 low-signal frame에서 가장 잘 보입니다.' },
          { en: 'Random noise averages down over frames, but fixed-pattern structure stays locked to pixel coordinates.', ko: 'Random noise는 frame average로 줄지만 fixed-pattern 구조는 pixel coordinate에 고정됩니다.' },
        ],
      },
      {
        title: { en: 'Calibration Strategy', ko: '보정 전략' },
        items: [
          { en: 'Use dark frames for offset maps and flat fields for gain maps; update them when temperature or analog gain changes.', ko: 'Offset map에는 dark frame, gain map에는 flat field를 사용하며 온도나 analog gain이 바뀌면 갱신합니다.' },
          { en: 'Normalize flat fields carefully so lens shading is not mistaken for pixel PRNU.', ko: 'Lens shading이 pixel PRNU로 오인되지 않도록 flat field를 신중히 정규화합니다.' },
          { en: 'Residual fixed pattern after correction often reveals unmodeled temperature, exposure, or column effects.', ko: '보정 후 residual fixed pattern은 보통 모델링되지 않은 온도, 노출, column effect를 드러냅니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The synthetic field does not represent wafer reticle patterns, column circuits, or per-color CFA mismatch.', ko: '합성장은 wafer reticle pattern, column circuit, per-color CFA mismatch를 나타내지 않습니다.' },
          { en: 'It does not model temporal instability such as RTS pixels or temperature-dependent offsets.', ko: 'RTS pixel 또는 temperature-dependent offset 같은 temporal instability를 모델링하지 않습니다.' },
          { en: 'Real correction should be evaluated after black-level, lens-shading, and demosaic stages.', ko: '실제 보정은 black-level, lens-shading, demosaic 단계 이후 평가해야 합니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
  'pixel-snr-vs-illuminance': {
    title: { en: 'Illuminance-to-SNR Conversion', ko: '조도-SNR 변환' },
    summary: {
      en: 'This tool shows how scene illuminance becomes photons, photoelectrons, and finally SNR after adding sensor noise sources.',
      ko: '이 도구는 장면 조도가 광자, 광전자, 그리고 센서 노이즈를 포함한 최종 SNR로 변환되는 과정을 보여줍니다.',
    },
    intuition: {
      en: 'How does scene brightness (measured in lux) turn into image quality? This tool walks the whole chain: lux becomes photons per second per pixel, photons become electrons through QE, and electrons then compete with shot, dark, and read noise. The output curve shows where you are read-noise-limited (low light) versus shot-noise-limited (well-lit).',
      ko: '장면의 밝기(lux)는 어떻게 이미지 품질로 이어질까요? 이 도구는 그 변환 과정을 단계별로 보여줍니다. lux는 픽셀당 광자 수가 되고, 광자는 QE를 거쳐 전자가 되며, 전자는 샷·암전류·읽기 노이즈와 경쟁합니다. 결과 곡선은 어디서부터 읽기 노이즈가 제한 요인인지(저조도)와 샷 노이즈가 지배적인지(충분한 조명)를 보여줍니다.',
    },
    formulas: [
      {
        label: { en: 'Electron signal', ko: '전자 신호' },
        equation: 'S = N_{\\text{ph}} \\cdot QE',
        variables: [
          { symbol: 'N_{\\text{ph}}', description: { en: 'Incident photons', ko: '입사 광자 수 $N_{\\text{ph}}$' } },
        ],
        note: { en: 'Optical throughput and quantum efficiency turn photons into collected electrons.', ko: '광학 처리량과 양자 효율이 광자를 수집 전자로 바꿉니다.' },
      },
      {
        label: { en: 'Total noise', ko: '전체 노이즈' },
        equation: '\\sigma = \\sqrt{S + D \\cdot t + \\sigma_{\\text{read}}^2}',
        variables: [
          { symbol: '\\sigma', description: { en: 'Total RMS noise', ko: '전체 노이즈 표준편차 $\\sigma$' } },
        ],
        note: { en: 'The simulator separates shot, dark, and read-noise contributions.', ko: '시뮬레이터는 샷, 암전류, 읽기 노이즈 기여를 구분합니다.' },
      },
      {
        label: { en: 'Ideal limit', ko: '이상적 한계' },
        equation: '\\text{SNR}_{\\text{ideal}} = \\sqrt{S}',
        variables: [
          { symbol: 'S', description: { en: 'Signal charge', ko: '신호 전하량 $S$' } },
        ],
        note: { en: 'This is the best possible photon shot-noise limit for a given signal.', ko: '주어진 신호에서 가능한 최선의 광자 샷 노이즈 한계입니다.' },
      },
    ],
    concepts: [
      { en: 'Low-light SNR is usually read-noise and photon-starvation limited.', ko: '저조도 SNR은 보통 읽기 노이즈와 광자 부족에 의해 제한됩니다.' },
      { en: 'Large pixels collect more photons at the same illuminance and exposure.', ko: '같은 조도와 노출에서 큰 픽셀은 더 많은 광자를 수집합니다.' },
      { en: 'Illuminance-to-photon conversion depends on spectrum, lens f-number, and calibration assumptions.', ko: '조도-광자 변환은 스펙트럼, 렌즈 f-number, 보정 가정에 의존합니다.' },
    ],
    sections: [
      {
        title: { en: 'Lux To Electrons', ko: 'Lux에서 전자까지' },
        items: [
          { en: 'Illuminance is photopic and human-eye weighted, so converting lux to photons requires an assumed spectrum.', ko: 'Illuminance는 photopic human-eye weighted 값이므로 lux를 photon으로 바꾸려면 spectrum 가정이 필요합니다.' },
          { en: 'Lens f-number and transmittance set how much scene radiance reaches the pixel.', ko: 'Lens f-number와 transmittance는 scene radiance 중 픽셀에 도달하는 양을 결정합니다.' },
          { en: 'Pixel area, exposure time, and QE convert the arriving photon flux into signal electrons.', ko: 'Pixel area, exposure time, QE는 도달한 photon flux를 signal electron으로 변환합니다.' },
        ],
      },
      {
        title: { en: 'Reading The Curve', ko: '곡선 해석' },
        items: [
          { en: 'At the dim end, the curve is flat because read noise dominates.', ko: '어두운 끝에서는 read noise가 지배하므로 곡선이 평평합니다.' },
          { en: 'In the middle, SNR rises roughly with $\\sqrt{S}$ as photon shot noise dominates.', ko: '중간 영역에서는 photon shot noise가 지배해 SNR이 대략 $\\sqrt{S}$로 증가합니다.' },
          { en: 'At high illuminance, full well and PRNU can limit further SNR improvement.', ko: '높은 조도에서는 full well과 PRNU가 추가 SNR 개선을 제한할 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The conversion from lux to photons is spectrum dependent and not unique.', ko: 'Lux에서 photon으로의 변환은 spectrum 의존적이며 유일하지 않습니다.' },
          { en: 'The model does not include lens flare, scene contrast, demosaic, denoise, or tone mapping.', ko: 'Lens flare, scene contrast, demosaic, denoise, tone mapping은 포함하지 않습니다.' },
          { en: 'Real low-light quality depends on color noise, fixed-pattern noise, and temporal processing as well as scalar SNR.', ko: '실제 저조도 품질은 scalar SNR뿐 아니라 color noise, fixed-pattern noise, temporal processing에도 의존합니다.' },
        ],
      },
    ],
    references: [refs.emva, refs.catrysse2002],
  },
  'responsivity-calculator': {
    title: { en: 'Quantum Efficiency to Responsivity', ko: '양자 효율에서 응답도 변환' },
    summary: {
      en: 'Spectral responsivity converts optical power at a wavelength into photocurrent, using the photon energy and quantum efficiency.',
      ko: '분광 응답도는 파장별 광파워를 광전류로 변환하며, 광자 에너지와 양자 효율을 사용합니다.',
    },
    intuition: {
      en: 'QE tells you what fraction of photons turn into electrons — a number between 0 and 1. But circuit designers usually need current per watt of optical power (A/W) instead. Responsivity converts between the two using the energy of one photon, and it naturally grows with wavelength: longer-wavelength photons carry less energy each, so the same QE produces more current per watt.',
      ko: 'QE는 광자 중 몇 %가 전자로 바뀌는지(0~1 사이 값)를 알려줍니다. 그런데 회로 설계자에게는 보통 “와트당 전류(A/W)”가 더 필요합니다. 응답도는 광자 하나의 에너지를 이용해 두 값 사이를 변환하며, 긴 파장의 광자는 에너지가 작기 때문에 같은 QE라도 더 큰 A/W가 나옵니다.',
    },
    formulas: [
      {
        label: { en: 'Photon energy', ko: '광자 에너지' },
        equation: 'E_{\\text{ph}} = \\frac{hc}{\\lambda}',
        variables: [
          { symbol: 'h', description: { en: 'Planck constant', ko: '플랑크 상수 $h$' } },
          { symbol: 'c', description: { en: 'Speed of light', ko: '빛의 속도 $c$' } },
        ],
        note: { en: 'Longer wavelengths carry less energy per photon.', ko: '긴 파장의 광자는 하나당 에너지가 더 낮습니다.' },
      },
      {
        label: { en: 'Responsivity', ko: '응답도' },
        equation: '\\mathcal{R}(\\lambda) = \\frac{QE(\\lambda) \\cdot q \\lambda}{hc}',
        variables: [
          { symbol: '\\mathcal{R}', description: { en: 'Responsivity (A/W)', ko: '응답도 $\\mathcal{R}$' } },
          { symbol: 'q', description: { en: 'Elementary charge', ko: '기본 전하량 $q$' } },
        ],
        note: { en: 'With lambda in micrometers, R ~= QE*lambda/1.2398 A/W.', ko: 'lambda를 micrometer로 쓰면 R ~= QE*lambda/1.2398 A/W입니다.' },
      },
      {
        label: { en: 'Photocurrent', ko: '광전류' },
        equation: 'I_{\\text{ph}} = \\mathcal{R}(\\lambda) \\cdot P_{\\text{opt}}',
        variables: [
          { symbol: 'I_{\\text{ph}}', description: { en: 'Photocurrent', ko: '광전류 $I_{\\text{ph}}$' } },
          { symbol: 'P_{\\text{opt}}', description: { en: 'Optical power', ko: '입사 광파워 $P_{\\text{opt}}$' } },
        ],
        note: { en: 'Responsivity links optical simulation to electrical current.', ko: '응답도는 광학 시뮬레이션을 전기적 전류와 연결합니다.' },
      },
    ],
    concepts: [
      { en: 'The same QE gives higher A/W at longer wavelengths until silicon absorption falls.', ko: '같은 QE라면 긴 파장에서 A/W는 커지지만, 실리콘 흡수가 떨어지면 감소합니다.' },
      { en: 'Responsivity is not color accuracy; it is a power-to-current metric.', ko: '응답도는 색 정확도가 아니라 파워-전류 변환 지표입니다.' },
      { en: 'Measured responsivity includes optics, fill factor, and collection efficiency.', ko: '측정 응답도에는 광학계, fill factor, 수집 효율이 포함됩니다.' },
    ],
    sections: [
      {
        title: { en: 'QE Versus A/W', ko: 'QE와 A/W 비교' },
        items: [
          { en: 'QE counts electrons per photon; responsivity counts amperes per watt.', ko: 'QE는 photon당 electron 수를, responsivity는 watt당 ampere를 셉니다.' },
          { en: 'Because $E_{\\text{ph}}=hc/\\lambda$, the same photon conversion efficiency produces more current per watt at longer wavelength.', ko: '$E_{\\text{ph}}=hc/\\lambda$이므로 같은 photon conversion efficiency는 긴 파장에서 watt당 더 큰 전류를 만듭니다.' },
          { en: 'Responsivity can rise with wavelength even when photon absorption is not improving.', ko: 'Photon absorption이 개선되지 않아도 responsivity는 파장 증가와 함께 커질 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Measurement Use', ko: '측정 활용' },
        items: [
          { en: 'Use monochromatic calibrated optical power to measure spectral responsivity.', ko: '분광 응답도 측정에는 monochromatic calibrated optical power를 사용합니다.' },
          { en: 'Subtract dark current and verify linearity before converting photocurrent to responsivity.', ko: 'Photocurrent를 responsivity로 변환하기 전에 dark current를 빼고 선형성을 확인합니다.' },
          { en: 'Compare measured $\\mathcal{R}(\\lambda)$ with optical QE only after accounting for fill factor and collection efficiency.', ko: 'Fill factor와 collection efficiency를 고려한 뒤 measured $\\mathcal{R}(\\lambda)$를 optical QE와 비교합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The conversion assumes one collected electron per successful photon event and omits avalanche gain or multiplication.', ko: '이 변환은 성공한 photon event당 수집 electron 하나를 가정하며 avalanche gain 또는 multiplication을 생략합니다.' },
          { en: 'It does not include bandwidth, capacitance, transimpedance gain, or readout circuit limitations.', ko: 'Bandwidth, capacitance, transimpedance gain, readout circuit limitation은 포함하지 않습니다.' },
          { en: 'Broadband responsivity requires spectral integration over the source spectrum, not a single-wavelength value.', ko: 'Broadband responsivity는 단일 파장 값이 아니라 source spectrum에 대한 분광 적분이 필요합니다.' },
        ],
      },
    ],
    references: [refs.green2008, refs.emva],
  },
  'linearity-analyzer': {
    title: { en: 'Transfer-Curve Linearity', ko: '전달 곡선 선형성' },
    summary: {
      en: 'Linearity measures how closely output code follows input exposure before saturation or knee compression.',
      ko: '선형성은 포화 또는 knee 압축 전까지 출력 코드가 입력 노출을 얼마나 선형적으로 따르는지를 측정합니다.',
    },
    intuition: {
      en: 'An ideal sensor doubles its output when the light doubles — a perfectly straight line on a chart. Real sensors curve slightly, especially near saturation, and that curvature breaks precise photometry, HDR stitching, and colour correction. Linearity analysis hunts for the curvature using the residual between measured points and a fitted straight line.',
      ko: '이상적인 센서는 빛이 두 배가 되면 출력도 두 배가 됩니다. 그래프상 완벽한 직선이죠. 실제 센서는 특히 포화 근처에서 살짝 곡선을 그리고, 그 곡률이 정밀 광도 측정, HDR 합성, 색 보정을 망칩니다. 선형성 분석은 측정점과 적합 직선의 잔차에서 그 곡률을 찾아냅니다.',
    },
    formulas: [
      {
        label: { en: 'Ideal response', ko: '이상 응답' },
        equation: 'Y_{\\text{ideal}} = gX + \\text{offset}',
        variables: [
          { symbol: 'g', description: { en: 'Slope (gain)', ko: '기울기 (이득) $g$' } },
          { symbol: 'X', description: { en: 'Input signal', ko: '입력 신호 $X$' } },
        ],
        note: { en: 'A linear sensor has constant slope over the usable exposure range.', ko: '선형 센서는 사용 가능한 노출 범위에서 일정한 기울기를 가집니다.' },
      },
      {
        label: { en: 'Residual', ko: '잔차' },
        equation: '\\text{residual} = Y_{\\text{meas}} - Y_{\\text{fit}}',
        variables: [
          { symbol: 'Y_{\\text{meas}}', description: { en: 'Measured digital value', ko: '실제 측정된 디지털 값 $Y_{\\text{meas}}$' } },
        ],
        note: { en: 'Residual plots reveal curvature that is hidden in the main transfer curve.', ko: '잔차 플롯은 주 전달 곡선에서 잘 보이지 않는 곡률을 드러냅니다.' },
      },
      {
        label: { en: 'Integral nonlinearity', ko: '적분 비선형성' },
        equation: 'INL = \\frac{\\max(|\\text{residual}|)}{\\text{full scale}}',
        variables: [
          { symbol: 'INL', description: { en: 'Integral Non-Linearity', ko: '적분 비선형성 $INL$' } },
        ],
        note: { en: 'INL normalizes the worst deviation by full-scale output.', ko: 'INL은 최대 편차를 full-scale 출력으로 정규화합니다.' },
      },
    ],
    concepts: [
      { en: 'Knee compression extends highlight range at the cost of linear radiometry.', ko: 'knee 압축은 선형 방사 측정성을 희생해 하이라이트 범위를 늘립니다.' },
      { en: 'Linearity errors affect photometry, color correction, HDR merge, and calibration.', ko: '선형성 오차는 광도 측정, 색 보정, HDR 병합, 보정에 영향을 줍니다.' },
      { en: 'Real measurements need black-level subtraction and stable illumination.', ko: '실측에는 black-level 제거와 안정적인 조명이 필요합니다.' },
    ],
    sections: [
      {
        title: { en: 'Residual Interpretation', ko: '잔차 해석' },
        items: [
          { en: 'A smooth residual curve indicates transfer nonlinearity; alternating residuals can indicate measurement noise or flicker.', ko: '매끄러운 residual curve는 transfer nonlinearity를, 번갈아 흔들리는 residual은 measurement noise 또는 flicker를 시사할 수 있습니다.' },
          { en: 'Near saturation, residuals often grow before hard clipping appears.', ko: '포화 근처에서는 hard clipping이 보이기 전에 residual이 커지는 경우가 많습니다.' },
          { en: 'Use a defined fit range; including the knee region can hide low-signal linearity.', ko: '정의된 fit range를 사용해야 합니다. Knee 영역을 포함하면 low-signal linearity가 가려질 수 있습니다.' },
        ],
      },
      {
        title: { en: 'Calibration Impact', ko: '보정 영향' },
        items: [
          { en: 'Color correction assumes linear channel ratios; nonlinearity changes those ratios with exposure.', ko: 'Color correction은 선형 channel ratio를 가정하며, 비선형성은 노출에 따라 그 비율을 바꿉니다.' },
          { en: 'HDR merge needs a known transfer function to align short and long exposures.', ko: 'HDR merge는 short/long exposure를 맞추기 위해 알려진 transfer function이 필요합니다.' },
          { en: 'Photometry and scientific imaging require tighter linearity than consumer tone-mapped imaging.', ko: 'Photometry와 scientific imaging은 consumer tone-mapped imaging보다 더 엄격한 선형성이 필요합니다.' },
        ],
      },
      {
        title: { en: 'Known Missing Physics', ko: '아직 빠진 물리' },
        items: [
          { en: 'The compact analyzer does not model ADC code transitions, column gain errors, or source-follower compression separately.', ko: 'Compact analyzer는 ADC code transition, column gain error, source-follower compression을 별도로 모델링하지 않습니다.' },
          { en: 'It does not include temperature drift, illumination drift, or exposure timing error during measurement.', ko: '측정 중 temperature drift, illumination drift, exposure timing error를 포함하지 않습니다.' },
          { en: 'Piecewise HDR and dual-gain sensors require multiple transfer curves, not one global line.', ko: 'Piecewise HDR 및 dual-gain sensor는 하나의 global line이 아니라 여러 transfer curve가 필요합니다.' },
        ],
      },
    ],
    references: [refs.emva],
  },
}

const entry = computed(() => theoryEntries[props.slug])
</script>

<style scoped>
.sim-theory {
  margin: 36px 0 24px;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
}

.sim-theory-eyebrow {
  margin-bottom: 4px;
  color: var(--vp-c-brand-1);
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.sim-theory h2 {
  margin: 0 0 8px;
  border-top: 0;
  padding-top: 0;
  font-size: 1.25em;
}

.sim-theory h3 {
  margin: 0 0 12px;
  font-size: 1em;
  color: var(--vp-c-text-1);
}

.sim-theory-summary {
  margin: 0 0 16px;
  color: var(--vp-c-text-2);
}

.sim-theory-intuition {
  margin: 0 0 16px;
  padding: 12px 14px;
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 6px;
  background: var(--vp-c-brand-soft);
}

.intuition-eyebrow {
  margin-bottom: 4px;
  color: var(--vp-c-brand-1);
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.sim-theory-intuition p {
  margin: 0;
  color: var(--vp-c-text-1);
  font-size: 0.95em;
  line-height: 1.55;
}

.sim-theory-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 14px;
}

.sim-theory-standard-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 16px 0;
}

.sim-theory-detail-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 14px;
}

.sim-theory-card {
  padding: 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
}

.sim-theory-card ul {
  margin: 0;
  padding-left: 18px;
}

.sim-theory-card li {
  margin: 6px 0;
}

.formula-row {
  padding: 10px 0;
  border-top: 1px solid var(--vp-c-divider);
}

.formula-row:first-of-type {
  border-top: 0;
  padding-top: 0;
}

.formula-label {
  margin-bottom: 4px;
  color: var(--vp-c-text-2);
  font-size: 0.82em;
  font-weight: 600;
}

.formula-equation {
  display: block;
  margin: 6px 0;
  padding: 10px;
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  overflow-x: auto;
  font-size: 0.95em;
}

.formula-variables {
  margin: 8px 0;
  padding: 8px 12px;
  border-left: 2px solid var(--vp-c-divider);
  font-size: 0.88em;
}

.formula-variables ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.formula-variables li {
  margin: 2px 0;
  color: var(--vp-c-text-1);
}

.var-symbol {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

/* Ensure MathJax block math doesn't add extra margins here */
.formula-equation :deep(mjx-container[display="true"]) {
  margin: 0 !important;
  text-align: left !important;
}

.formula-row p {
  margin: 6px 0 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.references-card {
  margin-top: 14px;
}

@media (max-width: 760px) {
  .sim-theory {
    padding: 14px;
  }

  .sim-theory-grid {
    grid-template-columns: 1fr;
  }

  .sim-theory-standard-grid {
    grid-template-columns: 1fr;
  }

  .sim-theory-detail-grid {
    grid-template-columns: 1fr;
  }
}
</style>
