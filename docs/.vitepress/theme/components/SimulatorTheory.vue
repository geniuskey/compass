<template>
  <section v-if="entry" ref="theoryRoot" class="sim-theory">
    <div class="sim-theory-eyebrow">{{ t('Physics Notes', '물리 수식과 이론') }}</div>
    <h2>{{ pick(entry.title) }}</h2>

    <div v-if="entry.intuition" class="sim-theory-intuition">
      <div class="intuition-eyebrow">{{ t('Plain-English Intuition', '쉽게 이해하기') }}</div>
      <p>{{ pick(entry.intuition) }}</p>
    </div>

    <p class="sim-theory-summary">{{ pick(entry.summary) }}</p>

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

interface TheoryEntry {
  title: Localized
  summary: Localized
  intuition?: Localized
  formulas: Formula[]
  concepts: Localized[]
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
      en: 'The calculator treats the pixel stack as a one-dimensional sequence of planar films. It solves coherent reflection and transmission first, then reports silicon-layer absorption as the optical upper bound for quantum efficiency.',
      ko: '이 계산기는 픽셀 스택을 1차원 평면 박막의 연속으로 보고, 먼저 간섭에 의한 반사/투과를 푼 뒤 실리콘층 흡수를 양자 효율의 광학적 상한으로 표시합니다.',
    },
    intuition: {
      en: 'Imagine sunlight hitting a stack of clear films layered like a sandwich. At every boundary some light bounces back and some passes through, and those waves can either reinforce or cancel each other — the same trick that paints rainbow colours on a soap bubble. This tool tracks every bouncing wave and reports how much light actually reaches the silicon, where photons become electrons.',
      ko: '햇빛이 투명한 필름이 여러 겹 쌓인 샌드위치를 통과한다고 상상해 보세요. 경계마다 빛의 일부는 반사되고 일부는 통과하며, 이 파동들이 서로 보강하거나 상쇄되어 비누 거품의 무지개색 같은 간섭을 만듭니다. 이 도구는 그 모든 반사파를 추적해 광자가 전자로 바뀌는 곳, 즉 실리콘에 빛이 얼마나 도달하는지를 알려줍니다.',
    },
    formulas: [
      {
        label: { en: 'Layer phase', ko: '레이어 위상' },
        equation: '\\delta_j = \\frac{2\\pi n_j d_j \\cos\\theta_j}{\\lambda}',
        variables: [
          { symbol: '\\delta_j', description: { en: 'Phase delay in layer $j$', ko: '레이어 $j$의 위상 지연' } },
          { symbol: 'n_j', description: { en: 'Refractive index of layer $j$', ko: '레이어 $j$의 굴절률' } },
          { symbol: 'd_j', description: { en: 'Thickness of layer $j$', ko: '레이어 $j$의 두께' } },
          { symbol: '\\theta_j', description: { en: 'Angle of refraction in layer $j$', ko: '레이어 $j$ 내의 굴절각' } },
          { symbol: '\\lambda', description: { en: 'Wavelength of light', ko: '빛의 파장 $\\lambda$' } },
        ],
        note: { en: 'Each layer contributes a wavelength- and angle-dependent phase delay.', ko: '각 레이어는 파장과 입사각에 의존하는 위상 지연을 만듭니다.' },
      },
      {
        label: { en: 'Energy balance', ko: '에너지 보존' },
        equation: 'R(\\lambda) + T(\\lambda) + \\sum_j A_j(\\lambda) = 1',
        variables: [
          { symbol: 'R', description: { en: 'Reflectance (fraction of power reflected)', ko: '반사율 $R$' } },
          { symbol: 'T', description: { en: 'Transmittance (fraction of power transmitted)', ko: '투과율 $T$' } },
          { symbol: 'A_j', description: { en: 'Absorbance in layer $j$', ko: '레이어 $j$의 흡수율' } },
        ],
        note: { en: 'A physically consistent TMM result should conserve incident optical power.', ko: '물리적으로 일관된 TMM 결과는 입사 광파워를 보존해야 합니다.' },
      },
      {
        label: { en: 'Optical QE proxy', ko: '광학 QE 근사' },
        equation: 'QE_{\\text{opt}}(\\lambda) \\approx A_{\\text{Si}}(\\lambda)',
        variables: [
          { symbol: 'QE_{\\text{opt}}', description: { en: 'Optimistic estimate of quantum efficiency', ko: '낙관적으로 추정된 양자 효율 $QE_{\\text{opt}}$' } },
          { symbol: 'A_{\\text{Si}}', description: { en: 'Absorption fraction in the photodiode silicon', ko: '실리콘 광검출층에서의 흡수율 $A_{\\text{Si}}$' } },
        ],
        note: { en: 'Carrier collection loss is not modeled, so silicon absorption is an optimistic QE estimate.', ko: '전하 수집 손실은 포함하지 않으므로 실리콘 흡수율은 낙관적인 QE 추정치입니다.' },
      },
    ],
    concepts: [
      { en: 'Best for flat BSI stacks, BARL tuning, and first-pass spectral trends.', ko: '평탄 BSI 스택, BARL 조정, 1차 분광 경향 확인에 적합합니다.' },
      { en: 'It cannot capture lateral diffraction, metal-grid shadowing, DTI crosstalk, or microlens focusing.', ko: '횡방향 회절, 금속 그리드 차광, DTI 크로스토크, 마이크로렌즈 집광은 포착하지 못합니다.' },
      { en: 'Large angle sweeps should be read as planar-film angular response, not full camera CRA behavior.', ko: '큰 각도 스윕은 전체 카메라 CRA가 아니라 평면 박막의 각도 응답으로 해석해야 합니다.' },
    ],
    references: [refs.macleod, refs.green2008, refs.catrysse2002],
  },
  'barl-optimizer': {
    title: { en: 'Thin-Film Anti-Reflection Theory', ko: '박막 반사 방지 이론' },
    summary: {
      en: 'BARL and coating stacks reduce Fresnel reflection by arranging multiple reflected waves to cancel at target wavelengths.',
      ko: 'BARL 및 박막 코팅은 여러 계면 반사파가 목표 파장에서 상쇄되도록 두께와 굴절률을 조합해 프레넬 반사를 줄입니다.',
    },
    intuition: {
      en: 'Anti-reflection layers work like noise-cancelling headphones for light. By stacking the right films at the right thicknesses, the unwanted reflections from each interface cancel each other out, so more light passes through and reaches the sensor instead of bouncing back into the lens.',
      ko: '반사 방지 박막은 빛에 대한 노이즈 캔슬링 이어폰처럼 동작합니다. 알맞은 두께의 박막을 잘 쌓으면 각 경계면의 반사파끼리 서로 상쇄되어, 빛이 렌즈 쪽으로 튀어나가지 않고 더 많이 센서로 들어옵니다.',
    },
    formulas: [
      {
        label: { en: 'Normal-incidence Fresnel reflection', ko: '수직 입사 프레넬 반사' },
        equation: 'R = \\left| \\frac{n_0 - n_s}{n_0 + n_s} \\right|^2',
        variables: [
          { symbol: 'R', description: { en: 'Reflectance coefficient', ko: '반사 계수 $R$' } },
          { symbol: 'n_0', description: { en: 'Refractive index of the incident medium', ko: '입사 매질의 굴절률 $n_0$' } },
          { symbol: 'n_s', description: { en: 'Refractive index of the substrate', ko: '기판의 굴절률 $n_s$' } },
        ],
        note: { en: 'A bare polymer/silicon or oxide/silicon interface has high reflection because the index contrast is large.', ko: '폴리머/실리콘 또는 산화막/실리콘 계면은 굴절률 차이가 커서 반사가 큽니다.' },
      },
      {
        label: { en: 'Quarter-wave seed thickness', ko: '1/4파장 초기 두께' },
        equation: 'd \\approx \\frac{\\lambda_0}{4n_{\\text{layer}}}',
        variables: [
          { symbol: 'd', description: { en: 'Optimal thickness for destructive interference', ko: '상쇄 간섭을 위한 최적 두께 $d$' } },
          { symbol: '\\lambda_0', description: { en: 'Target wavelength', ko: '목표 파장 $\\lambda_0$' } },
          { symbol: 'n_{\\text{layer}}', description: { en: 'Refractive index of the layer', ko: '박막 레이어의 굴절률 $n_{\\text{layer}}$' } },
        ],
        note: { en: 'A quarter-wave layer gives a useful starting point, but broadband stacks require numerical tuning.', ko: '1/4파장 두께는 좋은 출발점이지만 광대역 스택은 수치 최적화가 필요합니다.' },
      },
      {
        label: { en: 'Merit function', ko: '목적 함수' },
        equation: '\\mathcal{L} = \\langle R(\\lambda) \\rangle_{\\lambda \\in \\text{band}}',
        variables: [
          { symbol: '\\mathcal{L}', description: { en: 'Loss function to be minimized', ko: '최소화해야 할 손실 함수 $\\mathcal{L}$' } },
          { symbol: '\\langle R(\\lambda) \\rangle', description: { en: 'Mean reflectance across the band', ko: '대역 내 평균 반사율' } },
        ],
        note: { en: 'The optimizer minimizes average reflectance over the selected spectral band.', ko: '최적화기는 선택한 파장 대역의 평균 반사율을 줄입니다.' },
      },
    ],
    concepts: [
      { en: 'The optimum depends on the incident medium, substrate, wavelength band, and angle.', ko: '최적점은 입사 매질, 기판, 파장 대역, 입사각에 따라 달라집니다.' },
      { en: 'A stack optimized for green peak QE can hurt blue or NIR response.', ko: '녹색 피크 QE에 맞춘 스택은 청색 또는 NIR 응답을 악화시킬 수 있습니다.' },
      { en: 'Real CIS BARL recipes are process-specific and usually use constrained material sets.', ko: '실제 CIS BARL 레시피는 공정별로 다르며 사용할 수 있는 재료 조합도 제한됩니다.' },
    ],
    references: [refs.macleod, refs.bornWolf, refs.green2008],
  },
  'energy-budget': {
    title: { en: 'Photon Energy Accounting', ko: '광자 에너지 예산' },
    summary: {
      en: 'The analyzer decomposes incident optical power into reflected power, transmitted power, and absorption in each stack layer.',
      ko: '이 분석기는 입사 광파워를 반사, 투과, 각 스택 레이어에서의 흡수로 분해합니다.',
    },
    intuition: {
      en: 'Think of incoming light as a budget. Some of it is lost to reflection (bounces away), some is spent as toll fees inside the colour filter or metal layers, and only the part absorbed in silicon becomes a useful electrical signal. This tool gives you an itemised receipt that shows exactly where every percent of the budget went.',
      ko: '들어오는 빛을 예산이라고 생각해 보세요. 일부는 반사로 튕겨 나가 잃어버리고, 일부는 컬러 필터와 금속층에서 통행료로 쓰이며, 실리콘 내부에서 흡수된 부분만 실제 전기 신호가 됩니다. 이 도구는 예산의 몇 %가 어디로 사용되었는지를 영수증처럼 항목별로 보여줍니다.',
    },
    formulas: [
      {
        label: { en: 'Conservation check', ko: '보존성 검증' },
        equation: '1 = R + T + A_{\\text{CF}} + A_{\\text{BARL}} + A_{\\text{Si}} + \\dots',
        variables: [
          { symbol: 'R', description: { en: 'Total reflectance', ko: '전체 반사율 $R$' } },
          { symbol: 'T', description: { en: 'Total transmittance', ko: '전체 투과율 $T$' } },
          { symbol: 'A_{\\text{Si}}', description: { en: 'Useful absorption in silicon photodiode', ko: '실리콘 포토다이오드에서의 유효 흡수율 $A_{\\text{Si}}$' } },
        ],
        note: { en: 'The residual should stay near zero; otherwise the optical model or sampling is inconsistent.', ko: '잔차는 0에 가까워야 하며, 그렇지 않다면 광학 모델이나 샘플링을 의심해야 합니다.' },
      },
      {
        label: { en: 'Layer absorption', ko: '레이어 흡수' },
        equation: 'A_j = P_{\\text{in},j} - P_{\\text{out},j}',
        variables: [
          { symbol: 'A_j', description: { en: 'Absorption in specific layer $j$', ko: '특정 레이어 $j$에서의 흡수율' } },
          { symbol: 'P_{\\text{in},j}', description: { en: 'Power flux entering layer $j$', ko: '레이어 $j$로 입사된 광파워 플럭스' } },
          { symbol: 'P_{\\text{out},j}', description: { en: 'Power flux exiting layer $j$', ko: '레이어 $j$를 빠져나간 광파워 플럭스' } },
        ],
        note: { en: 'Absorption is assigned by tracking the power flux entering and leaving a layer.', ko: '흡수는 해당 레이어로 들어간 파워와 나간 파워의 차이로 배분합니다.' },
      },
      {
        label: { en: 'Useful absorption', ko: '유효 흡수' },
        equation: 'QE_{\\text{opt}} \\leq A_{\\text{Si}}',
        variables: [
          { symbol: 'QE_{\\text{opt}}', description: { en: 'External quantum efficiency estimate', ko: '외부 양자 효율 추정치 $QE_{\\text{opt}}$' } },
        ],
        note: { en: 'Only absorption inside the photodiode silicon can become collected charge.', ko: '포토다이오드 실리콘 내부 흡수만 수집 전하로 변환될 수 있습니다.' },
      },
    ],
    concepts: [
      { en: 'Color-filter and metal absorption reduce throughput before photons reach silicon.', ko: '컬러 필터와 금속 흡수는 광자가 실리콘에 도달하기 전 처리량을 낮춥니다.' },
      { en: 'Transmission out of the silicon layer is lost unless backside reflection or light trapping is modeled.', ko: '후면 반사나 광 포획을 모델링하지 않으면 실리콘을 통과한 빛은 손실로 봅니다.' },
      { en: 'Use the budget to identify whether a QE loss is reflection-limited, filter-limited, or silicon-thickness-limited.', ko: '에너지 예산은 QE 손실이 반사, 필터, 실리콘 두께 중 무엇에 의해 제한되는지 구분하는 데 유용합니다.' },
    ],
    references: [refs.catrysse2002, refs.green2008, refs.macleod],
  },
  'angular-response': {
    title: { en: 'Chief-Ray Angle and Polarization Response', ko: '주광선각 및 편광 응답' },
    summary: {
      en: 'Angular response combines Snell refraction, polarization-dependent Fresnel coefficients, and longer optical path length through absorbing layers.',
      ko: '각도 응답은 스넬 굴절, 편광별 프레넬 계수, 흡수층 내 광경로 증가가 함께 만든 결과입니다.',
    },
    intuition: {
      en: 'Light coming straight down acts differently from light arriving at an angle, just as rain falling vertically differs from rain hitting a windshield. Tilted rays bend at each surface, travel a longer effective path through each absorbing layer, and behave differently for the two polarisations — so QE typically drops as the incidence angle grows.',
      ko: '곧장 위에서 떨어지는 빛과 비스듬히 들어오는 빛은 동작이 다릅니다. 수직으로 내리는 비와 자동차 앞유리에 비스듬히 부딪히는 비가 다른 것과 같죠. 기울어진 광선은 각 경계에서 굴절되고, 흡수층 내부에서 더 긴 경로를 지나며, 편광에 따라 다르게 반응하기 때문에 입사각이 커질수록 보통 QE가 떨어집니다.',
    },
    formulas: [
      {
        label: { en: 'Snell law', ko: '스넬 법칙' },
        equation: 'n_0 \\sin(\\theta_0) = n_j \\sin(\\theta_j)',
        variables: [
          { symbol: 'n_0, n_j', description: { en: 'Refractive indices of incident and $j$-th media', ko: '입사 매질 및 $j$번째 매질의 굴절률' } },
          { symbol: '\\theta_0, \\theta_j', description: { en: 'Angles of incidence and refraction', ko: '입사각 및 굴절각' } },
        ],
        note: { en: 'Higher-index layers bend the ray toward normal and reduce the internal angle.', ko: '고굴절률 레이어는 광선을 법선 쪽으로 굴절시켜 내부 각도를 줄입니다.' },
      },
      {
        label: { en: 'Path length increase', ko: '광경로 증가' },
        equation: 'd_{\\text{eff}} = \\frac{d}{\\cos(\\theta_j)}',
        variables: [
          { symbol: 'd_{\\text{eff}}', description: { en: 'Effective path length', ko: '유효 경로 길이' } },
          { symbol: 'd', description: { en: 'Physical layer thickness', ko: '레이어의 물리적 두께' } },
        ],
        note: { en: 'Oblique rays see a thicker effective absorber or filter.', ko: '경사 입사 광선은 흡수층이나 필터를 더 두껍게 통과합니다.' },
      },
      {
        label: { en: 'Relative angular QE', ko: '상대 각도 QE' },
        equation: 'AR(\\theta, \\lambda) = \\frac{QE(\\theta, \\lambda)}{QE(0, \\lambda)}',
        variables: [
          { symbol: 'AR', description: { en: 'Angular Response', ko: '각도 응답' } },
          { symbol: 'QE(\\theta)', description: { en: 'QE at angle $\\theta$', ko: '입사각 $\\theta$에서의 QE' } },
        ],
        note: { en: 'This normalizes away the absolute calibration and highlights roll-off.', ko: '절대 보정을 제거하고 각도에 따른 롤오프를 보여줍니다.' },
      },
    ],
    concepts: [
      { en: 's- and p-polarization can diverge strongly near high angles.', ko: '큰 입사각에서는 s/p 편광 응답이 크게 달라질 수 있습니다.' },
      { en: 'Real edge pixels also need microlens shift and finite-cone integration.', ko: '실제 주변부 픽셀은 마이크로렌즈 시프트와 유한 조리개 cone 적분도 필요합니다.' },
      { en: 'The planar model is useful for stack sensitivity but incomplete for full-pixel CRA optimization.', ko: '평면 모델은 스택 민감도 분석에는 유용하지만 전체 픽셀 CRA 최적화에는 불충분합니다.' },
    ],
    references: [refs.hwang2023, refs.goossens2018, refs.macleod],
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
    references: [refs.agranov2003, refs.hwang2023, refs.catrysse2002],
  },
  'microlens-process-shape': {
    title: { en: 'Layout-to-Reflow Shape Surrogate', ko: '레이아웃-리플로우 형상 대리 모델' },
    summary: {
      en: 'The process predictor uses simplified conservation and empirical knobs to connect mask layout, thermal reflow, and etch-transfer settings to final lens geometry.',
      ko: '공정 형상 예측기는 단순화한 보존 법칙과 경험적 계수를 사용해 마스크 레이아웃, 열 리플로우, 식각 전사 조건을 최종 렌즈 형상과 연결합니다.',
    },
    intuition: {
      en: 'When a foundry builds a microlens, it starts as a square block of photoresist, melts it into a dome (reflow), then etches that dome down into the underlying material. The final lens shape depends on the starting size, melt time, and etch settings. This tool predicts the final shape with simple physical rules — useful for first-cut layout-to-lens trends, not for an exact foundry recipe.',
      ko: '공장에서 마이크로렌즈를 만들 때는 먼저 직사각형 모양의 포토레지스트에서 시작해, 열로 녹여 돔 모양으로 만들고(리플로우), 그 돔을 아래 층으로 식각해 옮깁니다. 최종 렌즈 모양은 시작 크기, 녹이는 시간, 식각 조건에 따라 달라집니다. 이 도구는 그 결과 형상을 단순한 물리 규칙으로 예측하므로, 정확한 공정 레시피보다는 레이아웃과 렌즈 모양의 1차 경향을 살피는 데 적합합니다.',
    },
    formulas: [
      {
        label: { en: 'Volume conservation', ko: '체적 보존' },
        equation: 'V_{\\text{resist}} \\approx V_{\\text{lens, final}}',
        variables: [
          { symbol: 'V', description: { en: 'Material volume', ko: '재료 체적 $V$' } },
        ],
        note: { en: 'Reflow reshapes the resist island while approximately conserving material volume.', ko: '리플로우는 resist island의 형상을 바꾸지만 재료 체적은 대략 보존합니다.' },
      },
      {
        label: { en: 'Sag-radius relation', ko: 'sag-곡률 관계' },
        equation: 'R_{\\text{vtx}} \\approx \\frac{a^2 + h^2}{2h}',
        variables: [
          { symbol: 'R_{\\text{vtx}}', description: { en: 'Vertex radius of curvature', ko: '꼭짓점 곡률 반경' } },
          { symbol: 'a', description: { en: 'Lens aperture radius', ko: '렌즈 개구 반경 $a$' } },
          { symbol: 'h', description: { en: 'Lens sag (height)', ko: '렌즈 sag (높이) $h$' } },
        ],
        note: { en: 'For a spherical-cap approximation, aperture radius a and sag h set vertex curvature.', ko: '구면 cap 근사에서 개구 반경 a와 sag h가 꼭짓점 곡률을 결정합니다.' },
      },
      {
        label: { en: 'Gap closure trend', ko: 'gap closure 경향' },
        equation: 'g_{\\text{final}} = \\max(0, g_{\\text{lay}} - \\Delta_{\\text{reflow}} - \\Delta_{\\text{etch}})',
        variables: [
          { symbol: 'g', description: { en: 'Gap between lenses', ko: '렌즈 간 간격 $g$' } },
          { symbol: '\\Delta', description: { en: 'Process-induced gap reduction', ko: '공정에 의한 간격 감소량 $\\Delta$' } },
        ],
        note: { en: 'The simulator keeps the directional trend explicit so coefficients can later be calibrated.', ko: '나중에 계수를 보정할 수 있도록 방향성 관계를 명시적으로 둡니다.' },
      },
    ],
    concepts: [
      { en: 'Public papers usually report DOE trends, not universal foundry recipes.', ko: '공개 논문은 대개 범용 foundry recipe가 아니라 DOE 경향을 보고합니다.' },
      { en: 'AFM/SEM metrology is needed to calibrate real process coefficients.', ko: '실제 공정 계수 보정에는 AFM/SEM 계측이 필요합니다.' },
      { en: 'The output is best read as sensitivity and failure-mode guidance.', ko: '출력은 정량 recipe보다 민감도와 failure mode 지침으로 해석하는 것이 좋습니다.' },
    ],
    references: [refs.ristoiu2020, refs.baillie2004],
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
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  gap: 14px;
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
}
</style>
