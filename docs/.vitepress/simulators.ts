// Single source of truth for the browser simulators.
// Used by the EN/KO simulator sidebars (config.mts) and the
// <SimulatorGallery /> component on the simulator overview pages.

export interface SimulatorEntry {
  slug: string
  icon: string
  en: { title: string; desc: string }
  ko: { title: string; desc: string }
}

export interface SimulatorCategory {
  id: string
  en: string
  ko: string
  items: SimulatorEntry[]
}

export const simulatorCategories: SimulatorCategory[] = [
  {
    id: 'stack',
    en: 'Optical Stack Analysis',
    ko: '광학 스택 분석',
    items: [
      {
        slug: 'tmm-qe',
        icon: '📈',
        en: { title: 'TMM QE Calculator', desc: 'BSI stack QE spectrum via transfer matrices, in real time.' },
        ko: { title: 'TMM QE 계산기', desc: '전달 행렬법으로 BSI 스택의 QE 스펙트럼을 실시간 계산.' },
      },
      {
        slug: 'barl-optimizer',
        icon: '🧱',
        en: { title: 'Thin Film Stack Designer', desc: 'Design multilayer coatings and watch R/T spectra update.' },
        ko: { title: '박막 스택 설계기', desc: '다층 박막 코팅을 설계하고 R/T 스펙트럼을 확인.' },
      },
      {
        slug: 'energy-budget',
        icon: '⚖️',
        en: { title: 'Energy Budget', desc: 'Where each photon goes: reflection, per-layer absorption, silicon.' },
        ko: { title: '에너지 버짓', desc: '반사·층별 흡수·실리콘 흡수로 광자 에너지 분배를 추적.' },
      },
    ],
  },
  {
    id: 'performance',
    en: 'Performance Analysis',
    ko: '성능 분석',
    items: [
      {
        slug: 'angular-response',
        icon: '📐',
        en: { title: 'Angular Response', desc: 'QE versus angle of incidence and chief ray angle.' },
        ko: { title: '각도 응답', desc: '입사각·주광선각(CRA)에 따른 QE 변화.' },
      },
      {
        slug: 'snr-calculator',
        icon: '📶',
        en: { title: 'SNR Calculator', desc: 'Pixel SNR from QE, pitch, noise sources and illuminance.' },
        ko: { title: 'SNR 계산기', desc: 'QE·피치·노이즈·조도로부터 픽셀 SNR 계산.' },
      },
      {
        slug: 'photon-transfer-curve',
        icon: '〰️',
        en: { title: 'Photon Transfer Curve', desc: 'Extract read noise, conversion gain and FWC from a PTC.' },
        ko: { title: '광자 전달 곡선', desc: 'PTC에서 읽기 노이즈·변환 이득·FWC 추출.' },
      },
      {
        slug: 'pixel-snr-vs-illuminance',
        icon: '🌗',
        en: { title: 'SNR vs Illuminance', desc: 'Noise breakdown against the ideal shot-noise limit.' },
        ko: { title: 'SNR vs 조도', desc: '노이즈 성분별 분해와 이상적 샷 노이즈 한계 비교.' },
      },
      {
        slug: 'dynamic-range',
        icon: '🎚️',
        en: { title: 'Dynamic Range', desc: 'Single-exposure vs HDR dynamic range over temperature.' },
        ko: { title: '다이나믹 레인지', desc: '단일 노출과 HDR 모드의 다이나믹 레인지 비교.' },
      },
      {
        slug: 'responsivity-calculator',
        icon: '⚡',
        en: { title: 'Responsivity', desc: 'Spectral responsivity (A/W) from TMM-based QE.' },
        ko: { title: '응답도 계산기', desc: 'TMM 기반 QE로부터 분광 응답도(A/W) 계산.' },
      },
    ],
  },
  {
    id: 'design',
    en: 'Design Tools',
    ko: '설계 도구',
    items: [
      {
        slug: 'color-filter',
        icon: '🎨',
        en: { title: 'Color Filter Designer', desc: 'Shape RGB filter spectra and see the CIE 1931 gamut.' },
        ko: { title: '컬러 필터 설계', desc: 'RGB 필터 스펙트럼을 설계하고 CIE 1931 색역 확인.' },
      },
      {
        slug: 'pixel-playground',
        icon: '🧪',
        en: { title: 'Pixel Design Playground', desc: 'Every BSI pixel parameter in one multi-panel designer.' },
        ko: { title: '픽셀 설계 놀이터', desc: 'BSI 픽셀의 모든 파라미터를 한 화면에서 조정.' },
      },
    ],
  },
  {
    id: 'optics',
    en: 'Optics & Wave Physics',
    ko: '광학 & 파동 물리',
    items: [
      {
        slug: 'si-absorption',
        icon: '🔻',
        en: { title: 'Si Absorption Depth', desc: 'Beer–Lambert penetration depth across the spectrum.' },
        ko: { title: 'Si 흡수 깊이', desc: 'Beer–Lambert 법칙으로 파장별 침투 깊이 시각화.' },
      },
      {
        slug: 'microlens-raytrace',
        icon: '🔦',
        en: { title: 'Microlens Ray Trace', desc: 'Superellipse microlens focusing, spot size and CRA shift.' },
        ko: { title: '마이크로렌즈 광선추적', desc: '초타원 마이크로렌즈의 집광·스팟 크기·CRA 효과.' },
      },
      {
        slug: 'microlens-process-shape',
        icon: '🫧',
        en: { title: 'Microlens Process Shape', desc: 'Predict gap, height and profile from reflow and etch.' },
        ko: { title: '마이크로렌즈 공정 형상', desc: '리플로우·식각 조건으로 갭·높이·프로파일 예측.' },
      },
      {
        slug: 'mla-array',
        icon: '🔵',
        en: { title: 'MLA Array Visualizer', desc: 'Microlens array contours, cross-sections and 3D surfaces.' },
        ko: { title: 'MLA 어레이 시각화', desc: 'MLA 등고선·단면·3D 표면 시각화.' },
      },
      {
        slug: 'fdti-pixel',
        icon: '🧊',
        en: { title: 'FDTI / BDTI Pixel', desc: 'Trench type, depth and liner effects on confinement.' },
        ko: { title: 'FDTI / BDTI 픽셀', desc: '트렌치 종류·깊이·라이너가 광 구속에 주는 영향.' },
      },
      {
        slug: 'fabry-perot',
        icon: '🌈',
        en: { title: 'Fabry-Pérot Visualizer', desc: 'Thin-film interference explained with phasor diagrams.' },
        ko: { title: '파브리-페로 시각화', desc: '페이저 다이어그램으로 보는 박막 간섭.' },
      },
      {
        slug: 'diffraction-psf',
        icon: '🎯',
        en: { title: 'Diffraction PSF', desc: 'Airy pattern on the pixel grid and encircled energy.' },
        ko: { title: '회절 PSF', desc: '픽셀 격자 위의 Airy 패턴과 집중 에너지.' },
      },
      {
        slug: 'lens-shading',
        icon: '🌘',
        en: { title: 'Lens Shading', desc: 'Relative illumination, cos⁴ fall-off and color shading.' },
        ko: { title: '렌즈 쉐이딩', desc: '상대 조도, cos⁴ 감쇠, 채널별 색 쉐이딩.' },
      },
    ],
  },
  {
    id: 'system',
    en: 'System Analysis',
    ko: '시스템 분석',
    items: [
      {
        slug: 'mtf-analyzer',
        icon: '🔍',
        en: { title: 'MTF Analyzer', desc: 'Aperture, diffraction and defocus MTF up to Nyquist.' },
        ko: { title: 'MTF 분석기', desc: '개구·회절·디포커스 MTF와 나이퀴스트 분석.' },
      },
      {
        slug: 'pixel-scaling',
        icon: '📉',
        en: { title: 'Pixel Scaling Trends', desc: 'How FWC, SNR and diffraction scale with pitch.' },
        ko: { title: '픽셀 스케일링', desc: '피치 축소에 따른 FWC·SNR·회절 한계 변화.' },
      },
      {
        slug: 'color-accuracy',
        icon: '🟩',
        en: { title: 'Color Accuracy (ΔE)', desc: 'CCM fit and ΔE on ColorChecker patches.' },
        ko: { title: '색 정확도 (ΔE)', desc: 'ColorChecker 패치의 CCM 계산과 ΔE 평가.' },
      },
      {
        slug: 'dark-current',
        icon: '🌡️',
        en: { title: 'Dark Current & Temp', desc: 'Arrhenius dark current and its image impact.' },
        ko: { title: '암전류 & 온도', desc: 'Arrhenius 암전류 모델과 화질 영향.' },
      },
      {
        slug: 'linearity-analyzer',
        icon: '📏',
        en: { title: 'Linearity Analyzer', desc: 'Transfer-curve non-linearity, knee point, residuals.' },
        ko: { title: '선형성 분석기', desc: '전달 곡선 비선형성·니 포인트·잔차 분석.' },
      },
      {
        slug: 'emva1288',
        icon: '📊',
        en: { title: 'EMVA 1288 Dashboard', desc: 'All key EMVA 1288 parameters from pixel inputs.' },
        ko: { title: 'EMVA 1288 대시보드', desc: '픽셀 입력으로 EMVA 1288 핵심 파라미터 계산.' },
      },
      {
        slug: 'prnu-visualizer',
        icon: '🧩',
        en: { title: 'PRNU / DSNU', desc: 'Fixed pattern noise maps and distributions.' },
        ko: { title: 'PRNU / DSNU', desc: '고정 패턴 노이즈의 공간 분포와 통계.' },
      },
    ],
  },
]

export const simulatorCount = simulatorCategories.reduce((n, c) => n + c.items.length, 0)

/** Build the simulator sidebar for a locale ('' for English, '/ko' for Korean). */
export function simulatorSidebar(lang: 'en' | 'ko') {
  const prefix = lang === 'ko' ? '/ko' : ''
  return [
    { text: lang === 'ko' ? '개요' : 'Overview', link: `${prefix}/simulator/` },
    ...simulatorCategories.map((cat) => ({
      text: cat[lang],
      collapsed: false,
      items: cat.items.map((s) => ({ text: s[lang].title, link: `${prefix}/simulator/${s.slug}` })),
    })),
  ]
}
