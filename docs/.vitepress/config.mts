import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const introSidebar = [
  { text: 'Introduction', items: [
    { text: 'What is a CMOS Image Sensor?', link: '/introduction/what-is-cmos-sensor' },
    { text: 'Optics Primer', link: '/introduction/optics-primer' },
    { text: 'Pixel Anatomy', link: '/introduction/pixel-anatomy' },
    { text: 'Understanding QE', link: '/introduction/qe-intuitive' },
  ]}
]

const introSidebarKo = [
  { text: '소개', items: [
    { text: 'CMOS 이미지 센서란?', link: '/ko/introduction/what-is-cmos-sensor' },
    { text: '광학 기초 입문', link: '/ko/introduction/optics-primer' },
    { text: '픽셀 해부학', link: '/ko/introduction/pixel-anatomy' },
    { text: '양자 효율 이해', link: '/ko/introduction/qe-intuitive' },
  ]}
]

const theorySidebar = [
  { text: 'Theory', items: [
    { text: 'Light Basics', link: '/theory/light-basics' },
    { text: 'Electromagnetic Waves', link: '/theory/electromagnetic-waves' },
    { text: 'Thin Film Optics', link: '/theory/thin-film-optics' },
    { text: 'Diffraction', link: '/theory/diffraction' },
    { text: 'RCWA Explained', link: '/theory/rcwa-explained' },
    { text: 'FDTD Explained', link: '/theory/fdtd-explained' },
    { text: 'RCWA vs FDTD', link: '/theory/rcwa-vs-fdtd' },
    { text: 'Image Sensor Optics', link: '/theory/image-sensor-optics' },
    { text: 'Quantum Efficiency', link: '/theory/quantum-efficiency' },
    { text: 'Signal Chain', link: '/theory/signal-chain' },
    { text: 'Numerical Stability', link: '/theory/numerical-stability' },
  ]}
]

const guideSidebar = [
  { text: 'Getting Started', items: [
    { text: 'Installation', link: '/guide/installation' },
    { text: 'Quick Start', link: '/guide/quickstart' },
    { text: 'First Simulation', link: '/guide/first-simulation' },
  ]},
  { text: 'Configuration', items: [
    { text: 'Pixel Stack Config', link: '/guide/pixel-stack-config' },
    { text: 'Material Database', link: '/guide/material-database' },
    { text: 'Choosing a Solver', link: '/guide/choosing-solver' },
  ]},
  { text: 'Running Solvers', items: [
    { text: 'Running RCWA', link: '/guide/running-rcwa' },
    { text: 'Running FDTD', link: '/guide/running-fdtd' },
    { text: 'Cross-Validation', link: '/guide/cross-validation' },
  ]},
  { text: 'Advanced', items: [
    { text: 'Cone Illumination', link: '/guide/cone-illumination' },
    { text: 'Signal Simulation', link: '/guide/signal-simulation' },
    { text: 'ROI Sweep', link: '/guide/roi-sweep' },
    { text: 'Inverse Design', link: '/guide/inverse-design' },
    { text: 'Visualization', link: '/guide/visualization' },
    { text: 'Troubleshooting', link: '/guide/troubleshooting' },
  ]},
]

const referenceSidebar = [
  { text: 'API Reference', items: [
    { text: 'Overview', link: '/reference/api-overview' },
    { text: 'PixelStack', link: '/reference/pixel-stack' },
    { text: 'MaterialDB', link: '/reference/material-db' },
    { text: 'SolverBase', link: '/reference/solver-base' },
    { text: 'Sources', link: '/reference/sources' },
    { text: 'Analysis', link: '/reference/analysis' },
    { text: 'Config Reference', link: '/reference/config-reference' },
    { text: 'Glossary', link: '/reference/glossary' },
  ]}
]

const cookbookSidebar = [
  { text: 'Recipes', items: [
    { text: 'BSI 2x2 Basic', link: '/cookbook/bsi-2x2-basic' },
    { text: 'Metal Grid Effect', link: '/cookbook/metal-grid-effect' },
    { text: 'Microlens & CRA', link: '/cookbook/microlens-optimization' },
    { text: 'Wavelength Sweep', link: '/cookbook/wavelength-sweep' },
    { text: 'Solver Comparison', link: '/cookbook/solver-benchmark' },
    { text: 'BARL Design', link: '/cookbook/barl-design' },
    { text: 'DTI Crosstalk', link: '/cookbook/dti-crosstalk' },
    { text: 'Signal Chain Color Accuracy', link: '/cookbook/signal-chain-color-accuracy' },
    { text: 'TMM Validation Results', link: '/cookbook/tmm-validation-results' },
    { text: 'Convergence Study', link: '/cookbook/convergence-study' },
  ]}
]

const researchSidebar = [
  { text: 'Research', items: [
    { text: 'EM Solver Survey', link: '/research/open-source-em-solvers-survey' },
    { text: 'CIS Technology Trends', link: '/research/cis-technology-trends' },
    { text: 'Simulation Methods', link: '/research/simulation-methods-comparison' },
    { text: 'Key Papers', link: '/research/key-papers' },
    { text: 'Benchmarks & Validation', link: '/research/benchmarks-and-validation' },
  ]}
]

const aboutSidebar = [
  { text: 'About', items: [
    { text: 'References', link: '/about/references' },
    { text: 'Changelog', link: '/about/changelog' },
    { text: 'Roadmap', link: '/about/roadmap' },
    { text: 'License', link: '/about/license' },
    { text: 'Contributing', link: '/about/contributing' },
  ]}
]

// Korean sidebar translations
const theorySidebarKo = [
  { text: '이론', items: [
    { text: '빛의 기초', link: '/ko/theory/light-basics' },
    { text: '전자기파', link: '/ko/theory/electromagnetic-waves' },
    { text: '박막 광학', link: '/ko/theory/thin-film-optics' },
    { text: '회절', link: '/ko/theory/diffraction' },
    { text: 'RCWA 설명', link: '/ko/theory/rcwa-explained' },
    { text: 'FDTD 설명', link: '/ko/theory/fdtd-explained' },
    { text: 'RCWA vs FDTD', link: '/ko/theory/rcwa-vs-fdtd' },
    { text: '이미지 센서 광학', link: '/ko/theory/image-sensor-optics' },
    { text: '양자 효율', link: '/ko/theory/quantum-efficiency' },
    { text: '신호 체인', link: '/ko/theory/signal-chain' },
    { text: '수치 안정성', link: '/ko/theory/numerical-stability' },
  ]}
]

const guideSidebarKo = [
  { text: '시작하기', items: [
    { text: '설치', link: '/ko/guide/installation' },
    { text: '빠른 시작', link: '/ko/guide/quickstart' },
    { text: '첫 번째 시뮬레이션', link: '/ko/guide/first-simulation' },
  ]},
  { text: '설정', items: [
    { text: '픽셀 스택 구성', link: '/ko/guide/pixel-stack-config' },
    { text: '재료 데이터베이스', link: '/ko/guide/material-database' },
    { text: '솔버 선택', link: '/ko/guide/choosing-solver' },
  ]},
  { text: '솔버 실행', items: [
    { text: 'RCWA 실행', link: '/ko/guide/running-rcwa' },
    { text: 'FDTD 실행', link: '/ko/guide/running-fdtd' },
    { text: '교차 검증', link: '/ko/guide/cross-validation' },
  ]},
  { text: '고급', items: [
    { text: '원뿔 조명', link: '/ko/guide/cone-illumination' },
    { text: '신호 시뮬레이션', link: '/ko/guide/signal-simulation' },
    { text: 'ROI 스윕', link: '/ko/guide/roi-sweep' },
    { text: '역설계', link: '/ko/guide/inverse-design' },
    { text: '시각화', link: '/ko/guide/visualization' },
    { text: '문제 해결', link: '/ko/guide/troubleshooting' },
  ]},
]

const referenceSidebarKo = [
  { text: 'API 레퍼런스', items: [
    { text: '개요', link: '/ko/reference/api-overview' },
    { text: 'PixelStack', link: '/ko/reference/pixel-stack' },
    { text: 'MaterialDB', link: '/ko/reference/material-db' },
    { text: 'SolverBase', link: '/ko/reference/solver-base' },
    { text: 'Sources', link: '/ko/reference/sources' },
    { text: 'Analysis', link: '/ko/reference/analysis' },
    { text: '설정 레퍼런스', link: '/ko/reference/config-reference' },
    { text: '용어집', link: '/ko/reference/glossary' },
  ]}
]

const cookbookSidebarKo = [
  { text: '레시피', items: [
    { text: 'BSI 2x2 기본', link: '/ko/cookbook/bsi-2x2-basic' },
    { text: '메탈 그리드 효과', link: '/ko/cookbook/metal-grid-effect' },
    { text: '마이크로렌즈 & CRA', link: '/ko/cookbook/microlens-optimization' },
    { text: '파장 스윕', link: '/ko/cookbook/wavelength-sweep' },
    { text: '솔버 비교 가이드', link: '/ko/cookbook/solver-benchmark' },
    { text: 'BARL 설계', link: '/ko/cookbook/barl-design' },
    { text: 'DTI 크로스토크', link: '/ko/cookbook/dti-crosstalk' },
    { text: '신호 체인 색 정확도', link: '/ko/cookbook/signal-chain-color-accuracy' },
    { text: 'TMM 검증 결과', link: '/ko/cookbook/tmm-validation-results' },
    { text: '수렴 연구', link: '/ko/cookbook/convergence-study' },
  ]}
]

const researchSidebarKo = [
  { text: '리서치', items: [
    { text: 'EM 솔버 서베이', link: '/ko/research/open-source-em-solvers-survey' },
    { text: 'CIS 기술 동향', link: '/ko/research/cis-technology-trends' },
    { text: '시뮬레이션 방법론', link: '/ko/research/simulation-methods-comparison' },
    { text: '핵심 논문', link: '/ko/research/key-papers' },
    { text: '벤치마크 & 검증', link: '/ko/research/benchmarks-and-validation' },
  ]}
]

const aboutSidebarKo = [
  { text: '정보', items: [
    { text: '참고 문헌', link: '/ko/about/references' },
    { text: '변경 이력', link: '/ko/about/changelog' },
    { text: '로드맵', link: '/ko/about/roadmap' },
    { text: '라이선스', link: '/ko/about/license' },
    { text: '기여 가이드', link: '/ko/about/contributing' },
  ]}
]

const simulatorSidebar = [
  { text: 'Simulator', items: [
    { text: 'Overview', link: '/simulator/' },
    { text: 'TMM QE Calculator', link: '/simulator/tmm-qe' },
    { text: 'Thin Film Stack Designer', link: '/simulator/barl-optimizer' },
    { text: 'Energy Budget', link: '/simulator/energy-budget' },
    { text: 'Angular Response', link: '/simulator/angular-response' },
    { text: 'SNR Calculator', link: '/simulator/snr-calculator' },
    { text: 'Color Filter Designer', link: '/simulator/color-filter' },
    { text: 'Pixel Design Playground', link: '/simulator/pixel-playground' },
  ]}
]

const simulatorSidebarKo = [
  { text: '시뮬레이터', items: [
    { text: '개요', link: '/ko/simulator/' },
    { text: 'TMM QE 계산기', link: '/ko/simulator/tmm-qe' },
    { text: '박막 스택 설계기', link: '/ko/simulator/barl-optimizer' },
    { text: '에너지 버짓', link: '/ko/simulator/energy-budget' },
    { text: '각도 응답', link: '/ko/simulator/angular-response' },
    { text: 'SNR 계산기', link: '/ko/simulator/snr-calculator' },
    { text: '컬러 필터 설계', link: '/ko/simulator/color-filter' },
    { text: '픽셀 설계 놀이터', link: '/ko/simulator/pixel-playground' },
  ]}
]

export default withMermaid(defineConfig({
  base: '/compass/',
  title: 'COMPASS',
  description: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/compass/favicon.svg' }],
    ['meta', { property: 'og:title', content: 'COMPASS' }],
    ['meta', { property: 'og:description', content: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:image', content: '/compass/logo.svg' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    ['meta', { name: 'twitter:title', content: 'COMPASS' }],
    ['meta', { name: 'twitter:description', content: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation' }],
    ['meta', { name: 'theme-color', content: '#3451b2' }],
    ['meta', { name: 'keywords', content: 'RCWA, FDTD, image sensor, pixel simulation, quantum efficiency, optical simulation, CMOS' }],
  ],

  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Learn', items: [
            { text: 'Introduction', link: '/introduction/what-is-cmos-sensor' },
            { text: 'Theory', link: '/theory/light-basics' },
          ]},
          { text: 'Guide', items: [
            { text: 'Getting Started', link: '/guide/installation' },
            { text: 'Cookbook', link: '/cookbook/bsi-2x2-basic' },
          ]},
          { text: 'Simulator', link: '/simulator/' },
          { text: 'Reference', items: [
            { text: 'API Reference', link: '/reference/api-overview' },
            { text: 'Research', link: '/research/open-source-em-solvers-survey' },
            { text: 'About', link: '/about/references' },
          ]},
        ],
        sidebar: {
          '/introduction/': introSidebar,
          '/theory/': theorySidebar,
          '/guide/': guideSidebar,
          '/reference/': referenceSidebar,
          '/cookbook/': cookbookSidebar,
          '/research/': researchSidebar,
          '/about/': aboutSidebar,
          '/simulator/': simulatorSidebar,
        },
      },
    },
    ko: {
      label: '한국어',
      lang: 'ko',
      themeConfig: {
        nav: [
          { text: '학습', items: [
            { text: '소개', link: '/ko/introduction/what-is-cmos-sensor' },
            { text: '이론', link: '/ko/theory/light-basics' },
          ]},
          { text: '가이드', items: [
            { text: '시작하기', link: '/ko/guide/installation' },
            { text: '레시피', link: '/ko/cookbook/bsi-2x2-basic' },
          ]},
          { text: '시뮬레이터', link: '/ko/simulator/' },
          { text: '레퍼런스', items: [
            { text: 'API 레퍼런스', link: '/ko/reference/api-overview' },
            { text: '리서치', link: '/ko/research/open-source-em-solvers-survey' },
            { text: '정보', link: '/ko/about/references' },
          ]},
        ],
        sidebar: {
          '/ko/introduction/': introSidebarKo,
          '/ko/theory/': theorySidebarKo,
          '/ko/guide/': guideSidebarKo,
          '/ko/reference/': referenceSidebarKo,
          '/ko/cookbook/': cookbookSidebarKo,
          '/ko/research/': researchSidebarKo,
          '/ko/about/': aboutSidebarKo,
          '/ko/simulator/': simulatorSidebarKo,
        },
        docFooter: { prev: '이전', next: '다음' },
        outline: { label: '목차' },
        lastUpdated: { text: '최종 수정' },
        returnToTopLabel: '맨 위로',
        sidebarMenuLabel: '메뉴',
        darkModeSwitchLabel: '다크 모드',
        langMenuLabel: '언어 변경',
      },
    },
  },

  themeConfig: {
    socialLinks: [
      { icon: 'github', link: 'https://github.com/geniuskey/compass' }
    ],
    search: { provider: 'local' },
  },

  markdown: {
    math: true
  },

  mermaid: {
    theme: 'default',
  },
}))
