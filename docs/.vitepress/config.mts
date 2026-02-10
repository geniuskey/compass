import { defineConfig } from 'vitepress'

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
    { text: 'ROI Sweep', link: '/guide/roi-sweep' },
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
    { text: 'Microlens Optimization', link: '/cookbook/microlens-optimization' },
    { text: 'Wavelength Sweep', link: '/cookbook/wavelength-sweep' },
    { text: 'Solver Benchmark', link: '/cookbook/solver-benchmark' },
    { text: 'CRA Shift Analysis', link: '/cookbook/cra-shift-analysis' },
    { text: 'BARL Design', link: '/cookbook/barl-design' },
    { text: 'DTI Crosstalk', link: '/cookbook/dti-crosstalk' },
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
    { text: 'ROI 스윕', link: '/ko/guide/roi-sweep' },
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
    { text: '마이크로렌즈 최적화', link: '/ko/cookbook/microlens-optimization' },
    { text: '파장 스윕', link: '/ko/cookbook/wavelength-sweep' },
    { text: '솔버 벤치마크', link: '/ko/cookbook/solver-benchmark' },
    { text: 'CRA 시프트 분석', link: '/ko/cookbook/cra-shift-analysis' },
    { text: 'BARL 설계', link: '/ko/cookbook/barl-design' },
    { text: 'DTI 크로스토크', link: '/ko/cookbook/dti-crosstalk' },
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

export default defineConfig({
  base: '/compass/',
  title: 'COMPASS',
  description: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation',

  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: 'Guide', link: '/guide/installation' },
          { text: 'Theory', link: '/theory/light-basics' },
          { text: 'Reference', link: '/reference/api-overview' },
          { text: 'Cookbook', link: '/cookbook/bsi-2x2-basic' },
        ],
        sidebar: {
          '/theory/': theorySidebar,
          '/guide/': guideSidebar,
          '/reference/': referenceSidebar,
          '/cookbook/': cookbookSidebar,
          '/about/': aboutSidebar,
        },
      },
    },
    ko: {
      label: '한국어',
      lang: 'ko',
      themeConfig: {
        nav: [
          { text: '가이드', link: '/ko/guide/installation' },
          { text: '이론', link: '/ko/theory/light-basics' },
          { text: '레퍼런스', link: '/ko/reference/api-overview' },
          { text: '요리책', link: '/ko/cookbook/bsi-2x2-basic' },
        ],
        sidebar: {
          '/ko/theory/': theorySidebarKo,
          '/ko/guide/': guideSidebarKo,
          '/ko/reference/': referenceSidebarKo,
          '/ko/cookbook/': cookbookSidebarKo,
          '/ko/about/': aboutSidebarKo,
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
  }
})
