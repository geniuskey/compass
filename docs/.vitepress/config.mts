import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'COMPASS',
  description: 'Cross-solver Optical Modeling Platform for Advanced Sensor Simulation',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/installation' },
      { text: 'Theory', link: '/theory/light-basics' },
      { text: 'Reference', link: '/reference/api-overview' },
      { text: 'Cookbook', link: '/cookbook/bsi-2x2-basic' },
    ],
    sidebar: {
      '/theory/': [
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
      ],
      '/guide/': [
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
      ],
      '/reference/': [
        { text: 'API Reference', items: [
          { text: 'Overview', link: '/reference/api-overview' },
          { text: 'PixelStack', link: '/reference/pixel-stack' },
          { text: 'MaterialDB', link: '/reference/material-db' },
          { text: 'SolverBase', link: '/reference/solver-base' },
          { text: 'Sources', link: '/reference/sources' },
          { text: 'Analysis', link: '/reference/analysis' },
          { text: 'Config Reference', link: '/reference/config-reference' },
        ]}
      ],
      '/cookbook/': [
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
      ],
      '/about/': [
        { text: 'About', items: [
          { text: 'References', link: '/about/references' },
          { text: 'Changelog', link: '/about/changelog' },
          { text: 'Roadmap', link: '/about/roadmap' },
          { text: 'License', link: '/about/license' },
          { text: 'Contributing', link: '/about/contributing' },
        ]}
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/compass-sim/compass' }
    ],
    search: { provider: 'local' },
  },
  markdown: {
    math: true
  }
})
