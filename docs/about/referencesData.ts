export const referencesEn = [
  {
    id: "moharam_1995_stable",
    category: "RCWA Theory",
    authors: "M.G. Moharam, E.B. Grann, D.A. Pommet, and T.K. Gaylord",
    title: "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings",
    journal: "J. Opt. Soc. Am. A",
    year: "1995",
    link: "https://doi.org/10.1364/JOSAA.12.001068",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Diffraction_grating_principle_1.svg/512px-Diffraction_grating_principle_1.svg.png",
    usedIn: [
      { label: "RCWA Explained", href: "/compass/theory/simulation/rcwa-explained" },
      { label: "Numerical Stability", href: "/compass/theory/simulation/numerical-stability" },
      { label: "Key Papers", href: "/compass/research/key-papers" }
    ],
    summary: `
      <p>This landmark paper introduces a numerically stable formulation for Rigorous Coupled-Wave Analysis (RCWA) when applied to binary gratings. Prior to this work, RCWA suffered from severe numerical instabilities, particularly when calculating thick gratings or highly conductive materials due to the unbounded growth of evanescent wave components (exponentially growing terms) in the eigenvalue decomposition.</p>
      <ul>
        <li><strong>Key Innovation:</strong> The authors introduced a state-variable method and a normalized formulation that scales the eigen-modes, thereby eliminating the numerical overflow caused by these evanescent waves.</li>
        <li><strong>Impact:</strong> It transformed RCWA from a theoretically interesting but practically limited method into a robust, standard tool for simulating subwavelength optical structures.</li>
      </ul>
      <p>In COMPASS, this formulation is the bedrock of our RCWA solvers (like <code>torcwa</code> and <code>meent</code>). It guarantees that when we simulate deep sub-micron structures like Deep Trench Isolation (DTI) or thick color filters, the simulation remains stable and energy conservation is maintained.</p>
    `
  },
  {
    id: "li_1996_fourier",
    category: "Fourier Factorization (Li Rules)",
    authors: "L. Li",
    title: "Use of Fourier series in the analysis of discontinuous periodic structures",
    journal: "J. Opt. Soc. Am. A",
    year: "1996",
    link: "https://doi.org/10.1364/JOSAA.13.001870",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Fourier_transform_time_and_frequency_domains_%28small%29.gif/512px-Fourier_transform_time_and_frequency_domains_%28small%29.gif",
    usedIn: [
      { label: "RCWA Explained", href: "/compass/theory/simulation/rcwa-explained" },
      { label: "RCWA vs FDTD", href: "/compass/theory/simulation/rcwa-vs-fdtd" },
      { label: "Key Papers", href: "/compass/research/key-papers" }
    ],
    summary: `
      <p>This paper resolved a critical, long-standing issue in computational electromagnetics: the slow or absent convergence of Fourier expansions at material interfaces with high index contrast. Li demonstrated that concurrently expanding discontinuous functions requires specific mathematical rules.</p>
      <ul>
        <li><strong>Laurent's Rule (Inverse Rule):</strong> Li proved that when multiplying two discontinuous functions with concurrent jump discontinuities, one must invert the Toeplitz matrix of the reciprocal of one function before multiplying.</li>
        <li><strong>Impact:</strong> Utilizing Li's factorization rules drastically improves the convergence rate of RCWA for TM (Transverse Magnetic) polarization and 2D crossed gratings, reducing the required number of Fourier harmonics.</li>
      </ul>
      <p>In COMPASS, Li's rules are strictly implemented in all RCWA backends. This is why our solvers can accurately model sharp metallic grid edges or high-index silicon/oxide boundaries in CMOS pixels using fewer Fourier orders, saving significant GPU memory and computation time.</p>
    `
  },
  {
    id: "li_1996_smatrix",
    category: "S-Matrix Algorithm",
    authors: "L. Li",
    title: "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings",
    journal: "J. Opt. Soc. Am. A",
    year: "1996",
    link: "https://doi.org/10.1364/JOSAA.13.001024",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Scattering_Matrix_Port_Definition.svg/512px-Scattering_Matrix_Port_Definition.svg.png",
    usedIn: [
      { label: "RCWA Explained", href: "/compass/theory/simulation/rcwa-explained" },
      { label: "Solver Benchmark", href: "/compass/cookbook/solver-benchmark" },
      { label: "Key Papers", href: "/compass/research/key-papers" }
    ],
    summary: `
      <p>While Moharam addressed single-layer stability, multi-layered structures still suffered from instabilities when cascading Transmission (T) matrices. Li comprehensively evaluated recursive matrix algorithms and championed the Scattering Matrix (S-Matrix) approach.</p>
      <ul>
        <li><strong>The S-Matrix Method:</strong> Instead of relating fields at the input to the output (which involves exponentially growing terms), the S-matrix relates incoming waves (to a layer) to outgoing waves (from the layer). This inherently bounds all matrix elements.</li>
        <li><strong>Stability:</strong> It guarantees unconditional numerical stability for an arbitrary number of layers and total grating thickness.</li>
      </ul>
      <p>Every multi-layer pixel stack in COMPASS (from microlens down to the photodiode) is solved by recursively cascading S-matrices. This allows us to simulate massive 3D stacks with hundreds of sliced layers (e.g., staircase approximations of curved microlenses) without any numerical blow-up.</p>
    `
  },
  {
    id: "yee_1966_fdtd",
    category: "FDTD Method",
    authors: "K.S. Yee",
    title: "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media",
    journal: "IEEE Trans. Antennas Propag.",
    year: "1966",
    link: "https://doi.org/10.1109/TAP.1966.1138693",
    usedIn: [
      { label: "FDTD Explained", href: "/compass/theory/simulation/fdtd-explained" },
      { label: "RCWA vs FDTD", href: "/compass/theory/simulation/rcwa-vs-fdtd" },
      { label: "Key Papers", href: "/compass/research/key-papers" }
    ],
    summary: `
      <p>This is the foundational paper for the Finite-Difference Time-Domain (FDTD) method, introducing the legendary "Yee Cell" grid.</p>
      <ul>
        <li><strong>The Yee Grid:</strong> Yee proposed staggering the Electric (E) and Magnetic (H) field components in both space and time. E-fields are evaluated at the edges of a cube, while H-fields are evaluated at the faces.</li>
        <li><strong>Leapfrog Integration:</strong> The method updates E-fields and H-fields in alternating time steps, perfectly mirroring the curl equations of Maxwell's theory.</li>
      </ul>
      <p>In COMPASS, our FDTD engines (like <code>flaport</code> and references to <code>Meep</code>) strictly adhere to the Yee grid architecture. This ensures divergence-free fields and robust, accurate time-domain simulations of broadband optical pulses interacting with sub-wavelength pixel structures.</p>
    `
  },
  {
    id: "el_gamal_2005",
    category: "Image Sensor Physics",
    authors: "A. El Gamal and H. Eltoukhy",
    title: "CMOS image sensors",
    journal: "IEEE Circuits and Devices Magazine",
    year: "2005",
    link: "https://doi.org/10.1109/MCD.2005.1438751",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Active_Pixel_Sensor.svg/512px-Active_Pixel_Sensor.svg.png",
    usedIn: [
      { label: "Noise and SNR", href: "/compass/theory/sensor/noise-and-snr" },
      { label: "EMVA 1288 Dashboard", href: "/compass/simulator/emva1288" },
      { label: "SNR Calculator", href: "/compass/simulator/snr-calculator" }
    ],
    summary: `
      <p>A comprehensive tutorial and review of CMOS Active Pixel Sensor (APS) technology that marked the transition from CCDs to CMOS.</p>
      <ul>
        <li><strong>Sensor Architecture:</strong> Explains the 3T and 4T pixel architectures, correlated double sampling (CDS), and column-parallel readout.</li>
        <li><strong>Noise & SNR:</strong> Details the fundamental noise sources in CMOS pixels (read noise, dark current, photon shot noise) and how they limit the signal-to-noise ratio.</li>
      </ul>
      <p>COMPASS leverages these principles in its <code>SignalChainDiagram</code> and SNR calculators. While COMPASS focuses heavily on optics, understanding the downstream electronic conversion (quantum efficiency to digital numbers) is crucial for our end-to-end pixel modeling.</p>
    `
  },
  {
    id: "macleod_2017_thinfilm",
    category: "Thin-Film Optics",
    authors: "H.A. Macleod",
    title: "Thin-Film Optical Filters",
    journal: "CRC Press",
    year: "2017",
    link: "https://www.routledge.com/Thin-Film-Optical-Filters/Macleod/p/book/9781138198241",
    usedIn: [
      { label: "TMM QE Calculator", href: "/compass/simulator/tmm-qe" },
      { label: "BARL Optimizer", href: "/compass/simulator/barl-optimizer" },
      { label: "Thin-Film Optics", href: "/compass/theory/optics/thin-film-optics" }
    ],
    summary: `
      <p>This book is the practical reference behind the characteristic-matrix view of multilayer coatings. It explains optical admittance, phase thickness, reflection minima, and how coating stacks trade bandwidth, angle, polarization, and manufacturability.</p>
      <p>In COMPASS, it anchors the TMM and BARL browser tools: those pages use the same coherent-film logic to turn layer thickness and refractive index into reflection, transmission, and absorption trends.</p>
    `
  },
  {
    id: "green_2008_silicon",
    category: "Material Optical Data",
    authors: "M.A. Green",
    title: "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients",
    journal: "Solar Energy Materials and Solar Cells",
    year: "2008",
    link: "https://doi.org/10.1016/j.solmat.2008.06.009",
    usedIn: [
      { label: "Si Absorption", href: "/compass/simulator/si-absorption" },
      { label: "TMM QE Calculator", href: "/compass/simulator/tmm-qe" },
      { label: "Quantum Efficiency", href: "/compass/theory/sensor/quantum-efficiency" }
    ],
    summary: `
      <p>Green provides a self-consistent set of intrinsic silicon optical constants across wavelength, including temperature coefficients. This is one of the most useful public sources for turning wavelength into absorption depth in silicon.</p>
      <p>COMPASS uses this reference whenever a simplified browser model needs a physically grounded silicon absorption trend rather than arbitrary visible-band attenuation.</p>
    `
  },
  {
    id: "catrysse_2002_pixels",
    category: "Image Sensor Optics",
    authors: "P.B. Catrysse and B.A. Wandell",
    title: "Optical efficiency of image sensor pixels",
    journal: "J. Opt. Soc. Am. A",
    year: "2002",
    link: "https://doi.org/10.1364/JOSAA.19.001610",
    usedIn: [
      { label: "Quantum Efficiency", href: "/compass/theory/sensor/quantum-efficiency" },
      { label: "Pixel Optical Effects", href: "/compass/theory/sensor/pixel-optical-effects" },
      { label: "TMM QE Calculator", href: "/compass/simulator/tmm-qe" }
    ],
    summary: `
      <p>This paper connects optical simulation to image-sensor pixel efficiency: incident optical power must be mapped to absorbed power in the intended photosensitive volume, and that mapping depends on stack geometry and material losses.</p>
      <p>It is the conceptual bridge between electromagnetic stack calculations and the optical QE proxies used throughout the COMPASS simulator pages.</p>
    `
  },
  {
    id: "ristoiu_2020_microlens_doe",
    category: "Microlens Process",
    authors: "Ristoiu et al.",
    title: "A DOE study of plasma etched microlens shape for CMOS image sensors",
    journal: "SPIE",
    year: "2020",
    link: "https://doi.org/10.1117/12.2551857",
    usedIn: [
      { label: "Microlens Process Shape", href: "/compass/simulator/microlens-process-shape" },
      { label: "Microlens Ray Trace", href: "/compass/simulator/microlens-raytrace" },
      { label: "Microlens Optimization", href: "/compass/cookbook/microlens-optimization" }
    ],
    summary: `
      <p>This is the closest public CIS-specific process reference for the layout-to-final-shape question. It studies reflowed microlenses transferred by plasma etch and models final gap and height evolution across process variables.</p>
      <p>COMPASS uses it to justify exposing mask thickness, polymerizing gas, and etch time as first-class controls in the microlens process-shape surrogate.</p>
    `
  },
  {
    id: "baillie_2004_zero_space",
    category: "Microlens Process",
    authors: "Baillie and Gendler",
    title: "Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development",
    journal: "SPIE",
    year: "2004",
    link: "https://doi.org/10.1117/12.533453",
    usedIn: [
      { label: "Microlens Process Shape", href: "/compass/simulator/microlens-process-shape" },
      { label: "MLA Array Visualizer", href: "/compass/simulator/mla-array" },
      { label: "Image Sensor Optics", href: "/compass/theory/sensor/image-sensor-optics" }
    ],
    summary: `
      <p>This paper frames the zero-space microlens fabrication problem: residual lens spacing reduces optical fill factor, but insufficient lithographic spacing can cause lens merger during reflow.</p>
      <p>That tradeoff is exactly why the COMPASS process-shape tool reports both final gap and merger/over-etch warnings instead of treating zero gap as automatically optimal.</p>
    `
  },
  {
    id: "hwang_2023_stack_alignment",
    category: "Microlens and CRA",
    authors: "Hwang and Kim",
    title: "A Numerical Method of Aligning the Optical Stacks for All Pixels",
    journal: "Sensors",
    year: "2023",
    link: "https://doi.org/10.3390/s23020702",
    usedIn: [
      { label: "Microlens Ray Trace", href: "/compass/simulator/microlens-raytrace" },
      { label: "Lens Shading", href: "/compass/simulator/lens-shading" },
      { label: "Cone Illumination", href: "/compass/guide/cone-illumination" }
    ],
    summary: `
      <p>This work motivates numerical chief-ray alignment across the optical stack. For sensor-edge pixels, the microlens position and stack geometry must be matched to the chief ray so the focused spot lands on the intended photodiode.</p>
      <p>COMPASS uses this idea in CRA-shift explanations, microlens ray tracing, and lens-shading discussions.</p>
    `
  },
  {
    id: "cie_2018_colorimetry",
    category: "Color Science",
    authors: "Commission Internationale de l'Eclairage",
    title: "Colorimetry, 4th Edition",
    journal: "CIE 015:2018",
    year: "2018",
    link: "https://www.cie.co.at/publications/colorimetry-4th-edition",
    usedIn: [
      { label: "Color Reproduction", href: "/compass/theory/sensor/color-reproduction" },
      { label: "Color Filter Designer", href: "/compass/simulator/color-filter" },
      { label: "Color Accuracy Analyzer", href: "/compass/simulator/color-accuracy" }
    ],
    summary: `
      <p>This CIE technical report is the reference behind the XYZ, chromaticity, Lab, standard observer, illuminant, and color-difference definitions used by the COMPASS color pages.</p>
      <p>It anchors the distinction between sensor-dependent camera RGB and standard colorimetric coordinates.</p>
    `
  },
  {
    id: "iec_61966_2_1_srgb",
    category: "Color Science",
    authors: "IEC",
    title: "Multimedia systems and equipment - Colour measurement and management - Part 2-1: Default RGB colour space - sRGB",
    journal: "IEC 61966-2-1:1999",
    year: "1999",
    link: "https://webstore.iec.ch/en/publication/6169",
    usedIn: [
      { label: "Color Reproduction", href: "/compass/theory/sensor/color-reproduction" },
      { label: "Color Accuracy Analyzer", href: "/compass/simulator/color-accuracy" }
    ],
    summary: `
      <p>The sRGB standard defines the display-referred RGB space and nonlinear encoding used for web and common image interchange.</p>
      <p>COMPASS uses it as the endpoint after sensor RGB has been mapped through XYZ, not as a substitute for raw camera RGB.</p>
    `
  },
  {
    id: "sharma_2005_ciede2000",
    category: "Color Science",
    authors: "G. Sharma, W. Wu, and E.N. Dalal",
    title: "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations",
    journal: "Color Research & Application",
    year: "2005",
    link: "https://doi.org/10.1002/col.20070",
    usedIn: [
      { label: "Color Reproduction", href: "/compass/theory/sensor/color-reproduction" },
      { label: "Color Accuracy Analyzer", href: "/compass/simulator/color-accuracy" },
      { label: "Signal Chain Color Accuracy", href: "/compass/cookbook/signal-chain-color-accuracy" }
    ],
    summary: `
      <p>This paper is the practical implementation reference for CIEDE2000 color difference calculations.</p>
      <p>It is relevant to COMPASS because camera color optimization needs a perceptual error metric after the spectral response is mapped into a standard color space.</p>
    `
  },
  {
    id: "nist_2002_color_by_numbers",
    category: "Color Science",
    authors: "S.W. Brown and Y. Ohno",
    title: "Color By Numbers: Using a Calibration Source Spectrally Matched To Your Test Source Is Key To Measurement Accuracy",
    journal: "NIST",
    year: "2002",
    link: "https://www.nist.gov/publications/color-numbers-using-calibration-source-spectrally-matched-your-test-source-key",
    usedIn: [
      { label: "Color Reproduction", href: "/compass/theory/sensor/color-reproduction" },
      { label: "Color Filter Designer", href: "/compass/simulator/color-filter" }
    ],
    summary: `
      <p>This NIST note explains why spectral responsivity mismatch changes color measurement error when the measured spectra differ from the calibration source.</p>
      <p>That principle is directly analogous to CIS color design: the camera channel responses must be evaluated against realistic illuminants and object spectra, not only against ideal RGB endpoints.</p>
    `
  }
];
