export const referencesKo = [
  {
    id: "moharam_1995_stable",
    category: "RCWA 이론 (RCWA Theory)",
    authors: "M.G. Moharam, E.B. Grann, D.A. Pommet, and T.K. Gaylord",
    title: "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings",
    journal: "J. Opt. Soc. Am. A",
    year: "1995",
    link: "https://doi.org/10.1364/JOSAA.12.001068",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Diffraction_grating_principle_1.svg/512px-Diffraction_grating_principle_1.svg.png",
    usedIn: [
      { label: "RCWA 설명", href: "/compass/ko/theory/simulation/rcwa-explained" },
      { label: "수치 안정성", href: "/compass/ko/theory/simulation/numerical-stability" },
      { label: "핵심 논문", href: "/compass/ko/research/key-papers" }
    ],
    summary: `
      <p>이 기념비적인 논문은 이진 회절 격자에 적용될 때 엄밀 결합파 해석(RCWA)의 수치적으로 안정적인 공식을 처음으로 소개했습니다. 이 연구 이전에는, 고유값 분해 과정에서 기하급수적으로 증가하는 근접장(evanescent wave) 성분 때문에 두꺼운 격자나 전도성이 높은 물질을 계산할 때 심각한 수치적 불안정성이 발생했습니다.</p>
      <ul>
        <li><strong>핵심 혁신:</strong> 상태 변수(state-variable) 방식과 고유 모드의 크기를 적절히 조절하는 정규화된 공식을 도입하여, 근접장에 의한 수치적 오버플로우를 완전히 제거했습니다.</li>
        <li><strong>영향력:</strong> 이론적으로는 흥미롭지만 실제 사용에는 제약이 많았던 RCWA를 오늘날 서브파장 광학 구조 시뮬레이션의 강력한 표준 도구로 탈바꿈시켰습니다.</li>
      </ul>
      <p>COMPASS에서 이 공식은 <code>torcwa</code>, <code>meent</code> 등 핵심 RCWA 솔버의 기반이 됩니다. DTI(Deep Trench Isolation)나 두꺼운 컬러 필터와 같은 깊은 서브마이크론 픽셀 구조를 시뮬레이션할 때 시뮬레이션이 발산하지 않고 에너지 보존 법칙을 완벽히 지키도록 보장합니다.</p>
    `
  },
  {
    id: "li_1996_fourier",
    category: "푸리에 분해 (Li's Rules)",
    authors: "L. Li",
    title: "Use of Fourier series in the analysis of discontinuous periodic structures",
    journal: "J. Opt. Soc. Am. A",
    year: "1996",
    link: "https://doi.org/10.1364/JOSAA.13.001870",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Fourier_transform_time_and_frequency_domains_%28small%29.gif/512px-Fourier_transform_time_and_frequency_domains_%28small%29.gif",
    usedIn: [
      { label: "RCWA 설명", href: "/compass/ko/theory/simulation/rcwa-explained" },
      { label: "RCWA vs FDTD", href: "/compass/ko/theory/simulation/rcwa-vs-fdtd" },
      { label: "핵심 논문", href: "/compass/ko/research/key-papers" }
    ],
    summary: `
      <p>이 논문은 굴절률 차이가 큰 물질의 경계면에서 푸리에 급수가 매우 느리게 수렴하거나 아예 수렴하지 않는 전산 전자기학의 치명적이고 오래된 난제를 해결했습니다. Li는 불연속 함수들을 동시에 전개할 때 반드시 지켜야 할 특수한 수학적 규칙들을 증명했습니다.</p>
      <ul>
        <li><strong>로랑 규칙 (역행렬 규칙):</strong> 점프 불연속성을 가진 두 불연속 함수를 곱할 때, 단순 곱셈이 아니라 한 함수의 역수에 대한 톱리츠(Toeplitz) 행렬을 구한 뒤 이를 다시 역행렬로 취해 곱해야 한다는 것을 증명했습니다.</li>
        <li><strong>영향력:</strong> Li의 규칙을 사용하면 TM 편광 및 2D 교차 격자에 대한 RCWA의 수렴 속도가 극적으로 향상되어, 필요한 푸리에 차수(Fourier order)를 크게 줄일 수 있습니다.</li>
      </ul>
      <p>COMPASS의 모든 RCWA 백엔드는 Li의 규칙을 엄격하게 준수합니다. 이를 통해 CMOS 픽셀의 날카로운 금속 그리드 모서리나 굴절률 차이가 큰 실리콘/산화물 경계를 훨씬 적은 연산량과 메모리로 정확하게 모델링할 수 있습니다.</p>
    `
  },
  {
    id: "li_1996_smatrix",
    category: "S-Matrix (산란 행렬) 알고리즘",
    authors: "L. Li",
    title: "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings",
    journal: "J. Opt. Soc. Am. A",
    year: "1996",
    link: "https://doi.org/10.1364/JOSAA.13.001024",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Scattering_Matrix_Port_Definition.svg/512px-Scattering_Matrix_Port_Definition.svg.png",
    usedIn: [
      { label: "RCWA 설명", href: "/compass/ko/theory/simulation/rcwa-explained" },
      { label: "솔버 벤치마크", href: "/compass/ko/cookbook/solver-benchmark" },
      { label: "핵심 논문", href: "/compass/ko/research/key-papers" }
    ],
    summary: `
      <p>Moharam의 연구가 단일 층의 안정성을 해결했다면, Li의 이 논문은 여러 층을 쌓아 올릴 때 발생하는 전달 행렬(T-Matrix)의 불안정성 문제를 완벽히 해결한 산란 행렬(S-Matrix) 기법을 정립했습니다.</p>
      <ul>
        <li><strong>S-Matrix 방식:</strong> 입력 단의 전자기장과 출력 단의 전자기장을 직접 연결(이 과정에서 기하급수적 발산 발생)하는 대신, 한 층으로 '들어오는 파동'과 '나가는 파동'의 관계를 정의합니다. 이를 통해 행렬의 모든 원소 값이 물리적으로 안전한 범위를 벗어나지 않도록 제한합니다.</li>
        <li><strong>안정성:</strong> 층(layer)의 개수나 전체 격자의 두께에 상관없이 무조건적인 수치적 안정성을 보장합니다.</li>
      </ul>
      <p>COMPASS의 모든 다층 픽셀 스택(마이크로렌즈부터 포토다이오드 깊숙한 곳까지)은 S-matrix를 재귀적으로 결합하여 계산됩니다. 곡면 마이크로렌즈를 수십~수백 개의 얇은 층(계단 근사)으로 쪼개어 시뮬레이션하더라도 수치적으로 전혀 발산하지 않는 이유는 바로 이 S-matrix 덕분입니다.</p>
    `
  },
  {
    id: "yee_1966_fdtd",
    category: "FDTD 방법론 (FDTD Method)",
    authors: "K.S. Yee",
    title: "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media",
    journal: "IEEE Trans. Antennas Propag.",
    year: "1966",
    link: "https://doi.org/10.1109/TAP.1966.1138693",
    usedIn: [
      { label: "FDTD 설명", href: "/compass/ko/theory/simulation/fdtd-explained" },
      { label: "RCWA vs FDTD", href: "/compass/ko/theory/simulation/rcwa-vs-fdtd" },
      { label: "핵심 논문", href: "/compass/ko/research/key-papers" }
    ],
    summary: `
      <p>전설적인 "Yee Cell" 그리드를 처음 소개하며 시간 영역 유한 차분법(FDTD)의 기초를 다진 기념비적인 논문입니다.</p>
      <ul>
        <li><strong>Yee 그리드:</strong> Yee는 전기장(E)과 자기장(H) 성분을 공간과 시간 상에서 엇갈리게(staggered) 배치할 것을 제안했습니다. 전기장은 큐브의 모서리에, 자기장은 큐브의 면 중앙에 위치합니다.</li>
        <li><strong>립프로그(Leapfrog) 적분:</strong> 전기장과 자기장을 번갈아가며 시간 스텝별로 업데이트하는 방식으로, 맥스웰 방정식의 회전(Curl) 연산을 물리적으로 완벽하게 모사합니다.</li>
      </ul>
      <p>COMPASS의 FDTD 엔진(<code>flaport</code> 및 <code>Meep</code> 등)은 이 Yee 그리드 아키텍처를 엄격하게 따릅니다. 이는 발산이 없는(divergence-free) 전자기장을 보장하며, 광대역 광 펄스가 서브파장 픽셀 구조와 상호작용하는 복잡한 과정을 시간 영역에서 매우 정교하게 모사할 수 있게 해줍니다.</p>
    `
  },
  {
    id: "el_gamal_2005",
    category: "이미지 센서 물리 (Image Sensor Physics)",
    authors: "A. El Gamal and H. Eltoukhy",
    title: "CMOS image sensors",
    journal: "IEEE Circuits and Devices Magazine",
    year: "2005",
    link: "https://doi.org/10.1109/MCD.2005.1438751",
    imageUrl: "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Active_Pixel_Sensor.svg/512px-Active_Pixel_Sensor.svg.png",
    usedIn: [
      { label: "Noise and SNR", href: "/compass/ko/theory/sensor/noise-and-snr" },
      { label: "EMVA 1288 Dashboard", href: "/compass/ko/simulator/emva1288" },
      { label: "SNR Calculator", href: "/compass/ko/simulator/snr-calculator" }
    ],
    summary: `
      <p>CCD에서 CMOS 이미지 센서로의 기술적 전환기를 장식한, CMOS Active Pixel Sensor(APS) 기술에 대한 포괄적인 리뷰 및 튜토리얼 논문입니다.</p>
      <ul>
        <li><strong>센서 아키텍처:</strong> 3T 및 4T 픽셀 아키텍처의 구조, 상관 이중 샘플링(CDS)의 원리, 컬럼 병렬 판독(readout) 회로 등 핵심 기술을 명쾌하게 설명합니다.</li>
        <li><strong>노이즈 및 SNR:</strong> 리드 노이즈(Read noise), 암전류(Dark current), 광자 샷 노이즈(Photon shot noise) 등 CMOS 픽셀의 근본적인 노이즈 원인들을 분석하고, 이것이 어떻게 신호대잡음비(SNR)를 제한하는지 다룹니다.</li>
      </ul>
      <p>COMPASS는 광학 시뮬레이션에 집중하고 있지만, 양자 효율(QE)이 최종적인 디지털 신호(DN)로 변환되는 하류(downstream) 전자 회로 프로세스를 이해하는 것은 매우 중요합니다. COMPASS 내장 SNR 계산기 및 <code>SignalChainDiagram</code> 모델링은 이 논문의 이론적 토대 위에 구축되어 있습니다.</p>
    `
  },
  {
    id: "macleod_2017_thinfilm",
    category: "박막 광학 (Thin-Film Optics)",
    authors: "H.A. Macleod",
    title: "Thin-Film Optical Filters",
    journal: "CRC Press",
    year: "2017",
    link: "https://www.routledge.com/Thin-Film-Optical-Filters/Macleod/p/book/9781138198241",
    usedIn: [
      { label: "TMM QE 계산기", href: "/compass/ko/simulator/tmm-qe" },
      { label: "BARL Optimizer", href: "/compass/ko/simulator/barl-optimizer" },
      { label: "박막 광학", href: "/compass/ko/theory/optics/thin-film-optics" }
    ],
    summary: `
      <p>다층 박막 코팅을 characteristic matrix 관점에서 해석하는 실무 기준 문헌입니다. Optical admittance, phase thickness, reflection minimum, 그리고 coating stack이 bandwidth, angle, polarization, manufacturability 사이에서 어떤 절충을 만드는지 설명합니다.</p>
      <p>COMPASS에서는 TMM 및 BARL 브라우저 도구의 이론적 기준으로 사용됩니다. 해당 페이지들은 같은 coherent-film 논리로 layer thickness와 refractive index를 reflection, transmission, absorption trend로 변환합니다.</p>
    `
  },
  {
    id: "green_2008_silicon",
    category: "재료 광학 데이터 (Material Optical Data)",
    authors: "M.A. Green",
    title: "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients",
    journal: "Solar Energy Materials and Solar Cells",
    year: "2008",
    link: "https://doi.org/10.1016/j.solmat.2008.06.009",
    usedIn: [
      { label: "Si Absorption", href: "/compass/ko/simulator/si-absorption" },
      { label: "TMM QE 계산기", href: "/compass/ko/simulator/tmm-qe" },
      { label: "Quantum Efficiency", href: "/compass/ko/theory/sensor/quantum-efficiency" }
    ],
    summary: `
      <p>Green은 intrinsic silicon의 wavelength-dependent optical constants와 temperature coefficient를 self-consistent하게 정리합니다. Silicon에서 wavelength를 absorption depth로 변환할 때 가장 유용한 공개 문헌 중 하나입니다.</p>
      <p>COMPASS에서는 단순 브라우저 모델이 임의 감쇠 대신 물리적으로 근거 있는 silicon absorption trend를 가져야 할 때 이 레퍼런스를 기준으로 둡니다.</p>
    `
  },
  {
    id: "catrysse_2002_pixels",
    category: "이미지 센서 광학 (Image Sensor Optics)",
    authors: "P.B. Catrysse and B.A. Wandell",
    title: "Optical efficiency of image sensor pixels",
    journal: "J. Opt. Soc. Am. A",
    year: "2002",
    link: "https://doi.org/10.1364/JOSAA.19.001610",
    usedIn: [
      { label: "Quantum Efficiency", href: "/compass/ko/theory/sensor/quantum-efficiency" },
      { label: "Pixel Optical Effects", href: "/compass/ko/theory/sensor/pixel-optical-effects" },
      { label: "TMM QE 계산기", href: "/compass/ko/simulator/tmm-qe" }
    ],
    summary: `
      <p>이 논문은 optical simulation을 image-sensor pixel efficiency와 연결합니다. 입사 optical power를 의도한 photosensitive volume 안의 absorbed power로 매핑해야 하며, 그 매핑은 stack geometry와 material loss에 의존합니다.</p>
      <p>COMPASS의 여러 simulator 페이지에서 사용하는 optical QE proxy가 전자기 스택 계산과 어떻게 연결되는지 설명하는 개념적 다리 역할을 합니다.</p>
    `
  },
  {
    id: "ristoiu_2020_microlens_doe",
    category: "마이크로렌즈 공정 (Microlens Process)",
    authors: "Ristoiu et al.",
    title: "A DOE study of plasma etched microlens shape for CMOS image sensors",
    journal: "SPIE",
    year: "2020",
    link: "https://doi.org/10.1117/12.2551857",
    usedIn: [
      { label: "마이크로렌즈 공정 형상", href: "/compass/ko/simulator/microlens-process-shape" },
      { label: "마이크로렌즈 광선 추적", href: "/compass/ko/simulator/microlens-raytrace" },
      { label: "마이크로렌즈 최적화", href: "/compass/ko/cookbook/microlens-optimization" }
    ],
    summary: `
      <p>Layout에서 final microlens shape로 이어지는 질문에 가장 가까운 공개 CIS 공정 문헌입니다. Reflowed microlens를 plasma etch로 transfer하고, 공정 변수에 따른 final gap 및 height evolution을 모델링합니다.</p>
      <p>COMPASS는 이 논문을 근거로 mask thickness, polymerizing gas, etch time을 microlens process-shape surrogate의 핵심 제어 변수로 노출합니다.</p>
    `
  },
  {
    id: "baillie_2004_zero_space",
    category: "마이크로렌즈 공정 (Microlens Process)",
    authors: "Baillie and Gendler",
    title: "Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development",
    journal: "SPIE",
    year: "2004",
    link: "https://doi.org/10.1117/12.533453",
    usedIn: [
      { label: "마이크로렌즈 공정 형상", href: "/compass/ko/simulator/microlens-process-shape" },
      { label: "MLA 어레이 시각화", href: "/compass/ko/simulator/mla-array" },
      { label: "이미지 센서 광학", href: "/compass/ko/theory/sensor/image-sensor-optics" }
    ],
    summary: `
      <p>이 논문은 zero-space microlens fabrication 문제를 정리합니다. Residual lens spacing은 optical fill factor를 낮추지만, lithographic spacing이 너무 작으면 reflow 중 lens merger가 발생할 수 있습니다.</p>
      <p>그래서 COMPASS process-shape 도구는 zero gap을 무조건 좋은 것으로 보지 않고 final gap과 merger/over-etch warning을 함께 표시합니다.</p>
    `
  },
  {
    id: "hwang_2023_stack_alignment",
    category: "마이크로렌즈와 CRA (Microlens and CRA)",
    authors: "Hwang and Kim",
    title: "A Numerical Method of Aligning the Optical Stacks for All Pixels",
    journal: "Sensors",
    year: "2023",
    link: "https://doi.org/10.3390/s23020702",
    usedIn: [
      { label: "마이크로렌즈 광선 추적", href: "/compass/ko/simulator/microlens-raytrace" },
      { label: "렌즈 쉐이딩", href: "/compass/ko/simulator/lens-shading" },
      { label: "Cone Illumination", href: "/compass/ko/guide/cone-illumination" }
    ],
    summary: `
      <p>이 연구는 optical stack 전체에서 chief ray alignment를 수치적으로 맞추는 문제를 다룹니다. 센서 가장자리 픽셀에서는 microlens 위치와 stack geometry가 chief ray에 맞아야 focused spot이 의도한 photodiode에 도달합니다.</p>
      <p>COMPASS에서는 CRA shift 설명, microlens ray tracing, lens shading 논의에서 이 관점을 사용합니다.</p>
    `
  },
  {
    id: "cie_2018_colorimetry",
    category: "색과학 (Color Science)",
    authors: "Commission Internationale de l'Eclairage",
    title: "Colorimetry, 4th Edition",
    journal: "CIE 015:2018",
    year: "2018",
    link: "https://www.cie.co.at/publications/colorimetry-4th-edition",
    usedIn: [
      { label: "색 재현과 색공간", href: "/compass/ko/theory/sensor/color-reproduction" },
      { label: "컬러 필터 설계", href: "/compass/ko/simulator/color-filter" },
      { label: "색 정확도 분석기", href: "/compass/ko/simulator/color-accuracy" }
    ],
    summary: `
      <p>이 CIE 기술 보고서는 COMPASS 색 관련 페이지에서 쓰는 XYZ, chromaticity, Lab, standard observer, illuminant, color-difference 정의의 기준입니다.</p>
      <p>Sensor-dependent camera RGB와 standard colorimetric coordinate의 차이를 설명하는 기준 문헌으로 사용됩니다.</p>
    `
  },
  {
    id: "iec_61966_2_1_srgb",
    category: "색과학 (Color Science)",
    authors: "IEC",
    title: "Multimedia systems and equipment - Colour measurement and management - Part 2-1: Default RGB colour space - sRGB",
    journal: "IEC 61966-2-1:1999",
    year: "1999",
    link: "https://webstore.iec.ch/en/publication/6169",
    usedIn: [
      { label: "색 재현과 색공간", href: "/compass/ko/theory/sensor/color-reproduction" },
      { label: "색 정확도 분석기", href: "/compass/ko/simulator/color-accuracy" }
    ],
    summary: `
      <p>sRGB 표준은 웹과 일반 이미지 교환에서 쓰는 display-referred RGB 공간과 nonlinear encoding을 정의합니다.</p>
      <p>COMPASS에서는 raw camera RGB의 대체물이 아니라, sensor RGB가 XYZ를 거친 뒤 도달하는 표시 endpoint로 다룹니다.</p>
    `
  },
  {
    id: "sharma_2005_ciede2000",
    category: "색과학 (Color Science)",
    authors: "G. Sharma, W. Wu, and E.N. Dalal",
    title: "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations",
    journal: "Color Research & Application",
    year: "2005",
    link: "https://doi.org/10.1002/col.20070",
    usedIn: [
      { label: "색 재현과 색공간", href: "/compass/ko/theory/sensor/color-reproduction" },
      { label: "색 정확도 분석기", href: "/compass/ko/simulator/color-accuracy" },
      { label: "신호 체인 색 정확도", href: "/compass/ko/cookbook/signal-chain-color-accuracy" }
    ],
    summary: `
      <p>이 논문은 CIEDE2000 색차 계산의 실무 구현 기준으로 널리 쓰입니다.</p>
      <p>센서 분광 응답을 표준 색공간으로 매핑한 뒤 perceptual error metric으로 평가해야 하는 COMPASS 색 최적화 흐름과 직접 연결됩니다.</p>
    `
  },
  {
    id: "nist_2002_color_by_numbers",
    category: "색과학 (Color Science)",
    authors: "S.W. Brown and Y. Ohno",
    title: "Color By Numbers: Using a Calibration Source Spectrally Matched To Your Test Source Is Key To Measurement Accuracy",
    journal: "NIST",
    year: "2002",
    link: "https://www.nist.gov/publications/color-numbers-using-calibration-source-spectrally-matched-your-test-source-key",
    usedIn: [
      { label: "색 재현과 색공간", href: "/compass/ko/theory/sensor/color-reproduction" },
      { label: "컬러 필터 설계", href: "/compass/ko/simulator/color-filter" }
    ],
    summary: `
      <p>이 NIST 문서는 측정 대상 스펙트럼이 calibration source와 달라질 때 spectral responsivity mismatch가 색 측정 오차를 어떻게 바꾸는지 설명합니다.</p>
      <p>CIS 색 설계에서도 같은 원리가 적용됩니다. Camera channel response는 이상적인 RGB endpoint가 아니라 실제 illuminant와 object spectrum으로 평가해야 합니다.</p>
    `
  }
];
