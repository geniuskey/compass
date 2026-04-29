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
    summary: `
      <p>CCD에서 CMOS 이미지 센서로의 기술적 전환기를 장식한, CMOS Active Pixel Sensor(APS) 기술에 대한 포괄적인 리뷰 및 튜토리얼 논문입니다.</p>
      <ul>
        <li><strong>센서 아키텍처:</strong> 3T 및 4T 픽셀 아키텍처의 구조, 상관 이중 샘플링(CDS)의 원리, 컬럼 병렬 판독(readout) 회로 등 핵심 기술을 명쾌하게 설명합니다.</li>
        <li><strong>노이즈 및 SNR:</strong> 리드 노이즈(Read noise), 암전류(Dark current), 광자 샷 노이즈(Photon shot noise) 등 CMOS 픽셀의 근본적인 노이즈 원인들을 분석하고, 이것이 어떻게 신호대잡음비(SNR)를 제한하는지 다룹니다.</li>
      </ul>
      <p>COMPASS는 광학 시뮬레이션에 집중하고 있지만, 양자 효율(QE)이 최종적인 디지털 신호(DN)로 변환되는 하류(downstream) 전자 회로 프로세스를 이해하는 것은 매우 중요합니다. COMPASS 내장 SNR 계산기 및 <code>SignalChainDiagram</code> 모델링은 이 논문의 이론적 토대 위에 구축되어 있습니다.</p>
    `
  }
];
