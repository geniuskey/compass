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
    summary: `
      <p>A comprehensive tutorial and review of CMOS Active Pixel Sensor (APS) technology that marked the transition from CCDs to CMOS.</p>
      <ul>
        <li><strong>Sensor Architecture:</strong> Explains the 3T and 4T pixel architectures, correlated double sampling (CDS), and column-parallel readout.</li>
        <li><strong>Noise & SNR:</strong> Details the fundamental noise sources in CMOS pixels (read noise, dark current, photon shot noise) and how they limit the signal-to-noise ratio.</li>
      </ul>
      <p>COMPASS leverages these principles in its <code>SignalChainDiagram</code> and SNR calculators. While COMPASS focuses heavily on optics, understanding the downstream electronic conversion (quantum efficiency to digital numbers) is crucial for our end-to-end pixel modeling.</p>
    `
  }
];
