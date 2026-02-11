# Key Academic Papers

An annotated list of key papers that form the theoretical and practical foundation of the COMPASS project.
Each paper includes full citation information, an English summary, and relevance tags.

---

## 1. RCWA Fundamentals

Papers providing the theoretical foundations of RCWA (Rigorous Coupled-Wave Analysis), the core solver in COMPASS.

### 1.1 Moharam & Gaylord (1981) -- Original RCWA Formulation

- **Citation**: M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction," *J. Opt. Soc. Am.*, vol. 71, no. 7, pp. 811--818, 1981. DOI: [10.1364/JOSA.71.000811](https://doi.org/10.1364/JOSA.71.000811)
- **Summary**: The seminal paper that first established the coupled-wave analysis method (RCWA) for rigorously solving electromagnetic wave diffraction in periodic structures. By combining Fourier series expansion of permittivity with an eigenvalue problem, it presented a method for computing exact transmission/reflection efficiencies for arbitrary diffraction gratings. This is the theoretical starting point for all RCWA solvers used in COMPASS (torcwa, grcwa, meent).
- **Tags**: [RCWA] [Fundamentals]

### 1.2 Moharam & Gaylord (1986) -- Metallic Surface-Relief Gratings

- **Citation**: M. G. Moharam and T. K. Gaylord, "Rigorous coupled-wave analysis of metallic surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 3, no. 11, pp. 1780--1787, 1986. DOI: [10.1364/JOSAA.3.001780](https://doi.org/10.1364/JOSAA.3.001780)
- **Summary**: This paper extended RCWA to metallic surface-relief gratings, enabling the treatment of TE/TM polarization and arbitrary incidence angles for materials with complex permittivity. It provides the essential theoretical basis for simulating absorptive metallic structures in COMPASS, such as tungsten grids and metal light shields.
- **Tags**: [RCWA] [Metallic Structures]

### 1.3 Moharam et al. (1995a) -- Stable Implementation and S-matrix

- **Citation**: M. G. Moharam, E. B. Grann, D. A. Pommet, and T. K. Gaylord, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1068--1076, 1995. DOI: [10.1364/JOSAA.12.001068](https://doi.org/10.1364/JOSAA.12.001068)
- **Summary**: This paper systematically analyzed numerical stability issues in RCWA and introduced the S-matrix algorithm. It proposed a Redheffer star product-based approach to solve the exponential overflow problem inherent in the T-matrix method. The `StableSMatrixAlgorithm` module in COMPASS directly implements the methodology from this paper.
- **Tags**: [RCWA] [Numerical Stability]

### 1.4 Moharam et al. (1995b) -- Enhanced Transmittance Matrix Approach

- **Citation**: M. G. Moharam, D. A. Pommet, E. B. Grann, and T. K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1077--1086, 1995. DOI: [10.1364/JOSAA.12.001077](https://doi.org/10.1364/JOSAA.12.001077)
- **Summary**: As a companion paper to the above, this work presents the enhanced transmittance matrix (ETM) approach in detail. It provides a comparative analysis of the S-matrix and ETM as two stable implementation methods, offering practical guidance for ensuring numerical stability in multilayer structure simulations.
- **Tags**: [RCWA] [Numerical Stability]

### 1.5 Li (1996a) -- Treatment of Discontinuous Functions in Fourier Series

- **Citation**: L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A*, vol. 13, no. 9, pp. 1870--1876, 1996. DOI: [10.1364/JOSAA.13.001870](https://doi.org/10.1364/JOSAA.13.001870)
- **Summary**: This paper mathematically identified convergence issues in Fourier series analysis of discontinuous periodic structures. It established the correct Fourier factorization rules (Li's rules) for the product of discontinuous functions, which dramatically improved TM polarization convergence at metal/dielectric interfaces. This is the theoretical basis for the `LiFactorization.convolution_matrix_inverse_rule` method in COMPASS.
- **Tags**: [RCWA] [Convergence] [Li's Rules]

### 1.6 Li (1996b) -- Comparison of Recursive Matrix Algorithms for Scattering

- **Citation**: L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings," *J. Opt. Soc. Am. A*, vol. 13, no. 5, pp. 1024--1035, 1996. DOI: [10.1364/JOSAA.13.001024](https://doi.org/10.1364/JOSAA.13.001024)
- **Summary**: This paper formulated and compared two recursive matrix algorithms (S-matrix and R-matrix) for modeling multilayer diffraction gratings. It proved that the S-matrix method is unconditionally stable numerically, providing the rationale for COMPASS's exclusive use of S-matrix.
- **Tags**: [RCWA] [Numerical Stability] [S-matrix]

### 1.7 Li (1997) -- FMM for Crossed Surface-Relief Gratings

- **Citation**: L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 10, pp. 2758--2767, 1997. DOI: [10.1364/JOSAA.14.002758](https://doi.org/10.1364/JOSAA.14.002758)
- **Summary**: This paper proposed a new formulation of the Fourier modal method (FMM) for two-dimensional crossed gratings. By applying correct Fourier factorization rules to 2D structures, convergence rates were significantly improved. This is the core algorithmic foundation for handling 2D periodic structures in COMPASS, such as 2x2 Bayer patterns.
- **Tags**: [RCWA] [2D Gratings] [Convergence]

### 1.8 Lalanne (1997) -- Improved Formulation of the Coupled-Wave Method

- **Citation**: P. Lalanne, "Improved formulation of the coupled-wave method for two-dimensional gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 7, pp. 1592--1598, 1997. DOI: [10.1364/JOSAA.14.001592](https://doi.org/10.1364/JOSAA.14.001592)
- **Summary**: Independently of Li's factorization rules, this paper proposed a new formulation to improve the convergence of the coupled-wave method for 2D gratings. It numerically demonstrated faster convergence compared to existing methods across various diffraction problems, including dielectric, metallic, volume, and surface-relief gratings.
- **Tags**: [RCWA] [Convergence] [2D Gratings]

### 1.9 Popov & Neviere (2001) -- Fast-Converging Formulation

- **Citation**: E. Popov and M. Neviere, "Maxwell equations in Fourier space: fast-converging formulation for diffraction by arbitrary shaped, periodic, anisotropic media," *J. Opt. Soc. Am. A*, vol. 18, no. 11, pp. 2886--2894, 2001. DOI: [10.1364/JOSAA.18.002886](https://doi.org/10.1364/JOSAA.18.002886)
- **Summary**: This paper proposed a fast-converging formulation of Maxwell's equations in Fourier space for periodic structures of arbitrary shape. Through the Fast Fourier Factorization (FFF) method, it improved convergence rates at irregular boundaries and can be extended to anisotropic media.
- **Tags**: [RCWA] [Convergence] [FFF]

---

## 2. FDTD Fundamentals

Papers forming the theoretical foundations of FDTD (Finite-Difference Time-Domain), the alternative solver in COMPASS.

### 2.1 Yee (1966) -- Original FDTD Proposal

- **Citation**: K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302--307, 1966. DOI: [10.1109/TAP.1966.1138693](https://doi.org/10.1109/TAP.1966.1138693)
- **Summary**: The landmark paper that first proposed the finite-difference time-domain method for Maxwell's equations. It introduced the Yee grid, where electric and magnetic fields are computed alternately on a spatially staggered grid. This is the fundamental algorithm underlying the flaport FDTD solver in COMPASS.
- **Tags**: [FDTD] [Fundamentals]

### 2.2 Taflove & Hagness (2005) -- FDTD Textbook

- **Citation**: A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed. Norwood, MA: Artech House, 2005. ISBN: 978-1580538329
- **Summary**: The most comprehensive reference textbook on FDTD methodology. It covers all aspects of FDTD from basic algorithms to absorbing boundary conditions, dispersive media modeling, and near-to-far-field transformations. Used as a reference for numerical dispersion, stability conditions (CFL), and grid resolution criteria in COMPASS's FDTD solver implementation.
- **Tags**: [FDTD] [Textbook]

### 2.3 Berenger (1994) -- PML Absorbing Boundary Condition

- **Citation**: J.-P. Berenger, "A perfectly matched layer for the absorption of electromagnetic waves," *J. Comput. Phys.*, vol. 114, no. 2, pp. 185--200, 1994. DOI: [10.1006/jcph.1994.1159](https://doi.org/10.1006/jcph.1994.1159)
- **Summary**: The paper that first proposed the Perfectly Matched Layer (PML) for reflection-free wave absorption in electromagnetic simulations. It revolutionized the open boundary condition problem in FDTD simulations and has since become the standard boundary condition for all EM simulators. Essential for computational domain termination in COMPASS's FDTD solver.
- **Tags**: [FDTD] [PML] [Boundary Conditions]

### 2.4 Gedney (1996) -- Uniaxial PML (UPML)

- **Citation**: S. D. Gedney, "An anisotropic perfectly matched layer-absorbing medium for the truncation of FDTD lattices," *IEEE Trans. Antennas Propag.*, vol. 44, no. 12, pp. 1630--1639, 1996. DOI: [10.1109/8.546249](https://doi.org/10.1109/8.546249)
- **Summary**: This paper reformulated Berenger's split-field PML as a uniaxial anisotropic medium (Uniaxial PML). The UPML approach is more physically intuitive and simpler to implement, and has been widely adopted in modern FDTD codes. It also facilitates PML extension to dispersive and anisotropic media.
- **Tags**: [FDTD] [PML] [UPML]

---

## 3. Silicon Optical Constants

Papers on the optical constants of silicon, the key material for image sensor simulation.

### 3.1 Green (2008) -- Standard Silicon Optical Parameter Data

- **Citation**: M. A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," *Sol. Energy Mater. Sol. Cells*, vol. 92, no. 11, pp. 1305--1310, 2008. DOI: [10.1016/j.solmat.2008.06.009](https://doi.org/10.1016/j.solmat.2008.06.009)
- **Summary**: Standard data providing self-consistent optical constants (absorption coefficient, refractive index, extinction coefficient) of intrinsic silicon at 300K across the 0.25--1.45um wavelength range. Derived from updated reflectance data using Kramers-Kronig analysis, including temperature coefficients. This is the reference paper for silicon data in COMPASS's MaterialDB (n~4.08, k~0.028 at 550nm).
- **Tags**: [Materials] [Silicon] [Optical Constants]

<MaterialBrowser />

### 3.2 Aspnes & Studna (1983) -- Spectroscopic Ellipsometry Data

- **Citation**: D. E. Aspnes and A. A. Studna, "Dielectric functions and optical parameters of Si, Ge, GaP, GaAs, GaSb, InP, InAs, and InSb from 1.5 to 6.0 eV," *Phys. Rev. B*, vol. 27, no. 2, pp. 985--1009, 1983. DOI: [10.1103/PhysRevB.27.985](https://doi.org/10.1103/PhysRevB.27.985)
- **Summary**: This paper measured the dielectric functions and optical parameters of 8 semiconductors including Si using spectroscopic ellipsometry over the 1.5--6.0eV range. Cited over 3,100 times, it is a high-impact paper and a standard reference for semiconductor optical constants.
- **Tags**: [Materials] [Silicon] [Spectroscopy]

### 3.3 Palik (1998) -- Handbook of Optical Constants

- **Citation**: E. D. Palik, ed., *Handbook of Optical Constants of Solids*, Academic Press, 1998. ISBN: 978-0125444156
- **Summary**: A comprehensive reference book containing optical constants for most materials used in COMPASS, including SiO2, Si3N4, HfO2, TiO2, and tungsten. It provides n and k values over a wide wavelength range (UV to infrared) and is used for verification of built-in material data in COMPASS's MaterialDB.
- **Tags**: [Materials] [Optical Constants] [Reference Book]

---

## 4. Image Sensor Optics

Key papers on the optical simulation of CMOS image sensor (CIS) pixels.

### 4.1 Catrysse & Wandell (2002) -- CIS Optical Efficiency Analysis

- **Citation**: P. B. Catrysse and B. A. Wandell, "Optical efficiency of image sensor pixels," *J. Opt. Soc. Am. A*, vol. 19, no. 8, pp. 1610--1620, 2002. DOI: [10.1364/JOSAA.19.001610](https://doi.org/10.1364/JOSAA.19.001610)
- **Summary**: A pioneering paper that analyzed the optical efficiency of CMOS image sensor pixels using a geometrical-optics phase-space approach. It presented theoretical predictions matching experimental measurements to within 3% accuracy, and laid the groundwork for subsequent FDTD-based CIS simulation research. This is the starting point for QE calculation and crosstalk analysis methodology in COMPASS.
- **Tags**: [CIS] [QE] [Optical Efficiency]

### 4.2 Agranov, Berezin & Tsai (2003) -- Crosstalk and Microlens Study

- **Citation**: G. Agranov, V. Berezin, and R. H. Tsai, "Crosstalk and microlens study in a color CMOS image sensor," *IEEE Trans. Electron Devices*, vol. 50, no. 1, pp. 4--11, 2003. DOI: [10.1109/TED.2002.806473](https://doi.org/10.1109/TED.2002.806473)
- **Summary**: An experimental study on optical crosstalk and the role of microlenses in color CMOS image sensors. It quantitatively analyzed the increase in crosstalk with pixel size reduction and demonstrated the potential for improvement through microlens optimization. This provides the basis for microlens shift and crosstalk analysis in COMPASS.
- **Tags**: [CIS] [Crosstalk] [Microlens]

### 4.3 El Gamal & Eltoukhy (2005) -- CIS Review

- **Citation**: A. El Gamal and H. Eltoukhy, "CMOS image sensors," *IEEE Circuits and Devices Magazine*, vol. 21, no. 3, pp. 6--20, 2005. DOI: [10.1109/MCD.2005.1438751](https://doi.org/10.1109/MCD.2005.1438751)
- **Summary**: A comprehensive review of the operating principles, architecture, and performance metrics of CMOS image sensors. An essential reference for understanding photoconversion efficiency, noise sources, and pixel circuit design across all aspects of CIS.
- **Tags**: [CIS] [Review]

### 4.4 Yokogawa et al. (2017) -- Light Trapping in BSI Sensors

- **Citation**: S. Yokogawa, I. Oshiyama, H. Ikeda, Y. Ebiko, T. Hirano, S. Saito, T. Oinoue, Y. Hagimoto, and H. Iwamoto, "IR sensitivity enhancement of CMOS Image Sensor with diffractive light trapping pixels," *Sci. Rep.*, vol. 7, 3832, 2017. DOI: [10.1038/s41598-017-04200-y](https://doi.org/10.1038/s41598-017-04200-y)
- **Summary**: This paper reported on diffractive light trapping technology using 2D inverted pyramid array (IPA) structures and DTI (Deep Trench Isolation) in BSI (backside illumination) CMOS image sensors. It achieved an 80% sensitivity improvement at 850nm and presented a comparative analysis of FDTD simulations and experimental data. An important validation reference for DTI and BSI pixel simulations in COMPASS.
- **Tags**: [CIS] [BSI] [Light Trapping] [DTI]

---

## 5. Numerical Stability

Papers on ensuring numerical stability in RCWA simulations.

### 5.1 Schuster et al. (2007) -- Normal Vector Method

- **Citation**: T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, and W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *J. Opt. Soc. Am. A*, vol. 24, no. 9, pp. 2880--2890, 2007. DOI: [10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880)
- **Summary**: This paper proposed the Normal Vector Method for improving RCWA convergence in crossed gratings. By utilizing normal vectors at material boundaries to accurately apply electric field continuity conditions, it achieves fast convergence even in complex 2D patterns where Li's rules alone are insufficient. This is the theoretical basis for the `fourier_factorization: "normal_vector"` option in COMPASS.
- **Tags**: [RCWA] [Numerical Stability] [Convergence]

### 5.2 Kim & Lee (2023) -- Eigenvalue Broadening Technique

- **Citation**: S. Kim and D. Lee, "Eigenvalue broadening technique for stable RCWA simulation of high-contrast gratings," *Comput. Phys. Commun.*, vol. 282, 108547, 2023. DOI: [10.1016/j.cpc.2022.108547](https://doi.org/10.1016/j.cpc.2022.108547)
- **Summary**: This paper proposed a broadening technique for the degenerate eigenvalue problem that arises in RCWA simulations of high-contrast gratings. Through a post-processing method that detects closely spaced eigenvalue pairs and re-orthogonalizes eigenvectors, COMPASS's `EigenvalueStabilizer.fix_degenerate_eigenvalues` method implements this technique. This is the theoretical basis for the `eigenvalue_broadening: 1e-10` parameter.
- **Tags**: [RCWA] [Numerical Stability] [Eigenvalues]

---

## 6. Microlens & Color Filter

Papers on microlenses and color filters, key elements of image sensor optical structures.

### 6.1 Macleod (2017) -- Thin-Film Optical Filters

- **Citation**: H. A. Macleod, *Thin-Film Optical Filters*, 5th ed. Boca Raton, FL: CRC Press, 2017. ISBN: 978-1138198241
- **Summary**: The standard textbook on the design and analysis of thin-film optical filters. It covers in detail the theory and design methods for anti-reflection coatings, interference filters, and dielectric multilayer films. Used as a reference for BARL (Bottom Anti-Reflection Layer) design and color filter optimization in COMPASS.
- **Tags**: [Thin-Film Optics] [Color Filter] [BARL]

### 6.2 Born & Wolf (1999) -- Principles of Optics

- **Citation**: M. Born and E. Wolf, *Principles of Optics*, 7th ed. Cambridge: Cambridge University Press, 1999. ISBN: 978-0521642224
- **Summary**: The standard textbook of classical optics based on electromagnetic theory. A theoretical reference for all optical phenomena that form the physical foundation of COMPASS, including Fresnel equations, interference, diffraction, and polarization. Particularly used for derivation of the transfer matrix method (TMM) for multilayer thin films.
- **Tags**: [Optical Theory] [Textbook] [Fresnel]

### 6.3 Catrysse et al. (2003) -- Integrated Color Pixels

- **Citation**: P. B. Catrysse, W. Suh, S. Fan, and M. Peeters, "Integrated color pixels in 0.18-um complementary metal oxide semiconductor technology," *J. Opt. Soc. Am. A*, vol. 20, no. 12, pp. 2293--2306, 2003. DOI: [10.1364/JOSAA.20.002293](https://doi.org/10.1364/JOSAA.20.002293)
- **Summary**: This paper analyzed the optical performance of integrated color pixels fabricated in a 0.18um CMOS process using FDTD. It presented electromagnetic simulation methodology for actual pixel structures including color filter arrays, microlenses, and metal interconnects. This is prior work for the full-stack pixel simulation approach in COMPASS.
- **Tags**: [CIS] [Color Filter] [FDTD]

---

## 7. Inverse Design & Optimization

Key papers on nanophotonic inverse design and optimization.

### 7.1 Molesky et al. (2018) -- Nanophotonic Inverse Design Review

- **Citation**: S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vuckovic, and A. W. Rodriguez, "Inverse design in nanophotonics," *Nat. Photonics*, vol. 12, pp. 659--670, 2018. DOI: [10.1038/s41566-018-0246-9](https://doi.org/10.1038/s41566-018-0246-9)
- **Summary**: A systematic review of key advances in nanophotonic inverse design. It covers inverse design techniques in diverse applications including nonlinear, topological, near-field, and on-chip optics, and provides the theoretical foundations of topology optimization and the adjoint method. This informs the direction of future optimization feature development in COMPASS.
- **Tags**: [Optimization] [Inverse Design] [Review]

### 7.2 Hughes et al. (2018) -- Adjoint Method for Inverse Design

- **Citation**: T. W. Hughes, M. Minkov, I. A. D. Williamson, and S. Fan, "Adjoint method and inverse design for nonlinear nanophotonic devices," *ACS Photonics*, vol. 5, no. 12, pp. 4781--4787, 2018. DOI: [10.1021/acsphotonics.8b01522](https://doi.org/10.1021/acsphotonics.8b01522)
- **Summary**: This paper presented an extension of the adjoint method for frequency-domain inverse design of nonlinear nanophotonic devices. By directly incorporating nonlinear responses into gradient calculations, it demonstrated the inverse design of Kerr nonlinear devices. A theoretical reference for differentiable simulation-based optimization in COMPASS.
- **Tags**: [Optimization] [Adjoint Method] [Nonlinear]

### 7.3 Piggott et al. (2015) -- Nanophotonic Inverse Design Demonstration

- **Citation**: A. Y. Piggott, J. Lu, K. G. Lagoudakis, J. Petykiewicz, T. M. Babinec, and J. Vuckovic, "Inverse design and demonstration of a compact and broadband on-chip wavelength demultiplexer," *Nat. Photonics*, vol. 9, pp. 374--377, 2015. DOI: [10.1038/nphoton.2015.69](https://doi.org/10.1038/nphoton.2015.69)
- **Summary**: A landmark paper reporting the inverse design and experimental demonstration of a compact, broadband on-chip wavelength demultiplexer on a silicon photonics platform. Using adjoint method-based topology optimization, it demonstrated performance unattainable by conventional design methods, and became a cornerstone of the nanophotonic inverse design field.
- **Tags**: [Optimization] [Inverse Design] [Experimental Demonstration]

---

## 8. Differentiable Simulation

Original papers for the differentiable EM simulator libraries used in COMPASS.

### 8.1 Jin et al. (2020) -- grcwa (Differentiable RCWA)

- **Citation**: W. Jin, W. Li, M. Orenstein, and S. Fan, "Inverse design of lightweight broadband reflector for relativistic lightsail propulsion," *ACS Photonics*, vol. 7, no. 9, pp. 2350--2355, 2020. DOI: [10.1021/acsphotonics.0c00768](https://doi.org/10.1021/acsphotonics.0c00768)
- **Summary**: This paper developed and applied grcwa, an automatic differentiation (autograd)-enabled RCWA implementation, for the inverse design of a relativistic broadband reflector. By replacing adjoint method gradient computation with automatic differentiation, it simplified the inverse design pipeline. This is the source library for the grcwa solver wrapper in COMPASS.
- **Tags**: [Differentiable Simulation] [RCWA] [grcwa] [Optimization]

### 8.2 Kim & Lee (2023) -- torcwa (GPU-Accelerated RCWA)

- **Citation**: C. Kim and B. Lee, "TORCWA: GPU-accelerated Fourier modal method and gradient-based optimization for metasurface design," *Comput. Phys. Commun.*, vol. 282, 108552, 2023. DOI: [10.1016/j.cpc.2022.108552](https://doi.org/10.1016/j.cpc.2022.108552)
- **Summary**: This paper introduced torcwa, a PyTorch-based GPU-accelerated RCWA simulator. It achieved significant speedup over CPU while maintaining comparable accuracy, and supports gradient-based optimization through PyTorch's backpropagation automatic differentiation. This is the default RCWA solver in COMPASS and the most frequently used backend.
- **Tags**: [Differentiable Simulation] [RCWA] [torcwa] [GPU]

### 8.3 Kim et al. (2024) -- meent (Differentiable EM Simulator)

- **Citation**: Y. Kim, A. W. Jung, S. Kim, K. Octavian, D. Heo, C. Park, J. Shin, S. Nam, C. Park, J. Park, et al., "Meent: Differentiable electromagnetic simulator for machine learning," *arXiv preprint*, arXiv:2406.12904, 2024. [arXiv:2406.12904](https://arxiv.org/abs/2406.12904)
- **Summary**: This paper introduced meent, a differentiable RCWA solver developed for the integration of machine learning and EM simulation. It supports analytical eigenvalue decomposition to improve numerical stability and demonstrates three applications: neural operator learning, reinforcement learning-based optimization, and gradient-based inverse problem solving. This is the third RCWA backend in COMPASS.
- **Tags**: [Differentiable Simulation] [RCWA] [meent] [ML]

---

## Appendix: COMPASS Solver Library Links

| Solver | Type | Repository | License |
|-------|------|--------|----------|
| torcwa | RCWA (PyTorch) | [github.com/kch3782/torcwa](https://github.com/kch3782/torcwa) | MIT |
| grcwa | RCWA (autograd) | [github.com/weiliangjinca/grcwa](https://github.com/weiliangjinca/grcwa) | MIT |
| meent | RCWA (multi-backend) | [github.com/kc-ml2/meent](https://github.com/kc-ml2/meent) | MIT |
| fdtd (flaport) | FDTD (PyTorch) | [github.com/flaport/fdtd](https://github.com/flaport/fdtd) | MIT |

---

## Notes

- DOI links have been verified as of 2024. Some DOIs may redirect due to publisher changes.
- For arXiv preprints, DOIs may change upon journal publication.
- In-code citations in COMPASS can also be found in the `compass/solvers/rcwa/stability.py` module docstrings and in `docs/about/references.md`.
