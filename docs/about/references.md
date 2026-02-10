# References

Key publications and resources that underpin the physics and algorithms in COMPASS.

## RCWA theory

- M.G. Moharam, E.B. Grann, D.A. Pommet, and T.K. Gaylord, "Formulation for stable and efficient implementation of the rigorous coupled-wave analysis of binary gratings," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1068-1076, 1995.

- M.G. Moharam, D.A. Pommet, E.B. Grann, and T.K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1077-1086, 1995.

## Fourier factorization (Li rules)

- L. Li, "Use of Fourier series in the analysis of discontinuous periodic structures," *J. Opt. Soc. Am. A*, vol. 13, no. 9, pp. 1870-1876, 1996.

- L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 10, pp. 2758-2767, 1997.

## S-matrix algorithm

- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings," *J. Opt. Soc. Am. A*, vol. 13, no. 5, pp. 1024-1035, 1996.

- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, and W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *J. Opt. Soc. Am. A*, vol. 24, no. 9, pp. 2880-2890, 2007.

## FDTD method

- A. Taflove and S.C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed. Artech House, 2005.

- K.S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.*, vol. 14, no. 3, pp. 302-307, 1966.

## Numerical stability

- S. Kim and D. Lee, "Eigenvalue broadening technique for stable RCWA simulation of high-contrast gratings," *Comput. Phys. Commun.*, vol. 282, 108547, 2023.

## Silicon optical properties

- M.A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," *Solar Energy Materials and Solar Cells*, vol. 92, no. 11, pp. 1305-1310, 2008.

- E.D. Palik, *Handbook of Optical Constants of Solids*, Academic Press, 1998.

## Image sensor physics

- A. El Gamal and H. Eltoukhy, "CMOS image sensors," *IEEE Circuits and Devices Magazine*, vol. 21, no. 3, pp. 6-20, 2005.

- S.K. Mendis, S.E. Kemeny, R.C. Gee, B. Pain, C.O. Staller, Q. Kim, and E.R. Fossum, "CMOS active pixel image sensors for highly integrated imaging systems," *IEEE J. Solid-State Circuits*, vol. 32, no. 2, pp. 187-197, 1997.

## Thin film optics

- H.A. Macleod, *Thin-Film Optical Filters*, 5th ed. CRC Press, 2017.

- M. Born and E. Wolf, *Principles of Optics*, 7th ed. Cambridge University Press, 1999.

## Solver libraries

- **torcwa**: PyTorch-based RCWA. GPU-accelerated.
- **grcwa**: GPU RCWA implementation.
- **meent**: Metasurface electromagnetic solver with analytic eigendecomposition.
- **fdtd (flaport)**: PyTorch-based FDTD.
- **Meep**: MIT Electromagnetic Equation Propagation (open-source FDTD).

## Software tools

- **PyTorch**: [pytorch.org](https://pytorch.org/) -- GPU computation framework
- **Hydra**: [hydra.cc](https://hydra.cc/) -- Configuration management
- **Pydantic**: [pydantic.dev](https://docs.pydantic.dev/) -- Data validation
- **VitePress**: [vitepress.dev](https://vitepress.dev/) -- Documentation framework
