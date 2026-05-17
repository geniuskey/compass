<script setup>
import { referencesEn } from './referencesData'
</script>

# Reference Map

This page is not the authoritative citation list for every topic. Topic-specific references live at the bottom of the relevant theory, guide, and simulator pages. Use this page as a cross-document reading map for the foundational papers, solver methods, optical data sources, and software tools that appear repeatedly across COMPASS.

::: tip How to use this page
Start with the cards for foundational method papers. For implementation-specific or simulator-specific claims, follow the references on the page where the claim appears.
:::

The separate map is useful only for cross-cutting sources. It should stay compact instead of duplicating every local citation.

<ReferenceInteractiveList :references="referencesEn" />

## Method and Data References

### RCWA Theory & Enhancements
- M.G. Moharam, D.A. Pommet, E.B. Grann, and T.K. Gaylord, "Stable implementation of the rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix approach," *J. Opt. Soc. Am. A*, vol. 12, no. 5, pp. 1077-1086, 1995.
- L. Li, "New formulation of the Fourier modal method for crossed surface-relief gratings," *J. Opt. Soc. Am. A*, vol. 14, no. 10, pp. 2758-2767, 1997.
- T. Schuster, J. Ruoff, N. Kerwien, S. Rafler, and W. Osten, "Normal vector method for convergence improvement using the RCWA for crossed gratings," *J. Opt. Soc. Am. A*, vol. 24, no. 9, pp. 2880-2890, 2007.
- S. Kim and D. Lee, "Eigenvalue broadening technique for stable RCWA simulation of high-contrast gratings," *Comput. Phys. Commun.*, vol. 282, 108547, 2023.

### FDTD Method
- A. Taflove and S.C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed. Artech House, 2005.

### Silicon Optical Properties
- M.A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," *Solar Energy Materials and Solar Cells*, vol. 92, no. 11, pp. 1305-1310, 2008.
- E.D. Palik, *Handbook of Optical Constants of Solids*, Academic Press, 1998.

### Image Sensor Physics
- S.K. Mendis, S.E. Kemeny, R.C. Gee, B. Pain, C.O. Staller, Q. Kim, and E.R. Fossum, "CMOS active pixel image sensors for highly integrated imaging systems," *IEEE J. Solid-State Circuits*, vol. 32, no. 2, pp. 187-197, 1997.

### Thin Film Optics
- H.A. Macleod, *Thin-Film Optical Filters*, 5th ed. CRC Press, 2017.
- M. Born and E. Wolf, *Principles of Optics*, 7th ed. Cambridge University Press, 1999.

## Solver Libraries

- **torcwa**: PyTorch-based RCWA. GPU-accelerated.
- **grcwa**: GPU RCWA implementation.
- **meent**: Metasurface electromagnetic solver with analytic eigendecomposition.
- **fdtd (flaport)**: PyTorch-based FDTD.
- **Meep**: MIT Electromagnetic Equation Propagation (open-source FDTD).

## Software Tools

- **PyTorch**: [pytorch.org](https://pytorch.org/) -- GPU computation framework
- **Hydra**: [hydra.cc](https://hydra.cc/) -- Configuration management
- **Pydantic**: [pydantic.dev](https://docs.pydantic.dev/) -- Data validation
- **VitePress**: [vitepress.dev](https://vitepress.dev/) -- Documentation framework
