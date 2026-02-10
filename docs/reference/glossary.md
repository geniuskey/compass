# Glossary

A comprehensive glossary of terms used throughout the COMPASS (Cross-solver Optical Modeling Platform for Advanced Sensor Simulation) documentation. Terms are organized alphabetically for quick reference.

---

## A

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Absorption Coefficient ($\alpha$) | Rate at which light intensity decreases per unit distance as it propagates through an absorbing medium. Measured in cm$^{-1}$. Wavelength-dependent and critical for calculating photon absorption profiles in silicon. | Beer-Lambert Law, Extinction Coefficient |
| Angular Spectrum | Decomposition of an electromagnetic field into a superposition of plane waves propagating at different angles. Used in propagation and diffraction calculations. | Diffraction, Fourier Order |

## B

| Term | Definition | Related Topics |
|------|-----------|----------------|
| BARL (Bottom Anti-Reflection Layer) | A thin dielectric film deposited at the silicon interface to reduce optical reflection and increase light transmission into the photodiode. Thickness and refractive index are tuned for target wavelengths. | Fresnel Equations, QE |
| Bayer Pattern / CFA | The most common color filter array arrangement placed over image sensor pixels, using a repeating pattern of red, green, and blue filters. Standard layouts include RGGB, GRBG, BGGR, and GBRG. | CFA, Crosstalk |
| Beer-Lambert Law | The exponential decay law describing how light intensity decreases as it passes through an absorbing medium: $I(z) = I_0 e^{-\alpha z}$. Fundamental for computing photon absorption depth profiles in silicon. | Absorption Coefficient, QE |
| Bloch/Floquet Theorem | States that electromagnetic fields in periodic structures are themselves periodic, up to a phase shift determined by the wave vector. Forms the theoretical basis of RCWA. | Floquet Mode, RCWA, Periodic Boundary Conditions |
| BSI (Back-Side Illumination) | A sensor architecture in which light enters the silicon substrate from the back side (opposite the metal wiring layers), allowing higher fill factor and improved quantum efficiency compared to FSI. | FSI, QE, Pixel Pitch |
| Boundary Conditions (PML, PBC, Bloch) | Mathematical conditions applied at the edges of a finite simulation domain to model an infinite or semi-infinite physical space. PML absorbs outgoing waves, PBC enforces periodicity, and Bloch conditions handle oblique-incidence periodicity. | PML, Unit Cell, CRA |

## C

| Term | Definition | Related Topics |
|------|-----------|----------------|
| CFA (Color Filter Array) | An array of patterned color filters (typically RGB) placed above the photodiodes to enable color imaging. Each pixel sees only one color; full-color images are reconstructed via demosaicing. | Bayer Pattern, Crosstalk |
| CIS (CMOS Image Sensor) | An image sensor fabricated using complementary metal-oxide-semiconductor (CMOS) technology, integrating photodiodes and readout circuits on the same chip. | BSI, FSI, Photodiode |
| CRA (Chief Ray Angle) | The angle between the chief (central) ray from the imaging lens and the normal to the sensor surface. Varies across the sensor and affects pixel optical design, especially microlens offset. | Microlens, Snell's Law |
| Courant Condition (CFL) | The Courant-Friedrichs-Lewy stability criterion that constrains the maximum allowable time step in FDTD simulations relative to the spatial grid size: $c \Delta t \leq \Delta x / \sqrt{d}$ (in $d$ dimensions). | FDTD, Yee Grid |
| Crosstalk | Unwanted leakage of light from one pixel into an adjacent pixel, degrading spatial resolution and color accuracy. Can be optical (through the stack) or electrical (carrier diffusion). | DTI, CFA, Pixel Pitch |

## D

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Deep Trench Isolation (DTI) | Narrow, oxide-filled trenches etched between adjacent pixels in a CMOS image sensor to physically block optical and electrical crosstalk. | Crosstalk, BSI |
| Diffraction | The bending and spreading of electromagnetic waves when they encounter obstacles or apertures with dimensions comparable to the wavelength. Governs light behavior in sub-wavelength pixel structures. | Diffraction Order, Grating Equation, Huygens' Principle |
| Diffraction Order | An integer index ($m$) labeling the discrete directions into which light is diffracted by a periodic structure, as determined by the grating equation. | Grating Equation, RCWA, Fourier Order |
| Dispersion | The dependence of a material's optical properties (refractive index, extinction coefficient) on wavelength. Must be accounted for in broadband simulations. | Refractive Index, Extinction Coefficient, Permittivity |

## E

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Eigenvalue Problem | The core mathematical problem solved within each layer during RCWA computation. The eigenvalues and eigenvectors of the coupling matrix determine the propagating and evanescent modes in each layer. | RCWA, Fourier Order |
| Electromagnetic Wave | A self-propagating wave of coupled, oscillating electric (**E**) and magnetic (**H**) fields, governed by Maxwell's equations. Light is an electromagnetic wave in the optical frequency range. | Maxwell's Equations, Poynting Vector |
| Extinction Coefficient ($k$) | The imaginary part of the complex refractive index ($\tilde{n} = n + ik$). Quantifies how strongly a material absorbs light at a given wavelength. | Refractive Index, Absorption Coefficient, Permittivity |

## F

| Term | Definition | Related Topics |
|------|-----------|----------------|
| FDTD (Finite-Difference Time-Domain) | A numerical method that solves Maxwell's equations directly in the time domain on a discrete (Yee) grid. Naturally handles broadband and nonlinear problems. One of the two primary solvers in COMPASS. | Yee Grid, Courant Condition, PML |
| Floquet Mode | A characteristic field pattern in a periodic structure whose spatial profile repeats with a phase factor determined by the Bloch wave vector. Each diffraction order corresponds to a Floquet mode. | Bloch/Floquet Theorem, Diffraction Order |
| Fourier Order ($N$) | The number of Fourier harmonics retained in the RCWA plane-wave expansion. Higher $N$ increases accuracy but also computational cost (matrix size scales as $(2N+1)^2$ in 2D). | RCWA, Eigenvalue Problem, Gibbs Phenomenon |
| Fresnel Equations | Analytical expressions for the reflection and transmission coefficients of light at a planar interface between two media, as functions of incidence angle and polarization. | Snell's Law, TE/TM Polarization, S-Matrix |
| FSI (Front-Side Illumination) | The traditional sensor architecture in which light enters from the same side as the metal interconnect layers. Metal wiring partially blocks incoming light, limiting fill factor. | BSI, QE |

## G

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Gibbs Phenomenon | Oscillatory ringing artifacts that appear near sharp discontinuities in a truncated Fourier series representation. In RCWA, it can degrade convergence when representing abrupt material boundaries. | Fourier Order, Li's Factorization Rules, RCWA |
| Grating Equation | The fundamental relation $n_t \sin\theta_m = n_i \sin\theta_i + m\lambda / \Lambda$ linking the diffraction angles ($\theta_m$) to the grating period ($\Lambda$), wavelength ($\lambda$), and diffraction order ($m$). | Diffraction Order, Diffraction, Snell's Law |

## H

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Huygens' Principle | The concept that every point on a wavefront acts as a source of secondary spherical wavelets, and the new wavefront is the envelope of these wavelets. Provides intuitive understanding of diffraction and propagation. | Diffraction, Angular Spectrum |

## L

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Li's Factorization Rules | Correct rules for computing Fourier coefficients of products of discontinuous functions, essential for achieving proper convergence in RCWA (especially for TM polarization). Introduced by Lifeng Li in 1996. | RCWA, Fourier Order, Gibbs Phenomenon |

## M

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Maxwell's Equations | The four fundamental partial differential equations governing all classical electromagnetic phenomena. Both RCWA and FDTD are numerical methods for solving these equations. | Electromagnetic Wave, RCWA, FDTD |
| Microlens | A small curved lens fabricated on top of each pixel to collect and focus incoming light onto the photodiode, improving quantum efficiency. Shape is often modeled as a superellipse. | Superellipse, CRA, QE |

## P

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Permittivity ($\varepsilon$) | A material property relating the electric field **E** to the electric displacement field **D**. Related to the complex refractive index by $\varepsilon = (n + ik)^2$. | Refractive Index, Extinction Coefficient, Dispersion |
| Photodiode (PD) | The light-sensitive semiconductor region within each pixel that absorbs photons and generates electron-hole pairs (photocurrent). The fundamental detection element of a CMOS image sensor. | QE, BSI, Beer-Lambert Law |
| Pixel Pitch | The center-to-center distance between adjacent pixels, typically measured in micrometers. Determines spatial resolution and sets the periodic unit cell size in simulations. | Unit Cell, Microlens, Crosstalk |
| Planarization Layer | A dielectric layer deposited and polished to create a flat surface above the color filter array, providing a uniform substrate for microlens fabrication. | Microlens, CFA |
| PML (Perfectly Matched Layer) | An artificial absorbing boundary layer surrounding the simulation domain, designed to absorb outgoing electromagnetic waves with minimal reflection. Used in both FDTD and RCWA. | Boundary Conditions, FDTD |
| Poynting Vector | The vector $\mathbf{S} = \mathbf{E} \times \mathbf{H}$ representing the directional energy flux (power per unit area) of an electromagnetic wave. Used to compute power flow and absorption in sensor structures. | Electromagnetic Wave, Maxwell's Equations, QE |

## Q

| Term | Definition | Related Topics |
|------|-----------|----------------|
| QE (Quantum Efficiency) | The ratio of the number of charge carriers (electrons) collected by a photodiode to the number of incident photons. The primary figure of merit for image sensor pixel optical performance. | Photodiode, BSI, Absorption Coefficient |

## R

| Term | Definition | Related Topics |
|------|-----------|----------------|
| RCWA (Rigorous Coupled-Wave Analysis) | A frequency-domain numerical method that solves Maxwell's equations in Fourier space for layered periodic structures. Decomposes fields into plane-wave harmonics and solves an eigenvalue problem in each layer. One of the two primary solvers in COMPASS. | Fourier Order, Eigenvalue Problem, S-Matrix |
| Refractive Index ($n$) | The ratio of the speed of light in vacuum to its phase velocity in a medium. The real part of the complex refractive index $\tilde{n} = n + ik$. | Extinction Coefficient, Permittivity, Snell's Law |

## S

| Term | Definition | Related Topics |
|------|-----------|----------------|
| S-Matrix (Scattering Matrix) | A numerically stable algorithm for cascading the transfer matrices of multiple layers, avoiding the exponential overflow problems of the direct transfer-matrix method. Essential for deep multilayer stacks. | RCWA, Fresnel Equations |
| Snell's Law | The relation $n_1 \sin\theta_1 = n_2 \sin\theta_2$ governing the refraction angle of light crossing an interface between two media with different refractive indices. | Fresnel Equations, Refractive Index, Total Internal Reflection |
| Staircase Approximation | The technique of discretizing curved or sloped surfaces (e.g., microlenses) into a stack of flat, laterally uniform layers for RCWA computation. Accuracy improves with finer slicing. | RCWA, Superellipse, Microlens |
| Superellipse | A generalized ellipse described by $\|x/a\|^p + \|y/b\|^p = 1$, used to parameterize microlens cross-sectional profiles. The exponent $p$ controls the shape from diamond ($p<2$) to rounded square ($p>2$). | Microlens, Staircase Approximation |

## T

| Term | Definition | Related Topics |
|------|-----------|----------------|
| TE/TM Polarization | Two fundamental polarization states for light incident on a layered structure. TE (transverse electric) has the electric field parallel to the interface; TM (transverse magnetic) has the magnetic field parallel to the interface. | Fresnel Equations, Maxwell's Equations |
| Total Internal Reflection (TIR) | The complete reflection of light at an interface when the angle of incidence exceeds the critical angle $\theta_c = \arcsin(n_2/n_1)$ and $n_1 > n_2$. Relevant for light trapping in high-index silicon. | Snell's Law, Refractive Index |

## U

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Unit Cell | The smallest repeating unit of a periodic structure. In image sensor simulation, typically corresponds to one or a small group of pixels. Defines the simulation domain for RCWA. | Pixel Pitch, Bayer Pattern, Bloch/Floquet Theorem |

## Y

| Term | Definition | Related Topics |
|------|-----------|----------------|
| Yee Grid/Cell | The staggered spatial grid introduced by Kane Yee (1966) in which **E** and **H** field components are offset by half a grid cell in both space and time. The standard discretization scheme for FDTD. | FDTD, Courant Condition |
