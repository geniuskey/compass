# Comprehensive Survey of CMOS Image Sensor (CIS) Technology Trends

> Survey date: 2026-02-11 | Reference for the COMPASS project

---

## 1. Pixel Pitch Scaling History

### 1.1 Key Milestones

The pixel pitch of CMOS image sensors has been reduced by approximately 20x over the past 25 years. The table below summarizes representative pixel pitches and key technological turning points by era.

| Era | Representative Pixel Pitch | Resolution Range | Key Technology Transition | Representative Products/Companies |
|------|---------------|------------|---------------|---------------|
| 2000-2004 | 5.6-12 um | 1-5 MP | FSI 3T/4T APS, 180nm process | Canon D30 (2000), Sony ICX282 |
| 2005-2007 | 2.2-5.6 um | 5-12 MP | Column ADC, shared pixel architecture | Sony Exmor (2007), column A/D commercialization |
| 2008-2010 | 1.4-2.2 um | 8-16 MP | **BSI commercialization**, gapless microlens | Sony BSI (2009), OmniVision OmniBSI |
| 2011-2014 | 1.1-1.4 um | 13-20 MP | Stacked BSI, PDAF (Phase Detection AF) | Sony Exmor RS (2012), beginning of stacked architecture |
| 2015-2017 | 0.9-1.12 um | 16-48 MP | DTI (Deep Trench Isolation), Dual PD | Samsung ISOCELL (2013~), Sony Dual PD |
| 2018-2020 | 0.7-0.9 um | 48-108 MP | Quad Bayer/Nonacell, Full DTI | Samsung HP1 108MP, Sony IMX586 48MP |
| 2021-2023 | 0.56-0.7 um | 108-200 MP | CDTI, 3-layer stacking, meta-optics | Samsung HP3 (0.56um), OV OVB0A (0.56um) |
| 2024-2026+ | 0.5-0.56 um | 200 MP+ | QQBC, on-sensor AI, 2-layer transistor | Sony 200MP QQBC (2025), next-gen research |

### 1.2 Physical Limits

The current 0.56 um pixel pitch is nearly equal to the wavelength of green light (~0.55 um). Fundamental constraints arising in this regime include:

- **Diffraction limit:** When pixel size approaches the wavelength, the Airy disk encroaches on adjacent pixels, degrading optical resolution
- **Photon starvation:** Reduced pixel area decreases the number of collectible photons, lowering SNR (Signal-to-Noise Ratio)
- **Reduced full-well capacity:** Shrinking photodiode volume reduces the maximum storable charge, limiting dynamic range
- **Read noise:** Skipper-in-CMOS technology achieving ~0.15 e- rms has emerged, but at the cost of increased circuit complexity
- **Lens requirements:** Fast (low f-number) lenses become necessary, increasing module design complexity

Due to these limitations, the industry is shifting strategy from simple pixel miniaturization to **stacked architectures + computational photography**.

---

## 2. BSI (Backside Illumination) vs FSI (Frontside Illumination)

### 2.1 Structural Differences

```
FSI (Frontside Illumination)         BSI (Backside Illumination)
 ┌─────────────────┐                 ┌─────────────────┐
 │   Microlens      │                 │   Microlens      │
 │   Color Filter   │                 │   Color Filter   │
 │   ─────────────  │                 │   Photodiode     │  ← Light reaches directly
 │   Metal layers   │  ← Blocks light│   ─────────────  │
 │   (M1-M5)       │                 │   Metal layers   │
 │   Photodiode     │                 │   (M1-M5)       │
 │   Si substrate   │                 │   Si substrate   │
 └─────────────────┘                 │   (thinned)      │
       ↓ Light                       └─────────────────┘
                                           ↓ Light
```

<PixelAnatomyViewer />

In the FSI structure, incident light must pass through multiple metal interconnect layers before reaching the photodiode. Reflection, scattering, and shielding in the metal layers significantly reduce the effective fill factor. This becomes particularly problematic when pixel pitch shrinks below 2 um, as competition for area between metal lines and the photodiode intensifies.

### 2.2 Why BSI Won

| Parameter | FSI | BSI | Improvement |
|------|-----|-----|--------|
| Quantum Efficiency (QE) @550nm | ~40-60% | ~80-90%+ | 1.5-2x |
| Fill factor | 30-60% | ~100% | 1.5-3x |
| CRA tolerance range | Wide | Limited (improving) | - |
| Manufacturing cost | Low | High (wafer thinning) | - |
| Parasitic light sensitivity | High | Low | Significantly improved |
| Crosstalk @1.0um pitch | 15-25% | 5-10% | 2-3x |

### 2.3 Key BSI Manufacturing Process Steps

1. **Wafer thinning:** Grinding/etching the Si substrate to <~3 um -- mechanical strength is a challenge
2. **Carrier wafer bonding:** Bonding to a support substrate before thinning to enable handling
3. **TSV (Through-Silicon Via):** Electrical connections from the backside to the front-side circuitry
4. **Backside processing:** Forming anti-reflection coatings (ARC), color filters, and microlenses on the backside

### 2.4 Current Status

As of 2025, mobile/consumer CIS has essentially completed a **100% transition to BSI**. The global BSI CIS market is approximately $15 billion in 2025, with projected annual growth of over 7.4% through 2033. Sony, Samsung, and OmniVision dominate the market.

---

## 3. Stacked Sensors

### 3.1 Evolution of Stacking Technology

Stacked CIS is a technology where the photoelectric conversion section and signal processing circuits of a sensor are fabricated on separate wafers and then bonded together.

| Generation | Structure | Era | Key Features | Representative Products |
|------|------|------|----------|----------|
| 1st Gen | 2-layer (Pixel + Logic) | 2012~ | Separate pixel and logic wafers | Sony Exmor RS (IMX135) |
| 2nd Gen | 2-layer + DRAM | 2017~ | Stacked DRAM for high-speed readout | Sony IMX400 (Xperia XZs) |
| 3rd Gen | 3-layer (Pixel + Logic + Memory) | 2021~ | Full 3-layer stacking, Global Shutter | Sony IMX900, Exmor T |
| 4th Gen | 2-layer transistor pixel | 2021~ | Separate photodiode/transistor layers | Sony 2-layer transistor (announced 2021) |
| 5th Gen | 3-layer + AI processor | 2025~ | On-sensor AI processing | Sony 200MP + AI (announced 2025) |

<StackVisualizer />

### 3.2 Bonding Technology

The core of stacked sensor technology lies in wafer-to-wafer bonding:

- **Cu-Cu hybrid bonding:** Direct copper pad bonding. Pitch miniaturization progressing to ~3-5 um. Currently the industry mainstream
- **Oxide bonding:** SiO2 surface bonding followed by TSV electrical connections. Used in earlier generations
- **Micro-bump:** Solder bump-based bonding. Large bonding pitch makes it unsuitable for latest products
- **Pixel-level connection:** Miniaturization of Cu-Cu hybrid bonding enables per-pixel vertical connections, allowing each pixel to independently access the underlying logic

### 3.3 Sony's 3-Layer Strategy

Shinji Sahsida, CEO of Sony Semiconductor Solutions, revealed the **long-term roadmap for 3-layer stacked sensors** at a 2025 investor presentation:

- **Separate optimization of photodiode and transistor layers** to approximately double the saturation signal level compared to conventional designs
- **Extended dynamic range:** Dramatic improvement in HDR performance
- **Multimodal sensing + on-chip AI:** Transitioning beyond image processing to intelligent sensing

### 3.4 Samsung's Response

Samsung is developing a **3-layer stacked CIS** targeting mass production for Apple iPhones around 2026. Starting with a 1/2.6-inch 48MP ultra-wide-angle sensor, Samsung aims to challenge Sony's long-standing exclusive supply dominance.

---

## 4. DTI (Deep Trench Isolation) Evolution

### 4.1 The Need for DTI Technology

As pixel pitch shrinks, **optical crosstalk** and **electrical crosstalk** between adjacent pixels increase dramatically. DTI is a critical technology that etches deep trenches into the silicon substrate and fills them with insulating material to achieve pixel-to-pixel isolation.

### 4.2 Generational Evolution of DTI

| Generation | Technology Name | Trench Depth | Fill Material | Crosstalk Improvement | Adoption Era |
|------|----------|------------|----------|---------------|----------|
| Gen 0 | STI (Shallow Trench) | <0.5 um | SiO2 | Baseline | ~2008 |
| Gen 1 | Partial BDTI (Backside DTI) | 1-2 um | SiO2 | 30-50% reduction | 2012~ |
| Gen 2 | Full DTI (F-DTI) | Full depth (~3 um) | SiO2 / Poly-Si | 60-80% reduction | 2015~ |
| Gen 3 | Metal-filled DTI | Full depth | W (tungsten) + liner | 80-90% reduction, light shielding | 2018~ |
| Gen 4 | CDTI (Capacitive DTI) | Full depth | MOS capacitor structure | 90%+ reduction, FWC improvement | 2020~ |
| Gen 5 | Air-gap DTI | Full depth | Air gap + ARC | Maximum isolation, reflection utilization | Research stage |

### 4.3 CDTI (Capacitive DTI) Details

CDTI forms a MOS capacitor structure on the trench sidewalls, providing the following advantages:

- **Ultra-low dark current:** Approximately 1 aA (attoampere) for 1.4 um pixels at 60C
- **High FWC (Full-well Capacity):** ~12,000 e- (for 1.4 um pixels), improved over conventional DTI
- **Quantum efficiency improvement:** QE enhancement achieved without sidewall implants
- **Pinning voltage control:** Active control of the trench sidewall potential to optimize the depletion region

### 4.4 DTI + BSI Synergy

Full DTI delivers maximum effectiveness when combined with BSI:

```
         Light incidence ↓
    ┌──────┬──────┬──────┐
    │ ML   │ ML   │ ML   │  Microlens
    │ CF   │ CF   │ CF   │  Color Filter
    ├──┤   ├──┤   ├──┤   │
    │  │PD │  │PD │  │PD │  Photodiode
    │  │   │  │   │  │   │
    │D │   │D │   │D │   │
    │T │   │T │   │T │   │  DTI (Full depth)
    │I │   │I │   │I │   │
    │  │   │  │   │  │   │
    ├──┤   ├──┤   ├──┤   │
    │  Metal    │  Metal    │  Metal    │
    └──────┴──────┴──────┘
```

In this structure, DTI acts as an optical waveguide, confining light within the photodiode to maximize absorption efficiency. This is particularly effective in the NIR (Near-Infrared) region.

### 4.5 Automotive F-DTI

In automotive image sensors, a **2.1 um Full-Depth DTI** process has been applied, achieving 120 dB dynamic range in a single exposure at 85C junction temperature. By integrating storage capacitors within the pixel, both HDR performance and functional safety requirements are simultaneously met.

---

## 5. Microlens Technology

### 5.1 Role of Microlenses

Microlenses are positioned on top of each pixel and serve as critical optical elements that focus incident light onto the effective photodiode area. As pixel pitch shrinks, the importance of microlens design increases dramatically.

### 5.2 Technology Evolution

| Generation | Technology | Features | Era |
|------|------|------|------|
| 1st Gen | Spherical microlens | Simple hemispherical, gaps between pixels | ~2005 |
| 2nd Gen | Gapless microlens | Eliminated gaps between adjacent lenses, improved light collection efficiency | 2008~ |
| 3rd Gen | Aspherical microlens | Superellipse profile, CRA optimization | 2012~ |
| 4th Gen | Multi-layer microlens | Inner lens + outer lens combination | 2016~ |
| 5th Gen | Meta-optics microlens | Nanostructure-based flat lens, wavelength-selective focusing | Research stage |

### 5.3 Superellipse Profile

The microlens model used in COMPASS:

```
z(x, y) = h * (1 - r^2)^(1/2a)

Where:
  h = Lens height (sag height)
  r = Normalized radius (0 to 1)
  a = Superellipse parameter (a=1: hemisphere, a<1: flattened, a>1: steepened)
```

This profile is critical for optimizing light collection efficiency as a function of CRA (Chief Ray Angle). At high CRA designs, asymmetric profiles are required, which is an important element that must be accurately reproduced in simulation.

### 5.4 CRA (Chief Ray Angle) Optimization

CRA is the angle at which the chief ray of the lens system is incident on the sensor surface:

- **Low CRA (<15 deg):** Light is incident nearly perpendicular -- microlens design is straightforward, QE is uniform
- **High CRA (>25 deg):** Light is incident at oblique angles at the sensor edges -- microlens shift and asymmetric profiles are required
- **CRA matching:** In mobile camera modules, the lens CRA and sensor CRA must be precisely matched to minimize peripheral vignetting

In recent 2024 research, **adjoint sensitivity analysis**-based microlens shape optimization was proposed, enabling computation of the figure of merit gradient with only two electromagnetic simulations.

### 5.5 Meta-optics Microlenses

To overcome the performance limitations of conventional refractive microlenses in the sub-micron pixel era, meta-optics technology is being researched:

- **Nanopillar arrays:** Sub-wavelength pillar structures for phase control
- **Wavelength-selective focusing:** Potential to integrate color filtering and light focusing functions in a single layer
- **Polarization sensitivity:** Selective response to specific polarization states can be achieved

---

## 6. Color Filter Technology

### 6.1 Traditional Bayer Pattern and Its Evolution

| Pattern | Arrangement | Effective Resolution | Binning | Representative Application |
|------|------|-----------|----------------|----------|
| Bayer (RGGB) | 2x2 | 1x | 2x2 → 1/4 | Most CIS |
| Quad Bayer (QQBC) | 4x4 same color | 1x or 1/4x | 4x4 → 1/16 | Sony IMX586+ |
| Nona-Bayer (Nonacell) | 3x3 same color | 1x or 1/9x | 3x3 → 1/9 | Samsung HP1 108MP |
| Hexadeca Bayer | 4x4 same color | 1x or 1/16x | Dual binning possible | Samsung HP3 200MP |
| QQBC (Quad-Quad) | 16px cluster | 1x or 1/16x | 16→1 ultra-high sensitivity | Sony 200MP (2025) |

<BayerPatternViewer />

### 6.2 Advantages of Quad Bayer / Nona-Bayer

By forming clusters of adjacent pixels with the same color:

1. **Low-light mode:** Summing (binning) signals from N pixels for N-fold sensitivity improvement
2. **High-resolution mode:** Reading each pixel independently to utilize full resolution
3. **Phase Detection AF:** Phase difference detection using same-color pixel pairs (Super QPD)
4. **HDR:** Single-frame HDR by setting different exposure times within same-color pixels

### 6.3 Sony QQBC (2025)

The 200MP sensor announced by Sony in 2025 adopted a **Quad-Quad Bayer Coding (QQBC)** arrangement:

- 0.7 um pixel pitch, 1/1.12-inch format
- 16 (4x4) same-color pixel clusters -- 16-pixel binning for high sensitivity in night/indoor conditions
- **On-sensor AI** built in: Resolution restoration and noise reduction processed within the sensor chip
- Maintains high image quality at high-magnification zoom from a monocular camera

### 6.4 Organic Color Filters

Research to overcome limitations of conventional pigment/dye-based color filters:

- **Organic Photodetector (OPD):** Selective wavelength absorption
- **Stacked organic sensors:** Vertically stacking R, G, B layers -- eliminates the need for a Bayer pattern, full color information from every pixel
- **Overcoming Foveon limitations:** Improved color separation through absorption spectrum engineering of organic materials
- **Panasonic/Fujifilm:** Continued research on organic CMOS sensors (commercialization remains limited)

### 6.5 Inorganic Quantum Dot Color Filters (QD Color Filter)

Leveraging the size-dependent absorption/emission properties of quantum dots:

- Precise control of absorption wavelength via quantum dot size (2-10 nm)
- Narrower absorption bands compared to conventional pigment filters -- potential for improved color purity
- Easy extension to SWIR (Short-Wave Infrared): PbS quantum dots for 900-1700 nm detection

---

## 7. Photoelectric Conversion Efficiency (QE) Optimization

### 7.1 Factors Determining QE

Quantum efficiency is the ratio of incident photons converted into electron-hole pairs, determined by the product of the following factors:

```
QE_total = T_lens * T_filter * (1 - R_surface) * eta_absorption * eta_collection

Where:
  T_lens      = Microlens transmittance
  T_filter    = Color filter transmittance
  R_surface   = Surface reflectance
  eta_absorption = Silicon optical absorption efficiency (thickness and wavelength dependent)
  eta_collection = Photogenerated carrier collection efficiency
```

### 7.2 Anti-Reflection Coating (ARC)

| Technology | Reflectance | Wavelength Range | Features |
|------|--------|----------|------|
| Single-layer ARC (SiN) | ~5% | Narrow | Simple, low cost |
| Multi-layer ARC (SiO2/SiN/HfO2) | ~1-2% | 400-700 nm | Standard technology |
| Nanostructured ARC (moth-eye) | <0.5% | 300-1000 nm | Broadband, complex processing |
| 3D nanocone | <1% | Visible | Compatible with flexible substrates, 7% EQE improvement reported |

Recent research (2023) applying surface nanoengineering to commercial BSI CIS achieved **QE above 90% across the 300-700 nm range**. This technology also demonstrated the additional benefit of reducing dark current by 3x.

### 7.3 Light-Trapping Structures

Structures to compensate for insufficient absorption (especially in NIR) due to thin silicon thickness in sub-micron pixels:

- **Inverted Pyramid Array (IPA):** 2D periodic structures formed on the backside of BSI CIS. An **80% improvement** in sensitivity at 850 nm reported for 1.2 um pixels with 400 nm pitch IPA
- **Single hole structure:** Optimally sized holes on the photodiode enhance NIR absorption by **60%** (for 3 um Si)
- **Diffraction grating:** Backside diffractive structures extend the optical path -- increasing the effective absorption thickness
- **DTI optical waveguide:** Total internal reflection at Full DTI sidewalls confines light within the pixel

### 7.4 Energy Conservation Verification

A core physical constraint in COMPASS:

```
R + T + A = 1  (tolerance < 1%)

Where:
  R = Reflectance
  T = Transmittance
  A = Absorptance ≥ QE (some absorption is dissipated as heat)
```

If this energy conservation relationship does not hold in a simulation, numerical errors should be suspected. In particular, energy non-conservation can occur in RCWA when the Fourier order is insufficient.

---

## 8. Next-Generation Technologies

### 8.1 Quantum Dot Image Sensors

2024 was called the **"Year of the Quantum Dot"** as quantum dot sensor technology advanced rapidly:

- **PbS quantum dot SWIR sensors:** Monolithic integration with silicon ROIC (Readout IC). Expanding applications in autonomous driving, food inspection, and medical imaging
- **Global shutter operation:** Global shutter operation demonstrated in PbS QD SWIR sensors
- **Key performance:** Detectivity >4.2x10^17 Jones, responsivity >8.3x10^3 A/W, detection range 365-1310 nm
- **Environmentally friendly synthesis:** Research on fabricating sensors from PbS QDs extracted from waste lead-acid batteries (2024)

### 8.2 Organic Photodetector (OPD)

Photoelectric conversion devices based on organic semiconductors:

- **Advantages:** Easy absorption spectrum tuning, large-area coating capability, flexible substrate compatibility
- **Vertical stacked color separation:** Vertically stacking R/G/B organic layers for full-color information without a Bayer pattern
- **Challenges:** CMOS process compatibility, long-term stability, dark current control
- **Perovskite photodetectors:** Organic-inorganic hybrid materials with high absorption coefficients and long carrier lifetimes

### 8.3 Event-Based Vision Sensor (EVS)

EVS/DVS technology commercialized through the collaboration between Sony and Prophesee:

- **Operating principle:** Each pixel asynchronously detects luminance changes, outputting only the coordinates and timestamp of pixels where changes occur
- **Sony IMX636:** 1280x720 HD, industry-smallest 4.86 um pixel, dynamic range 86 dB+ (5-100,000 lux)
- **Maximum event rate:** 1.06 Giga-events/sec
- **Advantages:** Ultra-low latency (<1 us), low power, low bandwidth (no data when the scene is static)
- **Applications:** Industrial inspection, autonomous driving, robot vision, gesture recognition

IDS launched an industrial event camera series (uEye XCP-E) based on the Sony-Prophesee IMX636 in 2025.

### 8.4 Computational Imaging

Overcoming the physical limitations of conventional sensors through hardware-software convergence:

- **On-sensor AI:** AI processor embedded in Sony's 200MP QQBC sensor -- noise reduction and resolution restoration at the sensor level
- **Multi-frame synthesis:** SNR improvement through multi-frame compositing for HDR and low-light photography
- **Depth sensing:** Integration of ToF (Time-of-Flight) and structured light with CIS
- **Neural ISP:** Replacing the traditional ISP pipeline with neural networks -- end-to-end optimization

---

## 9. Simulation Perspective: The Role and Challenges of COMPASS

### 9.1 Impact of Technology Trends on Simulation

Each of the technology trends discussed above imposes new requirements on optical simulation tools like COMPASS:

| Technology Trend | Simulation Requirement | COMPASS Response |
|-----------|-------------------|-------------|
| Sub-micron pixels | Wave optics (RCWA/FDTD) essential, geometrical optics inadequate | Multiple RCWA solvers (torcwa, grcwa, meent) |
| BSI structure | Accurate modeling of multi-layer thin-film interference | PixelStack layer-based structure |
| Stacked sensors | Complex 3D structures, large-scale computation | GPU acceleration (PyTorch/JAX) |
| Full DTI | Accurate permittivity modeling of metal/dielectric trenches | MaterialDB (built-in + CSV + dispersion models) |
| Microlens | Superellipse profile, CRA dependence | GeometryBuilder, cone illumination |
| Color filter | Material dispersion (n, k vs wavelength) | Cauchy/Sellmeier fitting |
| QE optimization | Energy conservation verification (R+T+A=1) | energy_balance.py |
| Meta-optics | Sub-wavelength structures require high Fourier orders | RCWA stability (S-matrix) |

### 9.2 Core Problems COMPASS Solves

1. **Cross-solver validation:** Comparing results from RCWA (torcwa, grcwa, meent) and FDTD (flaport) for the same structure to establish simulation reliability
2. **Single YAML configuration:** Declaratively defining complex pixel structures -- enabling reproducibility and easy parameter sweeps
3. **Solver-agnostic abstraction:** Automating PixelStack-to-solver conversion to facilitate adding new solvers
4. **Automatic physical consistency checks:** Automatic verification of QE range (0-1) and energy conservation (R+T+A=1)

### 9.3 Future Simulation Challenges

Challenges COMPASS must address to prepare for the sub-0.5 um pixel era:

- **Computational cost management:** For 3D FDTD, the computational domain is not large even for sub-micron pixels, but wavelength sweeps and parameter optimization require thousands of iterations
- **Inverse design integration:** Automatic differentiation (AD)-based microlens/meta-optics optimization -- increasing importance of AD-capable solvers such as meent and fmmax
- **Multiphysics coupling:** Integration of optical simulation + carrier transport + circuit simulation
- **Quantum dot/organic material modeling:** Need to expand the complex permittivity database for new photoelectric conversion materials
- **Event sensor simulation:** Time-domain response modeling -- natural synergy with FDTD

### 9.4 Simulation Accuracy Criteria

Key verification criteria for ensuring simulation result reliability in COMPASS:

| Verification Item | Reference Value | Notes |
|----------|--------|------|
| Energy conservation (R+T+A) | = 1 (error < 1%) | Verified at all wavelengths |
| Si refractive index @550nm | n ~ 4.08, k ~ 0.028 | Based on Green 2008 data |
| QE range | 0 <= QE <= 1 | Physical upper/lower bounds |
| Inter-solver QE deviation | < 5% (same conditions) | RCWA vs FDTD cross-validation |
| RCWA convergence | Confirm convergence with increasing Fourier order | S-matrix only (T-matrix not allowed) |
| Crosstalk | Signal leakage ratio to adjacent pixels | Verify difference with/without DTI |

---

## References and Sources

### Industry Reports
- Yole Group, "Status of the CMOS Image Sensor Industry 2025"
- Mordor Intelligence, "Image Sensors Market Size, Trends, Share Analysis 2030"
- IDTechEx, "Quantum Dots Revolutionizing Image Sensors" (2024)

### Corporate Technical Documents
- Sony Semiconductor Solutions, QQBC technology and 200MP sensor announcement (2025)
- Samsung Semiconductor, ISOCELL HP3 0.56um pixel technical document (2022)
- OmniVision, OmniBSI technology white paper
- Prophesee/Sony, IMX636 Event-Based Vision Sensor specifications

### Academic Papers
- "CMOS Image Sensor for Broad Spectral Range with >90% Quantum Efficiency" (Small, 2023)
- "Deep Trench Isolation and Inverted Pyramid Array Structures for CMOS Image Sensor" (Sensors, 2020)
- "Automotive 2.1um Full-Depth DTI CMOS Image Sensor with 120dB Dynamic Range" (Sensors, 2023)
- "Adjoint-Assisted Shape Optimization of Microlenses for CMOS Image Sensors" (PMC, 2024)
- "IR Sensitivity Enhancement of CMOS Image Sensor with Diffractive Light Trapping Pixels" (Scientific Reports, 2017)

### Industry Presentations and News
- IEEE Spectrum, "Samsung and OmniVision Claim Smallest Camera Pixels" (2022)
- SK hynix Newsroom, "Evolution of Pixel Technology in CMOS Image Sensor"
- DPReview, "Tech Timeline: Milestones in Sensor Development"
- Image Sensors World (imagesensors.org), IISW workshop papers

---

> This document is a survey compiled to provide technical context for the COMPASS project and serves as the basis for simulation parameter configuration and validation criteria. Technical data is current as of February 2026; given the rapidly evolving nature of the CIS industry, periodic updates are required.
