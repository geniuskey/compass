# Understanding Quantum Efficiency

Quantum Efficiency answers one simple question: out of all the photons that hit this pixel, how many actually got detected?

If 100 photons arrive at the pixel surface and 60 of them successfully generate electrons in the photodiode, the Quantum Efficiency (QE) is 60%. The remaining 40 photons were lost somewhere along the way — reflected, absorbed in the wrong layer, leaked into a neighbor, or passed straight through the silicon.

QE is the single most important performance metric for an image sensor pixel. It directly determines how sensitive the camera is: a higher QE means more signal from the same amount of light, which means cleaner images, especially in dim conditions. When camera engineers evaluate a new pixel design, QE is the first number they look at.

It is worth noting that QE is a property of the pixel at a specific wavelength. A single pixel does not have "a QE" — it has a QE spectrum, a curve that tells you how efficiently the pixel detects each wavelength of light from ultraviolet through visible to near-infrared. When someone says "this sensor has 80% QE," they typically mean the peak value of that curve, which usually occurs in the green wavelength range (around 530-560 nm).

---

## The Photon Budget

Let's follow 100 photons as they enter a pixel and see where each one ends up. This "photon budget" reveals exactly where light is gained and lost. It is the accountant's view of pixel physics — every photon must be accounted for.

Imagine 100 green photons (550 nm) arriving at a typical modern BSI pixel:

**Step 1: Microlens surface reflection**

When light crosses from air (refractive index 1.0) into the microlens material (refractive index ~1.5), some reflects back. This is a fundamental consequence of the refractive index mismatch — it happens at every glass or plastic surface you encounter in daily life. Even with optimized lens curvature, about 4 photons out of 100 bounce off the surface and are lost.

Remaining: **96 photons**

**Step 2: Microlens focusing**

The microlens bends the remaining photons inward, concentrating them toward the center of the pixel. Most are focused correctly, but a few are directed toward the edges or into the gaps between pixels — areas occupied by the metal grid or DTI structures. Let's say 2 photons miss the active area entirely.

Remaining: **94 photons**

**Step 3: Color filter absorption**

The photons pass through the color filter. For a green photon hitting a green filter, most are transmitted — but the filter is not perfectly transparent, even at its peak wavelength. The dye molecules in the filter absorb some of the "right" color along with all of the "wrong" colors. The filter absorbs about 10 photons.

Remaining: **84 photons**

**Step 4: Interface reflections (filter to BARL to silicon)**

As photons cross from the color filter into the BARL and then into silicon, some reflect at each interface. The BARL's job is to minimize this, and a well-designed multi-layer BARL does an excellent job — but it cannot eliminate reflection entirely. Perhaps 3 photons are reflected back upward and lost.

Remaining: **81 photons**

**Step 5: Absorption in silicon**

The photons enter the silicon. At 550 nm (green), silicon absorbs efficiently within the first 1-2 micrometers. Most of the 81 photons are absorbed and generate electron-hole pairs. Perhaps 3 photons pass through the silicon without being absorbed (this number would be much higher for red or NIR light, and nearly zero for blue).

Remaining: **78 photons absorbed in silicon**

**Step 6: Collection efficiency**

Not every electron generated in the silicon is collected by the photodiode. Some are generated too far from the junction, or in regions where the electric field is weak. Others recombine before they can be swept into the collection node. A few diffuse sideways into neighboring pixels (electrical crosstalk). Perhaps 5 photons effectively generate electrons that are lost to these mechanisms.

Detected: **73 photons**

**Final QE: 73%**

This is a reasonable number for a modern smartphone sensor pixel at green wavelengths. Premium sensors from leading manufacturers can achieve 80% or higher through careful optimization of every single layer.

<EnergyBalanceDiagram />

The key insight is that no single layer is responsible for most of the loss. Losses are distributed across multiple mechanisms, each contributing a few percent. Improving QE requires reducing losses at every stage simultaneously. This is what makes pixel optimization challenging — and why simulation tools like COMPASS are valuable. You need to see the full picture, not just one layer at a time.

::: details Math Detail
Energy conservation requires that for each wavelength:

R + T + A_filter + A_silicon(photodiode) + A_silicon(outside PD) + Crosstalk = 1

where R is total reflectance (summed over all interfaces), T is transmittance through the bottom of the silicon, A_filter is absorption in the color filter, A_silicon(photodiode) is useful absorption (this becomes QE), A_silicon(outside PD) is wasted silicon absorption (absorbed but not collected), and Crosstalk is energy that ends up detected in neighboring pixels. COMPASS computes all of these terms and verifies that they sum to unity.
:::

---

## Why Wavelength Matters

QE is not a single number — it varies dramatically with wavelength. A pixel might detect 75% of green photons but only 30% of near-infrared photons. Understanding why requires knowing how each layer in the pixel stack responds to different wavelengths.

### Blue Light (450 nm)

Blue photons have the shortest wavelength in the visible range. In silicon, they are absorbed very quickly — within about 0.2 micrometers of the surface. This means even a thin silicon layer captures nearly all blue photons, which is good for absorption completeness.

However, blue light has two disadvantages. First, shorter wavelengths diffract more readily around small structures. In a 1-micrometer pixel, diffraction can spread blue light beyond the pixel boundary, increasing crosstalk. The metal grid edges, DTI walls, and microlens boundaries all act as diffracting apertures.

Second, the blue color filter must reject a broad range of wavelengths (all of green and red), and the filter is not perfectly transparent even at its peak blue transmission. Blue filters tend to have lower peak transmission than green or red filters.

On the positive side, because blue is absorbed so close to the surface, there is very little opportunity for the generated electron to diffuse sideways — it is collected almost immediately. So electrical crosstalk is minimal for blue.

Typical blue QE: **50-65%**

### Green Light (550 nm)

Green represents the sweet spot for silicon imaging. The absorption depth in silicon (~1-2 micrometers) is well matched to typical silicon thickness. Diffraction effects are moderate — not as severe as blue, not as benign as red. The color filter transmission at peak green wavelengths is usually the best of the three colors, often exceeding 85%.

Green pixels also benefit from having two representatives in each Bayer group (RGGB), but this is an image processing advantage rather than a physics one — each individual green pixel has the same QE as it would in any other pattern.

The combination of good silicon absorption, moderate diffraction, and high filter transmission makes green the highest-QE color in virtually every sensor design.

Typical green QE: **70-85%**

### Red Light (650 nm)

Red photons need more silicon to be fully absorbed — about 3 micrometers or more. If the silicon layer is too thin, red photons pass through without being detected, and QE drops. In a sensor with 3 micrometers of silicon, roughly 60% of 650 nm light is absorbed; with 5 micrometers, about 80%.

On the positive side, longer wavelengths diffract less, so red light stays better contained within the pixel boundary. Red filters also tend to have good peak transmission.

The main challenge for red is absorption completeness. Modern sensors with 3-6 micrometer silicon thickness achieve reasonable red QE, but there is always a trade-off with thickness-related costs, fabrication complexity, and crosstalk. Deeply absorbed red photons generate electrons far from the surface where DTI may not extend (in partial-DTI designs), leading to increased electrical crosstalk.

Typical red QE: **55-75%**

### Near-Infrared (850 nm and beyond)

NIR is where silicon's limitations become severe. The absorption coefficient drops dramatically beyond 800 nm. At 850 nm, a photon can travel 20 micrometers in silicon before being absorbed — far more than the 3-6 micrometer thickness of a typical pixel. At 940 nm (used in face recognition systems and LiDAR time-of-flight sensors), the absorption length exceeds 50 micrometers.

Standard image sensor silicon is simply not thick enough to capture most NIR photons. This is why dedicated NIR sensors often use special architectures — thicker silicon (10-20 micrometers), different semiconductor materials (InGaAs for wavelengths beyond 1100 nm), or resonant structures that trap light and increase the effective optical path length through multiple internal reflections.

For standard sensors, NIR QE is always low. Applications that rely on NIR (like face-unlock in smartphones, which typically uses 940 nm) must compensate with powerful IR emitters and sensitive readout circuits.

Typical NIR QE at 850 nm: **15-35%**
Typical NIR QE at 940 nm: **5-15%**

### The QE Spectrum Shape

When you plot QE across all wavelengths, the resulting curve typically shows:

- Low QE in the UV (below 400 nm) due to surface absorption losses and poor filter transmission
- Rising QE through blue as filter transmission improves
- A broad peak somewhere around 530-600 nm where silicon absorption, filter transmission, and diffraction effects are all favorable
- Gradually falling QE through red as silicon absorption becomes incomplete
- A steep drop-off in the NIR beyond 800 nm as silicon becomes nearly transparent

This characteristic curve shape is fundamental to silicon-based imaging. It is one of the primary outputs of a COMPASS simulation, and comparing the QE spectrum across different pixel designs is the most common way to evaluate and optimize pixel performance.

When comparing two pixel designs, the differences in their QE curves tell a clear story. If design A has higher QE than design B at 650 nm, it means design A captures more red light — perhaps because it has thicker silicon or a better-optimized BARL for that wavelength range. If design B has higher QE at 450 nm, it might have a better microlens shape that reduces blue-light diffraction losses. Reading QE curves is a core skill for pixel engineers.

<QESpectrumChart />

---

## Real-World Impact

Why does a few percent of QE matter? Because QE directly translates to image quality, especially in the conditions where it matters most.

### Signal and Noise

In a camera, the signal is the number of detected photons (which is proportional to QE). The dominant noise source in low light is "shot noise" — the inherent statistical variation in photon arrival. If you expect 100 photons on average, you might actually get 90 or 110 in any given exposure. Shot noise follows a Poisson distribution and scales as the square root of the signal.

The signal-to-noise ratio (SNR) therefore scales as the square root of QE. Doubling QE improves SNR by about 40%. This is a meaningful and perceptible improvement — often the difference between a grainy, noisy photo and a clean one. In camera reviews, a 40% SNR improvement is immediately visible in side-by-side comparisons.

::: details Math Detail
For shot-noise-limited imaging:

SNR = signal / noise = (QE * N) / sqrt(QE * N) = sqrt(QE * N)

where N is the number of incident photons. Thus SNR is proportional to sqrt(QE), and improving QE by a factor of 2 improves SNR by sqrt(2) ~ 1.41, which is a 3 dB improvement.
:::

### Low-Light Photography

QE improvements matter most when light is scarce. In bright daylight, even a sensor with 30% QE captures thousands of photons per pixel per frame — plenty for a clean image. But at night, indoors without flash, or in deep shadow, each pixel might receive only a few tens of photons. In these conditions, every detected photon counts.

The difference between 60% QE and 80% QE means the higher-QE sensor gets 33% more signal from the same scene. That extra signal translates to visibly less noise, better color accuracy, and more detail in dark regions.

This is why flagship smartphone cameras invest heavily in pixel optimization. The "night mode" capabilities that consumers now take for granted are built on decades of QE improvement, combined with computational photography techniques that further amplify the signal.

### 20 Years of Progress

The history of CMOS image sensor QE is a story of relentless engineering:

- **Early 2000s (FSI, large pixels, 5-8 um pitch)**: ~25-35% peak QE. Front-side illumination meant light had to pass through metal wiring layers before reaching the silicon. Large portions of the pixel area were blocked by metal routing.
- **Late 2000s (BSI introduced, 1.75-2.2 um pitch)**: ~45-55% peak QE. Moving to back-side illumination removed the wiring obstacle and immediately boosted QE by 15-20 points. This was a paradigm shift in sensor architecture.
- **Mid 2010s (advanced BSI with DTI, 1.0-1.4 um pitch)**: ~60-70% peak QE. Deep trench isolation reduced crosstalk, better anti-reflection coatings reduced losses, and optimized microlens profiles improved light collection.
- **2020s (sub-micron pixels, full DTI, advanced ARC, 0.56-0.8 um pitch)**: ~75-85%+ peak QE. Mature full-depth DTI, precisely optimized microlenses, advanced multi-layer BARL stacks, and nanometer-level thickness control push QE toward the theoretical limits of a color-filtered silicon pixel.

Each of these advances was guided by simulation. Engineers could not afford to fabricate hundreds of test chips to try every possible layer thickness, microlens shape, and DTI dimension. Simulation tools allowed them to explore the design space computationally, identify promising configurations, and validate designs before committing to silicon fabrication.

This is exactly the workflow that COMPASS supports.

### The Physics Limit

Is there a theoretical maximum QE for a silicon pixel? For a monochrome (unfiltered) pixel with perfect anti-reflection coatings, infinitely thick silicon, and perfect carrier collection, the theoretical limit approaches 100%. But real pixels have color filters that inherently absorb a large fraction of incident light. For a Bayer-pattern pixel, the theoretical maximum is typically in the range of 85-95%, depending on the color and the filter characteristics. Reaching even 80% in production is a remarkable engineering achievement.

---

## What Affects QE

Every layer in the pixel stack influences QE. Here is a summary of the key levers that sensor designers — and COMPASS users — can adjust:

### Microlens Design
- **Lens height and curvature**: Controls the focal length and how tightly light is concentrated. Too much curvature can cause light to focus above or below the photodiode (defocus). Too little curvature fails to concentrate light away from dead zones.
- **Fill factor**: How much of the pixel area is covered by the lens. Gaps between microlenses waste light. Modern sensors achieve >95% microlens fill factor.
- **CRA matching**: Mismatched CRA compensation causes light to miss the photodiode at the sensor edges, leading to brightness falloff (shading) in the corners of the image.

### Anti-Reflection Coatings
- **Number of layers**: More layers provide broader wavelength coverage but add manufacturing complexity. Two to four layers are typical.
- **Layer thicknesses**: Each layer must be precisely tuned. A few nanometers of error can shift the anti-reflection band and degrade QE at certain wavelengths.
- **Material choice**: The refractive indices of coating materials determine the achievable reflection reduction. Higher-index materials (like HfO2 or TiO2) provide stronger anti-reflection effects but may have limited transparency at short wavelengths.

### Silicon Thickness
- **Thicker silicon**: Absorbs more red and NIR photons, improving long-wavelength QE. A sensor designed for NIR applications may use 6+ micrometers of silicon.
- **Thinner silicon**: Reduces crosstalk (photons have less distance to drift sideways) and manufacturing cost. Visible-only sensors may use 3 micrometers.
- The optimal thickness depends on the target application. A sensor designed for visible photography (400-700 nm) needs less silicon than one designed for NIR sensing (800-950 nm).

### Color Filter Properties
- **Filter transmission**: Higher peak transmission means less light wasted in the filter. Modern pigment-based filters have peak transmission of 85-95%.
- **Filter bandwidth**: A narrower filter provides better color purity but transmits fewer photons total. There is a fundamental trade-off between color accuracy and sensitivity.
- **Filter thickness**: Thinner filters have less absorption at peak wavelengths but may provide less rejection of off-band light, degrading color purity.

### DTI Effectiveness
- **Trench depth**: Full-depth DTI provides the best isolation but is more challenging to fabricate. Partial DTI leaves leakage paths at the bottom.
- **Trench fill material**: Determines the optical reflection coefficient at the trench wall. Higher refractive index contrast (silicon vs. oxide) means better optical isolation. Some designs use metal-filled DTI for even higher reflectivity.
- **Trench width**: Wider trenches provide better isolation but steal area from the photodiode, reducing the active collection volume and potentially lowering QE.

---

## QE vs. Crosstalk Trade-off

One of the most important tensions in pixel design is the relationship between QE and crosstalk. In many cases, improving one degrades the other. This is not a bug in the physics — it is a fundamental design constraint.

**Example: Microlens focusing**

A strongly curved microlens focuses light tightly onto the center of the pixel. This improves QE by concentrating photons on the photodiode and away from dead zones. But if the focus is too tight, the light cone diverges rapidly after the focal point and can spill into neighboring pixels as it enters the silicon. A more gently curved lens spreads the light more evenly, reducing crosstalk but also reducing peak collection efficiency.

The optimal lens shape depends on all the layers below it — filter thickness, BARL design, silicon thickness, and DTI depth all affect where light needs to be directed for the best overall performance.

**Example: Silicon thickness**

Thicker silicon absorbs more photons (higher QE for long wavelengths), but deeply absorbed photons generate electrons far from the surface, where the electric field is weaker. These electrons are more likely to diffuse sideways into a neighboring pixel (higher crosstalk). With full DTI, this trade-off is less severe because the trench walls block lateral diffusion. But without full DTI, thickness optimization becomes a careful balancing act.

**Example: DTI reflections**

DTI walls reflect light back into the correct pixel, which can increase QE by giving poorly directed photons a second chance at absorption. However, if the reflected light bounces at the wrong angle, it can eventually end up in a different neighboring pixel, merely redirecting the crosstalk rather than eliminating it. Multiple reflections between DTI walls can also create resonant effects that are wavelength-dependent, adding complexity to the optimization.

These trade-offs mean that pixel optimization is rarely about maximizing a single parameter. It is about finding the best balance across all metrics simultaneously — maximizing QE while keeping crosstalk below acceptable limits, across all wavelengths and angles of incidence. Simulation makes this tractable — you can sweep thousands of design combinations and find the Pareto-optimal configurations that offer the best trade-offs.

---

## How COMPASS Computes QE

Understanding the conceptual journey of photons through a pixel is important, but at some point, we need actual numbers. Here is how COMPASS turns a pixel description into a QE value:

**Step 1: Define the pixel structure**

You provide a YAML configuration file describing every layer — materials, thicknesses, geometries. COMPASS parses this into an internal representation called a PixelStack, which is a solver-agnostic description of the complete pixel structure.

**Step 2: Choose a solver**

COMPASS supports multiple electromagnetic solvers, each with different strengths. RCWA (Rigorous Coupled-Wave Analysis) solvers decompose the structure into layers and solve Maxwell's equations in the frequency domain. They are fast for layered periodic structures — which is exactly what a pixel array is. FDTD (Finite-Difference Time-Domain) solvers simulate the time evolution of electromagnetic fields on a spatial grid. They handle arbitrary geometries but are typically slower.

You can run the same pixel through multiple solvers to cross-validate results — a unique capability of COMPASS. If two independent solvers produce the same QE spectrum for a given pixel, you can be confident the results are physically accurate rather than numerical artifacts.

**Step 3: Run the simulation**

The solver calculates the electromagnetic field distribution throughout the entire pixel stack for each wavelength of interest. This includes all reflections, diffractions, and interference effects — the full wave-optics picture. No approximations or ray-tracing shortcuts.

**Step 4: Extract absorption**

From the field distribution, COMPASS calculates how much optical power is absorbed in each layer. The critical quantity is the absorption specifically within the photodiode region of the silicon — this is the light that actually generates detectable signal. Absorption in other layers (filter, BARL, non-photodiode silicon) represents losses.

**Step 5: Compute QE**

QE is simply the ratio of power absorbed in the photodiode to the total incident power:

QE(wavelength) = Power absorbed in photodiode / Incident power

This is computed at each wavelength to produce the full QE spectrum. COMPASS can also compute QE as a function of angle, polarization, and pixel position within the array.

**Step 6: Validate with energy conservation**

COMPASS checks energy conservation: the sum of all reflections, transmissions, and absorptions across all layers must equal the incident power (within numerical tolerance, typically less than 1%). If this balance is violated, it indicates a numerical issue with the simulation — perhaps too few Fourier orders in RCWA, or too coarse a grid in FDTD.

::: details Math Detail
For a periodic structure (pixel array), RCWA expresses the fields as a sum of Fourier spatial harmonics (diffraction orders). The absorption in a layer is computed from the Poynting vector flux difference:

A_layer = (S_in - S_out) / S_incident

where S_in and S_out are the power flux entering and leaving the layer. QE = A_photodiode / S_incident. The S-matrix method is used for numerical stability, avoiding the exponential growth issues of the T-matrix approach.
:::

---

## Where to Go Next

You now have an intuitive understanding of what QE is, why it matters, and what physical mechanisms determine it. From here, you can explore:

- **[Theory: Quantum Efficiency](../theory/quantum-efficiency.md)** — For the full mathematical treatment, including derivations of absorption calculations, spectral averaging, and angular dependence.
- **[Guide: First Simulation](../guide/first-simulation.md)** — Set up and run your first COMPASS simulation to see QE computed for a real pixel structure.
- **[Theory: RCWA Explained](../theory/rcwa-explained.md)** — Understand how the RCWA solver works and why it is well suited to periodic pixel structures.

---

## Summary

| Concept | Key Takeaway |
|---|---|
| QE definition | Fraction of incident photons that become detected electrons |
| Typical peak QE | 70-85% for modern BSI sensors at green wavelengths |
| Wavelength dependence | Blue absorbed quickly, green is the sweet spot, red needs thick Si, NIR is hard |
| Main loss mechanisms | Reflection, filter absorption, incomplete Si absorption, crosstalk |
| Why QE matters | Directly determines signal-to-noise, especially in low light |
| Design trade-offs | QE vs. crosstalk, thickness vs. cost, focus vs. spill |
| COMPASS workflow | YAML config -> solver -> field calculation -> absorption -> QE |

Every percentage point of QE improvement translates to better images in the conditions that matter most. Simulation tools like COMPASS make it possible to explore, optimize, and validate pixel designs before committing to fabrication — saving time, money, and silicon.
