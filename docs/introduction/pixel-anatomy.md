# Pixel Anatomy

Let's take a journey through a single pixel — from the top where light enters, to the bottom where it becomes an electrical signal.

A modern smartphone camera sensor contains millions of pixels, each one a carefully engineered stack of optical and electronic layers. Every layer has a specific job: bending light, filtering colors, reducing reflections, or converting photons into electrons. Understanding this stack is the first step toward understanding how image sensor simulation works — and why it matters.

<PixelAnatomyViewer />

In this page, we will walk through a **BSI (Back-Side Illuminated)** pixel architecture from top to bottom. BSI is the dominant architecture in modern sensors because it places the wiring layers behind the silicon, giving light a clear path to the photodetector. Think of it as removing obstacles from the hallway before inviting guests in.

Before BSI became standard, sensors used **FSI (Front-Side Illuminated)** architectures where light had to navigate through layers of metal interconnects and dielectric films before reaching the silicon. This was like trying to catch rainwater through a forest canopy — significant light was blocked or scattered by the wiring above. The shift to BSI in the late 2000s was one of the most important advances in image sensor history, immediately boosting Quantum Efficiency by 15-20 percentage points.

To put the scale in perspective: a typical pixel in a modern smartphone is around 0.7 to 1.0 micrometers wide. That is roughly 1/100th the width of a human hair. The entire optical stack — from the top of the microlens to the bottom of the silicon — is only about 5 to 10 micrometers tall. Yet within that tiny volume, light is focused, filtered by color, anti-reflection coated, absorbed, and converted to electrical charge. It is one of the most densely packed optical systems ever engineered.

---

## The Layer Stack

A pixel is not a single "thing" — it is a stack of thin films and structures, each measured in fractions of a micrometer. The total stack height is typically 5-10 micrometers, which is about one-tenth the width of a human hair. Here is what you encounter as you follow a ray of light from the sky down into the sensor.

<StackVisualizer />

### 1. Air — Where Light Arrives

Everything begins in open air. Light from the scene — whether it is sunlight reflecting off a face, or the glow of a streetlamp — arrives at the sensor surface as an electromagnetic wave.

At this stage, the light contains all colors (wavelengths) mixed together. It is the pixel's job to sort, focus, and measure just the right portion of that light.

The transition from air into the first solid layer is already important. Whenever light crosses from one material to another, some fraction reflects back. The bigger the difference in optical properties between the two materials, the stronger the reflection. This is why you see reflections on glass windows — and why sensor designers work hard to minimize reflections at every interface in the pixel stack.

For a camera, the "air" above the pixel is actually the space between the last element of the camera module's lens system and the sensor surface. In a smartphone, this gap is typically only a fraction of a millimeter. The quality of light arriving at each pixel — its angle, intensity, and uniformity — is shaped by the entire camera optical system above.

One important property of the air-to-solid transition is the Fresnel reflection. For normal incidence (light hitting straight on), the reflection at an air-to-polymer interface is about 4%. This seems small, but remember: every reflected photon is a photon that does not contribute to the image. Across millions of pixels and billions of photons, 4% adds up to a measurable sensitivity loss.

The angle at which light arrives also matters significantly. At the center of the sensor, light comes in nearly perpendicular. At the edges and corners, light can arrive at 20-35 degrees from normal. This oblique incidence changes how light interacts with every layer in the stack — the effective path length through each layer increases, diffraction angles change, and reflection coefficients shift. A pixel design that works well for normal incidence may perform poorly at high angles, and vice versa. This angular dependence is one reason why pixel optimization is so challenging, and why simulation tools that handle arbitrary angles of incidence are essential.

### 2. Microlens — A Tiny Focusing Lens

Sitting on top of each pixel is a miniature lens, typically made of a transparent polymer (such as a photoresist-based resin). These microlenses are incredibly small — roughly the same width as the pixel itself, often around 1 micrometer or less in modern sensors.

**Why is it there?**

Without a microlens, a significant amount of incoming light would land in the gaps between pixels — on top of metal grids, isolation structures, or neighboring pixels — and be wasted. The microlens acts like a funnel that directs rainwater into a bucket. It gathers light from across the full pixel area and focuses it down toward the active detection region at the center.

The improvement is substantial. A well-designed microlens can increase the amount of light reaching the photodiode by 30-50% compared to a flat pixel surface. For small pixels (below 1.5 micrometers), the microlens is not merely helpful — it is essential. Without it, the pixel would be nearly unusable in low light.

**Shape and Profile**

Microlenses are not perfect hemispheres. Their shape is typically described by a "superellipse" profile — a smooth, dome-like curve that can be tuned by adjusting parameters like height, radius of curvature, and a shape exponent. A rounder lens focuses more aggressively, while a flatter lens provides gentler bending. Sensor designers choose the profile that best matches the pixel size and the optics of the camera module sitting above.

The gap between adjacent microlenses also matters. Ideally, microlenses would tile the surface perfectly with no gaps, capturing every incoming photon. In practice, there are always small gaps at the boundaries. Modern fabrication techniques have pushed microlens "fill factors" above 95%, meaning less than 5% of the pixel area is uncovered.

::: details Math Detail
The superellipse profile is given by:

z(r) = h * (1 - (r/R)^2)^(1/(2a))

where h is the lens height, R is the radius, r is the radial distance from center, and a is the shape parameter. When a = 1, this reduces to a standard ellipsoid. Values of a < 1 produce more "boxy" lens shapes with steeper edges, while a > 1 produces more pointed profiles.
:::

**CRA Shift: Off-Center Lenses at the Sensor Edge**

Light arriving at the center of the sensor comes in nearly straight (perpendicular to the surface). But at the corners of the sensor, light arrives at an angle — this angle is called the Chief Ray Angle (CRA). For a typical smartphone camera, the CRA can reach 25-35 degrees at the sensor corners.

To compensate, microlenses at the edge of the sensor are deliberately shifted off-center. The lens is moved slightly toward the center of the sensor so that it still focuses the angled light onto the photodiode below. Without this shift, corner pixels would receive much less light, and your photos would have dark edges (a phenomenon called "vignetting" or "shading").

This shift varies smoothly across the sensor — zero at the center, maximum at the corners. The shift distance is typically a fraction of the pixel pitch. In COMPASS, you can specify the CRA and the simulation automatically accounts for this offset, adjusting the microlens position for each pixel location.

### 3. Planarization Layer — Smoothing the Surface

Between the microlens and the color filter below lies a thin planarization layer, typically 0.1 to 0.5 micrometers thick. This layer serves a straightforward but essential purpose: it provides a flat, smooth surface for the microlens to sit on.

The color filter layer below has a grid structure with slight height variations and gaps between filters. If you tried to form a microlens directly on this uneven surface, the lens shape would be distorted, and its focusing performance would suffer. Even small surface irregularities — on the order of tens of nanometers — can degrade the lens profile enough to measurably reduce QE.

The planarization layer fills in these gaps and creates a uniform foundation. It is typically made of a transparent, low-refractive-index material (often a spin-on polymer or oxide). You can think of it as pouring a thin layer of clear resin over a bumpy surface to make it smooth before placing a lens on top.

Its optical role is modest — it mainly needs to be transparent and have a refractive index that does not cause strong reflections. But getting its thickness right matters for controlling the distance between the microlens and the color filter, which affects how the focused light cone lands on the filter. Too thick, and the focal point shifts below the optimal position. Too thin, and the surface may not be adequately smoothed.

In some advanced designs, the planarization layer also serves as part of the anti-reflection system. By choosing its refractive index to be intermediate between the microlens material and the color filter material, it can help reduce reflection at these interfaces — a small but meaningful improvement when every percentage point of QE counts.

### 4. Color Filter Array (CFA) — Giving Each Pixel Its Color Identity

This is where the pixel gets its "color." A digital camera sensor is inherently colorblind — silicon absorbs all visible wavelengths and cannot tell red from blue. Color information comes entirely from the filter placed above each pixel.

**The Bayer Pattern**

The most common color filter arrangement is the Bayer pattern, invented by Bryce Bayer at Kodak in 1976: a repeating 2x2 grid of Red, Green, Green, Blue (RGGB). If you look at a small section of the sensor, you see:

```
 G  R  G  R  G  R
 B  G  B  G  B  G
 G  R  G  R  G  R
 B  G  B  G  B  G
```

Why two greens and only one red and one blue? Because the human visual system is most sensitive to green light. Our eyes have roughly twice the spatial resolution for green/luminance information compared to red or blue. Having twice as many green pixels provides better perceived sharpness and luminance accuracy. The missing color information for each pixel is filled in later by a process called demosaicing — an algorithm that estimates the full RGB value at each pixel location based on its neighbors.

Other color filter patterns exist — Quad Bayer (2x2 groups of same-color pixels), RYYB (replacing green with yellow for higher sensitivity), and specialized patterns for automotive or machine vision applications. But the standard Bayer RGGB remains by far the most common.

**How the Filter Works**

Each color filter is essentially a thin layer of dyed or pigmented material, typically 0.5 to 1.0 micrometers thick. A red filter absorbs blue and green light while transmitting red wavelengths. A blue filter absorbs red and green. A green filter absorbs red and blue.

The key insight is that filtering means absorbing. The unwanted wavelengths do not pass through and disappear — they get absorbed by the filter material and converted to heat. This is one of the reasons Quantum Efficiency can never reach 100% for a colored pixel: the filter itself removes a large portion of the incoming light.

Even at the peak transmission wavelength, no filter is perfectly transparent. A typical green filter might transmit 85-90% of light at 550 nm, meaning even the "right" color loses 10-15% in the filter.

**Metal Grid Between Filters**

Between adjacent color filters, there is typically a thin metal grid (often tungsten or aluminum), about 0.1 to 0.3 micrometers wide. This grid serves as an optical barrier — it prevents light that passed through one color filter from leaking sideways into a neighboring pixel that has a different color filter.

Without this grid, a red photon that entered through a red filter could scatter sideways and end up being detected by the green pixel next door. This type of optical crosstalk degrades color accuracy and image quality. The metal grid acts as a wall between apartments, keeping each pixel's light contained.

The grid comes with a trade-off: it occupies space that could otherwise collect light. In a 1-micrometer pixel, a 100-nanometer-wide grid on all four sides consumes about 19% of the pixel area. This is one reason why the microlens is so important — it focuses light away from these dead zones and toward the center.

The height and material of the metal grid also affect its performance. A taller grid blocks more angled light but also increases diffraction effects. A lower grid allows more leakage but diffracts less. The grid material's reflectivity matters too — a highly reflective grid can bounce light back into the correct pixel rather than simply absorbing it, potentially recovering some of the "lost" photons. Optimizing grid dimensions is a multi-variable problem that COMPASS handles by modeling the grid as part of the full electromagnetic simulation.

### 5. BARL (Bottom Anti-Reflection Layer) — Reducing Interface Reflections

When light exits the color filter and enters the silicon below, it encounters a large change in refractive index. Air has a refractive index of 1.0, typical filter materials are around 1.5-1.7, and silicon has a refractive index of about 4.0 at visible wavelengths. That jump from ~1.6 to ~4.0 causes significant reflection — similar to how you see a strong reflection when looking at the surface of a pool of water, but much worse.

The BARL is a thin-film stack designed to reduce this reflection. It works on the same principle as anti-glare coatings on eyeglasses or camera lenses.

**How It Works**

The BARL typically consists of alternating layers of high-refractive-index and low-refractive-index materials — for example, hafnium oxide (HfO2, n ~ 2.0) and silicon dioxide (SiO2, n ~ 1.46). Each layer is extremely thin, on the order of 30-80 nanometers. By carefully choosing the thickness of each layer, designers create constructive interference for the transmitted light and destructive interference for the reflected light.

In simple terms: the BARL creates a "gradient" that eases the optical transition from the low-index filter to the high-index silicon. Instead of one abrupt jump (which causes a big reflection), light steps through a series of smaller jumps (which causes much smaller reflections that partially cancel each other out).

A single-layer ARC can be optimized for one specific wavelength. But pixels need to work across a broad wavelength range (the full band of the color filter above). Multi-layer designs with 2-4 alternating films provide good anti-reflection performance across the entire visible spectrum, at the cost of more complex manufacturing.

::: details Math Detail
For a single-layer anti-reflection coating, the ideal refractive index is:

n_ARC = sqrt(n1 * n2)

where n1 and n2 are the indices of the surrounding layers. The ideal thickness is one quarter of the target wavelength divided by the ARC refractive index (quarter-wave condition):

d = lambda / (4 * n_ARC)

Multi-layer designs extend this principle for broader wavelength coverage, using transfer matrix methods to optimize layer thicknesses simultaneously.
:::

**Why It Matters**

Without the BARL, reflections at the silicon interface could waste 30% or more of the light that made it through the color filter. With a well-designed BARL, reflection losses can be reduced to just a few percent across the target wavelength range. For QE optimization, this is one of the most impactful layers to get right — small changes in BARL layer thickness (even 5-10 nanometers) can shift QE by several percentage points.

### 6. Silicon Substrate — Where Photons Become Electrons

This is the heart of the pixel. Everything above exists to deliver light here. The silicon substrate is where the actual detection happens — photons are absorbed and generate electron-hole pairs that are collected and measured.

**The Photodiode**

Embedded within the silicon is a photodiode — a junction between differently doped regions of silicon (typically a p-n junction). When a photon is absorbed in or near this junction, it creates a free electron (and a corresponding "hole"). An electric field in the junction region sweeps these carriers apart and collects the electron, adding it to the pixel's signal charge.

The photodiode does not occupy the entire silicon volume. It is a defined region, typically concentrated in the upper portion of the silicon (the side where light enters in a BSI device). The depth and lateral extent of the photodiode determine the "collection volume" — the region where generated electrons have a high probability of being captured.

The more photons absorbed in the photodiode region, the larger the signal. This is the core of what Quantum Efficiency measures.

**Absorption Depth and Wavelength**

Silicon does not absorb all wavelengths equally. This is one of the most important concepts in pixel physics:

- **Blue light (450 nm)**: Absorbed very quickly, within about 0.2 micrometers of the surface. Almost all blue photons are captured in a very thin layer. This is why even thin silicon works well for blue.
- **Green light (550 nm)**: Absorbed within about 1-2 micrometers. A good balance between quick absorption and reasonable silicon thickness. This is the "sweet spot" for silicon imaging.
- **Red light (650 nm)**: Needs about 3 or more micrometers of silicon to be mostly absorbed. Red photons penetrate deeper before being captured. A 3-micrometer silicon layer absorbs about 60% of 650 nm light; 5 micrometers captures about 80%.
- **Near-infrared (850 nm+)**: Can pass through several micrometers of silicon without being absorbed at all. Many NIR photons simply exit the bottom of the pixel undetected. At 850 nm, the absorption length in silicon is about 20 micrometers — far thicker than a typical pixel.

This is why silicon thickness is a critical design parameter. Thicker silicon captures more red and NIR light but costs more and can increase crosstalk (photons that travel deep have more opportunity to drift sideways into neighboring pixels).

::: details Math Detail
The absorption follows Beer-Lambert's law:

I(z) = I0 * exp(-alpha * z)

where alpha is the absorption coefficient of silicon, which is strongly wavelength-dependent. At 450 nm, alpha ~ 3.5 x 10^4 /cm. At 550 nm, alpha ~ 7 x 10^3 /cm. At 650 nm, alpha ~ 3 x 10^3 /cm — more than 10x smaller than blue. The absorption depth (1/alpha) gives the thickness at which ~63% of light is absorbed.
:::

**DTI (Deep Trench Isolation)**

Between adjacent pixels, deep narrow trenches are etched into the silicon and filled with an insulating material (often silicon dioxide). These trenches extend from the front surface deep into the silicon — sometimes through the entire wafer thickness (2-6 micrometers).

DTI serves two purposes:

1. **Optical isolation**: The trench walls act as mirrors (due to the large refractive index contrast between silicon, n ~ 4.0, and oxide, n ~ 1.46), reflecting light that would otherwise leak into a neighboring pixel back into the correct pixel. This dramatically reduces optical crosstalk. The reflection at a silicon-oxide interface can exceed 50%, making DTI walls effective light guides.

2. **Electrical isolation**: The insulating fill prevents electrons generated in one pixel from diffusing into the neighbor's photodiode. Without DTI, electrons generated near a pixel boundary have roughly equal probability of being collected by either pixel — a 50% chance of crosstalk.

Modern sensors use "full DTI" (FDTI) where the trench goes all the way through the silicon, creating complete isolation between pixels. Partial DTI (extending only part way through the silicon) was used in earlier generations and still allows some leakage at the bottom. The transition from partial to full DTI has been one of the most important innovations for small-pixel image quality.

Typical DTI trenches are 50-150 nanometers wide, which is remarkably narrow considering they may be several micrometers deep. The aspect ratio (depth to width) can exceed 30:1, making DTI fabrication one of the most challenging steps in sensor manufacturing.

The combination of optical reflection and electrical isolation makes DTI arguably the most important structural innovation in modern pixel design. Before DTI became widespread, crosstalk between adjacent pixels was one of the primary factors limiting how small pixels could be made while still delivering acceptable image quality.

---

## How Light Travels Through the Pixel

Now that we know all the layers, let us trace the complete journey of a photon:

1. **Arrival**: A photon arrives from the camera lens, traveling downward toward the sensor surface. If the pixel is near the edge of the sensor, the photon arrives at an angle (the Chief Ray Angle). The photon carries a specific wavelength — say 550 nm (green).

2. **Microlens refraction**: The photon hits the curved surface of the microlens and bends inward. The lens shape focuses it toward the center of the pixel. If there is a CRA offset, the shifted lens corrects for the incoming angle. The photon's trajectory changes from its original direction to a more downward path aimed at the photodiode.

3. **Planarization transit**: The photon passes through the planarization layer. Not much happens here optically — it is a smooth, transparent transit region. The photon continues along the path set by the microlens refraction.

4. **Color filter**: The photon reaches the color filter.
   - If the photon's wavelength matches the filter color (e.g., a 550 nm photon hitting a green filter), it passes through with moderate absorption — perhaps 10-15% is lost.
   - If the wavelength does not match (e.g., a 450 nm blue photon hitting a red filter), the photon is absorbed by the filter and its energy is converted to a tiny amount of heat. The photon's journey ends here.

5. **Anti-reflection layers**: The surviving photon passes through the BARL. If the coating is well-designed for this wavelength, the photon transmits through with minimal reflection. A small percentage of photons (2-5%) are reflected back upward at this interface and lost.

6. **Silicon entry**: The photon enters the silicon. Now its wavelength determines what happens next.
   - A blue photon is absorbed almost immediately, within a fraction of a micrometer.
   - A green photon travels about 1 micrometer before being absorbed.
   - A red photon may travel 3 or more micrometers before being absorbed.
   - A NIR photon might pass all the way through the silicon without being absorbed at all.

7. **Electron generation**: When the photon is absorbed, its energy is transferred to an electron in the silicon crystal, freeing it from its atomic bond. This creates an electron-hole pair. If this happens within the photodiode's collection region, the electric field sweeps the electron into the charge storage node, where it joins other collected electrons. The electron is captured and becomes part of the signal.

8. **Signal readout**: After the exposure period (which may be a few milliseconds to a few seconds), the accumulated charge is read out by the pixel's transistor circuits, converted to a voltage, digitized by an analog-to-digital converter, and sent to the image processor. The number of collected electrons is proportional to the light intensity at that pixel.

This entire journey — from photon arriving at the microlens to electron being collected in the photodiode — happens at the speed of light through the stack. The optical transit takes roughly 30 femtoseconds (30 x 10^-15 seconds). From the photon's perspective, it is essentially instantaneous. But the wave-optics effects that occur during that transit — interference, diffraction, reflection at each interface — collectively determine whether the photon is detected or lost. This is why accurate electromagnetic simulation matters.

---

## What Can Go Wrong

A pixel stack may look simple on paper, but many things can degrade performance. Understanding these failure modes is central to pixel optimization — and to understanding what COMPASS simulations reveal.

### Optical Crosstalk

Crosstalk occurs when light intended for one pixel ends up being detected by a neighboring pixel. There are several mechanisms:

- **Optical crosstalk at the color filter level**: Light passes through one filter and scatters sideways before entering silicon. The metal grid helps but cannot eliminate this entirely, especially for light arriving at steep angles. Diffraction around the grid edges can redirect light laterally.
- **Optical crosstalk in silicon**: A photon enters the correct pixel but is absorbed deep in the silicon, where it generates an electron that diffuses sideways into the neighbor's photodiode. This is especially problematic for red and NIR photons, which penetrate deep before being absorbed. DTI is the primary defense here.
- **Diffraction-induced crosstalk**: When pixel sizes approach the wavelength of light, diffraction effects become significant. Light bends around the edges of structures and can spread into neighboring pixels regardless of physical barriers. This is a fundamental wave-optics effect that cannot be eliminated by engineering — only managed.

Crosstalk reduces color accuracy and image sharpness. A red pixel that detects some green light, or a green pixel that detects some blue, produces incorrect color readings that the image processing pipeline must try to correct. Heavy correction amplifies noise, so minimizing crosstalk at the optical level is always preferable.

Typical crosstalk values for modern sensors range from 1-3% for nearest neighbors in well-optimized designs, to 5-10% or more in challenging conditions (small pixels, large CRA, long wavelengths).

### Reflection Losses

Every interface between layers with different refractive indices causes some reflection. In a typical pixel stack, there are five or more such interfaces. Without anti-reflection measures, cumulative reflection losses could waste 20-40% of incoming light.

The BARL addresses the largest single reflection (at the silicon interface), but reflections at the microlens surface, the filter interfaces, and other boundaries all contribute. Each interface might reflect only 2-4%, but five interfaces compound to a significant total loss.

Optimizing the stack means minimizing reflection at every interface, not just the biggest one. This involves matching refractive indices between adjacent layers where possible, and using anti-reflection coatings where large index mismatches are unavoidable.

### Diffraction Effects

When pixel pitch shrinks below about 1.5 micrometers, diffraction becomes a dominant optical effect. The pixel aperture is comparable to the wavelength of visible light (0.4-0.7 micrometers), and light no longer travels in neat straight lines through the stack.

Instead, the electromagnetic wave spreads out as it passes through small openings (like the gaps in the metal grid) or around structures (like DTI walls). This spreading can push light energy outside the intended pixel boundary, contributing to crosstalk and reducing the effective collection area.

Diffraction affects blue light more than red (shorter wavelengths diffract at wider angles for a given aperture size). This is somewhat counterintuitive — smaller features affect longer wavelengths less. For a 0.7-micrometer pixel, blue light (450 nm) is seriously affected by diffraction, while red light (650 nm) is somewhat less affected but still not immune.

Simulating diffraction accurately requires wave-optics solvers like RCWA or FDTD — simple ray tracing is not sufficient at these scales. This is one of the core motivations for a tool like COMPASS. Ray-based models fail to capture the interference and diffraction effects that dominate sub-2-micrometer pixel behavior.

### Absorption Inefficiency

Not all wavelengths are absorbed efficiently in silicon. Near-infrared photons (beyond 800 nm) require very thick silicon layers for complete absorption. Since making silicon thicker increases cost, processing complexity, and crosstalk risk, there is always a practical limit.

Photons that pass through the silicon unabsorbed are simply lost — they exit the back side of the sensor and contribute nothing to the signal. For visible-light photography, this is mainly an issue for deep red wavelengths. For NIR applications (face recognition, distance sensing, night vision), absorption inefficiency is the dominant performance limiter.

Some advanced designs use light-trapping structures — textured surfaces or embedded reflectors that bounce light back through the silicon for a second pass, effectively doubling the absorption path length. These techniques are particularly important for NIR-optimized sensors.

### Stray Light and Flare

In addition to the mechanisms above, stray light within the pixel stack can degrade performance. Light that reflects off the silicon surface and travels back upward can re-reflect off the bottom of the color filter or metal grid and return to the silicon — potentially in a different pixel. These multiple-bounce paths are difficult to predict with simple models and require full wave-optics simulation to capture accurately.

Stray light can create subtle artifacts: slightly elevated background signal, reduced contrast, and color tinting. While these effects may be small on a per-pixel basis, they accumulate across the image and can be visible in high-dynamic-range scenes.

### Polarization Effects

At small pixel sizes, the pixel stack can respond differently to different polarizations of light (the orientation of the electromagnetic wave oscillation). Structures like the metal grid and DTI trenches can preferentially transmit or reflect one polarization over another, leading to polarization-dependent QE. While this is often a small effect, it becomes more pronounced at large angles of incidence and can contribute to spatial non-uniformity across the sensor.

---

## BSI vs. FSI: Why Architecture Matters

We have been discussing BSI architecture throughout this page, but it is worth briefly understanding why it replaced FSI.

In an **FSI** (Front-Side Illuminated) sensor, the transistors, metal wiring, and inter-layer dielectrics are all on the same side as the incoming light. Photons must navigate through several metal layers (which block and scatter light) and dielectric stacks before reaching the silicon. The metal wiring occupies a significant fraction of the pixel area, creating optical dead zones.

In a **BSI** sensor, the silicon wafer is flipped after fabrication. Light enters from the back side (which has no wiring), while the transistors and metal are on the opposite side, out of the light path. The only structures light encounters before reaching silicon are the optical layers we described: microlens, planarization, color filter, and BARL.

The advantage is dramatic. BSI pixels have:
- 30-50% higher QE than equivalent FSI pixels
- More uniform light collection across the pixel area
- Better performance at large chief ray angles
- More freedom in metal routing (since wiring no longer competes with light collection)

Today, virtually all smartphone sensors and most high-performance imaging sensors use BSI technology. The remaining FSI applications are primarily in specialized areas where BSI's wafer-thinning process is incompatible with the required sensor characteristics — such as certain scientific imaging or radiation-hardened sensors.

Understanding the BSI advantage is important context for COMPASS simulations: when you define a pixel stack in COMPASS, you are describing the clean optical path that BSI enables. The simulation does not need to model metal routing or interconnect layers in the light path, because BSI has moved them out of the way.

The BSI architecture also simplifies the simulation geometry. Since the optical path contains only the deliberately designed layers (microlens, planarization, color filter, BARL, silicon, DTI), the structure has a clean periodicity that is well suited to RCWA solvers. In an FSI pixel, the irregular metal routing would break this periodicity and make the simulation significantly more complex.

---

## Connection to Simulation

Every layer described above corresponds directly to a section in a COMPASS YAML configuration file. When you define a simulation, you are specifying:

- **Microlens**: shape, height, radius, material, CRA offset
- **Planarization**: thickness, material
- **Color filter**: thickness, material (which encodes the spectral transmission), metal grid dimensions and material
- **BARL**: layer count, thickness of each layer, materials
- **Silicon**: thickness, photodiode depth, DTI dimensions and material

COMPASS takes this layer-by-layer description and constructs a complete optical model called a PixelStack. It then hands this model to one or more electromagnetic solvers (RCWA, FDTD) to calculate how light propagates through the entire stack.

The result tells you exactly what happens to light at every point in the pixel: how much reflects, how much absorbs in each layer, how much reaches the photodiode, and how much leaks into neighbors. This is the foundation for computing Quantum Efficiency, which we explore in the next page.

The power of simulation is that you can change any parameter — make the microlens taller, add another BARL layer, widen the DTI — and instantly see the effect on QE and crosstalk. Physical prototyping of a sensor takes months and costs millions of dollars. Simulation takes minutes and costs nothing beyond compute time.

Furthermore, COMPASS allows you to run the same pixel structure through multiple independent solvers (such as torcwa, grcwa, meent for RCWA, or flaport for FDTD) and compare results. This cross-validation gives confidence that the simulated numbers are physically meaningful, not just artifacts of a particular numerical method.

When working with COMPASS, you will often find yourself adjusting parameters iteratively. For example, you might start by setting up a baseline pixel, then sweep the microlens height to find the optimum, then optimize the BARL layer thicknesses for that microlens configuration, and so on. Each layer interacts with the others, so optimizing one layer may shift the optimal parameters for another. COMPASS's sweep and comparison runners are designed to make this iterative workflow efficient.

---

## Summary

A modern BSI pixel is a precision-engineered optical stack:

| Layer | Typical Thickness | Primary Role |
|---|---|---|
| Air | -- | Medium where light arrives |
| Microlens | 0.3 - 0.8 um | Focus light onto the active area |
| Planarization | 0.1 - 0.5 um | Provide smooth surface for microlens |
| Color Filter | 0.5 - 1.0 um | Select wavelength band (R, G, or B) |
| Metal Grid | 0.1 - 0.3 um wide | Block light leakage between filters |
| BARL | 0.05 - 0.2 um total | Reduce reflection at silicon interface |
| Silicon | 2 - 6 um | Absorb photons and generate electrons |
| DTI | Full depth, 50-150 nm wide | Isolate pixels optically and electrically |

Each layer interacts with light through refraction, absorption, reflection, and diffraction. Understanding these interactions — and how they change with wavelength, angle, and pixel geometry — is what pixel simulation is all about.

Note that the thickness values in this table are representative ranges. Actual dimensions vary by manufacturer, pixel size, and target application. COMPASS lets you specify exact dimensions for your particular design.

The total pixel stack is an astonishingly compact optical system. In a height of just 5-10 micrometers, it performs focusing, color filtering, anti-reflection, photon absorption, and carrier collection — functions that in traditional optics would require centimeters of space. Getting each layer right, and making all layers work together harmoniously, is the central challenge of pixel engineering.

As a final note, remember that a real sensor contains millions of these pixels packed side by side in a periodic array. The electromagnetic behavior of each pixel is influenced by its neighbors — the periodic boundary conditions that RCWA solvers exploit are not just a mathematical convenience but a reflection of physical reality. Light diffracted by one pixel can interfere with light in adjacent pixels, and the collective behavior of the array differs from what you would calculate for a single isolated pixel. This is why COMPASS simulates a unit cell with periodic boundaries rather than a single pixel in isolation.

Understanding the anatomy of a single pixel — what each layer does, why it is there, and how it interacts with light — gives you the foundation to interpret simulation results, diagnose performance issues, and design better pixels. Every optimization begins with understanding the structure.

**Next**: [Understanding Quantum Efficiency](./qe-intuitive.md) — Where we quantify how well this pixel stack actually performs.
