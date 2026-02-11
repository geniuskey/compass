# Optics Primer for Beginners

Light is the raw material of every image sensor. To understand how COMPASS simulates a pixel, you need to understand how light behaves -- how it travels, bends, reflects, and gets absorbed. This page introduces the essential optical concepts from scratch. No physics background is assumed, and no equations appear in the main text. (If you are curious about the math, expand the optional "Math Detail" sections.)

By the end of this page, you will understand the physical phenomena that COMPASS solvers compute for you automatically.

---

## What is Light?

Light is one of the most studied yet most mysterious things in physics. It has a dual nature:

- **As a wave**: Light is a ripple in the electromagnetic field, similar to how a water wave is a ripple on a lake's surface. It has a wavelength (the distance between wave crests), a frequency (how fast the crests arrive), and it can interfere with other waves.
- **As a particle**: Light also comes in discrete packets of energy called **photons**. Each photon carries a specific amount of energy related to its wavelength. When a photon is absorbed by silicon in an image sensor pixel, it knocks one electron free -- that is the fundamental event in digital photography.

For **simulation purposes**, we primarily use the wave picture. The electromagnetic solvers in COMPASS (RCWA and FDTD) solve Maxwell's equations, which describe light as an electromagnetic wave. The wave picture naturally captures diffraction, interference, and all the phenomena that matter at the sub-micrometer scale of modern pixels.

Think of it this way: when light travels across open space, you can think of it as rays (like arrows pointing in the direction of travel). But when light encounters structures that are similar in size to its wavelength -- like the layers and features inside a tiny pixel -- the ray picture breaks down and you must treat light as a wave. COMPASS handles this wave physics for you.

---

## Wavelength and Color

The most important property of a light wave is its **wavelength** -- the distance from one wave crest to the next. Wavelength determines everything: what color you see, how deep light penetrates into silicon, and how it interacts with the nanometer-scale structures in a pixel.

### The Visible Spectrum

Human eyes can see light with wavelengths roughly between **380 nm and 780 nm** (nanometers). Within this range:

| Color | Approximate Wavelength |
|---|---|
| Violet | 380 -- 450 nm |
| Blue | 450 -- 495 nm |
| Green | 495 -- 570 nm |
| Yellow | 570 -- 590 nm |
| Orange | 590 -- 620 nm |
| Red | 620 -- 780 nm |

For quick mental reference: **Blue is around 450 nm, Green is around 550 nm, Red is around 650 nm.** These three are especially important because they match the color filters in a Bayer-pattern image sensor.

### Beyond Visible: Near-Infrared

Silicon image sensors can actually detect light beyond what human eyes can see, out to about **1100 nm** (near-infrared). This is useful for applications like night-vision cameras, depth sensing (the face-unlock feature on smartphones uses near-IR), and industrial inspection. COMPASS can simulate wavelengths across this entire range.

### Units: Nanometers and Micrometers

You will see two units used for wavelength and for the physical dimensions of pixel structures:

- **Nanometers (nm)**: 1 nm = one billionth of a meter. Used for wavelengths and very thin layers.
- **Micrometers (um)**: 1 um = one millionth of a meter = 1000 nm. Used for pixel pitch and layer thicknesses.

COMPASS uses **micrometers internally** for all lengths. So a wavelength of 550 nm is entered as 0.55 um in configuration files.

Here is a sense of scale: a human hair is about 70 um wide. A modern smartphone pixel is about 0.7 um wide -- 100 times thinner than that hair. And the wavelength of green light (0.55 um) is not much smaller than the pixel itself. This is precisely why wave optics simulation is necessary.

<WavelengthExplorer />

---

## Refractive Index (n): Why Light Bends

When light travels from one material to another -- say, from air into glass -- two things happen: it **slows down** and it **changes direction** (bends). The property that describes how much light slows down in a material is called the **refractive index**, represented by the letter **n**.

### The Intuition

In vacuum (empty space), light travels at its maximum speed -- roughly 300,000 kilometers per second. When light enters a material like glass or water, it interacts with the atoms in the material and effectively slows down. The refractive index tells you by how much:

- **Air**: n = 1.0 (light travels at essentially full speed)
- **Water**: n = 1.33 (light travels at about 75% of its vacuum speed)
- **Glass**: n = 1.5 (light travels at about 67% of its vacuum speed)
- **Silicon**: n = 4.0 at green wavelengths (light travels at only 25% of its vacuum speed!)

The higher the refractive index, the slower light travels in that material.

### Why It Matters

Refractive index is arguably the most important material property in pixel simulation:

- **Reflection at interfaces**: When light hits a boundary between two materials with different refractive indices, some light reflects. The bigger the difference in n, the stronger the reflection. The air-silicon interface (n = 1 to n = 4) reflects a lot of light -- that is why anti-reflection coatings are essential.
- **Refraction (bending)**: Light changes direction when it crosses an interface between materials with different n values. This is how microlenses focus light -- the curved surface of a high-n material bends light toward the pixel center.
- **Interference effects**: The refractive index determines how fast light travels through each layer, which in turn determines the phase of the wave. Phase differences between layers cause constructive or destructive interference, making layer thicknesses critically important.

::: details Math Detail
The refractive index n is defined as the ratio of the speed of light in vacuum to the speed in the material:

n = c / v

where c is the speed of light in vacuum and v is the speed in the material.

When light crosses an interface, Snell's law relates the angles:

n1 * sin(theta1) = n2 * sin(theta2)

The Fresnel equations give the fraction of reflected and transmitted light at an interface as a function of the refractive indices and the angle of incidence.
:::

### Refractive Index Changes with Wavelength

An important subtlety: the refractive index of most materials is not a fixed number. It **changes with wavelength**. This phenomenon is called **dispersion** -- it is the reason a prism splits white light into a rainbow.

For silicon, the refractive index is higher at shorter (blue) wavelengths and lower at longer (red) wavelengths. COMPASS accounts for this by looking up the correct n value at each simulated wavelength from its built-in material database.

---

## Absorption (k): How Materials Swallow Light

So far we have talked about light slowing down and bending. But some materials also **absorb** light -- they convert the light's energy into something else (typically heat, or in the case of a photodetector, electrical current).

Absorption is described by a second number called **k**, the **extinction coefficient** or the **imaginary part of the refractive index**. Together, n and k form the **complex refractive index**: n + ik.

### Why Absorption Matters for Image Sensors

Silicon absorbs visible light -- and that is a good thing! Absorption is the very mechanism that makes silicon photodetectors work. When a photon is absorbed in silicon, it transfers its energy to an electron, knocking it free. The pixel collects this free electron, and that is how light becomes an electrical signal.

But the absorption is not the same for all colors:

| Wavelength | Color | Absorption Depth in Silicon |
|---|---|---|
| 450 nm | Blue | ~0.2 um |
| 550 nm | Green | ~1.5 um |
| 650 nm | Red | ~3.0 um |
| 850 nm | Near-IR | ~15 um |

**Absorption depth** is the distance light must travel in silicon before most of it (about 63%) is absorbed. Blue light is absorbed very quickly -- within the first 0.2 um of silicon. Red light penetrates much deeper, needing about 3 um. Near-infrared light can travel even further.

This has huge implications for pixel design:

- **Silicon thickness matters**: If the silicon layer is only 2 um thick, it will capture almost all blue and green light but miss a significant fraction of red light.
- **BSI orientation matters**: In a BSI sensor, light enters from the top of the silicon. Blue photons are absorbed near the top surface (close to the color filter), while red photons are absorbed deeper. This affects charge collection efficiency.
- **Crosstalk depends on wavelength**: Red light that penetrates deep into the silicon can drift laterally into a neighboring pixel before being collected, causing crosstalk.

::: details Math Detail
The absorption coefficient alpha is related to the extinction coefficient k by:

alpha = 4 * pi * k / lambda

where lambda is the free-space wavelength. The intensity of light decays exponentially as it penetrates:

I(z) = I_0 * exp(-alpha * z)

The absorption depth d = 1/alpha is the distance at which intensity drops to 1/e (about 37%) of its initial value.

In COMPASS, the complex permittivity is:

epsilon = (n + ik)^2

This is the form used by all EM solvers internally.
:::

---

## Reflection and Refraction

When light hits a boundary between two materials, it splits into two parts: a **reflected** portion that bounces back, and a **transmitted** portion that continues forward (possibly in a new direction). Understanding this splitting is essential to pixel design.

### Refraction: Light Bends at Interfaces

When light passes from a low-n material (like air) into a high-n material (like glass or silicon), it bends toward the perpendicular to the surface. When it goes from high-n to low-n, it bends away from the perpendicular. This bending is called **refraction**.

Refraction is how lenses work -- including the microlenses on top of each pixel. A curved surface of a high-n material bends light rays so they converge toward a focal point. In a pixel, the microlens bends incoming light toward the center of the photodiode, concentrating photons where they can be detected.

::: details Math Detail
Snell's law of refraction:

n1 * sin(theta1) = n2 * sin(theta2)

where n1 and n2 are the refractive indices of the two media, and theta1 and theta2 are the angles of the light ray measured from the surface normal.
:::

### Reflection: Light Bounces Back

At every interface, some fraction of light is reflected. How much depends on the difference in refractive index between the two materials and the angle of incidence.

Consider a few examples:

- **Air to glass** (n=1 to n=1.5): about 4% of light reflects at normal incidence. This is why you see faint reflections in windows.
- **Air to silicon** (n=1 to n=4): about 36% of light reflects! That is a huge loss.
- **Glass to silicon** (n=1.5 to n=4): about 18% reflects. Still significant, but better than air-silicon.

This is why image sensor pixels use **anti-reflection coatings** -- thin layers of intermediate-n materials placed between the low-n oxide and the high-n silicon. By carefully choosing the thickness and refractive index of these coatings, engineers can dramatically reduce reflection and get more light into the photodiode.

::: details Math Detail
At normal incidence (light hitting the surface straight on), the reflectance from the Fresnel equations simplifies to:

R = ((n1 - n2) / (n1 + n2))^2

For air (n1=1) to silicon (n2=4): R = (3/5)^2 = 0.36, or 36%.

For non-normal incidence, the full Fresnel equations must be used, and the reflectance depends on the polarization of the light (s-polarization vs p-polarization).
:::

<FresnelCalculator />

---

## Interference: When Waves Add Up (or Cancel Out)

Interference is perhaps the most important wave phenomenon in image sensor optics. It occurs when two or more light waves overlap and combine. The result depends on how the wave crests align:

- **Constructive interference**: When the crests of two waves line up, they add together to produce a stronger wave. The result is brighter light.
- **Destructive interference**: When the crest of one wave lines up with the trough of another, they cancel out. The result is dimmer light -- or no light at all.

### Thin Film Interference: Everyday Examples

You have seen interference before, even if you did not know the name:

- **Soap bubbles** show swirling rainbow colors because light reflecting off the front and back surfaces of the thin soap film interferes. The film thickness determines which colors interfere constructively (appear bright) and which interfere destructively (appear dark).
- **Oil on water** produces rainbow patterns for the same reason.
- **Anti-reflection coatings on eyeglasses** look slightly purple or green -- that is the residual color from the wavelengths that are not perfectly canceled by the coating.

### Why Interference Matters in Pixels

Inside a pixel, light passes through multiple thin layers (microlens, color filter, oxide layers, anti-reflection coatings) before reaching the silicon. At every interface, some light reflects. These reflected waves travel back and forth between layers, interfering with each other and with the incoming wave.

The result is that the amount of light reaching the photodiode **depends sensitively on the thickness of each layer**. Change an oxide layer by just 10 nm (a tenth of a percent of the pixel's total height), and the QE at certain wavelengths can shift noticeably.

This is why precise simulation matters. You cannot predict these interference effects by intuition alone -- they require solving the full wave equations, which is exactly what RCWA and FDTD solvers do.

### Constructive and Destructive Conditions

Whether interference is constructive or destructive depends on the **optical path length** -- the physical thickness of a layer multiplied by its refractive index. When the optical path length equals a whole number of wavelengths, you tend to get constructive interference (more light). When it equals a half-integer number of wavelengths, you get destructive interference (less light).

This is why anti-reflection coatings work: a coating with an optical thickness of exactly one quarter wavelength causes the reflection from its top surface to destructively interfere with the reflection from its bottom surface. The reflections cancel out, and more light is transmitted.

::: details Math Detail
For a thin film of thickness d and refractive index n, the optical path difference between light reflecting from the top and bottom surfaces is:

delta = 2 * n * d

Constructive interference occurs when delta = m * lambda (where m is an integer).
Destructive interference occurs when delta = (m + 1/2) * lambda.

For a quarter-wave anti-reflection coating:

n_coating * d = lambda / 4

The optimal refractive index for the coating is:

n_coating = sqrt(n1 * n2)

where n1 and n2 are the refractive indices of the materials on either side.
:::

---

## Diffraction: Light Bends Around Corners

In everyday life, light seems to travel in straight lines. But when light encounters an obstacle or aperture that is similar in size to its wavelength, it **spreads out** and bends around edges. This is called **diffraction**.

### Why Diffraction Matters in Small Pixels

For a pixel that is 0.7 um wide, illuminated by green light (0.55 um wavelength), the pixel aperture is only about 1.3 wavelengths across. At this scale, diffraction is significant:

- **Microlens focusing is limited**: A microlens this small cannot focus light to a sharp point. The diffraction limit means the focused spot is comparable in size to the pixel itself.
- **Light spreads into neighbors**: Diffracted light does not respect pixel boundaries. Even if a microlens is perfectly designed for its own pixel, some light inevitably spills into adjacent pixels.
- **Color filters have diffraction effects**: The finite size of color filter patches means their spectral response is not perfectly sharp.

Ray-tracing models (geometric optics) completely ignore diffraction. This is acceptable when structures are much larger than the wavelength, but for modern sub-micrometer pixels, ignoring diffraction leads to significantly wrong predictions. Full-wave solvers like those in COMPASS naturally include diffraction effects.

---

## Energy Conservation: Where Does the Light Go?

One of the most fundamental principles in optics is **energy conservation**. When light hits a structure, all of its energy must be accounted for. It can only go three places:

- **Reflected (R)**: Bounced back toward the source.
- **Transmitted (T)**: Passed through the structure and continued forward.
- **Absorbed (A)**: Converted to another form of energy (heat or electrical current) within the structure.

The energy conservation rule is simple:

**R + T + A = 1**

This means: the fraction of light reflected, plus the fraction transmitted, plus the fraction absorbed, must always add up to 100%.

In the context of an image sensor pixel:

- **R** is the light lost to reflection at the top surface and internal interfaces. We want this to be as small as possible.
- **A in silicon** is the useful part -- this is the light that generates signal. We want this to be as large as possible. This is essentially the quantum efficiency.
- **A in other layers** (metal, color filter absorption beyond the intended band) is parasitic loss. We want this to be small.
- **T** is light that passes all the way through the silicon without being absorbed. For very thin silicon or long wavelengths (near-IR), this can be significant.

COMPASS checks energy conservation as a diagnostic. If R + T + A differs from 1 by more than 1%, it flags a potential numerical problem with the simulation. This is one of the ways you can verify that your simulation results are trustworthy.

::: details Math Detail
For a plane wave at a given wavelength and angle of incidence, the energy conservation relation is:

R + T + A = 1

where R is the total reflectance (fraction of incident power reflected), T is the total transmittance (fraction of incident power transmitted), and A is the total absorptance (fraction of incident power absorbed).

In COMPASS, absorptance can be decomposed by layer:

A_total = A_silicon + A_metal + A_color_filter + A_other

The quantum efficiency QE is closely related to A_silicon but also accounts for charge collection efficiency within the silicon.
:::

---

## Polarization: The Direction of Vibration

Light is a **transverse wave**, meaning the electromagnetic field oscillates perpendicular to the direction the light is traveling. The direction of this oscillation is called the **polarization**.

For simulation purposes, there are two important polarization states:

- **s-polarization (TE)**: The electric field oscillates parallel to the surface of the pixel.
- **p-polarization (TM)**: The electric field oscillates in the plane containing the light's direction and the surface normal.

Why does this matter? The amount of light reflected at an interface depends on the polarization. At non-normal incidence (light hitting at an angle), s-polarized and p-polarized light reflect differently. For an image sensor under typical conditions, the incoming light is **unpolarized** (a random mix of both polarizations), so COMPASS averages over both polarization states by default.

---

## Putting It All Together: Connection to COMPASS

Every concept on this page maps directly to what COMPASS simulates:

| Optical Concept | How COMPASS Uses It |
|---|---|
| **Wavelength** | Simulation sweeps across wavelengths to compute QE spectrum |
| **Refractive index (n)** | Material database provides n(lambda) for all materials |
| **Absorption (k)** | Material database provides k(lambda); silicon absorption generates QE |
| **Reflection** | Computed at every interface; minimized by anti-reflection design |
| **Refraction** | Microlens focusing and light bending through layers |
| **Interference** | Automatically captured by RCWA and FDTD solvers |
| **Diffraction** | Naturally included in full-wave solvers |
| **Energy conservation** | R + T + A = 1 checked as simulation diagnostic |
| **Polarization** | User selects TE, TM, or unpolarized (averaged) |

The beauty of using a full electromagnetic solver is that you do not need to manually account for each of these effects. You define the pixel structure and materials, and the solver handles all the wave physics simultaneously. COMPASS makes it easy to set up the problem and interpret the results.

---

## Where to Go Next

With these optical foundations in place, you are ready to go deeper:

- **[What is a CMOS Image Sensor?](./what-is-cmos-sensor.md)** -- If you have not read it yet, start here for the big picture of image sensors and pixels.
- **[Light Basics (Theory)](/theory/light-basics)** -- A more detailed treatment of electromagnetic wave theory with full mathematical development.
- **[Thin Film Optics](/theory/thin-film-optics)** -- Deep dive into interference and anti-reflection coating design.
- **[Quantum Efficiency](/theory/quantum-efficiency)** -- The physics of photon-to-electron conversion and QE calculation.
- **[First Simulation](/guide/first-simulation)** -- Apply what you have learned by running an actual COMPASS simulation.
- **[Material Database Guide](/guide/material-database)** -- How COMPASS stores and looks up refractive index data for all materials.

---

*This page is part of the COMPASS Introduction series, designed for readers with no prior background in optics. All technical terms are introduced as they appear. Optional "Math Detail" sections provide equations for readers who want them.*
