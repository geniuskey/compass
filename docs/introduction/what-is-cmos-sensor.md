# What is a CMOS Image Sensor?

Every photo you take with your smartphone, every frame of video from a security camera, every image captured by a self-driving car -- they all start at the same place: a tiny chip called an **image sensor**. This chip sits behind the camera lens, silently converting light into electrical signals millions of times per second. Understanding how this chip works is the first step toward understanding what COMPASS simulates and why it matters.

Let's start from the big picture and zoom all the way down to the individual pixel.

---

## From Camera to Pixel

When you press the shutter button on a camera (or simply open your phone's camera app), here is what happens:

1. **Light from the scene** passes through the camera lens.
2. The lens focuses that light onto a flat chip -- the **image sensor**.
3. The image sensor is covered with **millions of tiny light detectors** arranged in a grid, like a mosaic.
4. Each tiny detector is called a **pixel**. It measures how much light hits it.
5. The electrical signals from all pixels are read out, processed, and assembled into the image you see on screen.

Think of the image sensor as a sheet of graph paper, where each tiny square on the paper independently measures brightness. A 12-megapixel sensor has 12 million of these squares, each one capturing its own little piece of the scene.

Here is a sense of scale: a modern smartphone image sensor is typically about 6 mm by 4 mm -- smaller than a fingernail. Yet it packs 50 million or more pixels onto that tiny area. A full-frame DSLR sensor is much larger (36 mm by 24 mm) but follows the same principle: more pixels means more detail, but each pixel must still capture enough light to produce a clean signal.

The key insight: **image quality begins at the pixel**. Everything -- color accuracy, low-light performance, dynamic range, sharpness -- depends on how well each individual pixel captures light. That is why pixel design matters so much, and that is exactly what COMPASS helps you study.

---

## What is a CMOS Image Sensor?

There are two main types of image sensors that have been used in digital cameras:

### CCD (Charge-Coupled Device)

CCDs were the original digital image sensor technology, dominant from the 1970s through the early 2000s. They produce excellent image quality but require specialized manufacturing processes and consume relatively high power. Think of a CCD like a bucket brigade -- charge is passed from pixel to pixel in a chain until it reaches a single output amplifier.

### CMOS (Complementary Metal-Oxide-Semiconductor)

CMOS sensors use the same manufacturing process as the chips in your computer's processor. Each pixel has its own amplifier and can be read individually, like giving every person in a stadium their own microphone instead of passing one microphone around.

### Why CMOS Won

By the mid-2000s, CMOS sensors overtook CCDs for almost all applications. Here is why:

- **Integration**: CMOS sensors can include processing circuits right on the same chip as the pixels. Signal processing, analog-to-digital conversion, and even some image processing can happen on-sensor.
- **Cost**: Since CMOS uses standard semiconductor fabrication, it benefits from the same economies of scale as computer chips. No special manufacturing line needed.
- **Speed**: Each pixel has its own amplifier, so pixels can be read in parallel rather than sequentially. This enables high frame rates.
- **Power**: CMOS sensors consume significantly less power, which is critical for smartphones and battery-powered devices.

Today, virtually every consumer camera -- from smartphones to DSLRs to medical imaging devices -- uses a CMOS image sensor. The market shift was decisive: by 2020, CMOS sensors accounted for over 99% of all image sensors shipped worldwide.

When we say "image sensor" in this documentation, we mean CMOS.

---

## The Pixel: The Fundamental Unit

A single pixel on a CMOS image sensor is a remarkable piece of engineering packed into an incredibly tiny space. Modern smartphone pixels are as small as 0.56 micrometers (um) across -- that is about 100 times thinner than a human hair.

Despite its tiny size, each pixel contains several functional components stacked on top of each other:

- **Microlens**: A tiny dome-shaped lens on top of the pixel that focuses incoming light down onto the active area. Without it, a lot of light would hit the gaps between pixels and be wasted. The microlens is typically made of a transparent polymer and shaped like a tiny dome or half-sphere. Its curvature and height are critical design parameters.
- **Color filter**: A thin layer that only allows certain wavelengths (colors) of light through. Most sensors use a **Bayer pattern** -- a repeating 2x2 grid of one red, one blue, and two green filters. (Two green because human vision is most sensitive to green.) The color filter array is what gives a single-chip sensor the ability to capture color images.
- **Oxide / passivation layers**: Thin transparent layers (typically silicon dioxide, SiO2) that provide electrical insulation and optical spacing. These seem passive, but their exact thickness affects how light waves interfere -- a seemingly small change can significantly impact performance.
- **Metal wiring layers**: Electrical connections that carry signals away from the pixel. These are opaque, so the pixel structure must be designed so they don't block light. In modern BSI sensors, the metal is placed behind the silicon, but in some designs, metal trenches between pixels are used to prevent optical crosstalk.
- **Photodiode**: The silicon region where the magic happens. When a photon of light enters the silicon, it can knock an electron free (a process called photoelectric absorption). The pixel collects these freed electrons and counts them. More photons = more electrons = brighter pixel.

Think of a pixel like a tiny bucket catching rain -- but instead of rain, it catches photons. The microlens is a funnel on top to direct more rain into the bucket. The color filter is like a screen that only lets certain sizes of raindrops through. And the photodiode is the bucket itself, collecting and counting what arrives.

### The Bayer Pattern: How Pixels See Color

A single pixel can only measure the total amount of light it receives -- it cannot distinguish colors on its own. To capture color, most image sensors use the **Bayer color filter array**: a mosaic of red, green, and blue filters laid over the pixel grid in a repeating 2x2 pattern:

```
 G  R  G  R  G  R
 B  G  B  G  B  G
 G  R  G  R  G  R
 B  G  B  G  B  G
```

Each pixel sees only one color. The full-color image is reconstructed by a process called **demosaicing**, which uses the color information from neighboring pixels to estimate the missing colors at each location.

Notice that there are twice as many green pixels as red or blue. This matches the human visual system, which is most sensitive to green light and uses it heavily for perceiving detail and luminance.

In COMPASS, when you simulate a 2x2 unit cell, you are simulating exactly one Bayer pattern repeat: one red, one blue, and two green pixels together.

<PhotonJourneyAnimation />

---

## BSI vs FSI: Which Way is Up?

One of the most important developments in modern image sensor technology is **backside illumination (BSI)**. To understand why, let's first look at the older approach.

### FSI (Frontside Illumination)

In a traditional FSI sensor, light enters from the **same side** as the metal wiring layers. Imagine trying to catch rain in a bucket, but there is a jungle gym of metal pipes above the bucket. Some of the rain hits the pipes and never reaches the bucket. Similarly, in FSI sensors, the metal wiring partially blocks and scatters light before it can reach the photodiode.

This was not a big problem when pixels were large (5 um or bigger) -- there was enough open area between the wires. But as pixels shrank below 2 um, the metal wiring started blocking a significant fraction of the incoming light.

### BSI (Backside Illumination)

BSI flips the sensor upside down. The silicon wafer is thinned from the back, and light enters from the **backside** -- the side **without** metal wiring. Now photons travel directly into the silicon photodiode with nothing in the way.

Here is a simplified view:

```
FSI (old approach):            BSI (modern approach):

  Light ↓                        Light ↓
  ──────────────                 ──────────────
  Color filter                   Color filter
  ──────────────                 ──────────────
  Metal wires  ← blocks light!  Silicon (photodiode)
  ──────────────                 ──────────────
  Silicon (photodiode)           Metal wires  ← behind silicon
  ──────────────                 ──────────────
```

### Why BSI is Now Standard

BSI offers several major advantages for small pixels:

- **Higher sensitivity**: No metal wiring blocking the light path.
- **Better angular response**: Light arriving at an angle is not clipped by metal edges.
- **Improved uniformity**: More consistent performance across the sensor area.

Since around 2010, BSI has become the standard for smartphone image sensors, and it is increasingly used in larger sensors as well. The manufacturing process involves an extra step -- thinning the silicon wafer from the back to about 2-3 micrometers -- but the optical benefits are well worth it for small pixels.

One additional feature in modern BSI sensors is **deep trench isolation (DTI)**: narrow trenches filled with oxide or metal that extend vertically through the silicon between adjacent pixels. These trenches act as optical walls, preventing light from leaking sideways from one pixel to its neighbor. DTI is especially important for small pixels where crosstalk would otherwise be severe.

**COMPASS primarily simulates BSI pixel structures**, though FSI configurations are also supported.

---

## Why Simulate?

You might wonder: if engineers have been making image sensors for decades, why do we need computer simulations? The answer lies in the relentless shrinking of pixel size.

### The Shrinking Pixel

The drive for higher resolution and smaller camera modules has pushed pixel sizes down dramatically:

| Year (approx.) | Pixel Pitch | Notable Devices |
|---|---|---|
| 2005 | 2.2 um | Early smartphone cameras |
| 2010 | 1.4 um | iPhone 4 era |
| 2015 | 1.0 um | Mainstream smartphones |
| 2020 | 0.7 um | High-resolution sensors |
| 2023+ | 0.56 um | 200MP smartphone sensors |

### When Pixels Shrink, Wave Optics Matters

Here is the critical point: **visible light has wavelengths between about 0.4 um and 0.7 um**. When pixels were 2 um across, a pixel was several wavelengths wide, and simple ray tracing (geometric optics) could predict light behavior reasonably well.

But when the pixel pitch approaches or drops below the wavelength of light, something fundamental changes. Light no longer behaves like simple rays bouncing around. Instead, **wave effects** become dominant:

- **Diffraction**: Light bends around edges and spreads out. A microlens smaller than a wavelength cannot focus light the way geometric optics predicts.
- **Interference**: Light waves reflected from different layers can add together (constructive) or cancel out (destructive). The thickness of each layer matters at the nanometer scale.
- **Coupling**: Light intended for one pixel can leak into neighboring pixels (crosstalk) through wave effects that ray tracing cannot capture.

This means that to design a 0.7 um or 0.56 um pixel, you cannot rely on intuition or simple optical models. You need **full electromagnetic (EM) simulation** that solves Maxwell's equations.

### Why Not Just Build and Test?

Physical prototyping of image sensors is:

- **Expensive**: Fabricating a new sensor design costs millions of dollars.
- **Slow**: A single fabrication cycle takes months.
- **Limited**: You can only test the exact designs you fabricate. Want to try a slightly different microlens shape? That is another fabrication run.

Simulation, by contrast, is:

- **Fast**: A wavelength sweep takes minutes to hours on a workstation, not months.
- **Cheap**: Computational cost is a tiny fraction of fabrication cost.
- **Flexible**: Change any parameter -- pixel pitch, layer thickness, microlens curvature, color filter material -- and re-run instantly.
- **Insightful**: Simulation gives you the full electromagnetic field distribution inside the pixel. You can see exactly where light goes, where it is lost, and why. Physical measurements can only tell you the final output, not the internal behavior.

### The Role of Electromagnetic Simulation

The type of simulation COMPASS performs is called **electromagnetic (EM) simulation** or **full-wave simulation**. This means it solves Maxwell's equations -- the fundamental equations that describe how electric and magnetic fields behave -- for the specific pixel geometry and materials you define.

There are two main families of EM simulation methods used in COMPASS:

- **RCWA (Rigorous Coupled-Wave Analysis)**: Works in the frequency domain (one wavelength at a time). It is very efficient for periodic structures like pixel arrays, where the same pattern repeats across the sensor. RCWA decomposes the fields into Fourier harmonics and propagates them through each layer.
- **FDTD (Finite-Difference Time-Domain)**: Works in the time domain by stepping through time and tracking how electromagnetic fields evolve. A single FDTD run can capture the response across many wavelengths at once. It is more flexible with geometry but typically more computationally expensive.

You do not need to understand the mathematical details of these methods to use COMPASS -- the platform handles the solver setup and execution. But knowing that these are rigorous, first-principles methods (not approximations or ray tracing) helps you appreciate the accuracy and reliability of the results.

---

## What COMPASS Does

COMPASS is a simulation platform designed specifically for CMOS image sensor pixels. Here is the workflow:

### 1. Define Your Pixel

You describe the pixel structure in a single **YAML configuration file**. This includes the pixel pitch, the microlens shape, color filter materials, oxide layers, metal trenches, and silicon thickness. One file captures the entire pixel design.

```yaml
pixel:
  pitch: 1.0        # Pixel size in micrometers
  unit_cell: [2, 2]  # Simulate a 2x2 Bayer pattern

  layers:
    - name: microlens
      type: microlens
      height: 0.6
    - name: color_filter
      type: color_filter
      thickness: 0.6
    - name: oxide
      type: planar
      material: SiO2
      thickness: 0.2
    - name: silicon
      type: photodiode
      material: Si
      thickness: 3.0
```

### 2. Choose a Solver

COMPASS supports multiple electromagnetic solvers through a unified interface:

- **RCWA solvers** (Rigorous Coupled-Wave Analysis): torcwa, grcwa, meent -- fast frequency-domain methods ideal for periodic pixel arrays.
- **FDTD solver** (Finite-Difference Time-Domain): flaport -- a time-domain method that captures broadband behavior in a single run.

You can run the same pixel design through different solvers and compare results for confidence. This **cross-validation** capability is one of COMPASS's unique strengths -- if two independent solvers agree on the QE, you can be much more confident in the result than if you relied on just one.

### 3. Run the Simulation

A single Python command (or script) launches the simulation. COMPASS handles the translation from your pixel description to the solver's internal representation, runs the computation, and collects results.

### 4. Analyze Results

COMPASS computes and visualizes:

- **Quantum Efficiency (QE)**: What fraction of incoming photons are actually captured by each pixel? This is the single most important metric for pixel performance.
- **Crosstalk**: How much light intended for one pixel ends up in a neighboring pixel? Lower is better.
- **Field distributions**: Full 2D or 3D maps of the electromagnetic field inside the pixel, showing exactly where light goes.
- **Energy balance**: Verification that all light is accounted for (reflected + transmitted + absorbed = 100%).

---

## Key Metrics at a Glance

Before diving deeper, here are the three metrics you will encounter most often in COMPASS:

### Quantum Efficiency (QE)

QE answers the question: "Out of all the photons that hit this pixel, what fraction actually gets detected?" A QE of 80% means 80 out of 100 photons successfully generate an electrical signal. Higher is better. QE depends on wavelength -- a pixel might have 85% QE for green light but only 40% QE for blue light.

### Crosstalk

Crosstalk measures how much light leaks from one pixel to its neighbors. If you shine a focused beam on the green pixel but 5% of the energy ends up in the adjacent red pixel, that is 5% optical crosstalk. It causes color errors and reduces image sharpness. Smaller pixels have higher crosstalk risk, which is one reason simulation is so important at small pixel pitches.

### Angular Response

Real camera light does not always arrive straight on. Light at the edges of the sensor arrives at significant angles (up to 30 degrees or more). Angular response measures how well a pixel performs as the angle of incoming light increases. A good pixel design maintains high QE and low crosstalk even at large angles.

Why do angles matter? In a camera, the main lens projects the image onto the sensor plane. Pixels near the center of the sensor receive light that arrives nearly straight on (normal incidence). But pixels near the edges and corners receive light at an angle -- this is called the **chief ray angle (CRA)**. In smartphone cameras, where the lens module must be very thin, the CRA can reach 25-35 degrees at the sensor corners. If the pixel is not designed to handle these angles well, the corners of your photos will appear darker and have worse color than the center.

COMPASS allows you to simulate pixel performance at any angle of incidence, and even sweep across a range of angles to produce a complete angular response curve.

---

## A Day in the Life of a Photon

To tie everything together, let's follow a single photon on its journey through a BSI pixel:

1. **Arrival**: A green photon (wavelength ~550 nm) arrives at the sensor surface, having traveled from the scene through the camera lens.

2. **Microlens**: The photon hits the curved microlens surface. The lens bends its path, directing it toward the center of the pixel rather than letting it wander off to the side.

3. **Color filter**: The photon passes through the green color filter. If it had been a red photon arriving at a green pixel, it would have been absorbed by the filter and never made it further -- that is how color selectivity works.

4. **Oxide layers**: The photon passes through one or more thin oxide layers. At each interface, there is a small chance of reflection. The thicknesses of these layers are tuned so that reflected waves interfere destructively, minimizing loss.

5. **Silicon entry**: The photon enters the silicon. The refractive index jumps from about 1.5 (oxide) to about 4.0 (silicon). An anti-reflection coating at this interface reduces what would otherwise be a 18% reflection loss.

6. **Absorption**: Within the first 1-2 micrometers of silicon, the green photon is absorbed. Its energy kicks an electron free from the crystal lattice, creating an electron-hole pair.

7. **Collection**: The freed electron drifts toward the pixel's collection well, guided by electric fields within the silicon. If the pixel has deep trench isolation, the electron is prevented from wandering into the neighboring pixel.

8. **Signal**: The collected electron contributes to the pixel's stored charge. After the exposure is complete, this charge is read out as a voltage, digitized, and becomes one tiny piece of the final image.

This entire journey takes less than a billionth of a second. COMPASS simulates steps 1 through 6 in detail -- the electromagnetic interaction of light with the pixel structure -- allowing engineers to optimize each layer for maximum photon capture.

---

## Where to Go Next

Now that you have a high-level understanding of what CMOS image sensors are and why simulation matters, here are some suggested next steps:

- **[Optics Primer for Beginners](./optics-primer.md)** -- Learn the fundamental optical concepts (wavelength, refractive index, interference) that underpin all COMPASS simulations. No prior physics background required.
- **[Pixel Anatomy](/theory/image-sensor-optics)** -- A deeper look at each layer of a modern BSI pixel and the physics at each interface.
- **[First Simulation](/guide/first-simulation)** -- Jump straight in and run your first COMPASS simulation with a step-by-step guide.
- **[Installation Guide](/guide/installation)** -- Set up COMPASS and its solver backends on your machine.

---

*This page is part of the COMPASS Introduction series, designed for readers with no prior background in image sensor optics. All technical terms are introduced as they appear.*
