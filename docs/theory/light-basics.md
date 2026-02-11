# Light Basics

::: tip 학습 경로 | Learning Path
처음이라면? [CMOS 이미지 센서란?](/introduction/what-is-cmos-sensor)부터 시작하세요.
New here? Start with [What is a CMOS Image Sensor?](/introduction/what-is-cmos-sensor)
:::

This page covers the fundamental properties of light that underpin every simulation in COMPASS.

## What is light?

Light is electromagnetic radiation -- oscillating electric and magnetic fields that propagate through space. In the context of image sensor simulation, we care about visible and near-infrared wavelengths (roughly 380 nm to 1100 nm).

Light has a dual nature:

- **Wave**: Light exhibits interference and diffraction, which are critical when pixel features are comparable to the wavelength.
- **Particle (photon)**: Each photon carries energy $E = h\nu = hc/\lambda$, where $h$ is Planck's constant, $c$ is the speed of light, and $\lambda$ is the wavelength.

## Key properties

### Wavelength and frequency

The wavelength $\lambda$ and frequency $\nu$ are related by:

$$\lambda = \frac{c}{n \cdot \nu}$$

where $n$ is the refractive index of the medium and $c = 3 \times 10^8$ m/s. COMPASS works in micrometers (um), so visible light spans $0.38$ to $0.78$ um.

### Refractive index

A material's optical properties are described by its **complex refractive index**:

$$\tilde{n} = n + ik$$

| Part | Symbol | Meaning |
|------|--------|---------|
| Real | $n$ | Ratio of phase velocity in vacuum to phase velocity in the material. Controls refraction and interference. |
| Imaginary | $k$ | Extinction coefficient. Controls absorption -- how quickly light intensity decays as it propagates. |

For example, silicon at 550 nm has $n \approx 4.08$ and $k \approx 0.028$, meaning it strongly refracts light and absorbs it over a few micrometers.

### Permittivity

The complex permittivity $\varepsilon$ is the square of the complex refractive index:

$$\varepsilon = \tilde{n}^2 = (n + ik)^2 = (n^2 - k^2) + 2ink$$

RCWA and FDTD solvers work with permittivity internally. COMPASS stores materials in $(n, k)$ form and converts to $\varepsilon$ when building the simulation geometry.

### Absorption

When light travels a distance $d$ through an absorbing medium, the intensity decays according to the Beer-Lambert law:

$$I(d) = I_0 \, e^{-\alpha d}$$

where $\alpha$ is the absorption coefficient, related to $k$ by:

$$\alpha = \frac{4\pi k}{\lambda}$$

This is why silicon absorbs short-wavelength (blue) light near the surface but long-wavelength (red/NIR) light penetrates several micrometers deeper. The different absorption depths are a core challenge in image sensor design.

<WavelengthSlider />

## Refraction: Snell's law

When light passes from one medium to another, it changes direction according to Snell's law: $n_1 \sin\theta_1 = n_2 \sin\theta_2$. If the incidence angle exceeds the critical angle (when going from a denser to a less dense medium), total internal reflection occurs.

<SnellCalculator />

## Polarization

Light is a transverse wave -- the electric field oscillates perpendicular to the direction of propagation. The orientation of this oscillation is the **polarization state**.

- **TE (s-polarization)**: Electric field perpendicular to the plane of incidence.
- **TM (p-polarization)**: Electric field in the plane of incidence.
- **Unpolarized**: Equal mixture of TE and TM. Natural sunlight and most ambient light sources are unpolarized.

COMPASS supports TE, TM, and unpolarized excitation. For unpolarized light, the simulation runs both TE and TM and averages the results:

$$\text{QE}_\text{unpol} = \frac{1}{2}(\text{QE}_\text{TE} + \text{QE}_\text{TM})$$

<PolarizationViewer />

## Relevance to COMPASS

Every COMPASS simulation begins by defining wavelength range, incidence angle, and polarization state through the source configuration. These parameters determine:

1. The permittivity of each material at each wavelength (via `MaterialDB`).
2. The interference conditions in thin-film stacks (anti-reflection coatings).
3. The diffraction behavior of sub-wavelength structures (color filter grids, DTI).
4. The absorption depth in silicon, which directly impacts QE.

::: tip
For most image sensor simulations, use `polarization: "unpolarized"` and a wavelength sweep from 0.38 to 0.78 um to capture the full visible spectrum.
:::
