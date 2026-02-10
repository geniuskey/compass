# Thin Film Optics

Thin film interference is one of the most important optical effects in image sensor design. Anti-reflection coatings, BARL layers, and even the planarization layer all rely on thin-film principles.

## Single thin film

Consider a thin film of thickness $d$ and refractive index $n_f$ on a substrate. Light reflected from the top and bottom surfaces of the film interferes:

$$\text{Path difference} = 2 n_f d \cos\theta_f$$

where $\theta_f$ is the angle of propagation inside the film. The interference condition for minimum reflection (destructive interference of the two reflected beams) is:

$$2 n_f d \cos\theta_f = \left(m + \frac{1}{2}\right) \lambda, \quad m = 0, 1, 2, \ldots$$

For a quarter-wave anti-reflection (AR) coating at normal incidence ($m=0$):

$$d = \frac{\lambda}{4 n_f}$$

And zero reflection occurs when the film index satisfies:

$$n_f = \sqrt{n_\text{air} \cdot n_\text{substrate}}$$

<FresnelCalculator />

## Multi-layer stacks

Real image sensors use multiple thin films. The BARL (Bottom Anti-Reflection Layer) in a typical COMPASS configuration consists of alternating SiO2, HfO2, and Si3N4 layers:

```yaml
barl:
  layers:
    - thickness: 0.010   # SiO2
      material: "sio2"
    - thickness: 0.025   # HfO2
      material: "hfo2"
    - thickness: 0.015   # SiO2
      material: "sio2"
    - thickness: 0.030   # Si3N4
      material: "si3n4"
```

For multi-layer stacks, the transfer matrix method (TMM) provides an exact solution. Each layer is represented by a 2x2 matrix:

$$M_j = \begin{pmatrix} \cos\delta_j & -\frac{i}{\eta_j}\sin\delta_j \\ -i\eta_j\sin\delta_j & \cos\delta_j \end{pmatrix}$$

where $\delta_j = k_0 n_j d_j \cos\theta_j$ is the phase thickness and $\eta_j$ is the admittance (different for TE and TM). The total system matrix is the product:

$$M = M_1 \cdot M_2 \cdots M_N$$

The overall reflection and transmission coefficients follow from the matrix elements.

<ThinFilmReflectance />

## Role in BSI pixels

In a backside-illuminated pixel, light enters through the silicon backside and must pass through several layers before reaching the photodiode:

```
Incident light
      |
      v
  [Air]
  [Microlens]         -- focuses light onto pixel
  [Planarization]     -- uniform dielectric
  [Color filter]      -- wavelength-selective absorption
  [BARL layers]       -- anti-reflection at color-filter/silicon interface
  [Silicon + DTI]     -- photodiode region
```

The BARL stack is designed to minimize reflection at the color-filter-to-silicon interface. Without it, the large refractive index mismatch (color filter $n \approx 1.55$, silicon $n \approx 4$) would cause roughly 30--40% reflection, severely reducing QE.

## Spectral response effects

Thin film interference produces wavelength-dependent oscillations in the QE spectrum. These "Fabry-Perot" fringes are a common feature of image sensor simulations:

- **Constructive interference** at certain wavelengths boosts the QE.
- **Destructive interference** at other wavelengths creates dips.

The fringe spacing is approximately:

$$\Delta\lambda \approx \frac{\lambda^2}{2 n d}$$

For a 3 um silicon layer at 600 nm, $\Delta\lambda \approx 15$ nm. This means you need a wavelength step of at most 5 nm to resolve the fringes (Nyquist criterion).

::: warning
If your QE spectrum looks jagged or shows unexpected oscillations, check that your wavelength step is fine enough to resolve thin-film fringes. A step of 10 nm or smaller is recommended.
:::

## COMPASS implementation

COMPASS handles thin films differently depending on the solver:

- **RCWA**: Each uniform thin film layer is represented exactly as a single layer slice with thickness $d$ and permittivity $\varepsilon(\lambda)$. No approximation is needed.
- **FDTD**: Thin films must be resolved by the spatial grid. A 10 nm BARL layer requires grid spacing $\leq 5$ nm, which can increase memory and computation time.

The `PixelStack` class automatically constructs all layers from the YAML configuration and provides the appropriate representation to each solver type.
