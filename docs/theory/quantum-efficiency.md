# Quantum Efficiency

Quantum Efficiency (QE) is the primary figure of merit computed by COMPASS. It measures how effectively a pixel converts incident photons into electrical signal.

## Definition

The external quantum efficiency at wavelength $\lambda$ is:

$$\text{QE}(\lambda) = \frac{\text{Number of electron-hole pairs collected}}{\text{Number of incident photons}}$$

Equivalently, in terms of optical power:

$$\text{QE}(\lambda) = \frac{P_\text{absorbed in PD}(\lambda)}{P_\text{incident}(\lambda)}$$

where $P_\text{absorbed in PD}$ is the power absorbed within the photodiode volume and $P_\text{incident}$ is the total incident power. QE is dimensionless and ranges from 0 to 1 (0% to 100%).

## Factors affecting QE

The total QE is the product of several loss factors:

$$\text{QE} = (1 - R) \times T_\text{optics} \times \eta_\text{abs} \times \text{FF}_\text{PD}$$

| Factor | Symbol | Description |
|--------|--------|-------------|
| Surface reflection | $R$ | Light reflected by the pixel stack (BARL reduces this) |
| Optical transmission | $T_\text{optics}$ | Fraction transmitted through color filter, planarization, etc. |
| Silicon absorption | $\eta_\text{abs}$ | Fraction of light absorbed within the silicon thickness |
| Photodiode fill factor | $\text{FF}_\text{PD}$ | Fraction of absorbed photons within the PD collection volume |

In COMPASS, the full-wave simulation captures all these effects simultaneously. The QE is not computed from this factored form but rather from the total electromagnetic solution.

## Computing QE in COMPASS

COMPASS computes QE by two methods:

### Method 1: Absorption integral

The absorbed power density at position $\mathbf{r}$ and wavelength $\lambda$ is:

$$p_\text{abs}(\mathbf{r}) = \frac{1}{2} \omega \varepsilon_0 \text{Im}(\varepsilon_r) |\mathbf{E}(\mathbf{r})|^2$$

The QE for a specific pixel (photodiode region $V_\text{PD}$) is:

$$\text{QE} = \frac{\int_{V_\text{PD}} p_\text{abs} \, dV}{P_\text{incident}}$$

### Method 2: Poynting flux difference

Alternatively, the power absorbed in a region can be found from the net Poynting flux entering and leaving the region:

$$P_\text{absorbed} = \oint_S \langle \mathbf{S} \rangle \cdot \hat{n} \, dA = S_{z,\text{top}} - S_{z,\text{bottom}}$$

where $S_{z,\text{top}}$ and $S_{z,\text{bottom}}$ are the z-components of the Poynting vector at the top and bottom of the photodiode region.

Both methods are implemented in the `QECalculator` class.

## Per-pixel QE and color channels

A 2x2 Bayer unit cell has four pixels, each with its own photodiode:

```
  +--------+--------+
  | R_0_0  | G_0_1  |
  +--------+--------+
  | G_1_0  | B_1_1  |
  +--------+--------+
```

COMPASS computes QE independently for each photodiode. The naming convention is `{Color}_{row}_{col}`.

The `spectral_response` function averages QE across pixels of the same color to produce per-channel QE curves:

```python
from compass.analysis.qe_calculator import QECalculator

# result.qe_per_pixel = {"R_0_0": array, "G_0_1": array, "G_1_0": array, "B_1_1": array}
channel_qe = QECalculator.spectral_response(result.qe_per_pixel, result.wavelengths)
# channel_qe = {"R": (wavelengths, qe_R), "G": (wavelengths, qe_G_avg), "B": (wavelengths, qe_B)}
```

## Crosstalk

Optical crosstalk occurs when light intended for one pixel is absorbed by a neighboring pixel. COMPASS quantifies this with a **crosstalk matrix**:

$$\text{CT}_{ij}(\lambda) = \frac{\text{QE}_j(\lambda, \text{illuminating pixel } i)}{\sum_k \text{QE}_k(\lambda, \text{illuminating pixel } i)}$$

The diagonal elements represent correctly detected signal; off-diagonal elements represent crosstalk. Lower crosstalk means better color separation.

The `QECalculator.compute_crosstalk` method computes this matrix from the per-pixel QE data.

<CrosstalkHeatmap />

## Energy balance

A fundamental physical constraint is energy conservation:

$$R(\lambda) + T(\lambda) + A(\lambda) = 1$$

where $R$ is total reflection, $T$ is total transmission (through the bottom), and $A$ is total absorption (in all materials). If this balance is violated by more than 1-2%, the simulation may have numerical issues.

The total QE across all pixels is bounded by the total absorption in silicon:

$$\sum_\text{pixels} \text{QE}_i \leq A_\text{Si}$$

The inequality is strict because some light is absorbed in the color filter, metal grid, and other non-photodiode regions.

## Typical QE spectra

<QESpectrumChart />

For a well-designed 1 um pitch BSI pixel:

| Channel | Peak QE | Peak wavelength |
|---------|---------|-----------------|
| Blue | 50-70% | 450-470 nm |
| Green | 60-80% | 530-560 nm |
| Red | 50-70% | 600-630 nm |

The QE spectrum typically shows:
- A sharp rise at the blue edge due to increasing silicon absorption
- A peak in the passband of each color filter
- A gradual decline at the red edge due to decreasing silicon absorption (absorption depth exceeds pixel thickness)
- Spectral ripples from thin-film interference in the BARL stack

::: info
QE above 80% for a single channel is rare because color filter absorption, reflection losses, and photodiode fill factor all reduce the total efficiency.
:::
