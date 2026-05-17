---
title: Color Reproduction and Color Spaces
description: Sensor-facing color science for CMOS image sensors, from spectral response and camera RGB to CIE XYZ, Lab, sRGB, color correction matrices, and color error.
---

# Color Reproduction and Color Spaces

::: tip Prerequisites
[Signal Chain](/theory/sensor/signal-chain) -> this page -> [Color Accuracy Analyzer](/simulator/color-accuracy)
:::

Color spaces matter in image sensor work because a sensor does not measure "red", "green", and "blue" in the same way the eye or an sRGB display defines them. A CIS pixel measures three device-dependent spectral integrals. Color reproduction is the process of turning those integrals into a standard colorimetric space, usually CIE XYZ, then into a display or evaluation space such as sRGB or CIE Lab.

For COMPASS, this page is the bridge between optical simulation and image-quality metrics:

1. Electromagnetic or browser models produce channel spectral response.
2. Scene spectra and illuminants are integrated through that response.
3. White balance and a color correction matrix map camera RGB to XYZ.
4. Lab and color-difference metrics quantify the remaining error.

## Sensor Color Is Spectral Matching

For channel $c \in \{R,G,B\}$, a compact sensor response model is:

$$
Q_c(\lambda) =
T_{\text{lens}}(\lambda)
T_{\text{IR}}(\lambda)
T_{\text{CF},c}(\lambda)
\eta_{\text{abs},c}(\lambda)
\eta_{\text{col},c}(\lambda)
$$

where:

- $Q_c(\lambda)$ is the end-to-end spectral response of channel $c$.
- $T_{\text{lens}}(\lambda)$ is camera lens transmittance.
- $T_{\text{IR}}(\lambda)$ is the IR-cut filter transmittance.
- $T_{\text{CF},c}(\lambda)$ is the color filter transmittance.
- $\eta_{\text{abs},c}(\lambda)$ is optical absorption in the target photodiode region.
- $\eta_{\text{col},c}(\lambda)$ is the carrier collection efficiency after absorption.

For a surface patch $j$ under illuminant $E(\lambda)$ with reflectance $\rho_j(\lambda)$, the raw channel response is:

$$
r_{c,j} =
\int_{\lambda_1}^{\lambda_2}
E(\lambda)\rho_j(\lambda)Q_c(\lambda)\,d\lambda
$$

This equation is the core reason color is a sensor-design topic. CFA thickness, dye spectrum, microlens focus, DTI leakage, IR-cut slope, and silicon absorption depth all change $Q_c(\lambda)$, and therefore change the camera RGB triplet before any ISP correction is applied.

::: info Camera RGB is not display RGB
Raw camera RGB is a device-dependent coordinate system. Its axes are the sensor channel responses, not the CIE standard observer primaries and not the sRGB display primaries. Treating raw RGB as sRGB hides illuminant dependence, metameric error, and CFA trade-offs.
:::

## CIE XYZ Reference Color

CIE XYZ is the usual device-independent reference for color reproduction. Given the same illuminant and surface reflectance, the reference tristimulus vector is:

$$
\begin{bmatrix}
X_j \\
Y_j \\
Z_j
\end{bmatrix}
=
k
\int_{\lambda_1}^{\lambda_2}
E(\lambda)\rho_j(\lambda)
\begin{bmatrix}
\bar{x}(\lambda) \\
\bar{y}(\lambda) \\
\bar{z}(\lambda)
\end{bmatrix}
d\lambda
$$

where:

- $\bar{x}(\lambda)$, $\bar{y}(\lambda)$, and $\bar{z}(\lambda)$ are the CIE standard observer color matching functions.
- $k$ is a normalization constant, often chosen so the reference white has $Y=100$ or $Y=1$.
- $Y$ corresponds to photopic luminance in the CIE system.

Chromaticity removes absolute luminance:

$$
x = \frac{X}{X+Y+Z}, \qquad
y = \frac{Y}{X+Y+Z}
$$

The CIE 1931 chromaticity diagram used by the color-filter simulator plots these $(x,y)$ coordinates. It is useful for visualizing gamut, but it does not by itself guarantee low color error because luminance and nonlinear perceptual uniformity are not represented well in the 2D diagram.

## White Balance

White balance removes the first-order illuminant cast by forcing a neutral target to become neutral after channel scaling. If a neutral patch has raw response $\mathbf{r}_{w}=[r_R,r_G,r_B]^T$, a common green-referenced gain set is:

$$
g_R = \frac{r_G}{r_R}, \qquad
g_G = 1, \qquad
g_B = \frac{r_G}{r_B}
$$

The balanced camera vector is:

$$
\mathbf{c}_j =
\begin{bmatrix}
g_R & 0 & 0 \\
0 & g_G & 0 \\
0 & 0 & g_B
\end{bmatrix}
\left(\mathbf{r}_j-\mathbf{b}\right)
$$

where $\mathbf{b}$ is the black-level or dark offset. In a simulated sensor, $\mathbf{b}$ may be zero. In real calibration data it must be removed before color correction.

White balance alone cannot make the sensor match the human observer. It equalizes one white point, but it does not fix spectral mismatch across colored objects. That residual mismatch is handled by a color correction matrix.

## Color Correction Matrix

A 3x3 color correction matrix (CCM) maps white-balanced camera RGB into a target color space. For sensor evaluation, the target is usually XYZ:

$$
\hat{\mathbf{x}}_j =
\begin{bmatrix}
\hat{X}_j \\
\hat{Y}_j \\
\hat{Z}_j
\end{bmatrix}
=
M
\mathbf{c}_j
$$

The matrix is fit from calibration patches:

$$
M^\star =
\arg\min_M
\sum_j
w_j
\left\|
M\mathbf{c}_j - \mathbf{x}_j
\right\|_2^2
$$

where:

- $\mathbf{c}_j$ is the white-balanced camera vector for patch $j$.
- $\mathbf{x}_j=[X_j,Y_j,Z_j]^T$ is the reference CIE XYZ vector.
- $w_j$ is an optional patch weight.
- $M^\star$ is the least-squares CCM.

In matrix form, with camera samples $C=[\mathbf{c}_1,\dots,\mathbf{c}_n]$ and references $X=[\mathbf{x}_1,\dots,\mathbf{x}_n]$, the unregularized solution is:

$$
M^\star = XC^T(CC^T)^{-1}
$$

If the calibration set is noisy or poorly conditioned, ridge regularization is more stable:

$$
M^\star_\alpha = XC^T(CC^T+\alpha I)^{-1}
$$

The CCM can hide some optical compromises, but it cannot recover information that the sensor never measured. If two different spectra produce the same camera RGB but different XYZ values, no 3x3 matrix can separate them.

## Metamerism and Spectral Mismatch

Two spectra are metamers for a sensor when they produce the same camera response:

$$
\int E(\lambda)\rho_a(\lambda)Q_c(\lambda)\,d\lambda
=
\int E(\lambda)\rho_b(\lambda)Q_c(\lambda)\,d\lambda
\quad \text{for all } c
$$

They are visually equivalent only if the same equality also holds for the CIE color matching functions. Color error appears when a pair is a camera metamer but not a human-observer metamer, or the reverse.

This is why CFA spectra cannot be optimized only for narrow, saturated primaries. Very narrow RGB filters can increase chromatic separation, but they often reduce signal and can make color correction unstable under LED, fluorescent, or mixed illuminants. Broad filters increase sensitivity but reduce channel independence. The practical design target is a balanced spectral response that supports stable CCM fitting across expected illuminants and materials.

## From XYZ to sRGB

sRGB is a display-referred encoding. After estimating XYZ, a D65-adapted XYZ value can be converted to linear sRGB with:

$$
\begin{bmatrix}
R_{\text{lin}} \\
G_{\text{lin}} \\
B_{\text{lin}}
\end{bmatrix}
=
\begin{bmatrix}
3.2406 & -1.5372 & -0.4986 \\
-0.9689 & 1.8758 & 0.0415 \\
0.0557 & -0.2040 & 1.0570
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
$$

The nonlinear sRGB code value for each clipped linear component $u$ is:

$$
v =
\begin{cases}
12.92u, & u \le 0.0031308 \\
1.055u^{1/2.4} - 0.055, & u > 0.0031308
\end{cases}
$$

where $u$ is a linear-light display RGB component and $v$ is the nonlinear encoded component. This encoding is for display and file interchange. Do not apply it before CCM fitting or Lab error calculations unless the model explicitly expects display-coded values.

## Lab and Color Difference

CIE Lab is derived from XYZ relative to a reference white $(X_n,Y_n,Z_n)$:

$$
L^\star = 116 f\!\left(\frac{Y}{Y_n}\right)-16
$$

$$
a^\star = 500\left[
f\!\left(\frac{X}{X_n}\right) -
f\!\left(\frac{Y}{Y_n}\right)
\right]
$$

$$
b^\star = 200\left[
f\!\left(\frac{Y}{Y_n}\right) -
f\!\left(\frac{Z}{Z_n}\right)
\right]
$$

with:

$$
f(t)=
\begin{cases}
t^{1/3}, & t > \delta^3 \\
\dfrac{t}{3\delta^2} + \dfrac{4}{29}, & t \le \delta^3
\end{cases}
\qquad
\delta=\frac{6}{29}
$$

The simple 1976 color difference is:

$$
\Delta E_{ab}^\star =
\sqrt{
(\Delta L^\star)^2 +
(\Delta a^\star)^2 +
(\Delta b^\star)^2
}
$$

CIEDE2000, often written $\Delta E_{00}$, modifies the weighting of lightness, chroma, and hue terms to better match visual differences. It is more complex but is the preferred practical metric when comparing camera color accuracy because equal $\Delta E_{ab}^\star$ steps are not equally visible across the Lab space.

## Sensor Design Implications

Color reproduction constrains optical stack design in several ways:

| Design choice | Color consequence |
|---|---|
| CFA center wavelength and bandwidth | Sets channel separation, sensitivity, and CCM conditioning. |
| CFA thickness and relief | Changes both spectral passband and angular response, so off-axis color can differ from center-field color. |
| Microlens focus and CRA shift | Alters which photodiode collects each wavelength and can create color shading near the image edge. |
| IR-cut transition | Residual NIR leaks into all channels, reducing saturation and corrupting white balance. |
| Silicon thickness and absorption depth | Blue light is absorbed shallowly while red/NIR penetrates deeper, affecting crosstalk and channel balance. |
| Noise and clipping | CCM coefficients can amplify noisy channels, while saturated channels destroy chromatic information. |

A robust color design should therefore be evaluated with spectra, not only with RGB endpoint intuition. The minimum useful evaluation set is:

1. D65 and Illuminant A, plus the LED or fluorescent spectra relevant to the product.
2. A neutral scale for white balance and exposure.
3. A patch set with known reflectance spectra, such as a ColorChecker-style target.
4. Color error before and after CCM.
5. SNR or noise-amplification checks after the CCM.

## COMPASS Workflow

Use the pages in this order when connecting optical simulation to color:

1. [Pixel Optical Effects](/theory/sensor/pixel-optical-effects) for CFA, CRA, crosstalk, and stack-level effects.
2. [Quantum Efficiency](/theory/sensor/quantum-efficiency) for converting absorbed power into channel spectral response.
3. [Signal Chain](/theory/sensor/signal-chain) for illuminant, scene reflectance, lens, IR filter, and electron signal.
4. [Color Filter Designer](/simulator/color-filter) for quick spectral-shape and gamut exploration.
5. [Color Accuracy Analyzer](/simulator/color-accuracy) for CCM fitting and color-error intuition.
6. [Signal Chain Color Accuracy](/cookbook/signal-chain-color-accuracy) for an end-to-end recipe.

For a production sensor, the browser simulators are early design and communication tools. Sign-off requires measured CFA spectra, module transmittance, dark/flat-field calibration, realistic illuminant sets, and validation against silicon images.

## References

- [CIE 015:2018, Colorimetry, 4th Edition](https://www.cie.co.at/publications/colorimetry-4th-edition) defines the standard colorimetric observers, illuminants, tristimulus calculations, color spaces, and color-difference practices used here.
- [IEC 61966-2-1:1999, Default RGB colour space - sRGB](https://webstore.iec.ch/en/publication/6169) defines the sRGB color space and encoding used for display-referred RGB.
- [Sharma, Wu, and Dalal, "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations"](https://doi.org/10.1002/col.20070) is the practical implementation reference for $\Delta E_{00}$.
- [NIST, "Color By Numbers"](https://www.nist.gov/publications/color-numbers-using-calibration-source-spectrally-matched-your-test-source-key) explains why spectral responsivity mismatch affects color measurement accuracy.
