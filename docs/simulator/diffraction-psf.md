---
title: Diffraction PSF Viewer
---

# Diffraction PSF Viewer

Visualize the Airy diffraction pattern from a circular aperture and see how it maps onto a pixel grid to understand energy collection efficiency.

<DiffractionPsfViewer />

## Physics

A circular aperture produces the Airy pattern as its point spread function (PSF):

**PSF(x) = [2 J₁(x) / x]²**

where x = pi D r / (lambda f) , D is the aperture diameter, f is the focal length, and J₁ is the first-order Bessel function.

### Key Quantities

- **Airy disk radius** = 1.22 lambda F/# — the first dark ring of the diffraction pattern. This sets the fundamental resolution limit.
- **Encircled energy** — the fraction of total energy within a given radius. About 84% falls within the first Airy ring.
- **Pixel collection efficiency** — when the pixel grid is overlaid on the PSF, the fraction captured by the central pixel depends on the ratio of pixel pitch to Airy disk diameter.

### Design Trade-offs

| F/# | Airy radius (550 nm) | Relative to 1.0 um pixel |
|-----|---------------------|--------------------------|
| 1.4 | 0.94 um | ~1x pixel pitch |
| 2.0 | 1.34 um | ~1.3x pixel pitch |
| 2.8 | 1.88 um | ~1.9x pixel pitch |

As F/# increases, the PSF spreads across more pixels, reducing per-pixel collection efficiency and increasing optical crosstalk.

::: info
This model assumes an ideal circular aperture. Real camera lenses have aberrations that broaden the PSF beyond the diffraction limit, especially at wide apertures.
:::
