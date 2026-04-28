---
title: FDTI / BDTI Pixel Simulator
---

# FDTI / BDTI Pixel Simulator

Build a virtual deep trench isolation CMOS image sensor pixel and inspect how FDTI/BDTI choice, trench width, BDTI depth, liner material, wavelength, and chief ray angle affect optical confinement.

<FdtiPixelSimulator />

## Model

This simulator uses a fast browser-side paraxial/ray approximation. It is intended as a visual design review and QA aid:

- The left panel shows a 3-pixel cross section with microlens, color filter, front oxide/gate stack, silicon, photodiodes, and DTI trenches.
- FDTI is modeled as a full-depth trench through the silicon. BDTI is modeled from the back-side optical surface into silicon with selectable depth.
- The right panel renders an approximate optical intensity map with photodiode collection regions overlaid.
- The visual QA scenarios compare the current design with no-DTI, shallow-BDTI, and full-FDTI oblique cases so layout changes can be checked by sight.

For sign-off accuracy, run the same geometry through COMPASS RCWA or FDTD workflows and use this view as a fast pre-check.
