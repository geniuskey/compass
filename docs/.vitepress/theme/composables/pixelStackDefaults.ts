export type BayerChannel = 'R' | 'G' | 'B'

export const pixelStackDefaults = {
  pitch: 1.0,
  unitCell: [2, 2] as const,
  air: {
    thickness: 1.0,
  },
  microlens: {
    height: 0.6,
    radiusX: 0.48,
    radiusY: 0.48,
    gap: 0.0,
    profileN: 2.5,
  },
  planarization: {
    thickness: 0.3,
  },
  colorFilter: {
    grid: {
      width: 0.05,
      thickness: 0.47,
    },
    channels: {
      R: { thickness: 0.62, contactAngle: 66, diagramFill: '#e74c3c', sectionFill: '#f87171' },
      G: { thickness: 0.60, contactAngle: 72, diagramFill: '#27ae60', sectionFill: '#4ade80' },
      B: { thickness: 0.65, contactAngle: 62, diagramFill: '#3498db', sectionFill: '#60a5fa' },
    } satisfies Record<BayerChannel, {
      thickness: number
      contactAngle: number
      diagramFill: string
      sectionFill: string
    }>,
  },
  barl: {
    layers: [
      { thickness: 0.010, material: 'SiO2', fill: '#7fb3d8' },
      { thickness: 0.025, material: 'HfO2', fill: '#6c71c4' },
      { thickness: 0.015, material: 'SiO2', fill: '#e8d44d' },
      { thickness: 0.030, material: 'Si3N4', fill: '#2aa198' },
    ],
  },
  silicon: {
    thickness: 3.0,
    dti: {
      width: 0.1,
      depth: 3.0,
    },
    photodiode: {
      centerDepth: 0.5,
      sizeX: 0.7,
      sizeY: 0.7,
      sizeZ: 2.0,
    },
  },
}

export const bayerCells2x2 = [
  { id: 'r0c0', key: 'R' as BayerChannel, row: 0, col: 0, x0: 0, x1: 1, y0: 0, y1: 1, cx: 0.5, cy: 0.5 },
  { id: 'r0c1', key: 'G' as BayerChannel, row: 0, col: 1, x0: 1, x1: 2, y0: 0, y1: 1, cx: 1.5, cy: 0.5 },
  { id: 'r1c0', key: 'G' as BayerChannel, row: 1, col: 0, x0: 0, x1: 1, y0: 1, y1: 2, cx: 0.5, cy: 1.5 },
  { id: 'r1c1', key: 'B' as BayerChannel, row: 1, col: 1, x0: 1, x1: 2, y0: 1, y1: 2, cx: 1.5, cy: 1.5 },
]

export function um(value: number, digits = 2): string {
  return `${value.toFixed(digits)} um`
}

export function umRange(values: number[], digits = 2): string {
  const sorted = [...values].sort((a, b) => a - b)
  return `${sorted[0].toFixed(digits)}-${sorted[sorted.length - 1].toFixed(digits)} um`
}
