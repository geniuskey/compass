import { describe, expect, it } from 'vitest'
import {
  GRID_N,
  LAYOUT_PRESETS,
  buildCellIndex,
  buildGridFromRects,
  densityFactor,
  deriveGroups,
  getGroupNeighbors,
  sigmaShape,
  type LensUnitShape,
} from '../../.vitepress/theme/composables/microlensProcessModel'

function presetGroups(preset: keyof typeof LAYOUT_PRESETS) {
  return deriveGroups(buildGridFromRects(LAYOUT_PRESETS[preset]))
}

function countsByKind(groups: ReturnType<typeof deriveGroups>) {
  const counts: Record<LensUnitShape, number> = { '1x1': 0, '2x1': 0, '1x2': 0, '2x2': 0 }
  for (const g of groups) counts[g.kind] += 1
  return counts
}

describe('buildGridFromRects', () => {
  it('fills 16 cells with unique ids when no rects are given', () => {
    const g = buildGridFromRects([])
    expect(g).toHaveLength(GRID_N * GRID_N)
    expect(new Set(g).size).toBe(16)
  })

  it('reuses the same id for cells covered by a rect', () => {
    const g = buildGridFromRects([[1, 1, 2, 2]])
    // four cells (1,1) (1,2) (2,1) (2,2) share id=0
    const id00 = g[1 * GRID_N + 1]
    expect(id00).toBe(0)
    expect(g[1 * GRID_N + 2]).toBe(0)
    expect(g[2 * GRID_N + 1]).toBe(0)
    expect(g[2 * GRID_N + 2]).toBe(0)
    // remaining 12 cells get unique ids starting at 1
    const others = g.filter((_, i) => {
      const r = Math.floor(i / GRID_N)
      const c = i % GRID_N
      return !(r >= 1 && r <= 2 && c >= 1 && c <= 2)
    })
    expect(new Set(others).size).toBe(12)
    expect(Math.min(...others)).toBe(1)
  })
})

describe('LAYOUT_PRESETS deriveGroups', () => {
  it('all-1x1 yields 16 × 1x1 groups, all valid', () => {
    const groups = presetGroups('all-1x1')
    expect(groups).toHaveLength(16)
    expect(countsByKind(groups)).toEqual({ '1x1': 16, '2x1': 0, '1x2': 0, '2x2': 0 })
    expect(groups.every(g => g.isValidShape)).toBe(true)
  })

  it('all-2x1 yields 8 × 2x1 groups, all valid', () => {
    const groups = presetGroups('all-2x1')
    expect(groups).toHaveLength(8)
    expect(countsByKind(groups)).toEqual({ '1x1': 0, '2x1': 8, '1x2': 0, '2x2': 0 })
  })

  it('all-1x2 yields 8 × 1x2 groups, all valid', () => {
    const groups = presetGroups('all-1x2')
    expect(groups).toHaveLength(8)
    expect(countsByKind(groups)).toEqual({ '1x1': 0, '2x1': 0, '1x2': 8, '2x2': 0 })
  })

  it('all-2x2 yields 4 × 2x2 groups, all valid', () => {
    const groups = presetGroups('all-2x2')
    expect(groups).toHaveLength(4)
    expect(countsByKind(groups)).toEqual({ '1x1': 0, '2x1': 0, '1x2': 0, '2x2': 4 })
  })

  it('mixed-2x2-pdaf is 1 × 2x2 + 12 × 1x1', () => {
    const groups = presetGroups('mixed-2x2-pdaf')
    expect(groups).toHaveLength(13)
    expect(countsByKind(groups)).toEqual({ '1x1': 12, '2x1': 0, '1x2': 0, '2x2': 1 })
  })

  it('sparse-2x1-pdaf is 2 × 2x1 + 12 × 1x1', () => {
    const groups = presetGroups('sparse-2x1-pdaf')
    expect(groups).toHaveLength(14)
    expect(countsByKind(groups)).toEqual({ '1x1': 12, '2x1': 2, '1x2': 0, '2x2': 0 })
  })
})

describe('deriveGroups invalid shapes', () => {
  it('flags an L-shaped group (three cells) as invalid', () => {
    // Cells (0,0), (0,1), (1,0) share id 0 → bbox is 2x2 with only 3 cells → invalid
    const g = new Array<number>(GRID_N * GRID_N).fill(0)
    let next = 1
    for (let i = 0; i < g.length; i += 1) {
      const r = Math.floor(i / GRID_N)
      const c = i % GRID_N
      const inL = (r === 0 && c === 0) || (r === 0 && c === 1) || (r === 1 && c === 0)
      if (!inL) g[i] = next++
    }
    const groups = deriveGroups(g)
    const lShape = groups.find(grp => grp.id === 0)!
    expect(lShape.cells).toHaveLength(3)
    expect(lShape.h).toBe(2)
    expect(lShape.w).toBe(2)
    expect(lShape.isValidShape).toBe(false)
  })

  it('flags a 3x1 group as invalid (only 1x1, 2x1, 1x2, 2x2 are allowed)', () => {
    const g = new Array<number>(GRID_N * GRID_N).fill(-1)
    g[0] = 0; g[1] = 0; g[2] = 0  // three cells in a row
    let next = 1
    for (let i = 0; i < g.length; i += 1) if (g[i] < 0) g[i] = next++
    const groups = deriveGroups(g)
    const tooWide = groups.find(grp => grp.id === 0)!
    expect(tooWide.h).toBe(1)
    expect(tooWide.w).toBe(3)
    expect(tooWide.isValidShape).toBe(false)
  })
})

describe('sigmaShape', () => {
  it('returns 0 for missing or invalid neighbor (grid edge)', () => {
    expect(sigmaShape(1, null)).toBe(0)
    expect(sigmaShape(1, undefined)).toBe(0)
    expect(sigmaShape(4, 0)).toBe(0)
  })

  it('returns 0 for equal-area pairs (uniform layout)', () => {
    expect(sigmaShape(1, 1)).toBe(0)
    expect(sigmaShape(2, 2)).toBe(0)
    expect(sigmaShape(4, 4)).toBe(0)
  })

  it('is +0.6 for a 1x1 with a 2x2 neighbor and -0.6 from the 2x2 side', () => {
    expect(sigmaShape(1, 4)).toBeCloseTo(0.6, 6)
    expect(sigmaShape(4, 1)).toBeCloseTo(-0.6, 6)
  })

  it('is +/-0.333 for a 1x1 next to a 2x1 (area 2)', () => {
    expect(sigmaShape(1, 2)).toBeCloseTo(1 / 3, 6)
    expect(sigmaShape(2, 1)).toBeCloseTo(-1 / 3, 6)
  })
})

describe('densityFactor', () => {
  it('is 0 for the sparsest pair (1x1 next to 1x1)', () => {
    expect(densityFactor(1, 1)).toBe(0)
  })

  it('treats grid-edge as a 1x1 (sparse padding)', () => {
    expect(densityFactor(1, null)).toBe(0)
    expect(densityFactor(4, null)).toBeCloseTo(0.5, 6)
  })

  it('is 0.5 for a mixed 1x1 + 2x2 boundary', () => {
    expect(densityFactor(1, 4)).toBeCloseTo(0.5, 6)
    expect(densityFactor(4, 1)).toBeCloseTo(0.5, 6)
  })

  it('is 1.0 for the densest pair (2x2 next to 2x2)', () => {
    expect(densityFactor(4, 4)).toBe(1)
  })
})

describe('getGroupNeighbors', () => {
  it('returns null on grid edges and the actual group on the inside', () => {
    const groups = presetGroups('all-1x1')
    const idx = buildCellIndex(groups)
    // Corner cell (0,0)
    const corner = groups.find(g => g.r0 === 0 && g.c0 === 0)!
    const n = getGroupNeighbors(corner, idx)
    expect(n.left).toBeNull()
    expect(n.top).toBeNull()
    expect(n.right).not.toBeNull()
    expect(n.bottom).not.toBeNull()
    expect(n.right!.r0).toBe(0)
    expect(n.right!.c0).toBe(1)
  })

  it('looks up the 2x2 OCL as the single neighbor on the matching side', () => {
    const groups = presetGroups('mixed-2x2-pdaf')
    const idx = buildCellIndex(groups)
    const oclLeftNeighbor = groups.find(g => g.r0 === 1 && g.c0 === 0)!
    const n = getGroupNeighbors(oclLeftNeighbor, idx)
    expect(n.right).not.toBeNull()
    expect(n.right!.kind).toBe('2x2')
  })
})
