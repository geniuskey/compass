// Pure functions used by MicrolensProcessShape.vue. Kept here so unit tests
// can import them without mounting the Vue component.

export type LensUnitShape = '1x1' | '2x1' | '1x2' | '2x2'

export type LayoutPreset =
  | 'all-1x1'
  | 'all-2x1'
  | 'all-1x2'
  | 'all-2x2'
  | 'mixed-2x2-pdaf'
  | 'sparse-2x1-pdaf'
  | 'custom'

export interface LensGroup {
  id: number
  cells: { r: number; c: number }[]
  r0: number
  c0: number
  h: number
  w: number
  kind: LensUnitShape
  isValidShape: boolean
}

export type Rect = [number, number, number, number]  // [r0, c0, h, w]

export const GRID_N = 4

export function buildGridFromRects(rects: Rect[]): number[] {
  const g = new Array<number>(GRID_N * GRID_N).fill(-1)
  let id = 0
  for (const [r0, c0, h, w] of rects) {
    for (let dr = 0; dr < h; dr += 1) {
      for (let dc = 0; dc < w; dc += 1) {
        g[(r0 + dr) * GRID_N + (c0 + dc)] = id
      }
    }
    id += 1
  }
  for (let i = 0; i < g.length; i += 1) {
    if (g[i] < 0) {
      g[i] = id
      id += 1
    }
  }
  return g
}

export const LAYOUT_PRESETS: Record<Exclude<LayoutPreset, 'custom'>, Rect[]> = {
  'all-1x1': [],
  'all-2x1': [
    [0, 0, 1, 2], [0, 2, 1, 2],
    [1, 0, 1, 2], [1, 2, 1, 2],
    [2, 0, 1, 2], [2, 2, 1, 2],
    [3, 0, 1, 2], [3, 2, 1, 2],
  ],
  'all-1x2': [
    [0, 0, 2, 1], [0, 1, 2, 1], [0, 2, 2, 1], [0, 3, 2, 1],
    [2, 0, 2, 1], [2, 1, 2, 1], [2, 2, 2, 1], [2, 3, 2, 1],
  ],
  'all-2x2': [[0, 0, 2, 2], [0, 2, 2, 2], [2, 0, 2, 2], [2, 2, 2, 2]],
  'mixed-2x2-pdaf': [[1, 1, 2, 2]],
  'sparse-2x1-pdaf': [[1, 1, 1, 2], [2, 1, 1, 2]],
}

export function deriveGroups(grid: number[]): LensGroup[] {
  const map = new Map<number, { r: number; c: number }[]>()
  for (let i = 0; i < grid.length; i += 1) {
    const id = grid[i]
    const r = Math.floor(i / GRID_N)
    const c = i % GRID_N
    if (!map.has(id)) map.set(id, [])
    map.get(id)!.push({ r, c })
  }
  const groups: LensGroup[] = []
  for (const [id, cells] of map) {
    const rs = cells.map(p => p.r)
    const cs = cells.map(p => p.c)
    const r0 = Math.min(...rs)
    const r1 = Math.max(...rs)
    const c0 = Math.min(...cs)
    const c1 = Math.max(...cs)
    const h = r1 - r0 + 1
    const w = c1 - c0 + 1
    const rectangular = cells.length === h * w
    const allowed = (h === 1 || h === 2) && (w === 1 || w === 2)
    const isValidShape = rectangular && allowed
    let kind: LensUnitShape = '1x1'
    if (h === 1 && w === 2) kind = '2x1'
    else if (h === 2 && w === 1) kind = '1x2'
    else if (h === 2 && w === 2) kind = '2x2'
    groups.push({ id, cells, r0, c0, h, w, kind, isValidShape })
  }
  groups.sort((a, b) => (a.r0 - b.r0) || (a.c0 - b.c0))
  return groups
}

export function buildCellIndex(groups: LensGroup[]): Map<string, LensGroup> {
  const m = new Map<string, LensGroup>()
  for (const g of groups) {
    for (const cell of g.cells) m.set(`${cell.r},${cell.c}`, g)
  }
  return m
}

export interface GroupNeighbors {
  left: LensGroup | null
  right: LensGroup | null
  top: LensGroup | null
  bottom: LensGroup | null
}

export function getGroupNeighbors(group: LensGroup, cellIndex: Map<string, LensGroup>): GroupNeighbors {
  const rMid = group.r0 + Math.floor(group.h / 2)
  const cMid = group.c0 + Math.floor(group.w / 2)
  const pick = (r: number, c: number): LensGroup | null => {
    if (r < 0 || r >= GRID_N || c < 0 || c >= GRID_N) return null
    const n = cellIndex.get(`${r},${c}`)
    return n && n.id !== group.id ? n : null
  }
  return {
    left: pick(rMid, group.c0 - 1),
    right: pick(rMid, group.c0 + group.w),
    top: pick(group.r0 - 1, cMid),
    bottom: pick(group.r0 + group.h, cMid),
  }
}

// Shape-asymmetry: positive when neighbor is bigger than this group.
// Returns 0 for missing/invalid neighbor or equal-area pair (so uniform
// layouts collapse to the independent-lens baseline).
export function sigmaShape(A_G: number, A_N: number | null | undefined): number {
  if (!A_N || A_N <= 0) return 0
  return (A_N - A_G) / (A_N + A_G)
}

// Per-edge pattern-density score for plasma microloading: 0 for the
// sparsest baseline (two 1x1 cells; A_G=A_N=1), 1 for the densest
// (two 2x2 cells; A_G=A_N=4). Missing neighbor is treated as 1x1.
export function densityFactor(A_G: number, A_N: number | null | undefined): number {
  const a = A_N && A_N > 0 ? A_N : 1
  return Math.max(0, ((A_G + a) - 2) / 6)
}
