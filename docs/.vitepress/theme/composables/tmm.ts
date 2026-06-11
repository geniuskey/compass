// Transfer Matrix Method (TMM) utilities for browser-based thin film calculations

// Complex number as [real, imag]
export type C = [number, number]

export function cadd(a: C, b: C): C { return [a[0]+b[0], a[1]+b[1]] }
export function csub(a: C, b: C): C { return [a[0]-b[0], a[1]-b[1]] }
export function cmul(a: C, b: C): C { return [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]] }
export function cdiv(a: C, b: C): C {
  const d = b[0]*b[0]+b[1]*b[1]
  return [(a[0]*b[0]+a[1]*b[1])/d, (a[1]*b[0]-a[0]*b[1])/d]
}
export function cabs2(a: C): number { return a[0]*a[0]+a[1]*a[1] }
export function cconj(a: C): C { return [a[0], -a[1]] }
export function cscale(a: C, s: number): C { return [a[0]*s, a[1]*s] }
export function csqrt(z: C): C {
  const r = Math.sqrt(z[0]*z[0]+z[1]*z[1])
  const t = Math.atan2(z[1], z[0])/2
  const sr = Math.sqrt(r)
  return [sr*Math.cos(t), sr*Math.sin(t)]
}
export function ccos(z: C): C {
  return [Math.cos(z[0])*Math.cosh(z[1]), -Math.sin(z[0])*Math.sinh(z[1])]
}
export function csin(z: C): C {
  return [Math.sin(z[0])*Math.cosh(z[1]), Math.cos(z[0])*Math.sinh(z[1])]
}

// Material system
export interface MaterialData {
  name: string
  type: 'tabulated' | 'sellmeier' | 'cauchy' | 'constant'
  table?: number[][]
  sellmeierB?: number[]
  sellmeierC?: number[]
  cauchyA?: number
  cauchyB?: number
  n?: number
  k?: number
}

function lerp(x: number, x0: number, x1: number, y0: number, y1: number): number {
  return y0 + (y1-y0)*(x-x0)/(x1-x0)
}

function interpTable(table: number[][], wl: number): C {
  if (wl <= table[0][0]) return [table[0][1], table[0][2]]
  if (wl >= table[table.length-1][0]) return [table[table.length-1][1], table[table.length-1][2]]
  for (let i = 0; i < table.length-1; i++) {
    if (wl >= table[i][0] && wl <= table[i+1][0]) {
      return [
        lerp(wl, table[i][0], table[i+1][0], table[i][1], table[i+1][1]),
        lerp(wl, table[i][0], table[i+1][0], table[i][2], table[i+1][2]),
      ]
    }
  }
  return [table[0][1], table[0][2]]
}

export function getN(mat: MaterialData, wl: number): C {
  switch (mat.type) {
    case 'constant': return [mat.n||1, mat.k||0]
    case 'sellmeier': {
      const l2 = wl*wl
      let n2 = 1
      for (let i = 0; i < mat.sellmeierB!.length; i++)
        n2 += mat.sellmeierB![i]*l2/(l2-mat.sellmeierC![i]**2)
      return [Math.sqrt(Math.max(n2,1)), 0]
    }
    case 'cauchy': return [mat.cauchyA!+mat.cauchyB!/(wl*wl), 0]
    case 'tabulated': return interpTable(mat.table!, wl)
    default: return [1,0]
  }
}

export const MATERIALS: Record<string, MaterialData> = {
  air: { name:'Air', type:'constant', n:1, k:0 },
  polymer: { name:'Polymer', type:'constant', n:1.56, k:0 },
  sio2: { name:'SiO₂', type:'sellmeier',
    sellmeierB:[0.6961663,0.4079426,0.8974794],
    sellmeierC:[0.0684043,0.1162414,9.896161] },
  si3n4: { name:'Si₃N₄', type:'sellmeier',
    sellmeierB:[2.8939], sellmeierC:[0.13967] },
  hfo2: { name:'HfO₂', type:'cauchy', cauchyA:1.88, cauchyB:0.0236 },
  tio2: { name:'TiO₂', type:'cauchy', cauchyA:2.2, cauchyB:0.06 },
  al2o3: { name:'Al₂O₃', type:'sellmeier',
    sellmeierB:[1.4313493,0.6505455,5.3414021],
    sellmeierC:[0.0726631,0.1193242,18.028251] },
  mgf2: { name:'MgF₂', type:'sellmeier',
    sellmeierB:[0.48755108,0.39875031,2.3120353],
    sellmeierC:[0.04338408,0.09461442,23.793604] },
  ta2o5: { name:'Ta₂O₅', type:'cauchy', cauchyA:2.06, cauchyB:0.045 },
  zro2: { name:'ZrO₂', type:'cauchy', cauchyA:2.12, cauchyB:0.042 },
  nb2o5: { name:'Nb₂O₅', type:'cauchy', cauchyA:2.23, cauchyB:0.055 },
  zns: { name:'ZnS', type:'cauchy', cauchyA:2.29, cauchyB:0.048 },
  znse: { name:'ZnSe', type:'cauchy', cauchyA:2.50, cauchyB:0.07 },
  caf2: { name:'CaF₂', type:'sellmeier',
    sellmeierB:[0.5675888,0.4710914,3.8484723],
    sellmeierC:[0.050263605,0.1003909,34.649040] },
  ito: { name:'ITO', type:'tabulated', table:[
    [.38,2.05,.028],[.40,2.02,.024],[.45,1.96,.016],[.50,1.92,.010],
    [.55,1.89,.007],[.60,1.87,.005],[.65,1.86,.004],[.70,1.85,.003],[.78,1.84,.002],
  ]},
  aln: { name:'AlN', type:'cauchy', cauchyA:2.02, cauchyB:0.030 },
  sic: { name:'SiC', type:'cauchy', cauchyA:2.55, cauchyB:0.05 },
  // Approximate values after Aspnes & Studna (1983); Ge absorbs strongly
  // across the visible (k ~ 2-2.5 below 560nm), E1 structure near 540-600nm.
  ge: { name:'Ge', type:'tabulated', table:[
    [.38,4.15,2.60],[.40,4.12,2.55],[.45,4.22,2.43],[.50,4.50,2.41],
    [.55,5.00,2.30],[.60,5.60,1.35],[.65,5.25,.700],[.70,4.95,.480],
    [.78,4.72,.340],[.85,4.60,.250],[.90,4.52,.180],[1.0,4.44,.080],
  ]},
  // Crystalline Si at 300K. n: Aspnes & Studna (1983); k = αλ/4π with α from
  // M.A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008).
  // Sanity anchors (1/α): ≈0.10µm @400nm, 1.4µm @550nm, 2.4µm @600nm, 12µm @800nm, 156µm @1000nm.
  silicon: { name:'Silicon', type:'tabulated', table:[
    [.35,5.44,2.90],[.36,6.01,2.91],[.37,6.86,2.06],[.38,6.55,.880],
    [.39,6.04,.440],[.40,5.57,.303],[.41,5.28,.220],[.42,5.10,.167],
    [.43,4.95,.134],[.44,4.82,.109],[.45,4.69,.0913],[.46,4.59,.0775],
    [.47,4.50,.0651],[.48,4.42,.0563],[.49,4.36,.0496],[.50,4.30,.0441],
    [.51,4.25,.0394],[.52,4.18,.0364],[.53,4.14,.0331],[.54,4.11,.0303],
    [.55,4.08,.0280],[.56,4.05,.0258],[.57,4.02,.0241],[.58,4.00,.0225],
    [.59,3.97,.0211],[.60,3.94,.0198],[.62,3.90,.0174],[.64,3.87,.0155],
    [.66,3.84,.0136],[.68,3.81,.0120],[.70,3.78,.0106],[.72,3.76,.0095],
    [.74,3.74,.0084],[.76,3.72,.0073],[.78,3.71,.0065],[.80,3.69,.0054],
    [.85,3.65,.0036],[.90,3.62,.0022],[.95,3.60,.0012],[1.0,3.58,.00051],
  ]},
  // Visible range after Palik; NIR points (>0.78µm) are an approximate
  // smooth extension (R ≈ 0.55-0.60 at 1µm, consistent with bulk W).
  tungsten: { name:'Tungsten', type:'tabulated', table:[
    [.38,3.39,2.66],[.40,3.46,2.72],[.45,3.55,2.86],[.50,3.61,2.98],
    [.55,3.65,3.08],[.60,3.68,3.17],[.65,3.70,3.25],[.70,3.72,3.33],
    [.78,3.74,3.44],[.85,3.74,3.50],[.90,3.74,3.56],[1.0,3.75,3.68],
  ]},
  // Pigment CFA dyes are nearly transparent in the NIR (hence IR-cut filters):
  // k rolls off beyond ~0.75µm. Red is a long-pass dye (transmits red→NIR).
  cf_red: { name:'CF Red', type:'tabulated', table:[
    [.38,1.55,.150],[.40,1.55,.150],[.44,1.55,.150],[.48,1.55,.149],
    [.50,1.55,.147],[.52,1.55,.141],[.54,1.55,.125],[.56,1.55,.095],
    [.58,1.55,.054],[.60,1.55,.016],[.62,1.55,.000],[.66,1.55,.000],
    [.70,1.55,.002],[.75,1.55,.003],[.80,1.55,.003],[.90,1.55,.003],
    [1.0,1.55,.003],
  ]},
  cf_green: { name:'CF Green', type:'tabulated', table:[
    [.38,1.55,.120],[.40,1.55,.120],[.42,1.55,.119],[.44,1.55,.115],
    [.46,1.55,.103],[.48,1.55,.076],[.50,1.55,.036],[.52,1.55,.005],
    [.53,1.55,.000],[.54,1.55,.005],[.56,1.55,.036],[.58,1.55,.076],
    [.60,1.55,.103],[.62,1.55,.115],[.64,1.55,.119],[.66,1.55,.120],
    [.68,1.55,.120],[.70,1.55,.118],[.74,1.55,.085],[.78,1.55,.055],
    [.82,1.55,.034],[.86,1.55,.020],[.90,1.55,.012],[.95,1.55,.007],
    [1.0,1.55,.005],
  ]},
  cf_blue: { name:'CF Blue', type:'tabulated', table:[
    [.38,1.55,.155],[.40,1.55,.114],[.42,1.55,.054],[.44,1.55,.007],
    [.45,1.55,.000],[.46,1.55,.007],[.48,1.55,.054],[.50,1.55,.114],
    [.52,1.55,.155],[.54,1.55,.173],[.56,1.55,.179],[.58,1.55,.180],
    [.60,1.55,.180],[.62,1.55,.180],[.64,1.55,.180],[.66,1.55,.180],
    [.68,1.55,.180],[.70,1.55,.178],[.74,1.55,.130],[.78,1.55,.090],
    [.82,1.55,.055],[.86,1.55,.032],[.90,1.55,.018],[.95,1.55,.010],
    [1.0,1.55,.006],
  ]},
}

export const MATERIAL_KEYS = Object.keys(MATERIALS)

/** Coating material keys suitable for thin film layer design (excludes air, substrates, color filters) */
export const COATING_MATERIALS = [
  'sio2','si3n4','hfo2','tio2','al2o3','mgf2','ta2o5','zro2',
  'nb2o5','zns','znse','caf2','ito','aln','sic','ge','polymer','tungsten',
] as const

/** Substrate material keys suitable as incident/exit media */
export const SUBSTRATE_MATERIALS = ['air','sio2','silicon','ge','polymer','glass'] as const

// Add glass as a simple constant material
MATERIALS['glass'] = { name:'Glass (BK7)', type:'sellmeier',
  sellmeierB:[1.03961212,0.231792344,1.01046945],
  sellmeierC:[0.00600069867,0.0200179144,103.560653] }

// TMM engine
export interface TmmLayer { material: string; thickness: number }
export interface TmmResult {
  R: number; T: number; A: number; layerA: number[]
}

type M2 = [C,C,C,C]
function m2mul(a: M2, b: M2): M2 {
  return [
    cadd(cmul(a[0],b[0]),cmul(a[1],b[2])),
    cadd(cmul(a[0],b[1]),cmul(a[1],b[3])),
    cadd(cmul(a[2],b[0]),cmul(a[3],b[2])),
    cadd(cmul(a[2],b[1]),cmul(a[3],b[3])),
  ]
}

function tmmPol(
  layers: TmmLayer[], n0Mat: string, nsMat: string,
  wl: number, angDeg: number, pol: 's'|'p'
): TmmResult {
  const n0 = getN(MATERIALS[n0Mat], wl)
  const ns = getN(MATERIALS[nsMat], wl)
  const th0 = angDeg*Math.PI/180
  const sinTh0: C = [Math.sin(th0),0]
  const cosTh0: C = [Math.cos(th0),0]
  const n0sin = cmul(n0, sinTh0)
  const k0 = 2*Math.PI/wl

  const cosTheta = (nj: C) => csqrt(csub([1,0], cmul(cdiv(n0sin,nj), cdiv(n0sin,nj))))
  const eta = (nj: C, ct: C) => pol==='s' ? cmul(nj,ct) : cdiv(nj,ct)

  const cosThs = cosTheta(ns)
  const eta0 = eta(n0, cosTh0)
  const etas = eta(ns, cosThs)

  const layerData: {nk:C, cosT:C, eta:C, delta:C}[] = []
  let M: M2 = [[1,0],[0,0],[0,0],[1,0]]

  for (const l of layers) {
    const nk = getN(MATERIALS[l.material], wl)
    const ct = cosTheta(nk)
    const et = eta(nk, ct)
    const delta = cscale(cmul(nk, ct), k0*l.thickness)
    const cd = ccos(delta), sd = csin(delta)
    const negI: C = [0,-1]
    const mj: M2 = [cd, cmul(negI, cdiv(sd,et)), cmul(negI, cmul(et,sd)), cd]
    M = m2mul(M, mj)
    layerData.push({nk, cosT:ct, eta:et, delta})
  }

  const e0M00 = cmul(eta0,M[0]), e0esM01 = cmul(cmul(eta0,etas),M[1])
  const esM11 = cmul(etas,M[3])
  const num = csub(csub(cadd(e0M00,e0esM01), M[2]), esM11)
  const den = cadd(cadd(e0M00,e0esM01), cadd(M[2], esM11))
  const r = cdiv(num, den)
  const t = cdiv(cscale(eta0,2), den)
  const R = cabs2(r)
  const Tr = (etas[0]/eta0[0])*cabs2(t)

  // Per-layer absorption via backward field propagation
  let E: C = t, H: C = cmul(etas, t)
  const layerA: number[] = new Array(layers.length)
  for (let j = layers.length-1; j >= 0; j--) {
    const Pbot = cmul(E, cconj(H))[0] / eta0[0]
    const {eta:et, delta} = layerData[j]
    const cd = ccos(delta), sd = csin(delta)
    const negI: C = [0,-1]
    const nE = cadd(cmul(cd,E), cmul(cmul(negI, cdiv(sd,et)), H))
    const nH = cadd(cmul(cmul(negI, cmul(et,sd)), E), cmul(cd,H))
    E = nE; H = nH
    const Ptop = cmul(E, cconj(H))[0] / eta0[0]
    layerA[j] = Math.max(0, Ptop - Pbot)
  }
  return { R, T:Tr, A:Math.max(0,1-R-Tr), layerA }
}

export function tmmCalc(
  layers: TmmLayer[], n0: string, ns: string,
  wl: number, angDeg = 0, pol: 's'|'p'|'avg' = 'avg'
): TmmResult {
  if (pol === 'avg') {
    const s = tmmPol(layers,n0,ns,wl,angDeg,'s')
    const p = tmmPol(layers,n0,ns,wl,angDeg,'p')
    return {
      R:(s.R+p.R)/2, T:(s.T+p.T)/2, A:(s.A+p.A)/2,
      layerA: s.layerA.map((a,i)=>(a+p.layerA[i])/2)
    }
  }
  return tmmPol(layers,n0,ns,wl,angDeg,pol)
}

export function tmmSpectrum(
  layers: TmmLayer[], n0: string, ns: string,
  wls: number[], angDeg = 0, pol: 's'|'p'|'avg' = 'avg'
): TmmResult[] {
  return wls.map(wl => tmmCalc(layers,n0,ns,wl,angDeg,pol))
}

export function wlRange(start: number, end: number, step: number): number[] {
  const r: number[] = []
  for (let w = start; w <= end+step*0.01; w += step) r.push(Math.round(w*1000)/1000)
  return r
}

/** Default BSI 1um pixel stack. Light: air → microlens → plnr → CF → BARL → Si. Substrate: SiO2 */
export function defaultBsiStack(cf: 'red'|'green'|'blue', siThick = 3.0): TmmLayer[] {
  const cfm = cf==='red'?'cf_red':cf==='green'?'cf_green':'cf_blue'
  return [
    {material:'polymer', thickness:0.6},
    {material:'sio2', thickness:0.3},
    {material:cfm, thickness:0.6},
    {material:'si3n4', thickness:0.030},
    {material:'sio2', thickness:0.015},
    {material:'hfo2', thickness:0.025},
    {material:'sio2', thickness:0.01},
    {material:'silicon', thickness:siThick},
  ]
}

/** Silicon layer index in defaultBsiStack */
export const SI_LAYER_IDX = 7
/** BARL layer indices in defaultBsiStack (si3n4, sio2, hfo2, sio2) */
export const BARL_INDICES = [3,4,5,6]

// CIE 1931 2° color matching functions (380-780nm, 5nm step)
export const CIE_WL: number[] = []
export const CIE_X: number[] = []
export const CIE_Y: number[] = []
export const CIE_Z: number[] = []
;(() => {
  // CIE 1931 standard colorimetric observer (2°), 380-780nm at 5nm steps
  const x = [0.0014,0.0022,0.0042,0.0076,0.0143,0.0232,0.0435,0.0776,0.1344,0.2148,0.2839,0.3285,0.3483,0.3481,0.3362,0.3187,0.2908,0.2511,0.1954,0.1421,0.0956,0.058,0.032,0.0147,0.0049,0.0024,0.0093,0.0291,0.0633,0.1096,0.1655,0.2257,0.2904,0.3597,0.4334,0.5121,0.5945,0.6784,0.7621,0.8425,0.9163,0.9786,1.0263,1.0567,1.0622,1.0456,1.0026,0.9384,0.8544,0.7514,0.6424,0.5419,0.4479,0.3608,0.2835,0.2187,0.1649,0.1212,0.0874,0.0636,0.0468,0.0329,0.0227,0.0158,0.0114,0.0081,0.0058,0.0041,0.0029,0.002,0.0014,0.001,0.0007,0.0005,0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,0]
  const yy = [0.0000,0.0001,0.0001,0.0002,0.0004,0.0006,0.0012,0.0022,0.0040,0.0073,0.0116,0.0168,0.0230,0.0298,0.0380,0.0480,0.0600,0.0739,0.0910,0.1126,0.1390,0.1693,0.2080,0.2586,0.3230,0.4073,0.5030,0.6082,0.7100,0.7932,0.8620,0.9149,0.9540,0.9803,0.9950,1.0000,0.9950,0.9786,0.9520,0.9154,0.8700,0.8163,0.7570,0.6949,0.6310,0.5668,0.5030,0.4412,0.3810,0.3210,0.2650,0.2170,0.1750,0.1382,0.1070,0.0816,0.0610,0.0446,0.0320,0.0232,0.0170,0.0119,0.0082,0.0057,0.0041,0.0029,0.0021,0.0015,0.0010,0.0007,0.0005,0.0004,0.0003,0.0002,0.0001,0.0001,0.0001,0.0000,0.0000,0.0000,0.0000]
  const z = [0.0065,0.0105,0.0201,0.0362,0.0679,0.1102,0.2074,0.3713,0.6456,1.0391,1.3856,1.6230,1.7471,1.7826,1.7721,1.7441,1.6692,1.5281,1.2876,1.0419,0.8130,0.6162,0.4652,0.3533,0.2720,0.2123,0.1582,0.1117,0.0782,0.0573,0.0422,0.0298,0.0203,0.0134,0.0087,0.0057,0.0039,0.0027,0.0021,0.0018,0.0017,0.0014,0.0011,0.001,0.0008,0.0006,0.0003,0.0002,0.0002,0.0001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for (let i = 0; i < 81; i++) {
    CIE_WL.push(0.380 + i*0.005)
    CIE_X.push(x[i] || 0)
    CIE_Y.push(yy[i] || 0)
    CIE_Z.push(z[i] || 0)
  }
})()

/** Convert spectral data to CIE XYZ. spectrum: transmittance at CIE_WL wavelengths */
export function spectrumToXYZ(spectrum: number[]): [number,number,number] {
  let X=0, Y=0, Z=0
  for (let i = 0; i < CIE_WL.length && i < spectrum.length; i++) {
    X += spectrum[i]*CIE_X[i]*5
    Y += spectrum[i]*CIE_Y[i]*5
    Z += spectrum[i]*CIE_Z[i]*5
  }
  return [X,Y,Z]
}

/** CIE XYZ to xy chromaticity */
export function xyzToXy(X: number, Y: number, Z: number): [number,number] {
  const s = X+Y+Z
  return s > 0 ? [X/s, Y/s] : [0.3127, 0.3290]
}
