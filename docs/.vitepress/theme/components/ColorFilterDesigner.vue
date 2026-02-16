<template>
  <div :class="['cf-container',{'cf-fullscreen':isFullscreen}]">
    <div class="cf-header">
      <div>
        <h4>{{ t('Color Filter Designer & Gamut Viewer', '컬러 필터 설계 및 색역 뷰어') }}</h4>
        <p v-if="!isFullscreen" class="component-description">{{ t('Design color filter spectral responses with multiple models, IR cut filter, and evaluate gamut, crosstalk, CCM quality, and Vora-Value.','다양한 모델로 컬러 필터 분광 응답을 설계하고, IR 차단 필터, 색역, 크로스토크, CCM 품질, Vora-Value를 평가합니다.') }}</p>
      </div>
      <button class="fs-btn" @click="toggleFullscreen" :title="t('Toggle fullscreen','전체화면 전환')">{{ isFullscreen ? '\u00d7' : '\u26f6' }}</button>
    </div>

    <!-- Top controls -->
    <div class="top-controls">
      <div class="ctrl-group">
        <label class="ctrl-label">{{ t('Model','모델') }}</label>
        <div class="btn-row">
          <button v-for="m in [['gaussian','Gaussian','가우시안'],['lorentzian','Lorentzian','로렌츠'],['realDye','Real Dye','실측 염료']]" :key="m[0]" :class="['toggle-btn',{active:filterModel===m[0]}]" @click="filterModel=m[0] as ModelType">{{ t(m[1],m[2]) }}</button>
        </div>
      </div>
      <div class="ctrl-group">
        <div class="slider-row"><label>{{ t('CF Thickness','CF 두께') }}: <strong>{{ cfThickness.toFixed(2) }} μm</strong></label>
        <input type="range" min="0.2" max="1.2" step="0.05" v-model.number="cfThickness" class="ctrl-range" /></div>
      </div>
      <div class="ctrl-group">
        <div class="slider-row"><label>{{ t('CRA','주광선각') }}: <strong>{{ craAngle }}°</strong></label>
        <input type="range" min="0" max="30" step="1" v-model.number="craAngle" class="ctrl-range" /></div>
      </div>
      <div class="ctrl-group">
        <label class="cb-label"><input type="checkbox" v-model="irCutEnabled" /> {{ t('IR Cut','IR 차단') }}</label>
        <div v-if="irCutEnabled" class="slider-row"><label>{{ t('Cutoff','차단') }}: <strong>{{ irCutoff }} nm</strong></label>
        <input type="range" min="620" max="720" step="5" v-model.number="irCutoff" class="ctrl-range" /></div>
      </div>
    </div>

    <!-- Main body: sidebar + charts -->
    <div class="cf-body">
      <div class="cf-sidebar">
        <!-- Filter controls -->
        <div v-if="filterModel!=='realDye'" class="filter-controls">
          <div v-for="f in filters" :key="f.id" class="filter-group" :style="{borderLeftColor:f.color}">
            <div class="filter-header"><span class="filter-dot" :style="{background:f.color}"></span><strong>{{ t(f.nameEn,f.nameKo) }}</strong></div>
            <div class="filter-sliders">
              <div class="slider-row"><label>{{ t('Center','중심') }}: <strong>{{ f.center.value }} nm</strong></label><input type="range" :min="f.centerMin" :max="f.centerMax" step="1" v-model.number="f.center.value" class="ctrl-range" /></div>
              <div class="slider-row"><label>{{ t('FWHM','반치폭') }}: <strong>{{ f.fwhm.value }} nm</strong></label><input type="range" min="20" max="120" step="2" v-model.number="f.fwhm.value" class="ctrl-range" /></div>
              <div class="slider-row"><label>{{ t('Peak','피크') }}: <strong>{{ f.peak.value }}%</strong></label><input type="range" min="50" max="100" step="1" v-model.number="f.peak.value" class="ctrl-range" /></div>
            </div>
          </div>
        </div>

        <!-- Metrics -->
        <div class="results-grid">
          <div class="result-card gamut-card"><div class="result-label">{{ t('Gamut','색역') }}</div><div class="result-value highlight">{{ gamutAreaPct.srgb.toFixed(1) }}% sRGB</div><div class="result-sub">{{ gamutAreaPct.ntsc.toFixed(1) }}% NTSC</div></div>
          <div class="result-card" style="border-top:3px solid #2ecc71"><div class="result-label">{{ t('Peak QE','피크 QE') }}</div><div class="result-value">{{ (Math.max(peakQE.r,peakQE.g,peakQE.b)*100).toFixed(1) }}%</div><div class="result-sub">R:{{ (peakQE.r*100).toFixed(0) }} G:{{ (peakQE.g*100).toFixed(0) }} B:{{ (peakQE.b*100).toFixed(0) }}</div></div>
          <div class="result-card" style="border-top:3px solid #e67e22"><div class="result-label">{{ t('Crosstalk','크로스토크') }}</div><div class="result-value">{{ (avgCrosstalk*100).toFixed(1) }}%</div><div class="result-sub">{{ t('avg off-diag','평균 비대각') }}</div></div>
          <div class="result-card" style="border-top:3px solid #9b59b6"><div class="result-label">CCM κ</div><div class="result-value" :style="{color:ccmCond<5?'#2ecc71':ccmCond<10?'#e67e22':'#e74c3c'}">{{ ccmCond.toFixed(1) }}</div><div class="result-sub">{{ t('condition #','조건수') }}</div></div>
          <div class="result-card" style="border-top:3px solid #1abc9c"><div class="result-label">Vora-Value</div><div class="result-value" :style="{color:voraVal>0.95?'#2ecc71':voraVal>0.9?'#e67e22':'#e74c3c'}">{{ voraVal.toFixed(3) }}</div><div class="result-sub">{{ t('colorimetric quality','색계측 품질') }}</div></div>
          <div v-for="f in filters" :key="'info-'+f.id" class="result-card" :style="{borderTop:`3px solid ${f.color}`}">
            <div class="result-label">{{ t(f.nameEn,f.nameKo) }}</div><div class="result-value">{{ filterChroma[f.id].domWl }} nm</div><div class="result-sub">{{ t('Purity','순도') }}: {{ (filterChroma[f.id].purity*100).toFixed(1) }}%</div>
          </div>
        </div>

        <!-- Analysis: Crosstalk + WB -->
        <div class="analysis-row">
          <div class="analysis-box">
            <h5>{{ t('Spectral Crosstalk Matrix','분광 크로스토크 매트릭스') }}</h5>
            <table class="xtalk-table">
              <tr><th></th><th>B</th><th>G</th><th>R</th></tr>
              <tr v-for="(f,i) in filters" :key="'xt-'+f.id">
                <td :style="{color:f.color,fontWeight:600}">{{ f.id.toUpperCase() }}</td>
                <td v-for="(v,j) in crosstalkMatrix[i]" :key="j" :class="{'xt-diag':(i===0&&j===2)||(i===1&&j===1)||(i===2&&j===0)}">{{ (v*100).toFixed(1) }}%</td>
              </tr>
            </table>
          </div>
          <div class="analysis-box">
            <h5>{{ t('WB Coefficients','화이트 밸런스 계수') }}</h5>
            <table class="wb-table">
              <tr><th>{{ t('Illum','광원') }}</th><th style="color:#e74c3c">R</th><th style="color:#27ae60">G</th><th style="color:#3498db">B</th></tr>
              <tr v-for="wb in wbCoeffs" :key="wb.key">
                <td class="wb-illum">{{ wb.label }}</td>
                <td v-for="(g,i) in wb.gains" :key="i" :style="{color:filters[i].color}">{{ g.toFixed(2) }}</td>
              </tr>
            </table>
          </div>
        </div>

        <div class="export-row"><button class="export-btn" @click="exportConfig">{{ t('Export Design (JSON)','설계 내보내기 (JSON)') }}</button></div>
      </div>

      <!-- Spectrum chart -->
      <div class="chart-section chart-spectrum">
        <h5>{{ t('Filter Spectra','필터 스펙트럼') }} <label class="qe-toggle"><input type="checkbox" v-model="showQE" /> {{ t('QE overlay','QE 오버레이') }}</label></h5>
        <div class="svg-wrapper">
          <svg :viewBox="`0 0 ${specW} ${specH}`" class="spec-svg" @mousemove="onSpecMouseMove" @mouseleave="specHover=null">
            <defs><linearGradient id="cfVisSpectrum" x1="0" y1="0" x2="1" y2="0"><stop v-for="s in spectrumStops" :key="s.offset" :offset="s.offset" :stop-color="s.color" /></linearGradient></defs>
            <rect :x="specXScale(380)" :y="specPad.top+specPlotH+2" :width="specXScale(780)-specXScale(380)" height="6" fill="url(#cfVisSpectrum)" rx="2" />
            <line v-for="tick in specXTicks" :key="'xg'+tick" :x1="specXScale(tick)" :y1="specPad.top" :x2="specXScale(tick)" :y2="specPad.top+specPlotH" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line v-for="tick in specYTicks" :key="'yg'+tick" :x1="specPad.left" :y1="specYScale(tick)" :x2="specPad.left+specPlotW" :y2="specYScale(tick)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
            <line :x1="specPad.left" :y1="specPad.top" :x2="specPad.left" :y2="specPad.top+specPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="specPad.left" :y1="specPad.top+specPlotH" :x2="specPad.left+specPlotW" :y2="specPad.top+specPlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <text v-for="tick in specXTicks" :key="'xl'+tick" :x="specXScale(tick)" :y="specPad.top+specPlotH+20" text-anchor="middle" class="tick-label">{{ tick }}</text>
            <text v-for="tick in specYTicks" :key="'yl'+tick" :x="specPad.left-6" :y="specYScale(tick)+3" text-anchor="end" class="tick-label">{{ tick }}%</text>
            <text :x="specPad.left+specPlotW/2" :y="specH-2" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)','파장 (nm)') }}</text>
            <text :x="12" :y="specPad.top+specPlotH/2" text-anchor="middle" class="axis-title" :transform="`rotate(-90,12,${specPad.top+specPlotH/2})`">{{ t('T / QE (%)','투과율 / QE (%)') }}</text>
            <path v-if="irCutEnabled" :d="irCutPath" fill="none" stroke="#c0392b" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7" />
            <path v-for="f in filters" :key="'fill-'+f.id" :d="filterAreaPath(f)" :fill="f.color" opacity="0.12" />
            <path v-for="f in filters" :key="'curve-'+f.id" :d="filterCurvePath(f)" fill="none" :stroke="f.color" stroke-width="2" />
            <template v-if="showQE"><path v-for="f in filters" :key="'qe-'+f.id" :d="qeCurvePath(f.id)" fill="none" :stroke="f.color" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8" /></template>
            <template v-if="specHover">
              <line :x1="specHover.sx" :y1="specPad.top" :x2="specHover.sx" :y2="specPad.top+specPlotH" stroke="var(--vp-c-text-2)" stroke-width="0.8" stroke-dasharray="4,3" />
              <rect :x="specHover.tx" :y="specPad.top+4" width="140" :height="showQE?86:50" rx="4" fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95" />
              <text :x="specHover.tx+6" :y="specPad.top+16" class="tooltip-text">&lambda; = {{ specHover.wl }} nm</text>
              <text :x="specHover.tx+6" :y="specPad.top+28" class="tooltip-text" fill="#e74c3c">R: {{ specHover.r.toFixed(1) }}%</text>
              <text :x="specHover.tx+6" :y="specPad.top+40" class="tooltip-text" fill="#27ae60">G: {{ specHover.g.toFixed(1) }}%</text>
              <text :x="specHover.tx+6" :y="specPad.top+52" class="tooltip-text" fill="#3498db">B: {{ specHover.b.toFixed(1) }}%</text>
              <template v-if="showQE">
                <text :x="specHover.tx+74" :y="specPad.top+28" class="tooltip-text" fill="#e74c3c">QE {{ specHover.qr.toFixed(1) }}%</text>
                <text :x="specHover.tx+74" :y="specPad.top+40" class="tooltip-text" fill="#27ae60">QE {{ specHover.qg.toFixed(1) }}%</text>
                <text :x="specHover.tx+74" :y="specPad.top+52" class="tooltip-text" fill="#3498db">QE {{ specHover.qb.toFixed(1) }}%</text>
              </template>
            </template>
          </svg>
        </div>
      </div>

      <!-- CIE 1931 -->
      <div class="chart-section chart-cie">
        <h5>{{ t('CIE 1931 Chromaticity Diagram','CIE 1931 색도도') }}</h5>
        <div class="svg-wrapper cie-wrapper">
          <svg :viewBox="`0 0 ${cieW} ${cieH}`" class="cie-svg">
            <line :x1="ciePad.left" :y1="ciePad.top" :x2="ciePad.left" :y2="ciePad.top+ciePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <line :x1="ciePad.left" :y1="ciePad.top+ciePlotH" :x2="ciePad.left+ciePlotW" :y2="ciePad.top+ciePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
            <template v-for="tick in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]" :key="'cx'+tick">
              <line :x1="cieXScale(tick)" :y1="ciePad.top" :x2="cieXScale(tick)" :y2="ciePad.top+ciePlotH" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
              <text :x="cieXScale(tick)" :y="ciePad.top+ciePlotH+14" text-anchor="middle" class="tick-label">{{ tick.toFixed(1) }}</text>
            </template>
            <template v-for="tick in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]" :key="'cy'+tick">
              <line :x1="ciePad.left" :y1="cieYScale(tick)" :x2="ciePad.left+ciePlotW" :y2="cieYScale(tick)" stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3" />
              <text :x="ciePad.left-6" :y="cieYScale(tick)+3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
            </template>
            <text :x="ciePad.left+ciePlotW/2" :y="cieH-2" text-anchor="middle" class="axis-title">x</text>
            <text :x="10" :y="ciePad.top+ciePlotH/2" text-anchor="middle" class="axis-title" :transform="`rotate(-90,10,${ciePad.top+ciePlotH/2})`">y</text>
            <path :d="locusPath" fill="none" stroke="var(--vp-c-text-2)" stroke-width="1.5" />
            <line :x1="cieXScale(locusPoints[0][0])" :y1="cieYScale(locusPoints[0][1])" :x2="cieXScale(locusPoints[locusPoints.length-1][0])" :y2="cieYScale(locusPoints[locusPoints.length-1][1])" stroke="var(--vp-c-text-2)" stroke-width="1.5" stroke-dasharray="4,3" />
            <template v-for="lbl in locusLabels" :key="'ll'+lbl.wl"><circle :cx="cieXScale(lbl.x)" :cy="cieYScale(lbl.y)" r="2" fill="var(--vp-c-text-3)" /><text :x="cieXScale(lbl.x)+lbl.dx" :y="cieYScale(lbl.y)+lbl.dy" class="locus-label">{{ lbl.wl }}</text></template>
            <polygon :points="srgbTrianglePoints" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="6,3" opacity="0.6" />
            <text :x="cieXScale(0.64)+4" :y="cieYScale(0.33)+3" class="srgb-label">sRGB</text>
            <polygon :points="gamutTrianglePoints" fill="var(--vp-c-brand-1)" fill-opacity="0.12" stroke="var(--vp-c-brand-1)" stroke-width="2" />
            <template v-for="f in filters" :key="'pt-'+f.id">
              <circle :cx="cieXScale(filterChroma[f.id].x)" :cy="cieYScale(filterChroma[f.id].y)" r="5" :fill="f.color" stroke="#fff" stroke-width="1.5" />
              <text :x="cieXScale(filterChroma[f.id].x)+(f.id==='r'?8:f.id==='b'?-8:0)" :y="cieYScale(filterChroma[f.id].y)+(f.id==='g'?-8:12)" :text-anchor="f.id==='b'?'end':'start'" class="point-label" :fill="f.color">{{ t(f.nameEn,f.nameKo) }}</text>
            </template>
            <line :x1="cieXScale(0.3127)-5" :y1="cieYScale(0.329)" :x2="cieXScale(0.3127)+5" :y2="cieYScale(0.329)" stroke="#555" stroke-width="1.5" />
            <line :x1="cieXScale(0.3127)" :y1="cieYScale(0.329)-5" :x2="cieXScale(0.3127)" :y2="cieYScale(0.329)+5" stroke="#555" stroke-width="1.5" />
            <text :x="cieXScale(0.3127)+7" :y="cieYScale(0.329)-4" class="d65-label">D65</text>
          </svg>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, type Ref } from 'vue'
import { useLocale } from '../composables/useLocale'
import { CIE_WL, CIE_X, CIE_Y, CIE_Z, spectrumToXYZ, xyzToXy, getN, MATERIALS, tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'
import type { TmmLayer } from '../composables/tmm'
import { useFullscreen } from '../composables/useFullscreen'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

// ---- State ----
type ModelType = 'gaussian' | 'lorentzian' | 'realDye'
const filterModel = ref<ModelType>('gaussian')
const irCutEnabled = ref(false)
const irCutoff = ref(680)
const craAngle = ref(0)
const cfThickness = ref(0.6)
const showQE = ref(true)

// ---- Filter definitions ----
interface FilterDef {
  id: string; nameEn: string; nameKo: string; color: string
  center: Ref<number>; centerMin: number; centerMax: number
  fwhm: Ref<number>; peak: Ref<number>
}
const filters: FilterDef[] = [
  { id:'r', nameEn:'Red', nameKo:'빨강', color:'#e74c3c', center:ref(620), centerMin:580, centerMax:660, fwhm:ref(60), peak:ref(90) },
  { id:'g', nameEn:'Green', nameKo:'초록', color:'#27ae60', center:ref(530), centerMin:500, centerMax:560, fwhm:ref(50), peak:ref(90) },
  { id:'b', nameEn:'Blue', nameKo:'파랑', color:'#3498db', center:ref(450), centerMin:420, centerMax:480, fwhm:ref(50), peak:ref(90) },
]

// ---- Filter models ----
const LN2 = Math.log(2)
function gaussian(wl: number, c: number, fw: number, pk: number): number {
  return (pk/100)*Math.exp(-4*LN2*((wl-c)/fw)**2)
}
function lorentzian(wl: number, c: number, fw: number, pk: number): number {
  const g = fw/2; return (pk/100)*g*g/((wl-c)**2+g*g)
}
function realDyeT(ch: string, wl_nm: number, d: number): number {
  const mat = ch==='r' ? MATERIALS.cf_red : ch==='g' ? MATERIALS.cf_green : MATERIALS.cf_blue
  const k = getN(mat, wl_nm/1000)[1]
  return Math.exp(-4*Math.PI*k*d/(wl_nm/1000))
}
function irCutT(wl_nm: number): number {
  return irCutEnabled.value ? 1/(1+Math.exp((wl_nm-irCutoff.value)/10)) : 1
}
function effThick(d: number, angDeg: number): number {
  if (angDeg === 0) return d
  const s = Math.sin(angDeg*Math.PI/180)/1.55
  return d/Math.sqrt(1-s*s)
}
function getT(f: FilterDef, wl_nm: number): number {
  const d = effThick(cfThickness.value, craAngle.value)
  let T: number
  if (filterModel.value === 'realDye') {
    T = realDyeT(f.id, wl_nm, d)
  } else {
    const fn = filterModel.value === 'lorentzian' ? lorentzian : gaussian
    const Tref = fn(wl_nm, f.center.value, f.fwhm.value, f.peak.value)
    T = Tref <= 0 ? 0 : Math.pow(Tref, d/0.6)
  }
  return T * irCutT(wl_nm)
}

// ---- TMM QE ----
const baseQE = computed(() => {
  const stack: TmmLayer[] = [
    {material:'polymer',thickness:0.6},{material:'sio2',thickness:0.3},
    {material:'si3n4',thickness:0.030},{material:'sio2',thickness:0.015},
    {material:'hfo2',thickness:0.025},{material:'sio2',thickness:0.01},
    {material:'silicon',thickness:3.0},
  ]
  return CIE_WL.map(wl => tmmCalc(stack,'air','sio2',wl,craAngle.value).layerA[6])
})

const channelQE = computed(() => {
  const r: Record<string,number[]> = {}
  if (filterModel.value === 'realDye') {
    for (const f of filters) {
      const ch = (f.id==='r'?'red':f.id==='g'?'green':'blue') as 'red'|'green'|'blue'
      const stack = defaultBsiStack(ch, 3.0)
      stack[2] = {...stack[2], thickness:cfThickness.value}
      r[f.id] = CIE_WL.map(wl => tmmCalc(stack,'air','sio2',wl,craAngle.value).layerA[SI_LAYER_IDX] * irCutT(wl*1000))
    }
  } else {
    const bqe = baseQE.value
    for (const f of filters) r[f.id] = CIE_WL.map((wl,i) => getT(f, wl*1000)*bqe[i])
  }
  return r
})

const peakQE = computed(() => {
  const r: Record<string,number> = {}
  for (const f of filters) r[f.id] = Math.max(...channelQE.value[f.id])
  return r
})

// ---- Crosstalk matrix ----
const crosstalkMatrix = computed(() => {
  const bands = [[4,24],[24,44],[44,64]] // B:400-500, G:500-600, R:600-700
  return filters.map(f => {
    const qe = channelQE.value[f.id]
    let total = 0; const bs = [0,0,0]
    for (let i=0;i<qe.length;i++) { total+=qe[i]; for (let b=0;b<3;b++) if (i>=bands[b][0]&&i<bands[b][1]) bs[b]+=qe[i] }
    return total>0 ? bs.map(s=>s/total) : [0,0,0]
  })
})
const avgCrosstalk = computed(() => {
  const M = crosstalkMatrix.value
  return ((M[0][0]+M[0][1])+(M[1][0]+M[1][2])+(M[2][1]+M[2][2]))/3
})

// ---- 3x3 matrix math ----
function inv3(m: number[][]): number[][] {
  const [[a,b,c],[d,e,f],[g,h,k]] = m
  const det = a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g)
  if (Math.abs(det)<1e-30) return [[1,0,0],[0,1,0],[0,0,1]]
  const id = 1/det
  return [[(e*k-f*h)*id,(c*h-b*k)*id,(b*f-c*e)*id],[(f*g-d*k)*id,(a*k-c*g)*id,(c*d-a*f)*id],[(d*h-e*g)*id,(b*g-a*h)*id,(a*e-b*d)*id]]
}

// ---- Vora-Value ----
const voraVal = computed(() => {
  const n = CIE_WL.length
  const C = filters.map(f => channelQE.value[f.id])
  const H = [CIE_X.slice(0,n), CIE_Y.slice(0,n), CIE_Z.slice(0,n)]
  function ortho(M: number[][]): number[][] {
    const Q = M.map(v => [...v])
    for (let col=0;col<3;col++) {
      for (let p=0;p<col;p++) {
        let dot=0,n2=0; for (let i=0;i<n;i++){dot+=Q[col][i]*Q[p][i];n2+=Q[p][i]*Q[p][i]}
        if (n2>0){const s=dot/n2;for (let i=0;i<n;i++) Q[col][i]-=s*Q[p][i]}
      }
      let nm=0;for (let i=0;i<n;i++) nm+=Q[col][i]*Q[col][i];nm=Math.sqrt(nm)
      if (nm>0) for (let i=0;i<n;i++) Q[col][i]/=nm
    }
    return Q
  }
  const QC=ortho(C), QH=ortho(H)
  let sum=0
  for (let i=0;i<3;i++) for (let j=0;j<3;j++){let d=0;for (let k=0;k<n;k++) d+=QH[i][k]*QC[j][k];sum+=d*d}
  return Math.min(1, sum/3)
})

// ---- CCM condition number ----
const ccmCond = computed(() => {
  const n = CIE_WL.length
  const C = filters.map(f => channelQE.value[f.id])
  const H = [CIE_X.slice(0,n), CIE_Y.slice(0,n), CIE_Z.slice(0,n)]
  const CtC = [[0,0,0],[0,0,0],[0,0,0]]
  const CtH = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i=0;i<3;i++) for (let j=0;j<3;j++) for (let k=0;k<n;k++){CtC[i][j]+=C[i][k]*C[j][k];CtH[i][j]+=C[i][k]*H[j][k]}
  const M = [[0,0,0],[0,0,0],[0,0,0]]
  const ci = inv3(CtC)
  for (let i=0;i<3;i++) for (let j=0;j<3;j++) for (let k=0;k<3;k++) M[i][j]+=ci[i][k]*CtH[k][j]
  const Mi = inv3(M)
  let nM=0,nMi=0
  for (let i=0;i<3;i++) for (let j=0;j<3;j++){nM+=M[i][j]**2;nMi+=Mi[i][j]**2}
  return Math.sqrt(nM)*Math.sqrt(nMi)
})

// ---- WB coefficients (Planck illuminant) ----
function planckRel(wl_um: number, T: number): number {
  return 1/(wl_um**5*(Math.exp(14388/(wl_um*T))-1))
}
const wbIlluminants = [
  {key:'D65',label:'D65',temp:6500},{key:'A',label:'A',temp:2856},
  {key:'F',label:'4100K',temp:4100},{key:'D50',label:'D50',temp:5000},
]
const wbCoeffs = computed(() => wbIlluminants.map(ill => {
  const sigs = filters.map(f => {
    let sum=0; for (let i=0;i<CIE_WL.length;i++) sum+=planckRel(CIE_WL[i],ill.temp)*channelQE.value[f.id][i]
    return sum
  })
  const gSig = sigs[1]||1
  return {...ill, gains:sigs.map(s=>s>0?gSig/s:0)}
}))

// ---- Spectra chart ----
const specW = 600, specH = 280
const specPad = { top:16, right:16, bottom:36, left:46 }
const specPlotW = specW-specPad.left-specPad.right
const specPlotH = specH-specPad.top-specPad.bottom
const specXTicks = [400,450,500,550,600,650,700,750]
const specYTicks = [0,25,50,75,100]

function specXScale(wl: number) { return specPad.left+((wl-380)/400)*specPlotW }
function specYScale(pct: number) { return specPad.top+specPlotH-(pct/100)*specPlotH }

function filterCurvePath(f: FilterDef): string {
  let d = ''
  for (let wl=380;wl<=780;wl+=2){const v=getT(f,wl)*100,x=specXScale(wl),y=specYScale(v);d+=d===''?`M${x.toFixed(1)},${y.toFixed(1)}`:` L${x.toFixed(1)},${y.toFixed(1)}`}
  return d
}
function filterAreaPath(f: FilterDef): string {
  return filterCurvePath(f)+` L${specXScale(780).toFixed(1)},${specYScale(0).toFixed(1)} L${specXScale(380).toFixed(1)},${specYScale(0).toFixed(1)} Z`
}
function qeCurvePath(fId: string): string {
  const qe=channelQE.value[fId]; let d=''
  for (let i=0;i<CIE_WL.length;i++){const wl=CIE_WL[i]*1000,x=specXScale(wl),y=specYScale(qe[i]*100);d+=d===''?`M${x.toFixed(1)},${y.toFixed(1)}`:` L${x.toFixed(1)},${y.toFixed(1)}`}
  return d
}
const irCutPath = computed(() => {
  if (!irCutEnabled.value) return ''
  let d=''; for (let wl=380;wl<=780;wl+=2){const x=specXScale(wl),y=specYScale(irCutT(wl)*100);d+=d===''?`M${x.toFixed(1)},${y.toFixed(1)}`:` L${x.toFixed(1)},${y.toFixed(1)}`}
  return d
})

function wavelengthToCSS(wl: number): string {
  let r=0,g=0,b=0
  if (wl>=380&&wl<440){r=-(wl-440)/60;b=1} else if (wl>=440&&wl<490){g=(wl-440)/50;b=1} else if (wl>=490&&wl<510){g=1;b=-(wl-510)/20}
  else if (wl>=510&&wl<580){r=(wl-510)/70;g=1} else if (wl>=580&&wl<645){r=1;g=-(wl-645)/65} else if (wl>=645&&wl<=780){r=1}
  let f=1.0; if (wl>=380&&wl<420) f=0.3+0.7*(wl-380)/40; else if (wl>=700&&wl<=780) f=0.3+0.7*(780-wl)/80
  return `rgb(${Math.round(255*Math.pow(r*f,0.8))},${Math.round(255*Math.pow(g*f,0.8))},${Math.round(255*Math.pow(b*f,0.8))})`
}
const spectrumStops = computed(() => {
  const s:{offset:string;color:string}[]=[]; for (let wl=380;wl<=780;wl+=20) s.push({offset:((wl-380)/400*100)+'%',color:wavelengthToCSS(wl)}); return s
})

const specHover = ref<{sx:number;tx:number;wl:number;r:number;g:number;b:number;qr:number;qg:number;qb:number}|null>(null)
function onSpecMouseMove(e: MouseEvent) {
  const svg=e.currentTarget as SVGSVGElement, rect=svg.getBoundingClientRect()
  const wl=380+((e.clientX-rect.left)*specW/rect.width-specPad.left)/specPlotW*400
  if (wl>=380&&wl<=780) {
    const sn=Math.round(wl), rv=getT(filters[0],sn)*100,gv=getT(filters[1],sn)*100,bv=getT(filters[2],sn)*100
    const idx=Math.min(80,Math.max(0,Math.round((sn-380)/5))), qe=channelQE.value
    const sx=specXScale(sn)
    specHover.value={sx,tx:sx+145>specW-specPad.right?sx-145:sx+10,wl:sn,r:rv,g:gv,b:bv,qr:qe.r[idx]*100,qg:qe.g[idx]*100,qb:qe.b[idx]*100}
  } else specHover.value=null
}

// ---- CIE Chromaticity ----
const cieW=400,cieH=400
const ciePad={top:16,right:16,bottom:32,left:36}
const ciePlotW=cieW-ciePad.left-ciePad.right, ciePlotH=cieH-ciePad.top-ciePad.bottom
function cieXScale(x: number) { return ciePad.left+(x/0.8)*ciePlotW }
function cieYScale(y: number) { return ciePad.top+ciePlotH-(y/0.9)*ciePlotH }

const locusPoints = computed(() => {
  const pts:[number,number][]=[]; for (let i=0;i<CIE_WL.length;i++){const s=CIE_X[i]+CIE_Y[i]+CIE_Z[i];if(s>0.001) pts.push([CIE_X[i]/s,CIE_Y[i]/s])}; return pts
})
const locusPath = computed(() => locusPoints.value.map((p,i)=>`${i===0?'M':'L'}${cieXScale(p[0]).toFixed(1)},${cieYScale(p[1]).toFixed(1)}`).join(' '))
const locusLabels = computed(() => {
  return [460,480,500,520,540,560,580,600,620,650,700].map(wl=>{
    const idx=Math.round((wl-380)/5); if(idx<0||idx>=CIE_WL.length) return null
    const s=CIE_X[idx]+CIE_Y[idx]+CIE_Z[idx]; if(s<0.001) return null
    const x=CIE_X[idx]/s,y=CIE_Y[idx]/s
    let dx=0,dy=0; if(wl<=490){dx=-10;dy=4}else if(wl<=520){dx=-8;dy=-6}else if(wl<=560){dx=0;dy=-10}else if(wl<=600){dx=6;dy=-6}else{dx=8;dy=4}
    return {wl,x,y,dx,dy}
  }).filter(Boolean) as any[]
})

const sRGBPts:[[number,number],[number,number],[number,number]] = [[0.64,0.33],[0.30,0.60],[0.15,0.06]]
const srgbTrianglePoints = computed(()=>sRGBPts.map(p=>`${cieXScale(p[0]).toFixed(1)},${cieYScale(p[1]).toFixed(1)}`).join(' '))

const filterChroma = computed(() => {
  const result: Record<string,{x:number;y:number;domWl:number;purity:number}> = {}
  for (const f of filters) {
    const spectrum = CIE_WL.map(wl=>getT(f, wl*1000))
    const [X,Y,Z] = spectrumToXYZ(spectrum)
    const [x,y] = xyzToXy(X,Y,Z)
    const d65x=0.3127,d65y=0.3290
    let domWl=f.center.value, bestDist=Infinity
    for (let i=0;i<CIE_WL.length;i++){
      const s=CIE_X[i]+CIE_Y[i]+CIE_Z[i]; if(s<0.001) continue
      const lx=CIE_X[i]/s,ly=CIE_Y[i]/s,dx1=x-d65x,dy1=y-d65y,dx2=lx-d65x,dy2=ly-d65y
      if(dx1*dx2+dy1*dy2>0){const cr=Math.abs(dx1*dy2-dy1*dx2),len=Math.sqrt(dx2*dx2+dy2*dy2);if(len>0&&cr/len<bestDist){bestDist=cr/len;domWl=Math.round(CIE_WL[i]*1000)}}
    }
    const distF=Math.sqrt((x-d65x)**2+(y-d65y)**2)
    const domIdx=Math.round((domWl/1000-0.380)/0.005)
    let purity=0
    if(domIdx>=0&&domIdx<CIE_WL.length){const s=CIE_X[domIdx]+CIE_Y[domIdx]+CIE_Z[domIdx];if(s>0.001){const lx=CIE_X[domIdx]/s,ly=CIE_Y[domIdx]/s;purity=Math.min(1,distF/Math.sqrt((lx-d65x)**2+(ly-d65y)**2))}}
    result[f.id]={x,y,domWl,purity}
  }
  return result
})

const gamutTrianglePoints = computed(()=>filters.map(f=>{const ch=filterChroma.value[f.id];return `${cieXScale(ch.x).toFixed(1)},${cieYScale(ch.y).toFixed(1)}`}).join(' '))

function triArea(pts:[number,number][]): number { const [a,b,c]=pts; return 0.5*Math.abs((b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])) }
const sRGBArea = triArea(sRGBPts)
const NTSCPts:[[number,number],[number,number],[number,number]] = [[0.67,0.33],[0.21,0.71],[0.14,0.08]]
const NTSCArea = triArea(NTSCPts)

const gamutAreaPct = computed(() => {
  const pts:[number,number][] = filters.map(f=>[filterChroma.value[f.id].x,filterChroma.value[f.id].y])
  return { srgb:triArea(pts)/sRGBArea*100, ntsc:triArea(pts)/NTSCArea*100 }
})

// ---- Export ----
function exportConfig() {
  const cfg = {model:filterModel.value,thickness:cfThickness.value,irCut:irCutEnabled.value?irCutoff.value:null,craAngle:craAngle.value,
    filters:filters.map(f=>({id:f.id,center:f.center.value,fwhm:f.fwhm.value,peak:f.peak.value})),
    metrics:{gamutSRGB:+gamutAreaPct.value.srgb.toFixed(1),gamutNTSC:+gamutAreaPct.value.ntsc.toFixed(1),voraValue:+voraVal.value.toFixed(3),ccmCond:+ccmCond.value.toFixed(1),avgCrosstalk:+(avgCrosstalk.value*100).toFixed(1),peakQE:{r:+(peakQE.value.r*100).toFixed(1),g:+(peakQE.value.g*100).toFixed(1),b:+(peakQE.value.b*100).toFixed(1)}}}
  const blob=new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'})
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='color-filter-design.json';a.click();URL.revokeObjectURL(a.href)
}
</script>

<style scoped>
.cf-container { border:1px solid var(--vp-c-divider); border-radius:12px; padding:24px; margin:24px 0; background:var(--vp-c-bg-soft); position:relative; }
.cf-container h4 { margin:0 0 4px 0; font-size:1.1em; color:var(--vp-c-brand-1); }
.cf-container h5 { margin:0 0 8px 0; font-size:0.95em; color:var(--vp-c-text-1); }
.component-description { margin:0 0 16px 0; color:var(--vp-c-text-2); font-size:0.9em; }
.cf-header { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:16px; }
.cf-header > div { flex:1; }
.cf-header h4, .cf-header .component-description { margin-bottom:4px; }
.fs-btn { width:36px; height:36px; border:1px solid var(--vp-c-divider); border-radius:8px; background:var(--vp-c-bg); cursor:pointer; font-size:1.2em; color:var(--vp-c-text-2); display:flex; align-items:center; justify-content:center; flex-shrink:0; transition:all 0.15s; }
.fs-btn:hover { border-color:var(--vp-c-brand-1); color:var(--vp-c-brand-1); background:var(--vp-c-bg-soft); }
/* Normal mode: wrappers are transparent */
.cf-body { display:contents; }
.cf-sidebar { display:contents; }
/* Fullscreen: 3-column dashboard */
.cf-fullscreen { position:fixed; inset:0; z-index:9999; overflow:hidden; background:var(--vp-c-bg); padding:12px 16px; margin:0; border:none; border-radius:0; display:flex; flex-direction:column; }
.cf-fullscreen .cf-header { flex-shrink:0; margin-bottom:8px; align-items:center; }
.cf-fullscreen .cf-header > div { flex:0 1 auto; }
.cf-fullscreen .cf-header h4 { margin:0; font-size:1em; }
.cf-fullscreen .fs-btn { width:36px; height:36px; font-size:1.4em; }
.cf-fullscreen .top-controls { flex-shrink:0; margin-bottom:8px; gap:8px; }
.cf-fullscreen .top-controls .ctrl-group { padding:6px 10px; min-width:auto; }
.cf-fullscreen .cf-body { display:grid; grid-template-columns:280px 1fr 1fr; gap:12px; flex:1; min-height:0; }
.cf-fullscreen .cf-sidebar { display:flex; flex-direction:column; gap:8px; overflow-y:auto; min-height:0; }
.cf-fullscreen .chart-spectrum { display:flex; flex-direction:column; min-height:0; margin:0; }
.cf-fullscreen .chart-spectrum h5 { flex-shrink:0; margin:0 0 4px 0; font-size:0.85em; }
.cf-fullscreen .chart-spectrum .svg-wrapper { flex:1; min-height:0; display:flex; align-items:flex-start; }
.cf-fullscreen .chart-spectrum .spec-svg { width:100%; height:100%; max-width:none; }
.cf-fullscreen .chart-cie { display:flex; flex-direction:column; min-height:0; margin:0; }
.cf-fullscreen .chart-cie h5 { flex-shrink:0; margin:0 0 4px 0; font-size:0.85em; }
.cf-fullscreen .chart-cie .svg-wrapper { flex:1; min-height:0; display:flex; justify-content:center; align-items:flex-start; }
.cf-fullscreen .chart-cie .cie-svg { width:100%; height:100%; max-width:none; }
.cf-fullscreen .filter-controls { display:flex; flex-direction:column; gap:6px; margin:0; }
.cf-fullscreen .filter-group { padding:8px; }
.cf-fullscreen .filter-header { margin-bottom:4px; font-size:0.82em; }
.cf-fullscreen .filter-sliders { gap:4px; }
.cf-fullscreen .filter-sliders .slider-row label { font-size:0.75em; margin-bottom:0; }
.cf-fullscreen .results-grid { grid-template-columns:repeat(2,1fr); gap:6px; margin:0; }
.cf-fullscreen .result-card { padding:6px 4px; }
.cf-fullscreen .result-label { font-size:0.7em; margin-bottom:1px; }
.cf-fullscreen .result-value { font-size:0.82em; }
.cf-fullscreen .result-sub { font-size:0.65em; }
.cf-fullscreen .analysis-row { flex-direction:column; gap:6px; margin:0; }
.cf-fullscreen .analysis-box { min-width:auto; padding:8px; }
.cf-fullscreen .analysis-box h5 { font-size:0.78em; margin:0 0 4px 0; }
.cf-fullscreen .xtalk-table, .cf-fullscreen .wb-table { font-size:0.75em; }
.cf-fullscreen .xtalk-table th, .cf-fullscreen .wb-table th { padding:2px 4px; font-size:0.7em; }
.cf-fullscreen .xtalk-table td, .cf-fullscreen .wb-table td { padding:2px 4px; }
.cf-fullscreen .export-row { margin:0; text-align:center; }
.top-controls { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:16px; align-items:flex-start; }
.ctrl-group { background:var(--vp-c-bg); border:1px solid var(--vp-c-divider); border-radius:8px; padding:10px 12px; min-width:140px; flex:1; }
.ctrl-label { font-size:0.8em; color:var(--vp-c-text-2); margin-bottom:6px; display:block; }
.btn-row { display:flex; gap:4px; flex-wrap:wrap; }
.toggle-btn { padding:4px 10px; border:1px solid var(--vp-c-divider); border-radius:6px; background:var(--vp-c-bg); cursor:pointer; font-size:0.78em; color:var(--vp-c-text-2); transition:all 0.15s; }
.toggle-btn.active { background:var(--vp-c-brand-1); color:#fff; border-color:var(--vp-c-brand-1); }
.toggle-btn:hover:not(.active) { border-color:var(--vp-c-brand-1); color:var(--vp-c-brand-1); }
.cb-label { font-size:0.8em; color:var(--vp-c-text-2); display:flex; align-items:center; gap:4px; }
.cb-label input { margin:0; }
.qe-toggle { font-size:0.8em; color:var(--vp-c-text-3); font-weight:400; margin-left:12px; cursor:pointer; }
.qe-toggle input { margin:0 3px 0 0; vertical-align:middle; }
.filter-controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:16px; }
.filter-group { background:var(--vp-c-bg); border:1px solid var(--vp-c-divider); border-left:4px solid; border-radius:8px; padding:12px; }
.filter-header { display:flex; align-items:center; gap:6px; margin-bottom:10px; font-size:0.9em; }
.filter-dot { width:10px; height:10px; border-radius:50%; }
.filter-sliders { display:flex; flex-direction:column; gap:8px; }
.slider-row label { display:block; font-size:0.8em; margin-bottom:2px; color:var(--vp-c-text-2); }
.ctrl-range { width:100%; -webkit-appearance:none; appearance:none; height:5px; border-radius:3px; background:var(--vp-c-divider); outline:none; }
.ctrl-range::-webkit-slider-thumb { -webkit-appearance:none; appearance:none; width:16px; height:16px; border-radius:50%; background:var(--vp-c-brand-1); cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.2); }
.ctrl-range::-moz-range-thumb { width:16px; height:16px; border-radius:50%; background:var(--vp-c-brand-1); cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.2); }
.results-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(115px,1fr)); gap:10px; margin-bottom:16px; }
.result-card { background:var(--vp-c-bg); border:1px solid var(--vp-c-divider); border-radius:8px; padding:10px; text-align:center; }
.result-card.gamut-card { border-top:3px solid var(--vp-c-brand-1); }
.result-label { font-size:0.78em; color:var(--vp-c-text-2); margin-bottom:3px; }
.result-value { font-weight:600; font-size:0.95em; font-family:var(--vp-font-family-mono); }
.result-value.highlight { color:var(--vp-c-brand-1); }
.result-sub { font-size:0.72em; color:var(--vp-c-text-3); margin-top:2px; font-family:var(--vp-font-family-mono); }
.chart-section { margin-bottom:20px; }
.svg-wrapper { margin-top:4px; }
.spec-svg { width:100%; max-width:600px; display:block; margin:0 auto; }
.cie-wrapper { display:flex; justify-content:center; }
.cie-svg { width:100%; max-width:400px; display:block; }
.tick-label { font-size:9px; fill:var(--vp-c-text-3); }
.axis-title { font-size:10px; fill:var(--vp-c-text-2); font-weight:600; }
.tooltip-text { font-size:9px; fill:var(--vp-c-text-1); font-family:var(--vp-font-family-mono); }
.locus-label { font-size:7px; fill:var(--vp-c-text-3); }
.srgb-label { font-size:8px; fill:#888; font-style:italic; }
.point-label { font-size:9px; font-weight:600; }
.d65-label { font-size:8px; fill:#555; font-weight:600; }
.analysis-row { display:flex; gap:16px; margin-bottom:20px; flex-wrap:wrap; }
.analysis-box { flex:1; min-width:220px; background:var(--vp-c-bg); border:1px solid var(--vp-c-divider); border-radius:8px; padding:12px; }
.analysis-box h5 { margin:0 0 8px 0; font-size:0.85em; }
.xtalk-table, .wb-table { width:100%; border-collapse:collapse; font-size:0.8em; font-family:var(--vp-font-family-mono); }
.xtalk-table th, .wb-table th { font-size:0.75em; color:var(--vp-c-text-3); font-weight:500; padding:4px 8px; text-align:center; border-bottom:1px solid var(--vp-c-divider); }
.xtalk-table td, .wb-table td { padding:4px 8px; text-align:center; }
.xt-diag { font-weight:700; color:var(--vp-c-brand-1); }
.wb-illum { font-weight:600; color:var(--vp-c-text-2); text-align:left !important; }
.export-row { text-align:right; margin-top:8px; }
.export-btn { padding:6px 16px; border:1px solid var(--vp-c-brand-1); border-radius:6px; background:var(--vp-c-bg); color:var(--vp-c-brand-1); cursor:pointer; font-size:0.82em; transition:all 0.15s; }
.export-btn:hover { background:var(--vp-c-brand-1); color:#fff; }
</style>
