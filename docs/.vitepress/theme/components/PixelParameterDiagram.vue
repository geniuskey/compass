<template>
  <div class="param-diagram">
    <div class="tab-row">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        class="tab-btn"
        :class="{ active: activeTab === tab.key }"
        :aria-pressed="activeTab === tab.key"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <p class="hint">
      {{ t(
        'Hover a parameter row in the legend below to highlight it on the diagram. Dimensions follow the default 1.0 µm BSI pixel (configs/pixel/default_bsi_1um.yaml), including per-channel color-filter relief and metal-grid thickness.',
        '아래 범례의 파라미터 행에 마우스를 올리면 다이어그램에서 해당 위치가 강조됩니다. 치수는 기본 1.0 µm BSI 픽셀(configs/pixel/default_bsi_1um.yaml) 기준이며, 색별 color-filter relief와 metal-grid thickness를 포함합니다.'
      ) }}
    </p>

    <div v-if="activeTab === 'xy'" class="xy-toggles">
      <span class="xy-toggles-label">{{ t('Layers:', '레이어:') }}</span>
      <label
        v-for="opt in xyToggleOptions"
        :key="opt.key"
        class="xy-toggle"
      >
        <input type="checkbox" v-model="xyVisible[opt.key]" />
        <span class="xy-toggle-swatch" :style="{ background: opt.color }"></span>
        {{ t(opt.labelEn, opt.labelKo) }}
      </label>
      <button
        type="button"
        class="xy-toggle-reset"
        @click="resetXyVisibility"
      >{{ t('Reset', '초기화') }}</button>
    </div>

    <!-- ==================== XZ Cross-Section ==================== -->
    <svg
      v-if="activeTab === 'xz'"
      :viewBox="`0 0 ${xzW} ${xzH}`"
      class="diagram-svg"
      role="img"
      :aria-label="t('XZ cross-section with parameter annotations', 'XZ 단면 파라미터 주석')"
      font-family="Arial, sans-serif"
      font-size="11"
    >
      <!-- Background media fills (bottom-to-top). Patterned layers use the surrounding medium here;
           the actual microlens, CFA relief, metal grid, DTI, and PD shapes are drawn below. -->
      <rect
        v-for="layer in layers"
        :key="'L-' + layer.id"
        :x="pad.left"
        :y="zToY(layer.zTop)"
        :width="plotW"
        :height="zToY(layer.zBot) - zToY(layer.zTop)"
        :fill="layer.fill"
        :opacity="dimLayer(layer.id) ? 0.25 : 0.85"
      />

      <!-- Patterned-layer z extents: outline only, so empty regions are not mistaken for material. -->
      <rect
        v-for="extent in patternedLayerExtents"
        :key="'E-' + extent.id"
        :x="pad.left"
        :y="zToY(extent.zTop)"
        :width="plotW"
        :height="zToY(extent.zBot) - zToY(extent.zTop)"
        fill="none"
        :stroke="highlight && extent.params.includes(highlight) ? '#e74c3c' : extent.stroke"
        :stroke-width="highlight && extent.params.includes(highlight) ? 1.4 : 0.8"
        stroke-dasharray="5 4"
        opacity="0.85"
      />

      <!-- BARL sublayers -->
      <rect
        v-for="(sub, i) in barlSublayers"
        :key="'B-' + i"
        :x="pad.left"
        :y="zToY(sub.zTop)"
        :width="plotW"
        :height="Math.max(zToY(sub.zBot) - zToY(sub.zTop), 1.5)"
        :fill="sub.color"
        :opacity="dimLayer('barl') ? 0.3 : 0.9"
        stroke="#666"
        stroke-width="0.3"
      />

      <!-- DTI walls -->
      <rect
        v-for="dx in dtiX"
        :key="'D-' + dx"
        :x="xToSvg(dx) - dtiHalfWPx"
        :y="zToY(siTop)"
        :width="dtiHalfWPx * 2"
        :height="zToY(siBot + (siTop - dtiDepth)) - zToY(siTop)"
        fill="#aed6f1"
        :opacity="highlight === 'dti' ? 1 : 0.8"
        :stroke="highlight === 'dti' ? '#e74c3c' : '#7fb3d3'"
        :stroke-width="highlight === 'dti' ? 1.5 : 0.5"
      />

      <!-- CF Bayer relief: channel top can rise above the metal grid and taper by contact angle. -->
      <path
        v-for="cf in cfProfiles"
        :key="'CF-' + cf.id"
        :d="cfProfilePath(cf)"
        :fill="cf.color"
        :opacity="dimLayer('color_filter') ? 0.25 : 0.72"
        :stroke="highlight === 'cf_t' || highlight === 'cf_angle' ? '#e74c3c' : '#1f6f45'"
        :stroke-width="highlight === 'cf_t' || highlight === 'cf_angle' ? 1.4 : 0.5"
      />

      <!-- Metal grid pillars at pixel boundaries -->
      <rect
        v-for="dx in dtiX"
        :key="'MG-' + dx"
        :x="xToSvg(dx) - mgHalfWPx"
        :y="zToY(cfGridZTop)"
        :width="mgHalfWPx * 2"
        :height="zToY(cfZBot) - zToY(cfGridZTop)"
        fill="#555"
        :opacity="highlight === 'grid' || highlight === 'grid_t' ? 1 : 0.85"
        :stroke="highlight === 'grid' || highlight === 'grid_t' ? '#e74c3c' : '#333'"
        :stroke-width="highlight === 'grid' || highlight === 'grid_t' ? 1.5 : 0.3"
      />

      <!-- Photodiodes (dashed) -->
      <rect
        v-for="(pd, i) in pdRectsXZ"
        :key="'PD-' + i"
        :x="xToSvg(pd.x0)"
        :y="zToY(pd.zTop)"
        :width="xToSvg(pd.x1) - xToSvg(pd.x0)"
        :height="zToY(pd.zBot) - zToY(pd.zTop)"
        :fill="highlight === 'pd' ? '#b85c5c' : '#b85c5c'"
        :opacity="highlight === 'pd' ? 0.55 : 0.25"
        :stroke="highlight === 'pd' ? '#e74c3c' : '#b85c5c'"
        :stroke-width="highlight === 'pd' ? 1.6 : 1.2"
        stroke-dasharray="4 2"
      />

      <!-- Microlens domes (left at default, right shifted to illustrate shift_x) -->
      <path
        v-for="(d, i) in mlPaths"
        :key="'ML-' + i"
        :d="d.path"
        fill="#dda0dd"
        :opacity="dimLayer('microlens') ? 0.3 : 0.7"
        :stroke="highlight === 'shift' && d.shifted ? '#e74c3c' : '#b07eb0'"
        :stroke-width="highlight === 'shift' && d.shifted ? 1.6 : 1"
      />

      <!-- Domain border -->
      <rect
        :x="pad.left"
        :y="pad.top"
        :width="plotW"
        :height="plotH"
        fill="none"
        stroke="var(--vp-c-divider)"
        stroke-width="1"
      />

      <!-- ===== Right-side dimension callouts: layer thicknesses ===== -->
      <g class="dim-group">
        <template v-for="dim in rightDims" :key="'rd-' + dim.id">
          <line
            :x1="pad.left + plotW + dim.offset"
            :y1="zToY(dim.zTop)"
            :x2="pad.left + plotW + dim.offset"
            :y2="zToY(dim.zBot)"
            :stroke="dimColor(dim.param)"
            :stroke-width="highlight === dim.param ? 2 : 1"
          />
          <line
            :x1="pad.left + plotW + dim.offset - 4"
            :y1="zToY(dim.zTop)"
            :x2="pad.left + plotW + dim.offset + 4"
            :y2="zToY(dim.zTop)"
            :stroke="dimColor(dim.param)"
            stroke-width="1"
          />
          <line
            :x1="pad.left + plotW + dim.offset - 4"
            :y1="zToY(dim.zBot)"
            :x2="pad.left + plotW + dim.offset + 4"
            :y2="zToY(dim.zBot)"
            :stroke="dimColor(dim.param)"
            stroke-width="1"
          />
          <text
            :x="pad.left + plotW + dim.offset + 8"
            :y="(zToY(dim.zTop) + zToY(dim.zBot)) / 2 + 4"
            :fill="dimColor(dim.param)"
            class="dim-text"
            :font-weight="highlight === dim.param ? '700' : '500'"
          >{{ dim.label }}</text>
        </template>
      </g>

      <!-- ===== Bottom callout: pitch ===== -->
      <g class="dim-group">
        <line
          :x1="xToSvg(0)" :y1="pad.top + plotH + 22"
          :x2="xToSvg(1)" :y2="pad.top + plotH + 22"
          :stroke="dimColor('pitch')"
          :stroke-width="highlight === 'pitch' ? 2 : 1"
          marker-start="url(#arr-l)" marker-end="url(#arr-r)"
        />
        <text
          :x="xToSvg(0.5)" :y="pad.top + plotH + 36"
          :fill="dimColor('pitch')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pitch' ? '700' : '500'"
        >pitch = 1.0 µm</text>
      </g>

      <!-- ===== Internal callouts ===== -->
      <!-- microlens.height: vertical sag arrow on left lens -->
      <g class="dim-group">
        <line
          :x1="xToSvg(0.18)" :y1="zToY(mlZBot)"
          :x2="xToSvg(0.18)" :y2="zToY(mlZBot + mlH)"
          :stroke="dimColor('ml_h')"
          :stroke-width="highlight === 'ml_h' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(0.06)" :y="zToY(mlZBot + mlH / 2) + 3"
          :fill="dimColor('ml_h')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'ml_h' ? '700' : '500'"
        >height</text>
      </g>

      <!-- microlens.radius_x: horizontal arrow across left lens base -->
      <g class="dim-group">
        <line
          :x1="xToSvg(0.5 - mlR)" :y1="zToY(mlZBot) - 3"
          :x2="xToSvg(0.5 + mlR)" :y2="zToY(mlZBot) - 3"
          :stroke="dimColor('ml_rx')"
          :stroke-width="highlight === 'ml_rx' ? 2 : 1"
          marker-start="url(#arr-l)" marker-end="url(#arr-r)"
        />
        <text
          :x="xToSvg(0.5)" :y="zToY(mlZBot) - 8"
          :fill="dimColor('ml_rx')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'ml_rx' ? '700' : '500'"
        >2 × radius_x</text>
      </g>

      <!-- microlens.gap: between lenses -->
      <g class="dim-group" v-if="mlGap > 0">
        <line
          :x1="xToSvg(0.5 + mlR)" :y1="zToY(mlZBot) - 3"
          :x2="xToSvg(1.5 - mlR)" :y2="zToY(mlZBot) - 3"
          :stroke="dimColor('ml_gap')"
          :stroke-width="highlight === 'ml_gap' ? 2 : 1"
          marker-start="url(#arr-l)" marker-end="url(#arr-r)"
        />
      </g>

      <!-- shift_x indicator on right lens (illustrative offset) -->
      <g class="dim-group">
        <line
          :x1="xToSvg(1.5)" :y1="zToY(mlZBot + mlH) - 6"
          :x2="xToSvg(1.5 + shiftXIllustrative)" :y2="zToY(mlZBot + mlH) - 6"
          :stroke="dimColor('shift')"
          :stroke-width="highlight === 'shift' ? 2 : 1"
          marker-start="url(#arr-l)" marker-end="url(#arr-r)"
        />
        <text
          :x="xToSvg(1.5 + shiftXIllustrative / 2)" :y="zToY(mlZBot + mlH) - 10"
          :fill="dimColor('shift')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'shift' ? '700' : '500'"
        >shift_x</text>
      </g>

      <!-- color_filter.grid.width: leader to metal pillar -->
      <g class="dim-group">
        <line
          :x1="xToSvg(1.0) - mgHalfWPx" :y1="zToY((cfGridZTop + cfZBot) / 2)"
          :x2="xToSvg(1.0) + mgHalfWPx" :y2="zToY((cfGridZTop + cfZBot) / 2)"
          :stroke="dimColor('grid_w')"
          :stroke-width="highlight === 'grid_w' ? 2 : 1"
        />
        <line
          :x1="xToSvg(1.0)" :y1="zToY((cfGridZTop + cfZBot) / 2)"
          :x2="xToSvg(1.4)" :y2="zToY(cfZBot) - 18"
          :stroke="dimColor('grid_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xToSvg(1.42)" :y="zToY(cfZBot) - 18"
          :fill="dimColor('grid_w')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'grid_w' ? '700' : '500'"
        >grid.width</text>
      </g>

      <!-- color_filter.grid.thickness: vertical height of the metal grid only -->
      <g class="dim-group">
        <line
          :x1="xToSvg(0.92)" :y1="zToY(cfZBot)"
          :x2="xToSvg(0.92)" :y2="zToY(cfGridZTop)"
          :stroke="dimColor('grid_t')"
          :stroke-width="highlight === 'grid_t' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(0.89)" :y="(zToY(cfZBot) + zToY(cfGridZTop)) / 2 + 3"
          :fill="dimColor('grid_t')"
          class="dim-text" text-anchor="end"
          :font-weight="highlight === 'grid_t' ? '700' : '500'"
        >grid.t</text>
      </g>

      <!-- color_filter.contact_angle: taper control above the grid top -->
      <g class="dim-group">
        <line
          :x1="xToSvg(1.86)" :y1="zToY(cfGridZTop + 0.1)"
          :x2="xToSvg(1.48)" :y2="zToY(cfZTop) - 18"
          :stroke="dimColor('cf_angle')"
          :stroke-width="highlight === 'cf_angle' ? 1.4 : 0.8"
          stroke-dasharray="2 2"
        />
        <text
          :x="xToSvg(1.46)" :y="zToY(cfZTop) - 19"
          :fill="dimColor('cf_angle')"
          class="dim-text" text-anchor="end"
          :font-weight="highlight === 'cf_angle' ? '700' : '500'"
        >contact_angle</text>
      </g>

      <!-- DTI width + depth -->
      <g class="dim-group">
        <line
          :x1="xToSvg(1.0) - dtiHalfWPx" :y1="zToY(siTop) + 14"
          :x2="xToSvg(1.0) + dtiHalfWPx" :y2="zToY(siTop) + 14"
          :stroke="dimColor('dti_w')"
          :stroke-width="highlight === 'dti_w' ? 2 : 1"
        />
        <line
          :x1="xToSvg(1.0)" :y1="zToY(siTop) + 14"
          :x2="xToSvg(1.45)" :y2="zToY(siTop) + 32"
          :stroke="dimColor('dti_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xToSvg(1.47)" :y="zToY(siTop) + 36"
          :fill="dimColor('dti_w')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'dti_w' ? '700' : '500'"
        >dti.width</text>

        <!-- DTI depth: vertical arrow alongside the trench, label below PD in silicon -->
        <line
          :x1="xToSvg(1.0) + dtiHalfWPx + 6" :y1="zToY(siTop)"
          :x2="xToSvg(1.0) + dtiHalfWPx + 6" :y2="zToY(siTop - dtiDepth)"
          :stroke="dimColor('dti_d')"
          :stroke-width="highlight === 'dti_d' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(1.0) + dtiHalfWPx + 10" :y="zToY(0.22) + 3"
          :fill="dimColor('dti_d')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'dti_d' ? '700' : '500'"
        >dti.depth</text>
      </g>

      <!-- Photodiode size dz (vertical, inside right PD) and position z (offset from Si top) -->
      <g class="dim-group">
        <!-- pd.size dz: horizontal label at PD top, arrow below it spanning the rest of PD -->
        <text
          :x="xToSvg(1.5)" :y="zToY(pdZTop) + 12"
          :fill="dimColor('pd_dz')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pd_dz' ? '700' : '500'"
        >size[dz]</text>
        <line
          :x1="xToSvg(1.5)" :y1="zToY(pdZTop) + 18"
          :x2="xToSvg(1.5)" :y2="zToY(pdZBot) - 2"
          :stroke="dimColor('pd_dz')"
          :stroke-width="highlight === 'pd_dz' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />

        <!-- pd.position z: top of Si to top of PD, label inside silicon gap above PD -->
        <line
          :x1="xToSvg(0.5)" :y1="zToY(siTop)"
          :x2="xToSvg(0.5)" :y2="zToY(pdZTop)"
          :stroke="dimColor('pd_pz')"
          :stroke-width="highlight === 'pd_pz' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(0.5) + 6" :y="(zToY(siTop) + zToY(pdZTop)) / 2 + 4"
          :fill="dimColor('pd_pz')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'pd_pz' ? '700' : '500'"
        >position[z]</text>
      </g>

      <!-- Layer name labels on left -->
      <g class="layer-labels">
        <text
          v-for="layer in layers"
          :key="'lab-' + layer.id"
          :x="pad.left + 6"
          :y="(zToY(layer.zTop) + zToY(layer.zBot)) / 2 + 4"
          class="layer-name"
        >{{ layer.label }}</text>
      </g>

      <!-- Z axis -->
      <text :x="pad.left - 8" :y="pad.top - 8" class="axis-label" text-anchor="end">z (µm)</text>
      <template v-for="z in zTicks" :key="'zt-' + z">
        <line
          :x1="pad.left - 4" :y1="zToY(z)"
          :x2="pad.left" :y2="zToY(z)"
          stroke="var(--vp-c-text-3)" stroke-width="0.8"
        />
        <text
          :x="pad.left - 7" :y="zToY(z) + 4"
          class="tick-label" text-anchor="end"
        >{{ z.toFixed(1) }}</text>
      </template>

      <!-- X axis -->
      <text
        :x="pad.left + plotW / 2" :y="xzH - 6"
        class="axis-label" text-anchor="middle"
      >x (µm) — {{ t('two adjacent pixels shown', '인접 두 픽셀 표시') }}</text>

      <!-- Title -->
      <text
        :x="pad.left + plotW / 2" :y="pad.top - 12"
        class="section-title" text-anchor="middle"
        font-size="16" font-weight="700"
      >{{ t('XZ Cross-Section (parameter map)', 'XZ 단면 (파라미터 맵)') }}</text>

      <!-- Arrow markers -->
      <defs>
        <marker id="arr-r" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L5,3 L0,6 z" fill="currentColor" />
        </marker>
        <marker id="arr-l" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto">
          <path d="M5,0 L0,3 L5,6 z" fill="currentColor" />
        </marker>
        <marker id="arr-u" markerWidth="6" markerHeight="6" refX="3" refY="1" orient="auto">
          <path d="M0,5 L3,0 L6,5 z" fill="currentColor" />
        </marker>
        <marker id="arr-d" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto">
          <path d="M0,0 L3,5 L6,0 z" fill="currentColor" />
        </marker>
      </defs>
    </svg>

    <!-- ==================== XY Top View ==================== -->
    <svg
      v-else
      :viewBox="`0 0 ${xyW} ${xyH}`"
      class="diagram-svg"
      role="img"
      :aria-label="t('XY top view with parameter annotations', 'XY 평면 파라미터 주석')"
      font-family="Arial, sans-serif"
      font-size="11"
    >
      <!-- 2x2 Bayer cells: solid area is the footprint at grid top, dashed area is the tapered top. -->
      <template v-if="xyVisible.color_filter">
        <rect
          v-for="cell in colorFilterFootprints"
          :key="'BY-' + cell.label + cell.cx + cell.cy"
          :x="xyRectX(cell.base.x0)"
          :y="xyRectY(cell.base.y1)"
          :width="xyRectW(cell.base.x0, cell.base.x1)"
          :height="xyRectH(cell.base.y0, cell.base.y1)"
          :fill="cell.fill"
          :opacity="dimLayer('color_filter') ? 0.2 : 0.62"
        />
        <rect
          v-for="cell in colorFilterFootprints"
          :key="'BYT-' + cell.label + cell.cx + cell.cy"
          :x="xyRectX(cell.top.x0)"
          :y="xyRectY(cell.top.y1)"
          :width="xyRectW(cell.top.x0, cell.top.x1)"
          :height="xyRectH(cell.top.y0, cell.top.y1)"
          :fill="cell.fill"
          fill-opacity="0.16"
          :stroke="highlight === 'cf_angle' || highlight === 'cf_t' ? '#e74c3c' : '#1f2937'"
          :stroke-width="highlight === 'cf_angle' || highlight === 'cf_t' ? 1.6 : 1.2"
          stroke-dasharray="5 4"
        />
        <text
          v-for="cell in bayerCells"
          :key="'BL-' + cell.label + cell.cx + cell.cy"
          :x="xyX(cell.cx)"
          :y="xyY(cell.cy) + 6"
          class="bayer-letter"
          text-anchor="middle"
          font-size="18"
          font-weight="700"
        >{{ cell.label }}</text>
      </template>

      <!-- Metal grid lines (CF boundaries) -->
      <g v-if="xyVisible.grid">
        <rect
          v-for="grid in metalGridRects"
          :key="'mg-' + grid.id"
          :x="xyRectX(grid.x0)"
          :y="xyRectY(grid.y1)"
          :width="xyRectW(grid.x0, grid.x1)"
          :height="xyRectH(grid.y0, grid.y1)"
          fill="#555"
          :opacity="highlight === 'grid_w' || highlight === 'grid_t' ? 1 : 0.72"
          :stroke="highlight === 'grid_w' || highlight === 'grid_t' ? '#e74c3c' : 'none'"
        />
      </g>

      <!-- DTI grid (slightly wider, dashed-blue) -->
      <g v-if="xyVisible.dti">
        <rect
          v-for="dti in dtiRects"
          :key="'dt-' + dti.id"
          :x="xyRectX(dti.x0)"
          :y="xyRectY(dti.y1)"
          :width="xyRectW(dti.x0, dti.x1)"
          :height="xyRectH(dti.y0, dti.y1)"
          fill="#aed6f1"
          :opacity="highlight === 'dti_w' ? 1 : 0.6"
          :stroke="highlight === 'dti_w' ? '#e74c3c' : '#7fb3d3'"
          :stroke-width="highlight === 'dti_w' ? 1.6 : 0.5"
        />
      </g>

      <!-- Photodiode footprints (dashed) -->
      <template v-if="xyVisible.photodiode">
        <rect
          v-for="(pd, i) in pdFootprints"
          :key="'pdxy-' + i"
          :x="xyRectX(pd.x0)"
          :y="xyRectY(pd.y1)"
          :width="xyRectW(pd.x0, pd.x1)"
          :height="xyRectH(pd.y0, pd.y1)"
          fill="#b85c5c"
          :opacity="highlight === 'pd' || highlight === 'pd_dxdy' ? 0.4 : 0.18"
          :stroke="highlight === 'pd' || highlight === 'pd_dxdy' ? '#e74c3c' : '#b85c5c'"
          :stroke-width="highlight === 'pd' || highlight === 'pd_dxdy' ? 1.6 : 1.2"
          stroke-dasharray="4 2"
        />
      </template>

      <!-- Microlens ellipses -->
      <template v-if="xyVisible.microlens">
        <ellipse
          v-for="(ml, i) in mlFootprints"
          :key="'mlxy-' + i"
          :cx="xyX(ml.cx)"
          :cy="xyY(ml.cy)"
          :rx="ml.rx * xyScale"
          :ry="ml.ry * xyScale"
          fill="#dda0dd"
          :opacity="dimLayer('microlens') ? 0.25 : 0.55"
          :stroke="highlight === 'ml_rx' || highlight === 'ml_ry' ? '#e74c3c' : '#b07eb0'"
          :stroke-width="highlight === 'ml_rx' || highlight === 'ml_ry' ? 1.6 : 1"
        />
      </template>

      <!-- pitch arrow (bottom) -->
      <g class="dim-group">
        <line
          :x1="xyX(0)" :y1="xyBottom + 18"
          :x2="xyX(1)" :y2="xyBottom + 18"
          :stroke="dimColor('pitch')"
          :stroke-width="highlight === 'pitch' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyX(0.5)" :y="xyBottom + 32"
          :fill="dimColor('pitch')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pitch' ? '700' : '500'"
        >pitch</text>
      </g>

      <!-- unit_cell arrow (bottom, full 2 µm) -->
      <g class="dim-group">
        <line
          :x1="xyX(0)" :y1="xyBottom + 42"
          :x2="xyX(2)" :y2="xyBottom + 42"
          :stroke="dimColor('unit_cell')"
          :stroke-width="highlight === 'unit_cell' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyX(1)" :y="xyBottom + 56"
          :fill="dimColor('unit_cell')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'unit_cell' ? '700' : '500'"
        >unit_cell = [2, 2] → 2·pitch</text>
      </g>

      <!-- microlens.radius_x (horizontal across one ML) -->
      <g class="dim-group" v-if="xyVisible.microlens">
        <line
          :x1="xyX(0.5 - mlR)" :y1="xyY(0.5)"
          :x2="xyX(0.5 + mlR)" :y2="xyY(0.5)"
          :stroke="dimColor('ml_rx')"
          :stroke-width="highlight === 'ml_rx' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyX(0.5)" :y="xyY(0.5) - 6"
          :fill="dimColor('ml_rx')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'ml_rx' ? '700' : '500'"
        >2·radius_x</text>
      </g>

      <!-- microlens.radius_y (vertical across one ML) -->
      <g class="dim-group" v-if="xyVisible.microlens">
        <line
          :x1="xyX(1.5)" :y1="xyY(0.5 + mlR)"
          :x2="xyX(1.5)" :y2="xyY(0.5 - mlR)"
          :stroke="dimColor('ml_ry')"
          :stroke-width="highlight === 'ml_ry' ? 2 : 1"
          marker-start="url(#xy-arr-u)" marker-end="url(#xy-arr-d)"
        />
        <text
          :x="xyX(1.5) + 6" :y="xyY(0.5) + 3"
          :fill="dimColor('ml_ry')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'ml_ry' ? '700' : '500'"
        >2·radius_y</text>
      </g>

      <!-- photodiode size dx/dy on bottom-right pixel -->
      <g class="dim-group" v-if="xyVisible.photodiode">
        <line
          :x1="xyX(1.5 - pdHalf)" :y1="xyY(0.5 - pdHalf) + 6"
          :x2="xyX(1.5 + pdHalf)" :y2="xyY(0.5 - pdHalf) + 6"
          :stroke="dimColor('pd_dxdy')"
          :stroke-width="highlight === 'pd_dxdy' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyX(1.5)" :y="xyY(0.5 - pdHalf) + 18"
          :fill="dimColor('pd_dxdy')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pd_dxdy' ? '700' : '500'"
        >photodiode.size[dx]</text>

        <line
          :x1="xyX(1.5 + pdHalf) + 6" :y1="xyY(0.5 + pdHalf)"
          :x2="xyX(1.5 + pdHalf) + 6" :y2="xyY(0.5 - pdHalf)"
          :stroke="dimColor('pd_dxdy')"
          :stroke-width="highlight === 'pd_dxdy' ? 2 : 1"
          marker-start="url(#xy-arr-u)" marker-end="url(#xy-arr-d)"
        />
        <text
          :x="xyX(1.5 + pdHalf) + 10" :y="xyY(0.5) + 3"
          :fill="dimColor('pd_dxdy')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'pd_dxdy' ? '700' : '500'"
        >size[dy]</text>
      </g>

      <!-- DTI width leader -->
      <g class="dim-group" v-if="xyVisible.dti">
        <line
          :x1="xyX(1) - dtiHalfXY" :y1="xyY(1.7)"
          :x2="xyX(1) + dtiHalfXY" :y2="xyY(1.7)"
          :stroke="dimColor('dti_w')"
          :stroke-width="highlight === 'dti_w' ? 2 : 1"
        />
        <line
          :x1="xyX(1)" :y1="xyY(1.7)"
          :x2="xyX(0.45)" :y2="xyY(1.95)"
          :stroke="dimColor('dti_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xyX(0.43)" :y="xyY(1.95) + 4"
          :fill="dimColor('dti_w')"
          class="dim-text" text-anchor="end"
          :font-weight="highlight === 'dti_w' ? '700' : '500'"
        >dti.width</text>
      </g>

      <!-- grid.width leader -->
      <g class="dim-group" v-if="xyVisible.grid">
        <line
          :x1="xyX(2) - mgHalfXY" :y1="xyY(0.3)"
          :x2="xyX(2) + mgHalfXY" :y2="xyY(0.3)"
          :stroke="dimColor('grid_w')"
          :stroke-width="highlight === 'grid_w' ? 2 : 1"
        />
        <line
          :x1="xyX(2)" :y1="xyY(0.3)"
          :x2="xyX(2.2)" :y2="xyY(0.1)"
          :stroke="dimColor('grid_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xyX(2.22)" :y="xyY(0.1) + 3"
          :fill="dimColor('grid_w')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'grid_w' ? '700' : '500'"
        >grid.width</text>
      </g>

      <!-- Domain border -->
      <rect
        :x="xyX(0)"
        :y="xyY(2)"
        :width="xyPlot"
        :height="xyPlot"
        fill="none"
        stroke="var(--vp-c-divider)"
        stroke-width="1"
      />

      <!-- Axes -->
      <text
        :x="xyX(1)" :y="xyH - 6"
        class="axis-label" text-anchor="middle"
      >x (µm)</text>
      <text
        :x="14" :y="xyY(1)"
        class="axis-label" text-anchor="middle"
        :transform="`rotate(-90, 14, ${xyY(1)})`"
      >y (µm)</text>

      <!-- Title -->
      <text
        :x="(xyX(0) + xyX(2)) / 2" :y="xyPadTop - 12"
        class="section-title" text-anchor="middle"
        font-size="16" font-weight="700"
      >{{ t('XY Top View — stack footprints (lower-left origin)', 'XY 평면도 — stack footprint (좌하단 원점)') }}</text>

      <!-- Markers -->
      <defs>
        <marker id="xy-arr-r" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L5,3 L0,6 z" fill="currentColor" />
        </marker>
        <marker id="xy-arr-l" markerWidth="6" markerHeight="6" refX="1" refY="3" orient="auto">
          <path d="M5,0 L0,3 L5,6 z" fill="currentColor" />
        </marker>
        <marker id="xy-arr-u" markerWidth="6" markerHeight="6" refX="3" refY="1" orient="auto">
          <path d="M0,5 L3,0 L6,5 z" fill="currentColor" />
        </marker>
        <marker id="xy-arr-d" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto">
          <path d="M0,0 L3,5 L6,0 z" fill="currentColor" />
        </marker>
      </defs>
    </svg>

    <!-- Parameter legend table -->
    <div class="legend-table">
      <table>
        <thead>
          <tr>
            <th></th>
            <th>{{ t('Parameter', '파라미터') }}</th>
            <th>{{ t('Default', '기본값') }}</th>
            <th>{{ t('Meaning', '의미') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in legendRows"
            :key="row.id"
            :class="{ active: highlight === row.id }"
            @mouseenter="highlight = row.id"
            @mouseleave="highlight = null"
          >
            <td>
              <span class="swatch" :style="{ background: row.color }"></span>
            </td>
            <td><code>{{ row.param }}</code></td>
            <td>{{ row.value }}</td>
            <td>{{ t(row.meaningEn, row.meaningKo) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const tabs = computed(() => [
  { key: 'xz', label: t('XZ Cross-Section', 'XZ 단면') },
  { key: 'xy', label: t('XY Top View', 'XY 평면도') },
])

const activeTab = ref<'xz' | 'xy'>('xz')
const highlight = ref<string | null>(null)

type XyLayerKey = 'microlens' | 'color_filter' | 'grid' | 'dti' | 'photodiode'
const defaultXyVisible: Record<XyLayerKey, boolean> = {
  microlens: true,
  color_filter: true,
  grid: true,
  dti: true,
  photodiode: true,
}
const xyVisible = ref<Record<XyLayerKey, boolean>>({ ...defaultXyVisible })
function resetXyVisibility() {
  xyVisible.value = { ...defaultXyVisible }
}

const xyToggleOptions: { key: XyLayerKey; labelEn: string; labelKo: string; color: string }[] = [
  { key: 'microlens',    labelEn: 'Microlens',    labelKo: '마이크로렌즈',  color: '#dda0dd' },
  { key: 'color_filter', labelEn: 'Color filter', labelKo: '컬러 필터',    color: '#27ae60' },
  { key: 'grid',         labelEn: 'Metal grid',   labelKo: '금속 격자',    color: '#555555' },
  { key: 'dti',          labelEn: 'DTI',          labelKo: 'DTI',         color: '#aed6f1' },
  { key: 'photodiode',   labelEn: 'Photodiode',   labelKo: '포토다이오드',  color: '#b85c5c' },
]

// ===== XZ geometry (matches default_bsi_1um.yaml) =====
// Axis equal: 100 svg-pixels per µm in both x and z.
const totalZ = 5.63
const totalX = 2.0  // two pixels of pitch 1 µm
const xzScale = 100
const plotW = totalX * xzScale       // 200
const plotH = totalZ * xzScale       // 558
const pad = { left: 110, right: 220, top: 30, bottom: 60 }
const xzW = pad.left + plotW + pad.right    // 530
const xzH = pad.top  + plotH + pad.bottom   // 648

const layers = [
  { id: 'silicon', label: 'silicon', fill: '#5d6d7e', zBot: 0,    zTop: 3.0 },
  { id: 'barl',    label: 'barl',    fill: '#8e44ad', zBot: 3.0,  zTop: 3.08 },
  { id: 'color_filter', label: 'color_filter', fill: '#d5dbdb', zBot: 3.08, zTop: 3.73 },
  { id: 'planarization', label: 'planarization', fill: '#d5dbdb', zBot: 3.73, zTop: 4.03 },
  { id: 'microlens', label: 'microlens', fill: '#d6eaf8', zBot: 4.03, zTop: 4.63 },
  { id: 'air',     label: 'air',     fill: '#d6eaf8', zBot: 4.63, zTop: 5.63 },
]

const patternedLayerExtents = [
  { id: 'color_filter', zBot: 3.08, zTop: 3.73, stroke: '#27ae60', params: ['cf_t', 'cf_angle', 'grid_w', 'grid_t', 'grid'] },
  { id: 'microlens', zBot: 4.03, zTop: 4.63, stroke: '#8e44ad', params: ['ml_h', 'ml_rx', 'ml_ry', 'ml_gap', 'shift'] },
]

const barlSublayers = [
  { color: '#7fb3d8', zBot: 3.0,   zTop: 3.01,  material: 'SiO2' },
  { color: '#6c71c4', zBot: 3.01,  zTop: 3.035, material: 'HfO2' },
  { color: '#e8d44d', zBot: 3.035, zTop: 3.05,  material: 'SiO2' },
  { color: '#2aa198', zBot: 3.05,  zTop: 3.08,  material: 'Si3N4' },
]

const cfZBot = 3.08
const cfZTop = 3.73
const cfGridThickness = 0.47
const cfGridZTop = cfZBot + cfGridThickness
const siBot = 0.0
const siTop = 3.0
const dtiDepth = 3.0
const dtiWidth = 0.1
const mgWidth = 0.05
const mlZBot = 4.03
const mlH = 0.6
const mlR = 0.48
const mlGap = 0.04
const shiftXIllustrative = 0.12

// Photodiode (1µm pixel default): position [0,0,0.5] from pixel center, size [0.7,0.7,2.0].
// In the parameter diagram we draw PD slightly biased upward (position[z] = 0.2 µm) so the
// rotated `size[dz]` label inside the PD does not align vertically with the silicon.thickness
// label on the right margin.
const pdSizeXY = 0.7
const pdSizeZ = 2.0
const pdPosZ = 0.2          // illustrative: PD top sits 0.2 µm below top of Si
const pdZTop = siTop - pdPosZ
const pdZBot = pdZTop - pdSizeZ

const pdRectsXZ = [
  { x0: 0.5 - pdSizeXY / 2, x1: 0.5 + pdSizeXY / 2, zTop: pdZTop, zBot: pdZBot },
  { x0: 1.5 - pdSizeXY / 2, x1: 1.5 + pdSizeXY / 2, zTop: pdZTop, zBot: pdZBot },
]

type CfProfile = {
  id: string
  x0: number
  x1: number
  topZ: number
  topInset: number
  color: string
}

const cfChannelSpecs: Record<string, { thickness: number; contactAngle: number }> = {
  R: { thickness: 0.62, contactAngle: 66 },
  G: { thickness: 0.60, contactAngle: 72 },
  B: { thickness: 0.65, contactAngle: 62 },
}

const cfProfiles: CfProfile[] = [
  makeCfProfile('G', 0.0, 1.0, cfChannelSpecs.G.thickness, cfChannelSpecs.G.contactAngle, '#27ae60'),
  makeCfProfile('B', 1.0, 2.0, cfChannelSpecs.B.thickness, cfChannelSpecs.B.contactAngle, '#3498db'),
]

function makeCfProfile(id: string, x0: number, x1: number, thickness: number, contactAngle: number, color: string): CfProfile {
  const baseInset = mgWidth / 2
  const protrusion = Math.max(0, thickness - cfGridThickness)
  const theta = (Math.PI / 180) * Math.max(1, Math.min(89.999, contactAngle))
  return {
    id,
    x0: x0 + baseInset,
    x1: x1 - baseInset,
    topZ: cfZBot + thickness,
    topInset: protrusion / Math.tan(theta),
    color,
  }
}

function cfProfilePath(cf: CfProfile) {
  const topLeft = Math.min(cf.x0 + cf.topInset, (cf.x0 + cf.x1) / 2)
  const topRight = Math.max(cf.x1 - cf.topInset, (cf.x0 + cf.x1) / 2)
  return [
    `M ${xToSvg(cf.x0)} ${zToY(cfZBot)}`,
    `L ${xToSvg(cf.x1)} ${zToY(cfZBot)}`,
    `L ${xToSvg(cf.x1)} ${zToY(cfGridZTop)}`,
    `L ${xToSvg(topRight)} ${zToY(cf.topZ)}`,
    `L ${xToSvg(topLeft)} ${zToY(cf.topZ)}`,
    `L ${xToSvg(cf.x0)} ${zToY(cfGridZTop)}`,
    'Z',
  ].join(' ')
}

// Pixel-pitch DTI lines at boundaries x = 0, 1, 2
const dtiX = [0.0, 1.0, 2.0]

// pixel-units → svg pixels (XZ uses x∈[0,2])
function xToSvg(x: number) { return pad.left + (x / 2.0) * plotW }
function zToY(z: number) { return pad.top + plotH - (z / totalZ) * plotH }
const pxPerUm = plotW / 2.0
const dtiHalfWPx = (dtiWidth / 2) * pxPerUm
const mgHalfWPx = (mgWidth / 2) * pxPerUm

const zTicks = [0, 1, 2, 3, 4, 5]

// Right-side dimension callouts (layers + sub-features)
const rightDims = [
  { id: 'air',     param: 'air_t',    zTop: 5.63, zBot: 4.63, offset: 14, label: 'air.thickness = 1.0 µm' },
  { id: 'ml',      param: 'ml_h',     zTop: 4.63, zBot: 4.03, offset: 14, label: 'microlens (height = 0.6)' },
  { id: 'plan',    param: 'plan_t',   zTop: 4.03, zBot: 3.73, offset: 14, label: 'planarization.thickness = 0.3' },
  { id: 'cf',      param: 'cf_t',     zTop: 3.73, zBot: 3.08, offset: 14, label: 'CF height = 0.60-0.65' },
  { id: 'cf-grid', param: 'grid_t',   zTop: 3.55, zBot: 3.08, offset: 54, label: 'grid.t = 0.47' },
  { id: 'barl',    param: 'barl_t',   zTop: 3.08, zBot: 3.00, offset: 14, label: 'barl Σ thickness ≈ 0.08' },
  { id: 'si',      param: 'si_t',     zTop: 3.00, zBot: 0.00, offset: 14, label: 'silicon.thickness = 3.0' },
]

// Microlens dome paths
const mlPaths = computed(() => {
  const samples = 40
  const pts = (cx: number, shifted: boolean) => {
    const dx = shifted ? shiftXIllustrative : 0
    const cxs = cx + dx
    let d = ''
    for (let i = 0; i <= samples; i++) {
      const r = -1 + (2 * i) / samples
      // superellipse approximation; here use spherical for visual clarity
      const z = mlZBot + mlH * Math.sqrt(Math.max(0, 1 - r * r))
      const x = cxs + r * mlR
      d += (i === 0 ? 'M' : 'L') + xToSvg(x) + ' ' + zToY(z) + ' '
    }
    d += 'L' + xToSvg(cxs + mlR) + ' ' + zToY(mlZBot) + ' '
    d += 'L' + xToSvg(cxs - mlR) + ' ' + zToY(mlZBot) + ' Z'
    return d
  }
  return [
    { path: pts(0.5, false), shifted: false },
    { path: pts(1.5, true),  shifted: true },
  ]
})

// ===== XY view =====
const xyW = 560
const xyH = 540
const xyPadLeft = 50
const xyPadTop = 40
const xyPlot = 380
const xyScale = xyPlot / 2.0
const xyBottom = xyPadTop + xyPlot
function xyX(v: number) { return xyPadLeft + v * xyScale }
function xyY(v: number) { return xyPadTop + (2.0 - v) * xyScale }
function xyRectX(x0: number) { return xyX(x0) }
function xyRectY(y1: number) { return xyY(y1) }
function xyRectW(x0: number, x1: number) { return (x1 - x0) * xyScale }
function xyRectH(y0: number, y1: number) { return (y1 - y0) * xyScale }

const dtiHalfXY = (dtiWidth / 2) * xyScale
const mgHalfXY = (mgWidth / 2) * xyScale
const pdHalf = pdSizeXY / 2
const pdHalfPx = pdHalf * xyScale

const bayerCells = [
  { x0: 0, x1: 1, y0: 0, y1: 1, cx: 0.5, cy: 0.5, label: 'R', fill: '#e74c3c' },
  { x0: 1, x1: 2, y0: 0, y1: 1, cx: 1.5, cy: 0.5, label: 'G', fill: '#27ae60' },
  { x0: 0, x1: 1, y0: 1, y1: 2, cx: 0.5, cy: 1.5, label: 'G', fill: '#27ae60' },
  { x0: 1, x1: 2, y0: 1, y1: 2, cx: 1.5, cy: 1.5, label: 'B', fill: '#3498db' },
]

type XyRect = { id: string; x0: number; x1: number; y0: number; y1: number }

const metalGridRects = boundaryRects(mgWidth, 'metal')
const dtiRects = boundaryRects(dtiWidth, 'dti')

const colorFilterFootprints = bayerCells.map((cell) => {
  const baseInset = mgWidth / 2
  const topInset = baseInset + cfTopInsetFor(cell.label)
  return {
    ...cell,
    base: {
      x0: cell.x0 + baseInset,
      x1: cell.x1 - baseInset,
      y0: cell.y0 + baseInset,
      y1: cell.y1 - baseInset,
    },
    top: {
      x0: cell.x0 + topInset,
      x1: cell.x1 - topInset,
      y0: cell.y0 + topInset,
      y1: cell.y1 - topInset,
    },
  }
})

function boundaryRects(width: number, prefix: string): XyRect[] {
  const half = width / 2
  const rects: XyRect[] = []
  for (const x of [0, 1, 2]) {
    rects.push({
      id: `${prefix}-v-${x}`,
      x0: Math.max(0, x - half),
      x1: Math.min(2, x + half),
      y0: 0,
      y1: 2,
    })
  }
  for (const y of [0, 1, 2]) {
    rects.push({
      id: `${prefix}-h-${y}`,
      x0: 0,
      x1: 2,
      y0: Math.max(0, y - half),
      y1: Math.min(2, y + half),
    })
  }
  return rects
}

function cfTopInsetFor(label: string) {
  const spec = cfChannelSpecs[label]
  const protrusion = Math.max(0, spec.thickness - cfGridThickness)
  const theta = (Math.PI / 180) * Math.max(1, Math.min(89.999, spec.contactAngle))
  return protrusion / Math.tan(theta)
}

const pdFootprints = [
  { x0: 0.5 - pdHalf, y0: 0.5 - pdHalf, x1: 0.5 + pdHalf, y1: 0.5 + pdHalf },
  { x0: 1.5 - pdHalf, y0: 0.5 - pdHalf, x1: 1.5 + pdHalf, y1: 0.5 + pdHalf },
  { x0: 0.5 - pdHalf, y0: 1.5 - pdHalf, x1: 0.5 + pdHalf, y1: 1.5 + pdHalf },
  { x0: 1.5 - pdHalf, y0: 1.5 - pdHalf, x1: 1.5 + pdHalf, y1: 1.5 + pdHalf },
]

const mlFootprints = [
  { cx: 0.5, cy: 0.5, rx: mlR, ry: mlR },
  { cx: 1.5, cy: 0.5, rx: mlR, ry: mlR },
  { cx: 0.5, cy: 1.5, rx: mlR, ry: mlR },
  { cx: 1.5, cy: 1.5, rx: mlR, ry: mlR },
]

// ===== Highlighting helpers =====
const groupColors: Record<string, string> = {
  pitch: '#1d70b8',
  unit_cell: '#1d70b8',
  air_t: '#7f8c8d',
  ml_h: '#8e44ad',
  ml_rx: '#8e44ad',
  ml_ry: '#8e44ad',
  ml_gap: '#8e44ad',
  shift: '#9b59b6',
  plan_t: '#7f8c8d',
  cf_t: '#27ae60',
  cf_angle: '#1f8a5b',
  grid_w: '#34495e',
  grid_t: '#34495e',
  barl_t: '#16a085',
  si_t: '#5d6d7e',
  dti_w: '#2980b9',
  dti_d: '#2980b9',
  pd_dz: '#c0392b',
  pd_pz: '#c0392b',
  pd_dxdy: '#c0392b',
  pd: '#c0392b',
  dti: '#2980b9',
  grid: '#34495e',
}
function dimColor(id: string) {
  return highlight.value === id ? '#e74c3c' : (groupColors[id] || 'var(--vp-c-text-2)')
}
function dimLayer(layerId: string) {
  if (!highlight.value) return false
  // Dim non-target layers when a parameter is hovered (keep target visible)
  const map: Record<string, string[]> = {
    ml_h: ['microlens'], ml_rx: ['microlens'], ml_ry: ['microlens'], ml_gap: ['microlens'], shift: ['microlens'],
    cf_t: ['color_filter'], cf_angle: ['color_filter'], grid_w: ['color_filter'], grid_t: ['color_filter'], grid: ['color_filter'],
    plan_t: ['planarization'], air_t: ['air'], barl_t: ['barl'],
    si_t: ['silicon'], dti_w: ['silicon'], dti_d: ['silicon'], dti: ['silicon'],
    pd_dz: ['silicon'], pd_pz: ['silicon'], pd_dxdy: ['silicon'], pd: ['silicon'],
  }
  const keep = map[highlight.value]
  if (!keep) return false
  return !keep.includes(layerId)
}

// ===== Legend rows =====
const legendRows = [
  { id: 'pitch',     param: 'pixel.pitch',                value: '1.0 µm',          color: '#1d70b8',
    meaningEn: 'In-plane pixel pitch (x and y)',          meaningKo: '면내 픽셀 피치 (x, y 모두)' },
  { id: 'unit_cell', param: 'pixel.unit_cell',            value: '[2, 2]',          color: '#1d70b8',
    meaningEn: 'Number of pixels in the periodic unit',   meaningKo: '주기 단위 셀의 픽셀 수' },
  { id: 'air_t',     param: 'air.thickness',              value: '1.0 µm',          color: '#7f8c8d',
    meaningEn: 'Air gap above the microlens',             meaningKo: '마이크로렌즈 위의 공기층 두께' },
  { id: 'ml_h',      param: 'microlens.height',           value: '0.6 µm',          color: '#8e44ad',
    meaningEn: 'Maximum lens sag (peak height)',          meaningKo: '렌즈 새그(최대 높이)' },
  { id: 'ml_rx',     param: 'microlens.radius_x',         value: '0.48 µm',         color: '#8e44ad',
    meaningEn: 'Lens semi-axis in x',                     meaningKo: 'x 방향 반축' },
  { id: 'ml_ry',     param: 'microlens.radius_y',         value: '0.48 µm',         color: '#8e44ad',
    meaningEn: 'Lens semi-axis in y',                     meaningKo: 'y 방향 반축' },
  { id: 'ml_gap',    param: 'microlens.gap',              value: '0.04 µm',         color: '#8e44ad',
    meaningEn: 'Gap between adjacent microlens bases',    meaningKo: '인접 렌즈 사이 간격' },
  { id: 'shift',     param: 'microlens.shift.shift_x/y',  value: '0 (auto_cra)',    color: '#9b59b6',
    meaningEn: 'Lateral lens offset (CRA correction)',    meaningKo: '주광선각(CRA) 보정용 횡방향 오프셋' },
  { id: 'plan_t',    param: 'planarization.thickness',    value: '0.3 µm',          color: '#7f8c8d',
    meaningEn: 'Spacer thickness between ML and CF',      meaningKo: '마이크로렌즈와 컬러 필터 사이 스페이서 두께' },
  { id: 'cf_t',      param: 'color_filter.{red,green,blue}.thickness', value: '0.60-0.65 µm', color: '#27ae60',
    meaningEn: 'Per-channel color-filter height above the CFA base', meaningKo: 'CFA base 위 색별 컬러 필터 높이' },
  { id: 'cf_angle',  param: 'color_filter.{red,green,blue}.contact_angle', value: '62-72°', color: '#1f8a5b',
    meaningEn: 'Sidewall taper above the metal-grid top', meaningKo: 'metal-grid top 위로 솟은 필터의 sidewall taper' },
  { id: 'grid_w',    param: 'color_filter.grid.width',    value: '0.05 µm',         color: '#34495e',
    meaningEn: 'Metal grid line width at pixel borders',  meaningKo: '픽셀 경계 금속 격자 선 너비' },
  { id: 'grid_t',    param: 'color_filter.grid.thickness', value: '0.47 µm',        color: '#34495e',
    meaningEn: 'Metal grid height from the CFA base',      meaningKo: 'CFA base 기준 metal grid 높이' },
  { id: 'barl_t',    param: 'barl.layers[i].thickness',   value: '0.01–0.03 µm',    color: '#16a085',
    meaningEn: 'Per-layer thickness of the AR stack',     meaningKo: '반사 방지 스택의 레이어별 두께' },
  { id: 'si_t',      param: 'silicon.thickness',          value: '3.0 µm',          color: '#5d6d7e',
    meaningEn: 'Total silicon substrate thickness',       meaningKo: '전체 실리콘 기판 두께' },
  { id: 'dti_w',     param: 'silicon.dti.width',          value: '0.1 µm',          color: '#2980b9',
    meaningEn: 'Trench width at pixel boundaries',        meaningKo: '픽셀 경계 트렌치 너비' },
  { id: 'dti_d',     param: 'silicon.dti.depth',          value: '3.0 µm',          color: '#2980b9',
    meaningEn: 'Trench depth from top of Si',             meaningKo: '실리콘 상단 기준 트렌치 깊이' },
  { id: 'pd_dxdy',   param: 'silicon.photodiode.size[dx,dy]', value: '0.7 × 0.7 µm', color: '#c0392b',
    meaningEn: 'PD lateral footprint per pixel',          meaningKo: '픽셀당 PD 횡방향 면적' },
  { id: 'pd_dz',     param: 'silicon.photodiode.size[dz]',     value: '2.0 µm',     color: '#c0392b',
    meaningEn: 'PD depth (z extent inside Si)',           meaningKo: '실리콘 내부 PD 깊이(z 방향 길이)' },
  { id: 'pd_pz',     param: 'silicon.photodiode.position[z]',  value: '0.5 µm (default; 0.2 shown)', color: '#c0392b',
    meaningEn: 'PD top below top of Si',                  meaningKo: '실리콘 상단 기준 PD 상단까지의 거리' },
]
</script>

<style scoped>
.param-diagram {
  margin: 1.5rem 0;
}

.tab-row {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.tab-btn {
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.15s ease;
}

.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
}

.tab-btn.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.hint {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  margin: 4px 0 8px 0;
}

.xy-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  padding: 8px 10px;
  margin: 0 0 8px 0;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  font-size: 0.85rem;
}

.xy-toggles-label {
  color: var(--vp-c-text-2);
  font-weight: 600;
}

.xy-toggle {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  user-select: none;
}

.xy-toggle input[type="checkbox"] {
  margin: 0;
  cursor: pointer;
}

.xy-toggle-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  border: 1px solid rgba(0, 0, 0, 0.2);
}

.xy-toggle-reset {
  margin-left: auto;
  padding: 3px 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
}

.xy-toggle-reset:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
}

.section-title {
  font-size: 13px;
  font-weight: 600;
  fill: var(--vp-c-text-1);
}

.axis-label {
  font-size: 11px;
  fill: var(--vp-c-text-2);
}

.tick-label {
  font-size: 10px;
  fill: var(--vp-c-text-3);
}

.dim-text {
  font-size: 11px;
  font-family: var(--vp-font-family-mono, monospace);
}

.layer-name {
  font-size: 11px;
  font-weight: 600;
  fill: #fff;
  paint-order: stroke;
  stroke: rgba(0, 0, 0, 0.35);
  stroke-width: 2.5;
}

.bayer-letter {
  font-size: 18px;
  font-weight: 700;
  fill: rgba(0, 0, 0, 0.55);
}

.legend-table {
  margin-top: 12px;
  overflow-x: auto;
}

.legend-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.legend-table th,
.legend-table td {
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid var(--vp-c-divider);
}

.legend-table th {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-weight: 600;
}

.legend-table tbody tr {
  cursor: pointer;
  transition: background 0.1s ease;
}

.legend-table tbody tr:hover,
.legend-table tbody tr.active {
  background: var(--vp-c-brand-soft);
}

.legend-table code {
  font-size: 0.85em;
  padding: 1px 4px;
  background: var(--vp-c-bg-soft);
  border-radius: 3px;
}

.swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 3px;
  border: 1px solid rgba(0, 0, 0, 0.15);
  vertical-align: middle;
}

@media (max-width: 640px) {
  .legend-table {
    font-size: 0.75rem;
  }
  .dim-text {
    font-size: 9px;
  }
}
</style>
