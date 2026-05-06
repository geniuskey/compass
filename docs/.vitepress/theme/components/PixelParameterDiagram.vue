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
        'Hover a parameter row in the legend below to highlight it on the diagram. All values shown match the default 1.0 µm BSI pixel (configs/pixel/default_bsi_1um.yaml).',
        '아래 범례의 파라미터 행에 마우스를 올리면 다이어그램에서 해당 위치가 강조됩니다. 표시된 값은 기본 1.0 µm BSI 픽셀(configs/pixel/default_bsi_1um.yaml) 기준입니다.'
      ) }}
    </p>

    <!-- ==================== XZ Cross-Section ==================== -->
    <svg
      v-if="activeTab === 'xz'"
      :viewBox="`0 0 ${xzW} ${xzH}`"
      class="diagram-svg"
      role="img"
      :aria-label="t('XZ cross-section with parameter annotations', 'XZ 단면 파라미터 주석')"
    >
      <!-- Layer fills (bottom→top) -->
      <rect
        v-for="layer in layers"
        :key="'L-' + layer.id"
        :x="pad.left"
        :y="zToY(layer.zTop)"
        :width="plotW"
        :height="zToY(layer.zBot) - zToY(layer.zTop)"
        :fill="layer.color"
        :opacity="dimLayer(layer.id) ? 0.25 : 0.85"
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

      <!-- CF Bayer columns -->
      <rect
        v-for="(cf, i) in cfCols"
        :key="'CF-' + i"
        :x="xToSvg(cf.x0)"
        :y="zToY(cfZTop)"
        :width="xToSvg(cf.x1) - xToSvg(cf.x0)"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        :fill="cf.color"
        :opacity="dimLayer('color_filter') ? 0.25 : 0.7"
      />

      <!-- Metal grid pillars at pixel boundaries -->
      <rect
        v-for="dx in dtiX"
        :key="'MG-' + dx"
        :x="xToSvg(dx) - mgHalfWPx"
        :y="zToY(cfZTop)"
        :width="mgHalfWPx * 2"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        fill="#555"
        :opacity="highlight === 'grid' ? 1 : 0.85"
        :stroke="highlight === 'grid' ? '#e74c3c' : '#333'"
        :stroke-width="highlight === 'grid' ? 1.5 : 0.3"
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
          :x1="xToSvg(1.0) - mgHalfWPx" :y1="zToY((cfZTop + cfZBot) / 2)"
          :x2="xToSvg(1.0) + mgHalfWPx" :y2="zToY((cfZTop + cfZBot) / 2)"
          :stroke="dimColor('grid_w')"
          :stroke-width="highlight === 'grid_w' ? 2 : 1"
        />
        <line
          :x1="xToSvg(1.0)" :y1="zToY((cfZTop + cfZBot) / 2)"
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

        <!-- DTI depth: vertical arrow alongside the trench -->
        <line
          :x1="xToSvg(1.0) + dtiHalfWPx + 6" :y1="zToY(siTop)"
          :x2="xToSvg(1.0) + dtiHalfWPx + 6" :y2="zToY(siTop - dtiDepth)"
          :stroke="dimColor('dti_d')"
          :stroke-width="highlight === 'dti_d' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(1.0) + dtiHalfWPx + 10" :y="zToY(siTop - dtiDepth / 2) + 3"
          :fill="dimColor('dti_d')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'dti_d' ? '700' : '500'"
        >dti.depth</text>
      </g>

      <!-- Photodiode size dz (vertical) and position z (offset from Si top) -->
      <g class="dim-group">
        <!-- pd.size dz on right photodiode -->
        <line
          :x1="xToSvg(1.5) + 60" :y1="zToY(pdZTop)"
          :x2="xToSvg(1.5) + 60" :y2="zToY(pdZBot)"
          :stroke="dimColor('pd_dz')"
          :stroke-width="highlight === 'pd_dz' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(1.5) + 64" :y="(zToY(pdZTop) + zToY(pdZBot)) / 2 + 3"
          :fill="dimColor('pd_dz')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'pd_dz' ? '700' : '500'"
        >photodiode.size[dz]</text>

        <!-- pd.position z: top of Si to top of PD -->
        <line
          :x1="xToSvg(0.5) - 18" :y1="zToY(siTop)"
          :x2="xToSvg(0.5) - 18" :y2="zToY(pdZTop)"
          :stroke="dimColor('pd_pz')"
          :stroke-width="highlight === 'pd_pz' ? 2 : 1"
          marker-start="url(#arr-d)" marker-end="url(#arr-u)"
        />
        <text
          :x="xToSvg(0.5) - 22" :y="(zToY(siTop) + zToY(pdZTop)) / 2 + 3"
          :fill="dimColor('pd_pz')"
          class="dim-text" text-anchor="end"
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
    >
      <!-- 2x2 Bayer cells (background) -->
      <rect
        v-for="cell in bayerCells"
        :key="'BY-' + cell.label + cell.cx + cell.cy"
        :x="xyToSvg(cell.cx - 0.5)"
        :y="xyToSvg(cell.cy - 0.5)"
        :width="xyToSvg(1) - xyToSvg(0)"
        :height="xyToSvg(1) - xyToSvg(0)"
        :fill="cell.fill"
        :opacity="dimLayer('color_filter') ? 0.2 : 0.55"
      />
      <text
        v-for="cell in bayerCells"
        :key="'BL-' + cell.label + cell.cx + cell.cy"
        :x="xyToSvg(cell.cx)"
        :y="xyToSvg(cell.cy) + 6"
        class="bayer-letter"
        text-anchor="middle"
      >{{ cell.label }}</text>

      <!-- Metal grid lines (CF boundaries) -->
      <g>
        <rect
          v-for="x in [0, 1, 2]"
          :key="'mgv-' + x"
          :x="xyToSvg(x) - mgHalfXY"
          :y="xyToSvg(0)"
          :width="mgHalfXY * 2"
          :height="xyToSvg(2) - xyToSvg(0)"
          fill="#555"
          :opacity="highlight === 'grid_w' ? 1 : 0.55"
          :stroke="highlight === 'grid_w' ? '#e74c3c' : 'none'"
        />
        <rect
          v-for="y in [0, 1, 2]"
          :key="'mgh-' + y"
          :x="xyToSvg(0)"
          :y="xyToSvg(y) - mgHalfXY"
          :width="xyToSvg(2) - xyToSvg(0)"
          :height="mgHalfXY * 2"
          fill="#555"
          :opacity="highlight === 'grid_w' ? 1 : 0.55"
          :stroke="highlight === 'grid_w' ? '#e74c3c' : 'none'"
        />
      </g>

      <!-- DTI grid (slightly wider, dashed-blue) -->
      <g>
        <rect
          v-for="x in [0, 1, 2]"
          :key="'dtv-' + x"
          :x="xyToSvg(x) - dtiHalfXY"
          :y="xyToSvg(0)"
          :width="dtiHalfXY * 2"
          :height="xyToSvg(2) - xyToSvg(0)"
          fill="#aed6f1"
          :opacity="highlight === 'dti_w' ? 1 : 0.6"
          :stroke="highlight === 'dti_w' ? '#e74c3c' : '#7fb3d3'"
          :stroke-width="highlight === 'dti_w' ? 1.6 : 0.5"
        />
        <rect
          v-for="y in [0, 1, 2]"
          :key="'dth-' + y"
          :x="xyToSvg(0)"
          :y="xyToSvg(y) - dtiHalfXY"
          :width="xyToSvg(2) - xyToSvg(0)"
          :height="dtiHalfXY * 2"
          fill="#aed6f1"
          :opacity="highlight === 'dti_w' ? 1 : 0.6"
          :stroke="highlight === 'dti_w' ? '#e74c3c' : '#7fb3d3'"
          :stroke-width="highlight === 'dti_w' ? 1.6 : 0.5"
        />
      </g>

      <!-- Photodiode footprints (dashed) -->
      <rect
        v-for="(pd, i) in pdFootprints"
        :key="'pdxy-' + i"
        :x="xyToSvg(pd.x0)"
        :y="xyToSvg(pd.y0)"
        :width="xyToSvg(pd.x1) - xyToSvg(pd.x0)"
        :height="xyToSvg(pd.y1) - xyToSvg(pd.y0)"
        fill="#b85c5c"
        :opacity="highlight === 'pd' || highlight === 'pd_dxdy' ? 0.4 : 0.18"
        :stroke="highlight === 'pd' || highlight === 'pd_dxdy' ? '#e74c3c' : '#b85c5c'"
        :stroke-width="highlight === 'pd' || highlight === 'pd_dxdy' ? 1.6 : 1.2"
        stroke-dasharray="4 2"
      />

      <!-- Microlens ellipses -->
      <ellipse
        v-for="(ml, i) in mlFootprints"
        :key="'mlxy-' + i"
        :cx="xyToSvg(ml.cx)"
        :cy="xyToSvg(ml.cy)"
        :rx="ml.rx * xyScale"
        :ry="ml.ry * xyScale"
        fill="#dda0dd"
        :opacity="dimLayer('microlens') ? 0.25 : 0.55"
        :stroke="highlight === 'ml_rx' || highlight === 'ml_ry' ? '#e74c3c' : '#b07eb0'"
        :stroke-width="highlight === 'ml_rx' || highlight === 'ml_ry' ? 1.6 : 1"
      />

      <!-- pitch arrow (bottom) -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(0)" :y1="xyToSvg(2) + 18"
          :x2="xyToSvg(1)" :y2="xyToSvg(2) + 18"
          :stroke="dimColor('pitch')"
          :stroke-width="highlight === 'pitch' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyToSvg(0.5)" :y="xyToSvg(2) + 32"
          :fill="dimColor('pitch')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pitch' ? '700' : '500'"
        >pitch</text>
      </g>

      <!-- unit_cell arrow (bottom, full 2 µm) -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(0)" :y1="xyToSvg(2) + 42"
          :x2="xyToSvg(2)" :y2="xyToSvg(2) + 42"
          :stroke="dimColor('unit_cell')"
          :stroke-width="highlight === 'unit_cell' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyToSvg(1)" :y="xyToSvg(2) + 56"
          :fill="dimColor('unit_cell')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'unit_cell' ? '700' : '500'"
        >unit_cell = [2, 2] → 2·pitch</text>
      </g>

      <!-- microlens.radius_x (horizontal across one ML) -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(0.5 - mlR)" :y1="xyToSvg(0.5)"
          :x2="xyToSvg(0.5 + mlR)" :y2="xyToSvg(0.5)"
          :stroke="dimColor('ml_rx')"
          :stroke-width="highlight === 'ml_rx' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyToSvg(0.5)" :y="xyToSvg(0.5) - 6"
          :fill="dimColor('ml_rx')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'ml_rx' ? '700' : '500'"
        >2·radius_x</text>
      </g>

      <!-- microlens.radius_y (vertical across one ML) -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(1.5)" :y1="xyToSvg(0.5 - mlR)"
          :x2="xyToSvg(1.5)" :y2="xyToSvg(0.5 + mlR)"
          :stroke="dimColor('ml_ry')"
          :stroke-width="highlight === 'ml_ry' ? 2 : 1"
          marker-start="url(#xy-arr-u)" marker-end="url(#xy-arr-d)"
        />
        <text
          :x="xyToSvg(1.5) + 6" :y="xyToSvg(0.5) + 3"
          :fill="dimColor('ml_ry')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'ml_ry' ? '700' : '500'"
        >2·radius_y</text>
      </g>

      <!-- photodiode size dx/dy on bottom-right pixel -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(1.5 - pdHalf)" :y1="xyToSvg(1.5) + pdHalfPx + 6"
          :x2="xyToSvg(1.5 + pdHalf)" :y2="xyToSvg(1.5) + pdHalfPx + 6"
          :stroke="dimColor('pd_dxdy')"
          :stroke-width="highlight === 'pd_dxdy' ? 2 : 1"
          marker-start="url(#xy-arr-l)" marker-end="url(#xy-arr-r)"
        />
        <text
          :x="xyToSvg(1.5)" :y="xyToSvg(1.5) + pdHalfPx + 18"
          :fill="dimColor('pd_dxdy')"
          class="dim-text" text-anchor="middle"
          :font-weight="highlight === 'pd_dxdy' ? '700' : '500'"
        >photodiode.size[dx]</text>

        <line
          :x1="xyToSvg(1.5) + pdHalfPx + 6" :y1="xyToSvg(1.5 - pdHalf)"
          :x2="xyToSvg(1.5) + pdHalfPx + 6" :y2="xyToSvg(1.5 + pdHalf)"
          :stroke="dimColor('pd_dxdy')"
          :stroke-width="highlight === 'pd_dxdy' ? 2 : 1"
          marker-start="url(#xy-arr-u)" marker-end="url(#xy-arr-d)"
        />
        <text
          :x="xyToSvg(1.5) + pdHalfPx + 10" :y="xyToSvg(1.5) + 3"
          :fill="dimColor('pd_dxdy')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'pd_dxdy' ? '700' : '500'"
        >size[dy]</text>
      </g>

      <!-- DTI width leader -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(1) - dtiHalfXY" :y1="xyToSvg(1.7)"
          :x2="xyToSvg(1) + dtiHalfXY" :y2="xyToSvg(1.7)"
          :stroke="dimColor('dti_w')"
          :stroke-width="highlight === 'dti_w' ? 2 : 1"
        />
        <line
          :x1="xyToSvg(1)" :y1="xyToSvg(1.7)"
          :x2="xyToSvg(0.45)" :y2="xyToSvg(1.95)"
          :stroke="dimColor('dti_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xyToSvg(0.43)" :y="xyToSvg(1.95) + 4"
          :fill="dimColor('dti_w')"
          class="dim-text" text-anchor="end"
          :font-weight="highlight === 'dti_w' ? '700' : '500'"
        >dti.width</text>
      </g>

      <!-- grid.width leader -->
      <g class="dim-group">
        <line
          :x1="xyToSvg(2) - mgHalfXY" :y1="xyToSvg(0.3)"
          :x2="xyToSvg(2) + mgHalfXY" :y2="xyToSvg(0.3)"
          :stroke="dimColor('grid_w')"
          :stroke-width="highlight === 'grid_w' ? 2 : 1"
        />
        <line
          :x1="xyToSvg(2)" :y1="xyToSvg(0.3)"
          :x2="xyToSvg(2.2)" :y2="xyToSvg(0.1)"
          :stroke="dimColor('grid_w')"
          stroke-width="0.7" stroke-dasharray="2 2"
        />
        <text
          :x="xyToSvg(2.22)" :y="xyToSvg(0.1) + 3"
          :fill="dimColor('grid_w')"
          class="dim-text" text-anchor="start"
          :font-weight="highlight === 'grid_w' ? '700' : '500'"
        >grid.width</text>
      </g>

      <!-- Domain border -->
      <rect
        :x="xyToSvg(0)"
        :y="xyToSvg(0)"
        :width="xyToSvg(2) - xyToSvg(0)"
        :height="xyToSvg(2) - xyToSvg(0)"
        fill="none"
        stroke="var(--vp-c-divider)"
        stroke-width="1"
      />

      <!-- Axes -->
      <text
        :x="xyToSvg(1)" :y="xyH - 6"
        class="axis-label" text-anchor="middle"
      >x (µm)</text>
      <text
        :x="14" :y="xyToSvg(1)"
        class="axis-label" text-anchor="middle"
        :transform="`rotate(-90, 14, ${xyToSvg(1)})`"
      >y (µm)</text>

      <!-- Title -->
      <text
        :x="(xyToSvg(0) + xyToSvg(2)) / 2" :y="xyPadTop - 12"
        class="section-title" text-anchor="middle"
      >{{ t('XY Top View — at silicon level (z ≈ 1.5 µm)', 'XY 평면도 — 실리콘 내부 (z ≈ 1.5 µm)') }}</text>

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

// ===== XZ geometry (matches default_bsi_1um.yaml) =====
const xzW = 720
const xzH = 520
const pad = { left: 110, right: 220, top: 30, bottom: 60 }
const plotW = xzW - pad.left - pad.right
const plotH = xzH - pad.top - pad.bottom
const totalZ = 5.58

const layers = [
  { id: 'silicon', label: 'silicon', color: '#5d6d7e', zBot: 0,    zTop: 3.0 },
  { id: 'barl',    label: 'barl',    color: '#8e44ad', zBot: 3.0,  zTop: 3.08 },
  { id: 'color_filter', label: 'color_filter', color: '#27ae60', zBot: 3.08, zTop: 3.68 },
  { id: 'planarization', label: 'planarization', color: '#d5dbdb', zBot: 3.68, zTop: 3.98 },
  { id: 'microlens', label: 'microlens', color: '#dda0dd', zBot: 3.98, zTop: 4.58 },
  { id: 'air',     label: 'air',     color: '#d6eaf8', zBot: 4.58, zTop: 5.58 },
]

const barlSublayers = [
  { color: '#7fb3d8', zBot: 3.0,   zTop: 3.01,  material: 'SiO2' },
  { color: '#6c71c4', zBot: 3.01,  zTop: 3.035, material: 'HfO2' },
  { color: '#e8d44d', zBot: 3.035, zTop: 3.05,  material: 'SiO2' },
  { color: '#2aa198', zBot: 3.05,  zTop: 3.08,  material: 'Si3N4' },
]

const cfZBot = 3.08
const cfZTop = 3.68
const siBot = 0.0
const siTop = 3.0
const dtiDepth = 3.0
const dtiWidth = 0.1
const mgWidth = 0.05
const mlZBot = 3.98
const mlH = 0.6
const mlR = 0.48
const mlGap = 0.04
const shiftXIllustrative = 0.12

// Photodiode (1µm pixel default): position [0,0,0.5] from pixel center, size [0.7,0.7,2.0]
// Pixel centers at (0.5, 0.5) and (1.5, 0.5) for the bottom row of the unit_cell.
// position[z] = 0.5 means the *top* of PD sits 0.5 µm below top of Si (z=2.5) per yaml convention.
const pdSizeXY = 0.7
const pdSizeZ = 2.0
const pdZTop = siTop - 0.5  // 2.5
const pdZBot = pdZTop - pdSizeZ // 0.5

const pdRectsXZ = [
  { x0: 0.5 - pdSizeXY / 2, x1: 0.5 + pdSizeXY / 2, zTop: pdZTop, zBot: pdZBot },
  { x0: 1.5 - pdSizeXY / 2, x1: 1.5 + pdSizeXY / 2, zTop: pdZTop, zBot: pdZBot },
]

const cfCols = [
  { x0: 0.0, x1: 1.0, color: '#27ae60' }, // G
  { x0: 1.0, x1: 2.0, color: '#3498db' }, // B
]

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
  { id: 'air',     param: 'air_t',    zTop: 5.58, zBot: 4.58, offset: 14, label: 'air.thickness = 1.0 µm' },
  { id: 'ml',      param: 'ml_h',     zTop: 4.58, zBot: 3.98, offset: 14, label: 'microlens (height = 0.6)' },
  { id: 'plan',    param: 'plan_t',   zTop: 3.98, zBot: 3.68, offset: 14, label: 'planarization.thickness = 0.3' },
  { id: 'cf',      param: 'cf_t',     zTop: 3.68, zBot: 3.08, offset: 14, label: 'color_filter.thickness = 0.6' },
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
function xyToSvg(v: number) { return xyPadLeft + v * xyScale }

const dtiHalfXY = (dtiWidth / 2) * xyScale
const mgHalfXY = (mgWidth / 2) * xyScale
const pdHalf = pdSizeXY / 2
const pdHalfPx = pdHalf * xyScale

const bayerCells = [
  { cx: 0.5, cy: 0.5, label: 'R', fill: '#e74c3c' },
  { cx: 1.5, cy: 0.5, label: 'G', fill: '#27ae60' },
  { cx: 0.5, cy: 1.5, label: 'G', fill: '#27ae60' },
  { cx: 1.5, cy: 1.5, label: 'B', fill: '#3498db' },
]

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
  grid_w: '#34495e',
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
    cf_t: ['color_filter'], grid_w: ['color_filter'], grid: ['color_filter'],
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
  { id: 'cf_t',      param: 'color_filter.thickness',     value: '0.6 µm',          color: '#27ae60',
    meaningEn: 'Color-filter (Bayer) layer thickness',    meaningKo: '컬러 필터(베이어) 레이어 두께' },
  { id: 'grid_w',    param: 'color_filter.grid.width',    value: '0.05 µm',         color: '#34495e',
    meaningEn: 'Metal grid line width at pixel borders',  meaningKo: '픽셀 경계 금속 격자 선 너비' },
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
  { id: 'pd_pz',     param: 'silicon.photodiode.position[z]',  value: '0.5 µm',     color: '#c0392b',
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
