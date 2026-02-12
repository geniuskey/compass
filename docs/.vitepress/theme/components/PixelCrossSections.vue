<template>
  <div class="pixel-cross-sections">
    <div class="tab-row">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        class="tab-btn"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- XZ Cross-Section -->
    <svg
      v-if="activeTab === 'xz'"
      :viewBox="`0 0 ${svgW} ${svgH}`"
      class="cross-section-svg"
      @mousemove="onMouseMoveXZ"
      @mouseleave="hoverInfo = null"
    >
      <!-- Layer rectangles -->
      <rect
        v-for="layer in layers"
        :key="'xz-' + layer.id"
        :x="pad.left"
        :y="zToY(layer.zTop)"
        :width="plotW"
        :height="zToY(layer.zBot) - zToY(layer.zTop)"
        :fill="layer.color"
        :opacity="0.85"
        stroke="none"
      />

      <!-- BARL sublayers -->
      <rect
        v-for="(sub, i) in barlSublayers"
        :key="'barl-sub-' + i"
        :x="pad.left"
        :y="zToY(sub.zTop)"
        :width="plotW"
        :height="Math.max(zToY(sub.zBot) - zToY(sub.zTop), 2)"
        :fill="sub.color"
        opacity="0.9"
        stroke="#666"
        stroke-width="0.3"
      />

      <!-- DTI walls -->
      <rect
        v-for="dx in dtiXPositions"
        :key="'dti-xz-' + dx"
        :x="xToSvg(dx) - dtiHalfW"
        :y="zToY(3.0)"
        :width="dtiHalfW * 2"
        :height="zToY(0) - zToY(3.0)"
        fill="#aed6f1"
        opacity="0.8"
        stroke="#7fb3d3"
        stroke-width="0.5"
      />

      <!-- Metal grid in CF layer -->
      <rect
        v-for="dx in dtiXPositions"
        :key="'mg-xz-' + dx"
        :x="xToSvg(dx) - metalHalfW"
        :y="zToY(cfZTop)"
        :width="metalHalfW * 2"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        fill="#555555"
        opacity="0.85"
        stroke="#333"
        stroke-width="0.3"
      />

      <!-- CF Bayer colors for XZ (y=1.0 → row1: G, B) -->
      <rect
        v-for="(cf, i) in xzCFSegments"
        :key="'cf-xz-' + i"
        :x="xToSvg(cf.x0)"
        :y="zToY(cfZTop)"
        :width="xToSvg(cf.x1) - xToSvg(cf.x0)"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        :fill="cf.color"
        opacity="0.7"
        stroke="none"
      />

      <!-- Photodiodes (dashed) -->
      <rect
        v-for="(pd, i) in xzPhotodiodes"
        :key="'pd-xz-' + i"
        :x="xToSvg(pd.x0)"
        :y="zToY(pd.zTop)"
        :width="xToSvg(pd.x1) - xToSvg(pd.x0)"
        :height="zToY(pd.zBot) - zToY(pd.zTop)"
        fill="#b85c5c"
        opacity="0.25"
        stroke="#b85c5c"
        stroke-width="1.2"
        stroke-dasharray="4 2"
      />

      <!-- Microlens domes -->
      <path
        v-for="(dome, i) in xzMicrolens"
        :key="'ml-xz-' + i"
        :d="dome.path"
        fill="#dda0dd"
        opacity="0.65"
        stroke="#b07eb0"
        stroke-width="1"
      />

      <!-- Border -->
      <rect
        :x="pad.left"
        :y="pad.top"
        :width="plotW"
        :height="plotH"
        fill="none"
        stroke="var(--vp-c-divider)"
        stroke-width="1"
      />

      <!-- Right dimension lines -->
      <template v-for="layer in layers" :key="'dim-' + layer.id">
        <line
          :x1="pad.left + plotW + 8"
          :y1="zToY(layer.zTop)"
          :x2="pad.left + plotW + 8"
          :y2="zToY(layer.zBot)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <line
          :x1="pad.left + plotW + 5"
          :y1="zToY(layer.zTop)"
          :x2="pad.left + plotW + 11"
          :y2="zToY(layer.zTop)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <line
          :x1="pad.left + plotW + 5"
          :y1="zToY(layer.zBot)"
          :x2="pad.left + plotW + 11"
          :y2="zToY(layer.zBot)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="pad.left + plotW + 15"
          :y="(zToY(layer.zTop) + zToY(layer.zBot)) / 2 + 4"
          class="dim-label"
        >{{ layer.thickness }}µm</text>
      </template>

      <!-- Z axis label -->
      <text
        :x="pad.left - 8"
        :y="pad.top - 8"
        class="axis-label"
        text-anchor="end"
      >z (µm)</text>

      <!-- Z axis ticks -->
      <template v-for="z in zTicks" :key="'ztick-xz-' + z">
        <line
          :x1="pad.left - 4"
          :y1="zToY(z)"
          :x2="pad.left"
          :y2="zToY(z)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="pad.left - 7"
          :y="zToY(z) + 4"
          class="tick-label"
          text-anchor="end"
        >{{ z.toFixed(1) }}</text>
      </template>

      <!-- X axis label -->
      <text
        :x="pad.left + plotW / 2"
        :y="svgH - 4"
        class="axis-label"
        text-anchor="middle"
      >x (µm)</text>

      <!-- X axis ticks -->
      <template v-for="x in [0, 0.5, 1.0, 1.5, 2.0]" :key="'xtick-xz-' + x">
        <line
          :x1="xToSvg(x)"
          :y1="pad.top + plotH"
          :x2="xToSvg(x)"
          :y2="pad.top + plotH + 4"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="xToSvg(x)"
          :y="pad.top + plotH + 16"
          class="tick-label"
          text-anchor="middle"
        >{{ x.toFixed(1) }}</text>
      </template>

      <!-- Scale bar -->
      <line
        :x1="xToSvg(0.5)"
        :y1="svgH - 22"
        :x2="xToSvg(1.5)"
        :y2="svgH - 22"
        stroke="var(--vp-c-text-1)"
        stroke-width="2"
      />
      <text
        :x="xToSvg(1.0)"
        :y="svgH - 26"
        class="scale-label"
        text-anchor="middle"
      >1.0 µm</text>

      <!-- Section label -->
      <text
        :x="pad.left + plotW / 2"
        :y="pad.top - 10"
        class="section-title"
        text-anchor="middle"
      >XZ Cross-Section (y = 1.0 µm)</text>

      <!-- Hover tooltip -->
      <template v-if="hoverInfo">
        <line
          :x1="hoverInfo.svgX"
          :y1="pad.top"
          :x2="hoverInfo.svgX"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="3 3"
          opacity="0.6"
        />
        <line
          :x1="pad.left"
          :y1="hoverInfo.svgY"
          :x2="pad.left + plotW"
          :y2="hoverInfo.svgY"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="3 3"
          opacity="0.6"
        />
        <rect
          :x="hoverInfo.tooltipX"
          :y="hoverInfo.tooltipY"
          :width="180"
          height="50"
          rx="4"
          fill="var(--vp-c-bg)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.8"
          opacity="0.95"
        />
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 15"
          class="tooltip-text"
          font-weight="600"
        >{{ hoverInfo.layerName }}</text>
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 30"
          class="tooltip-text"
          fill="var(--vp-c-text-2)"
        >{{ hoverInfo.material }}</text>
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 43"
          class="tooltip-text"
          fill="var(--vp-c-text-3)"
        >z = {{ hoverInfo.z.toFixed(2) }} µm</text>
      </template>
    </svg>

    <!-- YZ Cross-Section -->
    <svg
      v-if="activeTab === 'yz'"
      :viewBox="`0 0 ${svgW} ${svgH}`"
      class="cross-section-svg"
      @mousemove="onMouseMoveYZ"
      @mouseleave="hoverInfo = null"
    >
      <!-- Layer rectangles -->
      <rect
        v-for="layer in layers"
        :key="'yz-' + layer.id"
        :x="pad.left"
        :y="zToY(layer.zTop)"
        :width="plotW"
        :height="zToY(layer.zBot) - zToY(layer.zTop)"
        :fill="layer.color"
        :opacity="0.85"
        stroke="none"
      />

      <!-- BARL sublayers -->
      <rect
        v-for="(sub, i) in barlSublayers"
        :key="'barl-yz-sub-' + i"
        :x="pad.left"
        :y="zToY(sub.zTop)"
        :width="plotW"
        :height="Math.max(zToY(sub.zBot) - zToY(sub.zTop), 2)"
        :fill="sub.color"
        opacity="0.9"
        stroke="#666"
        stroke-width="0.3"
      />

      <!-- DTI walls -->
      <rect
        v-for="dy in dtiYPositions"
        :key="'dti-yz-' + dy"
        :x="yToSvg(dy) - dtiHalfW"
        :y="zToY(3.0)"
        :width="dtiHalfW * 2"
        :height="zToY(0) - zToY(3.0)"
        fill="#aed6f1"
        opacity="0.8"
        stroke="#7fb3d3"
        stroke-width="0.5"
      />

      <!-- Metal grid in CF layer -->
      <rect
        v-for="dy in dtiYPositions"
        :key="'mg-yz-' + dy"
        :x="yToSvg(dy) - metalHalfW"
        :y="zToY(cfZTop)"
        :width="metalHalfW * 2"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        fill="#555555"
        opacity="0.85"
        stroke="#333"
        stroke-width="0.3"
      />

      <!-- CF Bayer colors for YZ (x=1.0 → col1: G, B) -->
      <rect
        v-for="(cf, i) in yzCFSegments"
        :key="'cf-yz-' + i"
        :x="yToSvg(cf.y0)"
        :y="zToY(cfZTop)"
        :width="yToSvg(cf.y1) - yToSvg(cf.y0)"
        :height="zToY(cfZBot) - zToY(cfZTop)"
        :fill="cf.color"
        opacity="0.7"
        stroke="none"
      />

      <!-- Photodiodes (dashed) -->
      <rect
        v-for="(pd, i) in yzPhotodiodes"
        :key="'pd-yz-' + i"
        :x="yToSvg(pd.y0)"
        :y="zToY(pd.zTop)"
        :width="yToSvg(pd.y1) - yToSvg(pd.y0)"
        :height="zToY(pd.zBot) - zToY(pd.zTop)"
        fill="#b85c5c"
        opacity="0.25"
        stroke="#b85c5c"
        stroke-width="1.2"
        stroke-dasharray="4 2"
      />

      <!-- Microlens domes -->
      <path
        v-for="(dome, i) in yzMicrolens"
        :key="'ml-yz-' + i"
        :d="dome.path"
        fill="#dda0dd"
        opacity="0.65"
        stroke="#b07eb0"
        stroke-width="1"
      />

      <!-- Border -->
      <rect
        :x="pad.left"
        :y="pad.top"
        :width="plotW"
        :height="plotH"
        fill="none"
        stroke="var(--vp-c-divider)"
        stroke-width="1"
      />

      <!-- Right dimension lines -->
      <template v-for="layer in layers" :key="'dim-yz-' + layer.id">
        <line
          :x1="pad.left + plotW + 8"
          :y1="zToY(layer.zTop)"
          :x2="pad.left + plotW + 8"
          :y2="zToY(layer.zBot)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <line
          :x1="pad.left + plotW + 5"
          :y1="zToY(layer.zTop)"
          :x2="pad.left + plotW + 11"
          :y2="zToY(layer.zTop)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <line
          :x1="pad.left + plotW + 5"
          :y1="zToY(layer.zBot)"
          :x2="pad.left + plotW + 11"
          :y2="zToY(layer.zBot)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="pad.left + plotW + 15"
          :y="(zToY(layer.zTop) + zToY(layer.zBot)) / 2 + 4"
          class="dim-label"
        >{{ layer.thickness }}µm</text>
      </template>

      <!-- Z axis label -->
      <text
        :x="pad.left - 8"
        :y="pad.top - 8"
        class="axis-label"
        text-anchor="end"
      >z (µm)</text>

      <!-- Z axis ticks -->
      <template v-for="z in zTicks" :key="'ztick-yz-' + z">
        <line
          :x1="pad.left - 4"
          :y1="zToY(z)"
          :x2="pad.left"
          :y2="zToY(z)"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="pad.left - 7"
          :y="zToY(z) + 4"
          class="tick-label"
          text-anchor="end"
        >{{ z.toFixed(1) }}</text>
      </template>

      <!-- Y axis label and ticks -->
      <text
        :x="pad.left + plotW / 2"
        :y="svgH - 4"
        class="axis-label"
        text-anchor="middle"
      >y (µm)</text>
      <template v-for="y in [0, 0.5, 1.0, 1.5, 2.0]" :key="'ytick-yz-' + y">
        <line
          :x1="yToSvg(y)"
          :y1="pad.top + plotH"
          :x2="yToSvg(y)"
          :y2="pad.top + plotH + 4"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.8"
        />
        <text
          :x="yToSvg(y)"
          :y="pad.top + plotH + 16"
          class="tick-label"
          text-anchor="middle"
        >{{ y.toFixed(1) }}</text>
      </template>

      <!-- Scale bar -->
      <line
        :x1="yToSvg(0.5)"
        :y1="svgH - 22"
        :x2="yToSvg(1.5)"
        :y2="svgH - 22"
        stroke="var(--vp-c-text-1)"
        stroke-width="2"
      />
      <text
        :x="yToSvg(1.0)"
        :y="svgH - 26"
        class="scale-label"
        text-anchor="middle"
      >1.0 µm</text>

      <!-- Section label -->
      <text
        :x="pad.left + plotW / 2"
        :y="pad.top - 10"
        class="section-title"
        text-anchor="middle"
      >YZ Cross-Section (x = 1.0 µm)</text>

      <!-- Hover tooltip -->
      <template v-if="hoverInfo">
        <line
          :x1="hoverInfo.svgX"
          :y1="pad.top"
          :x2="hoverInfo.svgX"
          :y2="pad.top + plotH"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="3 3"
          opacity="0.6"
        />
        <line
          :x1="pad.left"
          :y1="hoverInfo.svgY"
          :x2="pad.left + plotW"
          :y2="hoverInfo.svgY"
          stroke="var(--vp-c-text-3)"
          stroke-width="0.5"
          stroke-dasharray="3 3"
          opacity="0.6"
        />
        <rect
          :x="hoverInfo.tooltipX"
          :y="hoverInfo.tooltipY"
          :width="180"
          height="50"
          rx="4"
          fill="var(--vp-c-bg)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.8"
          opacity="0.95"
        />
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 15"
          class="tooltip-text"
          font-weight="600"
        >{{ hoverInfo.layerName }}</text>
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 30"
          class="tooltip-text"
          fill="var(--vp-c-text-2)"
        >{{ hoverInfo.material }}</text>
        <text
          :x="hoverInfo.tooltipX + 8"
          :y="hoverInfo.tooltipY + 43"
          class="tooltip-text"
          fill="var(--vp-c-text-3)"
        >z = {{ hoverInfo.z.toFixed(2) }} µm</text>
      </template>
    </svg>

    <!-- XY Plan View -->
    <div v-if="activeTab === 'xy'" class="xy-container">
      <div class="slider-row">
        <label class="slider-label">z = {{ xyZ.toFixed(2) }} µm</label>
        <input
          type="range"
          :min="0"
          :max="5.58"
          :step="0.02"
          v-model.number="xyZ"
          class="z-slider"
        />
        <span class="layer-badge">{{ xyLayerName }}</span>
      </div>
      <svg
        :viewBox="`0 0 ${xySize} ${xySize}`"
        class="cross-section-svg xy-svg"
      >
        <!-- Background -->
        <rect
          :x="xyPad"
          :y="xyPad"
          :width="xyPlot"
          :height="xyPlot"
          :fill="xyBgColor"
        />

        <!-- Silicon zone: DTI grid + photodiodes -->
        <template v-if="xyZ < 3.0">
          <!-- DTI grid lines -->
          <rect
            v-for="gx in [0, 1.0, 2.0]"
            :key="'dti-v-' + gx"
            :x="xyX(gx) - xyScale * 0.05"
            :y="xyPad"
            :width="xyScale * 0.1"
            :height="xyPlot"
            fill="#aed6f1"
            opacity="0.8"
          />
          <rect
            v-for="gy in [0, 1.0, 2.0]"
            :key="'dti-h-' + gy"
            :x="xyPad"
            :y="xyY(gy) - xyScale * 0.05"
            :width="xyPlot"
            :height="xyScale * 0.1"
            fill="#aed6f1"
            opacity="0.8"
          />
          <!-- Photodiodes -->
          <rect
            v-for="(pd, i) in xyPhotodiodes"
            :key="'xy-pd-' + i"
            :x="xyX(pd.cx) - xyScale * 0.35"
            :y="xyY(pd.cy) - xyScale * 0.35"
            :width="xyScale * 0.7"
            :height="xyScale * 0.7"
            fill="#b85c5c"
            opacity="0.35"
            stroke="#b85c5c"
            stroke-width="1.5"
            stroke-dasharray="4 2"
          />
        </template>

        <!-- BARL zone -->
        <template v-if="xyZ >= 3.0 && xyZ < 3.08">
          <rect
            :x="xyPad"
            :y="xyPad"
            :width="xyPlot"
            :height="xyPlot"
            fill="#8e44ad"
            opacity="0.5"
          />
        </template>

        <!-- CF zone: Bayer + metal grid -->
        <template v-if="xyZ >= 3.08 && xyZ < 3.68">
          <rect
            v-for="(bf, i) in xyBayerCells"
            :key="'xy-bayer-' + i"
            :x="xyX(bf.x0)"
            :y="xyY(bf.y0)"
            :width="xyX(bf.x1) - xyX(bf.x0)"
            :height="xyY(bf.y1) - xyY(bf.y0)"
            :fill="bf.color"
            opacity="0.75"
          />
          <text
            v-for="(bf, i) in xyBayerCells"
            :key="'xy-bayer-label-' + i"
            :x="(xyX(bf.x0) + xyX(bf.x1)) / 2"
            :y="(xyY(bf.y0) + xyY(bf.y1)) / 2 + 5"
            text-anchor="middle"
            class="bayer-label"
          >{{ bf.label }}</text>
          <!-- Metal grid -->
          <rect
            v-for="gx in [0, 1.0, 2.0]"
            :key="'mg-v-' + gx"
            :x="xyX(gx) - xyScale * 0.025"
            :y="xyPad"
            :width="xyScale * 0.05"
            :height="xyPlot"
            fill="#555555"
            opacity="0.85"
          />
          <rect
            v-for="gy in [0, 1.0, 2.0]"
            :key="'mg-h-' + gy"
            :x="xyPad"
            :y="xyY(gy) - xyScale * 0.025"
            :width="xyPlot"
            :height="xyScale * 0.05"
            fill="#555555"
            opacity="0.85"
          />
        </template>

        <!-- Planarization zone -->
        <template v-if="xyZ >= 3.68 && xyZ < 3.98">
          <rect
            :x="xyPad"
            :y="xyPad"
            :width="xyPlot"
            :height="xyPlot"
            fill="#d5dbdb"
            opacity="0.7"
          />
        </template>

        <!-- Microlens zone -->
        <template v-if="xyZ >= 3.98 && xyZ < 4.58">
          <ellipse
            v-for="(ml, i) in xyMicrolenses"
            :key="'xy-ml-' + i"
            :cx="xyX(ml.cx)"
            :cy="xyY(ml.cy)"
            :rx="ml.rx * xyScale"
            :ry="ml.ry * xyScale"
            fill="#dda0dd"
            opacity="0.6"
            stroke="#b07eb0"
            stroke-width="1.5"
          />
        </template>

        <!-- Air zone -->
        <template v-if="xyZ >= 4.58">
          <rect
            :x="xyPad"
            :y="xyPad"
            :width="xyPlot"
            :height="xyPlot"
            fill="#d6eaf8"
            opacity="0.4"
          />
        </template>

        <!-- Border -->
        <rect
          :x="xyPad"
          :y="xyPad"
          :width="xyPlot"
          :height="xyPlot"
          fill="none"
          stroke="var(--vp-c-divider)"
          stroke-width="1"
        />

        <!-- Axis ticks -->
        <template v-for="v in [0, 0.5, 1.0, 1.5, 2.0]" :key="'xy-xtick-' + v">
          <line
            :x1="xyX(v)"
            :y1="xyPad + xyPlot"
            :x2="xyX(v)"
            :y2="xyPad + xyPlot + 4"
            stroke="var(--vp-c-text-3)"
            stroke-width="0.8"
          />
          <text
            :x="xyX(v)"
            :y="xyPad + xyPlot + 16"
            class="tick-label"
            text-anchor="middle"
          >{{ v.toFixed(1) }}</text>
        </template>
        <template v-for="v in [0, 0.5, 1.0, 1.5, 2.0]" :key="'xy-ytick-' + v">
          <line
            :x1="xyPad - 4"
            :y1="xyY(v)"
            :x2="xyPad"
            :y2="xyY(v)"
            stroke="var(--vp-c-text-3)"
            stroke-width="0.8"
          />
          <text
            :x="xyPad - 7"
            :y="xyY(v) + 4"
            class="tick-label"
            text-anchor="end"
          >{{ v.toFixed(1) }}</text>
        </template>

        <!-- Axis labels -->
        <text
          :x="xyPad + xyPlot / 2"
          :y="xySize - 4"
          class="axis-label"
          text-anchor="middle"
        >x (µm)</text>
        <text
          :x="12"
          :y="xyPad + xyPlot / 2"
          class="axis-label"
          text-anchor="middle"
          transform="rotate(-90, 12, 180)"
        >y (µm)</text>

        <!-- Title -->
        <text
          :x="xyPad + xyPlot / 2"
          :y="xyPad - 10"
          class="section-title"
          text-anchor="middle"
        >XY Plan View (z = {{ xyZ.toFixed(2) }} µm)</text>
      </svg>
    </div>

    <!-- Legend -->
    <div class="legend">
      <span
        v-for="item in legendItems"
        :key="item.label"
        class="legend-item"
      >
        <span class="legend-swatch" :style="{ background: item.color }"></span>
        {{ item.label }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const activeTab = ref('xz')
const tabs = [
  { key: 'xz', label: 'XZ Plane' },
  { key: 'yz', label: 'YZ Plane' },
  { key: 'xy', label: 'XY Plan View' },
]

// SVG dimensions
const svgW = 560
const svgH = 440
const pad = { left: 55, right: 60, top: 30, bottom: 40 }
const plotW = svgW - pad.left - pad.right
const plotH = svgH - pad.top - pad.bottom

// Physics
const totalZ = 5.58
const pitch = 1.0

// Layers (bottom to top)
const layers = [
  { id: 'silicon', label: 'Silicon', color: '#5d6d7e', zBot: 0, zTop: 3.0, thickness: '3.0', material: 'Si' },
  { id: 'barl', label: 'BARL', color: '#8e44ad', zBot: 3.0, zTop: 3.08, thickness: '0.08', material: 'SiO2/HfO2/SiO2/Si3N4' },
  { id: 'colorfilter', label: 'Color Filter', color: '#27ae60', zBot: 3.08, zTop: 3.68, thickness: '0.6', material: 'Organic dye' },
  { id: 'planarization', label: 'Planarization', color: '#d5dbdb', zBot: 3.68, zTop: 3.98, thickness: '0.3', material: 'SiO2' },
  { id: 'microlens', label: 'Microlens', color: '#dda0dd', zBot: 3.98, zTop: 4.58, thickness: '0.6', material: 'Polymer (n=1.56)' },
  { id: 'air', label: 'Air', color: '#d6eaf8', zBot: 4.58, zTop: 5.58, thickness: '1.0', material: 'Air' },
]

const cfZBot = 3.08
const cfZTop = 3.68
const mlZBot = 3.98
const mlH = 0.6
const mlR = 0.48
const mlN = 2.5

// BARL sublayers
const barlSublayers = [
  { color: '#b3cde0', zBot: 3.0, zTop: 3.01, material: 'SiO2' },
  { color: '#6c71c4', zBot: 3.01, zTop: 3.035, material: 'HfO2' },
  { color: '#b3cde0', zBot: 3.035, zTop: 3.05, material: 'SiO2' },
  { color: '#2aa198', zBot: 3.05, zTop: 3.08, material: 'Si3N4' },
]

// Coordinate transforms
function xToSvg(x: number) { return pad.left + (x / 2.0) * plotW }
function yToSvg(y: number) { return pad.left + (y / 2.0) * plotW }
function zToY(z: number) { return pad.top + plotH - (z / totalZ) * plotH }

const zTicks = [0, 1.0, 2.0, 3.0, 4.0, 5.0]

// DTI and metal grid
const dtiXPositions = [0, 1.0, 2.0]
const dtiYPositions = [0, 1.0, 2.0]
const dtiHalfW = (0.1 / 2.0) * plotW / 2.0
const metalHalfW = (0.05 / 2.0) * plotW / 2.0

// Superellipse profile
function superellipseZ(x: number, cx: number): number {
  const absXR = Math.abs((x - cx) / mlR)
  if (absXR >= 1) return 0
  return mlH * Math.pow(1 - Math.pow(absXR, mlN), 1 / mlN)
}

// Microlens footprint radius at a given z-height within the lens
function lensRadiusAtZ(zInLens: number): number {
  if (zInLens <= 0 || zInLens >= mlH) return 0
  const ratio = zInLens / mlH
  return mlR * Math.pow(1 - Math.pow(ratio, mlN), 1 / mlN)
}

// XZ Bayer segments (y=1.0 → row1: G, B)
const xzCFSegments = [
  { x0: 0, x1: 1.0, color: '#27ae60', label: 'G' },
  { x1: 2.0, x0: 1.0, color: '#2980b9', label: 'B' },
]

// YZ Bayer segments (x=1.0 → col1: G, B)
const yzCFSegments = [
  { y0: 0, y1: 1.0, color: '#27ae60', label: 'G' },
  { y0: 1.0, y1: 2.0, color: '#2980b9', label: 'B' },
]

// Photodiodes
const xzPhotodiodes = [
  { x0: 0.5 - 0.35, x1: 0.5 + 0.35, zBot: 0.5, zTop: 2.5 },
  { x0: 1.5 - 0.35, x1: 1.5 + 0.35, zBot: 0.5, zTop: 2.5 },
]
const yzPhotodiodes = [
  { y0: 0.5 - 0.35, y1: 0.5 + 0.35, zBot: 0.5, zTop: 2.5 },
  { y0: 1.5 - 0.35, y1: 1.5 + 0.35, zBot: 0.5, zTop: 2.5 },
]

// Microlens dome paths
function buildDomePath(centers: number[], axis: 'x' | 'y') {
  return centers.map(cx => {
    const pts: { sx: number; sy: number }[] = []
    const numPts = 40
    for (let i = 0; i <= numPts; i++) {
      const physPos = cx - mlR + (2 * mlR * i) / numPts
      const physZ = mlZBot + superellipseZ(physPos, cx)
      const sx = axis === 'x' ? xToSvg(physPos) : yToSvg(physPos)
      pts.push({ sx, sy: zToY(physZ) })
    }
    const baseY = zToY(mlZBot)
    let d = `M ${(axis === 'x' ? xToSvg(cx - mlR) : yToSvg(cx - mlR)).toFixed(1)} ${baseY.toFixed(1)}`
    for (const p of pts) {
      d += ` L ${p.sx.toFixed(1)} ${p.sy.toFixed(1)}`
    }
    d += ` L ${(axis === 'x' ? xToSvg(cx + mlR) : yToSvg(cx + mlR)).toFixed(1)} ${baseY.toFixed(1)} Z`
    return { path: d }
  })
}

const xzMicrolens = computed(() => buildDomePath([0.5, 1.5], 'x'))
const yzMicrolens = computed(() => buildDomePath([0.5, 1.5], 'y'))

// Hover
interface HoverInfo {
  svgX: number
  svgY: number
  tooltipX: number
  tooltipY: number
  layerName: string
  material: string
  z: number
}
const hoverInfo = ref<HoverInfo | null>(null)

function findLayerAtZ(z: number) {
  for (const layer of layers) {
    if (z >= layer.zBot && z <= layer.zTop) return layer
  }
  return null
}

function onMouseMoveXZ(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const scaleY = svgH / rect.height
  const mx = (event.clientX - rect.left) * scaleX
  const my = (event.clientY - rect.top) * scaleY

  if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) {
    hoverInfo.value = null
    return
  }

  const physX = ((mx - pad.left) / plotW) * 2.0
  const physZ = ((pad.top + plotH - my) / plotH) * totalZ
  const layer = findLayerAtZ(physZ)
  if (!layer) { hoverInfo.value = null; return }

  const tooltipW = 180
  hoverInfo.value = {
    svgX: mx,
    svgY: my,
    tooltipX: mx + tooltipW + 10 > svgW - pad.right ? mx - tooltipW - 10 : mx + 10,
    tooltipY: Math.max(pad.top, Math.min(my - 25, pad.top + plotH - 55)),
    layerName: layer.label,
    material: layer.material,
    z: physZ,
  }
}

function onMouseMoveYZ(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const scaleX = svgW / rect.width
  const scaleY = svgH / rect.height
  const mx = (event.clientX - rect.left) * scaleX
  const my = (event.clientY - rect.top) * scaleY

  if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) {
    hoverInfo.value = null
    return
  }

  const physY = ((mx - pad.left) / plotW) * 2.0
  const physZ = ((pad.top + plotH - my) / plotH) * totalZ
  const layer = findLayerAtZ(physZ)
  if (!layer) { hoverInfo.value = null; return }

  const tooltipW = 180
  hoverInfo.value = {
    svgX: mx,
    svgY: my,
    tooltipX: mx + tooltipW + 10 > svgW - pad.right ? mx - tooltipW - 10 : mx + 10,
    tooltipY: Math.max(pad.top, Math.min(my - 25, pad.top + plotH - 55)),
    layerName: layer.label,
    material: layer.material,
    z: physZ,
  }
}

// XY Plan View
const xyZ = ref(3.4)
const xySize = 400
const xyPad = 40
const xyPlot = xySize - xyPad * 2
const xyScale = xyPlot / 2.0

function xyX(x: number) { return xyPad + (x / 2.0) * xyPlot }
function xyY(y: number) { return xyPad + (y / 2.0) * xyPlot }

const xyLayerName = computed(() => {
  const z = xyZ.value
  if (z < 3.0) return 'Silicon'
  if (z < 3.08) return 'BARL'
  if (z < 3.68) return 'Color Filter'
  if (z < 3.98) return 'Planarization'
  if (z < 4.58) return 'Microlens'
  return 'Air'
})

const xyBgColor = computed(() => {
  const z = xyZ.value
  if (z < 3.0) return '#5d6d7e'
  if (z < 3.08) return '#8e44ad'
  if (z < 3.68) return '#27ae60'
  if (z < 3.98) return '#d5dbdb'
  if (z < 4.58) return '#dda0dd'
  return '#d6eaf8'
})

const xyPhotodiodes = [
  { cx: 0.5, cy: 0.5 },
  { cx: 1.5, cy: 0.5 },
  { cx: 0.5, cy: 1.5 },
  { cx: 1.5, cy: 1.5 },
]

// Bayer: R(0,0) G(1,0) G(0,1) B(1,1)
const xyBayerCells = [
  { x0: 0, x1: 1.0, y0: 0, y1: 1.0, color: '#c0392b', label: 'R' },
  { x0: 1.0, x1: 2.0, y0: 0, y1: 1.0, color: '#27ae60', label: 'G' },
  { x0: 0, x1: 1.0, y0: 1.0, y1: 2.0, color: '#27ae60', label: 'G' },
  { x0: 1.0, x1: 2.0, y0: 1.0, y1: 2.0, color: '#2980b9', label: 'B' },
]

const xyMicrolenses = computed(() => {
  const zInLens = xyZ.value - mlZBot
  const r = lensRadiusAtZ(zInLens)
  if (r <= 0) return []
  return [
    { cx: 0.5, cy: 0.5, rx: r, ry: r },
    { cx: 1.5, cy: 0.5, rx: r, ry: r },
    { cx: 0.5, cy: 1.5, rx: r, ry: r },
    { cx: 1.5, cy: 1.5, rx: r, ry: r },
  ]
})

// Legend
const legendItems = [
  { label: 'Silicon', color: '#5d6d7e' },
  { label: 'BARL', color: '#8e44ad' },
  { label: 'Color Filter', color: '#27ae60' },
  { label: 'CF Red', color: '#c0392b' },
  { label: 'CF Blue', color: '#2980b9' },
  { label: 'Planarization', color: '#d5dbdb' },
  { label: 'Microlens', color: '#dda0dd' },
  { label: 'Air', color: '#d6eaf8' },
  { label: 'DTI', color: '#aed6f1' },
  { label: 'Metal Grid', color: '#555555' },
  { label: 'Photodiode', color: '#b85c5c' },
]
</script>

<style scoped>
.pixel-cross-sections {
  padding: 16px 0;
}
.tab-row {
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.tab-btn {
  padding: 8px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.88em;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}
.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.tab-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
.cross-section-svg {
  width: 100%;
  max-width: 600px;
  height: auto;
  display: block;
  margin: 0 auto;
}
.xy-svg {
  max-width: 440px;
}
.xy-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.slider-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.slider-label {
  font-size: 0.9em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  min-width: 100px;
}
.z-slider {
  width: 200px;
  accent-color: var(--vp-c-brand-1);
}
.layer-badge {
  font-size: 0.8em;
  padding: 3px 10px;
  border-radius: 12px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  fill: var(--vp-c-text-1);
}
.axis-label {
  font-size: 11px;
  fill: var(--vp-c-text-2);
  font-weight: 500;
}
.tick-label {
  font-size: 10px;
  fill: var(--vp-c-text-3);
}
.dim-label {
  font-size: 9px;
  fill: var(--vp-c-text-2);
}
.scale-label {
  font-size: 10px;
  fill: var(--vp-c-text-1);
  font-weight: 600;
}
.tooltip-text {
  font-size: 11px;
  fill: var(--vp-c-text-1);
}
.bayer-label {
  font-size: 16px;
  font-weight: 700;
  fill: #fff;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 16px;
  margin-top: 16px;
  padding: 10px 14px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
}
.legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  border: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}
</style>
