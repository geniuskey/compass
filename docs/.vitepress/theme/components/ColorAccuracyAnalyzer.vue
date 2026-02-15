<template>
  <div class="ca-container">
    <h4>{{ t('Color Accuracy Analyzer', '색 정확도 분석기') }}</h4>
    <p class="component-description">
      {{ t(
        'Compute color reproduction accuracy (deltaE) for ColorChecker patches using TMM-based QE spectra and a least-squares Color Correction Matrix.',
        'TMM 기반 QE 스펙트럼과 최소제곱 색 보정 행렬(CCM)을 사용하여 ColorChecker 패치의 색 재현 정확도(deltaE)를 계산합니다.'
      ) }}
    </p>

    <!-- Controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Si Thickness', '실리콘 두께') }}: <strong>{{ siThickness.toFixed(1) }} &mu;m</strong>
        </label>
        <input type="range" min="1" max="5" step="0.1" v-model.number="siThickness" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('CF Bandwidth (FWHM)', 'CF 대역폭 (FWHM)') }}: <strong>{{ cfBandwidth }} nm</strong>
        </label>
        <input type="range" min="50" max="150" step="5" v-model.number="cfBandwidth" class="ctrl-range" />
      </div>
      <div class="chart-toggle">
        <button
          :class="['toggle-btn', { active: chartType === 'classic' }]"
          @click="chartType = 'classic'"
        >Classic 24</button>
        <button
          :class="['toggle-btn', { active: chartType === 'sg' }]"
          @click="chartType = 'sg'"
        >SG 140</button>
      </div>
      <div class="chart-toggle">
        <button
          v-for="m in DE_METHODS" :key="m.key"
          :class="['toggle-btn', { active: deMethod === m.key }]"
          @click="deMethod = m.key"
        >{{ m.label }}</button>
      </div>
      <div class="select-group">
        <label class="select-label">{{ t('Illuminant', '광원') }}</label>
        <select v-model="illuminant" class="ctrl-select">
          <option v-for="(ill, key) in ILLUMINANTS" :key="key" :value="key">{{ ill.label }}</option>
        </select>
      </div>
      <div class="chart-toggle">
        <button v-for="m in WB_METHODS" :key="m.key"
          :class="['toggle-btn', { active: wbMethod === m.key }]"
          @click="wbMethod = m.key"
        >{{ m.label }}</button>
      </div>
    </div>

    <!-- Noise controls -->
    <div class="controls-row">
      <div class="slider-group">
        <label>
          {{ t('Shot Noise', '샷 노이즈') }}: <strong>{{ shotNoise }} e&minus;</strong>
        </label>
        <input type="range" min="0" max="100" step="5" v-model.number="shotNoise" class="ctrl-range" />
      </div>
      <div class="slider-group">
        <label>
          {{ t('Read Noise', '리드 노이즈') }}: <strong>{{ readNoise }} e&minus;</strong>
        </label>
        <input type="range" min="0" max="50" step="1" v-model.number="readNoise" class="ctrl-range" />
      </div>
    </div>

    <!-- Action row -->
    <div class="action-row">
      <button class="action-btn" @click="runOptimization" :disabled="optimizing">
        {{ optimizing ? t('Optimizing...', '최적화 중...') : t('Auto Optimize', '자동 최적화') }}
      </button>
      <button class="action-btn" @click="takeSnapshot">{{ t('Snapshot', '스냅샷') }}</button>
      <button class="action-btn" v-if="snapshot" @click="snapshot = null">{{ t('Clear', '삭제') }}</button>
      <button class="action-btn" @click="exportCsv">CSV</button>
      <button class="action-btn" @click="exportPng">PNG</button>
      <button class="action-btn" @click="saveCondition">{{ t('Save Config', '설정 저장') }}</button>
    </div>

    <!-- Optimization result -->
    <div class="opt-result" v-if="optResult">
      {{ t('Optimal', '최적') }}: Si={{ optResult.si.toFixed(1) }}&mu;m, BW={{ optResult.bw }}nm &rarr; {{ deAxisLabel }}={{ optResult.avgDE.toFixed(2) }}
    </div>

    <!-- Snapshot comparison -->
    <div class="snapshot-bar" v-if="snapshot">
      <span class="snap-label">{{ t('Snapshot', '스냅샷') }}: {{ snapshot.label }}</span>
      <span class="snap-stat">Avg={{ snapshot.avgDE.toFixed(2) }}</span>
      <span class="snap-delta" :style="{ color: avgDeltaE - snapshot.avgDE < 0 ? '#27ae60' : '#e74c3c' }">
        ({{ avgDeltaE - snapshot.avgDE > 0 ? '+' : '' }}{{ (avgDeltaE - snapshot.avgDE).toFixed(2) }})
      </span>
    </div>

    <!-- Summary bar -->
    <div class="results-grid">
      <div class="result-card">
        <div class="result-label">{{ t('Average deltaE', '평균 deltaE') }}</div>
        <div class="result-value highlight">{{ avgDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Max deltaE', '최대 deltaE') }}</div>
        <div class="result-value">{{ maxDeltaE.toFixed(2) }}</div>
      </div>
      <div class="result-card">
        <div class="result-label">{{ t('Excellent (deltaE < 3)', '우수 (deltaE < 3)') }}</div>
        <div class="result-value highlight">{{ excellentPct.toFixed(0) }}%</div>
      </div>
    </div>

    <!-- QE Spectrum -->
    <div class="chart-section">
      <h5>{{ t('Sensor QE Spectrum', '센서 QE 스펙트럼') }}</h5>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${qeW} ${qeH}`" class="de-svg">
          <line
            v-for="tick in qeYTicks" :key="'qyg'+tick"
            :x1="qePad.left" :y1="qeYScale(tick)"
            :x2="qePad.left + qePlotW" :y2="qeYScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line :x1="qePad.left" :y1="qePad.top" :x2="qePad.left" :y2="qePad.top + qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="qePad.left" :y1="qePad.top + qePlotH" :x2="qePad.left + qePlotW" :y2="qePad.top + qePlotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <text
            v-for="tick in qeYTicks" :key="'qyl'+tick"
            :x="qePad.left - 6" :y="qeYScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick.toFixed(1) }}</text>
          <text
            v-for="wl in qeXTicks" :key="'qxl'+wl"
            :x="qeXScale(wl)" :y="qePad.top + qePlotH + 14"
            text-anchor="middle" class="tick-label"
          >{{ wl }}</text>
          <path :d="qePathR" fill="none" stroke="#e74c3c" stroke-width="2" opacity="0.8" />
          <path :d="qePathG" fill="none" stroke="#27ae60" stroke-width="2" opacity="0.8" />
          <path :d="qePathB" fill="none" stroke="#3498db" stroke-width="2" opacity="0.8" />
          <text :x="10" :y="qePad.top + qePlotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${qePad.top + qePlotH / 2})`">QE</text>
          <text :x="qePad.left + qePlotW / 2" :y="qePad.top + qePlotH + 26" text-anchor="middle" class="axis-title">{{ t('Wavelength (nm)', '파장 (nm)') }}</text>
          <!-- Legend -->
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 6" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 6" stroke="#e74c3c" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 9" class="tick-label">Red</text>
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 18" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 18" stroke="#27ae60" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 21" class="tick-label">Green</text>
          <line :x1="qePad.left + qePlotW - 80" :y1="qePad.top + 30" :x2="qePad.left + qePlotW - 65" :y2="qePad.top + 30" stroke="#3498db" stroke-width="2" />
          <text :x="qePad.left + qePlotW - 62" :y="qePad.top + 33" class="tick-label">Blue</text>
        </svg>
      </div>
    </div>

    <!-- CCM Matrix -->
    <div class="ccm-section" v-if="ccmMatrix">
      <h5>{{ t('Color Correction Matrix (CCM)', '색 보정 행렬 (CCM)') }}</h5>
      <table class="ccm-table">
        <thead>
          <tr><th></th><th>R<sub>in</sub></th><th>G<sub>in</sub></th><th>B<sub>in</sub></th></tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in ccmMatrix" :key="i">
            <th>{{ ['R','G','B'][i] }}<sub>out</sub></th>
            <td v-for="(v, j) in row" :key="j" :class="{ 'ccm-diag': i === j }">{{ v.toFixed(4) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Patch grid -->
    <div class="patch-section">
      <div class="section-header">
        <h5>{{ t('ColorChecker Patches', 'ColorChecker 패치') }}
          <span class="patch-count">({{ activePatches.length }})</span>
        </h5>
        <div class="chart-toggle view-toggle">
          <button
            :class="['toggle-btn', { active: patchView === 'swatch' }]"
            @click="patchView = 'swatch'"
          >{{ t('Swatch', '스와치') }}</button>
          <button
            :class="['toggle-btn', { active: patchView === 'heatmap' }]"
            @click="patchView = 'heatmap'"
          >{{ t('Heatmap', '히트맵') }}</button>
        </div>
      </div>
      <div :class="['patch-grid', chartType === 'sg' ? 'patch-grid-sg' : 'patch-grid-classic']">
        <div v-for="(patch, idx) in patchResults" :key="idx" class="patch-cell">
          <!-- Swatch view -->
          <template v-if="patchView === 'swatch'">
            <div class="patch-swatch">
              <div class="patch-ref" :style="{ background: rgbStr(patch.refSrgb) }"></div>
              <div class="patch-sensor" :style="{ background: rgbStr(patch.corrSrgb) }"></div>
            </div>
          </template>
          <!-- Heatmap view -->
          <template v-else>
            <div class="heatmap-cell" :style="{ background: deHeatBg(patch.deltaE) }">
              <span class="heatmap-val" :style="{ color: deHeatText(patch.deltaE) }">{{ patch.deltaE.toFixed(chartType === 'sg' ? 0 : 1) }}</span>
            </div>
          </template>
          <div v-if="chartType === 'classic'" class="patch-name">{{ patch.name }}</div>
          <div
            v-if="patchView === 'swatch'"
            class="patch-de"
            :class="{ 'patch-de-sg': chartType === 'sg' }"
            :style="{ color: deColor(patch.deltaE) }"
          >{{ patch.deltaE.toFixed(chartType === 'sg' ? 0 : 1) }}</div>
        </div>
      </div>
    </div>

    <!-- DeltaE bar chart -->
    <div class="chart-section">
      <h5>{{ t('deltaE per Patch', '패치별 deltaE') }}</h5>
      <div class="svg-wrapper">
        <svg
          ref="deChartSvg"
          :viewBox="`0 0 ${chartW} ${chartH}`"
          class="de-svg"
          @mousemove="onChartMouseMove"
          @mouseleave="chartHover = null"
        >
          <line
            v-for="tick in yTicks" :key="'yg'+tick"
            :x1="pad.left" :y1="yScale(tick)"
            :x2="pad.left + plotW" :y2="yScale(tick)"
            stroke="var(--vp-c-divider)" stroke-width="0.5" stroke-dasharray="3,3"
          />
          <line
            :x1="pad.left" :y1="yScale(3)"
            :x2="pad.left + plotW" :y2="yScale(3)"
            stroke="#27ae60" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(3) + 3" class="ref-label" fill="#27ae60">3</text>
          <line
            :x1="pad.left" :y1="yScale(6)"
            :x2="pad.left + plotW" :y2="yScale(6)"
            stroke="#e67e22" stroke-width="1" stroke-dasharray="6,3" opacity="0.7"
          />
          <text :x="pad.left + plotW + 4" :y="yScale(6) + 3" class="ref-label" fill="#e67e22">6</text>
          <line :x1="pad.left" :y1="pad.top" :x2="pad.left" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="pad.left" :y1="pad.top + plotH" :x2="pad.left + plotW" :y2="pad.top + plotH" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <text
            v-for="tick in yTicks" :key="'yl'+tick"
            :x="pad.left - 6" :y="yScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick }}</text>
          <!-- Snapshot ghost bars -->
          <template v-if="snapshot && snapshot.patchDEs.length === patchResults.length">
            <rect
              v-for="(de, idx) in snapshot.patchDEs" :key="'snap'+idx"
              :x="barX(idx)"
              :y="yScale(de)"
              :width="barWidth"
              :height="Math.max(0, pad.top + plotH - yScale(de))"
              fill="none"
              stroke="var(--vp-c-text-3)"
              stroke-width="1"
              stroke-dasharray="2,2"
              opacity="0.4"
            />
          </template>
          <rect
            v-for="(patch, idx) in patchResults" :key="'bar'+idx"
            :x="barX(idx)"
            :y="yScale(patch.deltaE)"
            :width="barWidth"
            :height="Math.max(0, pad.top + plotH - yScale(patch.deltaE))"
            :fill="deColor(patch.deltaE)"
            opacity="0.8"
            rx="1"
          />
          <template v-if="chartType === 'classic'">
            <text
              v-for="(patch, idx) in patchResults" :key="'xl'+idx"
              :x="barX(idx) + barWidth / 2"
              :y="pad.top + plotH + 10"
              text-anchor="end"
              class="tick-label"
              :transform="`rotate(-45, ${barX(idx) + barWidth / 2}, ${pad.top + plotH + 10})`"
            >{{ patch.name.substring(0, 8) }}</text>
          </template>
          <text :x="10" :y="pad.top + plotH / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 10, ${pad.top + plotH / 2})`">{{ deAxisLabel }}</text>
          <template v-if="chartHover">
            <rect
              :x="chartHover.tx" :y="pad.top + 4"
              width="140" height="34" rx="4"
              fill="var(--vp-c-bg)" stroke="var(--vp-c-divider)" stroke-width="0.8" opacity="0.95"
            />
            <text :x="chartHover.tx + 6" :y="pad.top + 18" class="tooltip-text">{{ chartHover.name }}</text>
            <text :x="chartHover.tx + 6" :y="pad.top + 30" class="tooltip-text">&Delta;E = {{ chartHover.de.toFixed(2) }}</text>
          </template>
        </svg>
      </div>
    </div>

    <!-- a*b* Chromaticity Diagram -->
    <div class="chart-section">
      <h5>{{ t('a*b* Chromaticity Diagram', 'a*b* 색도 다이어그램') }}</h5>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${abW} ${abH}`" class="ab-svg">
          <!-- Grid -->
          <line
            v-for="tick in abTicks" :key="'abxg'+tick"
            :x1="abXScale(tick)" :y1="abPad.top"
            :x2="abXScale(tick)" :y2="abPad.top + abPlotS"
            stroke="var(--vp-c-divider)" :stroke-width="tick === 0 ? 1 : 0.5"
            :stroke-dasharray="tick === 0 ? 'none' : '3,3'"
          />
          <line
            v-for="tick in abTicks" :key="'abyg'+tick"
            :x1="abPad.left" :y1="abYScale(tick)"
            :x2="abPad.left + abPlotS" :y2="abYScale(tick)"
            stroke="var(--vp-c-divider)" :stroke-width="tick === 0 ? 1 : 0.5"
            :stroke-dasharray="tick === 0 ? 'none' : '3,3'"
          />
          <!-- Tick labels -->
          <text
            v-for="tick in abTicks" :key="'abxl'+tick"
            :x="abXScale(tick)" :y="abPad.top + abPlotS + 14"
            text-anchor="middle" class="tick-label"
          >{{ tick }}</text>
          <text
            v-for="tick in abTicks" :key="'abyl'+tick"
            :x="abPad.left - 6" :y="abYScale(tick) + 3"
            text-anchor="end" class="tick-label"
          >{{ tick }}</text>
          <!-- Axis titles -->
          <text :x="abPad.left + abPlotS / 2" :y="abPad.top + abPlotS + 26" text-anchor="middle" class="axis-title">a*</text>
          <text :x="6" :y="abPad.top + abPlotS / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 6, ${abPad.top + abPlotS / 2})`">b*</text>
          <!-- Error lines -->
          <line
            v-for="(p, idx) in patchResults" :key="'aberr'+idx"
            :x1="abXScale(p.refLab[1])" :y1="abYScale(p.refLab[2])"
            :x2="abXScale(p.corrLab[1])" :y2="abYScale(p.corrLab[2])"
            stroke="var(--vp-c-text-3)" :stroke-width="chartType === 'sg' ? 0.5 : 1" :opacity="chartType === 'sg' ? 0.3 : 0.5"
          />
          <!-- Reference points -->
          <circle
            v-for="(p, idx) in patchResults" :key="'abref'+idx"
            :cx="abXScale(p.refLab[1])" :cy="abYScale(p.refLab[2])"
            :r="chartType === 'sg' ? 2 : 4"
            :fill="rgbStr(p.refSrgb)" stroke="var(--vp-c-text-3)" :stroke-width="chartType === 'sg' ? 0.3 : 0.5"
          />
          <!-- Corrected points -->
          <circle
            v-for="(p, idx) in patchResults" :key="'abcorr'+idx"
            :cx="abXScale(p.corrLab[1])" :cy="abYScale(p.corrLab[2])"
            :r="chartType === 'sg' ? 1.5 : 3"
            fill="none" :stroke="rgbStr(p.corrSrgb)" :stroke-width="chartType === 'sg' ? 0.8 : 1.5"
          />
          <!-- Legend -->
          <circle :cx="abPad.left + abPlotS - 50" :cy="abPad.top + 8" r="4" fill="#888" stroke="var(--vp-c-text-3)" stroke-width="0.5" />
          <text :x="abPad.left + abPlotS - 42" :y="abPad.top + 11" class="tick-label">{{ t('Reference', '기준') }}</text>
          <circle :cx="abPad.left + abPlotS - 50" :cy="abPad.top + 20" r="3" fill="none" stroke="#888" stroke-width="1.5" />
          <text :x="abPad.left + abPlotS - 42" :y="abPad.top + 23" class="tick-label">{{ t('Corrected', '보정') }}</text>
        </svg>
      </div>
    </div>

    <!-- CIE xy Chromaticity Diagram -->
    <div class="chart-section">
      <h5>{{ t('CIE xy Gamut Coverage', 'CIE xy 색역 커버리지') }}</h5>
      <div class="gamut-stats">
        <span v-for="g in gamutStats" :key="g.name" class="gamut-tag" :style="{ borderColor: g.color }">
          {{ g.name }}: {{ g.pct.toFixed(0) }}%
        </span>
      </div>
      <div class="svg-wrapper">
        <svg :viewBox="`0 0 ${xyW} ${xyH}`" class="ab-svg">
          <path :d="spectralLocusPath" fill="none" stroke="var(--vp-c-text-3)" stroke-width="1" />
          <polygon :points="gamutTriangle(SRGB_PRIM)" fill="none" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.7" />
          <polygon :points="gamutTriangle(P3_PRIM)" fill="none" stroke="#8e44ad" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.7" />
          <polygon :points="gamutTriangle(BT2020_PRIM)" fill="none" stroke="#2980b9" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.7" />
          <line :x1="xyPad.left" :y1="xyPad.top" :x2="xyPad.left" :y2="xyPad.top + xyPlotS" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <line :x1="xyPad.left" :y1="xyPad.top + xyPlotS" :x2="xyPad.left + xyPlotS" :y2="xyPad.top + xyPlotS" stroke="var(--vp-c-text-2)" stroke-width="1" />
          <template v-for="tick in xyTicks" :key="'xyt'+tick">
            <text :x="xyXScale(tick)" :y="xyPad.top + xyPlotS + 14" text-anchor="middle" class="tick-label">{{ tick.toFixed(1) }}</text>
            <text :x="xyPad.left - 6" :y="xyYScale(tick) + 3" text-anchor="end" class="tick-label">{{ tick.toFixed(1) }}</text>
          </template>
          <text :x="xyPad.left + xyPlotS / 2" :y="xyPad.top + xyPlotS + 26" text-anchor="middle" class="axis-title">x</text>
          <text :x="6" :y="xyPad.top + xyPlotS / 2" text-anchor="middle" class="axis-title" :transform="`rotate(-90, 6, ${xyPad.top + xyPlotS / 2})`">y</text>
          <circle v-for="(p, idx) in xyRefPoints" :key="'xyref'+idx"
            :cx="xyXScale(p.x)" :cy="xyYScale(p.y)"
            :r="chartType === 'sg' ? 2 : 4"
            :fill="rgbStr(p.rgb)" stroke="var(--vp-c-text-3)" :stroke-width="chartType === 'sg' ? 0.3 : 0.5"
          />
          <circle v-for="(p, idx) in xyCorrPoints" :key="'xycorr'+idx"
            :cx="xyXScale(p.x)" :cy="xyYScale(p.y)"
            :r="chartType === 'sg' ? 1.5 : 3"
            fill="none" :stroke="rgbStr(p.rgb)" :stroke-width="chartType === 'sg' ? 0.8 : 1.5"
          />
          <line :x1="xyPad.left + 4" :y1="xyPad.top + 8" :x2="xyPad.left + 16" :y2="xyPad.top + 8" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="4,2" />
          <text :x="xyPad.left + 20" :y="xyPad.top + 11" class="tick-label">sRGB</text>
          <line :x1="xyPad.left + 4" :y1="xyPad.top + 20" :x2="xyPad.left + 16" :y2="xyPad.top + 20" stroke="#8e44ad" stroke-width="1.5" stroke-dasharray="4,2" />
          <text :x="xyPad.left + 20" :y="xyPad.top + 23" class="tick-label">DCI-P3</text>
          <line :x1="xyPad.left + 4" :y1="xyPad.top + 32" :x2="xyPad.left + 16" :y2="xyPad.top + 32" stroke="#2980b9" stroke-width="1.5" stroke-dasharray="4,2" />
          <text :x="xyPad.left + 20" :y="xyPad.top + 35" class="tick-label">BT.2020</text>
        </svg>
      </div>
    </div>

    <!-- Multi-condition comparison -->
    <div class="compare-section" v-if="savedConditions.length > 0">
      <h5>{{ t('Condition Comparison', '조건 비교') }}</h5>
      <div class="compare-table-wrapper">
        <table class="compare-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Si (&mu;m)</th>
              <th>BW (nm)</th>
              <th>{{ t('Illuminant', '광원') }}</th>
              <th>WB</th>
              <th>{{ t('Noise', '노이즈') }}</th>
              <th>Avg {{ deAxisLabel }}</th>
              <th>Max {{ deAxisLabel }}</th>
              <th>&lt;3</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(cond, idx) in savedConditions" :key="idx"
              :class="{ 'best-row': cond.avgDE === bestSavedAvgDE }">
              <td>{{ idx + 1 }}</td>
              <td>{{ cond.si.toFixed(1) }}</td>
              <td>{{ cond.bw }}</td>
              <td>{{ cond.illuminant }}</td>
              <td>{{ cond.wb }}</td>
              <td>{{ cond.noiseLabel }}</td>
              <td class="mono-val">{{ cond.avgDE.toFixed(2) }}</td>
              <td class="mono-val">{{ cond.maxDE.toFixed(2) }}</td>
              <td class="mono-val">{{ cond.excellentPct.toFixed(0) }}%</td>
              <td><button class="del-btn" @click="savedConditions.splice(idx, 1)">&times;</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'
import { tmmCalc, defaultBsiStack, SI_LAYER_IDX } from '../composables/tmm'

const { t } = useLocale()

// ---- Controls ----
const siThickness = ref(3.0)
const cfBandwidth = ref(100)
const chartType = ref<'classic' | 'sg'>('classic')
const patchView = ref<'swatch' | 'heatmap'>('swatch')
const illuminant = ref('D65')
const wbMethod = ref<'none' | 'grayworld' | 'patch'>('none')
const WB_METHODS = [
  { key: 'none' as const, label: 'No WB' },
  { key: 'grayworld' as const, label: 'Gray World' },
  { key: 'patch' as const, label: 'Neutral Patch' },
]
const shotNoise = ref(0)
const readNoise = ref(0)
const optimizing = ref(false)
const optResult = ref<{ si: number; bw: number; avgDE: number } | null>(null)

interface Snapshot { label: string; avgDE: number; maxDE: number; patchDEs: number[] }
const snapshot = ref<Snapshot | null>(null)

interface SavedCondition {
  si: number; bw: number; illuminant: string; wb: string; noiseLabel: string
  avgDE: number; maxDE: number; excellentPct: number
}
const savedConditions = ref<SavedCondition[]>([])

type DEMethod = 'cie76' | 'cie94' | 'ciede2000'
const deMethod = ref<DEMethod>('ciede2000')
const DE_METHODS = [
  { key: 'cie76' as DEMethod, label: 'CIE76' },
  { key: 'cie94' as DEMethod, label: 'CIE94' },
  { key: 'ciede2000' as DEMethod, label: 'CIEDE2000' },
]

// ---- ColorChecker Classic 24 data (standard 24 patches, 7 wavelengths 400-700nm @ 50nm) ----
const CLASSIC_PATCHES: { name: string; srgb: number[]; refl: number[] }[] = [
  { name: 'Dark Skin',     srgb: [115,82,68],   refl: [0.055,0.058,0.069,0.099,0.132,0.143,0.146] },
  { name: 'Light Skin',    srgb: [194,150,130],  refl: [0.092,0.107,0.152,0.191,0.260,0.286,0.275] },
  { name: 'Blue Sky',      srgb: [98,122,157],   refl: [0.117,0.143,0.178,0.193,0.149,0.115,0.108] },
  { name: 'Foliage',       srgb: [87,108,67],    refl: [0.042,0.051,0.085,0.131,0.099,0.068,0.060] },
  { name: 'Blue Flower',   srgb: [133,128,177],  refl: [0.137,0.132,0.128,0.120,0.133,0.154,0.228] },
  { name: 'Bluish Green',  srgb: [103,189,170],  refl: [0.131,0.226,0.318,0.341,0.301,0.232,0.218] },
  { name: 'Orange',        srgb: [214,126,44],   refl: [0.050,0.054,0.063,0.170,0.395,0.413,0.266] },
  { name: 'Purplish Blue', srgb: [80,91,166],    refl: [0.153,0.128,0.091,0.058,0.042,0.060,0.157] },
  { name: 'Moderate Red',  srgb: [193,90,99],    refl: [0.065,0.053,0.058,0.078,0.200,0.305,0.231] },
  { name: 'Purple',        srgb: [94,60,108],    refl: [0.065,0.052,0.040,0.038,0.043,0.065,0.100] },
  { name: 'Yellow Green',  srgb: [157,188,64],   refl: [0.047,0.065,0.172,0.331,0.350,0.237,0.120] },
  { name: 'Orange Yellow', srgb: [224,163,46],   refl: [0.050,0.058,0.093,0.284,0.481,0.465,0.287] },
  { name: 'Blue',          srgb: [56,61,150],    refl: [0.142,0.103,0.056,0.033,0.026,0.032,0.088] },
  { name: 'Green',         srgb: [70,148,73],    refl: [0.035,0.060,0.140,0.175,0.099,0.058,0.044] },
  { name: 'Red',           srgb: [175,54,60],    refl: [0.043,0.036,0.035,0.047,0.153,0.331,0.271] },
  { name: 'Yellow',        srgb: [231,199,31],   refl: [0.042,0.054,0.124,0.397,0.559,0.504,0.316] },
  { name: 'Magenta',       srgb: [187,86,149],   refl: [0.094,0.065,0.067,0.073,0.127,0.219,0.276] },
  { name: 'Cyan',          srgb: [8,133,161],    refl: [0.073,0.139,0.210,0.219,0.137,0.083,0.076] },
  { name: 'White',         srgb: [243,243,242],  refl: [0.875,0.886,0.892,0.894,0.892,0.882,0.870] },
  { name: 'Neutral 8',     srgb: [200,200,200],  refl: [0.570,0.578,0.584,0.586,0.585,0.578,0.572] },
  { name: 'Neutral 6.5',   srgb: [160,160,160],  refl: [0.354,0.362,0.366,0.368,0.366,0.361,0.356] },
  { name: 'Neutral 5',     srgb: [122,122,121],  refl: [0.195,0.200,0.204,0.206,0.205,0.201,0.197] },
  { name: 'Neutral 3.5',   srgb: [85,85,85],     refl: [0.091,0.094,0.096,0.097,0.096,0.094,0.092] },
  { name: 'Black',         srgb: [52,52,52],      refl: [0.032,0.033,0.034,0.034,0.034,0.033,0.032] },
]

// ---- ColorChecker SG 140 data (L*a*b* D50/2°) ----
const SG_LAB: number[][] = [
  [96.04,-0.12,0.31],[53.35,-36.85,15.22],[70.48,-32.43,0.55],[48.52,-28.76,-8.49],[39.43,-16.58,-26.34],
  [55.28,9.35,-34.09],[53.34,14.25,-13.53],[80.57,3.73,-7.71],[50.72,51.66,-14.77],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[46.47,52.54,19.94],[50.76,50.96,27.64],[66.47,35.85,60.44],[62.27,33.28,57.71],
  [72.70,-0.97,69.36],[53.61,10.59,53.90],[44.19,-14.25,38.96],[35.17,-13.17,22.74],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[35.75,60.42,34.16],[40.06,48.08,27.44],[30.87,23.44,22.47],[25.53,13.84,15.33],
  [52.91,-1.37,55.42],[39.77,17.41,47.95],[27.38,-0.63,30.60],[20.17,-0.67,13.69],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[42.35,12.63,-44.77],[52.17,2.07,-30.04],[51.20,49.36,-16.78],[60.21,25.51,2.46],
  [53.13,12.31,17.49],[71.82,-23.83,57.10],[60.94,-29.79,41.53],[49.48,-29.99,22.66],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[38.63,12.21,-45.58],[62.86,36.15,57.33],[71.93,-23.53,57.21],[55.76,-38.34,31.73],
  [40.02,10.05,-44.52],[30.36,22.89,-20.64],[72.39,-27.57,1.35],[49.06,30.18,-4.58],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[42.53,52.21,28.87],[63.85,18.23,18.15],[71.15,11.26,17.58],[78.36,0.41,0.10],
  [64.16,-18.45,-17.62],[60.06,26.25,-19.86],[61.77,0.63,0.51],[50.50,-31.76,-27.86],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[83.68,3.73,79.79],[55.09,-38.56,32.12],[31.77,1.29,-23.25],[81.68,1.02,79.96],
  [52.00,48.63,-14.84],[32.79,18.35,21.36],[72.91,0.69,0.10],[79.66,-1.02,75.21],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[62.67,37.32,68.03],[39.46,49.57,31.76],[72.67,-23.21,58.12],[51.69,-28.81,49.32],
  [51.51,54.10,25.56],[81.22,2.51,80.51],[40.02,50.40,27.55],[30.25,26.53,-22.62],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[43.25,15.29,24.73],[61.50,10.04,17.26],[68.80,12.52,15.81],[78.36,0.41,0.10],
  [57.44,-8.73,-10.74],[50.62,-28.32,-1.05],[48.97,-1.02,-0.31],[23.53,2.10,-2.48],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[53.72,7.22,-25.45],[59.75,-28.34,-26.83],[49.30,-1.96,44.83],[34.95,14.40,24.37],
  [38.93,55.97,29.59],[67.52,-30.41,-0.67],[92.71,0.06,1.45],[32.65,-23.41,0.65],[81.02,-0.17,0.23],
  [65.73,-0.09,0.22],[44.44,23.12,31.85],[38.52,11.43,-45.87],[32.97,16.78,-30.95],[53.53,-41.67,34.15],
  [42.68,15.63,-44.01],[31.50,24.37,19.33],[68.48,0.37,0.37],[60.58,37.49,67.63],[65.73,-0.09,0.22],
  [50.87,-0.06,0.11],[66.84,17.81,-21.56],[42.15,53.06,28.98],[43.26,48.54,-5.29],[42.83,54.65,26.66],
  [62.32,-5.55,63.37],[50.45,50.81,-14.56],[34.63,10.14,27.26],[26.47,15.39,12.83],[50.87,-0.06,0.11],
  [96.04,-0.12,0.31],[49.81,-3.25,49.37],[38.73,-16.30,30.58],[28.61,-10.42,20.02],[21.17,-7.13,10.92],
  [52.72,47.41,56.15],[40.22,42.73,28.43],[30.49,28.63,16.00],[24.54,15.14,6.43],[96.04,-0.12,0.31],
  [96.04,-0.12,0.31],[81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],
  [81.02,-0.17,0.23],[65.73,-0.09,0.22],[50.87,-0.06,0.11],[96.04,-0.12,0.31],[96.04,-0.12,0.31],
]

const SG_NAMES: string[] = (() => {
  const rows = 'ABCDEFGHIJKLMN'
  const names: string[] = []
  for (let r = 0; r < 14; r++)
    for (let c = 1; c <= 10; c++)
      names.push(`${rows[r]}${c}`)
  return names
})()

// ---- Illuminant data (7 wavelengths 400-700nm @ 50nm) ----
const ILLUMINANTS: Record<string, { spd: number[]; label: string }> = {
  D65: { spd: [82.75, 109.35, 117.01, 114.86, 100.0, 90.01, 71.61], label: 'D65 (Daylight)' },
  D50: { spd: [54.65, 100.27, 122.97, 112.59, 100.0, 78.66, 55.78], label: 'D50 (Warm)' },
  A:   { spd: [11.46, 25.62, 51.12, 72.17, 100.0, 125.75, 145.11], label: 'A (Tungsten)' },
  F2:  { spd: [24.82, 65.85, 80.03, 95.72, 100.0, 64.89, 17.22], label: 'F2 (Fluorescent)' },
}

const activeIlluminant = computed(() => ILLUMINANTS[illuminant.value].spd)

// ---- Constants ----
const WL_UM = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
const LN2 = Math.log(2)
const CF_CENTERS = { red: 0.620, green: 0.530, blue: 0.450 }

// ---- QE Spectrum (31 points, 400-700nm @ 10nm) ----
const QE_WL_NM = Array.from({ length: 31 }, (_, i) => 400 + i * 10)
const QE_WL_UM = QE_WL_NM.map(w => w / 1000)

// ---- Color filter & TMM ----
function cfTransmittance(wlUm: number, centerUm: number, fwhmNm: number): number {
  const fwhmUm = fwhmNm / 1000
  return Math.exp(-4 * LN2 * ((wlUm - centerUm) / fwhmUm) ** 2)
}

function computeQEAt(color: 'red' | 'green' | 'blue', wlUm: number, siThick: number): number {
  const stack = defaultBsiStack(color, siThick)
  const result = tmmCalc(stack, 'air', 'sio2', wlUm, 0, 'avg')
  return result.layerA[SI_LAYER_IDX]
}

function sensorResponseAt(color: 'red' | 'green' | 'blue', wlUm: number, siThick: number, cfBw: number): number {
  return computeQEAt(color, wlUm, siThick) * cfTransmittance(wlUm, CF_CENTERS[color], cfBw)
}

function sensorResponse(color: 'red' | 'green' | 'blue', wlUm: number): number {
  return sensorResponseAt(color, wlUm, siThickness.value, cfBandwidth.value)
}

// ---- QE Spectra computed ----
const qeSpectra = computed(() => ({
  r: QE_WL_UM.map(wl => sensorResponse('red', wl)),
  g: QE_WL_UM.map(wl => sensorResponse('green', wl)),
  b: QE_WL_UM.map(wl => sensorResponse('blue', wl)),
}))

// QE chart dimensions
const qeW = 600
const qeH = 180
const qePad = { top: 12, right: 20, bottom: 32, left: 40 }
const qePlotW = qeW - qePad.left - qePad.right
const qePlotH = qeH - qePad.top - qePad.bottom

const qeYMax = computed(() => {
  const all = [...qeSpectra.value.r, ...qeSpectra.value.g, ...qeSpectra.value.b]
  return Math.max(0.1, Math.ceil(Math.max(...all) * 10) / 10)
})

const qeYTicks = computed(() => {
  const ticks: number[] = []
  const step = qeYMax.value <= 0.5 ? 0.1 : 0.2
  for (let v = 0; v <= qeYMax.value + 0.001; v += step) ticks.push(Math.round(v * 100) / 100)
  return ticks
})

const qeXTicks = [400, 450, 500, 550, 600, 650, 700]

function qeYScale(v: number): number {
  return qePad.top + qePlotH - (v / qeYMax.value) * qePlotH
}

function qeXScale(wlNm: number): number {
  return qePad.left + ((wlNm - 400) / 300) * qePlotW
}

function qePathD(data: number[]): string {
  return data.map((v, i) => {
    const x = qePad.left + (i / (data.length - 1)) * qePlotW
    const y = qePad.top + qePlotH - (v / qeYMax.value) * qePlotH
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join('')
}

const qePathR = computed(() => qePathD(qeSpectra.value.r))
const qePathG = computed(() => qePathD(qeSpectra.value.g))
const qePathB = computed(() => qePathD(qeSpectra.value.b))

// ---- sRGB / linear color math ----
function srgbToLinear(c: number): number {
  const s = c / 255
  return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
}

function linearToSrgb(c: number): number {
  const v = Math.max(0, Math.min(1, c))
  const s = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055
  return Math.round(Math.max(0, Math.min(255, s * 255)))
}

const SRGB_TO_XYZ = [
  [0.4124564, 0.3575761, 0.1804375],
  [0.2126729, 0.7151522, 0.0721750],
  [0.0193339, 0.1191920, 0.9503041],
]

function srgbToXYZ(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0]), g = srgbToLinear(srgb[1]), b = srgbToLinear(srgb[2])
  return [
    SRGB_TO_XYZ[0][0] * r + SRGB_TO_XYZ[0][1] * g + SRGB_TO_XYZ[0][2] * b,
    SRGB_TO_XYZ[1][0] * r + SRGB_TO_XYZ[1][1] * g + SRGB_TO_XYZ[1][2] * b,
    SRGB_TO_XYZ[2][0] * r + SRGB_TO_XYZ[2][1] * g + SRGB_TO_XYZ[2][2] * b,
  ]
}

const XYZ_TO_SRGB = [
  [ 3.2404542, -1.5371385, -0.4985314],
  [-0.9692660,  1.8760108,  0.0415560],
  [ 0.0556434, -0.2040259,  1.0572252],
]

function xyzToLinearRgb(xyz: number[]): number[] {
  return [
    XYZ_TO_SRGB[0][0] * xyz[0] + XYZ_TO_SRGB[0][1] * xyz[1] + XYZ_TO_SRGB[0][2] * xyz[2],
    XYZ_TO_SRGB[1][0] * xyz[0] + XYZ_TO_SRGB[1][1] * xyz[1] + XYZ_TO_SRGB[1][2] * xyz[2],
    XYZ_TO_SRGB[2][0] * xyz[0] + XYZ_TO_SRGB[2][1] * xyz[1] + XYZ_TO_SRGB[2][2] * xyz[2],
  ]
}

// ---- XYZ ↔ Lab (D65) ----
const D65_WP = { Xn: 0.9505, Yn: 1.0, Zn: 1.089 }

function labF(t: number): number {
  return t > 0.008856 ? Math.pow(t, 1 / 3) : 7.787 * t + 16 / 116
}

function xyzToLab(xyz: number[]): number[] {
  const fx = labF(xyz[0] / D65_WP.Xn)
  const fy = labF(xyz[1] / D65_WP.Yn)
  const fz = labF(xyz[2] / D65_WP.Zn)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

// ---- deltaE methods ----
function deltaE76(lab1: number[], lab2: number[]): number {
  return Math.sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2)
}

function deltaE94(lab1: number[], lab2: number[]): number {
  const dL = lab1[0] - lab2[0]
  const da = lab1[1] - lab2[1], db = lab1[2] - lab2[2]
  const C1 = Math.sqrt(lab1[1] ** 2 + lab1[2] ** 2)
  const dC = C1 - Math.sqrt(lab2[1] ** 2 + lab2[2] ** 2)
  const dH2 = da ** 2 + db ** 2 - dC ** 2
  const SC = 1 + 0.045 * C1, SH = 1 + 0.015 * C1
  return Math.sqrt(dL ** 2 + (dC / SC) ** 2 + (dH2 > 0 ? dH2 : 0) / SH ** 2)
}

function deltaE2000(lab1: number[], lab2: number[]): number {
  const L1 = lab1[0], a1 = lab1[1], b1 = lab1[2]
  const L2 = lab2[0], a2 = lab2[1], b2 = lab2[2]
  const deg = Math.PI / 180
  const Cab1 = Math.sqrt(a1 ** 2 + b1 ** 2), Cab2 = Math.sqrt(a2 ** 2 + b2 ** 2)
  const CabAvg = (Cab1 + Cab2) / 2, CabAvg7 = CabAvg ** 7
  const G = 0.5 * (1 - Math.sqrt(CabAvg7 / (CabAvg7 + 25 ** 7)))
  const a1p = a1 * (1 + G), a2p = a2 * (1 + G)
  const C1p = Math.sqrt(a1p ** 2 + b1 ** 2), C2p = Math.sqrt(a2p ** 2 + b2 ** 2)
  let h1p = Math.atan2(b1, a1p); if (h1p < 0) h1p += 2 * Math.PI
  let h2p = Math.atan2(b2, a2p); if (h2p < 0) h2p += 2 * Math.PI
  const dLp = L2 - L1, dCp = C2p - C1p
  let dhp: number
  if (C1p * C2p === 0) dhp = 0
  else if (Math.abs(h2p - h1p) <= Math.PI) dhp = h2p - h1p
  else if (h2p - h1p > Math.PI) dhp = h2p - h1p - 2 * Math.PI
  else dhp = h2p - h1p + 2 * Math.PI
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dhp / 2)
  const Lpm = (L1 + L2) / 2, Cpm = (C1p + C2p) / 2
  let hpm: number
  if (C1p * C2p === 0) hpm = h1p + h2p
  else if (Math.abs(h1p - h2p) <= Math.PI) hpm = (h1p + h2p) / 2
  else if (h1p + h2p < 2 * Math.PI) hpm = (h1p + h2p + 2 * Math.PI) / 2
  else hpm = (h1p + h2p - 2 * Math.PI) / 2
  const T = 1 - 0.17 * Math.cos(hpm - 30 * deg) + 0.24 * Math.cos(2 * hpm) + 0.32 * Math.cos(3 * hpm + 6 * deg) - 0.20 * Math.cos(4 * hpm - 63 * deg)
  const SL = 1 + 0.015 * (Lpm - 50) ** 2 / Math.sqrt(20 + (Lpm - 50) ** 2)
  const SC = 1 + 0.045 * Cpm, SH = 1 + 0.015 * Cpm * T
  const Cpm7 = Cpm ** 7, RC = 2 * Math.sqrt(Cpm7 / (Cpm7 + 25 ** 7))
  const dTheta = 30 * deg * Math.exp(-(((hpm / deg - 275) / 25) ** 2))
  const RT = -Math.sin(2 * dTheta) * RC
  return Math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))
}

function calcDeltaE(lab1: number[], lab2: number[]): number {
  switch (deMethod.value) {
    case 'cie94': return deltaE94(lab1, lab2)
    case 'ciede2000': return deltaE2000(lab1, lab2)
    default: return deltaE76(lab1, lab2)
  }
}

const deAxisLabel = computed(() => {
  switch (deMethod.value) {
    case 'cie94': return '\u0394E*94'
    case 'ciede2000': return '\u0394E00'
    default: return '\u0394E*ab'
  }
})

// ---- Lab (D50) → sRGB (D65) via Bradford ----
const D50_WP = { Xn: 0.9642, Yn: 1.0, Zn: 0.8251 }
const BRAD_M = [[ 0.8951, 0.2664,-0.1614],[-0.7502, 1.7135, 0.0367],[ 0.0389,-0.0685, 1.0296]]
const BRAD_MI = [[ 0.9870,-0.1471, 0.1600],[ 0.4323, 0.5184, 0.0493],[-0.0085, 0.0400, 0.9685]]
const D50_XYZ = [0.9642, 1.0, 0.8251]
const D65_XYZ = [0.9505, 1.0, 1.089]

const bradAdapt: number[][] = (() => {
  const coneS = [0, 0, 0].map((_, r) => BRAD_M[r][0] * D50_XYZ[0] + BRAD_M[r][1] * D50_XYZ[1] + BRAD_M[r][2] * D50_XYZ[2])
  const coneD = [0, 0, 0].map((_, r) => BRAD_M[r][0] * D65_XYZ[0] + BRAD_M[r][1] * D65_XYZ[1] + BRAD_M[r][2] * D65_XYZ[2])
  const scale = [[coneD[0]/coneS[0],0,0],[0,coneD[1]/coneS[1],0],[0,0,coneD[2]/coneS[2]]]
  const tmp: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) tmp[i][j] += scale[i][k] * BRAD_M[k][j]
  const result: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) result[i][j] += BRAD_MI[i][k] * tmp[k][j]
  return result
})()

function labFInv(t: number): number {
  return t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787
}

function labToSrgb(lab: number[]): number[] {
  const fy = (lab[0] + 16) / 116, fx = lab[1] / 500 + fy, fz = fy - lab[2] / 200
  const xyzD50 = [D50_WP.Xn * labFInv(fx), D50_WP.Yn * labFInv(fy), D50_WP.Zn * labFInv(fz)]
  const xyzD65 = [0, 1, 2].map(i => bradAdapt[i][0] * xyzD50[0] + bradAdapt[i][1] * xyzD50[1] + bradAdapt[i][2] * xyzD50[2])
  const lin = xyzToLinearRgb(xyzD65)
  return [linearToSrgb(lin[0]), linearToSrgb(lin[1]), linearToSrgb(lin[2])]
}

// ---- Gaussian basis spectral reconstruction ----
function reconstructRefl(srgb: number[]): number[] {
  const r = srgbToLinear(srgb[0]), g = srgbToLinear(srgb[1]), b = srgbToLinear(srgb[2])
  const wlNm = [400, 450, 500, 550, 600, 650, 700]
  return wlNm.map(wl => {
    const bR = Math.exp(-0.5 * ((wl - 610) / 55) ** 2)
    const bG = Math.exp(-0.5 * ((wl - 535) / 50) ** 2)
    const bB = Math.exp(-0.5 * ((wl - 445) / 35) ** 2)
    return Math.max(0.01, Math.min(1, r * bR + g * bG + b * bB))
  })
}

// ---- Seeded PRNG for noise ----
function mulberry32(seed: number): () => number {
  let a = seed | 0
  return () => {
    a = a + 0x6D2B79F5 | 0
    let t = Math.imul(a ^ a >>> 15, 1 | a)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function gaussRng(rng: () => number): number {
  return Math.sqrt(-2 * Math.log(rng() + 1e-10)) * Math.cos(2 * Math.PI * rng())
}

// ---- White Balance ----
function applyWB(sensorRGB: number[][], patches: PatchData[]): number[][] {
  if (wbMethod.value === 'none') return sensorRGB
  if (wbMethod.value === 'grayworld') {
    const n = sensorRGB.length
    const avg = [0, 1, 2].map(ch => sensorRGB.reduce((s, v) => s + v[ch], 0) / n)
    const mean = (avg[0] + avg[1] + avg[2]) / 3
    const gains = avg.map(a => a > 0 ? mean / a : 1)
    return sensorRGB.map(v => [v[0] * gains[0], v[1] * gains[1], v[2] * gains[2]])
  }
  // Neutral patch: find patches where R≈G≈B in reference
  const neutralIdxs: number[] = []
  for (let i = 0; i < patches.length; i++) {
    const s = patches[i].srgb
    if (Math.max(Math.abs(s[0] - s[1]), Math.abs(s[1] - s[2]), Math.abs(s[0] - s[2])) < 10)
      neutralIdxs.push(i)
  }
  if (neutralIdxs.length === 0) return sensorRGB
  const avg = [0, 1, 2].map(ch => neutralIdxs.reduce((s, i) => s + sensorRGB[i][ch], 0) / neutralIdxs.length)
  const mean = (avg[0] + avg[1] + avg[2]) / 3
  const gains = avg.map(a => a > 0 ? mean / a : 1)
  return sensorRGB.map(v => [v[0] * gains[0], v[1] * gains[1], v[2] * gains[2]])
}

// ---- Active patches ----
interface PatchData { name: string; srgb: number[]; refl: number[] }

const activePatches = computed<PatchData[]>(() => {
  if (chartType.value === 'classic') return CLASSIC_PATCHES
  return SG_LAB.map((lab, idx) => {
    const srgb = labToSrgb(lab)
    return { name: SG_NAMES[idx], srgb, refl: reconstructRefl(srgb) }
  })
})

// ---- 3x3 matrix ops ----
function mat3x3Inverse(m: number[][]): number[][] | null {
  const [a,b,c] = m[0], [d,e,f] = m[1], [g,h,i] = m[2]
  const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
  if (Math.abs(det) < 1e-12) return null
  const inv = 1 / det
  return [
    [(e*i-f*h)*inv, (c*h-b*i)*inv, (b*f-c*e)*inv],
    [(f*g-d*i)*inv, (a*i-c*g)*inv, (c*d-a*f)*inv],
    [(d*h-e*g)*inv, (b*g-a*h)*inv, (a*e-b*d)*inv],
  ]
}

function matMul3x3(a: number[][], b: number[][]): number[][] {
  const r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i][j] += a[i][k] * b[k][j]
  return r
}

function matTransposeMulNx3(s: number[][]): number[][] {
  const n = s.length, r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < n; k++) r[i][j] += s[k][i] * s[k][j]
  return r
}

function matSTmulT(s: number[][], tgt: number[][]): number[][] {
  const n = s.length, r: number[][] = [[0,0,0],[0,0,0],[0,0,0]]
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < n; k++) r[i][j] += s[k][i] * tgt[k][j]
  return r
}

function mat3Vec(m: number[][], v: number[]): number[] {
  return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2], m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2], m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]]
}

// ---- Pipeline: compute CCM + sensor data once ----
const pipeline = computed(() => {
  const patches = activePatches.value
  const illum = activeIlluminant.value
  const sR = WL_UM.map(wl => sensorResponse('red', wl))
  const sG = WL_UM.map(wl => sensorResponse('green', wl))
  const sB = WL_UM.map(wl => sensorResponse('blue', wl))
  let sensorRGB = patches.map(patch => {
    let r = 0, g = 0, b = 0
    for (let i = 0; i < 7; i++) { const w = patch.refl[i] * illum[i]; r += w * sR[i]; g += w * sG[i]; b += w * sB[i] }
    return [r, g, b]
  })
  // Apply noise
  if (shotNoise.value > 0 || readNoise.value > 0) {
    const rng = mulberry32(42)
    const fullWell = 10000
    sensorRGB = sensorRGB.map(rgb => rgb.map(v => {
      const signal = v * fullWell
      const shotSigma = Math.sqrt(Math.max(0, signal)) * (shotNoise.value / 100)
      const totalSigma = Math.sqrt(shotSigma ** 2 + readNoise.value ** 2)
      return Math.max(0, (signal + totalSigma * gaussRng(rng)) / fullWell)
    }))
  }
  // Apply WB
  sensorRGB = applyWB(sensorRGB, patches)
  const targetLinear = patches.map(patch => [srgbToLinear(patch.srgb[0]), srgbToLinear(patch.srgb[1]), srgbToLinear(patch.srgb[2])])
  const STS = matTransposeMulNx3(sensorRGB)
  const STSinv = mat3x3Inverse(STS)
  if (!STSinv) return { ccm: null as number[][] | null, sensorRGB, targetLinear }
  const CCM = matMul3x3(STSinv, matSTmulT(sensorRGB, targetLinear))
  return { ccm: CCM, sensorRGB, targetLinear }
})

const ccmMatrix = computed(() => pipeline.value.ccm)

// ---- Main patch results ----
interface PatchResult { name: string; refSrgb: number[]; corrSrgb: number[]; refLab: number[]; corrLab: number[]; deltaE: number }

const patchResults = computed<PatchResult[]>(() => {
  const patches = activePatches.value
  const { ccm, sensorRGB } = pipeline.value
  if (!ccm) return patches.map(p => ({ name: p.name, refSrgb: p.srgb, corrSrgb: [128,128,128], refLab: [50,0,0], corrLab: [50,0,0], deltaE: 99 }))
  return patches.map((patch, idx) => {
    const corrLinear = mat3Vec(ccm, sensorRGB[idx])
    const corrSrgb = [linearToSrgb(corrLinear[0]), linearToSrgb(corrLinear[1]), linearToSrgb(corrLinear[2])]
    const refLab = xyzToLab(srgbToXYZ(patch.srgb))
    const cc = corrLinear.map(v => Math.max(0, Math.min(1, v)))
    const corrXYZ = [0, 1, 2].map(r => SRGB_TO_XYZ[r][0]*cc[0] + SRGB_TO_XYZ[r][1]*cc[1] + SRGB_TO_XYZ[r][2]*cc[2])
    const corrLab = xyzToLab(corrXYZ)
    return { name: patch.name, refSrgb: patch.srgb, corrSrgb, refLab, corrLab, deltaE: calcDeltaE(refLab, corrLab) }
  })
})

const avgDeltaE = computed(() => { const r = patchResults.value; return r.reduce((s, p) => s + p.deltaE, 0) / r.length })
const maxDeltaE = computed(() => Math.max(...patchResults.value.map(p => p.deltaE)))
const excellentPct = computed(() => { const r = patchResults.value; return (r.filter(p => p.deltaE < 3).length / r.length) * 100 })

// ---- Helpers ----
function rgbStr(rgb: number[]): string { return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})` }

function deColor(de: number): string {
  if (de < 3) return '#27ae60'
  if (de < 6) return '#e67e22'
  return '#e74c3c'
}

function deHeatBg(de: number): string {
  const hue = Math.max(0, 120 * (1 - Math.min(de, 12) / 12))
  return `hsl(${hue}, 65%, 72%)`
}

function deHeatText(de: number): string {
  return de > 9 ? '#fff' : '#222'
}

// ---- DeltaE bar chart ----
const chartW = 600
const chartH = 200
const pad = { top: 16, right: 30, bottom: 60, left: 40 }
const plotW = chartW - pad.left - pad.right
const plotH = chartH - pad.top - pad.bottom

const yMax = computed(() => Math.max(8, Math.ceil(maxDeltaE.value + 2)))

const yTicks = computed(() => {
  const ticks: number[] = []
  const step = yMax.value <= 12 ? 2 : 5
  for (let v = 0; v <= yMax.value; v += step) ticks.push(v)
  return ticks
})

function yScale(v: number): number { return pad.top + plotH - (v / yMax.value) * plotH }

const barGap = computed(() => chartType.value === 'sg' ? 0.5 : 2)
const barWidth = computed(() => Math.max(2, (plotW - barGap.value * (activePatches.value.length - 1)) / activePatches.value.length))

function barX(idx: number): number { return pad.left + idx * (barWidth.value + barGap.value) }

const chartHover = ref<{ tx: number; name: string; de: number } | null>(null)
const deChartSvg = ref<SVGSVGElement | null>(null)

function onChartMouseMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const mouseX = (event.clientX - rect.left) * (chartW / rect.width)
  const idx = Math.floor((mouseX - pad.left) / (barWidth.value + barGap.value))
  if (idx >= 0 && idx < patchResults.value.length) {
    const patch = patchResults.value[idx]
    const bx = barX(idx) + barWidth.value / 2
    chartHover.value = { tx: bx + 150 > chartW - pad.right ? bx - 150 : bx + 10, name: patch.name, de: patch.deltaE }
  } else {
    chartHover.value = null
  }
}

// ---- Export CSV ----
function exportCsv() {
  const method = deMethod.value.toUpperCase()
  const header = `Patch,Ref_R,Ref_G,Ref_B,Corr_R,Corr_G,Corr_B,${method}\n`
  const rows = patchResults.value.map(p =>
    `${p.name},${p.refSrgb.join(',')},${p.corrSrgb.join(',')},${p.deltaE.toFixed(4)}`
  ).join('\n')
  const blob = new Blob([header + rows], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `colorchecker-${chartType.value}-${method}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

// ---- Export PNG (canvas redraw) ----
function exportPng() {
  const results = patchResults.value
  const dpr = 2
  const canvas = document.createElement('canvas')
  canvas.width = chartW * dpr
  canvas.height = chartH * dpr
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)

  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, chartW, chartH)

  // Grid
  ctx.strokeStyle = '#e0e0e0'
  ctx.lineWidth = 0.5
  ctx.setLineDash([3, 3])
  for (const tick of yTicks.value) {
    ctx.beginPath(); ctx.moveTo(pad.left, yScale(tick)); ctx.lineTo(pad.left + plotW, yScale(tick)); ctx.stroke()
  }
  ctx.setLineDash([])

  // Threshold lines
  ctx.setLineDash([6, 3])
  ctx.lineWidth = 1
  ctx.strokeStyle = '#27ae60'
  ctx.beginPath(); ctx.moveTo(pad.left, yScale(3)); ctx.lineTo(pad.left + plotW, yScale(3)); ctx.stroke()
  ctx.strokeStyle = '#e67e22'
  ctx.beginPath(); ctx.moveTo(pad.left, yScale(6)); ctx.lineTo(pad.left + plotW, yScale(6)); ctx.stroke()
  ctx.setLineDash([])

  // Axes
  ctx.strokeStyle = '#333'
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH); ctx.lineTo(pad.left + plotW, pad.top + plotH); ctx.stroke()

  // Y ticks
  ctx.fillStyle = '#666'
  ctx.font = '9px sans-serif'
  ctx.textAlign = 'right'
  for (const tick of yTicks.value) ctx.fillText(String(tick), pad.left - 6, yScale(tick) + 3)

  // Bars
  for (let i = 0; i < results.length; i++) {
    const p = results[i]
    ctx.fillStyle = deColor(p.deltaE)
    ctx.globalAlpha = 0.8
    const by = yScale(p.deltaE)
    ctx.fillRect(barX(i), by, barWidth.value, Math.max(0, pad.top + plotH - by))
  }
  ctx.globalAlpha = 1

  // Axis title
  ctx.save()
  ctx.fillStyle = '#333'
  ctx.font = 'bold 10px sans-serif'
  ctx.textAlign = 'center'
  ctx.translate(10, pad.top + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.fillText(deAxisLabel.value, 0, 0)
  ctx.restore()

  // X labels for classic
  if (chartType.value === 'classic') {
    ctx.fillStyle = '#666'
    ctx.font = '8px sans-serif'
    for (let i = 0; i < results.length; i++) {
      ctx.save()
      const x = barX(i) + barWidth.value / 2
      const y = pad.top + plotH + 10
      ctx.translate(x, y)
      ctx.rotate(-Math.PI / 4)
      ctx.textAlign = 'end'
      ctx.fillText(results[i].name.substring(0, 8), 0, 0)
      ctx.restore()
    }
  }

  // Summary text
  ctx.fillStyle = '#333'
  ctx.font = 'bold 9px sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText(`Avg ${deAxisLabel.value}=${avgDeltaE.value.toFixed(2)}  Max=${maxDeltaE.value.toFixed(2)}  <3: ${excellentPct.value.toFixed(0)}%`, pad.left, pad.top - 2)

  canvas.toBlob(blob => {
    if (!blob) return
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `colorchecker-${chartType.value}-${deMethod.value}.png`
    a.click()
    URL.revokeObjectURL(url)
  })
}

// ---- Snapshot ----
function takeSnapshot() {
  snapshot.value = {
    label: `Si=${siThickness.value}μm, BW=${cfBandwidth.value}nm, ${illuminant.value}`,
    avgDE: avgDeltaE.value,
    maxDE: maxDeltaE.value,
    patchDEs: patchResults.value.map(p => p.deltaE),
  }
}

// ---- Auto Optimization ----
function computeAvgDE(patches: PatchData[], illum: number[], siThick: number, cfBw: number): number {
  const sR = WL_UM.map(wl => sensorResponseAt('red', wl, siThick, cfBw))
  const sG = WL_UM.map(wl => sensorResponseAt('green', wl, siThick, cfBw))
  const sB = WL_UM.map(wl => sensorResponseAt('blue', wl, siThick, cfBw))
  const sensorRGB = patches.map(p => {
    let r = 0, g = 0, b = 0
    for (let i = 0; i < 7; i++) { const w = p.refl[i] * illum[i]; r += w * sR[i]; g += w * sG[i]; b += w * sB[i] }
    return [r, g, b]
  })
  const targetLinear = patches.map(p => [srgbToLinear(p.srgb[0]), srgbToLinear(p.srgb[1]), srgbToLinear(p.srgb[2])])
  const STSinv = mat3x3Inverse(matTransposeMulNx3(sensorRGB))
  if (!STSinv) return 99
  const CCM = matMul3x3(STSinv, matSTmulT(sensorRGB, targetLinear))
  let total = 0
  for (let idx = 0; idx < patches.length; idx++) {
    const corrLinear = mat3Vec(CCM, sensorRGB[idx])
    const refLab = xyzToLab(srgbToXYZ(patches[idx].srgb))
    const cc = corrLinear.map(v => Math.max(0, Math.min(1, v)))
    const corrXYZ = [0, 1, 2].map(r => SRGB_TO_XYZ[r][0] * cc[0] + SRGB_TO_XYZ[r][1] * cc[1] + SRGB_TO_XYZ[r][2] * cc[2])
    total += calcDeltaE(refLab, xyzToLab(corrXYZ))
  }
  return total / patches.length
}

function runOptimization() {
  optimizing.value = true
  optResult.value = null
  requestAnimationFrame(() => {
    const patches = activePatches.value
    const illum = activeIlluminant.value
    let bestAvg = Infinity, bestSi = 3.0, bestBw = 100
    for (let si = 10; si <= 50; si += 2) {
      for (let bw = 50; bw <= 150; bw += 5) {
        const avg = computeAvgDE(patches, illum, si / 10, bw)
        if (avg < bestAvg) { bestAvg = avg; bestSi = si / 10; bestBw = bw }
      }
    }
    // Fine pass
    const siMin = Math.max(10, Math.round(bestSi * 10) - 2)
    const siMax = Math.min(50, Math.round(bestSi * 10) + 2)
    const bwMin = Math.max(50, bestBw - 5)
    const bwMax = Math.min(150, bestBw + 5)
    for (let si = siMin; si <= siMax; si++) {
      for (let bw = bwMin; bw <= bwMax; bw += 5) {
        const avg = computeAvgDE(patches, illum, si / 10, bw)
        if (avg < bestAvg) { bestAvg = avg; bestSi = si / 10; bestBw = bw }
      }
    }
    siThickness.value = bestSi
    cfBandwidth.value = bestBw
    optResult.value = { si: bestSi, bw: bestBw, avgDE: bestAvg }
    optimizing.value = false
  })
}

// ---- a*b* Chromaticity Diagram ----
const abW = 360
const abH = 360
const abPad = { top: 12, right: 12, bottom: 32, left: 36 }
const abPlotS = Math.min(abW - abPad.left - abPad.right, abH - abPad.top - abPad.bottom)

const abRange = computed(() => {
  const results = patchResults.value
  let m = 40
  for (const p of results) {
    m = Math.max(m, Math.abs(p.refLab[1]), Math.abs(p.refLab[2]), Math.abs(p.corrLab[1]), Math.abs(p.corrLab[2]))
  }
  return Math.ceil(m / 20) * 20 + 10
})

const abTicks = computed(() => {
  const r = abRange.value
  const step = r <= 60 ? 20 : 40
  const ticks: number[] = []
  for (let v = -r; v <= r; v += step) ticks.push(v)
  return ticks
})

function abXScale(a: number): number {
  return abPad.left + ((a + abRange.value) / (2 * abRange.value)) * abPlotS
}

function abYScale(b: number): number {
  return abPad.top + abPlotS - ((b + abRange.value) / (2 * abRange.value)) * abPlotS
}

// ---- CIE xy Chromaticity Diagram ----
const xyW = 400, xyH = 400
const xyPad = { top: 12, right: 12, bottom: 32, left: 36 }
const xyPlotS = Math.min(xyW - xyPad.left - xyPad.right, xyH - xyPad.top - xyPad.bottom)
const xyTicks = [0, 0.2, 0.4, 0.6, 0.8]

function xyXScale(x: number): number { return xyPad.left + (x / 0.85) * xyPlotS }
function xyYScale(y: number): number { return xyPad.top + xyPlotS - (y / 0.85) * xyPlotS }

function xyzToXy(xyz: number[]): { x: number; y: number } {
  const sum = xyz[0] + xyz[1] + xyz[2]
  return sum > 0 ? { x: xyz[0] / sum, y: xyz[1] / sum } : { x: 0.3127, y: 0.3290 }
}

const SRGB_PRIM = [[0.6400,0.3300],[0.3000,0.6000],[0.1500,0.0600]]
const P3_PRIM = [[0.6800,0.3200],[0.2650,0.6900],[0.1500,0.0600]]
const BT2020_PRIM = [[0.7080,0.2920],[0.1700,0.7970],[0.1310,0.0460]]

const SPECTRAL_LOCUS: [number,number][] = [
  [0.1741,0.0050],[0.1740,0.0050],[0.1733,0.0048],[0.1726,0.0048],
  [0.1714,0.0051],[0.1689,0.0069],[0.1644,0.0109],[0.1566,0.0177],
  [0.1440,0.0297],[0.1241,0.0578],[0.0913,0.1327],[0.0454,0.2950],
  [0.0082,0.5384],[0.0139,0.7502],[0.0743,0.8338],[0.1547,0.8059],
  [0.2296,0.7543],[0.3016,0.6923],[0.3731,0.6245],[0.4441,0.5547],
  [0.5125,0.4866],[0.5752,0.4242],[0.6270,0.3725],[0.6658,0.3340],
  [0.6915,0.3083],[0.7079,0.2920],[0.7190,0.2809],[0.7260,0.2740],
  [0.7300,0.2700],[0.7320,0.2680],[0.7334,0.2666],[0.7347,0.2653],
]

const spectralLocusPath = computed(() => {
  const pts = SPECTRAL_LOCUS.map(([x, y]) => `${xyXScale(x).toFixed(1)},${xyYScale(y).toFixed(1)}`)
  return `M${pts.join('L')}Z`
})

function gamutTriangle(prim: number[][]): string {
  return prim.map(([x, y]) => `${xyXScale(x).toFixed(1)},${xyYScale(y).toFixed(1)}`).join(' ')
}

const xyRefPoints = computed(() => patchResults.value.map(p => {
  const { x, y } = xyzToXy(srgbToXYZ(p.refSrgb))
  return { x, y, rgb: p.refSrgb }
}))

const xyCorrPoints = computed(() => patchResults.value.map(p => {
  const { x, y } = xyzToXy(srgbToXYZ(p.corrSrgb))
  return { x, y, rgb: p.corrSrgb }
}))

function pointInTriangle(px: number, py: number, v0: number[], v1: number[], v2: number[]): boolean {
  const d1 = (px - v1[0]) * (v0[1] - v1[1]) - (v0[0] - v1[0]) * (py - v1[1])
  const d2 = (px - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (py - v2[1])
  const d3 = (px - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (py - v0[1])
  return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0))
}

const gamutStats = computed(() => {
  const pts = xyRefPoints.value
  if (pts.length === 0) return [
    { name: 'sRGB', color: '#e74c3c', pct: 0 },
    { name: 'DCI-P3', color: '#8e44ad', pct: 0 },
    { name: 'BT.2020', color: '#2980b9', pct: 0 },
  ]
  const coverage = (prim: number[][]) => (pts.filter(p => pointInTriangle(p.x, p.y, prim[0], prim[1], prim[2])).length / pts.length) * 100
  return [
    { name: 'sRGB', color: '#e74c3c', pct: coverage(SRGB_PRIM) },
    { name: 'DCI-P3', color: '#8e44ad', pct: coverage(P3_PRIM) },
    { name: 'BT.2020', color: '#2980b9', pct: coverage(BT2020_PRIM) },
  ]
})

// ---- Save Condition ----
function saveCondition() {
  savedConditions.value.push({
    si: siThickness.value,
    bw: cfBandwidth.value,
    illuminant: illuminant.value,
    wb: wbMethod.value,
    noiseLabel: shotNoise.value > 0 || readNoise.value > 0 ? `S${shotNoise.value}/R${readNoise.value}` : 'Off',
    avgDE: avgDeltaE.value,
    maxDE: maxDeltaE.value,
    excellentPct: excellentPct.value,
  })
}

const bestSavedAvgDE = computed(() => {
  if (savedConditions.value.length === 0) return Infinity
  return Math.min(...savedConditions.value.map(c => c.avgDE))
})
</script>

<style scoped>
.ca-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}
.ca-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}
.ca-container h5 {
  margin: 0 0 8px 0;
  font-size: 0.95em;
  color: var(--vp-c-text-1);
}
.patch-count {
  font-weight: 400;
  color: var(--vp-c-text-3);
  font-size: 0.85em;
}
.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}
.controls-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
  align-items: flex-end;
}
.slider-group {
  flex: 1;
  min-width: 200px;
}
.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
}
.ctrl-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: var(--vp-c-divider);
  outline: none;
}
.ctrl-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.chart-toggle {
  display: flex;
  gap: 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.toggle-btn {
  padding: 6px 14px;
  font-size: 0.82em;
  font-weight: 500;
  border: none;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}
.toggle-btn.active {
  background: var(--vp-c-brand-1);
  color: #fff;
}
.toggle-btn:not(.active):hover {
  background: var(--vp-c-bg-soft);
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.result-card {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.result-label {
  font-size: 0.8em;
  color: var(--vp-c-text-2);
  margin-bottom: 4px;
}
.result-value {
  font-weight: 600;
  font-size: 1.0em;
  font-family: var(--vp-font-family-mono);
}
.result-value.highlight {
  color: var(--vp-c-brand-1);
}

/* CCM table */
.ccm-section {
  margin-bottom: 20px;
}
.ccm-table {
  border-collapse: collapse;
  font-family: var(--vp-font-family-mono);
  font-size: 0.82em;
  margin: 0 auto;
}
.ccm-table th, .ccm-table td {
  border: 1px solid var(--vp-c-divider);
  padding: 6px 12px;
  text-align: center;
}
.ccm-table th {
  background: var(--vp-c-bg);
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.ccm-table td {
  background: var(--vp-c-bg-soft);
}
.ccm-diag {
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

/* Patch grid */
.patch-section {
  margin-bottom: 20px;
}
.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.section-header h5 {
  margin: 0;
}
.view-toggle {
  flex-shrink: 0;
}
.view-toggle .toggle-btn {
  padding: 4px 10px;
  font-size: 0.75em;
}
.patch-grid-classic {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 6px;
}
.patch-grid-sg {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 3px;
}
@media (max-width: 640px) {
  .patch-grid-classic {
    grid-template-columns: repeat(4, 1fr);
  }
  .patch-grid-sg {
    grid-template-columns: repeat(10, 1fr);
    gap: 2px;
  }
}
.patch-cell {
  text-align: center;
}
.patch-swatch {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  display: flex;
  flex-direction: column;
}
.patch-grid-sg .patch-swatch {
  border-radius: 3px;
}
.patch-ref {
  flex: 1;
}
.patch-sensor {
  flex: 1;
}
.patch-name {
  font-size: 0.6em;
  color: var(--vp-c-text-3);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.patch-de {
  font-size: 0.75em;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.patch-de-sg {
  font-size: 0.55em;
}

/* Heatmap */
.heatmap-cell {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  display: flex;
  align-items: center;
  justify-content: center;
}
.patch-grid-sg .heatmap-cell {
  border-radius: 3px;
}
.heatmap-val {
  font-size: 0.72em;
  font-weight: 700;
  font-family: var(--vp-font-family-mono);
}
.patch-grid-sg .heatmap-val {
  font-size: 0.5em;
}

/* Chart */
.chart-section {
  margin-bottom: 20px;
}
.svg-wrapper {
  margin-top: 4px;
}
.de-svg {
  width: 100%;
  max-width: 600px;
  display: block;
  margin: 0 auto;
}
.tick-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
}
.axis-title {
  font-size: 10px;
  fill: var(--vp-c-text-2);
  font-weight: 600;
}
.ref-label {
  font-size: 8px;
  font-weight: 600;
}
.tooltip-text {
  font-size: 9px;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

/* Illuminant select */
.select-group {
  display: flex;
  align-items: center;
  gap: 8px;
}
.select-label {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}
.ctrl-select {
  padding: 5px 10px;
  font-size: 0.82em;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  cursor: pointer;
}

/* Action row */
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.action-btn {
  padding: 6px 16px;
  font-size: 0.82em;
  font-weight: 500;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}
.action-btn:hover {
  background: var(--vp-c-brand-1);
  color: #fff;
  border-color: var(--vp-c-brand-1);
}
.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.action-btn:disabled:hover {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  border-color: var(--vp-c-divider);
}

/* Optimization result */
.opt-result {
  padding: 8px 14px;
  margin-bottom: 12px;
  background: var(--vp-c-brand-soft);
  border-radius: 8px;
  font-size: 0.85em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-brand-1);
}

/* Snapshot bar */
.snapshot-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 14px;
  margin-bottom: 12px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  font-size: 0.85em;
}
.snap-label {
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
}
.snap-stat {
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}
.snap-delta {
  font-weight: 700;
  font-family: var(--vp-font-family-mono);
}

/* a*b* diagram */
.ab-svg {
  width: 100%;
  max-width: 400px;
  display: block;
  margin: 0 auto;
}

/* Gamut stats */
.gamut-stats {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.gamut-tag {
  padding: 4px 10px;
  font-size: 0.82em;
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
  border: 2px solid;
  border-radius: 6px;
  color: var(--vp-c-text-1);
}

/* Comparison table */
.compare-section {
  margin-top: 20px;
}
.compare-table-wrapper {
  overflow-x: auto;
}
.compare-table {
  border-collapse: collapse;
  font-size: 0.82em;
  width: 100%;
  min-width: 600px;
}
.compare-table th, .compare-table td {
  border: 1px solid var(--vp-c-divider);
  padding: 6px 10px;
  text-align: center;
  white-space: nowrap;
}
.compare-table th {
  background: var(--vp-c-bg);
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.compare-table td {
  background: var(--vp-c-bg-soft);
}
.mono-val {
  font-family: var(--vp-font-family-mono);
  font-weight: 600;
}
.best-row td {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}
.del-btn {
  border: none;
  background: none;
  color: var(--vp-c-text-3);
  font-size: 1.1em;
  cursor: pointer;
  padding: 0 4px;
}
.del-btn:hover {
  color: #e74c3c;
}
</style>
