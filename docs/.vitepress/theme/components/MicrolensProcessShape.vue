<template>
  <div :class="['mlp-container', 'sim-fs-root', { 'sim-fullscreen': isFullscreen }]">
    <h4>{{ t('Microlens Process Shape Predictor', '마이크로렌즈 공정 형상 예측기') }}</h4>
    <button
      type="button"
      class="sim-fs-btn"
      :aria-label="t('Toggle fullscreen', '전체화면 전환')"
      :aria-pressed="isFullscreen"
      :title="t('Toggle fullscreen', '전체화면 전환')"
      @click="toggleFullscreen"
    >{{ isFullscreen ? '\u00d7' : '\u26f6' }}</button>
    <p class="component-description">
      {{ t(
        'Explore how layout gap, reflow budget, and etch-transfer settings can move a CIS microlens toward a final gap, height, and profile.',
        '레이아웃 gap, reflow budget, etch-transfer 조건이 CIS 마이크로렌즈의 최종 gap, 높이, profile에 어떤 방향으로 작용하는지 살펴봅니다.'
      ) }}
    </p>

    <div class="sim-fs-controls">
    <div class="control-section">
      <div class="control-heading">{{ t('Layout and resist', '레이아웃 및 resist') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Pixel pitch', '픽셀 피치') }}: <strong>{{ pitch.toFixed(2) }} um</strong></label>
          <input type="range" min="0.6" max="3.0" step="0.02" v-model.number="pitch" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Mask island width', '마스크 island 폭') }}: <strong>{{ maskWidth.toFixed(2) }} um</strong></label>
          <input type="range" :min="minMaskWidth" :max="maxMaskWidth" step="0.01" v-model.number="maskWidth" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Resist thickness', 'Resist 두께') }}: <strong>{{ resistThickness.toFixed(2) }} um</strong></label>
          <input type="range" min="0.12" max="1.20" step="0.01" v-model.number="resistThickness" class="ctrl-range" />
        </div>
      </div>
      <div class="controls-row">
        <div class="slider-group compact">
          <label>{{ t('Aperture shape', '개구 형상') }}</label>
          <select v-model="apertureShape" class="ctrl-select">
            <option value="circular">{{ t('Circular', '원형') }}</option>
            <option value="rounded-square">{{ t('Rounded square', '라운드 사각') }}</option>
            <option value="square">{{ t('Square-like', '사각형 근사') }}</option>
          </select>
        </div>
        <div class="slider-group compact">
          <label>{{ t('Lens unit layout', '렌즈 unit 배치') }}</label>
          <select v-model="layoutPreset" class="ctrl-select">
            <option value="all-1x1">{{ t('All 1x1', '전체 1x1') }}</option>
            <option value="all-2x1">{{ t('All 2x1 (Sony 2PD)', '전체 2x1 (Sony 2PD)') }}</option>
            <option value="all-1x2">{{ t('All 1x2', '전체 1x2') }}</option>
            <option value="all-2x2">{{ t('All 2x2 (Tetracell OCL)', '전체 2x2 (Tetracell OCL)') }}</option>
            <option value="mixed-2x2-pdaf">{{ t('Mixed 2x2 OCL + 1x1', '혼합 2x2 OCL + 1x1') }}</option>
            <option value="sparse-2x1-pdaf">{{ t('Sparse 2x1 PDAF', 'Sparse 2x1 PDAF') }}</option>
            <option value="custom">{{ t('Custom 4x4', 'Custom 4x4') }}</option>
          </select>
        </div>
      </div>
      <div v-if="layoutPreset === 'custom'" class="custom-grid-row">
        <div class="custom-grid-help">
          <strong>{{ t('Custom 4x4 editor', 'Custom 4x4 편집기') }}</strong>
          <span>{{ t('Click two adjacent cells to merge into a 2x1, 1x2, or 2x2 lens. Double-click to split back to 1x1.', '인접 셀 두 개를 차례로 클릭하면 2x1/1x2/2x2 렌즈로 병합, 더블클릭하면 1x1로 분리됩니다.') }}</span>
        </div>
        <svg
          :viewBox="`0 0 ${customGridSize} ${customGridSize}`"
          class="custom-grid-svg"
          role="img"
          :aria-label="t('4x4 lens unit editor', '4x4 렌즈 unit 편집기')"
        >
          <g v-for="(gid, idx) in cellGrid" :key="'gcell-' + idx">
            <rect
              :x="(idx % GRID_N) * customCellSize + 1"
              :y="Math.floor(idx / GRID_N) * customCellSize + 1"
              :width="customCellSize - 2"
              :height="customCellSize - 2"
              rx="3"
              :fill="customCellFill(gid, idx)"
              :stroke="selectedCell === idx ? 'var(--vp-c-brand-1)' : 'var(--vp-c-divider)'"
              :stroke-width="selectedCell === idx ? 2.5 : 1"
              class="custom-cell"
              @click="onCellClick(idx)"
              @dblclick="onCellDoubleClick(idx)"
            />
            <text
              :x="(idx % GRID_N) * customCellSize + customCellSize / 2"
              :y="Math.floor(idx / GRID_N) * customCellSize + customCellSize / 2 + 5"
              text-anchor="middle"
              class="custom-cell-label"
            >{{ groupKindForCell(idx) }}</text>
          </g>
        </svg>
        <span v-if="!isLayoutValid" class="custom-invalid">
          {{ t('Only 1x1, 2x1, 1x2, 2x2 rectangles are allowed.', '1x1, 2x1, 1x2, 2x2 사각형만 허용됩니다.') }}
        </span>
      </div>
    </div>

    <div class="control-section">
      <div class="control-heading">{{ t('Thermal reflow', 'Thermal reflow') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Reflow temperature', 'Reflow 온도') }}: <strong>{{ reflowTemp }} C</strong></label>
          <input type="range" min="125" max="220" step="1" v-model.number="reflowTemp" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Reflow time', 'Reflow 시간') }}: <strong>{{ reflowTime }} s</strong></label>
          <input type="range" min="20" max="300" step="5" v-model.number="reflowTime" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Lens index', '렌즈 굴절률') }}: <strong>{{ lensIndex.toFixed(2) }}</strong></label>
          <input type="range" min="1.40" max="1.75" step="0.01" v-model.number="lensIndex" class="ctrl-range" />
        </div>
      </div>
    </div>

    <div class="control-section">
      <div class="control-heading">{{ t('Etch transfer', 'Etch transfer') }}</div>
      <div class="controls-row">
        <div class="slider-group">
          <label>{{ t('Mask thickness', 'Mask 두께') }}: <strong>{{ maskThickness.toFixed(2) }} um</strong></label>
          <input type="range" min="0.10" max="1.00" step="0.01" v-model.number="maskThickness" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Polymerizing gas', 'Polymerizing gas') }}: <strong>{{ polymerGas }}%</strong></label>
          <input type="range" min="0" max="100" step="1" v-model.number="polymerGas" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Etch time', 'Etch 시간') }}: <strong>{{ etchTime }} s</strong></label>
          <input type="range" min="0" max="140" step="1" v-model.number="etchTime" class="ctrl-range" />
        </div>
      </div>
    </div>

    <details class="calibration-panel">
      <summary>{{ t('Calibration coefficients', '보정 계수') }}</summary>
      <p>
        {{ t(
          'Use these multipliers when fitting the surrogate to AFM/SEM or DOE data. Keep them near 1.0 until measured profiles justify a process-specific correction.',
          'AFM/SEM 또는 DOE 데이터에 surrogate를 맞출 때 쓰는 multiplier입니다. 실측 profile이 공정별 보정을 뒷받침하기 전에는 1.0 근처에서 사용하세요.'
        ) }}
      </p>
      <div class="controls-row calibration-controls">
        <div class="slider-group">
          <label>{{ t('Reflow spread gain', 'Reflow spread gain') }}: <strong>{{ reflowSpreadGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.60" max="1.60" step="0.02" v-model.number="reflowSpreadGain" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Volume retention gain', 'Volume retention gain') }}: <strong>{{ volumeRetentionGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.80" max="1.08" step="0.01" v-model.number="volumeRetentionGain" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Lateral etch gain', 'Lateral etch gain') }}: <strong>{{ lateralEtchGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.50" max="1.80" step="0.02" v-model.number="lateralEtchGain" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Vertical loss gain', 'Vertical loss gain') }}: <strong>{{ verticalLossGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.40" max="1.80" step="0.02" v-model.number="verticalLossGain" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Proximity coupling gain', '인접 결합 gain') }}: <strong>{{ proximityCouplingGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.00" max="2.00" step="0.05" v-model.number="proximityCouplingGain" class="ctrl-range" />
        </div>
        <div class="slider-group">
          <label>{{ t('Microloading gain', 'Microloading gain') }}: <strong>{{ microloadingGain.toFixed(2) }}x</strong></label>
          <input type="range" min="0.00" max="2.00" step="0.05" v-model.number="microloadingGain" class="ctrl-range" />
        </div>
      </div>
    </details>
    </div>

    <div class="sim-fs-view">
    <div class="metric-grid">
      <div class="metric-card">
        <span>{{ t('Initial gap', '초기 gap') }}</span>
        <strong>{{ initialGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('After reflow', 'Reflow 후 gap') }}</span>
        <strong>{{ reflowGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card accent">
        <span>{{ t('Final gap', '최종 gap') }}</span>
        <strong>{{ finalGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card accent">
        <span>{{ t('Final height', '최종 높이') }}</span>
        <strong>{{ finalHeight.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('ROC at vertex', 'Vertex ROC') }}</span>
        <strong>{{ roc.toFixed(2) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('f-number', 'f-number') }}</span>
        <strong>f/{{ fNumber.toFixed(2) }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Fill factor', 'Fill factor') }}</span>
        <strong>{{ fillFactor.toFixed(1) }}%</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Height retention', '높이 보존율') }}</span>
        <strong>{{ heightRetention.toFixed(0) }}%</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Zero-gap etch', 'Zero-gap etch') }}</span>
        <strong>{{ zeroGapEtchTimeLabel }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Profile exponent', 'Profile exponent') }}</span>
        <strong>{{ profilePower.toFixed(2) }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Aspect ratio (WX:WY)', '장단축비 (WX:WY)') }}</span>
        <strong>{{ aspectRatio.toFixed(2) }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Worst reflow gap', '최악 reflow gap') }}</span>
        <strong>{{ worstReflowGap.toFixed(3) }} um</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Lens unit', '렌즈 unit') }}</span>
        <strong>{{ representativeGroup.kind }}</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Lateral rate', 'Lateral rate') }}</span>
        <strong>{{ (lateralEtchRate * 1000).toFixed(2) }} nm/s</strong>
      </div>
      <div class="metric-card">
        <span>{{ t('Height-loss rate', 'Height-loss rate') }}</span>
        <strong>{{ (verticalLossRate * 100).toFixed(2) }}%/s</strong>
      </div>
    </div>

    <div class="status-row">
      <span v-for="flag in processFlags" :key="flag.text" :class="['status-pill', flag.tone]">
        {{ flag.text }}
      </span>
    </div>

    <div class="tab-row">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        type="button"
        :class="['tab-btn', { active: viewMode === tab.key }]"
        :aria-pressed="viewMode === tab.key"
        @click="viewMode = tab.key"
      >
        {{ t(tab.en, tab.ko) }}
      </button>
    </div>

    <div class="plot-panel">
      <svg
        v-if="viewMode === 'section'"
        :viewBox="`0 0 ${sectionW} ${sectionH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Microlens process cross-section', '마이크로렌즈 공정 단면')"
      >
        <rect x="0" y="0" :width="sectionW" :height="sectionH" fill="var(--vp-c-bg)" />
        <line
          v-for="tick in sectionXTicks"
          :key="'sxg-' + tick"
          :x1="sectionXScale(tick)"
          :y1="sectionPad.top"
          :x2="sectionXScale(tick)"
          :y2="sectionH - sectionPad.bottom"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line
          v-for="tick in sectionYTicks"
          :key="'syg-' + tick"
          :x1="sectionPad.left"
          :y1="sectionYScale(tick)"
          :x2="sectionW - sectionPad.right"
          :y2="sectionYScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line :x1="sectionPad.left" :y1="sectionYScale(0)" :x2="sectionW - sectionPad.right" :y2="sectionYScale(0)" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="sectionPad.left" :y1="sectionPad.top" :x2="sectionPad.left" :y2="sectionH - sectionPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <template v-for="center in lensCenters" :key="'rect-' + center">
          <rect
            :x="sectionXScale(center - sectionMaskW / 2)"
            :y="sectionYScale(resistThickness)"
            :width="Math.max(1, sectionXScale(center + sectionMaskW / 2) - sectionXScale(center - sectionMaskW / 2))"
            :height="sectionYScale(0) - sectionYScale(resistThickness)"
            fill="#9b59b6"
            fill-opacity="0.08"
            stroke="#9b59b6"
            stroke-width="1"
            stroke-dasharray="4,3"
          />
        </template>

        <path
          v-for="profile in reflowProfiles"
          :key="'reflow-' + profile"
          :d="profile"
          fill="none"
          stroke="#e67e22"
          stroke-width="1.5"
          stroke-dasharray="5,4"
          opacity="0.9"
        />
        <path
          v-for="profile in finalProfiles"
          :key="'final-' + profile"
          :d="profile"
          fill="#3498db"
          fill-opacity="0.20"
          stroke="#1f78b4"
          stroke-width="2"
        />

        <line
          :x1="sectionXScale(-finalGap / 2)"
          :y1="sectionYScale(-0.035)"
          :x2="sectionXScale(finalGap / 2)"
          :y2="sectionYScale(-0.035)"
          stroke="#c0392b"
          stroke-width="2"
          :stroke-dasharray="finalGap <= 0.002 ? '2,2' : 'none'"
        />
        <text :x="sectionXScale(0)" :y="sectionYScale(-0.08)" text-anchor="middle" class="plot-label" fill="#c0392b">
          {{ t('final gap', '최종 gap') }}
        </text>

        <line :x1="sectionXScale(0)" :y1="sectionYScale(0)" :x2="sectionXScale(0)" :y2="sectionYScale(finalHeight)" stroke="#27ae60" stroke-width="1.4" stroke-dasharray="4,3" />
        <text :x="sectionXScale(0) + 6" :y="sectionYScale(finalHeight / 2)" class="plot-label" fill="#27ae60">
          h={{ finalHeight.toFixed(2) }} um
        </text>

        <line x1="438" y1="22" x2="462" y2="22" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="4,3" />
        <text x="468" y="26" class="legend-label">{{ t('litho resist island', 'litho resist island') }}</text>
        <line x1="438" y1="40" x2="462" y2="40" stroke="#e67e22" stroke-width="1.5" stroke-dasharray="5,4" />
        <text x="468" y="44" class="legend-label">{{ t('after reflow', 'reflow 후') }}</text>
        <line x1="438" y1="58" x2="462" y2="58" stroke="#1f78b4" stroke-width="2" />
        <text x="468" y="62" class="legend-label">{{ t('after etch transfer', 'etch transfer 후') }}</text>

        <text v-for="tick in sectionXTicks" :key="'sxl-' + tick" :x="sectionXScale(tick)" :y="sectionH - 12" text-anchor="middle" class="axis-label">
          {{ tick.toFixed(1) }}
        </text>
        <text v-for="tick in sectionYTicks" :key="'syl-' + tick" :x="sectionPad.left - 8" :y="sectionYScale(tick) + 3" text-anchor="end" class="axis-label">
          {{ tick.toFixed(1) }}
        </text>
        <text :x="(sectionPad.left + sectionW - sectionPad.right) / 2" :y="sectionH - 2" text-anchor="middle" class="axis-label">x (um)</text>
        <text x="13" :y="(sectionPad.top + sectionH - sectionPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 13, ${(sectionPad.top + sectionH - sectionPad.bottom) / 2})`">z (um)</text>
      </svg>

      <svg
        v-if="viewMode === 'topview'"
        :viewBox="`0 0 ${topW} ${topH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Microlens footprint top view', '마이크로렌즈 footprint 위에서 본 모양')"
      >
        <rect x="0" y="0" :width="topW" :height="topH" fill="var(--vp-c-bg)" />
        <line
          v-for="line in topGridLines"
          :key="line.key"
          :x1="line.x1"
          :y1="line.y1"
          :x2="line.x2"
          :y2="line.y2"
          stroke="var(--vp-c-divider)"
          stroke-width="0.7"
        />

        <template v-for="grp in topGroupsRender" :key="'tgrp-' + grp.id">
          <path
            :d="grp.maskPath"
            fill="none"
            stroke="#9b59b6"
            stroke-width="1.1"
            stroke-dasharray="4,3"
            opacity="0.7"
          />
          <path
            :d="grp.reflowPath"
            fill="none"
            stroke="#e67e22"
            stroke-width="1.3"
            stroke-dasharray="5,4"
            opacity="0.85"
          />
          <path
            :d="grp.finalPath"
            :fill="grp.isValid ? heatColor(grp.finalHeightUm) : 'rgba(192, 57, 43, 0.25)'"
            fill-opacity="0.78"
            :stroke="grp.isValid ? '#1f4e79' : '#a93226'"
            stroke-width="1.5"
          />
          <text
            :x="grp.centerPx.x"
            :y="grp.centerPx.y + 4"
            text-anchor="middle"
            class="top-group-label"
          >{{ grp.labelText }}</text>
        </template>

        <line
          v-for="m in topGapMarkers"
          :key="m.key"
          :x1="m.x1"
          :y1="m.y1"
          :x2="m.x2"
          :y2="m.y2"
          :stroke="m.tone === 'good' ? '#27ae60' : m.tone === 'risk' ? '#c0392b' : '#d35400'"
          stroke-width="2"
        />

        <g transform="translate(20, 20)">
          <line x1="0" y1="6" x2="22" y2="6" stroke="#9b59b6" stroke-width="1.1" stroke-dasharray="4,3" />
          <text x="28" y="10" class="legend-label">{{ t('Mask footprint', '마스크 footprint') }}</text>
          <line x1="0" y1="24" x2="22" y2="24" stroke="#e67e22" stroke-width="1.3" stroke-dasharray="5,4" />
          <text x="28" y="28" class="legend-label">{{ t('Reflow footprint', 'Reflow footprint') }}</text>
          <line x1="0" y1="42" x2="22" y2="42" stroke="#1f4e79" stroke-width="1.5" />
          <text x="28" y="46" class="legend-label">{{ t('Final footprint', '최종 footprint') }}</text>
        </g>

        <g :transform="`translate(${topW - 130}, 28)`">
          <text x="0" y="0" class="legend-label">{{ t('Final height (um)', '최종 높이 (um)') }}</text>
          <g v-for="(stop, i) in topLegendStops" :key="'hstop-' + i">
            <rect :x="0" :y="8 + i * 16" width="22" height="12" :fill="stop.color" stroke="var(--vp-c-divider)" stroke-width="0.5" />
            <text :x="28" :y="18 + i * 16" class="legend-label">{{ stop.value.toFixed(2) }}</text>
          </g>
        </g>

        <text :x="topW / 2" :y="topH - 8" text-anchor="middle" class="axis-label">
          {{ t('4×4 cell grid · 1 cell = pitch', '4×4 셀 그리드 · 1 셀 = pitch') }}
        </text>
      </svg>

      <svg
        v-if="viewMode === 'surface'"
        :viewBox="`0 0 ${surfaceW} ${surfaceH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Microlens final surface wireframe', '마이크로렌즈 최종 표면 wireframe')"
      >
        <rect x="0" y="0" :width="surfaceW" :height="surfaceH" fill="var(--vp-c-bg)" />
        <polygon :points="cellBasePolygon" fill="#7f8c8d" opacity="0.08" stroke="var(--vp-c-divider)" stroke-width="1" />
        <path
          v-for="g in surfaceGridLines"
          :key="g.key"
          :d="g.d"
          fill="none"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
          opacity="0.7"
        />
        <path
          v-for="line in surfaceLines"
          :key="line.key"
          :d="line.d"
          fill="none"
          :stroke="line.stroke"
          :stroke-width="line.major ? 1.4 : 0.7"
          :opacity="line.major ? 0.95 : 0.6"
        />
        <path
          v-for="edge in surfaceFootprintEdges"
          :key="edge.key"
          :d="edge.d"
          fill="none"
          :stroke="edge.stroke"
          stroke-width="1.2"
          opacity="0.85"
        />
        <text x="24" y="28" class="surface-title">{{ t('Final 3D footprints (all lens units)', '최종 3D footprint (모든 lens unit)') }}</text>
        <text x="24" y="48" class="surface-note">
          {{ t('Each lens group is rendered at its grid location with its own anisotropic radii and height.', '각 lens group을 grid 위치에 자체 anisotropic 반경과 height로 렌더링합니다.') }}
        </text>
        <g transform="translate(24, 268)">
          <rect x="0" y="0" width="240" height="78" rx="5" fill="var(--vp-c-bg-soft)" stroke="var(--vp-c-divider)" />
          <text x="10" y="14" class="legend-label">{{ t('Lens unit counts', 'Lens unit 개수') }}</text>
          <g v-for="(kind, ki) in (['1x1','2x1','1x2','2x2'] as const)" :key="'su-' + kind">
            <rect :x="10 + ki * 56" y="22" width="14" height="10" :fill="SURFACE_PALETTE[kind].rowMajor" stroke="var(--vp-c-divider)" stroke-width="0.5" />
            <text :x="28 + ki * 56" y="31" class="legend-label">{{ kind }} × {{ surfaceKindCounts[kind] }}</text>
          </g>
          <text x="10" y="50" class="legend-label">{{ t('Max height', '최대 높이') }} {{ globalMaxHeight.toFixed(3) }} um</text>
          <text x="10" y="64" class="legend-label">{{ t('Profile exponent (rep.)', 'Profile exponent (대표)') }} {{ profilePower.toFixed(2) }}</text>
        </g>
      </svg>

      <svg
        v-if="viewMode === 'process'"
        :viewBox="`0 0 ${processW} ${processH}`"
        class="main-svg"
        role="img"
        :aria-label="t('Etch response curves', 'Etch response curve')"
      >
        <rect x="0" y="0" :width="processW" :height="processH" fill="var(--vp-c-bg)" />
        <line
          v-for="tick in processXTicks"
          :key="'pxg-' + tick"
          :x1="processXScale(tick)"
          :y1="processPad.top"
          :x2="processXScale(tick)"
          :y2="processH - processPad.bottom"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line
          v-for="tick in processGapTicks"
          :key="'pyg-' + tick"
          :x1="processPad.left"
          :y1="processGapYScale(tick)"
          :x2="processW - processPad.right"
          :y2="processGapYScale(tick)"
          stroke="var(--vp-c-divider)"
          stroke-width="0.6"
        />
        <line :x1="processPad.left" :y1="processPad.top" :x2="processPad.left" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="processPad.left" :y1="processH - processPad.bottom" :x2="processW - processPad.right" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />
        <line :x1="processW - processPad.right" :y1="processPad.top" :x2="processW - processPad.right" :y2="processH - processPad.bottom" stroke="var(--vp-c-text-2)" stroke-width="1" />

        <path :d="gapCurvePath" fill="none" stroke="#c0392b" stroke-width="2.4" />
        <path :d="heightCurvePath" fill="none" stroke="#27ae60" stroke-width="2.4" />
        <line :x1="currentProcessPoint.x" :y1="processPad.top" :x2="currentProcessPoint.x" :y2="processH - processPad.bottom" stroke="var(--vp-c-brand-1)" stroke-width="1.3" stroke-dasharray="5,4" />
        <circle :cx="currentProcessPoint.x" :cy="currentProcessPoint.gapY" r="4.5" fill="#c0392b" stroke="#fff" stroke-width="1.2" />
        <circle :cx="currentProcessPoint.x" :cy="currentProcessPoint.heightY" r="4.5" fill="#27ae60" stroke="#fff" stroke-width="1.2" />

        <line x1="88" y1="30" x2="118" y2="30" stroke="#c0392b" stroke-width="2.4" />
        <text x="124" y="34" class="legend-label">{{ t('Final gap (left axis)', '최종 gap (좌축)') }}</text>
        <line x1="88" y1="49" x2="118" y2="49" stroke="#27ae60" stroke-width="2.4" />
        <text x="124" y="53" class="legend-label">{{ t('Height retention (right axis)', '높이 보존율 (우축)') }}</text>

        <text v-for="tick in processXTicks" :key="'pxl-' + tick" :x="processXScale(tick)" :y="processH - 12" text-anchor="middle" class="axis-label">{{ tick }}</text>
        <text v-for="tick in processGapTicks" :key="'pyl-' + tick" :x="processPad.left - 8" :y="processGapYScale(tick) + 3" text-anchor="end" class="axis-label">{{ tick.toFixed(2) }}</text>
        <text v-for="tick in processRetentionTicks" :key="'prl-' + tick" :x="processW - processPad.right + 8" :y="processRetentionYScale(tick) + 3" class="axis-label">{{ tick }}%</text>
        <text :x="(processPad.left + processW - processPad.right) / 2" :y="processH - 2" text-anchor="middle" class="axis-label">{{ t('Etch time (s)', 'Etch 시간 (s)') }}</text>
        <text x="13" :y="(processPad.top + processH - processPad.bottom) / 2" text-anchor="middle" class="axis-label" :transform="`rotate(-90, 13, ${(processPad.top + processH - processPad.bottom) / 2})`">{{ t('Gap (um)', 'Gap (um)') }}</text>
      </svg>
    </div>

    <div class="formula-box">
      <strong>{{ t('Model note', '모델 메모') }}:</strong>
      {{ t(
        'This is a calibrated-by-user surrogate, not a foundry recipe. It combines volume-conserving reflow, parabolic/superellipse caps, and DOE-inspired etch trends: more etch time closes gap; polymerization mainly preserves height; mask thickness changes transfer robustness. Fit the calibration gains to measured gap, height, and profile data before making quantitative decisions.',
        '이 모델은 foundry recipe가 아니라 사용자가 보정해 쓰는 surrogate입니다. Volume-conserving reflow, parabolic/superellipse cap, DOE식 etch 경향을 결합합니다. Etch time은 gap closure를 키우고, polymerization은 주로 height 보존에, mask thickness는 transfer robustness에 영향을 준다고 둡니다. 정량 의사결정 전에는 calibration gain을 실측 gap, height, profile 데이터에 맞춰야 합니다.'
      ) }}
    </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useLocale } from '../composables/useLocale'
import { useFullscreen } from '../composables/useFullscreen'

const { t } = useLocale()
const { isFullscreen, toggleFullscreen } = useFullscreen()

type ApertureShape = 'circular' | 'rounded-square' | 'square'
type ViewMode = 'section' | 'topview' | 'surface' | 'process'
type LensUnitShape = '1x1' | '2x1' | '1x2' | '2x2'
type LayoutPreset =
  | 'all-1x1'
  | 'all-2x1'
  | 'all-1x2'
  | 'all-2x2'
  | 'mixed-2x2-pdaf'
  | 'sparse-2x1-pdaf'
  | 'custom'

interface LensGroup {
  id: number
  cells: { r: number; c: number }[]
  r0: number
  c0: number
  h: number
  w: number
  kind: LensUnitShape
  isValidShape: boolean
}

const GRID_N = 4

const pitch = ref(1.10)
const maskWidth = ref(0.88)
const resistThickness = ref(0.42)
const apertureShape = ref<ApertureShape>('rounded-square')
const reflowTemp = ref(170)
const reflowTime = ref(90)
const lensIndex = ref(1.55)
const maskThickness = ref(0.45)
const polymerGas = ref(55)
const etchTime = ref(55)
const viewMode = ref<ViewMode>('section')
const reflowSpreadGain = ref(1.00)
const volumeRetentionGain = ref(1.00)
const lateralEtchGain = ref(1.00)
const verticalLossGain = ref(1.00)
const proximityCouplingGain = ref(1.00)
const microloadingGain = ref(1.00)
const layoutPreset = ref<LayoutPreset>('all-1x1')
const cellGrid = ref<number[]>(buildGridFromRects([]))
const selectedCell = ref<number | null>(null)

const tabs = [
  { key: 'section' as const, en: 'Cross-section', ko: '단면' },
  { key: 'topview' as const, en: 'Top view (XY)', ko: '위에서 본 모양' },
  { key: 'surface' as const, en: '3D surface', ko: '3D 표면' },
  { key: 'process' as const, en: 'Etch response', ko: 'Etch 응답' },
]

function buildGridFromRects(rects: [number, number, number, number][]) {
  const g = new Array(GRID_N * GRID_N).fill(-1)
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

type Rect = [number, number, number, number]
const LAYOUT_PRESETS: Record<Exclude<LayoutPreset, 'custom'>, Rect[]> = {
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

function applyPreset(preset: LayoutPreset) {
  if (preset !== 'custom') {
    cellGrid.value = buildGridFromRects(LAYOUT_PRESETS[preset])
  }
  selectedCell.value = null
}

watch(layoutPreset, (preset) => applyPreset(preset))

function deriveGroups(grid: number[]): LensGroup[] {
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
    const expected = h * w
    const rectangular = cells.length === expected
    const allowed = (h === 1 || h === 2) && (w === 1 || w === 2)
    const isValidShape = rectangular && allowed
    let kind: LensUnitShape = '1x1'
    if (h === 1 && w === 1) kind = '1x1'
    else if (h === 1 && w === 2) kind = '2x1'
    else if (h === 2 && w === 1) kind = '1x2'
    else if (h === 2 && w === 2) kind = '2x2'
    groups.push({ id, cells, r0, c0, h, w, kind, isValidShape })
  }
  groups.sort((a, b) => (a.r0 - b.r0) || (a.c0 - b.c0))
  return groups
}

const customGridSize = 220
const customCellSize = customGridSize / GRID_N

function renumberGrid(grid: number[]) {
  const remap = new Map<number, number>()
  let next = 0
  return grid.map((id) => {
    if (!remap.has(id)) {
      remap.set(id, next)
      next += 1
    }
    return remap.get(id)!
  })
}

function tryMergeCells(idxA: number, idxB: number): boolean {
  if (idxA === idxB) return false
  const rA = Math.floor(idxA / GRID_N)
  const cA = idxA % GRID_N
  const rB = Math.floor(idxB / GRID_N)
  const cB = idxB % GRID_N
  const groups = lensGroups.value
  const groupA = groups.find(g => g.cells.some(p => p.r === rA && p.c === cA))!
  const groupB = groups.find(g => g.cells.some(p => p.r === rB && p.c === cB))!
  if (groupA.id === groupB.id) return false
  const combined = [...groupA.cells, ...groupB.cells]
  const rs = combined.map(p => p.r)
  const cs = combined.map(p => p.c)
  const h = Math.max(...rs) - Math.min(...rs) + 1
  const w = Math.max(...cs) - Math.min(...cs) + 1
  if ((h !== 1 && h !== 2) || (w !== 1 && w !== 2)) return false
  if (combined.length !== h * w) return false
  const newGrid = cellGrid.value.slice()
  const newId = Math.min(groupA.id, groupB.id)
  for (const p of combined) newGrid[p.r * GRID_N + p.c] = newId
  cellGrid.value = renumberGrid(newGrid)
  return true
}

function splitGroupAtCell(idx: number) {
  const r = Math.floor(idx / GRID_N)
  const c = idx % GRID_N
  const group = lensGroups.value.find(g => g.cells.some(p => p.r === r && p.c === c))
  if (!group || group.cells.length === 1) return
  const newGrid = cellGrid.value.slice()
  const maxId = Math.max(...newGrid)
  let nextId = maxId + 1
  for (const cell of group.cells) {
    newGrid[cell.r * GRID_N + cell.c] = nextId
    nextId += 1
  }
  cellGrid.value = renumberGrid(newGrid)
}

function onCellClick(idx: number) {
  if (layoutPreset.value !== 'custom') return
  if (selectedCell.value === null) {
    selectedCell.value = idx
    return
  }
  if (selectedCell.value === idx) {
    selectedCell.value = null
    return
  }
  tryMergeCells(selectedCell.value, idx)
  selectedCell.value = null
}

function onCellDoubleClick(idx: number) {
  if (layoutPreset.value !== 'custom') return
  splitGroupAtCell(idx)
  selectedCell.value = null
}

function groupKindForCell(idx: number): string {
  const r = Math.floor(idx / GRID_N)
  const c = idx % GRID_N
  const group = lensGroups.value.find(g => g.cells.some(p => p.r === r && p.c === c))
  if (!group) return ''
  if (!group.isValidShape) return '!'
  if (r !== group.r0 || c !== group.c0) return ''
  return group.kind
}

function customCellFill(gid: number, idx: number) {
  const r = Math.floor(idx / GRID_N)
  const c = idx % GRID_N
  const group = lensGroups.value.find(g => g.cells.some(p => p.r === r && p.c === c))
  if (!group) return 'var(--vp-c-bg)'
  if (!group.isValidShape) return 'rgba(192, 57, 43, 0.18)'
  const palette: Record<LensUnitShape, string> = {
    '1x1': 'rgba(52, 152, 219, 0.18)',
    '2x1': 'rgba(155, 89, 182, 0.22)',
    '1x2': 'rgba(230, 126, 34, 0.22)',
    '2x2': 'rgba(39, 174, 96, 0.22)',
  }
  return palette[group.kind]
}

const lensGroups = computed(() => deriveGroups(cellGrid.value))
const isLayoutValid = computed(() => lensGroups.value.every(g => g.isValidShape))
const representativeGroup = computed<LensGroup>(() => {
  const groups = lensGroups.value
  const nonUnit = groups.find(g => g.kind !== '1x1' && g.isValidShape)
  return nonUnit ?? groups[0]
})

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

const minMaskWidth = computed(() => Math.max(0.2, pitch.value * 0.35))
const maxMaskWidth = computed(() => Math.max(minMaskWidth.value + 0.02, pitch.value * 0.98))

watch([minMaskWidth, maxMaskWidth], () => {
  maskWidth.value = clamp(maskWidth.value, minMaskWidth.value, maxMaskWidth.value)
}, { immediate: true })

const boundedMaskWidth = computed(() => clamp(maskWidth.value, minMaskWidth.value, maxMaskWidth.value))
const boundaryGap = computed(() => Math.max(0, pitch.value - boundedMaskWidth.value))
const thermalDose = computed(() => {
  const tempNorm = clamp((reflowTemp.value - 125) / 80, 0, 1.35)
  const timeNorm = clamp(Math.log1p(reflowTime.value / 30) / Math.log1p(300 / 30), 0, 1.1)
  return clamp(0.62 * tempNorm + 0.38 * timeNorm, 0, 1.25)
})

const shapeExponent = computed(() => {
  if (apertureShape.value === 'circular') return 2
  if (apertureShape.value === 'rounded-square') return 4
  return 8
})

const areaFactor = computed(() => {
  if (apertureShape.value === 'circular') return Math.PI / 4
  if (apertureShape.value === 'rounded-square') return 0.91
  return 0.98
})

const polyNorm = computed(() => polymerGas.value / 100)
const maskRobustness = computed(() => clamp(0.72 + maskThickness.value / 0.75, 0.72, 1.70))
const lateralEtchRate = computed(() => {
  const poly = polyNorm.value
  const mask = maskRobustness.value
  return lateralEtchGain.value * (0.0018 + 0.0012 * poly + 0.00045 * thermalDose.value) * (0.84 + 0.16 * mask)
})
const verticalLossRate = computed(() => {
  const poly = polyNorm.value
  const mask = maskRobustness.value
  return verticalLossGain.value * 0.0030 * (1 - 0.62 * poly) * (1.12 - 0.16 * mask)
})

interface ProcessState {
  pitchX: number
  pitchY: number
  maskWX: number
  maskWY: number
  initialGapX: number
  initialGapY: number
  reflowSpreadX: number
  reflowSpreadY: number
  reflowWX: number
  reflowWY: number
  reflowGapX: number
  reflowGapY: number
  reflowHeight: number
  closure: number
  lossFraction: number
  finalGapX: number
  finalGapY: number
  finalWX: number
  finalWY: number
  finalHeight: number
  profilePower: number
  retention: number
  aspectRatio: number
}

function computeProcessAt(pitchX: number, pitchY: number, maskWX: number, maskWY: number, etchSeconds: number): ProcessState {
  const initialGapX = Math.max(0, pitchX - maskWX)
  const initialGapY = Math.max(0, pitchY - maskWY)
  const thicknessTerm = 0.10 * resistThickness.value
  const spreadBase = reflowSpreadGain.value * thermalDose.value
  const spreadX0 = spreadBase * (0.018 + thicknessTerm + 0.20 * initialGapX)
  const spreadY0 = spreadBase * (0.018 + thicknessTerm + 0.20 * initialGapY)
  // Surface-tension correction (Choi et al.): asymmetric masks reflow toward isotropy;
  // the long axis grows less and the short axis grows more.
  const maskAsym = (maskWX - maskWY) / Math.max(maskWX + maskWY, 1e-6)
  const tensionPull = 0.35 * thermalDose.value * maskAsym * ((spreadX0 + spreadY0) / 2)
  const reflowSpreadX = Math.max(0, spreadX0 - tensionPull)
  const reflowSpreadY = Math.max(0, spreadY0 + tensionPull)
  const reflowWX = Math.min(pitchX * 1.04, maskWX + 2 * reflowSpreadX)
  const reflowWY = Math.min(pitchY * 1.04, maskWY + 2 * reflowSpreadY)
  const reflowGapX = Math.max(0, pitchX - reflowWX)
  const reflowGapY = Math.max(0, pitchY - reflowWY)
  const r0x = maskWX / 2
  const r0y = maskWY / 2
  const r1x = reflowWX / 2
  const r1y = reflowWY / 2
  const retention = clamp((0.96 - 0.08 * thermalDose.value) * volumeRetentionGain.value, 0.74, 1.02)
  const reflowHeight = clamp((2 * resistThickness.value * r0x * r0y * retention) / Math.max(r1x * r1y, 0.01), 0.035, 2.0)
  const closure = etchSeconds * lateralEtchRate.value
  const lossFraction = clamp(etchSeconds * verticalLossRate.value, 0, 0.58)
  const finalGapX = Math.max(0, reflowGapX - 2 * closure)
  const finalGapY = Math.max(0, reflowGapY - 2 * closure)
  const finalWX = Math.max(0.05, pitchX - finalGapX)
  const finalWY = Math.max(0.05, pitchY - finalGapY)
  const finalHeight = Math.max(0.025, reflowHeight * (1 - lossFraction))
  const profilePower = clamp(2.0 + 0.42 * thermalDose.value + 1.2 * lossFraction - 0.22 * polyNorm.value, 1.7, 4.2)
  const aspectRatio = Math.max(finalWX, finalWY) / Math.max(Math.min(finalWX, finalWY), 1e-6)
  return {
    pitchX, pitchY, maskWX, maskWY,
    initialGapX, initialGapY,
    reflowSpreadX, reflowSpreadY,
    reflowWX, reflowWY,
    reflowGapX, reflowGapY,
    reflowHeight,
    closure, lossFraction,
    finalGapX, finalGapY,
    finalWX, finalWY,
    finalHeight,
    profilePower,
    retention,
    aspectRatio,
  }
}

function groupMaskDims(group: LensGroup) {
  const bGap = boundaryGap.value
  const maskWX = Math.max(0.02, group.w * pitch.value - bGap)
  const maskWY = Math.max(0.02, group.h * pitch.value - bGap)
  return { maskWX, maskWY, pitchX: group.w * pitch.value, pitchY: group.h * pitch.value }
}

function computeGroupProcessAt(group: LensGroup, etchSeconds: number) {
  const { maskWX, maskWY, pitchX, pitchY } = groupMaskDims(group)
  return computeProcessAt(pitchX, pitchY, maskWX, maskWY, etchSeconds)
}

// --- Per-side proximity coupling -----------------------------------------
// Mass-flow asymmetry (Choi et al., reflow toward isotropy; Stanford E241
// survey) drives the lens SHAPE to bulge toward larger neighbors.
// Microloading + ARDE (Mogab 1977; Gottscho 1992) reduces lateral etch
// closure near dense pattern, leaving a larger GAP next to bigger lens
// groups. Both effects vanish for uniform-size layouts.

interface GroupNeighbors {
  left: LensGroup | null
  right: LensGroup | null
  top: LensGroup | null
  bottom: LensGroup | null
}

function buildCellIndex(groups: LensGroup[]): Map<string, LensGroup> {
  const m = new Map<string, LensGroup>()
  for (const g of groups) {
    for (const cell of g.cells) m.set(`${cell.r},${cell.c}`, g)
  }
  return m
}

function getGroupNeighbors(group: LensGroup, cellIndex: Map<string, LensGroup>): GroupNeighbors {
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

interface SideOffsets {
  edgeLeft: number
  edgeRight: number
  edgeTop: number
  edgeBottom: number
  gapLeft: number
  gapRight: number
  gapTop: number
  gapBottom: number
  sigmaLeft: number
  sigmaRight: number
  sigmaTop: number
  sigmaBottom: number
  asymmetry: number
}

// Empirical baseline constants; the user-facing gains scale these.
const K_MASS_BASE_UM = 0.045
const K_LOAD_BASE = 0.55

function computeGroupSideOffsets(
  group: LensGroup,
  baseState: ProcessState,
  neighbors: GroupNeighbors,
  etchSeconds: number,
): SideOffsets {
  const A_G = group.h * group.w
  const td = thermalDose.value
  const ler = lateralEtchRate.value
  const kMass = proximityCouplingGain.value * K_MASS_BASE_UM
  const kLoad = microloadingGain.value * K_LOAD_BASE

  function sigmaShape(N: LensGroup | null) {
    if (!N || !N.isValidShape) return 0
    const A_N = N.h * N.w
    return (A_N - A_G) / (A_N + A_G)
  }
  function densityFactor(N: LensGroup | null) {
    // grid-edge neighbor: treat as 1x1-equivalent sparse padding
    const A_N = N && N.isValidShape ? N.h * N.w : 1
    const baseline = 2 // two 1x1 cells = sparsest dense baseline
    return Math.max(0, ((A_G + A_N) - baseline) / 6) // normalize to 1 when both are 2x2 (sum=8)
  }

  const sL = sigmaShape(neighbors.left)
  const sR = sigmaShape(neighbors.right)
  const sT = sigmaShape(neighbors.top)
  const sB = sigmaShape(neighbors.bottom)
  // mass-flow: my lens extends MORE on larger-neighbor side
  const dSpL = kMass * td * sL
  const dSpR = kMass * td * sR
  const dSpT = kMass * td * sT
  const dSpB = kMass * td * sB
  // microloading: gap stays LARGER (less closure) near denser pattern
  const dGapL = kLoad * ler * etchSeconds * densityFactor(neighbors.left)
  const dGapR = kLoad * ler * etchSeconds * densityFactor(neighbors.right)
  const dGapT = kLoad * ler * etchSeconds * densityFactor(neighbors.top)
  const dGapB = kLoad * ler * etchSeconds * densityFactor(neighbors.bottom)

  const halfX = baseState.finalWX / 2
  const halfY = baseState.finalWY / 2
  const edgeLeft = Math.max(0.02, halfX + dSpL)
  const edgeRight = Math.max(0.02, halfX + dSpR)
  const edgeTop = Math.max(0.02, halfY + dSpT)
  const edgeBottom = Math.max(0.02, halfY + dSpB)
  const gapLeft = Math.max(0, baseState.finalGapX + 2 * dGapL)
  const gapRight = Math.max(0, baseState.finalGapX + 2 * dGapR)
  const gapTop = Math.max(0, baseState.finalGapY + 2 * dGapT)
  const gapBottom = Math.max(0, baseState.finalGapY + 2 * dGapB)
  const asymmetry = Math.max(sL, sR, sT, sB) - Math.min(sL, sR, sT, sB)

  return {
    edgeLeft, edgeRight, edgeTop, edgeBottom,
    gapLeft, gapRight, gapTop, gapBottom,
    sigmaLeft: sL, sigmaRight: sR, sigmaTop: sT, sigmaBottom: sB,
    asymmetry,
  }
}

const groupCellIndex = computed(() => buildCellIndex(lensGroups.value))

function groupNeighborsFor(group: LensGroup) {
  return getGroupNeighbors(group, groupCellIndex.value)
}

interface GroupRenderRow {
  group: LensGroup
  state: ProcessState
  neighbors: GroupNeighbors
  sides: SideOffsets
  center: { x: number; y: number }
}

// Single per-group derivation reused by every consumer (top view, gap markers,
// surface wireframe, footprint edges, status flags).
const groupRenderData = computed<GroupRenderRow[]>(() =>
  lensGroups.value.map((group) => {
    const state = computeGroupProcessAt(group, etchTime.value)
    const neighbors = groupNeighborsFor(group)
    const sides = computeGroupSideOffsets(group, state, neighbors, etchTime.value)
    return { group, state, neighbors, sides, center: groupCenterUm(group) }
  }),
)

const representativeProcess = computed<ProcessState>(() => computeGroupProcessAt(representativeGroup.value, etchTime.value))

// Scalar computeds for the metric cards and the cross-section overlay
// (representative-group view; mixed-layout aggregates are intentionally not
// derived here — the Top view and 3D surface visualise per-group state).
const initialGap = computed(() => representativeProcess.value.initialGapX)
const reflowGap = computed(() => representativeProcess.value.reflowGapX)
const reflowHeight = computed(() => representativeProcess.value.reflowHeight)

function etchTransferAt(timeSeconds: number) {
  const state = computeGroupProcessAt(representativeGroup.value, timeSeconds)
  return {
    gap: Math.max(state.finalGapX, state.finalGapY),
    closure: state.closure,
    lossFraction: state.lossFraction,
    height: state.finalHeight,
  }
}

const finalGap = computed(() => Math.max(representativeProcess.value.finalGapX, representativeProcess.value.finalGapY))
const finalWidth = computed(() => representativeProcess.value.finalWX)
const finalHeight = computed(() => representativeProcess.value.finalHeight)
const heightRetention = computed(() => clamp((finalHeight.value / Math.max(reflowHeight.value, 0.001)) * 100, 0, 120))
const profilePower = computed(() => representativeProcess.value.profilePower)
const halfFinalWidth = computed(() => representativeProcess.value.finalWX / 2)
const roc = computed(() => {
  const rep = representativeProcess.value
  const a = (rep.finalWX + rep.finalWY) / 4
  const h = rep.finalHeight
  return (a * a + h * h) / Math.max(2 * h, 0.001)
})
const focalLength = computed(() => roc.value / Math.max(lensIndex.value - 1, 0.05))
const fNumber = computed(() => focalLength.value / Math.max(finalWidth.value, 0.05))
const fillFactor = computed(() => {
  const rep = representativeProcess.value
  return clamp((areaFactor.value * rep.finalWX * rep.finalWY / Math.max(rep.pitchX * rep.pitchY, 1e-6)) * 100, 0, 100)
})
const aspectRatio = computed(() => representativeProcess.value.aspectRatio)
const worstReflowGap = computed(() => Math.min(representativeProcess.value.reflowGapX, representativeProcess.value.reflowGapY))
const zeroGapEtchTime = computed(() => {
  if (worstReflowGap.value <= 0.002) return 0
  const seconds = worstReflowGap.value / Math.max(2 * lateralEtchRate.value, 1e-6)
  return Math.min(seconds, 999)
})
const zeroGapEtchTimeLabel = computed(() => {
  if (zeroGapEtchTime.value <= 0.1) return t('already closed', '이미 닫힘')
  if (zeroGapEtchTime.value > 140) return t('>140 s', '>140 s')
  return `${zeroGapEtchTime.value.toFixed(0)} s`
})

const processFlags = computed(() => {
  const flags: { text: string; tone: string }[] = []
  const rep = representativeProcess.value
  if (finalGap.value <= 0.015) flags.push({ text: t('zero-gap candidate', 'zero-gap 후보'), tone: 'good' })
  else if (finalGap.value <= 0.06) flags.push({ text: t('near zero-space', 'zero-space 근접'), tone: 'good' })
  else flags.push({ text: t('visible lens gap', '렌즈 gap 잔존'), tone: 'warn' })

  if (rep.reflowGapX <= 0.002 && rep.initialGapX > 0.01) flags.push({ text: t('X-direction merger risk', 'X 방향 merger 위험'), tone: 'risk' })
  if (rep.reflowGapY <= 0.002 && rep.initialGapY > 0.01) flags.push({ text: t('Y-direction merger risk', 'Y 방향 merger 위험'), tone: 'risk' })
  if (rep.lossFraction > 0.32) flags.push({ text: t('height loss risk', 'height loss 위험'), tone: 'risk' })
  if (zeroGapEtchTime.value > 140 && finalGap.value > 0.06) flags.push({ text: t('etch window too short', 'etch window 부족'), tone: 'risk' })
  if (zeroGapEtchTime.value > 0.1 && etchTime.value > zeroGapEtchTime.value + 30) flags.push({ text: t('over-etch margin', 'over-etch margin'), tone: 'warn' })
  if (fillFactor.value > 92) flags.push({ text: t('high fill factor', '높은 fill factor'), tone: 'good' })
  if (fNumber.value < 0.9 || fNumber.value > 3.8) flags.push({ text: t('check optical focus', 'optical focus 확인 필요'), tone: 'warn' })
  if (representativeGroup.value.kind !== '1x1' && aspectRatio.value > 1.08) {
    flags.push({ text: t('asymmetric reflow profile', '비대칭 reflow profile'), tone: 'warn' })
  }
  // Heterogeneous-neighbor coupling flag: any valid group with asymmetric sigma > 0.6
  for (const row of groupRenderData.value) {
    if (row.group.isValidShape && row.sides.asymmetry > 0.6) {
      flags.push({ text: t('proximity asymmetry', '인접 비대칭 결합'), tone: 'warn' })
      break
    }
  }
  if (!isLayoutValid.value) flags.push({ text: t('invalid lens layout', '유효하지 않은 lens 배치'), tone: 'risk' })
  return flags
})

function lensZ1D(x: number, halfWidth: number, height: number, power: number) {
  const u = Math.abs(x) / Math.max(halfWidth, 0.001)
  if (u >= 1) return 0
  return height * Math.pow(1 - Math.pow(u, power), 1.0)
}

function lensZ2D(x: number, y: number, halfWX: number, halfWY: number, height: number, profile: number) {
  const n = shapeExponent.value
  const r = Math.pow(
    Math.pow(Math.abs(x) / Math.max(halfWX, 0.001), n) +
      Math.pow(Math.abs(y) / Math.max(halfWY, 0.001), n),
    1 / n,
  )
  if (r >= 1) return 0
  return height * Math.pow(1 - Math.pow(r, profile), 1.0)
}

// Cross-section plot
const sectionW = 640
const sectionH = 330
const sectionPad = { left: 52, right: 22, top: 22, bottom: 42 }
const sectionPitch = computed(() => representativeProcess.value.pitchX)
const sectionMaskW = computed(() => representativeProcess.value.maskWX)
const lensCenters = computed(() => [-sectionPitch.value, 0, sectionPitch.value])
const sectionXMin = computed(() => -1.55 * sectionPitch.value)
const sectionXMax = computed(() => 1.55 * sectionPitch.value)
const sectionYMax = computed(() => Math.max(reflowHeight.value, finalHeight.value, resistThickness.value) * 1.22 + 0.08)
const sectionYMin = computed(() => -0.12)
const sectionXTicks = computed(() => {
  return [-1.5, -1, -0.5, 0, 0.5, 1, 1.5].map(v => v * sectionPitch.value).filter(v => v >= sectionXMin.value && v <= sectionXMax.value)
})
const sectionYTicks = computed(() => {
  const max = sectionYMax.value
  return [0, max * 0.25, max * 0.5, max * 0.75, max].map(v => Number(v.toFixed(2)))
})

function sectionXScale(x: number) {
  const plotW = sectionW - sectionPad.left - sectionPad.right
  return sectionPad.left + ((x - sectionXMin.value) / (sectionXMax.value - sectionXMin.value)) * plotW
}

function sectionYScale(y: number) {
  const plotH = sectionH - sectionPad.top - sectionPad.bottom
  return sectionPad.top + (1 - (y - sectionYMin.value) / (sectionYMax.value - sectionYMin.value)) * plotH
}

function buildSectionProfile(center: number, width: number, height: number, power: number, fill: boolean) {
  const half = width / 2
  const points: string[] = []
  for (let i = 0; i <= 72; i += 1) {
    const xLocal = -half + (2 * half * i) / 72
    const z = lensZ1D(xLocal, half, height, power)
    points.push(`${i === 0 ? 'M' : 'L'} ${sectionXScale(center + xLocal).toFixed(2)} ${sectionYScale(z).toFixed(2)}`)
  }
  if (fill) {
    points.push(`L ${sectionXScale(center + half).toFixed(2)} ${sectionYScale(0).toFixed(2)}`)
    points.push(`L ${sectionXScale(center - half).toFixed(2)} ${sectionYScale(0).toFixed(2)} Z`)
  }
  return points.join(' ')
}

const reflowProfiles = computed(() => lensCenters.value.map(center => buildSectionProfile(center, representativeProcess.value.reflowWX, reflowHeight.value, 2.0, false)))
const finalProfiles = computed(() => lensCenters.value.map(center => buildSectionProfile(center, representativeProcess.value.finalWX, finalHeight.value, profilePower.value, true)))

// 3D wireframe
const surfaceW = 640
const surfaceH = 360
const surfaceCx = 322
const surfaceCy = 214
const surfaceScale = computed(() => {
  const fitGrid = 230 / Math.max(GRID_N * pitch.value, 0.4)
  return Math.max(20, fitGrid)
})
const globalMaxHeight = computed(() => {
  let max = 0.08
  for (const { group, state } of groupRenderData.value) {
    if (group.isValidShape && state.finalHeight > max) max = state.finalHeight
  }
  return max
})
const surfaceZScale = computed(() => 80 / Math.max(globalMaxHeight.value, 0.08))

function projectSurface(x: number, y: number, z: number) {
  const sx = surfaceScale.value
  return {
    x: surfaceCx + (x - y) * sx * 0.82,
    y: surfaceCy + (x + y) * sx * 0.38 - z * surfaceZScale.value,
  }
}

function buildSurfaceLine(points: { x: number; y: number; z: number }[]) {
  return points.map((p, i) => {
    const projected = projectSurface(p.x, p.y, p.z)
    return `${i === 0 ? 'M' : 'L'} ${projected.x.toFixed(2)} ${projected.y.toFixed(2)}`
  }).join(' ')
}

interface SurfacePalette {
  rowMajor: string
  rowMinor: string
  colMajor: string
  colMinor: string
  edge: string
}

const SURFACE_PALETTE: Record<LensUnitShape, SurfacePalette> = {
  '1x1': { rowMajor: '#1f78b4', rowMinor: '#3498db', colMajor: '#8e44ad', colMinor: '#9b59b6', edge: '#34495e' },
  '2x1': { rowMajor: '#7d3c98', rowMinor: '#bb8fce', colMajor: '#6c3483', colMinor: '#a569bd', edge: '#5b2c6f' },
  '1x2': { rowMajor: '#c0392b', rowMinor: '#ec7063', colMajor: '#a04000', colMinor: '#dc7633', edge: '#922b21' },
  '2x2': { rowMajor: '#1e8449', rowMinor: '#52be80', colMajor: '#117a65', colMinor: '#48c9b0', edge: '#0e6251' },
}

const surfaceLines = computed(() => {
  const lines: { key: string; d: string; stroke: string; major: boolean; depth: number }[] = []
  const steps = 12
  const samples = 32
  // Sample wireframe along one axis with the other axis fixed.
  for (const { group, state, sides, center } of groupRenderData.value) {
    if (!group.isValidShape) continue
    const h = state.finalHeight
    const p = state.profilePower
    const palette = SURFACE_PALETTE[group.kind]
    const depth = center.x + center.y
    const xMin = -sides.edgeLeft
    const xMax = sides.edgeRight
    const yMin = -sides.edgeTop
    const yMax = sides.edgeBottom
    const hxFor = (x: number) => (x >= 0 ? sides.edgeRight : sides.edgeLeft)
    const hyFor = (y: number) => (y >= 0 ? sides.edgeBottom : sides.edgeTop)
    const axes = [
      { tag: 'x', major: palette.rowMajor, minor: palette.rowMinor, fixedLo: yMin, fixedHi: yMax, varLo: xMin, varHi: xMax, build: (fixed: number, vary: number) => ({ x: vary, y: fixed }) },
      { tag: 'y', major: palette.colMajor, minor: palette.colMinor, fixedLo: xMin, fixedHi: xMax, varLo: yMin, varHi: yMax, build: (fixed: number, vary: number) => ({ x: fixed, y: vary }) },
    ]
    for (const axis of axes) {
      for (let k = 0; k <= steps; k += 1) {
        const fixed = axis.fixedLo + (axis.fixedHi - axis.fixedLo) * (k / steps)
        const pts: { x: number; y: number; z: number }[] = []
        for (let i = 0; i <= samples; i += 1) {
          const vary = axis.varLo + (axis.varHi - axis.varLo) * (i / samples)
          const local = axis.build(fixed, vary)
          const z = lensZ2D(local.x, local.y, hxFor(local.x), hyFor(local.y), h, p)
          if (z > 0 || i === 0 || i === samples) {
            pts.push({ x: center.x + local.x, y: center.y + local.y, z })
          }
        }
        if (pts.length > 1) {
          const major = k === 0 || k === steps || k === steps / 2
          lines.push({
            key: `g${group.id}-${axis.tag}-${k}`,
            d: buildSurfaceLine(pts),
            stroke: major ? axis.major : axis.minor,
            major,
            depth,
          })
        }
      }
    }
  }
  // Painter's order: back-to-front (smaller depth first)
  lines.sort((a, b) => a.depth - b.depth)
  return lines
})

const surfaceFootprintEdges = computed(() => {
  const edges: { key: string; d: string; stroke: string }[] = []
  const n = shapeExponent.value
  for (const { group, sides, center } of groupRenderData.value) {
    if (!group.isValidShape) continue
    const pts: { x: number; y: number; z: number }[] = []
    for (let i = 0; i <= 80; i += 1) {
      const theta = (2 * Math.PI * i) / 80
      const c = Math.cos(theta)
      const s = Math.sin(theta)
      const denom = Math.pow(Math.pow(Math.abs(c), n) + Math.pow(Math.abs(s), n), 1 / n)
      const cNorm = c / Math.max(denom, 1e-6)
      const sNorm = s / Math.max(denom, 1e-6)
      const rx = cNorm >= 0 ? sides.edgeRight : sides.edgeLeft
      const ry = sNorm >= 0 ? sides.edgeBottom : sides.edgeTop
      pts.push({ x: center.x + rx * cNorm, y: center.y + ry * sNorm, z: 0 })
    }
    edges.push({ key: `edge-${group.id}`, d: buildSurfaceLine(pts), stroke: SURFACE_PALETTE[group.kind].edge })
  }
  return edges
})

const surfaceGridLines = computed(() => {
  const half = (GRID_N * pitch.value) / 2
  const items: { key: string; d: string }[] = []
  for (let i = 0; i <= GRID_N; i += 1) {
    const v = -half + i * pitch.value
    items.push({
      key: `sgv-${i}`,
      d: buildSurfaceLine([{ x: v, y: -half, z: 0 }, { x: v, y: half, z: 0 }]),
    })
    items.push({
      key: `sgh-${i}`,
      d: buildSurfaceLine([{ x: -half, y: v, z: 0 }, { x: half, y: v, z: 0 }]),
    })
  }
  return items
})

const cellBasePolygon = computed(() => {
  const half = (GRID_N * pitch.value) / 2
  return [
    projectSurface(-half, -half, 0),
    projectSurface(half, -half, 0),
    projectSurface(half, half, 0),
    projectSurface(-half, half, 0),
  ].map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ')
})

const surfaceKindCounts = computed(() => {
  const counts: Record<LensUnitShape, number> = { '1x1': 0, '2x1': 0, '1x2': 0, '2x2': 0 }
  for (const g of lensGroups.value) {
    if (g.isValidShape) counts[g.kind] += 1
  }
  return counts
})

// Top view (XY) plot
const topW = 640
const topH = 400
const topPad = { left: 70, right: 70, top: 30, bottom: 30 }
const topPlotW = computed(() => topW - topPad.left - topPad.right)
const topPlotH = computed(() => topH - topPad.top - topPad.bottom)
const topScale = computed(() => Math.min(topPlotW.value, topPlotH.value) / Math.max(GRID_N * pitch.value, 0.4))
const topOriginX = computed(() => topPad.left + topPlotW.value / 2)
const topOriginY = computed(() => topPad.top + topPlotH.value / 2)

function topXScale(xUm: number) {
  return topOriginX.value + xUm * topScale.value
}

function topYScale(yUm: number) {
  return topOriginY.value + yUm * topScale.value
}

function buildSuperellipsePath(cxUm: number, cyUm: number, halfXUm: number, halfYUm: number, n: number, samples = 96) {
  const pts: string[] = []
  for (let i = 0; i <= samples; i += 1) {
    const theta = (2 * Math.PI * i) / samples
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    const denom = Math.pow(Math.pow(Math.abs(c), n) + Math.pow(Math.abs(s), n), 1 / n)
    const xLocal = (halfXUm * c) / Math.max(denom, 1e-6)
    const yLocal = (halfYUm * s) / Math.max(denom, 1e-6)
    pts.push(`${i === 0 ? 'M' : 'L'} ${topXScale(cxUm + xLocal).toFixed(2)} ${topYScale(cyUm + yLocal).toFixed(2)}`)
  }
  pts.push('Z')
  return pts.join(' ')
}

function buildAsymmetricSuperellipsePath(
  cxUm: number, cyUm: number,
  edgeLeft: number, edgeRight: number, edgeTop: number, edgeBottom: number,
  n: number, samples = 128,
) {
  const pts: string[] = []
  for (let i = 0; i <= samples; i += 1) {
    const theta = (2 * Math.PI * i) / samples
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    const denom = Math.pow(Math.pow(Math.abs(c), n) + Math.pow(Math.abs(s), n), 1 / n)
    const cNorm = c / Math.max(denom, 1e-6)
    const sNorm = s / Math.max(denom, 1e-6)
    const rx = cNorm >= 0 ? edgeRight : edgeLeft
    // y+ is down in our SVG; "top" is y- direction, "bottom" is y+
    const ry = sNorm >= 0 ? edgeBottom : edgeTop
    const xLocal = rx * cNorm
    const yLocal = ry * sNorm
    pts.push(`${i === 0 ? 'M' : 'L'} ${topXScale(cxUm + xLocal).toFixed(2)} ${topYScale(cyUm + yLocal).toFixed(2)}`)
  }
  pts.push('Z')
  return pts.join(' ')
}

function groupCenterUm(group: LensGroup) {
  return {
    x: (group.c0 + group.w / 2 - GRID_N / 2) * pitch.value,
    y: (group.r0 + group.h / 2 - GRID_N / 2) * pitch.value,
  }
}

interface TopGroupRender {
  id: number
  kind: LensUnitShape
  isValid: boolean
  maskPath: string
  reflowPath: string
  finalPath: string
  centerPx: { x: number; y: number }
  labelText: string
  finalHeightUm: number
  sides: SideOffsets
}

const topGroupsRender = computed<TopGroupRender[]>(() => {
  const exponent = shapeExponent.value
  return groupRenderData.value.map(({ group, state, sides, center }) => {
    const maskPath = buildSuperellipsePath(center.x, center.y, state.maskWX / 2, state.maskWY / 2, exponent)
    const reflowPath = buildSuperellipsePath(center.x, center.y, state.reflowWX / 2, state.reflowWY / 2, exponent)
    const finalPath = buildAsymmetricSuperellipsePath(
      center.x, center.y,
      sides.edgeLeft, sides.edgeRight, sides.edgeTop, sides.edgeBottom,
      exponent,
    )
    return {
      id: group.id,
      kind: group.kind,
      isValid: group.isValidShape,
      maskPath,
      reflowPath,
      finalPath,
      centerPx: { x: topXScale(center.x), y: topYScale(center.y) },
      labelText: group.kind,
      finalHeightUm: state.finalHeight,
      sides,
    }
  })
})

const topMaxHeight = computed(() => {
  const heights = topGroupsRender.value.map(g => g.finalHeightUm)
  return heights.length ? Math.max(...heights, 0.05) : 0.05
})

function heatColor(heightUm: number) {
  const t01 = clamp(heightUm / Math.max(topMaxHeight.value, 1e-6), 0, 1)
  // viridis-ish 5-stop interpolation
  const stops: [number, [number, number, number]][] = [
    [0.00, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.50, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.00, [253, 231, 37]],
  ]
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [a, ca] = stops[i]
    const [b, cb] = stops[i + 1]
    if (t01 <= b) {
      const u = (t01 - a) / Math.max(b - a, 1e-6)
      const r = Math.round(ca[0] + (cb[0] - ca[0]) * u)
      const g = Math.round(ca[1] + (cb[1] - ca[1]) * u)
      const bl = Math.round(ca[2] + (cb[2] - ca[2]) * u)
      return `rgb(${r}, ${g}, ${bl})`
    }
  }
  return 'rgb(253, 231, 37)'
}

interface TopGapMarker {
  key: string
  x1: number
  y1: number
  x2: number
  y2: number
  gap: number
  tone: 'good' | 'warn' | 'risk'
}

function gapTone(gap: number): TopGapMarker['tone'] {
  return gap <= 0.08 ? 'good' : 'warn'
}

const topGapMarkers = computed<TopGapMarker[]>(() => {
  const markers: TopGapMarker[] = []
  // For each group, look at right and bottom neighbors only (avoid duplicates).
  for (const { group, neighbors, sides, center } of groupRenderData.value) {
    if (neighbors.right) {
      const y = topYScale(center.y)
      markers.push({
        key: `gx-${group.id}-${neighbors.right.id}`,
        x1: topXScale(center.x + sides.edgeRight),
        y1: y,
        x2: topXScale(center.x + sides.edgeRight + sides.gapRight),
        y2: y,
        gap: sides.gapRight,
        tone: gapTone(sides.gapRight),
      })
    }
    if (neighbors.bottom) {
      const x = topXScale(center.x)
      markers.push({
        key: `gy-${group.id}-${neighbors.bottom.id}`,
        x1: x,
        y1: topYScale(center.y + sides.edgeBottom),
        x2: x,
        y2: topYScale(center.y + sides.edgeBottom + sides.gapBottom),
        gap: sides.gapBottom,
        tone: gapTone(sides.gapBottom),
      })
    }
  }
  return markers
})

const topGridLines = computed(() => {
  const lines: { key: string; x1: number; y1: number; x2: number; y2: number }[] = []
  for (let i = 0; i <= GRID_N; i += 1) {
    const v = (i - GRID_N / 2) * pitch.value
    lines.push({
      key: `gv-${i}`,
      x1: topXScale(v),
      y1: topYScale(-GRID_N / 2 * pitch.value),
      x2: topXScale(v),
      y2: topYScale(GRID_N / 2 * pitch.value),
    })
    lines.push({
      key: `gh-${i}`,
      x1: topXScale(-GRID_N / 2 * pitch.value),
      y1: topYScale(v),
      x2: topXScale(GRID_N / 2 * pitch.value),
      y2: topYScale(v),
    })
  }
  return lines
})

const topLegendStops = computed(() => {
  const max = topMaxHeight.value
  const vals = [0, max * 0.25, max * 0.5, max * 0.75, max]
  return vals.map(v => ({ value: v, color: heatColor(v) }))
})

// Etch response plot
const processW = 640
const processH = 330
const processPad = { left: 58, right: 62, top: 24, bottom: 42 }
const processMaxTime = 140
const processXTicks = [0, 20, 40, 60, 80, 100, 120, 140]
const processGapTicks = computed(() => {
  const maxGap = Math.max(reflowGap.value, initialGap.value * 0.55, 0.08)
  return [0, maxGap * 0.25, maxGap * 0.5, maxGap * 0.75, maxGap].map(v => Number(v.toFixed(2)))
})
const processRetentionTicks = [0, 25, 50, 75, 100]
const processGapMax = computed(() => processGapTicks.value[processGapTicks.value.length - 1])

function processXScale(x: number) {
  const plotW = processW - processPad.left - processPad.right
  return processPad.left + (x / processMaxTime) * plotW
}

function processGapYScale(gap: number) {
  const plotH = processH - processPad.top - processPad.bottom
  return processPad.top + (1 - gap / Math.max(processGapMax.value, 0.01)) * plotH
}

function processRetentionYScale(retentionPct: number) {
  const plotH = processH - processPad.top - processPad.bottom
  return processPad.top + (1 - retentionPct / 100) * plotH
}

const processSamples = computed(() => {
  const rows: { time: number; gap: number; retention: number }[] = []
  for (let time = 0; time <= processMaxTime; time += 2) {
    const state = etchTransferAt(time)
    rows.push({
      time,
      gap: state.gap,
      retention: clamp((state.height / Math.max(reflowHeight.value, 0.001)) * 100, 0, 110),
    })
  }
  return rows
})

const gapCurvePath = computed(() => processSamples.value.map((p, i) => `${i === 0 ? 'M' : 'L'} ${processXScale(p.time).toFixed(2)} ${processGapYScale(p.gap).toFixed(2)}`).join(' '))
const heightCurvePath = computed(() => processSamples.value.map((p, i) => `${i === 0 ? 'M' : 'L'} ${processXScale(p.time).toFixed(2)} ${processRetentionYScale(Math.min(100, p.retention)).toFixed(2)}`).join(' '))
const currentProcessPoint = computed(() => ({
  x: processXScale(etchTime.value),
  gapY: processGapYScale(finalGap.value),
  heightY: processRetentionYScale(Math.min(100, heightRetention.value)),
}))

</script>

<style scoped>
.mlp-container {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.4rem;
  margin: 24px 0;
  background: var(--vp-c-bg-soft);
}

.mlp-container h4 {
  margin: 0 0 4px 0;
  font-size: 1.1em;
  color: var(--vp-c-brand-1);
}

.component-description {
  margin: 0 0 16px 0;
  color: var(--vp-c-text-2);
  font-size: 0.9em;
}

.control-section {
  padding: 12px 0 2px;
  border-top: 1px solid var(--vp-c-divider);
}

.control-section:first-of-type {
  border-top: 0;
  padding-top: 0;
}

.control-heading {
  margin-bottom: 9px;
  font-size: 0.82em;
  font-weight: 700;
  color: var(--vp-c-text-1);
  text-transform: uppercase;
  letter-spacing: 0;
}

.calibration-panel {
  margin: 10px 0 4px;
  padding: 10px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.calibration-panel summary {
  cursor: pointer;
  color: var(--vp-c-text-1);
  font-size: 0.84em;
  font-weight: 700;
}

.calibration-panel p {
  margin: 8px 0 12px;
  color: var(--vp-c-text-2);
  font-size: 0.82em;
  line-height: 1.5;
}

.calibration-controls {
  margin-bottom: 0;
}

.controls-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.slider-group {
  flex: 1;
  min-width: 156px;
}

.slider-group.compact {
  max-width: 230px;
}

.slider-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.85em;
  color: var(--vp-c-text-1);
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
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.ctrl-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--vp-c-brand-1);
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.ctrl-select {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-1);
  font-size: 0.9em;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 14px 0;
}

.metric-card {
  padding: 9px 10px;
  min-height: 58px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.metric-card span {
  display: block;
  color: var(--vp-c-text-2);
  font-size: 0.78em;
  line-height: 1.25;
}

.metric-card strong {
  display: block;
  margin-top: 4px;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 1.02em;
}

.metric-card.accent {
  border-top: 3px solid var(--vp-c-brand-1);
}

.status-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.status-pill {
  padding: 4px 9px;
  border-radius: 999px;
  font-size: 0.78em;
  font-weight: 600;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
}

.status-pill.good {
  color: #1f7a4d;
  border-color: rgba(39, 174, 96, 0.35);
  background: rgba(39, 174, 96, 0.09);
}

.status-pill.warn {
  color: #a85f00;
  border-color: rgba(230, 126, 34, 0.35);
  background: rgba(230, 126, 34, 0.10);
}

.status-pill.risk {
  color: #a93226;
  border-color: rgba(192, 57, 43, 0.35);
  background: rgba(192, 57, 43, 0.10);
}

.tab-row {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.tab-btn {
  padding: 6px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.85em;
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s, background 0.2s;
}

.tab-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.tab-btn.active {
  color: #fff;
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-1);
}

.plot-panel {
  display: flex;
  justify-content: center;
  padding: 8px 0 2px;
}

.main-svg {
  width: 100%;
  max-width: 680px;
  min-height: 260px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.axis-label,
.legend-label,
.plot-label,
.surface-note {
  font-size: 10px;
  fill: var(--vp-c-text-2);
}

.legend-label {
  font-weight: 600;
}

.surface-title {
  font-size: 14px;
  font-weight: 700;
  fill: var(--vp-c-text-1);
}

.surface-note {
  font-size: 10px;
}

.top-group-label {
  font-size: 11px;
  font-weight: 700;
  fill: var(--vp-c-bg);
  paint-order: stroke fill;
  stroke: rgba(0, 0, 0, 0.55);
  stroke-width: 2.2px;
  stroke-linejoin: round;
  pointer-events: none;
}

.custom-grid-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
  margin: 8px 0 4px;
  padding: 10px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
}

.custom-grid-help {
  flex: 1;
  min-width: 200px;
  font-size: 0.82em;
  line-height: 1.55;
  color: var(--vp-c-text-2);
}

.custom-grid-help strong {
  display: block;
  margin-bottom: 4px;
  color: var(--vp-c-text-1);
  font-size: 0.95em;
}

.custom-grid-svg {
  width: 220px;
  height: 220px;
  flex-shrink: 0;
}

.custom-cell {
  cursor: pointer;
  transition: stroke-width 0.15s;
}

.custom-cell:hover {
  stroke: var(--vp-c-brand-1);
}

.custom-cell-label {
  font-size: 10px;
  font-weight: 700;
  fill: var(--vp-c-text-1);
  pointer-events: none;
}

.custom-invalid {
  flex-basis: 100%;
  color: #a93226;
  font-size: 0.82em;
  font-weight: 600;
}

.formula-box {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 7px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.82em;
  line-height: 1.55;
}

.formula-box strong {
  color: var(--vp-c-text-1);
}

@media (max-width: 760px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .mlp-container {
    padding: 1rem;
  }

  .controls-row {
    flex-direction: column;
    gap: 8px;
  }

  .slider-group.compact {
    max-width: none;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }
}
</style>
