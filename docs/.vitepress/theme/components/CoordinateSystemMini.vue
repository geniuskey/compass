<template>
  <div class="coord-mini" role="img" :aria-label="t(
    'Compact coordinate sketch for the BSI pixel stack',
    'BSI 픽셀 스택 좌표계 간단 도식'
  )">
    <svg viewBox="0 0 720 230" class="coord-svg">
      <defs>
        <linearGradient id="coordAir" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#f8fbff" />
          <stop offset="100%" stop-color="#e7f0fb" />
        </linearGradient>
        <linearGradient id="coordLens" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#d9f0ff" />
          <stop offset="100%" stop-color="#8cc8e8" />
        </linearGradient>
        <marker id="coordArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
        </marker>
        <marker id="coordLightArrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
          <path d="M0,0 L9,4.5 L0,9 Z" fill="#f59e0b" />
        </marker>
      </defs>

      <rect x="0" y="0" width="720" height="230" rx="8" class="canvas" />

      <g class="axis">
        <line x1="74" y1="186" x2="74" y2="42" marker-end="url(#coordArrow)" />
        <line x1="74" y1="186" x2="224" y2="186" marker-end="url(#coordArrow)" />
        <circle cx="73.5" cy="186" r="11" />
        <circle cx="73.5" cy="186" r="3" class="axis-dot" />
        <text x="60" y="34">z</text>
        <text x="232" y="191">x</text>
        <text x="42" y="191">y</text>
        <text x="91" y="57" class="axis-note">{{ t('z max', 'z max') }}</text>
        <text x="91" y="177" class="axis-note">{{ t('z min', 'z min') }}</text>
      </g>

      <g transform="translate(286 28)">
        <line x1="74" y1="0" x2="74" y2="39" class="light-ray" marker-end="url(#coordLightArrow)" />
        <line x1="160" y1="0" x2="160" y2="39" class="light-ray" marker-end="url(#coordLightArrow)" />
        <text x="188" y="25" class="light-label">{{ t('light -z', '빛 -z') }}</text>

        <rect x="0" y="9" width="238" height="31" class="layer air layer-air" />
        <path
          d="M20 72 C38 35, 88 35, 106 72 Z M132 72 C150 35, 200 35, 218 72 Z"
          class="layer-lens"
          stroke-width="1.2"
        />
        <rect x="0" y="72" width="238" height="24" class="layer layer-planar" />
        <g>
          <rect x="0" y="96" width="238" height="38" class="layer layer-planar patterned-envelope" opacity="0.82" />
          <path d="M14 134 L48 134 L43 98 L19 98 Z" fill="#f87171" />
          <path d="M70 134 L104 134 L101 98 L73 98 Z" fill="#4ade80" />
          <path d="M126 134 L160 134 L153 98 L133 98 Z" fill="#60a5fa" />
          <path d="M182 134 L216 134 L213 98 L185 98 Z" fill="#4ade80" />
          <rect x="56" y="106" width="6" height="28" class="metal-grid" />
          <rect x="112" y="106" width="6" height="28" class="metal-grid" />
          <rect x="168" y="106" width="6" height="28" class="metal-grid" />
        </g>
        <rect x="0" y="134" width="238" height="10" class="layer layer-barl" />
        <rect x="0" y="144" width="238" height="42" class="layer silicon layer-silicon" />
        <rect x="22" y="154" width="32" height="22" rx="3" class="layer-dti" />
        <rect x="184" y="154" width="32" height="22" rx="3" class="layer-dti" />

        <g class="labels">
          <text x="254" y="30">{{ t('air', 'air') }}</text>
          <text x="254" y="64">{{ t('microlens', 'microlens') }}</text>
          <text x="254" y="89">{{ t('planarization', 'planarization') }}</text>
          <text x="254" y="119">{{ t('color filter / grid', 'color filter / grid') }}</text>
          <text x="254" y="143">BARL</text>
          <text x="254" y="171">{{ t('silicon', 'silicon') }}</text>
        </g>
      </g>
    </svg>
  </div>
</template>

<script setup lang="ts">
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()
</script>

<style scoped>
.coord-mini {
  margin: 18px 0 12px;
  max-width: 760px;
}

.coord-svg {
  width: 100%;
  height: auto;
  display: block;
}

.canvas {
  fill: var(--vp-c-bg-soft);
  stroke: var(--vp-c-divider);
}

.axis {
  color: var(--vp-c-text-1);
}

.axis line {
  stroke: currentColor;
  stroke-width: 1.7;
}

.axis circle {
  fill: var(--vp-c-bg);
  stroke: currentColor;
  stroke-width: 1.5;
}

.axis .axis-dot {
  fill: currentColor;
  stroke: none;
}

.axis text,
.labels text,
.light-label {
  fill: var(--vp-c-text-1);
  font-size: 13px;
  font-weight: 650;
}

.axis-note {
  fill: var(--vp-c-text-2);
  font-size: 11px;
  font-weight: 520;
}

.layer-air { fill: url(#coordAir); }
.layer-lens { fill: url(#coordLens); stroke: #4f9fcb; stroke-width: 1.2; }
.layer-planar { fill: #d9edf7; }
.layer-barl { fill: #fef3c7; }
.layer-silicon { fill: #c7cbd1; }
.layer-dti { fill: #9ca3af; }
.metal-grid { fill: #4b5563; }

:root.dark .layer-air { fill: #1a2230; }
:root.dark .layer-lens { fill: #1a3050; stroke: #3a7aab; stroke-width: 1.2; }
:root.dark .layer-planar { fill: #1e3040; }
:root.dark .layer-barl { fill: #302810; }
:root.dark .layer-silicon { fill: #2a2d32; }
:root.dark .layer-dti { fill: #3a3d42; }
:root.dark .metal-grid { fill: #8a9099; }

.light-ray {
  stroke: #f59e0b;
  stroke-width: 2.5;
  stroke-linecap: round;
}

.light-label {
  fill: #b45309;
}

:root.dark .light-label {
  fill: #fbbf24;
}

.layer {
  stroke: rgba(17, 24, 39, 0.28);
  stroke-width: 1;
}

:root.dark .layer {
  stroke: rgba(255, 255, 255, 0.18);
}

.patterned-envelope {
  stroke-dasharray: 5 4;
}

.silicon {
  stroke: rgba(17, 24, 39, 0.38);
}

:root.dark .silicon {
  stroke: rgba(255, 255, 255, 0.25);
}

.labels text {
  fill: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 560;
}

@media (max-width: 640px) {
  .axis text,
  .labels text,
  .light-label {
    font-size: 12px;
  }
}
</style>
