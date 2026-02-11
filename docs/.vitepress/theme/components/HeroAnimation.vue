<template>
  <div class="hero-animation-container">
    <svg :viewBox="`0 0 ${svgW} ${svgH}`" class="hero-svg" aria-hidden="true">
      <defs>
        <!-- Light ray gradient -->
        <linearGradient id="heroRayGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#fbbf24" stop-opacity="0.9" />
          <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.0" />
        </linearGradient>

        <!-- Microlens gradient -->
        <radialGradient id="heroLensGrad" cx="50%" cy="90%" r="70%">
          <stop offset="0%" stop-color="var(--hero-lens-inner)" stop-opacity="0.95" />
          <stop offset="100%" stop-color="var(--hero-lens-outer)" stop-opacity="0.75" />
        </radialGradient>

        <!-- Silicon gradient -->
        <linearGradient id="heroSiGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--hero-si-top)" />
          <stop offset="100%" stop-color="var(--hero-si-bot)" />
        </linearGradient>

        <!-- Glow filter -->
        <filter id="heroGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <!-- Soft glow for layers -->
        <filter id="layerGlow" x="-5%" y="-5%" width="110%" height="110%">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <!-- Background grid (subtle) -->
      <g opacity="0.08">
        <line v-for="i in 15" :key="'gv'+i"
          :x1="i * 50" y1="0" :x2="i * 50" :y2="svgH"
          stroke="var(--vp-c-text-1)" stroke-width="0.5" />
        <line v-for="i in 8" :key="'gh'+i"
          x1="0" :y1="i * 50" :x2="svgW" :y2="i * 50"
          stroke="var(--vp-c-text-1)" stroke-width="0.5" />
      </g>

      <!-- Animated light rays -->
      <g class="light-rays">
        <line v-for="(ray, idx) in lightRays" :key="'ray'+idx"
          :x1="ray.x1" :y1="ray.y1" :x2="ray.x2" :y2="ray.y2"
          stroke="#fbbf24" :stroke-width="ray.w" :opacity="ray.opacity"
          :class="'ray-anim ray-anim-' + (idx % 3)"
          stroke-linecap="round" />
      </g>

      <!-- Pixel stack cross-section -->
      <g :transform="`translate(${stackX}, ${stackY})`">
        <!-- Air region label -->
        <text :x="stackW / 2" y="-12" text-anchor="middle" class="region-label">
          {{ t('Air', '공기') }}
        </text>

        <!-- Microlens (elliptical dome) -->
        <ellipse
          :cx="stackW / 2" :cy="lensY + lensH * 0.3"
          :rx="stackW * 0.42" :ry="lensH * 0.7"
          fill="url(#heroLensGrad)"
          class="layer-pulse layer-pulse-0"
        />
        <text :x="stackW / 2" :y="lensY + lensH * 0.45" text-anchor="middle" class="layer-label">
          {{ t('Microlens', '마이크로렌즈') }}
        </text>

        <!-- Planarization layer -->
        <rect x="0" :y="planarY" :width="stackW" :height="planarH"
          fill="var(--hero-planar)" rx="2"
          class="layer-pulse layer-pulse-1" />
        <text :x="stackW / 2" :y="planarY + planarH / 2 + 4" text-anchor="middle" class="layer-label-sm">
          SiO2
        </text>

        <!-- Color filter (Bayer pattern: R G G B) -->
        <g>
          <rect :x="0" :y="cfY" :width="stackW * 0.25" :height="cfH"
            fill="var(--hero-cf-red)" class="layer-pulse layer-pulse-2" />
          <rect :x="stackW * 0.25" :y="cfY" :width="stackW * 0.25" :height="cfH"
            fill="var(--hero-cf-green)" class="layer-pulse layer-pulse-2" />
          <rect :x="stackW * 0.5" :y="cfY" :width="stackW * 0.25" :height="cfH"
            fill="var(--hero-cf-green)" class="layer-pulse layer-pulse-2" />
          <rect :x="stackW * 0.75" :y="cfY" :width="stackW * 0.25" :height="cfH"
            fill="var(--hero-cf-blue)" class="layer-pulse layer-pulse-2" />
          <!-- Metal grid lines -->
          <line v-for="i in 3" :key="'mg'+i"
            :x1="stackW * 0.25 * i" :y1="cfY"
            :x2="stackW * 0.25 * i" :y2="cfY + cfH"
            stroke="var(--hero-metal)" stroke-width="2" />
          <text :x="stackW / 2" :y="cfY + cfH / 2 + 4" text-anchor="middle" class="layer-label">
            {{ t('Color Filter', '컬러 필터') }}
          </text>
        </g>

        <!-- BARL -->
        <rect x="0" :y="barlY" :width="stackW" :height="barlH"
          fill="var(--hero-barl)" rx="1" />

        <!-- Silicon body with DTI -->
        <rect x="0" :y="siY" :width="stackW" :height="siH"
          fill="url(#heroSiGrad)" rx="2"
          class="layer-pulse layer-pulse-3" />

        <!-- DTI trenches -->
        <rect v-for="i in 3" :key="'dti'+i"
          :x="stackW * 0.25 * i - 3" :y="siY + 2"
          width="6" :height="siH - 4"
          fill="var(--hero-dti)" rx="1" opacity="0.7" />

        <!-- Photodiode regions -->
        <rect v-for="i in 4" :key="'pd'+i"
          :x="stackW * 0.25 * (i - 1) + stackW * 0.06"
          :y="siY + siH * 0.2"
          :width="stackW * 0.25 - stackW * 0.12"
          :height="siH * 0.6"
          fill="none" stroke="var(--hero-pd-stroke)" stroke-width="1"
          stroke-dasharray="3,2" rx="2" opacity="0.6" />

        <text :x="stackW / 2" :y="siY + siH / 2 + 4" text-anchor="middle" class="layer-label layer-label-light">
          {{ t('Silicon', '실리콘') }}
        </text>

        <!-- Substrate -->
        <rect x="0" :y="subY" :width="stackW" :height="subH"
          fill="var(--hero-sub)" rx="2" />
        <text :x="stackW / 2" :y="subY + subH / 2 + 4" text-anchor="middle" class="layer-label layer-label-light">
          {{ t('Metal / Substrate', '금속 / 기판') }}
        </text>

        <!-- Dimension annotations (right side) -->
        <line :x1="stackW + 15" :y1="lensY" :x2="stackW + 15" :y2="subY + subH"
          stroke="var(--vp-c-text-3)" stroke-width="0.8" />
        <line :x1="stackW + 12" :y1="lensY" :x2="stackW + 18" :y2="lensY"
          stroke="var(--vp-c-text-3)" stroke-width="0.8" />
        <line :x1="stackW + 12" :y1="subY + subH" :x2="stackW + 18" :y2="subY + subH"
          stroke="var(--vp-c-text-3)" stroke-width="0.8" />
        <text :x="stackW + 22" :y="(lensY + subY + subH) / 2 + 4" class="dim-label">
          ~4 um
        </text>
      </g>

      <!-- Labels: arrows pointing to features -->
      <g class="annotation-group">
        <!-- QE annotation -->
        <g :transform="`translate(${stackX + stackW + 65}, ${stackY + siY + 20})`">
          <rect x="-4" y="-14" width="110" height="50" rx="6"
            fill="var(--vp-c-bg-soft)" stroke="var(--vp-c-brand-1)" stroke-width="1" opacity="0.9" />
          <text x="2" y="2" class="annotation-title">
            {{ t('Quantum Efficiency', '양자 효율') }}
          </text>
          <text x="2" y="18" class="annotation-value">QE = 0 ~ 100%</text>
          <text x="2" y="30" class="annotation-sub">R + T + A = 1</text>
        </g>

        <!-- Wavelength annotation -->
        <g :transform="`translate(${stackX - 130}, ${stackY + 10})`">
          <rect x="-4" y="-14" width="115" height="40" rx="6"
            fill="var(--vp-c-bg-soft)" stroke="#f59e0b" stroke-width="1" opacity="0.9" />
          <text x="2" y="2" class="annotation-title" style="fill: #f59e0b;">
            {{ t('Wavelength Sweep', '파장 스윕') }}
          </text>
          <text x="2" y="18" class="annotation-value">400 - 700 nm</text>
        </g>
      </g>
    </svg>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()

const svgW = 700
const svgH = 360

// Pixel stack dimensions and positioning
const stackW = 280
const stackX = (svgW - stackW) / 2 + 10
const stackY = 40

const lensH = 40
const lensY = 0
const planarH = 25
const planarY = lensY + lensH
const cfH = 35
const cfY = planarY + planarH
const barlH = 8
const barlY = cfY + cfH
const siH = 130
const siY = barlY + barlH
const subH = 30
const subY = siY + siH

// Light rays configuration
const lightRays = computed(() => {
  const rays = []
  const cx = stackX + stackW / 2
  const targetY = stackY + lensY + lensH * 0.3

  for (let i = 0; i < 7; i++) {
    const spread = (i - 3) * 28
    rays.push({
      x1: cx + spread * 2.5,
      y1: 0,
      x2: cx + spread * 0.3,
      y2: targetY,
      w: 2.5 - Math.abs(i - 3) * 0.3,
      opacity: 0.7 - Math.abs(i - 3) * 0.08,
    })
  }
  return rays
})
</script>

<style scoped>
.hero-animation-container {
  --hero-lens-inner: #c8e6f5;
  --hero-lens-outer: #90c5e3;
  --hero-planar: #d5dbdb;
  --hero-cf-red: #e74c3c;
  --hero-cf-green: #27ae60;
  --hero-cf-blue: #2980b9;
  --hero-metal: #7f8c8d;
  --hero-barl: #8e44ad;
  --hero-si-top: #5d6d7e;
  --hero-si-bot: #34495e;
  --hero-dti: #aed6f1;
  --hero-sub: #1c2833;
  --hero-pd-stroke: #f1c40f;

  max-width: 700px;
  margin: -20px auto 32px auto;
  padding: 0 16px;
}

:root.dark .hero-animation-container {
  --hero-lens-inner: #2a5a7a;
  --hero-lens-outer: #1a3d5c;
  --hero-planar: #3d4e5e;
  --hero-cf-red: #c0392b;
  --hero-cf-green: #1e8449;
  --hero-cf-blue: #1f618d;
  --hero-metal: #566573;
  --hero-barl: #6c3483;
  --hero-si-top: #4a5a6a;
  --hero-si-bot: #2c3e50;
  --hero-dti: #5499c7;
  --hero-sub: #0e1a24;
  --hero-pd-stroke: #d4ac0d;
}

.hero-svg {
  width: 100%;
  display: block;
}

/* Animated light rays */
.ray-anim {
  animation: rayPulse 3s ease-in-out infinite;
}
.ray-anim-0 { animation-delay: 0s; }
.ray-anim-1 { animation-delay: 1s; }
.ray-anim-2 { animation-delay: 2s; }

@keyframes rayPulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 0.85; }
}

/* Subtle layer glow pulse */
.layer-pulse {
  animation: layerGlow 4s ease-in-out infinite;
}
.layer-pulse-0 { animation-delay: 0s; }
.layer-pulse-1 { animation-delay: 0.5s; }
.layer-pulse-2 { animation-delay: 1s; }
.layer-pulse-3 { animation-delay: 1.5s; }

@keyframes layerGlow {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.08); }
}

/* Labels */
.region-label {
  font-size: 11px;
  fill: var(--vp-c-text-3);
  font-weight: 500;
}

.layer-label {
  font-size: 11px;
  fill: var(--vp-c-text-1);
  font-weight: 600;
  pointer-events: none;
}

.layer-label-sm {
  font-size: 9px;
  fill: var(--vp-c-text-2);
  font-weight: 500;
  pointer-events: none;
}

.layer-label-light {
  fill: #ecf0f1;
}

.dim-label {
  font-size: 9px;
  fill: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.annotation-title {
  font-size: 10px;
  font-weight: 700;
  fill: var(--vp-c-brand-1);
}

.annotation-value {
  font-size: 10px;
  font-weight: 500;
  fill: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

.annotation-sub {
  font-size: 9px;
  fill: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

/* Responsive */
@media (max-width: 640px) {
  .hero-animation-container {
    margin: -10px auto 20px auto;
  }
  .annotation-group {
    display: none;
  }
}
</style>
