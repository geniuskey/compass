<template>
  <div class="feature-showcase">
    <div class="feature-grid">
      <div
        v-for="feature in features"
        :key="feature.id"
        class="feature-card"
        @mouseenter="hoveredId = feature.id"
        @mouseleave="hoveredId = null"
      >
        <div class="feature-icon" :class="{ hovered: hoveredId === feature.id }">
          <!-- Multi-Solver -->
          <svg v-if="feature.id === 'multi-solver'" viewBox="0 0 40 40" class="icon-svg">
            <rect x="2" y="4" width="14" height="14" rx="3" fill="var(--vp-c-brand-1)" opacity="0.8" />
            <rect x="24" y="4" width="14" height="14" rx="3" fill="var(--vp-c-brand-2)" opacity="0.8" />
            <rect x="13" y="22" width="14" height="14" rx="3" fill="var(--vp-c-brand-3)" opacity="0.8" />
            <line x1="16" y1="11" x2="24" y2="11" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="2,2" />
            <line x1="9" y1="18" x2="20" y2="22" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="2,2" />
            <line x1="31" y1="18" x2="20" y2="22" stroke="var(--vp-c-text-3)" stroke-width="1.5" stroke-dasharray="2,2" />
          </svg>

          <!-- GPU Accelerated -->
          <svg v-else-if="feature.id === 'gpu'" viewBox="0 0 40 40" class="icon-svg">
            <rect x="4" y="8" width="32" height="22" rx="3" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
            <rect x="8" y="12" width="6" height="6" rx="1" fill="var(--vp-c-brand-1)" opacity="0.6" />
            <rect x="17" y="12" width="6" height="6" rx="1" fill="var(--vp-c-brand-1)" opacity="0.7" />
            <rect x="26" y="12" width="6" height="6" rx="1" fill="var(--vp-c-brand-1)" opacity="0.8" />
            <rect x="8" y="21" width="6" height="6" rx="1" fill="var(--vp-c-brand-2)" opacity="0.6" />
            <rect x="17" y="21" width="6" height="6" rx="1" fill="var(--vp-c-brand-2)" opacity="0.7" />
            <rect x="26" y="21" width="6" height="6" rx="1" fill="var(--vp-c-brand-2)" opacity="0.8" />
            <line x1="10" y1="33" x2="10" y2="36" stroke="var(--vp-c-text-3)" stroke-width="1.5" />
            <line x1="20" y1="33" x2="20" y2="36" stroke="var(--vp-c-text-3)" stroke-width="1.5" />
            <line x1="30" y1="33" x2="30" y2="36" stroke="var(--vp-c-text-3)" stroke-width="1.5" />
          </svg>

          <!-- Differentiable -->
          <svg v-else-if="feature.id === 'differentiable'" viewBox="0 0 40 40" class="icon-svg">
            <path d="M 5 30 Q 12 30, 16 22 T 28 10 T 36 8"
              fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2.5" stroke-linecap="round" />
            <line x1="22" y1="8" x2="36" y2="8" stroke="var(--vp-c-brand-2)" stroke-width="1.5" stroke-dasharray="3,2" />
            <text x="20" y="38" font-size="12" font-weight="700" fill="var(--vp-c-brand-1)" font-family="serif">
              dJ/dx
            </text>
          </svg>

          <!-- Cross-Validation -->
          <svg v-else-if="feature.id === 'cross-validation'" viewBox="0 0 40 40" class="icon-svg">
            <circle cx="14" cy="20" r="10" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" opacity="0.7" />
            <circle cx="26" cy="20" r="10" fill="none" stroke="var(--vp-c-brand-2)" stroke-width="2" opacity="0.7" />
            <path d="M 19.5 13 A 10 10 0 0 1 19.5 27 A 10 10 0 0 1 19.5 13"
              fill="var(--vp-c-brand-soft)" stroke="none" />
            <text x="20" y="23" font-size="8" font-weight="700" fill="var(--vp-c-brand-1)" text-anchor="middle">
              OK
            </text>
          </svg>

          <!-- Rich Analysis -->
          <svg v-else-if="feature.id === 'analysis'" viewBox="0 0 40 40" class="icon-svg">
            <line x1="6" y1="34" x2="6" y2="6" stroke="var(--vp-c-text-3)" stroke-width="1.5" />
            <line x1="6" y1="34" x2="36" y2="34" stroke="var(--vp-c-text-3)" stroke-width="1.5" />
            <path d="M 8 30 L 14 22 L 20 26 L 26 14 L 34 10"
              fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            <path d="M 8 28 L 14 24 L 20 20 L 26 16 L 34 18"
              fill="none" stroke="var(--vp-c-brand-2)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="4,2" />
            <circle cx="26" cy="14" r="2.5" fill="var(--vp-c-brand-1)" />
          </svg>

          <!-- Open Source -->
          <svg v-else-if="feature.id === 'open-source'" viewBox="0 0 40 40" class="icon-svg">
            <circle cx="20" cy="16" r="8" fill="none" stroke="var(--vp-c-brand-1)" stroke-width="2" />
            <circle cx="20" cy="16" r="3" fill="var(--vp-c-brand-1)" />
            <line x1="14" y1="22" x2="8" y2="34" stroke="var(--vp-c-brand-1)" stroke-width="2" stroke-linecap="round" />
            <line x1="26" y1="22" x2="32" y2="34" stroke="var(--vp-c-brand-1)" stroke-width="2" stroke-linecap="round" />
            <circle cx="8" cy="34" r="3" fill="var(--vp-c-brand-2)" />
            <circle cx="32" cy="34" r="3" fill="var(--vp-c-brand-2)" />
          </svg>
        </div>

        <h3 class="feature-title">{{ feature.title }}</h3>
        <p class="feature-desc">{{ feature.description }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useLocale } from '../composables/useLocale'

const { t } = useLocale()
const hoveredId = ref(null)

const features = computed(() => [
  {
    id: 'multi-solver',
    title: t('Multi-Solver Engine', '멀티 솔버 엔진'),
    description: t(
      'Run the same pixel structure through RCWA (torcwa, grcwa, meent, fmmax) and FDTD (flaport, Meep, fdtdz) solvers from a single YAML config. Compare results head-to-head.',
      '하나의 YAML 설정으로 동일한 픽셀 구조를 RCWA(torcwa, grcwa, meent, fmmax)와 FDTD(flaport, Meep, fdtdz) 솔버로 실행합니다. 결과를 직접 비교할 수 있습니다.'
    ),
  },
  {
    id: 'gpu',
    title: t('GPU Accelerated', 'GPU 가속'),
    description: t(
      'Leverage PyTorch and JAX GPU backends for massively parallel wavelength sweeps. Achieve 10-100x speedup over CPU-only solvers on large parameter spaces.',
      'PyTorch 및 JAX GPU 백엔드를 활용하여 대규모 병렬 파장 스윕을 수행합니다. 넓은 파라미터 공간에서 CPU 전용 솔버 대비 10-100배 속도 향상을 달성합니다.'
    ),
  },
  {
    id: 'differentiable',
    title: t('Differentiable Simulation', '미분 가능 시뮬레이션'),
    description: t(
      'Automatic differentiation through the solver enables gradient-based inverse design. Optimize microlens profiles, BARL stacks, and color filter thicknesses directly.',
      '솔버를 통한 자동 미분으로 기울기 기반 역설계가 가능합니다. 마이크로렌즈 프로파일, BARL 스택, 컬러 필터 두께를 직접 최적화할 수 있습니다.'
    ),
  },
  {
    id: 'cross-validation',
    title: t('Cross-Validation', '교차 검증'),
    description: t(
      'Built-in solver comparison framework ensures physics accuracy. Compare energy balance (R+T+A=1), QE spectra, and field distributions across solver backends.',
      '내장된 솔버 비교 프레임워크로 물리적 정확성을 보장합니다. 에너지 균형(R+T+A=1), QE 스펙트럼, 필드 분포를 솔버 간 비교합니다.'
    ),
  },
  {
    id: 'analysis',
    title: t('Rich Analysis', '풍부한 분석'),
    description: t(
      'Compute QE per pixel per wavelength, crosstalk matrices, energy balance, and field distributions. Plot results with built-in matplotlib and 3D PyVista viewers.',
      '픽셀별/파장별 QE, 크로스토크 행렬, 에너지 균형, 필드 분포를 계산합니다. 내장된 matplotlib 및 3D PyVista 뷰어로 결과를 시각화합니다.'
    ),
  },
  {
    id: 'open-source',
    title: t('Open Source', '오픈 소스'),
    description: t(
      'MIT licensed. Fully documented with theory guides, cookbooks, and API references. Built on established open-source EM solver ecosystems.',
      'MIT 라이선스. 이론 가이드, 쿡북, API 레퍼런스가 포함된 완전한 문서를 제공합니다. 확립된 오픈 소스 EM 솔버 생태계 위에 구축되었습니다.'
    ),
  },
])
</script>

<style scoped>
.feature-showcase {
  margin: 24px 0;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

@media (max-width: 960px) {
  .feature-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
}

.feature-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
  cursor: default;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 16px rgba(52, 81, 178, 0.08);
  transform: translateY(-2px);
}

:root.dark .feature-card:hover {
  box-shadow: 0 4px 16px rgba(107, 141, 229, 0.1);
}

.feature-icon {
  width: 48px;
  height: 48px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  transition: border-color 0.25s, background 0.25s;
}

.feature-icon.hovered {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.icon-svg {
  width: 32px;
  height: 32px;
}

.feature-title {
  margin: 0 0 8px 0;
  font-size: 1em;
  font-weight: 700;
  color: var(--vp-c-text-1);
  line-height: 1.3;
}

.feature-desc {
  margin: 0;
  font-size: 0.88em;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}
</style>
