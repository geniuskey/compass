<template>
  <div class="cta-grid">
    <a v-for="c in cards" :key="c.link" class="cta-card" :href="withBase(prefix + c.link)">
      <strong>{{ isKo ? c.ko[0] : c.en[0] }}</strong>
      <span>{{ isKo ? c.ko[1] : c.en[1] }}</span>
    </a>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { useLocale } from '../composables/useLocale'

// Raw <a href="/..."> in Markdown is not rewritten with the site base
// (/compass/), so home-page link cards are rendered here with withBase().
const { isKo } = useLocale()
const prefix = computed(() => (isKo.value ? '/ko' : ''))

const cards = [
  { link: '/guide/quickstart', en: ['Quick Start', 'Run your first simulation in minutes'], ko: ['빠른 시작', '몇 분 안에 첫 시뮬레이션 실행'] },
  { link: '/guide/installation', en: ['Installation', 'Set up COMPASS and solver backends'], ko: ['설치 가이드', 'COMPASS와 솔버 백엔드 설치'] },
  { link: '/guide/choosing-solver', en: ['Choosing a Solver', 'RCWA, FDTD or TMM — which one when'], ko: ['솔버 선택', 'RCWA, FDTD, TMM 중 무엇을 언제'] },
  { link: '/cookbook/bsi-2x2-basic', en: ['Cookbook', 'Practical recipes for common tasks'], ko: ['쿡북', '자주 하는 작업을 위한 레시피'] },
  { link: '/reference/', en: ['API Reference', 'PixelStack, MaterialDB, SolverBase, config'], ko: ['API 레퍼런스', 'PixelStack, MaterialDB, SolverBase, 설정'] },
  { link: '/reports/', en: ['Validation Reports', 'Cross-solver parity, convergence, benchmarks'], ko: ['검증 리포트', '솔버 간 일치도, 수렴, 벤치마크'] },
]
</script>
