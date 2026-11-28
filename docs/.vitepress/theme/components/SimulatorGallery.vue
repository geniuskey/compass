<template>
  <div class="sim-gallery">
    <div class="sim-gallery-toolbar">
      <input
        v-model="query"
        type="search"
        class="sim-gallery-search"
        :placeholder="t(`Filter ${simulatorCount} simulators…`, `시뮬레이터 ${simulatorCount}개 검색…`)"
        :aria-label="t('Filter simulators', '시뮬레이터 검색')"
      />
      <div class="sim-gallery-chips" role="group" :aria-label="t('Category', '카테고리')">
        <button
          type="button"
          class="sim-chip"
          :class="{ active: active === 'all' }"
          :aria-pressed="active === 'all'"
          @click="active = 'all'"
        >
          {{ t('All', '전체') }}
        </button>
        <button
          v-for="cat in simulatorCategories"
          :key="cat.id"
          type="button"
          class="sim-chip"
          :class="{ active: active === cat.id }"
          :aria-pressed="active === cat.id"
          @click="active = cat.id"
        >
          {{ isKo ? cat.ko : cat.en }}
        </button>
      </div>
    </div>

    <section v-for="cat in filtered" :key="cat.id" class="sim-gallery-group">
      <h3 :id="`sim-${cat.id}`" class="sim-gallery-heading">
        {{ isKo ? cat.ko : cat.en }}
        <span class="sim-gallery-count ignore-header">{{ cat.items.length }}</span>
      </h3>
      <div class="sim-gallery-grid">
        <a
          v-for="s in cat.items"
          :key="s.slug"
          class="sim-card"
          :href="withBase(`${isKo ? '/ko' : ''}/simulator/${s.slug}`)"
        >
          <span class="sim-card-icon" aria-hidden="true">{{ s.icon }}</span>
          <span class="sim-card-body">
            <span class="sim-card-title">{{ isKo ? s.ko.title : s.en.title }}</span>
            <span class="sim-card-desc">{{ isKo ? s.ko.desc : s.en.desc }}</span>
          </span>
        </a>
      </div>
    </section>

    <p v-if="filtered.length === 0" class="sim-gallery-empty">
      {{ t('No simulator matches that filter.', '조건에 맞는 시뮬레이터가 없습니다.') }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { withBase } from 'vitepress'
import { simulatorCategories, simulatorCount } from '../../simulators'
import { useLocale } from '../composables/useLocale'

const { isKo, t } = useLocale()
const query = ref('')
const active = ref<string>('all')

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  return simulatorCategories
    .filter((cat) => active.value === 'all' || cat.id === active.value)
    .map((cat) => ({
      ...cat,
      items: cat.items.filter((s) => {
        if (!q) return true
        const hay = `${s.en.title} ${s.en.desc} ${s.ko.title} ${s.ko.desc} ${s.slug}`.toLowerCase()
        return hay.includes(q)
      }),
    }))
    .filter((cat) => cat.items.length > 0)
})
</script>

<style scoped>
.sim-gallery {
  margin: 24px 0 8px;
}

.sim-gallery-toolbar {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 8px;
}

.sim-gallery-search {
  width: 100%;
  padding: 10px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-size: 15px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.sim-gallery-search:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}

.sim-gallery-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.sim-chip {
  padding: 4px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 500;
  line-height: 22px;
  cursor: pointer;
  transition: color 0.2s, border-color 0.2s, background 0.2s;
}

.sim-chip:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

.sim-chip.active {
  color: var(--vp-c-white);
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

.sim-gallery-group {
  margin-top: 28px;
}

.vp-doc .sim-gallery-heading {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 12px;
  padding: 0;
  font-size: 17px;
}

.sim-gallery-count {
  padding: 0 8px;
  border-radius: 999px;
  background: var(--vp-c-default-soft);
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 600;
  line-height: 20px;
}

.sim-gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
}

.vp-doc a.sim-card {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  color: inherit;
  text-decoration: none;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}

.vp-doc a.sim-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: var(--compass-shadow-hover);
  transform: translateY(-2px);
}

.sim-card-icon {
  flex: none;
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 9px;
  background: var(--vp-c-bg);
  font-size: 18px;
}

.sim-card-body {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.sim-card-title {
  color: var(--vp-c-text-1);
  font-size: 14.5px;
  font-weight: 600;
  line-height: 1.4;
}

.vp-doc a.sim-card:hover .sim-card-title {
  color: var(--vp-c-brand-1);
}

.sim-card-desc {
  color: var(--vp-c-text-2);
  font-size: 13px;
  line-height: 1.5;
}

.sim-gallery-empty {
  color: var(--vp-c-text-2);
}

@media (prefers-reduced-motion: reduce) {
  .vp-doc a.sim-card,
  .vp-doc a.sim-card:hover {
    transition: none;
    transform: none;
  }
}
</style>
