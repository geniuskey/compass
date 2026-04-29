<script setup lang="ts">
import { ref, computed } from 'vue'

interface Reference {
  id: string
  category: string
  authors: string
  title: string
  journal: string
  year: string
  summary: string
  imageUrl?: string
  link?: string
}

const props = defineProps<{
  references: Reference[]
}>()

const selectedRef = ref<Reference | null>(null)
const isModalOpen = ref(false)

const openModal = (refItem: Reference) => {
  selectedRef.value = refItem
  isModalOpen.value = true
  document.body.style.overflow = 'hidden'
}

const closeModal = () => {
  isModalOpen.value = false
  selectedRef.value = null
  document.body.style.overflow = ''
}

const groupedReferences = computed(() => {
  const groups: Record<string, { name: string; items: Reference[] }> = {}
  props.references.forEach(r => {
    if (!groups[r.category]) {
      groups[r.category] = { name: r.category, items: [] }
    }
    groups[r.category].items.push(r)
  })
  return Object.values(groups)
})
</script>

<template>
  <div class="reference-interactive-list">
    <div v-for="(category, index) in groupedReferences" :key="index" class="category-group">
      <h2 class="category-title">{{ category.name }}</h2>
      <div class="cards-grid">
        <div v-for="item in category.items" :key="item.id" class="ref-card" @click="openModal(item)">
          <div class="ref-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          </div>
          <div class="ref-content">
            <div class="ref-title">{{ item.title }}</div>
            <div class="ref-authors">{{ item.authors }}</div>
            <div class="ref-journal">{{ item.journal }}, {{ item.year }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal -->
    <Teleport to="body">
      <div v-if="isModalOpen" class="modal-overlay" @click="closeModal">
        <div class="modal-window" @click.stop>
          <button class="close-btn" @click="closeModal" aria-label="Close modal">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
          
          <div class="modal-header">
            <h3>{{ selectedRef?.title }}</h3>
            <p class="modal-authors">{{ selectedRef?.authors }}</p>
            <p class="modal-journal">{{ selectedRef?.journal }} ({{ selectedRef?.year }})</p>
            <a v-if="selectedRef?.link" :href="selectedRef.link" target="_blank" class="doi-link">View Paper ↗</a>
          </div>

          <div class="modal-body">
            <div v-if="selectedRef?.imageUrl" class="modal-image-container">
               <img :src="selectedRef.imageUrl" :alt="selectedRef.title" class="modal-image" />
            </div>
            
            <div class="modal-summary">
              <h4>Abstract & Summary</h4>
              <div class="summary-content" v-html="selectedRef?.summary"></div>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.reference-interactive-list {
  margin-top: 2rem;
}

.category-group {
  margin-bottom: 3rem;
}

.category-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem;
}

.ref-card {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: 1.2rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-soft);
  cursor: pointer;
  transition: all 0.2s ease;
}

.ref-card:hover {
  border-color: var(--vp-c-brand);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.ref-icon {
  color: var(--vp-c-brand);
  flex-shrink: 0;
  margin-top: 0.2rem;
}

.ref-title {
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 0.4rem;
  line-height: 1.3;
  color: var(--vp-c-text-1);
}

.ref-authors {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.3rem;
}

.ref-journal {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
  font-style: italic;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 9999;
  padding: 2rem;
}

.modal-window {
  background-color: var(--vp-c-bg);
  border-radius: 12px;
  width: 100%;
  max-width: 800px;
  max-height: 90vh;
  overflow-y: auto;
  position: relative;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
}

.close-btn {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: none;
  border: none;
  color: var(--vp-c-text-2);
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s;
}

.close-btn:hover {
  background-color: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.modal-header {
  padding: 2rem 2rem 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  background-color: var(--vp-c-bg-soft);
}

.modal-header h3 {
  font-size: 1.4rem;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 0.8rem;
  padding-right: 2rem;
}

.modal-authors {
  font-size: 0.95rem;
  color: var(--vp-c-text-2);
  margin-bottom: 0.4rem;
}

.modal-journal {
  font-size: 0.9rem;
  color: var(--vp-c-text-3);
  font-style: italic;
  margin-bottom: 1rem;
}

.doi-link {
  display: inline-block;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--vp-c-brand);
  text-decoration: none;
}

.doi-link:hover {
  text-decoration: underline;
}

.modal-body {
  padding: 2rem;
}

.modal-image-container {
  margin-bottom: 2rem;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
  background-color: var(--vp-c-bg-mute);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1rem;
}

.modal-image {
  max-width: 100%;
  max-height: 400px;
  object-fit: contain;
}

.modal-summary h4 {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: var(--vp-c-text-1);
}

.summary-content {
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

.summary-content p {
  margin-bottom: 1rem;
}

.summary-content ul {
  margin-left: 1.5rem;
  margin-bottom: 1rem;
}

.summary-content li {
  margin-bottom: 0.5rem;
}
</style>