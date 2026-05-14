import { computed } from 'vue'
import { useRoute } from 'vitepress'

export function useLocale() {
  const route = useRoute()
  const isKo = computed(() => /(^|\/)ko(\/|$)/.test(route.path))

  function t(en: string, ko: string): string {
    return isKo.value ? ko : en
  }

  return { isKo, t }
}
