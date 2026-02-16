import { ref, onMounted, onUnmounted } from 'vue'

export function useFullscreen() {
  const isFullscreen = ref(false)

  function toggleFullscreen() {
    isFullscreen.value = !isFullscreen.value
    if (typeof document !== 'undefined')
      document.body.style.overflow = isFullscreen.value ? 'hidden' : ''
  }

  let handler: ((e: KeyboardEvent) => void) | null = null

  onMounted(() => {
    handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen.value) {
        e.preventDefault()
        toggleFullscreen()
      }
    }
    document.addEventListener('keydown', handler)
  })

  onUnmounted(() => {
    if (handler) document.removeEventListener('keydown', handler)
    if (typeof document !== 'undefined') document.body.style.overflow = ''
  })

  return { isFullscreen, toggleFullscreen }
}
