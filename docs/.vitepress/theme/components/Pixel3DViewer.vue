<template>
  <div class="pixel-3d-viewer">
    <div class="controls-row">
      <label class="ctrl-item">
        <input type="checkbox" v-model="exploded" />
        <span>Exploded View</span>
      </label>
      <label class="ctrl-item">
        <input type="checkbox" v-model="autoRotate" />
        <span>Auto Rotate</span>
      </label>
      <button class="ctrl-btn" @click="resetView">Reset View</button>
      <div class="layer-toggles">
        <label v-for="l in layerDefs" :key="l.id" class="toggle-item">
          <input type="checkbox" v-model="l.visible" />
          <span class="toggle-sw" :style="{ background: l.color }"></span>
          <span class="toggle-lbl">{{ l.label }}</span>
        </label>
      </div>
    </div>

    <div ref="canvasHost" class="viewer-host">
      <div class="hint">Drag: rotate · Right-drag: pan · Wheel: zoom</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onBeforeUnmount, watch } from 'vue'

interface LayerDef {
  id: string
  label: string
  color: string
  zBot: number
  zTop: number
  thickness: string
  material: string
  visible: boolean
}

const layerDefs = reactive<LayerDef[]>([
  { id: 'silicon', label: 'Silicon', color: '#5d6d7e', zBot: 0, zTop: 3.0, thickness: '3.0', material: 'Si', visible: true },
  { id: 'barl', label: 'BARL', color: '#8e44ad', zBot: 3.0, zTop: 3.08, thickness: '0.08', material: 'SiO2/HfO2/SiO2/Si3N4', visible: true },
  { id: 'colorfilter', label: 'Color Filter', color: '#27ae60', zBot: 3.08, zTop: 3.68, thickness: '0.6', material: 'Organic dye', visible: true },
  { id: 'planarization', label: 'Planarization', color: '#d5dbdb', zBot: 3.68, zTop: 3.98, thickness: '0.3', material: 'SiO2', visible: true },
  { id: 'microlens', label: 'Microlens', color: '#dda0dd', zBot: 3.98, zTop: 4.58, thickness: '0.6', material: 'Polymer (n=1.56)', visible: true },
  { id: 'air', label: 'Air', color: '#d6eaf8', zBot: 4.58, zTop: 5.58, thickness: '1.0', material: 'Air', visible: false },
])

const exploded = ref(false)
const autoRotate = ref(false)
const canvasHost = ref<HTMLDivElement | null>(null)

// Three.js context (kept outside reactive system)
let three: {
  THREE: typeof import('three')
  scene: import('three').Scene
  camera: import('three').PerspectiveCamera
  renderer: import('three').WebGLRenderer
  controls: import('three').EventDispatcher & {
    update: () => void
    dispose: () => void
    autoRotate: boolean
    autoRotateSpeed: number
    target: import('three').Vector3
    enableDamping: boolean
  }
  layerGroups: Record<string, import('three').Group>
  rafId: number
  resizeObs: ResizeObserver | null
  initialCamPos: import('three').Vector3
  initialTarget: import('three').Vector3
} | null = null

const PIXEL = 2.0 // 2 µm pitch
const EXPLODE_GAP = 0.45 // µm between layers when exploded

async function initThree() {
  if (!canvasHost.value) return
  const THREE = await import('three')
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')

  const host = canvasHost.value
  const width = host.clientWidth || 640
  const height = 480

  const scene = new THREE.Scene()
  scene.background = null

  const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100)
  const initialCamPos = new THREE.Vector3(5, 4.5, 6)
  const initialTarget = new THREE.Vector3(0, 2.3, 0)
  camera.position.copy(initialCamPos)

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  host.appendChild(renderer.domElement)

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.55))
  const key = new THREE.DirectionalLight(0xffffff, 1.05)
  key.position.set(4, 8, 5)
  key.castShadow = true
  key.shadow.mapSize.set(1024, 1024)
  key.shadow.camera.left = -4
  key.shadow.camera.right = 4
  key.shadow.camera.top = 4
  key.shadow.camera.bottom = -4
  key.shadow.camera.near = 1
  key.shadow.camera.far = 20
  scene.add(key)
  const fill = new THREE.DirectionalLight(0xbfd8ff, 0.35)
  fill.position.set(-5, 3, -3)
  scene.add(fill)
  const rim = new THREE.DirectionalLight(0xffe6c0, 0.25)
  rim.position.set(0, 2, -6)
  scene.add(rim)

  // Ground / shadow catcher
  const groundGeo = new THREE.PlaneGeometry(20, 20)
  const groundMat = new THREE.ShadowMaterial({ opacity: 0.18 })
  const ground = new THREE.Mesh(groundGeo, groundMat)
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -0.001
  ground.receiveShadow = true
  scene.add(ground)

  // Faint grid on ground
  const grid = new THREE.GridHelper(6, 12, 0x888888, 0x888888)
  ;(grid.material as import('three').Material).transparent = true
  ;(grid.material as import('three').Material).opacity = 0.15
  scene.add(grid)

  // Build pixel
  const layerGroups: Record<string, import('three').Group> = {}
  for (const l of layerDefs) {
    const g = new THREE.Group()
    g.name = l.id
    g.userData.zBot = l.zBot
    g.userData.zTop = l.zTop
    buildLayer(THREE, g, l)
    g.visible = l.visible
    scene.add(g)
    layerGroups[l.id] = g
  }

  // Center the pixel at origin in XZ
  for (const id in layerGroups) {
    layerGroups[id].position.x = -PIXEL / 2
    layerGroups[id].position.z = -PIXEL / 2
  }

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.target.copy(initialTarget)
  controls.minDistance = 2
  controls.maxDistance = 20
  controls.maxPolarAngle = Math.PI * 0.495 // prevent going below ground
  controls.autoRotateSpeed = 1.2

  function onResize() {
    if (!host) return
    const w = host.clientWidth
    if (w === 0) return
    renderer.setSize(w, height)
    camera.aspect = w / height
    camera.updateProjectionMatrix()
  }
  const resizeObs = new ResizeObserver(onResize)
  resizeObs.observe(host)

  let rafId = 0
  const tick = () => {
    rafId = requestAnimationFrame(tick)
    controls.autoRotate = autoRotate.value
    controls.update()
    renderer.render(scene, camera)
  }
  tick()

  three = { THREE, scene, camera, renderer, controls, layerGroups, rafId, resizeObs, initialCamPos, initialTarget }
  applyExplode()
}

function buildLayer(THREE: typeof import('three'), group: import('three').Group, l: LayerDef) {
  switch (l.id) {
    case 'silicon': {
      // Translucent silicon body
      const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x5d6d7e, roughness: 0.55, metalness: 0.15,
        transparent: true, opacity: 0.55,
      })
      const body = boxMesh(THREE, PIXEL, l.zTop - l.zBot, PIXEL, bodyMat)
      body.position.set(PIXEL / 2, (l.zBot + l.zTop) / 2, PIXEL / 2)
      body.castShadow = true
      body.receiveShadow = true
      group.add(body)

      // DTI walls (slightly brighter blue)
      const dtiMat = new THREE.MeshStandardMaterial({
        color: 0xaed6f1, roughness: 0.4, metalness: 0.2,
        transparent: true, opacity: 0.85,
      })
      const dw = 0.06
      const wallH = l.zTop - l.zBot
      const wx = boxMesh(THREE, dw, wallH, PIXEL, dtiMat)
      wx.position.set(PIXEL / 2, (l.zBot + l.zTop) / 2, PIXEL / 2)
      wx.castShadow = true
      group.add(wx)
      const wy = boxMesh(THREE, PIXEL, wallH, dw, dtiMat)
      wy.position.set(PIXEL / 2, (l.zBot + l.zTop) / 2, PIXEL / 2)
      wy.castShadow = true
      group.add(wy)

      // Photodiode regions on top
      const pdMat = new THREE.MeshStandardMaterial({
        color: 0xb85c5c, roughness: 0.65, metalness: 0.1,
        transparent: true, opacity: 0.55, side: THREE.DoubleSide,
      })
      for (const c of [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]) {
        const pd = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.7), pdMat)
        pd.rotation.x = -Math.PI / 2
        pd.position.set(c[0], l.zTop + 0.001, c[1])
        group.add(pd)
        // Outline
        const eg = new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.7, 0.7))
        const line = new THREE.LineSegments(
          eg, new THREE.LineBasicMaterial({ color: 0x9b3636 }))
        line.rotation.x = -Math.PI / 2
        line.position.set(c[0], l.zTop + 0.002, c[1])
        group.add(line)
      }
      break
    }
    case 'barl': {
      const subs = [
        { dz: 0, t: 0.01, col: 0x7fb3d8 },
        { dz: 0.01, t: 0.025, col: 0x6c71c4 },
        { dz: 0.035, t: 0.015, col: 0xe8d44d },
        { dz: 0.05, t: 0.030, col: 0x2aa198 },
      ]
      for (const s of subs) {
        const mat = new THREE.MeshStandardMaterial({
          color: s.col, roughness: 0.35, metalness: 0.25,
        })
        const m = boxMesh(THREE, PIXEL, s.t, PIXEL, mat)
        m.position.set(PIXEL / 2, l.zBot + s.dz + s.t / 2, PIXEL / 2)
        m.castShadow = true
        m.receiveShadow = true
        group.add(m)
      }
      break
    }
    case 'colorfilter': {
      const cfH = l.zTop - l.zBot
      const fracs = { R: 0.85, G: 1.00, B: 0.92, grid: 0.65 }
      const planar = 0xd5dbdb
      const cells = [
        { x0: 0, z0: 0, color: 0xc0392b, h: cfH * fracs.R },
        { x0: 1, z0: 0, color: 0x27ae60, h: cfH * fracs.G },
        { x0: 0, z0: 1, color: 0x27ae60, h: cfH * fracs.G },
        { x0: 1, z0: 1, color: 0x2980b9, h: cfH * fracs.B },
      ]
      for (const c of cells) {
        const mat = new THREE.MeshStandardMaterial({
          color: c.color, roughness: 0.45, metalness: 0.05,
          transparent: true, opacity: 0.92,
        })
        const m = boxMesh(THREE, 1, c.h, 1, mat)
        m.position.set(c.x0 + 0.5, l.zBot + c.h / 2, c.z0 + 0.5)
        m.castShadow = true
        m.receiveShadow = true
        group.add(m)
        if (c.h < cfH - 1e-6) {
          const fillMat = new THREE.MeshStandardMaterial({
            color: planar, roughness: 0.3, metalness: 0.1,
            transparent: true, opacity: 0.6,
          })
          const fillH = cfH - c.h
          const f = boxMesh(THREE, 1, fillH, 1, fillMat)
          f.position.set(c.x0 + 0.5, l.zBot + c.h + fillH / 2, c.z0 + 0.5)
          group.add(f)
        }
      }
      // Metal grid
      const mw = 0.05
      const gridH = cfH * fracs.grid
      const metalMat = new THREE.MeshStandardMaterial({
        color: 0x4a4a4a, roughness: 0.25, metalness: 0.85,
      })
      const gx = boxMesh(THREE, mw, gridH, PIXEL, metalMat)
      gx.position.set(PIXEL / 2, l.zBot + gridH / 2, PIXEL / 2)
      gx.castShadow = true
      gx.receiveShadow = true
      group.add(gx)
      const gz = boxMesh(THREE, PIXEL, gridH, mw, metalMat)
      gz.position.set(PIXEL / 2, l.zBot + gridH / 2, PIXEL / 2)
      gz.castShadow = true
      gz.receiveShadow = true
      group.add(gz)
      // Planarization above grid
      const fillMat2 = new THREE.MeshStandardMaterial({
        color: planar, roughness: 0.3, metalness: 0.1,
        transparent: true, opacity: 0.55,
      })
      const aboveH = cfH - gridH
      const fx = boxMesh(THREE, mw, aboveH, PIXEL, fillMat2)
      fx.position.set(PIXEL / 2, l.zBot + gridH + aboveH / 2, PIXEL / 2)
      group.add(fx)
      const fz = boxMesh(THREE, PIXEL, aboveH, mw, fillMat2)
      fz.position.set(PIXEL / 2, l.zBot + gridH + aboveH / 2, PIXEL / 2)
      group.add(fz)
      break
    }
    case 'planarization': {
      const mat = new THREE.MeshStandardMaterial({
        color: 0xd5dbdb, roughness: 0.35, metalness: 0.05,
        transparent: true, opacity: 0.6,
      })
      const m = boxMesh(THREE, PIXEL, l.zTop - l.zBot, PIXEL, mat)
      m.position.set(PIXEL / 2, (l.zBot + l.zTop) / 2, PIXEL / 2)
      m.castShadow = true
      m.receiveShadow = true
      group.add(m)
      break
    }
    case 'microlens': {
      const mat = new THREE.MeshPhysicalMaterial({
        color: 0xdda0dd, roughness: 0.12, metalness: 0.0,
        transmission: 0.35, thickness: 0.4, ior: 1.56,
        transparent: true, opacity: 0.92, clearcoat: 0.6, clearcoatRoughness: 0.15,
      })
      for (const c of [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]) {
        const dome = buildSuperellipseDome(THREE, 0.48, 0.6, 2.5, 36, mat)
        dome.position.set(c[0], l.zBot, c[1])
        dome.castShadow = true
        dome.receiveShadow = true
        group.add(dome)
      }
      break
    }
    case 'air': {
      const mat = new THREE.MeshStandardMaterial({
        color: 0xd6eaf8, roughness: 0.0, metalness: 0.0,
        transparent: true, opacity: 0.15,
      })
      const m = boxMesh(THREE, PIXEL, l.zTop - l.zBot, PIXEL, mat)
      m.position.set(PIXEL / 2, (l.zBot + l.zTop) / 2, PIXEL / 2)
      group.add(m)
      break
    }
  }
}

function boxMesh(
  THREE: typeof import('three'),
  w: number, h: number, d: number,
  mat: import('three').Material,
): import('three').Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat)
  return mesh
}

// Superellipse dome: |x/R|^n + |y/R|^n <= 1, height = h * (1 - r^n)^(1/n)
function buildSuperellipseDome(
  THREE: typeof import('three'),
  R: number, h: number, n: number, segments: number,
  material: import('three').Material,
): import('three').Mesh {
  // Build a heightmap-style mesh over the superellipse footprint.
  const verts: number[] = []
  const indices: number[] = []
  const idx: (number | null)[][] = []
  let counter = 0
  for (let i = 0; i <= segments; i++) {
    const row: (number | null)[] = []
    for (let j = 0; j <= segments; j++) {
      const u = -R + (2 * R * i) / segments
      const v = -R + (2 * R * j) / segments
      const au = Math.abs(u / R), av = Math.abs(v / R)
      const val = Math.pow(au, n) + Math.pow(av, n)
      if (val <= 1) {
        const z = h * Math.pow(1 - val, 1 / n)
        verts.push(u, z, v)
        row.push(counter++)
      } else {
        row.push(null)
      }
    }
    idx.push(row)
  }
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < segments; j++) {
      const a = idx[i][j], b = idx[i + 1][j], c = idx[i + 1][j + 1], d = idx[i][j + 1]
      if (a !== null && b !== null && c !== null && d !== null) {
        indices.push(a, b, c, a, c, d)
      }
    }
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  geo.setIndex(indices)
  geo.computeVertexNormals()
  return new THREE.Mesh(geo, material)
}

function applyExplode() {
  if (!three) return
  const { layerGroups } = three
  let cumulative = 0
  for (let i = 0; i < layerDefs.length; i++) {
    const l = layerDefs[i]
    const g = layerGroups[l.id]
    if (!g) continue
    g.position.y = exploded.value ? cumulative : 0
    if (exploded.value) cumulative += EXPLODE_GAP
  }
}

function resetView() {
  if (!three) return
  three.camera.position.copy(three.initialCamPos)
  three.controls.target.copy(three.initialTarget)
  three.controls.update()
}

watch(exploded, applyExplode)
watch(
  () => layerDefs.map(l => l.visible),
  (vals) => {
    if (!three) return
    for (let i = 0; i < layerDefs.length; i++) {
      const g = three.layerGroups[layerDefs[i].id]
      if (g) g.visible = vals[i]
    }
  },
)

onMounted(() => {
  initThree().catch(err => {
    console.error('Pixel3DViewer init failed', err)
  })
})

onBeforeUnmount(() => {
  if (!three) return
  cancelAnimationFrame(three.rafId)
  three.resizeObs?.disconnect()
  three.controls.dispose()
  three.renderer.dispose()
  three.scene.traverse((obj) => {
    const mesh = obj as import('three').Mesh
    if (mesh.geometry) mesh.geometry.dispose()
    const m = mesh.material as import('three').Material | import('three').Material[] | undefined
    if (Array.isArray(m)) m.forEach(mm => mm.dispose())
    else if (m) m.dispose()
  })
  if (three.renderer.domElement.parentNode) {
    three.renderer.domElement.parentNode.removeChild(three.renderer.domElement)
  }
  three = null
})
</script>

<style scoped>
.pixel-3d-viewer { padding: 16px 0; }
.controls-row {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 12px; flex-wrap: wrap;
}
.ctrl-item {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.88em; font-weight: 600; color: var(--vp-c-text-1);
  cursor: pointer; white-space: nowrap;
}
.ctrl-item input[type="checkbox"] { accent-color: var(--vp-c-brand-1); }
.ctrl-btn {
  font-size: 0.82em; padding: 4px 10px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft); color: var(--vp-c-text-1);
  border-radius: 4px; cursor: pointer;
}
.ctrl-btn:hover { background: var(--vp-c-bg-mute); }
.layer-toggles { display: flex; flex-wrap: wrap; gap: 6px 14px; }
.toggle-item {
  display: flex; align-items: center; gap: 4px;
  font-size: 0.82em; color: var(--vp-c-text-2); cursor: pointer;
}
.toggle-item input[type="checkbox"] { accent-color: var(--vp-c-brand-1); width: 14px; height: 14px; }
.toggle-sw { width: 10px; height: 10px; border-radius: 2px; border: 1px solid var(--vp-c-divider); }
.toggle-lbl { white-space: nowrap; }
.viewer-host {
  position: relative; width: 100%; max-width: 720px;
  height: 480px; margin: 0 auto;
  border: 1px solid var(--vp-c-divider); border-radius: 6px;
  background: linear-gradient(180deg, var(--vp-c-bg-soft) 0%, var(--vp-c-bg) 100%);
  overflow: hidden; user-select: none; -webkit-user-select: none;
}
.viewer-host :deep(canvas) { display: block; }
.hint {
  position: absolute; left: 0; right: 0; bottom: 6px;
  text-align: center; font-size: 11px; color: var(--vp-c-text-3);
  font-style: italic; pointer-events: none;
}
</style>
