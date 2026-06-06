import { chromium } from 'playwright'
import { mkdir } from 'node:fs/promises'

const outDir = '/tmp/microlens-shots'
await mkdir(outDir, { recursive: true })

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--no-sandbox'],
})

const ctx = await browser.newContext({ viewport: { width: 1500, height: 950 } })
const page = await ctx.newPage()
page.on('console', (msg) => {
  if (msg.type() === 'error') console.log('PAGE ERR:', msg.text())
})
page.on('pageerror', (err) => console.log('PAGE EXC:', err.message))

const BASE = 'http://localhost:4173/compass'
await page.goto(`${BASE}/simulator/microlens-process-shape.html`, { waitUntil: 'networkidle' })
await page.waitForSelector('.mlp-container', { timeout: 5000 })

async function selectPreset(value) {
  await page.selectOption('select.ctrl-select >> nth=1', value)
  await page.waitForTimeout(300)
}

async function selectTab(label) {
  await page.locator('.tab-btn', { hasText: label }).first().click()
  await page.waitForTimeout(300)
}

async function shot(name) {
  const file = `${outDir}/${name}.png`
  await page.locator('.mlp-container').screenshot({ path: file })
  console.log('saved', file)
}

// 1. All 1x1 baseline, Top view
await selectPreset('all-1x1')
await selectTab('Top view')
await shot('01-all-1x1-topview')

// 2. Mixed 2x2 OCL + 1x1, Top view (showcase per-edge asymmetry)
await selectPreset('mixed-2x2-pdaf')
await shot('02-mixed-2x2-topview')

// 3. Sparse 2x1 PDAF, Top view
await selectPreset('sparse-2x1-pdaf')
await shot('03-sparse-2x1-topview')

// 4. Mixed 2x2 with reflow time cranked up + microloading gain 2x to amplify asymmetry
await selectPreset('mixed-2x2-pdaf')
const detailsToggle = page.locator('details.calibration-panel summary').first()
const isOpen = await page.locator('details.calibration-panel').getAttribute('open')
if (isOpen === null) await detailsToggle.click()
await page.waitForTimeout(200)

async function setSliderByLabel(labelText, value) {
  const slider = page.locator(`label:has-text("${labelText}") + input[type="range"]`).first()
  await slider.evaluate((el, v) => {
    el.value = String(v)
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }, value)
}

await setSliderByLabel('Microloading gain', 2.0)
await setSliderByLabel('Proximity coupling gain', 2.0)
await page.waitForTimeout(300)
await shot('04-mixed-2x2-coupling-2x')

// 5. 3D surface view of the same Mixed
await selectTab('3D surface')
await shot('05-mixed-2x2-surface')
// 5b. Tight crop of just the SVG
await page.locator('.plot-panel svg.main-svg').screenshot({ path: `${outDir}/05b-surface-svg-only.png` })
console.log('saved 05b-surface-svg-only.png')
// 5c. Top view svg only too for clarity
await selectTab('Top view')
await page.locator('.plot-panel svg.main-svg').screenshot({ path: `${outDir}/04b-topview-svg-only.png` })
console.log('saved 04b-topview-svg-only.png')
await selectTab('3D surface')

// 6. Fullscreen Side-by-side
await setSliderByLabel('Proximity coupling gain', 1.0)
await setSliderByLabel('Microloading gain', 1.0)
await selectTab('Top view')
await page.locator('button.sim-fs-btn').first().click()
await page.waitForTimeout(500)
await page.screenshot({ path: `${outDir}/06-fullscreen-sidebyside.png` })
console.log('saved 06-fullscreen-sidebyside.png')

await browser.close()
console.log('DONE')
