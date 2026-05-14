import { expect, test, type Locator, type Page } from '@playwright/test'

async function setRangeValue(locator: Locator, value: number) {
  await locator.evaluate((input, nextValue) => {
    const range = input as HTMLInputElement
    range.value = String(nextValue)
    range.dispatchEvent(new Event('input', { bubbles: true }))
    range.dispatchEvent(new Event('change', { bubbles: true }))
  }, value)
}

async function numericAttribute(locator: Locator, name: string) {
  const value = await locator.getAttribute(name)
  expect(value, `${name} attribute`).not.toBeNull()
  const parsed = Number(value)
  expect(Number.isFinite(parsed), `${name}=${value}`).toBeTruthy()
  return parsed
}

async function expectNonEmptySvgPath(locator: Locator) {
  await expect(locator).toBeVisible()
  const d = await locator.getAttribute('d')
  expect(d?.length ?? 0).toBeGreaterThan(20)
}

async function gotoGuide(page: Page, path: string) {
  await page.goto(path.replace(/^\//, ''))
  await page.waitForLoadState('networkidle')
}

test.describe('pixel stack guide visuals', () => {
  for (const path of ['/guide/pixel-stack-config.html', '/ko/guide/pixel-stack-config.html']) {
    test(`renders physical XZ/XY pixel-stack visuals on ${path}`, async ({ page }) => {
      await gotoGuide(page, path)

      await expect(page.locator('[data-visual-id="xz-layer-air"]')).toHaveCount(1)
      await expect(page.locator('[data-visual-id="xz-layer-microlens"]')).toHaveCount(1)
      await expect(page.locator('[data-visual-id="xz-layer-color_filter"]')).toHaveCount(1)
      await expect(page.locator('[data-visual-id="xz-layer-planarization"]')).toHaveCount(1)
      await expect(page.locator('[data-visual-id="xz-cf-relief"]')).toHaveCount(2)
      await expect(page.locator('[data-visual-id="xz-metal-grid"]')).toHaveCount(3)
      await expect(page.locator('[data-visual-id="xz-photodiode"]')).toHaveCount(2)
      await expect(page.locator('[data-visual-id="xz-microlens-dome"]')).toHaveCount(2)

      await expect(page.locator('[data-visual-id="xz-pattern-extent-microlens"]').first()).toHaveAttribute('fill', 'none')
      await expect(page.locator('[data-visual-id="xz-pattern-extent-color_filter"]').first()).toHaveAttribute('fill', 'none')

      const xyTab = page.getByRole('button', { name: /XY|평면/ }).first()
      await xyTab.click()
      await expect(page.getByRole('img', { name: /XY|평면/ })).toBeVisible()
      await expect(page.getByLabel(/Microlens|마이크로렌즈/).or(page.getByText(/Microlens|마이크로렌즈/)).first()).toBeVisible()
    })
  }
})

test.describe('cone illumination guide visuals', () => {
  test('keeps CRA-corrected chief focus inside the photodiode', async ({ page }) => {
    await gotoGuide(page, '/guide/cone-illumination.html')

    const viewer = page.locator('.cone-illum-container').filter({ has: page.getByText('Interactive Cone Illumination Viewer') })
    await expect(viewer).toBeVisible()

    await setRangeValue(viewer.locator('input[type="range"]').first(), 30)

    const pd = viewer.locator('[data-visual-id="cone-photodiode"]').first()
    const chiefFocus = viewer.locator('[data-visual-id="cone-chief-focus"]').first()
    await expect(pd).toBeVisible()
    await expect(chiefFocus).toBeVisible()

    const pdX = await numericAttribute(pd, 'x')
    const pdW = await numericAttribute(pd, 'width')
    const focusX = await numericAttribute(chiefFocus, 'cx')

    expect(focusX).toBeGreaterThanOrEqual(pdX)
    expect(focusX).toBeLessThanOrEqual(pdX + pdW)
    await expect(viewer.locator('[data-visual-id="cone-bundle-focus"]')).toHaveCount(2)
  })

  test('renders distinct non-grid sampling patterns in the top view', async ({ page }) => {
    await gotoGuide(page, '/guide/cone-illumination.html')

    const topView = page.locator('.cone-illum-container').filter({ has: page.getByText('Cone Illumination') }).last()
    await expect(topView.locator('[data-visual-id="cone-top-footprint"]')).toBeVisible()

    const positions = async () => {
      return topView.locator('[data-visual-id="cone-top-sample"]').evaluateAll((nodes) =>
        nodes.slice(0, 16).map((node) => {
          const circle = node as SVGCircleElement
          return `${Number(circle.getAttribute('cx')).toFixed(1)},${Number(circle.getAttribute('cy')).toFixed(1)}`
        })
      )
    }

    const fibonacci = await positions()
    await topView.getByRole('button', { name: /Halton|할튼/ }).click()
    const halton = await positions()
    await topView.getByRole('button', { name: /Grid legacy|격자 legacy/ }).click()
    const grid = await positions()

    expect(new Set(fibonacci).size).toBeGreaterThan(10)
    expect(fibonacci.join('|')).not.toEqual(halton.join('|'))
    expect(halton.join('|')).not.toEqual(grid.join('|'))
  })

  test('renders Fabry-Perot cone samples and spectrum paths', async ({ page }) => {
    await gotoGuide(page, '/guide/cone-illumination.html')

    expect(await page.locator('[data-visual-id="fp-cone-sample"]').count()).toBeGreaterThan(50)
    await expectNonEmptySvgPath(page.locator('[data-visual-id="fp-cone-spectrum"]').first())
  })
})

test('material browser renders n/k curves and hover readout', async ({ page, isMobile }) => {
  await gotoGuide(page, '/guide/material-database.html')

  await expectNonEmptySvgPath(page.locator('[data-visual-id="material-n-curve"]').first())
  await expectNonEmptySvgPath(page.locator('[data-visual-id="material-k-curve"]').first())

  if (isMobile) return

  const chart = page.locator('.material-svg').first()
  const box = await chart.boundingBox()
  expect(box).not.toBeNull()
  await page.mouse.move(box!.x + box!.width * 0.5, box!.y + box!.height * 0.5)
  await expect(page.locator('.hover-info')).toContainText(/n =/)
  await expect(page.locator('.hover-info')).toContainText(/k =/)
})
