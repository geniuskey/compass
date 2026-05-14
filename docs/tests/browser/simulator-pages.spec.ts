import { expect, test, type Page } from '@playwright/test'

async function gotoDocs(page: Page, path: string) {
  await page.goto(path.replace(/^\//, ''))
  await page.waitForLoadState('networkidle')
}

test.describe('simulator pages', () => {
  const fullscreenPaths = [
    '/simulator/tmm-qe.html',
    '/simulator/barl-optimizer.html',
    '/simulator/pixel-playground.html',
    '/simulator/fdti-pixel.html',
    '/simulator/microlens-process-shape.html',
  ]

  for (const path of fullscreenPaths) {
    test(`control-heavy simulator toggles fullscreen on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)

      const toggle = page.getByRole('button', { name: 'Toggle fullscreen' }).first()
      await expect(toggle).toBeVisible()
      await toggle.click()
      await expect(page.locator('.sim-fullscreen')).toHaveCount(1)

      await page.keyboard.press('Escape')
      await expect(page.locator('.sim-fullscreen')).toHaveCount(0)
    })
  }

  const theoryPaths = [
    '/simulator/color-filter.html',
    '/simulator/tmm-qe.html',
    '/simulator/dark-current.html',
    '/ko/simulator/color-filter.html',
    '/ko/simulator/tmm-qe.html',
    '/ko/simulator/dark-current.html',
  ]

  for (const path of theoryPaths) {
    test(`renders simulator theory notes on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)

      await expect(page.locator('.sim-theory')).toHaveCount(1)
      await expect(page.locator('.sim-theory .formula-equation').first()).toBeVisible()
      await expect(page.locator('.sim-theory a').first()).toBeVisible()
    })
  }
})
