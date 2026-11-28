import { expect, test, type Page } from '@playwright/test'

async function gotoDocs(page: Page, path: string) {
  await page.goto(path.replace(/^\//, ''))
  await page.waitForLoadState('networkidle')
}

test.describe('simulator pages', () => {
  const fullscreenPaths = [
    '/simulator/tmm-qe.html',
    '/simulator/angular-response.html',
    '/simulator/barl-optimizer.html',
    '/simulator/color-accuracy.html',
    '/simulator/color-filter.html',
    '/simulator/emva1288.html',
    '/simulator/energy-budget.html',
    '/simulator/pixel-playground.html',
    '/simulator/fdti-pixel.html',
    '/simulator/mla-array.html',
    '/simulator/microlens-process-shape.html',
  ]

  for (const path of fullscreenPaths) {
    test(`control-heavy simulator toggles fullscreen on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)

      const toggle = page.getByRole('button', { name: 'Toggle fullscreen' }).first()
      await expect(toggle).toBeVisible()
      await toggle.click()
      const fullscreenRoot = page.locator('.sim-fullscreen, .cf-fullscreen')
      await expect(fullscreenRoot).toHaveCount(1)

      await page.keyboard.press('Escape')
      await expect(fullscreenRoot).toHaveCount(0)
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
      await expect(page.locator('.sim-theory .formula-variables mjx-container').first()).toBeVisible()
      await expect(page.locator('.sim-theory a').first()).toBeVisible()
    })
  }

  test('renders expanded theory notes on the microlens process simulator', async ({ page }) => {
    await gotoDocs(page, '/simulator/microlens-process-shape.html')

    await expect(page.locator('.sim-theory .formula-row')).toHaveCount(9)
    await expect(page.locator('.sim-theory-detail-grid')).toBeVisible()
    await expect(page.getByText('Validation Example')).toBeVisible()
    await expect(page.getByText('How To Calibrate This Surrogate')).toBeVisible()
    await expect(page.getByText('Known Missing Physics')).toBeVisible()
    await expect(page.getByText('Calibration coefficients')).toBeVisible()
    await expect(page.locator('.metric-card span').getByText('Zero-gap etch', { exact: true })).toBeVisible()
    await expect(page.locator('.sim-theory .formula-variables mjx-container').first()).toBeVisible()
  })

  test('renders expanded theory notes on the TMM QE simulator', async ({ page }) => {
    await gotoDocs(page, '/simulator/tmm-qe.html')

    await expect(page.locator('.sim-theory .formula-row')).toHaveCount(7)
    await expect(page.locator('.sim-theory-detail-grid')).toBeVisible()
    await expect(page.getByText('Validation Example')).toBeVisible()
    await expect(page.getByText('Calibration Checklist')).toBeVisible()
    await expect(page.getByText('Known Missing Physics')).toBeVisible()
    await expect(page.locator('.sim-theory .formula-variables mjx-container').first()).toBeVisible()
  })

  const expandedTheoryPages = [
    { path: '/simulator/barl-optimizer.html', formulas: 6, heading: 'Tuning Workflow' },
    { path: '/simulator/energy-budget.html', formulas: 6, heading: 'Diagnosis Workflow' },
    { path: '/simulator/angular-response.html', formulas: 6, heading: 'CRA Design Implications' },
    { path: '/simulator/snr-calculator.html', formulas: 3, heading: 'Regime Map' },
    { path: '/simulator/color-filter.html', formulas: 3, heading: 'Design Tradeoff' },
    { path: '/simulator/si-absorption.html', formulas: 3, heading: 'Wavelength Regimes' },
    { path: '/simulator/mtf-analyzer.html', formulas: 3, heading: 'Frequency Landmarks' },
    { path: '/simulator/dark-current.html', formulas: 3, heading: 'Temperature Scaling' },
    { path: '/simulator/responsivity-calculator.html', formulas: 3, heading: 'QE Versus A/W' },
    { path: '/simulator/linearity-analyzer.html', formulas: 3, heading: 'Residual Interpretation' },
  ]

  for (const { path, formulas, heading } of expandedTheoryPages) {
    test(`renders expanded theory notes on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)

      await expect(page.locator('.sim-theory .formula-row')).toHaveCount(formulas)
      await expect(page.locator('.sim-theory-detail-grid')).toBeVisible()
      await expect(page.getByRole('heading', { name: heading })).toBeVisible()
      await expect(page.getByText('Known Missing Physics')).toBeVisible()
      await expect(page.locator('.sim-theory .formula-variables mjx-container').first()).toBeVisible()
    })
  }

  const detailedTheoryPages = [
    '/simulator/dynamic-range.html',
    '/simulator/responsivity-calculator.html',
    '/simulator/pixel-snr-vs-illuminance.html',
    '/simulator/photon-transfer-curve.html',
    '/simulator/mtf-analyzer.html',
    '/simulator/pixel-scaling.html',
  ]

  for (const path of detailedTheoryPages) {
    test(`renders standard detailed theory blocks on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)

      await expect(page.locator('.sim-theory-standard-grid')).toBeVisible()
      await expect(page.getByText('Assumptions', { exact: true })).toBeVisible()
      await expect(page.getByText('Outputs', { exact: true })).toBeVisible()
      await expect(page.getByText('Validation Example', { exact: true })).toBeVisible()
    })
  }

  // Controls that previously had no effect on any output must now change the results.
  const liveControls = [
    { path: '/simulator/photon-transfer-curve.html', control: 'Conversion Gain', value: '150', results: '.results-grid' },
    { path: '/simulator/pixel-snr-vs-illuminance.html', control: 'Pixel Area', value: '9', results: '.results-grid' },
    { path: '/simulator/emva1288.html', control: 'Exposure Time', value: '2.5', results: '.param-grid' },
    { path: '/simulator/linearity-analyzer.html', control: 'Full Well Capacity', value: '50000', results: '.results-grid' },
    { path: '/simulator/dark-current.html', control: 'Dark current @ 27 °C', value: '20', results: '.results-grid' },
  ]

  for (const { path, control, value, results } of liveControls) {
    test(`"${control}" changes the outputs on ${path}`, async ({ page }) => {
      await gotoDocs(page, path)
      const out = page.locator(results).first()
      const before = await out.innerText()
      await page.getByRole('slider', { name: control, exact: true }).first().fill(value)
      await expect(out).not.toHaveText(before)
    })
  }

  test('PRNU heatmap contrast follows the PRNU slider', async ({ page }) => {
    await gotoDocs(page, '/simulator/prnu-visualizer.html')
    const cells = page.locator('.pixel-cell')
    const spread = async () =>
      cells.evaluateAll(els => {
        const v = els.map(e => Number(getComputedStyle(e).backgroundColor.match(/\d+/)?.[0] ?? 0))
        return Math.max(...v) - Math.min(...v)
      })
    await page.getByRole('slider', { name: 'DSNU', exact: true }).fill('0')
    await page.getByRole('slider', { name: 'PRNU', exact: true }).fill('0.5')
    const low = await spread()
    await page.getByRole('slider', { name: 'PRNU', exact: true }).fill('3')
    await expect.poll(spread).toBeGreaterThan(low)
  })
})
