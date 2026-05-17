import { expect, test, type Page } from '@playwright/test'

async function gotoDocs(page: Page, path: string) {
  await page.goto(path.replace(/^\//, ''))
  await page.waitForLoadState('networkidle')
}

test.describe('reference map', () => {
  test('shows where a reference is used in COMPASS', async ({ page }) => {
    await gotoDocs(page, '/about/references.html')

    await page.getByRole('button', { name: /Open reference details for/i }).first().click()

    await expect(page.getByText('Used in COMPASS')).toBeVisible()
    await expect(page.getByRole('link', { name: 'RCWA Explained' })).toBeVisible()
  })
})
