import { expect, test } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

// Every simulator page must render without console errors and without NaN/Infinity in its UI.
const slugs = fs
  .readdirSync(path.resolve(__dirname, '../../simulator'))
  .filter(f => f.endsWith('.md') && f !== 'index.md')
  .map(f => f.replace(/\.md$/, ''))

for (const locale of ['', 'ko/']) {
  for (const slug of slugs) {
    test(`simulator ${locale}${slug} renders cleanly`, async ({ page }) => {
      const errors: string[] = []
      page.on('pageerror', e => errors.push(e.message))
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()) })
      await page.goto(`${locale}simulator/${slug}.html`)
      await page.waitForLoadState('networkidle')
      const root = page.locator('.vp-doc')
      const text = await root.innerText()
      expect(text).not.toMatch(/\bNaN\b|Infinity|undefined/)
      const svgAttrs = await root.locator('svg path, svg line, svg rect, svg circle').evaluateAll(els =>
        els.flatMap(e => ['d', 'x1', 'y1', 'x', 'y', 'cx', 'cy', 'width', 'height']
          .map(a => e.getAttribute(a) ?? '')
          .filter(v => /NaN|Infinity/.test(v))),
      )
      expect(svgAttrs).toEqual([])
      expect(errors).toEqual([])
    })
  }
}
