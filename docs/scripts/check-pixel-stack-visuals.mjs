import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const docsRoot = path.resolve(scriptDir, '..')
const distRoot = path.join(docsRoot, '.vitepress', 'dist')

const pages = [
  'guide/pixel-stack-config.html',
  'ko/guide/pixel-stack-config.html',
]

const sourceFiles = [
  '.vitepress/theme/components/PixelParameterDiagram.vue',
  '.vitepress/theme/components/PixelStackBuilder.vue',
  '.vitepress/theme/composables/pixelStackDefaults.ts',
]

const failures = []

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function countToken(text, token) {
  return (text.match(new RegExp(escapeRegExp(token), 'g')) || []).length
}

function tagsWithVisualId(html, id) {
  const pattern = new RegExp(`<[^>]*data-visual-id="${escapeRegExp(id)}"[^>]*>`, 'g')
  return html.match(pattern) || []
}

function requireToken(label, text, token) {
  if (!text.includes(token)) failures.push(`${label}: missing token "${token}"`)
}

function requireAnyToken(label, text, tokens) {
  if (!tokens.some((token) => text.includes(token))) {
    failures.push(`${label}: missing one of ${tokens.map((token) => `"${token}"`).join(', ')}`)
  }
}

function requireVisualCount(page, html, id, minCount) {
  const count = countToken(html, `data-visual-id="${id}"`)
  if (count < minCount) {
    failures.push(`${page}: expected at least ${minCount} "${id}" visual node(s), found ${count}`)
  }
}

function requireVisualAttribute(page, html, id, attribute) {
  const tags = tagsWithVisualId(html, id)
  if (!tags.some((tag) => tag.includes(attribute))) {
    failures.push(`${page}: "${id}" visual node is missing ${attribute}`)
  }
}

for (const page of pages) {
  const filePath = path.join(distRoot, page)
  if (!fs.existsSync(filePath)) {
    failures.push(`${page}: built HTML not found. Run npm run docs:build before docs:visual-check.`)
    continue
  }

  const html = fs.readFileSync(filePath, 'utf8')

  requireAnyToken(page, html, ['Interactive Pixel Stack Builder', '인터랙티브 픽셀 스택 빌더'])
  requireAnyToken(page, html, ['Toggle fullscreen', '전체화면 전환'])
  requireToken(page, html, 'position[z] center')
  requireAnyToken(page, html, ['PD center depth below top of Si', '실리콘 상단 기준 PD 중심 깊이'])
  requireToken(page, html, 'microlens.gap')

  requireVisualCount(page, html, 'xz-layer-air', 1)
  requireVisualCount(page, html, 'xz-layer-microlens', 1)
  requireVisualCount(page, html, 'xz-layer-color_filter', 1)
  requireVisualCount(page, html, 'xz-layer-planarization', 1)
  requireVisualCount(page, html, 'xz-pattern-extent-microlens', 1)
  requireVisualCount(page, html, 'xz-pattern-extent-color_filter', 1)
  requireVisualCount(page, html, 'xz-cf-relief', 2)
  requireVisualCount(page, html, 'xz-metal-grid', 3)
  requireVisualCount(page, html, 'xz-dti-wall', 3)
  requireVisualCount(page, html, 'xz-photodiode', 2)
  requireVisualCount(page, html, 'xz-microlens-dome', 2)
  requireVisualCount(page, html, 'stack-cf-relief', 1)
  requireVisualCount(page, html, 'stack-metal-grid', 2)
  requireVisualCount(page, html, 'stack-microlens-material', 1)

  requireVisualAttribute(page, html, 'xz-pattern-extent-microlens', 'fill="none"')
  requireVisualAttribute(page, html, 'xz-pattern-extent-color_filter', 'fill="none"')
}

for (const file of sourceFiles) {
  const filePath = path.join(docsRoot, file)
  const source = fs.readFileSync(filePath, 'utf8')

  if (source.includes('PD top below top of Si')) {
    failures.push(`${file}: legacy PD top-depth wording returned`)
  }
  if (source.includes('mlGap = 0.04') || source.includes('gap: 0.04')) {
    failures.push(`${file}: microlens.gap must follow default_bsi_1um.yaml (0.0), not visual clearance`)
  }
  if (file.endsWith('PixelParameterDiagram.vue') && source.includes('zBot: 3.08')) {
    failures.push(`${file}: hard-coded default stack z coordinates returned`)
  }
}

if (failures.length > 0) {
  console.error('Pixel stack visual check failed:')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`Pixel stack visual check passed for ${pages.length} built page(s).`)
