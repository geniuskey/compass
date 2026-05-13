import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const docsRoot = path.resolve(scriptDir, '..')
const guideRoot = path.join(docsRoot, 'guide')
const koGuideRoot = path.join(docsRoot, 'ko', 'guide')
const themeIndex = path.join(docsRoot, '.vitepress', 'theme', 'index.ts')
const componentsRoot = path.join(docsRoot, '.vitepress', 'theme', 'components')

const failures = []

function walkMarkdown(dir) {
  const files = []
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) files.push(...walkMarkdown(full))
    else if (entry.isFile() && entry.name.endsWith('.md')) files.push(full)
  }
  return files.sort()
}

function componentTags(markdown) {
  const tags = []
  const pattern = /<([A-Z][A-Za-z0-9]*)\b/g
  let match
  while ((match = pattern.exec(markdown)) !== null) tags.push(match[1])
  return tags
}

function parseRegisteredComponents() {
  const source = fs.readFileSync(themeIndex, 'utf8')
  const registered = new Map()
  const pattern = /\['([^']+)',\s*\(\) => import\('\.\/components\/([^']+\.vue)'\)\]/g
  let match
  while ((match = pattern.exec(source)) !== null) registered.set(match[1], match[2])
  return registered
}

const registered = parseRegisteredComponents()
const guideFiles = walkMarkdown(guideRoot)
const usedByPage = new Map()
const allUsed = new Set()

for (const file of guideFiles) {
  const rel = path.relative(guideRoot, file)
  const koFile = path.join(koGuideRoot, rel)
  if (!fs.existsSync(koFile)) {
    failures.push(`ko/guide/${rel}: missing localized guide page`)
    continue
  }

  const enTags = componentTags(fs.readFileSync(file, 'utf8'))
  const koTags = componentTags(fs.readFileSync(koFile, 'utf8'))
  usedByPage.set(rel, enTags)
  enTags.forEach((tag) => allUsed.add(tag))
  koTags.forEach((tag) => allUsed.add(tag))

  if (JSON.stringify(enTags) !== JSON.stringify(koTags)) {
    failures.push(`${rel}: EN/KO interactive component sequence differs (${enTags.join(', ')} vs ${koTags.join(', ')})`)
  }
}

for (const name of allUsed) {
  if (!registered.has(name)) {
    failures.push(`${name}: used in guide markdown but not registered in theme/index.ts`)
    continue
  }
  const componentFile = path.join(componentsRoot, registered.get(name))
  if (!fs.existsSync(componentFile)) {
    failures.push(`${name}: registered component file not found (${registered.get(name)})`)
    continue
  }

  const source = fs.readFileSync(componentFile, 'utf8')
  const emptySvgAttrs = source.match(/\s(?:x1|x2|y1|y2|width|height)=""/g)
  if (emptySvgAttrs) {
    failures.push(`${registered.get(name)}: empty SVG coordinate/size attribute(s): ${[...new Set(emptySvgAttrs)].join(', ')}`)
  }
}

if (failures.length > 0) {
  console.error('Guide interactive component check failed:')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

const pageCount = usedByPage.size
const componentCount = allUsed.size
console.log(`Guide interactive check passed for ${pageCount} guide page(s) and ${componentCount} component(s).`)
