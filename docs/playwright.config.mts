import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.DOCS_BASE_URL || 'http://127.0.0.1:4173/compass/'

export default defineConfig({
  testDir: './tests/browser',
  timeout: 30_000,
  expect: {
    timeout: 7_500,
  },
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  webServer: {
    command: 'npm run docs:preview -- --host 127.0.0.1 --port 4173',
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium-desktop',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'chromium-mobile',
      use: { ...devices['Pixel 5'] },
    },
  ],
})
