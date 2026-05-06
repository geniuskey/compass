import DefaultTheme from 'vitepress/theme'
import { defineAsyncComponent } from 'vue'
import './custom.css'

const asyncComponent = (loader: () => Promise<any>) => defineAsyncComponent(loader)

const globalComponents = [
  ['WavelengthSlider', () => import('./components/WavelengthSlider.vue')],
  ['FresnelCalculator', () => import('./components/FresnelCalculator.vue')],
  ['FourierOrderDemo', () => import('./components/FourierOrderDemo.vue')],
  ['StackVisualizer', () => import('./components/StackVisualizer.vue')],
  ['BayerPatternViewer', () => import('./components/BayerPatternViewer.vue')],
  ['SnellCalculator', () => import('./components/SnellCalculator.vue')],
  ['MaterialBrowser', () => import('./components/MaterialBrowser.vue')],
  ['EMWaveAnimation', () => import('./components/EMWaveAnimation.vue')],
  ['RCWAConvergenceDemo', () => import('./components/RCWAConvergenceDemo.vue')],
  ['CrosstalkHeatmap', () => import('./components/CrosstalkHeatmap.vue')],
  ['QESpectrumChart', () => import('./components/QESpectrumChart.vue')],
  ['PixelStackBuilder', () => import('./components/PixelStackBuilder.vue')],
  ['ConeIlluminationViewer', () => import('./components/ConeIlluminationViewer.vue')],
  ['PrecisionComparison', () => import('./components/PrecisionComparison.vue')],
  ['ThinFilmReflectance', () => import('./components/ThinFilmReflectance.vue')],
  ['PolarizationViewer', () => import('./components/PolarizationViewer.vue')],
  ['YeeCellViewer', () => import('./components/YeeCellViewer.vue')],
  ['SolverComparisonChart', () => import('./components/SolverComparisonChart.vue')],
  ['BlackbodySpectrum', () => import('./components/BlackbodySpectrum.vue')],
  ['SignalChainDiagram', () => import('./components/SignalChainDiagram.vue')],
  ['ModuleArchitectureDiagram', () => import('./components/ModuleArchitectureDiagram.vue')],
  ['EnergyBalanceDiagram', () => import('./components/EnergyBalanceDiagram.vue')],
  ['SolverPipelineDiagram', () => import('./components/SolverPipelineDiagram.vue')],
  ['StaircaseMicrolensViewer', () => import('./components/StaircaseMicrolensViewer.vue')],
  ['HeroAnimation', () => import('./components/HeroAnimation.vue')],
  ['FeatureShowcase', () => import('./components/FeatureShowcase.vue')],
  ['ArchitectureOverview', () => import('./components/ArchitectureOverview.vue')],
  ['SolverShowcase', () => import('./components/SolverShowcase.vue')],
  ['PixelAnatomyViewer', () => import('./components/PixelAnatomyViewer.vue')],
  ['PhotonJourneyAnimation', () => import('./components/PhotonJourneyAnimation.vue')],
  ['WavelengthExplorer', () => import('./components/WavelengthExplorer.vue')],
  ['ConeIlluminationTopView', () => import('./components/ConeIlluminationTopView.vue')],
  ['CrossSolverValidation', () => import('./components/CrossSolverValidation.vue')],
  ['RcwaFdtdValidation', () => import('./components/RcwaFdtdValidation.vue')],
  ['ConvergenceStudyChart', () => import('./components/ConvergenceStudyChart.vue')],
  ['PerColorConvergenceChart', () => import('./components/PerColorConvergenceChart.vue')],
  ['PixelCrossSections', () => import('./components/PixelCrossSections.vue')],
  ['Pixel3DViewer', () => import('./components/Pixel3DViewer.vue')],
  ['TmmQeCalculator', () => import('./components/TmmQeCalculator.vue')],
  ['BarlOptimizer', () => import('./components/BarlOptimizer.vue')],
  ['EnergyBudgetAnalyzer', () => import('./components/EnergyBudgetAnalyzer.vue')],
  ['AngularResponseSimulator', () => import('./components/AngularResponseSimulator.vue')],
  ['SnrCalculator', () => import('./components/SnrCalculator.vue')],
  ['ColorFilterDesigner', () => import('./components/ColorFilterDesigner.vue')],
  ['PixelDesignPlayground', () => import('./components/PixelDesignPlayground.vue')],
  ['SiliconAbsorptionDepth', () => import('./components/SiliconAbsorptionDepth.vue')],
  ['FabryPerotVisualizer', () => import('./components/FabryPerotVisualizer.vue')],
  ['MicrolensRayTrace', () => import('./components/MicrolensRayTrace.vue')],
  ['DiffractionPsfViewer', () => import('./components/DiffractionPsfViewer.vue')],
  ['FdtiPixelSimulator', () => import('./components/FdtiPixelSimulator.vue')],
  ['MtfAnalyzer', () => import('./components/MtfAnalyzer.vue')],
  ['PixelScalingTrends', () => import('./components/PixelScalingTrends.vue')],
  ['ColorAccuracyAnalyzer', () => import('./components/ColorAccuracyAnalyzer.vue')],
  ['DarkCurrentSimulator', () => import('./components/DarkCurrentSimulator.vue')],
  ['MlaArrayVisualizer', () => import('./components/MlaArrayVisualizer.vue')],
  ['PhotonTransferCurve', () => import('./components/PhotonTransferCurve.vue')],
  ['DynamicRangeCalculator', () => import('./components/DynamicRangeCalculator.vue')],
  ['EMVA1288Dashboard', () => import('./components/EMVA1288Dashboard.vue')],
  ['LensShadingSimulator', () => import('./components/LensShadingSimulator.vue')],
  ['PRNUVisualizer', () => import('./components/PRNUVisualizer.vue')],
  ['PixelSNRvsIlluminance', () => import('./components/PixelSNRvsIlluminance.vue')],
  ['ResponsivityCalculator', () => import('./components/ResponsivityCalculator.vue')],
  ['LinearityAnalyzer', () => import('./components/LinearityAnalyzer.vue')],
  ['ReferenceInteractiveList', () => import('./components/ReferenceInteractiveList.vue')],
] as const

function isLocaleSwitch(from: string, to: string): boolean {
  const stripKo = (p: string) => p.replace(/^\/ko\//, '/')
  return stripKo(from) === stripKo(to) && from !== to
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router }) {
    if (typeof window !== 'undefined') {
      let savedScrollY = 0
      let pendingRestore = false

      router.onBeforeRouteChange = (to: string) => {
        const from = router.route.path
        if (isLocaleSwitch(from, to)) {
          savedScrollY = window.scrollY
          pendingRestore = true
        }
        return true
      }

      router.onAfterRouteChanged = () => {
        if (pendingRestore) {
          pendingRestore = false
          const y = savedScrollY
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              window.scrollTo(0, y)
            })
          })
        }
      }
    }

    globalComponents.forEach(([name, loader]) => {
      app.component(name, asyncComponent(loader))
    })
  },
}
