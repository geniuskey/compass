import DefaultTheme from 'vitepress/theme'
import './custom.css'

import WavelengthSlider from './components/WavelengthSlider.vue'
import FresnelCalculator from './components/FresnelCalculator.vue'
import FourierOrderDemo from './components/FourierOrderDemo.vue'
import StackVisualizer from './components/StackVisualizer.vue'
import BayerPatternViewer from './components/BayerPatternViewer.vue'
import SnellCalculator from './components/SnellCalculator.vue'
import MaterialBrowser from './components/MaterialBrowser.vue'
import EMWaveAnimation from './components/EMWaveAnimation.vue'
import RCWAConvergenceDemo from './components/RCWAConvergenceDemo.vue'
import CrosstalkHeatmap from './components/CrosstalkHeatmap.vue'
import QESpectrumChart from './components/QESpectrumChart.vue'
import PixelStackBuilder from './components/PixelStackBuilder.vue'
import ConeIlluminationViewer from './components/ConeIlluminationViewer.vue'
import PrecisionComparison from './components/PrecisionComparison.vue'
import ThinFilmReflectance from './components/ThinFilmReflectance.vue'
import PolarizationViewer from './components/PolarizationViewer.vue'
import YeeCellViewer from './components/YeeCellViewer.vue'
import SolverComparisonChart from './components/SolverComparisonChart.vue'
import BlackbodySpectrum from './components/BlackbodySpectrum.vue'
import SignalChainDiagram from './components/SignalChainDiagram.vue'
import ModuleArchitectureDiagram from './components/ModuleArchitectureDiagram.vue'
import EnergyBalanceDiagram from './components/EnergyBalanceDiagram.vue'
import SolverPipelineDiagram from './components/SolverPipelineDiagram.vue'
import StaircaseMicrolensViewer from './components/StaircaseMicrolensViewer.vue'
import HeroAnimation from './components/HeroAnimation.vue'
import FeatureShowcase from './components/FeatureShowcase.vue'
import ArchitectureOverview from './components/ArchitectureOverview.vue'
import SolverShowcase from './components/SolverShowcase.vue'
import PixelAnatomyViewer from './components/PixelAnatomyViewer.vue'
import PhotonJourneyAnimation from './components/PhotonJourneyAnimation.vue'
import WavelengthExplorer from './components/WavelengthExplorer.vue'
import ConeIlluminationTopView from './components/ConeIlluminationTopView.vue'
import CrossSolverValidation from './components/CrossSolverValidation.vue'
import RcwaFdtdValidation from './components/RcwaFdtdValidation.vue'
import ConvergenceStudyChart from './components/ConvergenceStudyChart.vue'
import PerColorConvergenceChart from './components/PerColorConvergenceChart.vue'
import PixelCrossSections from './components/PixelCrossSections.vue'
import Pixel3DViewer from './components/Pixel3DViewer.vue'
import TmmQeCalculator from './components/TmmQeCalculator.vue'
import BarlOptimizer from './components/BarlOptimizer.vue'
import EnergyBudgetAnalyzer from './components/EnergyBudgetAnalyzer.vue'
import AngularResponseSimulator from './components/AngularResponseSimulator.vue'
import SnrCalculator from './components/SnrCalculator.vue'
import ColorFilterDesigner from './components/ColorFilterDesigner.vue'
import PixelDesignPlayground from './components/PixelDesignPlayground.vue'
import SiliconAbsorptionDepth from './components/SiliconAbsorptionDepth.vue'
import FabryPerotVisualizer from './components/FabryPerotVisualizer.vue'
import MicrolensRayTrace from './components/MicrolensRayTrace.vue'
import DiffractionPsfViewer from './components/DiffractionPsfViewer.vue'
import MtfAnalyzer from './components/MtfAnalyzer.vue'
import PixelScalingTrends from './components/PixelScalingTrends.vue'
import ColorAccuracyAnalyzer from './components/ColorAccuracyAnalyzer.vue'
import DarkCurrentSimulator from './components/DarkCurrentSimulator.vue'

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

    app.component('WavelengthSlider', WavelengthSlider)
    app.component('FresnelCalculator', FresnelCalculator)
    app.component('FourierOrderDemo', FourierOrderDemo)
    app.component('StackVisualizer', StackVisualizer)
    app.component('BayerPatternViewer', BayerPatternViewer)
    app.component('SnellCalculator', SnellCalculator)
    app.component('MaterialBrowser', MaterialBrowser)
    app.component('EMWaveAnimation', EMWaveAnimation)
    app.component('RCWAConvergenceDemo', RCWAConvergenceDemo)
    app.component('CrosstalkHeatmap', CrosstalkHeatmap)
    app.component('QESpectrumChart', QESpectrumChart)
    app.component('PixelStackBuilder', PixelStackBuilder)
    app.component('ConeIlluminationViewer', ConeIlluminationViewer)
    app.component('PrecisionComparison', PrecisionComparison)
    app.component('ThinFilmReflectance', ThinFilmReflectance)
    app.component('PolarizationViewer', PolarizationViewer)
    app.component('YeeCellViewer', YeeCellViewer)
    app.component('SolverComparisonChart', SolverComparisonChart)
    app.component('BlackbodySpectrum', BlackbodySpectrum)
    app.component('SignalChainDiagram', SignalChainDiagram)
    app.component('ModuleArchitectureDiagram', ModuleArchitectureDiagram)
    app.component('EnergyBalanceDiagram', EnergyBalanceDiagram)
    app.component('SolverPipelineDiagram', SolverPipelineDiagram)
    app.component('StaircaseMicrolensViewer', StaircaseMicrolensViewer)
    app.component('HeroAnimation', HeroAnimation)
    app.component('FeatureShowcase', FeatureShowcase)
    app.component('ArchitectureOverview', ArchitectureOverview)
    app.component('SolverShowcase', SolverShowcase)
    app.component('PixelAnatomyViewer', PixelAnatomyViewer)
    app.component('PhotonJourneyAnimation', PhotonJourneyAnimation)
    app.component('WavelengthExplorer', WavelengthExplorer)
    app.component('ConeIlluminationTopView', ConeIlluminationTopView)
    app.component('CrossSolverValidation', CrossSolverValidation)
    app.component('RcwaFdtdValidation', RcwaFdtdValidation)
    app.component('ConvergenceStudyChart', ConvergenceStudyChart)
    app.component('PerColorConvergenceChart', PerColorConvergenceChart)
    app.component('PixelCrossSections', PixelCrossSections)
    app.component('Pixel3DViewer', Pixel3DViewer)
    app.component('TmmQeCalculator', TmmQeCalculator)
    app.component('BarlOptimizer', BarlOptimizer)
    app.component('EnergyBudgetAnalyzer', EnergyBudgetAnalyzer)
    app.component('AngularResponseSimulator', AngularResponseSimulator)
    app.component('SnrCalculator', SnrCalculator)
    app.component('ColorFilterDesigner', ColorFilterDesigner)
    app.component('PixelDesignPlayground', PixelDesignPlayground)
    app.component('SiliconAbsorptionDepth', SiliconAbsorptionDepth)
    app.component('FabryPerotVisualizer', FabryPerotVisualizer)
    app.component('MicrolensRayTrace', MicrolensRayTrace)
    app.component('DiffractionPsfViewer', DiffractionPsfViewer)
    app.component('MtfAnalyzer', MtfAnalyzer)
    app.component('PixelScalingTrends', PixelScalingTrends)
    app.component('ColorAccuracyAnalyzer', ColorAccuracyAnalyzer)
    app.component('DarkCurrentSimulator', DarkCurrentSimulator)
  },
}
