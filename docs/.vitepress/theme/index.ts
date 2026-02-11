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

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
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
  },
}
