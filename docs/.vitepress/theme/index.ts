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
  },
}
