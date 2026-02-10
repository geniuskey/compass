import DefaultTheme from 'vitepress/theme'
import './custom.css'

import WavelengthSlider from './components/WavelengthSlider.vue'
import FresnelCalculator from './components/FresnelCalculator.vue'
import FourierOrderDemo from './components/FourierOrderDemo.vue'
import StackVisualizer from './components/StackVisualizer.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('WavelengthSlider', WavelengthSlider)
    app.component('FresnelCalculator', FresnelCalculator)
    app.component('FourierOrderDemo', FourierOrderDemo)
    app.component('StackVisualizer', StackVisualizer)
  },
}
