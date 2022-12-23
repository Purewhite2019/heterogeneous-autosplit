import Vue from 'vue'
import App from './App.vue'
import "echarts"
import ECharts from 'vue-echarts'

Vue.config.productionTip = false
Vue.component('ECharts', ECharts)


new Vue({
  render: h => h(App),
}).$mount('#app')
