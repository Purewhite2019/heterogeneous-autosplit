<!-- 这是用来画【拓扑图】的vue组件 -->

<template>
    <e-charts class="chart" :option="option" />
</template>
  
<script>

export default {
    name: 'Topology',
    props:['network_layers', 'client_num', 'client_layers_arr'],

    computed: {
        option() {
            let legend_data = ['server']  //图例分类
            let series_categories = [{ name: 'server' }]  //对应legend中的data
            for (let i = 1; i <= this.client_num; ++i) {
                legend_data.push(`client ${i}`)
                series_categories.push({ name: `client ${i}` })
            }

            //下面开始计算server节点在拓扑中的位置（长1000，宽350）
            let server_nodes = []
            let node_yInterval = 18  //节点上下之间的间隔
            let node_xInterval = 200  //节点左右之间的间隔，主要是不同client的间距
            for (let i = this.network_layers; i >= 1; --i) {  //这个循环计算server节点位置，并且添加到nodes数组里
                server_nodes.push({
                    name: `Server    L${i}`,
                    symbolSize: [150, 10],  //节点的长宽
                    x: 1000,
                    y: 0 + (this.network_layers - i) * node_yInterval,
                    label: {
                        show: true,  //显示节点的name
                        fontSize: 15
                    },
                    value: `可以写入的一些数据，这里先用随机数占位 = ${Math.random()}\n`,
                    category: 'server'  //可以直接写图例分类名
                })
            }

            //下面计算client节点在拓扑中的位置（长1000，宽350）
            let client_nodes = []
            for (let client = 1; client <= this.client_num; ++client) {
                for (let client_node = this.client_layers_arr[client - 1]; client_node >= 1; --client_node) {
                    client_nodes.push({
                        name: `Client${client} L${client_node}`,
                        x: 0 + (client - 1) * node_xInterval,
                        y: 350 + (this.client_layers_arr[client - 1] - client_node) * node_yInterval,
                        label: {
                            show: true,  //显示节点的name
                            fontSize: 15,
                            position: 'right',  //位置靠右
                            fontFamily: "Courier New"  //字体偏细

                        },
                        value: `可以写入的一些数据，这里先用随机数占位 = ${Math.random()}\n`,
                        category: `client ${client}`  //可以直接写图例分类名
                    })
                }
            }

            //下面计算server和client节点的中继点，实现线段
            let continue_nodes = []
            for (let i = 1; i <= this.client_num; ++i) {
                continue_nodes.push({
                    name: `Client ${i}`,
                    symbolSize: 5,  //中继点的长宽
                    symbol: 'circle',
                    x: 0 + (i - 1) * node_xInterval,
                    y: 0 + (this.network_layers - this.client_layers_arr[i - 1]) * node_yInterval,
                    label: {
                        show: true,  //显示节点的name
                        position: 'top',  //位置靠右
                        fontSize: 20,
                        //fontFamily: "Courier New"  //字体偏细
                    },
                    value: `acc = ${Math.random()}\n`,
                    category: `client ${i}`  //可以直接写图例分类名
                })
            }

            //下面计算连线的位置
            let links = []
            for (let i = 1; i <= this.client_num; ++i) {
                links.push({
                    source: `Client${i} L${this.client_layers_arr[i - 1]}`,
                    target: `Client ${i}`,
                }, {
                    source: `Client ${i}`,
                    target: `Server    L${this.client_layers_arr[i - 1]}`,
                },

                )
            }


            return {
                title: {  //标题相关
                    text: 'Network Topology of Split Learning',
                    textStyle: {
                        fontSize: 20
                    },
                    top: "left",
                    left: "center",

                },
                tooltip: {},
                legend: {  //图例
                    data: legend_data,  //图例分类
                    left: "center",
                    top: 45

                },
                animationDurationUpdate: 1500,
                animationEasingUpdate: 'quinticInOut',

                series: {
                    categories: series_categories,  //要对应legend中的data
                    type: 'graph',
                    layout: 'none',
                    draggable: true,  //可拖动节点，echats v5.4.1之后才行
                    top: '15%',
                    symbol: 'Rect',  //ECharts 提供的节点类型包括'circle', 'rect', 'roundRect', 'triangle', 'diamond', 'pin', 'arrow', 'none'
                    symbolSize: [100, 10],  //节点的长宽
                    roam: false,  //不可放大缩小
                    // label: {
                    //     show: false,  //显示节点的name
                    //     fontSize: 16
                    // },

                    edgeSymbol: ['circle', 'circle'],  //边的两端的标记类型
                    //edgeLabel: { show: false, fontSize: 20, },  //边标签样式
                    edgeSymbolSize: [4, 10],
                    lineStyle: {  //线的默认样式
                        color: 'source',  //颜色取决于源
                        opacity: 0.9,
                        width: 3,
                        curveness: 0,
                        type: "dashed"  //虚线
                    },

                    nodes: [].concat(server_nodes, client_nodes, continue_nodes),  //全部节点的信息

                    links: links  //全部连线的信息

                }

            }
        }
    },
}
</script>
  
<style scoped>
.chart {
    height: 690px;
}
</style>
  