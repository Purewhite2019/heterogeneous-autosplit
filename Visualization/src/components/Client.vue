<!-- 这是用来画【折线图】的vue组件 -->

<template>
    <li>
        <e-charts class="chart" :option="option" />
    </li>
</template>
  
<script>

export default {
    name: 'Client',
    props: ['thisClient'],
    data() {
        return {
            dataSize: 1000,
            SES_a: 0.2  //指数平滑的系数
        }
    },

    computed: {
        option() {
            // console.log(this.thisClient.acc)  //离谱，de这个bug了一通宵

            let clientAcc = []
            if (this.thisClient.acc.length <= this.dataSize) {
                clientAcc = this.thisClient.acc
            }
            else {
                clientAcc = this.thisClient.acc.slice(this.thisClient.acc.length - this.dataSize)
            }

            for (let i = 1; i < clientAcc.length; ++i) {  //一次指数平滑
                clientAcc[i] = this.SES_a * clientAcc[i] + (1 - this.SES_a) * clientAcc[i - 1]
            }

            // let xAxis_data = new Array(this.thisClient.acc.length).fill('')
            let xAxis_data = new Array(this.dataSize).fill('')
            return {
                title: {
                    text: `Visualization of client ${this.thisClient.id} `,
                    textStyle: {
                        fontWeight: "bold",
                        fontSize: 14
                    },
                    left: "10%",  //左边空出10%
                },
                legend: {  //出现右上角能控制线条的标签与否的标识
                    right: "10%",
                    textStyle: {
                        fontSize: 14
                    }
                },
                xAxis: {
                    type: 'category',
                    data: xAxis_data,
                    // data: [1,2,3,4,5],
                    axisTick: {
                        show: false  //坐标轴刻度不显示
                    },
                    axisLabel: {
                        show: false,  //不显示x轴刻度
                        // interval: 0,  //X轴信息全部展示
                        // rotate: -60,  //60 标签倾斜的角度
                    }
                },
                yAxis: {
                    type: 'value',
                    max: 1.0
                    // boundaryGap: ['0%', 1],  //纵坐标留白
                },
                series: [
                    {
                        type: 'line',
                        // smooth: true,  //曲线图
                        showSymbol: false,  //不出现点点
                        data: clientAcc,
                        // data: [1,2],
                        name: `Accuracy of client ${this.thisClient.id} `,  //点出现的名字
                    },
                    // {
                    //     showSymbol: false,  //不出现点点
                    //     data: this.server_log2,
                    //     name: "Accuracy of client2",  //点出现的名字
                    //     type: 'line',
                    //     // smooth: true,  //曲线图
                    // },

                ],
                tooltip: {  //显示鼠标触发提示框
                    trigger: 'axis'
                }
            }
        }
    },

}
</script>
  
<style scoped>
.chart {
    height: 300px;
}

li {
    display: flex;
}
</style>
  