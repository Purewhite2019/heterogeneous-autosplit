<template>
    <div id="app">
        <div class="fancy" @click="isTopology = !isTopology">
            <span class="top-key"></span>
            <span class="text"> Switch Views</span>
            <span class="bottom-key-1"></span>
            <span class="bottom-key-2"></span>
        </div>

        <Topology v-if="isTopology == true" 
            :network_layers="network_layers" 
            :client_num="client_num"
            :client_layers_arr="client_layers_arr" 
        />

        <ul v-if="isTopology == false">
            <Client v-for="client in client_info" :key="client.id" 
                :thisClient="client" 
            />
        </ul>
    </div>

</template>
  
<script>
import axios from 'axios'
import Topology from './components/Topology.vue'
import Client from './components/Client.vue'

export default {
    name: 'App',
    components: { Topology, Client },
    data() {
        return {
            isTopology: true,
            client_info: [],
            network_layers: 10,
            client_num: 4,
            client_layers_arr: [3, 6, 4, 9],
        }
    },
    created() {
        let getData = () => {
            axios.get('./data.json', {}).then(value => {
                //console.log(value.data)
                this.client_info = value.data.client_info
                this.network_layers = value.data.network_layers
                this.client_num = value.data.client_num
                this.client_layers_arr = value.data.client_layers_arr
            });
        }

        let getRamdon = () => {
            axios.get('http://127.0.0.1:5000/accuracy', {}).then(value => {
                //console.log(value.data)
                this.client_info.forEach(item => {
                    // console.log(value.data.Accuracy)
                    item.acc = value.data.Accuracy  //离谱，这里居然还有个.Accuracy，de这个bug一通宵
                });
            });
        }

        getData()  //立即执行一次
        getRamdon()
        //this.timer = setInterval(getData, 5000)  //开启获取数据的定时器 
        this.timer2 = setInterval(getRamdon, 5000)  //从后端每x毫秒获取一次数据
    },
    beforeDestroy() {
        //clearInterval(this.timer)
        clearInterval(this.timer2)
        console.log('app即将驾鹤西游了')
    },
}
</script>
  
<style>
#app {
    margin: 10px;
}
.fancy {
 background-color: transparent;
 border: 2px solid #000;
 border-radius: 0;
 box-sizing: border-box;
 color: #fff;
 cursor: pointer;
 display: inline-block;
 /* float: right; */
 font-weight: 700;
 letter-spacing: 0.05em;
 margin: 0;
 outline: none;
 overflow: visible;
 padding: 1.25em 2em;
 position: relative;
 text-align: center;
 text-decoration: none;
 text-transform: none;
 transition: all 0.3s ease-in-out;
 user-select: none;
 font-size: 10px;
}

.fancy::before {
 content: " ";
 width: 1.5625rem;
 height: 2px;
 background: black;
 top: 50%;
 left: 1em; /* 调整黑线的定位  */
 position: absolute;
 transform: translateY(-50%);
 transform-origin: center;
 transition: background 0.3s linear, width 0.3s linear;
}

.fancy .text {
 font-size: 1.125em;
 line-height: 1.33333em;
 padding-left: 2em;
 display: block;
 text-align: left;
 transition: all 0.3s ease-in-out;
 text-transform: uppercase;
 text-decoration: none;
 color: black;
}

.fancy .top-key {
 height: 2px;
 width: 1.5625rem;
 top: -2px;
 left: 0.625rem;
 position: absolute;
 background: #e8e8e8;
 transition: width 0.5s ease-out, left 0.3s ease-out;
}

.fancy .bottom-key-1 {
 height: 2px;
 width: 1.5625rem;
 right: 1.875rem;
 bottom: -2px;
 position: absolute;
 background: #e8e8e8;
 transition: width 0.5s ease-out, right 0.3s ease-out;
}

.fancy .bottom-key-2 {
 height: 2px;
 width: 0.625rem;
 right: 0.625rem;
 bottom: -2px;
 position: absolute;
 background: #e8e8e8;
 transition: width 0.5s ease-out, right 0.3s ease-out;
}

.fancy:hover {
 color: white;
 background: black;
}

.fancy:hover::before {
 width: 0.9375rem;
 background: white;
}

.fancy:hover .text {
 color: white;
 padding-left: 1.5em;
}

.fancy:hover .top-key {
 left: -2px;
 width: 0px;
}

.fancy:hover .bottom-key-1,
 .fancy:hover .bottom-key-2 {
 right: 0;
 width: 0;
}
</style>
  
