import torch
from typing import Any
import numpy as np

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, gamma: float=0.9):
        self.reset()
        self.gamma = gamma
        self.should_record_values = False
        self.values_buffer = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = None
        self.should_record_values = False
        self.values_buffer = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        self.exp_avg = val if self.exp_avg is None else (self.gamma * self.exp_avg + (1 - self.gamma) * val)
        if self.should_record_values:
            self.values_buffer.append(val)
    
    def set_record_values(self, should_record_values: bool) -> None: 
        self.should_record_values = should_record_values
    
    @property
    def mean_without_min_max(self):
        return sum(sorted(self.values_buffer)[1:-1]) / (len(self.values_buffer) - 2)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k].shape)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def analyze(x: Any, depth: int=0) -> None:
    """util function used to analyze the structure of complex object

    Args:
        x (Any): Object to analyze
        depth (int, optional): the depth of current object, used for indent. Defaults to 0.
    """
    if isinstance(x, list):
        print('\t'*depth, '[', sep='')
        for v in x:
            analyze(v, depth+1);
        print('\t'*depth, ']', sep='')
        
    elif isinstance(x, dict):
        print('\t'*depth, '{', sep='')
        for k, v in x.items():
            print('\t'*depth, (k if not hasattr(k, 'shape') else k.shape), sep='')
            analyze(v, depth+1);
        print('\t'*depth, '}', sep='')
    
    else:
        print('\t'*depth, (x if not hasattr(x, 'shape') else x.shape), sep='')


class LinearFitting():
    def __init__(self, type:str, base_value, real_value, layers_num, c_number):
        if(type == "clientforward"):
            self.data_x = [base_value[f'client{c_number}-forwardbase-{i}'] for i in range(layers_num)]
            self.data_y = [real_value[f'client{c_number}-forward-{i}'] for i in range(layers_num)]
            self.cal_x = [base_value[f'client{c_number}-forwardbase-{i}'] for i in range(layers_num, 7)]
        elif(type == "clientbackward"):
            self.data_x = [base_value[f'client{c_number}-backwardbase-{i}'] for i in range(layers_num)]
            self.data_y = [real_value[f'client{c_number}-backward-{i}'] for i in range(layers_num)]
            self.cal_x = [base_value[f'client{c_number}-backwardbase-{i}'] for i in range(layers_num, 7)]
        elif(type == "client"):
            self.data_x = [(base_value[f'client{c_number}-forwardbase-{i}']+base_value[f'client{c_number}-backwardbase-{i}']) for i in range(layers_num)]
            self.data_y = [(real_value[f'client{c_number}-forward-{i}']+real_value[f'client{c_number}-backward-{i}']) for i in range(layers_num)]
            self.cal_x = [(base_value[f'client{c_number}-forwardbase-{i}']+base_value[f'client{c_number}-backwardbase-{i}']) for i in range(layers_num, 7)]
        elif(type == "serverforward"):
            self.data_x = [base_value[f'server-forwardbase-{i}'] for i in range(layers_num, 8)]
            # print(self.data_x)
            self.data_y = [real_value[f'server-forward-{i}'] for i in range(layers_num, 8)]
            # print(self.data_y)
            self.cal_x = [base_value[f'server-forwardbase-{i}'] for i in range(layers_num)]
        elif(type == "serverbackward"):
            self.data_x = [base_value[f'server-backwardbase-{i}'] for i in range(layers_num, 8)]
            self.data_y = [real_value[f'server-backward-{i}'] for i in range(layers_num, 8)]
            self.cal_x = [base_value[f'server-backwardbase-{i}'] for i in range(layers_num)]
        elif(type == "server"):
            self.data_x = [(base_value[f'server-forwardbase-{i}']+base_value[f'server-backwardbase-{i}']) for i in range(layers_num, 8)]
            self.data_y = [(real_value[f'server-forward-{i}']+real_value[f'server-backward-{i}']) for i in range(layers_num, 8)]
            self.cal_x = [(base_value[f'server-forwardbase-{i}']+base_value[f'server-backwardbase-{i}']) for i in range(layers_num)]
        elif(type == "network"):
            self.data_x = [(base_value[layers_num])/8, base_value[layers_num]]
            self.data_y = [real_value['network_zero'], real_value['network']]
            self.cal_x = base_value
        self.k = 0
        self.b = 0
        self.linear_fitting()
    
    def linear_fitting(self):
        size = len(self.data_x)
        i = 0
        sum_xy = 0
        sum_y = 0
        sum_x = 0
        sum_sqare_x = 0
        average_x = 0;
        average_y = 0;
        while i < size:
            sum_xy += self.data_x[i]*self.data_y[i];
            sum_y += self.data_y[i]
            sum_x += self.data_x[i]
            sum_sqare_x += self.data_x[i]*self.data_x[i]
            i += 1
        average_x = sum_x/size
        average_y = sum_y/size
        self.k = (size*sum_xy - sum_x*sum_y)/(size*sum_sqare_x - sum_x*sum_x)
        self.b = average_y - average_x*self.k
    
    def calculate(self):
        datay = []
        for x in self.cal_x:
            datay.append(self.k*x + self.b)
        return datay
