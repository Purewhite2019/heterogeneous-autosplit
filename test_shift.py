# coding: utf-8
from src.models.cifar.mobilenet import mobilenet
from src.core.trainer import DynamicNetwork, DynamicNetworkTrainer
import torch
import torchvision
from random import random

net = mobilenet()
backbone, classifier = net.dump_layers()
net = DynamicNetworkTrainer(backbone + classifier, 'sgd', {'lr' : 1e-3, 'momentum': 0.9})
x = torch.rand((8, 3, 32, 32), device='cpu')
z = net.forward(x)
net.backward(z)
net.step()

left = []
right = []
print(len(net.model.model_layers))
for i in range(50):
    print(len(left), len(net.model.model_layers), len(right), end='\t')
    if (len(net.model.model_layers) == 1 or random() < 0.5) and len(left)+len(right) > 0:
        if len(left) == 0 or (random() > 0.5 and len(right) > 0):
            print('push back')
            net.push_back_layer(*right[-1])
            right.pop(-1)
        else:
            print('push front')
            net.push_front_layer(*left[-1])
            left.pop(-1)
    else:
        if random() > 0.5:
            print('pop back')
            right.append(net.pop_back_layer())
        else:
            print('pop front')
            left.append(net.pop_front_layer())
            
while len(left) > 0:
    net.push_front_layer(*left[-1])
    left.pop(-1)
while len(right) > 0:
    net.push_back_layer(*right[-1])
    right.pop(-1)

x = torch.rand((8, 3, 32, 32), device='cpu')
z = net.forward(x)
net.backward(z)
net.step()
