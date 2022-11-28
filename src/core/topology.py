from src.core import Client, Server, MPIConnection
from src.models.cifar import mobilenet

import torch
import numpy as np
import time
from torch.utils.data import Subset, random_split
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class Topology():
    """Abstract class defining the topology between clients an servers and performs the experiment.
    """
    def __init__(self) -> None:
        pass
    
    def run(self) -> None:
        raise NotImplementedError()


class Topology_Example(Topology):
    """Implementation of Topology with 2 clients, 1 server, efficientnet_b0, CIFAR10
    """
    def __init__(self) -> None:
        feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
        whole_train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=True, download=True,
                                                           transform=T.Compose([T.RandomVerticalFlip(), T.RandomResizedCrop(32), T.ToTensor(),
                                                                                T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))]))
            
        client_datasets = random_split(whole_train_dataset, [len(whole_train_dataset)//2, len(whole_train_dataset)-len(whole_train_dataset)//2])
        
        def dataloader_fn(idx: int) -> DataLoader:
            return DataLoader(client_datasets[idx], batch_size=32, shuffle=True, pin_memory=True)
        
        self.client_to_server_connections = [] #TODO: write this
        self.server_to_client_connections = [] #TODO: write this
        
        self.dump_path = f'experiment/example/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'
        
        #TODO: Here, separate processes should be launched to work as clients and servers. 
        self.server = Server(self.dump_path, feat_extractor+classifier, 'sgd', dict(lr=1e-3, momentum=0.99), self.server_to_client_connections)
        self.clients = [
            Client(0, self.dump_path, feat_extractor, 'sgd', dict(lr=1e-2, momentum=0.99), dataloader_fn, server_connection=self.client_to_server_connections[0]),
            Client(1, self.dump_path, 'sgd', dict(lr=1e-2, momentum=0.99), dataloader_fn, server_connection=self.client_to_server_connections[1]),
        ]
    
    def run(self) -> None:
        #TODO: find some way to implement the following calls between processes
        self.server.listen()
        
        for c in self.clients:
            c.train(n_epoch=50)
    
        self.server.save_model('model_finished.pth')
        
        test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=False, download=True,
                                                    transform=T.Compose([T.ToTensor(), T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))]))
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.load(self.dump_path, 'model_finished.pth').to(device)
        
        for X, y in test_loader:
            pass
            #TODO: test model accuracy here