import sys
sys.path.append('../')

import time
import torch
import logging
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from src.core import Client, Server, MPIConnection
from src.models.cifar import mobilenet
from mpi4py import MPI

import multiprocessing
from multiprocessing.connection import Connection
from server_flask import init_flask

import os

comm = MPI.COMM_WORLD
MAX_RANK = comm.Get_size()
SERVER_RANK = 0

rank = comm.Get_rank()

def is_server(rankp: int):
    return rankp == 0

if __name__ == '__main__':
    feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
    whole_train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=True, download=True,
                                                       transform=T.Compose(
                                                           [T.RandomVerticalFlip(), T.RandomResizedCrop(32),
                                                            T.ToTensor(),
                                                            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                        std=(0.247, 0.243, 0.261))]))
    whole_train_dataset = Subset(whole_train_dataset, range(1000))
    client_datasets = random_split(whole_train_dataset, [len(whole_train_dataset) // 2,
                                                         len(whole_train_dataset) - len(whole_train_dataset) // 2])

    def dataloader_fn(idx: int) -> DataLoader:
        return DataLoader(client_datasets[idx - 1], batch_size=32, shuffle=True, pin_memory=True)

    dump_path = f'experiment/example/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'

    client_layer_num = [2, 5]
    if is_server(rank):
        server_to_client_connection = MPIConnection(0)
        print('Server2Clients\' connections are all established')
        
        pipe = multiprocessing.Pipe()
        proc_flask = multiprocessing.Process(target=init_flask, args=(pipe[0], 
                                                                      dict(n_layers_total=len(feat_extractor)+len(classifier),
                                                                           client_num=len(client_layer_num))))
        proc_flask.start()
        
        runner = Server(dump_path, feat_extractor + classifier, 'sgd', dict(lr=1e-3, momentum=0.99),
        server_to_client_connection, MAX_RANK - 1, pipe=pipe[1])
        print('Server begins listening')
        runner.listen()
        print('Server terminate listening')
        runner.save_model('model_finished.pth')
        proc_flask.join()
    else:
        client_to_server_connection = MPIConnection(rank)
        runner = Client(rank, dump_path, feat_extractor, client_layer_num[rank-1], 'sgd', dict(lr=1e-2, momentum=0.99), dataloader_fn,
               server_connection=client_to_server_connection)
        print(f'Client {rank} begins training')
        runner.train(n_epoch=50)
        print(f'Client {rank} terminate training')
        runner.wait_for_sync()
        print(f'Client {rank} sync completed')
        test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=False, download=True,
                                                    transform=T.Compose([T.ToTensor(),
                                                                         T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                     std=(0.247, 0.243, 0.261))]))
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)
        runner.test(test_dataloader)