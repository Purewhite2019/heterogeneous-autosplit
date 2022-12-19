import torch
import numpy as np
import time
import random
from torch.utils.data import Subset, random_split
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

def get_rotated_dataset(dataset: str, n_clients: int, angles: list, istrain: bool):
    """ distribute data rotated to a certain angle to each client
    Args:
        dataset: type of dataset
        n_clients: number of clients
        angles: a list of angles
        istrain: train or test dataset
    """
    assert n_clients % len(angles) == 0
    clients_per_angle = n_clients // len(angles)
    client_datasets = []
    if dataset == 'cifar10':
        for angle in angles:
            whole_train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=istrain, download=True,
                                                                   transform=T.Compose([
                                                                        T.RandomResizedCrop(32), T.ToTensor(),
                                                                        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
                                                                                        MyRotateTransform(angle)]))
            client_datasets += random_split(whole_train_dataset, split_integer(len(whole_train_dataset), clients_per_angle))

    return client_datasets



def get_dataset_with_skewed_label(labels_per_client: int, dataset: str, n_clients: int, istrain: bool):
    """ distribute data with the same number of label types to each client
    Args:
        labels_per_client: number of labels assigned to each client
        dataset: type of dataset
        n_clients: number of clients
        istrain: train or test dataset
    """
    if dataset == 'cifar10':
        class_num = 10
        whole_train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=istrain, download=True,
                                                           transform=T.Compose(
                                                               [T.RandomVerticalFlip(), T.RandomResizedCrop(32),
                                                                T.ToTensor(),
                                                                T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                            std=(0.247, 0.243, 0.261))]))

        y_train = np.array(whole_train_dataset.targets)

        if labels_per_client == 10:
            client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_clients)
                for j in range(n_clients):
                    client_dataidx_map[j]=np.append(client_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(class_num)]
            contain=[]
            for i in range(n_clients):
                current=[i%class_num]
                times[i%class_num]+=1
                j=1
                while (j<labels_per_client):
                    ind=random.randint(0,class_num-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_clients)}
            for i in range(class_num):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                if times[i] == 0:
                    continue
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_clients):
                    if i in contain[j]:
                        client_dataidx_map[j]=np.append(client_dataidx_map[j],split[ids])
                        ids+=1
        client_datasets = [torch.utils.data.Subset(whole_train_dataset, client_dataidx_map[i]) for i in range(n_clients)]
        return client_datasets


class MyRotateTransform:
    def __init__(self, angle: int):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

