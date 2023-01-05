import sys
sys.path.append('../')

import time
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from src.utils.data_process import get_raw_dataset, get_test_dataset, get_rotated_dataset, get_dataset_with_quantity_skewed_label, get_dataset_with_distribution_skewed_label
from src.core import Client, Server, MPIConnection
from src.models.cifar import mobilenet
from mpi4py import MPI

comm = MPI.COMM_WORLD
MAX_RANK = comm.Get_size()
SERVER_RANK = 0

rank = comm.Get_rank()

def is_server(rankp: int):
    return rankp == 0

DATASET = 'cifar10'


if __name__ == '__main__':
    if DATASET == 'cifar10':
        feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
    elif DATASET == 'cifar100':
        feat_extractor, classifier = mobilenet(class_num=100).dump_layers()
    elif DATASET == 'fashionmnist':
        feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
    
    client_datasets = get_raw_dataset(dataset=DATASET, n_clients=2)

    def dataloader_fn(idx: int) -> DataLoader:
        return DataLoader(client_datasets[idx - 1], batch_size=32, shuffle=True, pin_memory=True)

    dump_path = f'experiment/example/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'

    client_layer_num = [2, 5]
    if is_server(rank):
        server_to_client_connection = MPIConnection(0)
        print('Server2Clients\' connections are all established')
        runner = Server(dump_path, feat_extractor + classifier, 'sgd', dict(lr=1e-3, momentum=0.99),
               server_to_client_connection, MAX_RANK - 1)
        print('Server begins listening')
        runner.listen()
        print('Server terminate listening')
        runner.save_model('model_finished.pth')
    else:
        client_to_server_connection = MPIConnection(rank)
        runner = Client(rank, dump_path, feat_extractor, client_layer_num[rank-1], 'sgd', dict(lr=1e-2, momentum=0.99), dataloader_fn,
               server_connection=client_to_server_connection)
        print(f'Client {rank} begins training')
        runner.train(n_epoch=50)
        print(f'Client {rank} terminate training')
        runner.wait_for_sync()
        print(f'Client {rank} sync completed')
        test_dataset = get_test_dataset(dataset=DATASET)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)
        runner.test(test_dataloader)

