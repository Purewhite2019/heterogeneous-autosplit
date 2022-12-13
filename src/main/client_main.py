import sys
sys.path.append('../')

import time
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.core import Client, TCPConnection
from src.models.cifar import mobilenet

client_idx = 0
client_layer_num = 3
#need to be assigned respectively

if __name__ == '__main__':
    feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
    whole_train_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=True, download=True,
                                                       transform=T.Compose(
                                                           [T.RandomVerticalFlip(), T.RandomResizedCrop(32),
                                                            T.ToTensor(),
                                                            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                        std=(0.247, 0.243, 0.261))]))
    client_datasets = random_split(whole_train_dataset, [len(whole_train_dataset) // 2,
                                                         len(whole_train_dataset) - len(whole_train_dataset) // 2])

    def dataloader_fn(idx: int) -> DataLoader:
        return DataLoader(client_datasets[idx - 1], batch_size=32, shuffle=True, pin_memory=True)

    dump_path = f'experiment/example/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'

    client_to_server_connection = TCPConnection(is_server=False, server_ip='59.78.9.42', server_port=50000)
    runner = Client(client_idx, dump_path, feat_extractor[:client_layer_num], 'sgd', dict(lr=1e-2, momentum=0.99), dataloader_fn,
           server_connection=client_to_server_connection)
    print(f'Client {client_idx} begins training')
    runner.train(n_epoch=50)
    print(f'Client {client_idx} terminate training')
    runner.wait_for_sync()
    print(f'Client {client_idx} sync completed')
    test_dataset = torchvision.datasets.CIFAR10(root='data/cifar10/', train=False, download=True,
                                                transform=T.Compose([T.ToTensor(),
                                                                     T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                 std=(0.247, 0.243, 0.261))]))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)
    runner.test(test_dataloader)
