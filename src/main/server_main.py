import sys
sys.path.append('../')

import time
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader



from src.core import Server, TCPConnection
from src.models.cifar import mobilenet


if __name__ == '__main__':
    feat_extractor, classifier = mobilenet(class_num=10).dump_layers()
    dump_path = f'experiment/example/{time.strftime("%Y%m%d-%H%M%S", time.localtime())}'

    server_to_client_connection = TCPConnection(is_server=True, server_ip='59.78.9.42', server_port=50000)
    print('Server2Clients\' connections are all established')
    runner = Server(dump_path, feat_extractor + classifier, 'sgd', dict(lr=1e-3, momentum=0.99),
           server_to_client_connection, 2)
    print('Server begins listening')
    runner.listen()
    print('Server terminate listening')
    runner.save_model('model_finished.pth')
