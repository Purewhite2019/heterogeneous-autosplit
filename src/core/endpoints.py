from src.core.trainer import DynamicNetworkTrainer

import os
import logging

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from mpi4py import MPI
from typing import List, Tuple, Any, Callable, Dict, Union


LOG_LEVEL = logging.DEBUG

# TODO: Write this using MPI or other communication libraries
class Connection():
    """Abstract class defining the connection between 2 endpoints.
    """
    def __init__(self, dst_rank: int) -> None:
        self.dst_rank = dst_rank

    def send(self, *msg, **kwmsg) -> None:
        """Send a group of message to message queue of the other endpoint.
        Args:
            msg (Tuple, optional): Message to be sent in Tuple
            kwmsg (Dict, optional): Message to be sent in Dict
        """
        MPI.COMM_WORLD.send([msg, kwmsg], self.dst_rank)
    
    def recv(self) -> List[Tuple, Dict]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        this function keeps waiting until a message comes.
        """
        return MPI.COMM_WORLD.recv(self.dst_rank)


class Client(DynamicNetworkTrainer):
    def __init__(self, number: int, dump_path: str, model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict,
                 dataloader_fn: Callable[[int], DataLoader], server_connection: Connection) -> None:
        super().__init__(model_layers, optim_alg, optim_kwargs)
        
        self.number = number
        self.dump_path = dump_path
        self.dataloader = dataloader_fn(self.number)
        self.server_connection = server_connection
        self.split_layer_idx = len(model_layers)
        
        os.makedirs(self.dump_path, exist_ok=True)
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=LOG_LEVEL,
                    filename=os.path.join(self.dump_path, f'client{self.number}.log'),
                    filemode='a')
    
    def left_shift_split_point():
        pass
    
    def right_shift_split_point():
        pass
    
    def train(self, n_epoch: int=1) -> None:
        for e in range(n_epoch):
            for X, y in self.dataloader:
                feat_client = self.forward(X)
                
                logging.debug(f'Sending forward information to server: {self.split_layer_idx}, {feat_client.shape}, {y.shape}')
                self.server_connection.send(f'Client{self.number}Forward', self.split_layer_idx, feat_client, y)

                msgs = self.server_connection.recv()
                assert len(msgs) == 1
                assert len(msgs[0][1]) == None
                msg_type, feat_grad = msgs[0][0]
                assert msg_type == f'ServerBackwardToClient{self.number}'
                logging.debug(f'Received backward information from server: {msg_type}, {feat_grad.shape}')
                
                self.zero_grad()
                self.backward(feat_grad)
                self.step()


class Server(DynamicNetworkTrainer):
    def __init__(self, model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict) -> None:
        super().__init__(model_layers, optim_alg, optim_kwargs)


