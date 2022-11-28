from src.core.trainer import DynamicNetwork, DynamicNetworkTrainer

import os
import logging
from collections import deque

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
    
    def recv(self) -> List[Tuple[Tuple, Dict]]:
        """Retrieve all messages in the message queue. If the message queue is empty,
        this function keeps waiting until a message comes.
        it blocks until a message comes.

        Raises:
            NotImplementedError: this is an abstract class that should`n be called.

        Returns:
            List[Tuple, Dict]: List of messages in the message queue
        """
        return MPI.COMM_WORLD.recv(self.dst_rank)


# DNN Execution sequence: client front -> ... -> client back -> Communication() -> server front -> server back

class Client(DynamicNetworkTrainer):
    def __init__(self, number: int, dump_path: str,
                 model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict,
                 dataloader_fn: Callable[[int], DataLoader],
                 server_connection: Connection) -> None:
        super().__init__(model_layers, optim_alg, optim_kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.number = number
        self.dump_path = dump_path
        self.dataloader = dataloader_fn(self.number)
        self.server_connection = server_connection
        
        os.makedirs(self.dump_path, exist_ok=True)
        logging.basicConfig(format=f'%(asctime)s - %(filename)s[line:%(lineno)d] - [Client {self.number}] %(levelname)s: %(message)s',
                    level=LOG_LEVEL,
                    filename=os.path.join(self.dump_path, f'client{self.number}.log'),
                    filemode='a')

        self.model.to(self.device)
    
    def left_shift_split_point(self):
        logging.info('Left-shifting split point...')
        self.pop_back_layer()
        logging.info('Left-shift finished')
    
    def right_shift_split_point(self):
        logging.info('Right-shifting split point...')
        
        self.server_connection.send(f'Client{self.number}RequestParameters', len(self.model.model_layers))
        msgs = self.server_connection.recv()
        assert len(msgs) == 1 and len(msgs[0][1]) == None
        msg_type, layer, optim_state_diff = msgs[0][0]
        assert msg_type == f'ServerReplyParametersToClient{self.number}' and isinstance(layer, nn.Module)
        logging.debug(f'Received backward information from server: {msg_type}, {type(layer)}, {type(optim_state_diff)}')

        self.push_back_layer(layer, optim_state_diff)
        logging.info('Right-shift finished')

    def train(self, n_epoch: int=1) -> None:
        for e in range(n_epoch):
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                feat_client = self.forward(X)
                
                logging.debug(f'Sending forward information to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
                self.server_connection.send(f'Client{self.number}Forward', len(self.model.model_layers), feat_client, y)

                msgs = self.server_connection.recv()
                assert len(msgs) == 1 and len(msgs[0][1]) == None
                msg_type, feat_grad = msgs[0][0]
                assert msg_type == f'ServerBackwardToClient{self.number}' and isinstance(feat_grad, torch.Tensor)
                feat_grad = feat_grad.to(self.device)
                logging.debug(f'Received backward information from server: {msg_type}, {feat_grad.shape}')
                
                self.zero_grad()
                self.backward(feat_grad)
                self.step()
    
    def wait_for_sync(self) -> None:
        logging.info('Synchronizing with the server...')

        self.server_connection.send(f'Client{self.number}Sync', self.model)
        msgs = self.server_connection.recv()
        assert len(msgs) == 1 and len(msgs[0][1]) == None
        msg_type, model_synced = msgs[0][0]
        assert msg_type == f'ServerSyncWithClient{self.number}' and isinstance(model_synced, DynamicNetwork)
        self.model.load_state_dict(model_synced.state_dict())
        
        # TODO: Should we Wait until all are synced?
        # self.server_connection.send(f'Client{self.number}SyncFinished', self.model)
        # msgs = self.server_connection.recv()
        # assert len(msgs) == 1 and len(msgs[0][1]) == None
        # assert msgs[0][0] == (f'ServerReady', )

        logging.info('Synchronization succeeded...')

    #! TODO: when to sync with server? how to sync with server?
    #* my idea: when an epoch of all client ends or a timer expires, they synchronize with the server.
    #* and upon synchronization with server, clients adapt its split points accordingly. 


class Server(DynamicNetworkTrainer):
    def __init__(self, dump_path: str,
                 whole_model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict,
                 client_connections: List[Connection]) -> None:
        super().__init__(whole_model_layers, optim_alg, optim_kwargs)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dump_path = dump_path
        self.client_connections = client_connections
        self.messages = deque()

        os.makedirs(self.dump_path, exist_ok=True)
        logging.basicConfig(format=f'%(asctime)s - %(filename)s[line:%(lineno)d] - [Server] %(levelname)s: %(message)s',
                    level=LOG_LEVEL,
                    filename=os.path.join(self.dump_path, f'server.log'),
                    filemode='a')

        self.model.to(self.device)
    
    def listen(self):
        pass
