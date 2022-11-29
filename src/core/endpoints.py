from src.core.trainer import DynamicNetwork, DynamicNetworkTrainer
from src.core.connections import Connection

import os
import re
import logging
from collections import deque

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from mpi4py import MPI
from typing import List, Tuple, Any, Callable, Dict, Union


LOG_LEVEL = logging.DEBUG


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
        self.model.train()
    
    def left_shift_split_point(self):
        logging.info('Left-shifting split point...')
        self.pop_back_layer()
        logging.info('Left-shift finished')
    
    def right_shift_split_point(self):
        logging.info('Right-shifting split point...')
        
        self.server_connection.send(0, f'Client{self.number}RequestParameters', len(self.model.model_layers))
        msgs = self.server_connection.recv()
        assert len(msgs) == 1 and len(msgs[0][1]) == None
        msg_type, layer, optim_state_diff = msgs[0][0]
        assert msg_type == f'ServerReplyParametersToClient{self.number}' and isinstance(layer, nn.Module)
        logging.debug(f'Received layer from server: {msg_type}, {type(layer)}, {type(optim_state_diff)}')

        self.push_back_layer(layer, optim_state_diff)
        logging.info('Right-shift finished')

    def train(self, n_epoch: int=1) -> None:
        for e in range(n_epoch):
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                feat_client = self.forward(X)
                
                logging.debug(f'Sending forward information to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
                self.server_connection.send(0, False, f'Client{self.number}Forward', len(self.model.model_layers), feat_client, y)

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

        self.server_connection.send(0, False, f'Client{self.number}Sync', self.model.state_dict(), self.optim.state_dict())
        msgs = self.server_connection.recv()
        assert len(msgs) == 1 and len(msgs[0][1]) == None
        msg_type, model_synced, optim_synced = msgs[0][0]
        assert msg_type == f'ServerSyncWithClient{self.number}' and isinstance(model_synced, DynamicNetwork)
        self.model.load_state_dict(model_synced.state_dict())
        self.optim.load_state_dict(optim_synced)
        
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
                 client_connection: Connection, client_num: int) -> None:
        super().__init__(whole_model_layers, optim_alg, optim_kwargs)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dump_path = dump_path
        self.client_connection = client_connection
        self.wait_for_sync = [None] * client_num
        
        os.makedirs(self.dump_path, exist_ok=True)
        logging.basicConfig(format=f'%(asctime)s - %(filename)s[line:%(lineno)d] - [Server] %(levelname)s: %(message)s',
                    level=LOG_LEVEL,
                    filename=os.path.join(self.dump_path, f'server.log'),
                    filemode='a')

        self.model.to(self.device)
        self.model.train()
        
    def listen(self):
        while (True):
            msgs = self.client_connection.recv()
            for (msg, kwmsg) in msgs:
                assert kwmsg == {}    # We don`t use kwmsg currently.
                if re.match(r'Client\d+Forward', msg[0]):
                    layer_idx, feat_client, y = msg[1:]
                    client_idx = int(re.sub('Forward', "", re.sub('Client', "", msg[0])))

                    logging.debug(f'Received forward information from client {client_idx}: {layer_idx}, {feat_client.shape}, {y.shape}')

                    feat_client, y = feat_client.to(self.device), y.to(self.device)
                    feat_client.requires_grad_(True)
                    logits = self.forward(feat_client, layer_idx)
                    loss = F.cross_entropy(logits, y)

                    self.zero_grad()
                    loss.backward()
                    self.step()

                    logging.debug(f'Sending backward information to client {client_idx}: {feat_client.shape}')
                    self.client_connection.send(client_idx, False, f'ServerBackwardToClient{client_idx}', feat_client.grad.detach())

                elif re.match(r'Client\d+RequestParameters', msg[0]):
                    layer_idx = msg[1]
                    client_idx = int(re.sub('RequestParameters', "", re.sub('Client', "", msg[0])))

                    layer, optim_state_diff = self.dump_layer(layer_idx)
                    logging.debug(f'Sending layer to client {client_idx}: {type(layer)}, {type(optim_state_diff)}')
                    client_idx.send(client_idx, False, f'ServerReplyParametersToClient{client_idx}', layer, optim_state_diff)

                elif re.match(r'Client\d+Sync', msg[0]):
                    model_state, optim_state = msg[1:]
                    client_idx = int(re.sub('Sync', "", re.sub('Client', "", msg[0])))

                    if self.wait_for_sync[client_idx - 1] is not None:
                        raise RuntimeError(f'Client {client_idx} is already waiting for sync but sync signal was received again.')
                    self.wait_for_sync[client_idx - 1] = client_idx, model_state, optim_state

                else:
                    raise ValueError(f'Unknown message type "{msg[0]}".')
            
            if all([x is not None for x in self.wait_for_sync]):
                #* perform sync
                with torch.no_grad():
                    new_model_state = self.model.state_dict()
                    new_optim_state = self.optim.state_dict()
                    
                    # Update model params
                    for k, v in new_model_state.items():
                        new_layer_state = [v] + [model_state[k] for (_, model_state, _) in self.wait_for_sync if k in model_state.keys()]
                        new_layer_state = torch.stack(new_layer_state)
                        if new_layer_state.dtype == torch.int64:
                            new_layer_state = new_layer_state.sum(dim=0) // new_layer_state.shape[0]
                        else:
                            new_layer_state = new_layer_state.mean(dim=0)
                        
                        new_model_state[k] = new_layer_state
                        for (_, model_state, _) in self.wait_for_sync:
                            if k in model_state.keys():
                                model_state[k] = new_layer_state
                    
                    # Update optim states
                    if isinstance(self.optim, torch.optim.SGD): # For SGD
                        for k, v in new_optim_state['state'].items():
                            new_layer_state = [v['momentum_buffer']] + [optim_state['state'][k]['momentum_buffer'] for (_, _, optim_state) in self.wait_for_sync if k in optim_state['state'].keys()]
                            new_layer_state = torch.stack(new_layer_state)
                            new_layer_state = new_layer_state.mean(dim=0)
                            
                            new_optim_state[k] = new_layer_state
                            for (_, _, optim_state) in self.wait_for_sync:
                                if k in optim_state['state'].keys():
                                    optim_state['state'][k]['momentum_buffer'] = new_layer_state

                    else:   # For others
                        raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')

                    self.model.load_state_dict(new_model_state)
                    self.optim.load_state_dict(new_optim_state)
                    
                    for i, (client_idx, model_state, optim_state) in enumerate(self.wait_for_sync):
                        self.client_connections.send(client_idx, False, f'ServerSyncWithClient{client_idx}', model_state, optim_state)
                    
            

    def save_model(self, name=None):
        path = os.path.join(self.dump_path, 'model_checkpoint.pth' if name is None else name)
        torch.save(self.model, path)
        logging.info(f'Model saved in {path}')
