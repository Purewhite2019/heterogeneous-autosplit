from src.core.trainer import DynamicNetwork, DynamicNetworkTrainer
from src.core.connections import Connection
from src.utils.utils import AverageMeter, LinearFitting

import os
import time
import re
import logging
from collections import deque
from tqdm import trange
from pprint import pformat
from copy import deepcopy
import signal

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
from sklearn.linear_model import LinearRegression

import multiprocessing.connection

from typing import List, Tuple, Any, Callable, Dict, Union, Optional
from functools import reduce

LOG_LEVEL = logging.DEBUG
BASIC_FORMAT_CLIENT = '%(asctime)s - %(filename)s[line:%(lineno)d] - [Client (CLIENTNO)] %(levelname)s: %(message)s'
BASIC_FORMAT_SERVER = f'%(asctime)s - %(filename)s[line:%(lineno)d] - [Server] %(levelname)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

PRE_RUN_ITERS = 16
ZERO_RUN_ITERS = 8
SPLIT_POINT_CHANGE_GAMMA = 0.9

# DNN Execution sequence: client front -> ... -> client back -> Communication() -> server front -> server back

class Client(DynamicNetworkTrainer):
    def __init__(self, number: int, dump_path: str,
                 model_layers: List[nn.Module], n_layers_left: int, optim_alg: str, optim_kwargs: dict,
                 dataloader_fn: Callable[[int], DataLoader],
                 server_connection: Connection) -> None:
        super().__init__(model_layers, optim_alg, optim_kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.number = number
        self.dump_path = dump_path
        self.dataloader = dataloader_fn(self.number)
        self.server_connection = server_connection

        self.unchange_times = 0

        os.makedirs(self.dump_path, exist_ok=True)
        self.server_connection.send(0, False, f'Client{self.number}Ready')  # This message helps TCPConnection to match client sockets with their client IDs. 

        # logging settings
        logger = logging.getLogger()
        
        logger.setLevel(LOG_LEVEL)
        
        basic_format = re.sub('(CLIENTNO)', str(self.number), BASIC_FORMAT_CLIENT)
        formatter = logging.Formatter(basic_format, DATE_FORMAT)
        
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        
        fhlr = logging.FileHandler(filename=os.path.join(self.dump_path, f'client{self.number}.log'), mode='a')
        fhlr.setFormatter(formatter)
        
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        self.model.to(self.device)
        self.model.train()
        
        def exit_handler(signum, frame):
            logging.warning(f'Interrupt caught ({signum}, {frame}), saving model and shutting down...')
            self.save_model(f'model_client{self.number}.pth')
            exit(0)

        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)
        
        self.n_layer_total = len(model_layers)+1
        logging.info(f'Preliminary initialization succeeded, using device {self.device}, begin pre-running...')
        
        # Record input shapes of each layer
        self.input_shapes = self.get_shapes(next(iter(self.dataloader))[0].shape, 0)
        logging.info(f'Input shapes: {self.input_shapes}')
        self.input_sizes = np.array([reduce((lambda x, y: x * y), x) for x in self.input_shapes])
        logging.info(f'Input sizes: {self.input_sizes}')
        
        self.network_meter = AverageMeter()
        self.network_meter_zero = AverageMeter()
        self.summary_server = None
        self.network_time = []
        
        # Client Pre-run
        self.client_forwards_base = np.empty(len(self.model.model_layers))
        self.client_backwards_base = np.empty(len(self.model.model_layers))
        self.server_forwards_base = np.empty(len(self.model.model_layers)+1)
        self.server_backwards_base = np.empty(len(self.model.model_layers)+1)
        
        server_base_value = dict()
        
        for i in range(len(self.model.model_layers)):
            self.model.forward_meters[i].set_record_values(True)
            self.model.backward_meters[i].set_record_values(True)
        
        
        network_meter_first = AverageMeter()
        network_meter_last = AverageMeter()
        
        network_meter_first.set_record_values(True)
        network_meter_last.set_record_values(True)
        for i, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Client pre-runs from layer 0, server runs classifier
            feat_client = self.forward(X)
            
            begin = time.time()
            logging.debug(f'Sending forward pre-run information to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
            self.server_connection.send(0, False, f'Client{self.number}Forward', len(self.model.model_layers), feat_client.cpu(), y.cpu(), {})

            msgs = self.server_connection.recv(False, source=0)
            assert len(msgs) == 1 and msgs[0][1] == {}
            msg_type, feat_grad, summary_server = msgs[0][0]
            assert msg_type == f'ServerBackwardToClient_{self.number}' and isinstance(feat_grad, torch.Tensor)
            feat_grad = feat_grad.to(self.device)
            logging.debug(f'Received backward pre-run information from server: {msg_type}, {feat_grad.shape}')
            end = time.time()
            
            network_meter_first.update(end - begin - sum([v[1] for v in summary_server.values()]))
            
            self.zero_grad()
            self.backward(feat_grad)
            self.step()

            # Server pre-runs from layer 0
            self.server_connection.send(0, False, f'Client{self.number}Forward', 0, X.cpu(), y.cpu(), {})

            begin = time.time()
            logging.debug(f'Sending forward raw-data to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
            msgs = self.server_connection.recv(False, source=0)
            assert len(msgs) == 1 and msgs[0][1] == {}
            msg_type, feat_grad, summary_server = msgs[0][0]
            assert msg_type == f'ServerBackwardToClient_{self.number}' and isinstance(feat_grad, torch.Tensor)
            feat_grad = feat_grad.to(self.device)
            logging.debug(f'Received backward raw-data information from server: {msg_type}, {feat_grad.shape}')
            end = time.time()
            
            network_meter_last.update(end - begin - sum([v[1] for v in summary_server.values()]))
            
            for k, v in summary_server.items():
                if k == 'network':          # Don't record network time in base value
                    continue
                if k not in server_base_value.keys():
                    server_base_value[k] = []
                server_base_value[k].append(v[1])
            
            self.merge_server_summary(summary_server)

            if i > PRE_RUN_ITERS:
                break
        
        for i in range(len(self.model.model_layers)):
            self.client_forwards_base[i] = self.model.forward_meters[i].mean_without_min_max
            self.client_backwards_base[i] = self.model.backward_meters[i].mean_without_min_max

        for k, v in server_base_value.items():
            mean_without_min_max = sum(sorted(v)[1:-1]) / (len(v) - 2)
            # server-backward-{i}
            _, direction, layer_idx = k.strip().split('-')
            layer_idx = int(layer_idx)
            if direction == 'forward':
                self.server_forwards_base[layer_idx] = mean_without_min_max
            elif direction == 'backward':
                self.server_backwards_base[layer_idx] = mean_without_min_max
            else:
                raise RuntimeError(f'Unrecognized key: {k}')

        logger.info(f'client_forwards_base: {self.client_forwards_base}')
        logger.info(f'client_backwards_base: {self.client_backwards_base}')
        logger.info(f'server_forwards_base: {self.server_forwards_base}')
        logger.info(f'server_backwards_base: {self.server_backwards_base}')
        
        for i in range(len(self.model.model_layers)):
            self.model.forward_meters[i].set_record_values(False)
            self.model.backward_meters[i].set_record_values(False)
        
        while len(self.model.model_layers) > n_layers_left:
            self.left_shift_split_point()

    def left_shift_split_point(self):
        logging.info(f'Left-shifting split point from {len(self.model.model_layers)}...')
        self.pop_back_layer()
        self.network_meter.reset()
        self.network_meter_zero.reset()
        logging.info('Left-shift finished')
    
    def right_shift_split_point(self):
        logging.info(f'Right-shifting split point from {len(self.model.model_layers)}...')
        
        self.server_connection.send(0, False, f'Client{self.number}RequestParameters', len(self.model.model_layers))
        msgs = self.server_connection.recv(False, source=0)
        assert len(msgs) == 1 and msgs[0][1] == {}
        msg_type, layer, optim_state_diff = msgs[0][0]
        assert msg_type == f'ServerReplyParametersToClient{self.number}' and isinstance(layer, nn.Module)
        logging.debug(f'Received layer from server: {msg_type}, {type(layer)}, {type(optim_state_diff)}')

        self.push_back_layer(layer, optim_state_diff)
        self.network_meter.reset()
        self.network_meter_zero.reset()
        logging.info('Right-shift finished')

    def auto_split(self, client_forwards: np.ndarray, client_backwards: np.ndarray, server_forwards: np.ndarray, server_backwards: np.ndarray, network_sizes: np.ndarray, network_times: np.ndarray, cur_time: float) -> None:
        if len(client_forwards) > 0 and len(self.model.model_layers) < len(self.client_forwards_base): 
            regr = LinearRegression(fit_intercept=(len(client_forwards) > 1))
            regr.fit(self.client_forwards_base[:len(self.model.model_layers)].reshape(-1, 1), client_forwards.reshape(-1, 1))
            client_forwards = np.concatenate([client_forwards, regr.predict(self.client_forwards_base[len(self.model.model_layers):].reshape(-1, 1)).reshape(-1)])
            del regr

            regr = LinearRegression(fit_intercept=(len(client_backwards) > 1))
            regr.fit(self.client_backwards_base[:len(self.model.model_layers)].reshape(-1, 1), client_backwards.reshape(-1, 1))
            client_backwards = np.concatenate([client_backwards, regr.predict(self.client_backwards_base[len(self.model.model_layers):].reshape(-1, 1)).reshape(-1)])
            del regr
        else:
            client_forwards = self.client_forwards_base
            client_backwards = self.client_backwards_base
        
        if len(server_forwards) > 0 and len(self.model.model_layers) > 0: 
            regr = LinearRegression(fit_intercept=(len(server_forwards) > 1))
            regr.fit(self.server_forwards_base[len(self.model.model_layers):].reshape(-1, 1), server_forwards.reshape(-1, 1))
            server_forwards = np.concatenate([regr.predict(self.server_forwards_base[:len(self.model.model_layers)].reshape(-1, 1)).reshape(-1), server_forwards])
            del regr

            regr = LinearRegression(fit_intercept=(len(server_backwards) > 1))
            regr.fit(self.server_backwards_base[len(self.model.model_layers):].reshape(-1, 1), server_backwards.reshape(-1, 1))
            server_backwards = np.concatenate([regr.predict(self.server_backwards_base[:len(self.model.model_layers)].reshape(-1, 1)).reshape(-1), server_backwards])
            del regr
        else:
            server_forwards = self.server_forwards_base
            server_backwards = self.server_backwards_base
        
        regr = LinearRegression(fit_intercept=(len(network_sizes) > 1))
        regr.fit(network_sizes.reshape(-1, 1), network_times.reshape(-1, 1))
        network_times = regr.predict(self.input_sizes.reshape(-1, 1)).reshape(-1)
        
        # client_forwards: shape: (self.n_layer_total, ), i-th element represents client forward time of i-th layer
        # client_backwards: shape: (self.n_layer_total, ), i-th element represents client backward time of i-th layer
        # server_forwards: shape: (self.n_layer_total+1, ), i-th element represents server forward time of i-th layer
        # server_backwards: shape: (self.n_layer_total+1, ), i-th element represents server backward time of i-th layer
        # network_times: shape: (self.n_layer_total+1, ), i-th element represents network time w.r.t. the **input** of i-th layer
        
        # total_times[i] = "client forward & backward time from layer 0 to layer i" + \
        #                  "network time to transport output of layer i (input of layer i+1) to server and receive gradient" + \
        #                  "server forward & backward time from layer i+1 to end"

        total_times = np.cumsum(client_forwards + client_forwards) + \
                      (np.sum(server_forwards[1:] + server_backwards[1:]) - np.cumsum(server_forwards[1:] + server_backwards[1:])) + \
                      network_times[1:]
        
        logging.info(f'Predicted total times w.r.t. split point: {total_times}')
        
        min_time_pred, best_split_point = np.min(total_times), np.argmin(total_times)
        best_split_point += 1   # best_split_point \in {1, 2, ..., self.n_layer_total}, leaving at least 1 layer and at most 1 layer on client and server, respectively.
        
        logging.info(f'Current time consumption: {cur_time}, Estimated min time consumption: {min_time_pred}')
        
        if (min_time_pred < SPLIT_POINT_CHANGE_GAMMA * cur_time):
            logging.info(f'Estimated best split point is {best_split_point} / {self.n_layer_total}, shifting...')
            while(best_split_point > len(self.model.model_layers)):
                self.right_shift_split_point()
            while(best_split_point < len(self.model.model_layers)):
                self.left_shift_split_point()
        
    def summarize(self, idx_start: int = 0) -> Dict:
        summary = dict()
        for i in range(idx_start, len(self.model.model_layers)):
            summary[f'client{self.number}-forward-{i}'] = self.model.forward_meters[i].exp_avg
            summary[f'client{self.number}-backward-{i}'] = self.model.backward_meters[i].exp_avg
        summary['network'] = self.network_meter.exp_avg #! Internet time
        summary.update(self.summary_server)
        return summary

    def short_summarize(self, idx_start: int = 0) -> Dict:
        if self.network_meter.count == 0:
            return {}
        summary = dict()
        for i in range(idx_start, len(self.model.model_layers)):
            summary[f'Client-{i}'] = self.model.forward_meters[i].exp_avg + self.model.backward_meters[i].exp_avg
        summary['Network'] = self.network_meter.exp_avg
        return summary

    def merge_server_summary(self, summary_server: dict) -> None:
        self.summary_server = summary_server
        
        for k, v in self.summary_server.items():
            self.summary_server[k] = v[0]

    def train(self, n_epoch: int=1) -> None:
        for e in trange(n_epoch):
            for i, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)
                feat_client = self.forward(X)

                feat_shape_last = feat_client.shape

                begin = time.time()
                logging.debug(f'Sending forward information to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
                self.server_connection.send(0, False, f'Client{self.number}Forward', len(self.model.model_layers), feat_client.cpu(), y.cpu(), self.short_summarize())

                msgs = self.server_connection.recv(False, source=0)
                assert len(msgs) == 1 and msgs[0][1] == {}
                msg_type, feat_grad, summary_server = msgs[0][0]
                assert msg_type == f'ServerBackwardToClient_{self.number}' and isinstance(feat_grad, torch.Tensor)
                feat_grad = feat_grad.to(self.device)
                logging.debug(f'Received backward information from server: {msg_type}, {feat_grad.shape}')

                end = time.time()
                self.merge_server_summary(summary_server)
                self.network_meter.update(end - begin - sum(summary_server.values()))   # Server total time - server train/inference time
                
                self.zero_grad()
                self.backward(feat_grad)
                self.step()
            
            # After 1 epoch: auto-split
            real_value = self.summarize()
            logging.info(f'Epoch {e}, timing summary: {pformat(real_value)}')
    
            # sent an empty data to measure network time
            feat_shape_empty = list(feat_shape_last)
            feat_shape_empty[0] = 0
            
            self.network_meter_zero.reset()
            self.network_meter_zero.set_record_values(True)
            
            for _ in range(ZERO_RUN_ITERS):
                feat_empty = torch.rand(feat_shape_empty)
                y = torch.zeros(size=[0], dtype=torch.int64)

                begin = time.time()
                logging.debug(f'Sending forward zero information to server: {len(self.model.model_layers)}, {feat_empty.shape}, {y.shape}')
                self.server_connection.send(0, False, f'Client{self.number}Forward', len(self.model.model_layers), feat_empty.cpu(), y.cpu(), {})

                msgs = self.server_connection.recv(False, source=0)
                assert len(msgs) == 1 and msgs[0][1] == {}
                msg_type, feat_grad, summary_server = msgs[0][0]
                assert msg_type == f'ServerBackwardToClient_{self.number}' and isinstance(feat_grad, torch.Tensor)
                feat_grad = feat_grad.to(self.device)
                logging.debug(f'Received backward zero information from server: {msg_type}, {feat_grad.shape}')

                end = time.time()
                self.network_meter_zero.update(end - begin - sum([v[1] for v in summary_server.values()]))   # Server total time - server train/inference time
            
            client_forwards = np.array([real_value[f'client{self.number}-forward-{i}'] for i in range(len(self.model.model_layers))])
            client_backwards = np.array([real_value[f'client{self.number}-backward-{i}'] for i in range(len(self.model.model_layers))])
            server_forwards = np.array([real_value[f'server-forward-{i}'] for i in range(len(self.model.model_layers), self.n_layer_total)])
            server_backwards = np.array([real_value[f'server-backward-{i}'] for i in range(len(self.model.model_layers), self.n_layer_total)])
            network_sizes = np.array([torch.prod(torch.tensor(feat_shape_last)), 0])
            network_times = np.array([real_value['network'], self.network_meter_zero.exp_avg])
            cur_time = sum(real_value.values())
            
            self.auto_split(client_forwards, client_backwards, server_forwards, server_backwards, network_sizes, network_times, cur_time)
    
    def wait_for_sync(self) -> None:
        logging.info('Synchronizing with the server...')

        self.server_connection.send(0, False, f'Client{self.number}Sync', self.model.state_dict(), self.optim.state_dict())
        msgs = self.server_connection.recv(False, source=0)
        assert len(msgs) == 1 and msgs[0][1] == {}
        msg_type, model_synced, optim_synced = msgs[0][0]
        assert msg_type == f'ServerSyncWithClient{self.number}' # and isinstance(model_synced, DynamicNetwork)
        self.model.load_state_dict(model_synced)
        self.optim.load_state_dict(optim_synced)
        
        # TODO: Should we Wait until all are synced?
        # self.server_connection.send(f'Client{self.number}SyncFinished', self.model)
        # msgs = self.server_connection.recv()
        # assert len(msgs) == 1 and len(msgs[0][1]) == None
        # assert msgs[0][0] == (f'ServerReady', )

        logging.info('Synchronization succeeded...')

    def test(self, test_dataloader: DataLoader):
        for X, y in test_dataloader:
            X, y = X.to(self.device), y.to(self.device)
            feat_client = self.forward(X)

            logging.debug(
                f'Sending forward information to server: {len(self.model.model_layers)}, {feat_client.shape}, {y.shape}')
            self.server_connection.send(0, False, f'Client{self.number}TestForward', len(self.model.model_layers),
                                        feat_client, y)
            logging.debug(
                f'Client {self.number} received the ACK signal from server')
            msgs = self.server_connection.recv(False, source=0)
            assert len(msgs) == 1 and msgs[0][1] == {}
    #! TODO: when to sync with server? how to sync with server?
    #* my idea: when an epoch of all client ends or a timer expires, they synchronize with the server.
    #* and upon synchronization with server, clients adapt its split points accordingly. 

class Server(DynamicNetworkTrainer):
    def __init__(self, dump_path: str,
                 whole_model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict,
                 client_connection: Connection, client_num: int,
                 pipe: multiprocessing.connection=None) -> None:
        super().__init__(whole_model_layers, optim_alg, optim_kwargs)


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dump_path = dump_path
        self.client_connection = client_connection
        self.client_num = client_num
        self.wait_for_sync = [None] * client_num
        
        self.pipe = pipe
        
        self.forward_meters = dict()
        self.backward_meters = dict()
        
        os.makedirs(self.dump_path, exist_ok=True)
        
        # logging settings
        logger = logging.getLogger()
        
        logger.setLevel(LOG_LEVEL)
        
        formatter = logging.Formatter(BASIC_FORMAT_SERVER, DATE_FORMAT)
        
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        
        fhlr = logging.FileHandler(filename=os.path.join(self.dump_path, f'server.log'), mode='a')
        fhlr.setFormatter(formatter)
        
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        self.model.to(self.device)
        self.model.train()
        
        def exit_handler(signum, frame):
            logging.warning(f'Interrupt caught ({signum}, {frame}), saving model and shutting down...')
            self.save_model('model_server.pth')
            exit(0)

        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)

    def summarize(self, idx_start: int=0) -> Dict:
        summary = dict()
        for i in range(idx_start, len(self.model.model_layers)):
            summary[f'server-forward-{i}'] = (self.model.forward_meters[i].exp_avg, self.model.forward_meters[i].val)
            summary[f'server-backward-{i}'] = (self.model.backward_meters[i].exp_avg, self.model.backward_meters[i].val)
        return summary

    def listen(self):
        logging.debug(f'Server is listening, using device {self.device}')
        while True:
            msgs = self.client_connection.recv()
            for (msg, kwmsg) in msgs:
                assert kwmsg == {}    # We don`t use kwmsg currently.
                if re.match(r'Client\d+Forward', msg[0]):
                    layer_idx, feat_client, y, short_summary = msg[1:]
                    client_idx = int(re.sub('Forward', "", re.sub('Client', "", msg[0])))

                    logging.debug(f'Received forward information from client {client_idx}: {layer_idx}, {feat_client.shape}, {y.shape}')
                    
                    # Measure training/inference time w.r.t. batch
                    if feat_client.shape[0] not in self.forward_meters.keys():
                        self.forward_meters[feat_client.shape[0]] = [AverageMeter() for _ in range(len(self.model.model_layers))]
                        self.backward_meters[feat_client.shape[0]] = [AverageMeter() for _ in range(len(self.model.model_layers))]
                    self.model.forward_meters = self.forward_meters[feat_client.shape[0]]
                    self.model.backward_meters = self.backward_meters[feat_client.shape[0]]
                    
                    # Load features and labels to device
                    feat_client, y = feat_client.to(self.device), y.to(self.device)
                    feat_client.requires_grad_(True)
                    # Server doesn't need self.forward(), self.model.forward() is OK, because it doesn't need to store any extra information..
                    logits = self.model(feat_client, layer_idx).detach().requires_grad_(True)
                    loss = F.cross_entropy(logits, y)

                    self.zero_grad()
                    loss.backward()
                    self.backward(logits.grad.detach(), layer_idx)
                    self.step()
                    
                    corrects = (torch.argmax(logits, dim=-1) == y).sum()
                    n_samples = y.shape[0]
                    logging.info(f'Accuracy of batch from client {client_idx}: {corrects/n_samples:.2f}({corrects}/{n_samples})')
                    
                    if self.pipe is not None and len(short_summary) != 0:
                        logging.debug(f'Sending information of client {client_idx} to flask server...')
                        for i in range(layer_idx, len(self.model.model_layers)):
                            short_summary[f'Server-{i}'] = self.model.forward_meters[i].exp_avg + self.model.backward_meters[i].exp_avg
                        self.pipe.send((client_idx, layer_idx, corrects.item(), n_samples, short_summary, str(list(feat_client.shape))))
                    logging.debug(f'Sending backward information to client {client_idx}: {feat_client.shape}')
                    self.client_connection.send(client_idx, False, f'ServerBackwardToClient_{client_idx}', feat_client.grad.detach().cpu(), self.summarize(layer_idx))
                    logging.debug(f'Backward information are successfully sent to client {client_idx}')

                elif re.match(r'Client\d+RequestParameters', msg[0]):
                    layer_idx = msg[1]
                    client_idx = int(re.sub('RequestParameters', "", re.sub('Client', "", msg[0])))

                    layer, optim_state_diff = self.dump_layer(layer_idx)
                    logging.debug(f'Sending layer to client {client_idx}: {type(layer)}, {type(optim_state_diff)}')
                    self.client_connection.send(client_idx, False, f'ServerReplyParametersToClient{client_idx}', layer, optim_state_diff)

                elif re.match(r'Client\d+Sync', msg[0]):
                    model_state, optim_state = msg[1:]
                    client_idx = int(re.sub('Sync', "", re.sub('Client', "", msg[0])))

                    if self.wait_for_sync[client_idx - 1] is not None:
                        raise RuntimeError(f'Client {client_idx} is already waiting for sync but sync signal was received again.')
                    self.wait_for_sync[client_idx - 1] = client_idx, model_state, optim_state
                
                elif re.match(r'Client\d+TestForward', msg[0]):
                    layer_idx, feat_client, y = msg[1:]
                    client_idx = int(re.sub('TestForward', "", re.sub('Client', "", msg[0])))

                    logging.debug(
                        f'Received forward information from client {client_idx}: {layer_idx}, {feat_client.shape}, {y.shape}')

                    feat_client, y = feat_client.to(self.device), y.to(self.device)
                    # Server doesn't need self.forward(), self.model.forward() is OK, because it doesn't need to store any extra information..
                    logits = self.model(feat_client, layer_idx)

                    corrects = (torch.argmax(logits, dim=-1) == y).sum()
                    n_samples = y.shape[0]
                    logging.info(
                        f'Test accuracy of batch from client {client_idx}: {corrects / n_samples:.2f}({corrects}/{n_samples})')

                    logging.debug(f'Sending acknowledge information to client {client_idx}: {feat_client.shape}')
                    self.client_connection.send(client_idx, False, f'ServerToClient{client_idx}')
                    logging.debug(f'Acknowledge information are successfully sent to client {client_idx}')

                elif re.match(r'Client\d+Ready', msg[0]):
                    client_idx = int(re.sub('Ready', "", re.sub('Client', "", msg[0])))
                    logging.info(f'Client {client_idx} has connected with server')
                
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
                            
                            if (len(new_layer_state) > 1):  # When more than 1 client participates in training the layer, an update is needed.
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
                        self.client_connection.send(client_idx, False, f'ServerSyncWithClient{client_idx}', model_state, optim_state)
                    self.wait_for_sync = [None] * self.client_num
