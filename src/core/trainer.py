from src.utils.utils import AverageMeter

import os
import logging
from typing import List, Tuple, Any, Dict
from time import time

import torch
import torch.nn as nn
import torch.optim as optim


class DynamicNetwork(nn.Module):
    """Neural network which supports layer insertion and deletion

    Args:
        model_layers (List[nn.Module]): initial layers of the model
    """
    def __init__(self, model_layers: List[nn.Module]) -> None:
        super().__init__()
        self.model_layers = nn.ModuleList(model_layers)
        self.output_size = [None for _ in range(len(self.model_layers))]
        self.forward_meters = [AverageMeter() for _ in range(len(self.model_layers))]
        self.backward_meters = [AverageMeter() for _ in range(len(self.model_layers))]
    
    def forward(self, x: torch.Tensor, idx_start: int=0) -> torch.Tensor:
        for i, layer in enumerate(self.model_layers[idx_start:]):
            begin = time()
            x = layer(x)
            end = time()
            self.output_size[idx_start + i] = x.shape
            self.forward_meters[idx_start+i].update(end - begin)
        return x

    def push_front_layer(self, layer: nn.Module) -> None:
        self.model_layers.append(layer)
        self.output_size.append(None)
        self.forward_meters.append(AverageMeter())
        self.backward_meters.append(AverageMeter())

    def push_back_layer(self, layer: nn.Module) -> None:
        self.model_layers.insert(0, layer)
        self.output_size.insert(0, None)
        self.forward_meters.insert(0, AverageMeter())
        self.backward_meters.insert(0, AverageMeter())

    #! Caution: the return values shouldn't be modified otherwise the running model parameters will be modified as well.
    def dump_layer(self, idx: int) -> nn.Module:
        layer = self.model_layers[idx]
        return layer

    def pop_front_layer(self) -> nn.Module:
        layer = self.model_layers[0]
        self.model_layers.pop(0)
        self.output_size.pop(0)
        self.forward_meters.pop(0)
        self.backward_meters.pop(0)
        return layer

    def pop_back_layer(self) -> nn.Module:
        layer = self.model_layers[-1]
        self.model_layers.pop(-1)
        self.output_size.pop(-1)
        self.forward_meters.pop(-1)
        self.backward_meters.pop(-1)
        return layer


class DynamicNetworkTrainer():
    """Trainer for DynamicNetwork class
    
    Args:
        model_layers (List[nn.Module]): initial layers of the model
        optim_alg (str): name of optimizer algorithm used in this Trainer
        optim_kwargs (dict) arguments of the optimizer used in this Trainer
    
    Example:
        ```
        model = torchvision.models.efficientnet_b0()
        feat_extractor = list(model.features) + [model.avgpool, nn.Flatten(1)]
        classifier = list(model.classifier)

        client_trainer = DynamicNetworkTrainer(model_layers=feat_extractor[:2], optim_alg='sgd', lr=1e-3, momentum=0.99)
        server_trainer = DynamicNetworkTrainer(model_layers=feat_extractor[2:]+classifier, optim_alg='sgd', lr=1e-3, momentum=0.99)

        for imgs, labels in dataloader:
            # Run at client
            feat_client = client_trainer.forward(imgs)
            
            # Run at server
            # send `feat_client` from client to server...
            feat_client.requires_grad_(True)
            logits = server_trainer.forward(feat_client)    
            loss = F.cross_entropy(logits, labels)
            
            server_trainer.zero_grad()
            loss.backward()
            server_trainer.step()
            
            # Run at client
            # send `feat_client.grad` as `grad` from server to client...
            client_trainer.zero_grad()
            client_trainer.backward(grad)
            client_trainer.step()
        ```
    """
    def __init__(self, model_layers: List[nn.Module], optim_alg: str, optim_kwargs: dict) -> None:
        self.model = DynamicNetwork(model_layers)
        

        if optim_alg.lower() == 'sgd':
            self.optim = optim.SGD(self.model.parameters(), **optim_kwargs)
            # 此处可以修改，但最多支持两组optim.param_groups，不然信息传递之类的会比较麻烦
        else:
            raise NotImplementedError(f'Optimizer "{optim_alg}" is not supported.')
        
        self.last_forward_output = None
    
    def summarize(self, idx_start: int=0) -> Dict:
        summary = dict()
        for i in range(idx_start, len(self.model.model_layers)):
            summary[f'forward-{i}'] = self.model.forward_meters[i].avg
            summary[f'backward-{i}'] = self.model.backward_meters[i].avg
        return summary
    
    def forward(self, x: torch.Tensor, idx_start: int=0) -> torch.Tensor:
        self.last_forward_output = self.model(x, idx_start)
        return self.last_forward_output.detach()
    
    def backward(self, grad: torch.Tensor) -> None:
        self.last_forward_output.backward(grad)
    
    def zero_grad(self) -> None:
        self.optim.zero_grad(set_to_none=True)
        #* `set_to_none=True` to avoid performing optimizer algorithm on layers that only exists in client end.
        #* This may affect, for example, momentum in SGD.
    
    def step(self) -> None:
        self.optim.step()

    def push_front_layer(self, layer: nn.Module, optim_state_diff: Any) -> None:
        self.model.push_front_layer(layer)
        
        optim_state = self.optim.state_dict()
        
        if isinstance(self.optim, optim.SGD):
            for param_groups in optim_state['param_groups']:
                param_groups['params'] = [idx+1 for idx in param_groups['params']]
            optim_state['state'] = {(k+1, v) for k, v in optim_state['state'].items()}
            
            optim_state['param_groups'][0]['params'].insert(0, 0)
            optim_state['state'][0] = dict(momentum_buffer=optim_state_diff)
        else:
            raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')
        
        self.optim.load_state_dict(optim_state)

    def push_back_layer(self, layer: nn.Module, optim_state_diff: Any) -> None:
        self.model.push_back_layer(layer)
        layer_idx = len(self.model.model_layers)
        
        optim_state = self.optim.state_dict()
        
        if isinstance(self.optim, optim.SGD):
            
            optim_state['param_groups'][-1]['params'].append(layer_idx)
            optim_state['state'][layer_idx] = dict(momentum_buffer=optim_state_diff)
        else:
            raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')

        self.optim.load_state_dict(optim_state)

    #! Caution: the returned layer parameters shouldn't be modified otherwise the running model parameters will be modified as well.
    def dump_layer(self, idx: int) -> Tuple[nn.Module, Any]:
        layer = self.model.dump_layer(idx)
        optim_state = self.optim.state_dict()
        
        if isinstance(self.optim, optim.SGD):
            optim_state_diff = optim_state['state'][idx]
        else:
            raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')
        
        return layer, optim_state_diff

    def pop_front_layer(self) -> Tuple[nn.Module, Any]:
        layer = self.model.pop_front_layer()
        optim_state = self.optim.state_dict()
        
        if isinstance(self.optim, optim.SGD):
            optim_state_diff = optim_state['state'][0]
            optim_state['state'].pop(0)
            optim_state['state'] = {(k-1, v) for k, v in optim_state['state'].items()}

            succeed = False
            for param_groups in optim_state['param_groups']:
                try:
                    param_groups['params'].remove(0)
                    succeed = True
                except:
                    pass
            if not succeed:
                raise RuntimeError(f'Fatal error: layer 0 doesn`t exist in all param_groups of the current optimizer.')

            for param_groups in optim_state['param_groups']:
                param_groups['params'] = [idx+1 for idx in param_groups['params']]
            
        else:
            raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')
        
        self.optim.load_state_dict(optim_state)
        
        return layer, optim_state_diff

    def pop_back_layer(self) -> Tuple[nn.Module, Any]:
        layer_idx = len(self.model.model_layers)
        layer = self.model.pop_back_layer()
        optim_state = self.optim.state_dict()
        
        if isinstance(self.optim, optim.SGD):
            optim_state_diff = optim_state['state'][layer_idx]
            optim_state['state'].pop(layer_idx)
            
            succeed = False
            for param_groups in optim_state['param_groups']:
                try:
                    param_groups['params'].remove(layer_idx)
                    succeed = True
                except:
                    pass
            if not succeed:
                raise RuntimeError(f'Fatal error: layer {layer_idx} doesn`t exist in all param_groups of the current optimizer.')

        else:
            raise NotImplementedError(f'Optimizer "{type(self.optim)}" is not supported.')
        
        self.optim.load_state_dict(optim_state)
        
        return layer, optim_state_diff

    def save_model(self, name=None):
        path = os.path.join(self.dump_path, 'model_checkpoint.pth' if name is None else name)
        torch.save(self.model, path)
        logging.info(f'Model saved in {path}')
