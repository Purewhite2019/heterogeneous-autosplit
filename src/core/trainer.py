import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from mpi4py import MPI

from typing import List, Tuple, Any


class DynamicNetwork(nn.Module):
    """Neural network which supports layer insertion and deletion

    Args:
        model_layers (List[nn.Module]): initial layers of the model
    """
    def __init__(self, model_layers: List[nn.Module]) -> None:
        super().__init__()
        self.model_layers = model_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model_layers:
            x = layer(x)

    def push_front_layer(self, layer: nn.Module) -> None:
        self.model_layers.append(layer)

    def push_back_layer(self, layer: nn.Module) -> None:
        self.model_layers.insert(0, layer)
    
    def pop_front_layer(self) -> nn.Module:
        layer = self.model_layers[0]
        self.model_layers.pop(0)
        return layer

    def pop_back_layer(self) -> nn.Module:
        layer = self.model_layers[-1]
        self.model_layers.pop(-1)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_forward_output = self.model.forward(x)
        return self.last_forward_output.detach()
    
    def backward(self, grad: torch.Tensor) -> None:
        self.last_forward_output.backward(grad)
    
    def zero_grad(self) -> None:
        self.optim.zero_grad()
    
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


def analyze(x: Any, depth: int=0):
    """util function used to analyze the structure of complex object

    Args:
        x (Any): Object to analyze
        depth (int, optional): the depth of current object, used for indent. Defaults to 0.
    """
    if isinstance(x, list):
        print('\t'*depth, '[', sep='')
        for v in x:
            analyze(v, depth+1);
        print('\t'*depth, ']', sep='')
        
    elif isinstance(x, dict):
        print('\t'*depth, '{', sep='')
        for k, v in x.items():
            print('\t'*depth, (k if not hasattr(k, 'shape') else k.shape), sep='')
            analyze(v, depth+1);
        print('\t'*depth, '}', sep='')
    
    else:
        print('\t'*depth, (x if not hasattr(x, 'shape') else x.shape), sep='')
