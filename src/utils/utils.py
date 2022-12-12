import torch
from typing import Any


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k].shape)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def analyze(x: Any, depth: int=0) -> None:
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
