from typing import Any

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
