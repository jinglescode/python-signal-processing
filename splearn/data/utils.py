import numpy as np

def onehot_targets(targets):
    return (np.arange(targets.max()+1) == targets[...,None]).astype(int)
