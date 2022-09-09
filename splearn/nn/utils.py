import torch
from itertools import product

from splearn.utils import Config


def get_class_name(obj):
    return obj.__class__.__name__

def get_backbone_and_fc(backbone):
    backbone.output_dim = backbone.fc.in_features
    classifier = backbone.fc
    backbone.fc = torch.nn.Identity()
    return backbone, classifier


class HyperParametersTuning():
    '''
    Example usage:
    >>> configs = {
    >>>     'num_layers': [8,16],
    >>>     'dim': [128,256],
    >>>     'dropout': [0.5],
    >>> }
    >>> 
    >>> all_model_config = HyperParametersTuning(configs)
    >>> 
    >>> for i in range(all_model_config.get_num_configs()):
    >>>     print(all_model_config.get_config(i))
    '''
    def __init__(self, config):
        self.all_model_config = [dict(zip(configs, v)) for v in product(*configs.values())]
    
    def get_num_configs(self):
        return len(self.all_model_config)
    
    def get_config(self, i, return_config_object=True):
        if return_config_object:
            config = Config(self.all_model_config[i])
        else:
            config = self.all_model_config[i]
        return 