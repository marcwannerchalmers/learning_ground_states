from torch import nn 
import torch
from util.transforms import Identity, UnitInterval
from model.geometry import GridMap
from omegaconf import OmegaConf

class Sin(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.sin(x)

def get_activation(activation_string: str):
    if activation_string == "tanh":
        return nn.Tanh()
    elif activation_string == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def get_transform(tf_string: str, **tf_args):
    if tf_string=="id":
        return Identity(**tf_args)
    elif tf_string=="unit":
        return UnitInterval(**tf_args)
    else:
        raise NotImplementedError
    
def get_n_terms(mode: str, gm: GridMap):
    if mode == "edges": # terms correspond to edges in lattice
        return gm.m
    else:
        raise NotImplementedError
    
def update_cfg(cfg: OmegaConf, gm: GridMap):
    if cfg.ds_parameters.y_indices == "edges": # terms correspond to edges in lattice
        cfg.ds_parameters.y_indices = gm.edges.tolist()
    else:
        raise NotImplementedError
    
    return cfg
