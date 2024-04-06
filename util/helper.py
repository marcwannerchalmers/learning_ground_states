from torch import nn 
import torch
from util.transforms import Identity, UnitInterval
from model.geometry import GridMap


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
    if mode == "edges":
        return gm.m
    else:
        raise NotImplementedError