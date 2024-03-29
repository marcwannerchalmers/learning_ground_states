from torch import nn 
import torch
from util.transforms import Identity


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
        m.bias.data.fill_(0.01)

def get_transform(tf_string: str, **tf_args):
    if tf_string=="id":
        return Identity(**tf_args)