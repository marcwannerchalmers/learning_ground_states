import torch
from torch import nn 
from model.geometry import GridMap
from util.helper import get_activation, init_weights, get_n_terms


class LocalDNN(nn.Module):
    def __init__(self, in_dim, width=10, depth=2, act_fun="tanh") -> None:
        super().__init__()
        assert(depth >= 1)

        self.in_dim = in_dim
        self.depth = depth
        self.act_fun = get_activation(act_fun)
        self.width = width

        # construct network
        self.layers = [nn.Linear(in_dim, width)]
        for _ in range(depth-1):
            self.layers.append(self.act_fun)
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, 1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
    
class SimpleFullDNN(nn.Module):
    def __init__(self, 
                 n_terms, 
                 geometry_parameters={},
                 local_parameters={}) -> None:
        super().__init__()
        
        self.gm = GridMap(**geometry_parameters)
        self.local_map = self.gm.get_layer()
        self.n_terms = n_terms if isinstance(n_terms, int) else get_n_terms(n_terms, self.gm) 
        self.models = nn.ModuleList([LocalDNN(len(loc_ind), **local_parameters) 
                                     for loc_ind in self.local_map.parameter_map])
        self.last_layer = nn.Linear(self.n_terms, 1, bias=False)
    

    def forward(self, x):
        # can be sped up if necessary using vmap
        # use more efficient map in more sophisticated version
        x = self.local_map(x)
        x = torch.cat([model(x_P) for model, x_P in zip(self.models, x)], dim=-1)
        x = self.last_layer(x).flatten()

        return x, self.last_layer.weight
    
    def init_xavier(self):
        self.apply(init_weights)

# this is the one consistent with the paper
class SeparateFullDNN(SimpleFullDNN):
    def __init__(self, n_terms, geometry_parameters={}, local_parameters={}) -> None:
        super().__init__(n_terms, geometry_parameters, local_parameters)

    def forward(self, x):
        # can be sped up if necessary using vmap
        # use more efficient map in more sophisticated version
        x = self.local_map(x)
        x = torch.cat([model(x_P) for model, x_P in zip(self.models, x)], dim=-1)
        x = (x * self.last_layer.weight)

        return x, self.last_layer.weight
        

    

