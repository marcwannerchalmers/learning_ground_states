import torch
from torch import nn 


class PartialDNN(nn.Module):
    def __init__(self, in_dim, width, depth=2, act_fun=nn.Tanh) -> None:
        assert(depth >= 1)

        self.in_dim = in_dim
        self.depth = depth
        self.act_fun = act_fun
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
    def __init__(self, n_terms, in_dim, width, depth=2, act_fun=nn.Tanh, mult_res=False) -> None:
        self.n_terms = n_terms
        self.mult_res = mult_res
        self.models = nn.ModuleList([PartialDNN(in_dim, width, depth, act_fun) for _ in range(n_terms)])

    def forward(self, x):
        # can be sped up if necessary using vmap
        # use more efficient map in more sophisticated version
        
        last_layer = x if self.mult_res else torch.ones_like(x)
        res = 0

        # also slow
        for i, model in enumerate(self.models):
            res += model*last_layer[i]

        return res