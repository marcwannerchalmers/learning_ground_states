import torch
from torch import nn 
from model.geometry import GridMap
from util.helper import get_activation, init_weights, get_n_terms
from torch.func import stack_module_state, functional_call
import copy

# standard fully connected deep network: [d_input] --> float
class LocalDNN(nn.Module):
    def __init__(self, in_dim, width=10, depth=2, act_fun="tanh", dropout=0.0) -> None:
        super().__init__()
        assert(depth >= 1)

        self.in_dim = in_dim
        self.depth = depth
        self.act_fun = get_activation(act_fun)
        self.width = width
        self.dropout = dropout

        # construct network
        self.layers = [nn.Linear(in_dim, width)]
        for _ in range(depth-1):
            self.layers.append(self.act_fun)
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.Dropout1d(dropout))
            # self.layers.append(nn.LayerNorm((width)))
        self.layers.append(nn.Linear(width, 1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

# model according to paper: [N.o. parameters] --> float
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

        return torch.stack((x, torch.ones_like(x) * self.last_layer.weight.abs().sum()))
    
    def init_xavier(self):
        self.apply(init_weights)


# Stacked SimpleFullDNNs using torch.vmap
# used to predict several observables in parallel
# [N.o. parameters] --> [N.o. observables]
class CombinedFullDNN(nn.Module):
    def __init__(self, n_terms, geometry_parameters={}, local_parameters={}, device="cpu") -> None:
        super().__init__()
        self.geometry_parameters = geometry_parameters
        self.local_parameters = local_parameters
        base_model = SimpleFullDNN(n_terms, self.geometry_parameters, self.local_parameters)
        self.n_terms = base_model.n_terms
        self.gm = base_model.gm
        # for performance
        base_model = base_model.to('meta')
        def f_model(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))
        self.f_model = f_model
       
        #self.f_model = lambda params, buffers, x: functional_call(base_model, (params, buffers), (x,))                                     
        self.models = nn.ModuleList([SimpleFullDNN(n_terms, geometry_parameters, local_parameters) for _ in range(self.n_terms)]).to(torch.device(device))
        self.params, self.buffs = stack_module_state(self.models)
        # seems a bit hacky, but works
        self._parameters = self.params
        """self.params = nn.ParameterDict(params)
        self.register_buffer('buffs', buffs, persistent=False)"""
  

    def forward(self, x):
        # can be sped up if necessary using vmap/or by making it a Linear layer
        # use more efficient map in more sophisticated version
        """xs = []
        ws = []
        for model in self.models:
            pred, w = model(x)
            xs.append(pred)
            ws.append(w)
        xs = torch.stack(xs, dim=-1)
        ws = torch.stack(ws, dim=-1)"""
        
        res_vmap = torch.vmap(self.f_model, in_dims=(0, 0, None))(self.params, self.buffs, x)
        xs, ws = res_vmap.permute(1, 2, 0)
        #print(torch.allclose(xs_v, xs_v, atol=1e-3, rtol=1e-5), torch.allclose(ws_v, ws_v, atol=1e-3, rtol=1e-5))
        return xs, ws
    
    def init_xavier(self):
        for model in self.models:
            model.init_xavier()
        

    

