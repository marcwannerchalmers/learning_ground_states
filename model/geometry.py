import torch
from torch import nn
import numpy as np
from netket.graph import Graph

class LocalLayer(nn.Module):
    def __init__(self, parameter_map) -> None:
        super().__init__()
        self.parameter_map = parameter_map
    
    def forward(self, x):
        return [x[...,params] for params in self.parameter_map]

# pauli_qubits: list of iterable of qubits each pauli acts on
class GridMap:
    def __init__(self, shape, pauli_qubits=None, delta1=0) -> None:
        shape = tuple(shape)
        self.shape = shape
        self.d = len(shape)
        self.n = torch.prod(torch.tensor(shape))
        self.delta1 = delta1
        # indices of qubits on n-d grid structure (coordinate-to-qubit index)
        self.indices = torch.arange(self.n).reshape(shape)
        # list of all coordinates (qubit-index-to-coordinate)
        self.coordinates = torch.from_numpy(np.array([index for index in np.ndindex(shape)]))
        self.edges = self.get_edges()
        self.m = len(self.edges)
        self.pauli_qubits = self.edges if pauli_qubits is None else pauli_qubits
        
        self.parameter_map = [self.get_local_parameters(qubits) for qubits in self.pauli_qubits]

    def get_layer(self):
        return LocalLayer(self.parameter_map)
        
    # potentially generalize to different models
    def get_edges(self):
        # maybe improve this, but one-time calc
        edges = []
        for ind1 in range(self.n):
            for ind2 in range(ind1):
                # adjacent, so l1 distance 1
                if self.distance(ind1, ind2) == 1:
                    edges.append((ind2, ind1)) # index 1 can be larger than index 2, want lexographic order

        return torch.tensor(edges)
    
    # Given two qubits q1, q2 (1-indexed integers) in length x width grid
    # Output l1 distance between q1 and q2 in grid
    def distance(self, q1, q2):
        return torch.abs(self.coordinates[q1] - self.coordinates[q2]).sum()
    
    def get_nearby_qubits(self, q, delta1):
        return torch.argwhere((self.coordinates - self.coordinates[q].repeat(self.n,1)).float().norm(dim=1) <= delta1)

    # qubits: iterable of integers
    # returns indices of all parameters (parameters are edges) for a tensor with qubit indices
    def get_local_parameters(self, qubits):
        local_qubits = []
        for qubit in qubits:
            local_qubits.append(self.get_nearby_qubits(qubit, self.delta1))
        local_qubits = torch.tensor(local_qubits).flatten().unique()

        local_parameters = []
        for qubit in local_qubits:
            for j in range(2):
                loc_param = torch.argwhere(self.edges[:,j] == torch.ones((self.m,)) * qubit)
                if len(loc_param) > 0:
                    local_parameters.append(loc_param.flatten())

        return torch.cat(local_parameters).flatten().unique()
    
    def get_graph(self, all_unique_colors=True):
        edges = self.edges.tolist()
        if all_unique_colors:
            edges = [(edge[0], edge[1], i) for i, edge in enumerate(edges)]
        graph = Graph(edges)
        
        return graph
    
### LEGACY ###

def lllewis234_get_local():

    length = 0
    width = 0
    grid = 0
    # generate all edges in grid in same order as Xfull
    all_edges = []
    for i in range(0, length):
        for j in range(1, width + 1):
            if i != length - 1:
                all_edges.append((width * i + j, width * (i + 1) + j))
            if j != width:
                all_edges.append((width * i + j, width * i + j + 1))
    print(all_edges)
                
    def calc_distance(q1, q2):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid
        # Output l1 distance between q1 and q2 in grid

        pos1 = np.array(np.where(grid == q1)).T[0]
        pos2 = np.array(np.where(grid == q2)).T[0]

        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def get_nearby_qubit_pairs(d):
        # Given distance d > 0
        # Output all pairs of qubits that are within distance d of each other
        
        if d == 1:
            return all_edges
        
        qubit_pairs = []
        for q1 in range(1, length * width + 1):
            for q2 in range(1, length * width + 1):
                dist = calc_distance(q1, q2)
                pair = tuple(sorted((q1, q2)))
                if dist == d and pair not in qubit_pairs:
                    qubit_pairs.append(pair)
        
        return qubit_pairs

    # Finding local patches of a given radius

    def get_local_region_qubits(q, delta1):
        # Given a qubit q (1-indexed integer) in length x width grid and radius delta1
        # delta1 = -1 if all qubits are in local region
        # Output list of qubits (1-indexed integers) within a radius of delta1 of q
        
        if delta1 == 0:
            return [q]
        elif delta1 == -1:
            return list(range(1, length * width + 1))
        
        local_qubits = []
        for q2 in range(1, length * width + 1):
            dist = calc_distance(q, q2)
            
            if dist <= delta1:
                local_qubits.append(q2)
        
        return local_qubits

    def get_local_region_edges(q1, q2, delta1):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid and radius delta1
        # delta1 = -1 if all qubits are in local region
        # Output list of tuples of qubits (1-indexed integers) corresponding to edges in local region of radius delta1

        if delta1 == 0:
            return [(q1, q2)]
        elif delta1 == -1:
            return all_edges

        local_qubits = list(set(get_local_region_qubits(q1, delta1) + get_local_region_qubits(q2, delta1)))
        
        local_edges = []
        for edge in all_edges:
            (q1, q2) = edge
            if q1 in local_qubits and q2 in local_qubits:
                local_edges.append(edge)

        return local_edges

    def get_local_region_params(q1, q2, delta1, data, i):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid, radius delta1, and input data (i.e., Xfull)
        # delta1 = -1 if all qubits are considered nearby
        # Output data but only for parameters corresponding to edges within radius delta1
        
        edges = get_local_region_edges(q1, q2, delta1)
        
        indices = [all_edges.index(edge) for edge in edges]
        
        return np.array([data[i][j] for j in sorted(indices)])


# TESTING

def main():
    grid = GridMap((5,5))
    grid.get_edges()

if __name__ == "__main__":
    main()