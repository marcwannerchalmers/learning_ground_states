# Categorizing pairs of qubits by distance

# grid of qubits
grid = np.array(range(1, length * width + 1)).reshape((length, width))

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
    