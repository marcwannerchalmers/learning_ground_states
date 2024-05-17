from model.geometry import GridMap
from netket.hilbert import Spin
from netket.operator import Heisenberg
from netket.graph import Graph
import os
from data import Data
from netket.exact import lanczos_ed
#os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time
from jax.experimental.sparse import BCOO
from jax.experimental.sparse.linalg import lobpcg_standard
import jax.numpy as jnp
import sys

# THIS ATTEMPT IS DISCONTINUED

def get_heisenberg_operator(shape: tuple, x: list):
    graph = GridMap(shape).get_graph()
    hi = Spin(s=0.5, N=graph.n_nodes)
    op = Heisenberg(hilbert=hi, graph=graph, J=x)
    print(op.n_operators)
    return op

# convert to jax sparse matrix
# but jax has no smallest eigenvalue method yet
def heisenberg_to_jbcoo(op: Heisenberg):
    op_sparse = op.to_sparse().tocoo()
    indices = jnp.stack([jnp.array(op_sparse.row), jnp.array(op_sparse.col)], axis=1)
    data = jnp.array(op_sparse.data)
    return BCOO((data, indices), shape=op_sparse.shape, indices_sorted=True, unique_indices=True)

def main():
    
    ds = Data("data_torch", (4,5), 0, 500)
    x, y = ds[2]
    op = get_heisenberg_operator((4,5), x.tolist())
    print("computing energy...")
    start = time.time()
    evals = lanczos_ed(op, compute_eigenvectors=False)
    #evals, U, i = lobpcg_standard(jax_sparse, jnp.ones((jax_sparse.shape[0], 1)))
    end = time.time()
    print(evals[0])
    print("GT: ", y)
    print("Time: ", end-start)


if __name__ == "__main__":
    main()

