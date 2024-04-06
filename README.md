# Learning Ground States

## Setup
Run the following commands:
1. `python -m pip install - r requirements.txt`
2. `python generate_tensor_dataset.py`

## Implemented so far 

The ($d$-dimensional) lattice model and the corresponding functionality to generate $I_P$ can be found in [geometry.py](model/geometry.py). The complete neural network is in [model.py](model/model.py). 

Usage: Enter hyperparameters for training in [config.yaml](conf/config.yaml). \
Then, run `python train.py`.

### Netket Heisenberg model
Implemented our model using the [Netket library](https://netket.readthedocs.io/en/latest/). Exact Lanczos ground state computation gets killed locally due to too much memory usage from $5 \times 5$ grid onwards. Can potentially use other approximate ground state methods they provide.

## Possible resources for data generation
### iTensor
The approach in [generate_heisenberg_data.jl](data_generation/generate_heisenberg_data.jl) seems to have attempted to use GPUs, but the corresponding parts are commented. 
Suggestion: Learn the basics of Julia, tune the script a bit using [Performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/). Use the new [CUDA version](https://itensor.org/news/23_10_23_gpu.html), by adding `using CUDA`to the script, which is from Oct 2023. 
### Pennylane
The paper `provably efficient machine learning for quantum many-body problems´ has been implemented using [Pennylane](https://pennylane.ai/qml/demos/tutorial_ml_classical_shadows/). But it looks like they are missing an efficient method to compute the ground state (e.g. DMRG, as in Robert's version), which also caused the Netket version to be slow (and memory-inefficient). But it looks like pennylane can also do [DMRG states](https://pennylane.ai/qml/demos/tutorial_initial_state_preparation/), which is what I currently consider most promising. But there does not seem to be a GPU implementation. 
### cuDMRG
There is a [GPU-based DMRG implementation](https://github.com/ClarkResearchGroup/cuDMRG/tree/master) based on [CuPy](https://cupy.dev). We might manage to get an implementation by altering the file [heisenberg.py](https://github.com/ClarkResearchGroup/cuDMRG/blob/master/cuDMRG/apps/heisenberg.py). They also mention that `Hopefully more sparse solvers can be added in the future´, which might indicate that this is not optimal with respect to state of the art.
### DMRG Material
Among an [introductory paper](https://web.mit.edu/8.334/www/grades/projects/projects14/Yu-An%20Chen+Hung-I%20Yang.pdf), I came across a paper where they claim to have done a [JAX implementation of DMRG](https://arxiv.org/pdf/2204.05693.pdf), which is incredibly fast on strong TPUs. However, I could not find the code yet. This here is the [original DMRG paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.69.2863), and here a longer, [comprehensive introduction](https://arxiv.org/pdf/1008.3477.pdf) connecting DMRG and matrix product states.