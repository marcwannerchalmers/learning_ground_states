# Predicting ground state properties: Constant sample complexity and deep learning

## Setup
Run the following commands:
1. `python -m pip install - r requirements.txt`
2. `python generate_tensor_dataset.py`

The second command will populate the folder [data_torch](data_torch) with the datasets in the appropriate formate.
## Deep learning-based model

### Training
In order to train your deep learning-based model, run the following command:

`python -u train.py --config-name=config.yaml --config-path=conf`

The hyperparameters are specified via the config file [config.yaml](conf/config.yaml), where each parameter is documented in its comments. Adapt it following the same format or create your own config file and adapt the command accordingly. 

You may also try to use automated hyperparameter tuning via the file [hp_tuning.py](learner/hp_tuning.py). This feature is however experimantal and may not work yet.

### Evaluation

Run the following command: 
`python -u evaluate.py --config-name=config.yaml --config-path=conf`

This will evaluate metrics of interest, such as training and test error and magnitude of certain weights.
Please make sure to use the same config filename and path as for training.

## Regression-based model
The code is a slightly adapted version of the code by Lewis et al. 2023. In particular, we added a jax backend to the regression solver.
In order to train and evaluate the regression model, run the following command:
`python -u train_regression.py`

In order to specify hyperparameters, have a look at [train_regression.py](train_regression.py) and either use the flags of your choise or change the default options in the file.

## Plots

More general functionality for generating plots from the evaluation data will be added soon. In the mean time, we refer to [plot_results.py](plot_results.py) and [plot_results2.py](plot_results2.py) as an example. 

## Other files

The ($d$-dimensional) lattice model and the corresponding functionality to generate $I_P$ can be found in [geometry.py](model/geometry.py). The complete neural network is in [model.py](model/model.py). 

## Custom Heisenberg data generation
The main script for data generation is [generate_heisenberg_data.jl](data_generation/generate_heisenberg_data.jl), which is an adapted version from Huang et al. Our adapted version makes use of the new [CUDA version](https://itensor.org/news/23_10_23_gpu.html), by adding `using CUDA`to the script, which is from Oct 2023. Run the following command:
`julia generate_heisenberg_data.jl`
Make sure to navigate to the respective folder. See the file for flags specifying hyperparameters.

### DMRG Literature
Among an [introductory paper](https://web.mit.edu/8.334/www/grades/projects/projects14/Yu-An%20Chen+Hung-I%20Yang.pdf), there a paper where they claim to have done a [JAX implementation of DMRG](https://arxiv.org/pdf/2204.05693.pdf) tailored to TPUs. This here is the [original DMRG paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.69.2863), and here a longer, [comprehensive introduction](https://arxiv.org/pdf/1008.3477.pdf) connecting DMRG and matrix product states.