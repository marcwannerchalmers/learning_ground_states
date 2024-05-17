#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-397 -p alvis
#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=A40:1

module load CUDA/12.1.1

cd ~/GS_experiments/
apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand_nonloc.jl --Lx 7 --start_id 1 --npoints 256