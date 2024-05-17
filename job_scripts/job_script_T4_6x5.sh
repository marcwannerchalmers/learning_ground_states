#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-887 -p alvis
#SBATCH -t 0-10:00:00
#SBATCH --gpus-per-node=T4:1

module load CUDA/12.1.1

cd ~/GS_experiments/
apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand.jl --Lx 6 --start_id 1 --npoints 16384