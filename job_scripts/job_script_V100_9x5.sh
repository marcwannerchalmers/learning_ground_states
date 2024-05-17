#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-887 -p alvis
#SBATCH -t 0-2:00:00
#SBATCH --gpus-per-node=V100:1

module load CUDA/12.1.1

cd ~/GS_experiments/
apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand.jl --Lx 9 --start_id 1 --npoints 512