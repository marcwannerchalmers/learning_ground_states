#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-887 -p alvis
#SBATCH -t 0-12:00:00
#SBATCH --gpus-per-node=T4:1

module load CUDA/12.1.1

cd ~/GS_experiments/

let npoints=1024
let offset=0
let startid=$SLURM_ARRAY_TASK_ID*$npoints+1+offset

apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand.jl --Lx 4 --start_id $startid --npoints $npoints