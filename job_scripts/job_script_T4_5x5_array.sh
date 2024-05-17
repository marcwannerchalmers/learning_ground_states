#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-397 -p alvis
#SBATCH -t 0-12:00:00
#SBATCH --gpus-per-node=T4:1

module load CUDA/12.1.1

cd ~/GS_experiments/

let npoints_orig=1024
let npoints=525
let offset=500
let startid=$SLURM_ARRAY_TASK_ID*$npoints_orig+1+offset

apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand.jl --Lx 5 --start_id $startid --npoints $npoints