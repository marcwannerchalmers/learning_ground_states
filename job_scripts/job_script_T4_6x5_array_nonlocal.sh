#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-397 -p alvis
#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=A40:1

module load CUDA/12.1.1

cd ~/GS_experiments/

startids=(65 329 575 842 1100 1355 1615 1860 2122 2383 2633 2890 3129 3405 3654 3917)

let npoints_orig=64
let npoints=$npoints_orig
let offset=0
let startid_idx=$SLURM_ARRAY_TASK_ID/3
let offset=($SLURM_ARRAY_TASK_ID%3)*$npoints
let startid=${startids[$startid_idx]}+$offset

apptainer exec new_julia.sif julia learning_ground_states/data_generation/generate_heisenberg_data_rand_nonloc.jl --Lx 6 --start_id $startid --npoints $npoints