#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-397 -p alvis
#SBATCH -t 0-14:00:00
#SBATCH --gpus-per-node=T4:1

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
cd ~/GS_experiments/
source learn_gs/bin/activate
cd learning_ground_states/
python -u train.py
