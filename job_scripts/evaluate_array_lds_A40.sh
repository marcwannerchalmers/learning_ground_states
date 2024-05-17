#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-397 -p alvis
#SBATCH -t 0-10:00:00
#SBATCH --gpus-per-node=A40:1

dir="dl_lds_A40/"

# get filenames
cd ~/GS_experiments/learning_ground_states/conf/$dir
configs=(*)

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
cd ~/GS_experiments/
source learn_gs/bin/activate
cd learning_ground_states/
python -u evaluate.py --config-path conf/$dir --config-name ${configs[$SLURM_ARRAY_TASK_ID]}