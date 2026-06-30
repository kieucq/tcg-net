#!/bin/bash -l
#SBATCH -A r00043
#SBATCH -J step_4
#SBATCH -t 00:59:00
#SBATCH -N 1
#SBATCH -p gpu-debug
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
conda deactivate
cd /N/slate/ckieu/typhoon-formation/models/resnet18/
step=$1
echo "Training model step ${step}"
module load python/gpu/3.10.10
python ./train_job_only.py \
    --project ResNet \
    --seed 53 \
    --ratio 30 \
    --step ${step} \
    --weight 15 \
