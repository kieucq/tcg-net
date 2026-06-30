#!/bin/bash -l
#SBATCH -A r00043
#SBATCH -J finetune
#SBATCH -t 05:59:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=240G
##SBATCH --cpus-per-gpu=64
set -x
src="/N/slate/ckieu/tcg-net/"
output_dir="/N/scratch/ckieu/tcg-net/output"
cd "${src}/models/finetune/"
step=$1
if [ "${step}" = "" ]; then
    echo "Must have an input for the step (leadtime) to finetune...exit 1"
    exit 1
fi
echo "Finetuning model for forecast lead time (step) = ${step}"
conda deactivate
module load python/gpu/3.10.10

# set up parameters
ratio=30
weight=6   
pretrain_model="ResNet"
data_source="ERA5"
train_start_year=1955
train_end_year=2023
test_start_year=1940
test_end_year=1954
seed=45
batch_size=512

# link data source
rm -f ${src}/output/DynamicDomain ${src}/output/PastDomain ${src}/output/POSITIVE
ln -sf ${output_dir}/${data_source}_dynamic ${src}/output/DynamicDomain
ln -sf ${output_dir}/${data_source}_past ${src}/output/PastDomain
ln -sf $output_dir/${data_source}_positive ${src}/output/POSITIVE

# creating a csv for all data
python generate_csv_all.py --inp_dir "${src}/output/" --out_dir "${src}/output/csv/"

# splitting the csv for train/test/validation
python split_data_training.py \
    --inp_dir "${src}/output/csv/" \
    --out_dir "${src}/output/csv/" \
    --train_start_year ${train_start_year} \
    --train_end_year ${train_end_year} \
    --test_start_year ${test_start_year} \
    --test_end_year ${test_end_year} \
    --rus_ratio ${ratio}

# runing finetune job
python ./finetune_job.py \
    --project ${pretrain_model} \
    --seed ${seed} \
    --pretrained "${src}/models/pre-trained/dynamic/${pretrain_model}_r${ratio}_w${weight}/Step_${step}_v0/checkpoints/last.ckpt" \
    --step ${step} \
    --ratio ${ratio} \
    --weight ${weight} \
    --batch_size ${batch_size} \
    --inp_dir "${src}/output/csv/DynamicRemain_rus${ratio}/Step_${step}" \
    --out_dir "${src}/models/finetune/${pretrain_model}_r${ratio}_w${weight}/Step_${step}" \
