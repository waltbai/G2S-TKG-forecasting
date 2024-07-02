#!/bin/bash
#SBATCH --gpus=4
#SBATCH -x paraai-n32-h-01-agent-[1,7-8,16-17,24]
module load anaconda/2021.11
source activate bailong

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.stage1.train config/llama3-icews14/train-stage1.yaml

llamafactory-cli export config/llama3-icews14/merge-stage1.yaml
