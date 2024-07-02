#!/bin/bash
#SBATCH --gpus=4
module load anaconda/2021.11
source activate bailong
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.stage1.inference config/llama3-icews14/infer-icl.yaml
