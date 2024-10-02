#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,7-8,16-17,24] --gpus=4
module load anaconda/2021.11
source activate bailong
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.stage1.inference \
    --model_name_or_path /home/bingxing2/home/scx6592/bailong/models/llama3-icews14-stage1-sft \
    --num_predictions 30 \
    --dataset_dir /home/bingxing2/home/scx6592/bailong/data/tkg_data \
    --prepare_dir /home/bingxing2/home/scx6592/bailong/data/llm4tkg/prepare \
    --train_dataset ICEWS18 \
    --valid_dataset ICEWS18 \
    --test_dataset ICEWS18 \
    --history_finder rule \
    --history_type entity \
    --history_direction uni \
    --history_length 50 \
    --anonymizer session \
    --anonymize_entity True \
    --anonymize_rel True \
    --time_processor query \
    --prompt_construct_strategy qa \
    --cutoff_len 2048 \
    --output_dir /home/bingxing2/home/scx6592/bailong/data/llm4tkg/saves/llama3-icews18-stage1-sft \
    --do_train False \
    --do_eval True \
    --do_predict True \
    --bf16 True \
    --template default \
    --finetuning_type lora

