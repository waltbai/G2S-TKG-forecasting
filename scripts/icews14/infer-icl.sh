#!/bin/bash
#SBATCH --gpus=4
module load anaconda/2021.11
source activate bailong
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.stage1.inference \
    --model_name_or_path /home/bingxing2/home/scx6592/zuoyuxin/LLM-Chat/Meta-Llama-3-8B-Instruct \
    --num_predictions 30 \
    --dataset_dir /home/bingxing2/home/scx6592/bailong/data/tkg_data \
    --prepare_dir /home/bingxing2/home/scx6592/bailong/data/llm4tkg/prepare \
    --train_dataset ICEWS14 \
    --valid_dataset ICEWS14 \
    --test_dataset ICEWS14 \
    --history_finder 1-hop \
    --history_type entity \
    --history_direction uni \
    --history_length 30 \
    --anonymizer session \
    --anonymize_entity True \
    --anonymize_rel True \
    --time_processor query \
    --prompt_construct_strategy qa \
    --cutoff_len 2048 \
    --output_dir /home/bingxing2/home/scx6592/bailong/data/llm4tkg/outputs/llama3-icews14-stage1-icl \
    --do_train False \
    --do_eval True \
    --do_predict True \
    --bf16 True \
    --template default \
    --finetuning_type lora
