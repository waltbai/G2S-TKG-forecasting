#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,7-8,16-17,24] --gpus=4
module load anaconda/2021.11
source activate bailong

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.stage1.train \
    --model_name_or_path /home/bingxing2/home/scx6592/zuoyuxin/LLM/LLaMA3-8B \
    --num_predictions 30 \
    --stage sft \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
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
    --overwrite_output_dir True \
    --logging_steps 10 \
    --save_steps 500 \
    --plot_loss True \
    --do_train True \
    --do_eval False \
    --do_predict False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1.0e-4 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True

llamafactory-cli export \
    --model_name_or_path /home/bingxing2/home/scx6592/zuoyuxin/LLM/LLaMA3-8B \
    --adapter_name_or_path /home/bingxing2/home/scx6592/bailong/data/llm4tkg/saves/llama3-icews18-stage1-sft \
    --template default \
    --finetuning_type lora \
    --export_dir /home/bingxing2/home/scx6592/bailong/models/llama3-icews18-stage1-sft \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
