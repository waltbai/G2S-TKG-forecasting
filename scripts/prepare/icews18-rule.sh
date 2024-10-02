#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2021.11
source activate bailong
python -m src.stage1.prepare \
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
  --prompt_construct_strategy qa
