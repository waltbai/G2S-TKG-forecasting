# Train, default to use 2 GPU cores
llamafactory-cli train config/fewshot/train-ICEWS14-from-pt.yaml

# Evaluate all checkpoints, default to use 2 GPU cores
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 2 \
    -m src.stage2.eval config/fewshot/eval-ICEWS14-from-pt.yaml
