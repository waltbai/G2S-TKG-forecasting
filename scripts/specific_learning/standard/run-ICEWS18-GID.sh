# Train, default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/standard/train/ICEWS18-GID.yaml

# Evaluate all checkpoints, default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.eval config/standard/eval/ICEWS18-GID.yaml
