# Train, default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/low-resource/train/ICEWS14-05-GID.yaml

# Evaluate all checkpoints, default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.eval config/low-resource/eval/ICEWS14-05-GID.yaml
