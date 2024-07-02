CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    -m src.stage2.train config/opt-350m-icews14/train-stage2.yaml
