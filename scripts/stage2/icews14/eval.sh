CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    -m src.stage2.eval config/stage2/ICEWS14/train.yaml
