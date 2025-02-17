# default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    -m src.eval_zs config/specific_learning/zero-shot/eval/FID-GL-GDELT-100k-RID.yaml
