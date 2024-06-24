CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --num_processes 8 \
    -m src.stage1.train config/opt-350m-icews14/train-stage1.yaml
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
#    --standalone \
#    --nnodes 1 \
#    --nproc_per_node 8 \
#    -m src.stage1.train config/opt-350m-icews14/train-stage1.yaml

llamafactory-cli export config/opt-350m-icews14/merge-stage1.yaml
