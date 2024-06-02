CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file config/accelerate_config.yml \
    -m src.stage1.train config/opt-350m-stage1-icews14-multi.yaml
