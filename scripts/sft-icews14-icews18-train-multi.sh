CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file config/accelerate_config.yml \
    -m src.stage1.train config/opt-350m-stage1-icews14-icews18-multi.yaml
