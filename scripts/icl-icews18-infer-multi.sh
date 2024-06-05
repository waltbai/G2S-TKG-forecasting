CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file config/accelerate_config.yml \
     -m src.stage1.dist_inference config/opt-350m-stage1-ICEWS18-icl.yaml
