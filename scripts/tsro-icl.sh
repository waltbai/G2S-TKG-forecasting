CUDA_VISIBLE_DEVICES=0 python -m src.tsro.workflow \
    --dataset_dir /data/bailong/data/tkg_data \
    --dataset ICEWS14 \
    --prepare_dir /data/bailong/data/tkg_data/prepare \
    --model_name_or_path /data/bailong/models/gpt2 \
    --output_dir /data/bailong/data/tkg_data/result \
    --predict_with_generate \
    --do_predict
