python -m src.prepare.prepare_tsro \
    --dataset_dir /data/bailong/data/tkg_data \
    --dataset ICEWS14 \
    --output_dir /data/bailong/data/tkg_data/prepare \
    --do_predict
python -m src.icl.run_icl_tsro \
    --model_name_or_path /data/bailong/models/gpt2 \
    --dataset_dir /data/bailong/data/tkg_data/prepare \
    --dataset ICEWS14 \
    --output_dir /data/bailong/data/tkg_data/results \
    --device cuda:0 \
    --num_predictions 30
