# Train, default to use 4 GPU cores
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/general_learning/train/GDELT-100k-FID-with-map.yaml

# Merge
llamafactory-cli export config/general_learning/merge/GDELT-100k-FID-with-map.yaml
