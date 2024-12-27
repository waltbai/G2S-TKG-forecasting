# Train, default to use 2 GPU cores
llamafactory-cli train config/stage1/train-GDELT-100k-anonym.yaml

# Merge
llamafactory-cli export config/stage1/merge-GDELT-100k-anonym.yaml
