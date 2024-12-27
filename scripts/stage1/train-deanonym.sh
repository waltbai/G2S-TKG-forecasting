# Train, default to use 2 GPU cores
llamafactory-cli train config/stage1/train-GDELT-100k-deanonym.yaml

# Merge
llamafactory-cli export config/stage1/merge-GDELT-100k-deanonym.yaml
