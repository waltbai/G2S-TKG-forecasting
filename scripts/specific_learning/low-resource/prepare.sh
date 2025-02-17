# Prepare FID data
python -m src.prepare config/specific_learning/low-resource/prepare/train-FID.yaml
python -m src.prepare config/specific_learning/low-resource/prepare/eval-FID.yaml

# Prepare GID data
python -m src.prepare config/specific_learning/low-resource/prepare/train-GID.yaml
python -m src.prepare config/specific_learning/low-resource/prepare/eval-GID.yaml

# Prepare RID data
python -m src.prepare config/specific_learning/low-resource/prepare/train-RID.yaml
python -m src.prepare config/specific_learning/low-resource/prepare/eval-RID.yaml
