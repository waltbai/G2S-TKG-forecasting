# Prepare FID data
python -m src.prepare config/specific_learning/zero-shot/prepare/FID.yaml

# Prepare GID data
python -m src.prepare config/specific_learning/zero-shot/prepare/GID.yaml

# Prepare RID data
python -m src.prepare config/specific_learning/zero-shot/prepare/RID.yaml
