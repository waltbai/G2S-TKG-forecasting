# Prepare FID data
python -m src.prepare config/specific_learning/standard/prepare/FID.yaml

# Prepare GID data
python -m src.prepare config/specific_learning/standard/prepare/GID.yaml

# Prepare RID data
python -m src.prepare config/specific_learning/standard/prepare/RID.yaml

# Prepare ablation data
python -m src.prepare config/specific_learning/standard/prepare/ICEWS14-GID-wo-ent.yaml
python -m src.prepare config/specific_learning/standard/prepare/ICEWS14-GID-wo-rel.yaml
python -m src.prepare config/specific_learning/standard/prepare/ICEWS14-GID-wo-ent-rel.yaml
