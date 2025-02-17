# Prepare GDELT data
python -m src.prepare config/general_learning/prepare/GDELT-100k-RID.yaml
python -m src.prepare config/general_learning/prepare/GDELT-100k-FID.yaml
python -m src.prepare config/general_learning/prepare/GDELT-100k-FID-with-map.yaml

# Prepare WIKI data
python -m src.prepare config/general_learning/prepare/WIKI-30k-RID.yaml
