# G2S-tkg-forecasting
This is the official code for paper: "G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models".

## Develop platform
- 4 A800 (80G) GPUs
- cuda==12.2
- python==3.11
- llama_factory==0.9.1.dev0

## Usage
**[Optional]** Create virtual environment:  
```shell
conda create -n g2s python=3.11
```

Install dependencies:  
```shell
pip install -r requirements.txt
```

Run unit tests to ensure each module works:
```shell
python -m unittest tests
```

### General Learning Stage
Data preparation:
```shell
bash scripts/general_learning/prepare.sh
```

Training:
```shell
bash scripts/general_learning/train-GDELT-WIKI-130k-RID.sh
```

### Specific Learning Stage

#### Standard setting
Data preparation:
```shell
bash scripts/specific_learning/standard/prepare.sh
```

Run training and evaluation:
```shell
bash scripts/specific_learning/run-ICEWS14-GID.sh
```

#### Zero-shot setting
Data preparation:
```shell
bash scripts/specific_learning/zero-shot/prepare.sh
```

Run evaluation:
```shell
bash scripts/specific_learning/zero-shot/run-FID-GL-GDELT-WIKI-130k-RID.sh
```

#### Low-resource setting
Data preparation:
```shell
bash scripts/specific_learning/low-resource/prepare.sh
```

Run training and evaluation:
```shell
bash scripts/specific_learning/low-resource/run-ICEWS14-05-FID.sh
```

## Dataset Statistics
| Dataset | Schema   | # Entities | # Relations | # Train facts | # Valid Facts | # Test Facts | Time Granularity |
| :------ | -------- | ---------: | ----------: | ------------: | ------------: | -----------: | ---------------: |
| ICEWS14 | CAMEO    |      7,128 |         230 |        74,845 |         8,514 |        7,371 |            1 day |
| ICEWS18 | CAMEO    |     23,033 |         256 |       373,018 |        45,995 |       49,545 |            1 day |
| YAGO    | YAGO     |     10,623 |          10 |       161,540 |        19,523 |       20,026 |           1 year |
| GDELT   | CAMEO    |      7,691 |         240 |     1,734,399 |       238,765 |      305,241 |           15 min |
| WIKI    | Wikidata |     12,554 |          24 |       539,286 |        67,538 |       63,110 |           1 year |
