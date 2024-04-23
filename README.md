# Large Language Model for Temporal Knowledge Graph Reasoning
This project aims to investigate how to utilize LLMs for TKG reasoning.

## Installation
**[Optional]** Create virtual environment:  
```shell
conda create -n llm4tkg python=3.8
```

Install dependencies:  
```shell
pip install -r requirements.txt
```

## Usage
Run unit test to ensure each module works:
```shell
python -m unittest discover
```

Evaluate in-context learning model:
```shell
scripts/run_icl.sh
```

## Plan
- [ ] Small-scale Preliminary Experiments on GPT2
  - [x] Preprocess
    - [x] TKG data loader
  - [x] Quadruple Prompt
    - [x] Quadruple-style Prompt
      - [x] Time anonymize
      - [x] Entity and Relation anonymize
      - [x] Unit test
  - [ ] Model
    - [x] In-Context Learning Model (2024.4.8~2024.4.14)
      - [x] Tokenization and indexing
      - [x] Predict: memory limit, do not use batching
      - [x] Evaluate
    - [ ] Fine-Tuning Model
      - [ ] Train
      - [ ] Checkpoint
    - [ ] Preferece-Tuning Model
      - [ ] Train
      - [ ] Checkpoint
  - [x] Metric
    - [x] Raw Hit@k
    - [x] Time-Filter Hit@k

## Anonymous Pretrain and Entity Assignment
### Pretrain
Learn variables and computations
[ENT_0, REL_0, ENT_1, xxx]
...

### Fine-tune
Baseline:  
[China, agreement, Russia, xxx]
[xxx]
Russia 

Assignment:  
Map: ENT_0: China, ENT_1: Russia
[ENT_0, REL_0, ENT_1, xxx]
ENT_1 -> Russia

[ENT_0: China, REL_0: make agreement, ENT_1: Russia, xxx]
