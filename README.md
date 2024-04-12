# Large Language Model for Temporal Knowledge Graph Reasoning
This project aims to investigate how to utilize LLMs for TKG reasoning.

## Installation
**[Optional]** Create virtual environment:  
```conda create -n llm4tkg python=3.10```

Install dependencies:  
```pip install -r requirements.txt```

## Usage
To be described.

## Plan
- [ ] Small-scale Preliminary Experiments on GPT2
  - [x] Preprocess (2024.4.1~2024.4.7)
    - [x] TKG data loader
  - [x] Prompt
    - [x] Quadruple-style Prompt (2024.4.8~2024.4.14)
      - [x] Time anonymize
      - [x] Entity and Relation anonymize
    - [ ] Natural Language-style Prompt
      - [ ] Time anonymize
      - [ ] Entity and Relation anonymize
  - [ ] Model
    - [x] In-Context Learning Model (2024.4.8~2024.4.14)
      - [x] Tokenization and indexing
      - [x] Predict
      - [x] Evaluate
      - [x] Metric
        - [x] Time-Filter Hit@k
    - [ ] Fine-Tuning Model
