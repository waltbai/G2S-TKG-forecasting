# Large Language Model for Temporal Knowledge Graph Reasoning
This project aims to investigate how to utilize LLMs for TKG reasoning.

## Develop platform
- cuda==12.2
- python==3.10
- llama_factory==0.9.0
- tqdm

## Usage
**[Optional]** Create virtual environment:  
```shell
conda create -n llm4tkg python=3.10
```

Install dependencies:  
```shell
pip install -r requirements.txt
```

Run unit tests to ensure each module works:
```shell
python -m unittest tests
```

## Project Structure
```
root/
|-- data/                   # Dataset directory
  |-- ICWES05-15/
  |-- ICEWS14/
  |-- ICEWS18/
  |-- rules/                # Rules from TLogic
    |-- ICEWS14.json
    |-- ICEWS18.json
  |-- README.md             # Dataset descriptions
|-- src/                    # Source codes
  |-- stage1/               # Stage-2: dataset adaptation 
    |-- args.py             # Arguments
    |-- prepare.py          # Data preparation process
    |-- prompt.py           # Prompt construct strategies 
    |-- train.py            # Train process
    |-- inference.py        # Valid and Test process
  |-- utils/                # Common classes and functions
    |-- data/               # Data classes
      |-- fact.py           # Fact class
      |-- query.py          # Query class
      |-- tkg.py            # TKG class
    |-- common.py           # Common functions
    |-- metric.py           # Metric functions
|-- tests/                  # Unit test cases
```

## Dataset Statistics
| Dataset    | # Train facts | # Valid Facts | # Test Facts |
|:-----------|--------------:|--------------:|-------------:|
| ICEWS14    |        74,845 |         8,514 |        7,371 |
| ICEWS18    |       373,018 |        45,995 |       49,545 |
| ICEWS05-15 |       368,868 |        46,302 |       46,159 |
| WIKI       |       539,286 |        67,538 |       63,110 |
| YAGO       |       161,540 |        19,523 |       20,026 |
