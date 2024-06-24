# Large Language Model for Temporal Knowledge Graph Reasoning
This project aims to investigate how to utilize LLMs for TKG reasoning.

## Develop platform
- cuda==12.2
- python==3.10
- torch==2.3.0
- transformers==4.41.2
- datasets==2.19.2
- llama_factory>=0.7.0
- tqdm

## Usage
**[Optional]** Create virtual environment:  
```shell
conda create -n llm4tkg python=3.8
```

Install dependencies:  
```shell
pip install -r requirements.txt
```

Run unit tests to ensure each module works:
```shell
python -m unittest discover
```

## Project Structure
```
root/
|-- src/                    # Source codes
  |-- stage1                # Stage-1: anonymized training stage
    |-- prepare.py          # Data preparation process
    |-- anonymizer.py       # Anonymize strategies
    |-- time_processor.py   # Time process strategies
    |-- prompt.py           # Prompt construct strategies 
    |-- train.py            # Train process
    |-- inference.py        # Valid and Test process
  |-- stage2                # Stage-2: de-anonymized training stage
  |-- utils                 # Common classes and functions
    |-- fact.py             # Fact class
    |-- tkg.py              # TKG class
    |-- query.py            # Query class
    |-- metric.py           # Metric functions
    |-- common.py           # Common functions
  |-- args.py               # Arguments
|-- config/                 # Configuration files
|-- tests/                  # Unit test cases
|-- docs/                   # Documents
```

## Dataset Statistics
| Dataset    | # Train facts | # Valid Facts | # Test Facts |
| :--------- | ------------: | ------------: | -----------: |
| ICEWS14    |        74,845 |         8,514 |        7,371 |
| ICEWS18    |       373,018 |        45,995 |       49,545 |
| ICEWS05-15 |       368,868 |        46,302 |       46,159 |
| WIKI       |       539,286 |        67,538 |       63,110 |
| YAGO       |       161,540 |        19,523 |       20,026 |

## Validation Experiment

Common settings:
| Param            |  Value  |
| :--------------- | :-----: |
| Test Dataset     | ICEWS14 |
| Stage            | stage1  |
| Triple Anonymize | session |
| Time Process     |  query  |
| Prompt Template  | inline  |

Results:

| Method | Train Datasets  | Valid Raw Hit@(1/3/10) | Valid Filter Hit@(1/3/10) | Test Raw Hit@(1/3/10) | Test Filter Hit@(1/3/10) |
| :----- | :-------------- | :--------------------: | :-----------------------: | :-------------------: | :----------------------: |
| ICL    | -               | 27.28 / 41.54 / 54.66  |   28.72 / 42.39 / 54.82   | 28.04 / 40.95 / 53.47 |  28.90 / 41.43 / 53.67   |
| SFT-1  | ICEWS14         | 30.54 / 44.40 / 55.00  |   32.49 / 45.10 / 55.21   | 31.70 / 43.64 / 53.96 |  33.01 / 44.15 / 54.09   |
| SFT-1  | ICEWS14,ICEWS18 |                        |                           |                       |                          |
