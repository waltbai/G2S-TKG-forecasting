# Large Language Model for Temporal Knowledge Graph Reasoning
This project aims to investigate how to utilize LLMs for TKG reasoning.

## Requirements
DCU requirements:
- DTK==22.10
- python==3.8
- torch==1.13.0
- transformers==4.39.3
- datasets==2.19.0
- llama_factory>=0.7.0
- tqdm

Ideal requirements:
- python==3.10
- torch>=2.2
- transformers>=4.40
- datasets>=2.19
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

Run unit test to ensure each module works:
```shell
python -m unittest discover
```

## Project Structure
```
root/
|-- src/                    # Source codes
  |-- stage1                # Stage-1: anonymized training stage
    |-- workflow.py         # Overall controller of stage-1
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
| Dataset     |  # Train facts |  # Valid Facts |  # Test Facts |
|:------------|---------------:|---------------:|--------------:|
| ICEWS14     |         74,845 |          8,514 |         7,371 |
| ICEWS18     |        373,018 |         45,995 |        49,545 |
| ICEWS05-15  |        368,868 |         46,302 |        46,159 |
| WIKI        |        539,286 |         67,538 |        63,110 |
| YAGO        |        161,540 |         19,523 |        20,026 |
