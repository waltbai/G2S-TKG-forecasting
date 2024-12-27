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
conda create -n llm4tkg python=3.11
```

Install dependencies:  
```shell
pip install -r requirements.txt
```

Run unit tests to ensure each module works:
```shell
python -m unittest tests
```

## Dataset Statistics
| Dataset    | # Train facts | # Valid Facts | # Test Facts |
|:-----------|--------------:|--------------:|-------------:|
| ICEWS14    |        74,845 |         8,514 |        7,371 |
| ICEWS18    |       373,018 |        45,995 |       49,545 |
| ICEWS05-15 |       368,868 |        46,302 |       46,159 |
| GDELT      |               |               |              |
| WIKI       |       539,286 |        67,538 |       63,110 |
| YAGO       |       161,540 |        19,523 |       20,026 |
