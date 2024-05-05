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
|-- src/            # Source codes
  |-- workflow/     # Experiment workflow control
  |-- prepare/      # Data preparation
  |-- train/        # Training
  |-- inference/    # Evaluation and Prediction
  |-- utils/        # Utility classes and functions
|-- config/         # Configuration files
|-- tests/          # Unit test cases
|-- docs/           # Documents
```
