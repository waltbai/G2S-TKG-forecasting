# Neural Network-based Experiment Template
This is a template for neural network-based experiments.

Change ``package_name`` to your module name.

## Usage
Install dependency:  
```pip install -r requirements.txt```

## preprocess
Preprocess transform raw corpus or source data into model-readable data, i.e., tensors.
It consists of a series of customized components, which will be run one-by-one.


## TODO list
- [x] config module
  - [x] load config
  - [x] import lib via path
- [ ] preprocess module
  - [x] preprocess pipeline
  - [ ] single-processing component
  - [ ] multi-processing component
