#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2021.11
source activate bailong
python -m src.stage1.prepare config/prepare/prepare-stage1-inline.yaml
