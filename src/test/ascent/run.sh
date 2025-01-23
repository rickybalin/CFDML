#!/bin/bash

module use /soft/modulefiles
module load conda
conda activate

source /eagle/datascience/balin/CFDML/venv/_ssim/bin/activate

export PYTHONPATH=$PYTHONPATH:/soft/visualization/ascent/develop/2024-05-03-8baa78c/conduit-v0.9.1/python-modules
export PYTHONPATH=$PYTHONPATH:/soft/visualization/ascent/develop/2024-05-03-8baa78c/ascent-develop/python-modules

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

python driver.py
 
