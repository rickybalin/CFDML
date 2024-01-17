#!/bin/bash

module load conda/2023-10-04
conda activate
module load gcc/11.2.0
module load oneapi/release
export LD_LIBRARY_PATH=/soft/compilers/oneapi/release/2023.2/compiler/2023.2.1/linux/lib/:$LD_LIBRARY_PATH
 
