#!/bin/bash

CC=cc
CXX=CC

cmake \
    -DCMAKE_CXX_FLAGS="" \
    -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
    ./

make
