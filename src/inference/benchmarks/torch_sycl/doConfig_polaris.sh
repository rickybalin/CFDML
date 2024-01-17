#!/bin/bash

CC=icx
CXX=icpx

cmake \
  -DCMAKE_C_COMPILER="icx" \
  -DCMAKE_CXX_COMPILER="icpx" \
  -DCMAKE_CXX_FLAGS="-std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80" \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  ./

make

