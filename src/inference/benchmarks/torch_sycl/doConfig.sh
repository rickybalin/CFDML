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

# Intel Extension for Torch path needed for Intel GPU
# -DINTEL_EXTENSION_FOR_PYTORCH_PATH=`python -c 'import torch; print(torch.__path__[0].replace("torch","intel_extension_for_pytorch"))'` \
