#!/bin/bash

cmake \
  -DCMAKE_CXX_FLAGS="-std=c++17 -fsycl" \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -DUSE_XPU=True \
  -DINTEL_EXTENSION_FOR_PYTORCH_PATH=`python -c 'import torch; print(torch.__path__[0].replace("torch","intel_extension_for_pytorch"))'` \
  ./

make

# Intel Extension for Torch path needed for Intel GPU
# -DINTEL_EXTENSION_FOR_PYTORCH_PATH=`python -c 'import torch; print(torch.__path__[0].replace("torch","intel_extension_for_pytorch"))'` \
