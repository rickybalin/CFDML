#!/bin/bash

cmake \
  -DCMAKE_CXX_FLAGS="-std=c++17 -fsycl" \
  ./

make

