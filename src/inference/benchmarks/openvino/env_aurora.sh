#!/bin/bash

module use /soft/modulefiles
module load spack-pe-gcc 
module load cmake
source /gecko/Aurora_deployment/balin/OpenVINO/openvino/setupvars.sh 
# Need to add a path to the libtbb.so.2 library needed by OpenVINO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/datascience/llm_ds/basekit_2023_0_25537/vtune/2023.0.0/lib64
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
export ZE_AFFINITY_MASK=0.0

 
