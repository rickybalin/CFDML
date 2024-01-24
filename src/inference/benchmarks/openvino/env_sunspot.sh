#!/bin/bash

module load frameworks/2023.12.15.001
source /gila/Aurora_deployment/balin/OpenVINO/openvino/setupvars.sh 
# Need to add a path to the libtbb.so.2 library needed by OpenVINO
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/datascience/llm_ds/basekit_2023_0_25537/vtune/2023.0.0/lib64 
