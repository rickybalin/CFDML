#!/bin/bash

source /flare/cfdml_aesp_CNDA/frameworks_2024.2.1_postAT/env.sh

LIBCEED_PATH=/flare/cfdml_aesp_CNDA/frameworks_2024.2.1_postAT/libCEED/libCEED
SIM_EXE=$LIBCEED_PATH/examples/fluids/navierstokes
#export MPICH_GPU_SUPPORT_ENABLED=1

# Setup env
echo "Loaded modules"
module list
echo

# Set up env variables
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_CQ_FILL_PERCENT=20
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE

# Print nodes
echo Running on nodes:
echo `cat $PBS_NODEFILE`
echo

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
SIM_PROCS_PER_NODE=12
SIM_RANKS=$((NODES * SIM_PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of simulation total ranks: $SIM_RANKS
echo Using libCEED from $LIBCEED_PATH
echo

# Run
TIME_STAMP=$(date '+%d-%m-%Y_%H-%M-%S')
echo "`date`: Starting run ..."
mpiexec -n $SIM_RANKS --ppn $SIM_PROCS_PER_NODE \
    --cpu-bind list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99  ./affinity_sim.sh \
    $SIM_EXE -options_file blasiusNGA.yaml \
    2>&1 | tee flatplate_${TIME_STAMP}.out
echo "`date`: Finished run "



