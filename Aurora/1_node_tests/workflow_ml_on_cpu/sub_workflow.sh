#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N cfdml_cpu_scale
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -k doe
#PBS -j oe
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR 
BASE_PATH=/flare/cfdml_aesp_CNDA/frameworks_2024.1

# Simulation
LIBCEED_PATH=$BASE_PATH/libCEED/libCEED
SIM_EXE=$LIBCEED_PATH/examples/fluids/navierstokes
SIM_CONFIG=$PWD/blasiusNGA_ssim.yaml
SIM_GPU_AFFINITY=$PWD/affinity_sim.sh

# Workflow and training
CFDML_PATH=$BASE_PATH/CFDML/CFDML
DRIVER=$CFDML_PATH/src/train/ssim_driver.py
DRIVER_CONFIG=$PWD
TRAIN_EXE=$CFDML_PATH/src/train/main.py
TRAIN_CONFIG=$PWD
TRAIN_GPU_AFFINITY=$PWD/affinity_ml.sh

# Setup env
module load frameworks/2024.1
source /lus/flare/projects/cfdml_aesp_CNDA/frameworks_2024.1/SmartSim/_ssim_env/bin/activate
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
echo "Loaded modules"
module list
echo

# Set up env vars
export PETSC_DIR=/lus/flare/projects/Aurora_deployment/balin/cfdml_aesp/PETSc/petsc_kej
export PETSC_ARCH=frameworks24.1
#export MPICH_GPU_SUPPORT_ENABLED=1

#export SR_LOG_FILE=stdout
export SR_LOG_LEVEL=QUIET
export SR_CONN_INTERVAL=10 # default is 1000 ms
export SR_CONN_TIMEOUT=1000 # default is 100 ms
export SR_CMD_INTERVAL=10 # default is 1000 ms
export SR_CMD_TIMEOUT=1000 # default is 100 ms
export SR_THREAD_COUNT=4 # default is 4

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
ML_PROCS_PER_NODE=12
ML_RANKS=$((NODES * ML_PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of simulation total ranks: $SIM_RANKS
echo Number of ML ranks per node: $ML_PROCS_PER_NODE
echo Number of ML total ranks: $ML_RANKS
echo Using libCEED from $LIBCEED_PATH
echo Running driver from $DRIVER
echo

# Run
echo "`date`: Starting run ..."
#SIM_ARGS="-options_file ${SIM_CONFIG}"
python $DRIVER --config-path $DRIVER_CONFIG \
    sim.executable=$SIM_EXE \
    run_args.simprocs=${SIM_RANKS} run_args.simprocs_pn=${SIM_PROCS_PER_NODE} \
    train.executable=$TRAIN_EXE train.config=${TRAIN_CONFIG} \
    run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_PROCS_PER_NODE} \
    sim.affinity=${SIM_GPU_AFFINITY}

echo "`date`: Finished run "

TIME_STAMP=$(date '+%d-%m-%Y_%H-%M-%S')
cd cfdml && mkdir $TIME_STAMP && cd ..
mv cfdml/sim cfdml/${TIME_STAMP}/sim
mv cfdml/train cfdml/${TIME_STAMP}/train

