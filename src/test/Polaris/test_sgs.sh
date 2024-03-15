#!/bin/bash

# Set env
module load conda/2023-10-04
conda activate
source /eagle/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/_ssim_env/bin/activate
module list

# Set executables
BASE_DIR=/eagle/datascience/balin/CFDML/CFDML_GNN
DRIVER=$BASE_DIR/src/train/ssim_driver.py
SIM_EXE=$BASE_DIR/src/data_producers/smartredis/load_data.py
ML_EXE=$BASE_DIR/src/train/main.py
TRAIN_CONFIG=$PWD

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
SIM_PROCS_PER_NODE=8
SIM_RANKS=$((NODES * SIM_PROCS_PER_NODE))
ML_PROCS_PER_NODE=2
ML_RANKS=$((NODES * ML_PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of simulation total ranks: $SIM_RANKS
echo Number of ML ranks per node: $ML_PROCS_PER_NODE
echo Number of ML total ranks: $ML_RANKS
echo

# Run
SIM_ARGS="--model\=sgs --problem_size\=small --db_launch\=colocated --ppn\=${SIM_PROCS_PER_NODE}"
python $DRIVER \
    database.network_interface=uds \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config="./sgs_train_config.yaml" \
    run_args.simprocs=${SIM_RANKS} run_args.simprocs_pn=${SIM_PROCS_PER_NODE} \
    run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_PROCS_PER_NODE} \
    train.config=${TRAIN_CONFIG}
