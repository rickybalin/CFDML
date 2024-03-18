#!/bin/bash

# Set executables
BASE_DIR=$PWD
DRIVER=$BASE_DIR/../../train/ssim_driver.py
SIM_EXE=$BASE_DIR/../../data_producers/smartredis/load_data.py
ML_EXE=$BASE_DIR/../../train/main.py
TRAIN_CONFIG=$PWD

# Set up run
SIM_RANKS=4
ML_RANKS=1
echo Number of simulation total ranks: $SIM_RANKS
echo Number of ML total ranks: $ML_RANKS
echo

# Run
SIM_ARGS="--model\=sgs --problem_size\=small --db_launch\=colocated --ppn\=${SIM_RANKS} --reproducibility\=True"
python $DRIVER \
    database.network_interface=lo database.launcher=local \
    sim.executable=$SIM_EXE sim.arguments="${SIM_ARGS}" \
    train.executable=$ML_EXE train.config="./sgs_train_config.yaml" \
    run_args.simprocs=${SIM_RANKS}  run_args.mlprocs=${ML_RANKS} run_args.mlprocs_pn=${ML_RANKS} \
    train.config=${TRAIN_CONFIG}
