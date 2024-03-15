#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

/eagle/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/_ssim_env/bin/python -m smartsim._core.entrypoints.colocated +lockfile smartsim-75ec903.lock +db_cpus 1 +command taskset -c 0 /lus/eagle/projects/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/SmartSim/smartsim/_core/bin/redis-server /lus/eagle/projects/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/eagle/projects/datascience/balin/Polaris/SmartSim_envs/venv_conda-2023-10-04/SmartSim/smartsim/_core/lib/redisai.so THREADS_PER_QUEUE 4 INTER_OP_PARALLELISM 1 INTRA_OP_PARALLELISM 1 --port 0 --unixsocket /tmp/redis.socket --unixsocketperm 755 --logfile /dev/null --maxclients 100000 --cluster-node-timeout 30000 &
DBPID=$!

$@

