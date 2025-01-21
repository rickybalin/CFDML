#!/bin/bash

num_ranks_per_node=$1
shift
#num_gpu=6
#num_tile=2

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_LRANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_LRANKID=$PALS_LOCAL_RANKID
  _MPI_RANKID=$PALS_RANKID
else
  display_help
fi

#gpu_id=$(((_MPI_LRANKID / num_tile) % num_gpu))
#gpu_id=$((_MPI_LRANKID / num_tile))
#tile_id=$((_MPI_LRANKID % num_tile))
#gpu_name=$gpu_id.$tile_id
gpu_id=$((_MPI_LRANKID % num_ranks_per_node))
gpu_name=$gpu_id

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$gpu_id.$tile_id
echo ?RANK= ${_MPI_RANKID} LOCAL_RANK= ${_MPI_LRANKID} gpu= ${gpu_name}?

exec "$@"
