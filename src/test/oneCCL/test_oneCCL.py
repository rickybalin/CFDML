import os
import sys
import datetime
from argparse import ArgumentParser
#import socket

#from mpi4py import MPI

import torch
import intel_extension_for_pytorch as ipex
import torchvision
#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP

## Main function
def main():
    # Init MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(f"Hello from MPI rank {rank}/{size}", flush=True)
    comm.Barrier()

    # Intel imports
    try:
        import oneccl_bindings_for_pytorch as ccl
    except ModuleNotFoundError as err:
        if rank==0: print(err)

    # Parse arguments
    parser = ArgumentParser(description='CCL and torch distributes tests')
    parser.add_argument('--device', default="xpu", type=str, choices=['cpu','xpu'], help='Device to use for data and model')
    args = parser.parse_args()
    if rank==0:
        print('Running with arguments: \n',args,'\n',flush=True)

    # Initialize DDP process group
    import socket
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    backend = 'ccl'
    dist.init_process_group(backend,
                            rank=int(rank),
                            world_size=int(size),
                            init_method='env://',
                            timeout=datetime.timedelta(seconds=120))

    # Perform an allreduce with torch.distributed
    tensor = torch.ones(10,10).to(args.device)
    tensor_reduced = dist.all_reduce(tensor)
    comm.Barrier()
    if rank==0: print(f'\nPast torch.distributed.all_reduce \n', flush=True)
    
    # Get model and wrap with DDP
    model = torchvision.models.resnet50()
    model.to(args.device)
    model = DDP(model)
    comm.Barrier()
    if rank==0: print(f'Past DDP wrapper \n', flush=True)

    # Distroy DDP process group
    dist.destroy_process_group()

    # Final MPI comm and print
    comm.Barrier()
    if (rank==0):
        print('\nExiting ...')

## Run main
if __name__ == "__main__":
    main()
