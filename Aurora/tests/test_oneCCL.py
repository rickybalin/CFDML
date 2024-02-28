import os
import sys
import datetime

# Import ML libraries
import torch
import intel_extension_for_pytorch as ipex
import torchvision

## Main function
def main():
    # Import and init MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(f"Hello from MPI rank {rank}/{size}")
    sys.stdout.flush()

    # Intel imports
    try:
        import intel_extension_for_pytorch
    except ModuleNotFoundError as err:
        if rank==0: print(err)
    try:
        import oneccl_bindings_for_pytorch
    except ModuleNotFoundError as err:
        if rank==0: print(err)

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

    # Get model and wrap with DDP
    model = torchvision.models.resnet50()
    model.to('xpu')
    model = DDP(model)
    print(f'[{rank}]: past DDP wrapper')
    sys.stdout.flush()

    # Distroy DDP process group
    dist.destroy_process_group()

    # Final MPI comm and print
    comm.Barrier()
    if (rank==0):
        print('\nExiting ...')

## Run main
if __name__ == "__main__":
    main()
