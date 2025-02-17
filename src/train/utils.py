##### 
##### This script contains general utilities that can be useful
##### to many training applications and driver scripts 
#####

from os.path import exists
import sys
import numpy as np
import torch
import torch.distributed as dist


### MPI Communicator class
class MPI_COMM:
    def __init__(self):
        """
        MPI Communicator class
        """
        self.comm = None
        self.size = None
        self.rank = None
        self.name = None
        self.rankl = None
        self.sum = None
        self.minloc = None
        self.maxloc = None

    def init(self, cfg, print_hello=False):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()
        self.rankl = self.rank % cfg.ppn
        self.sum = MPI.SUM
        self.min = MPI.MIN
        self.max = MPI.MAX
        self.minloc = MPI.MINLOC
        self.maxloc = MPI.MAXLOC
        if print_hello:
            print(f"Hello from MPI rank {self.rank}/{self.size} and local rank {self.rankl}",flush=True)

### Compute the average of a quantity across all ranks
def metric_average_mpi(comm, val):
    avg_val = comm.comm.allreduce(val, op=comm.sum)
    avg_val = avg_val / comm.size
    return avg_val

def metric_average_ccl(val, size):
    avg_val = dist.all_reduce(val, op=dist.ReduceOp.SUM)
    avg_val = avg_val/size
    return avg_val

### Compute the correlation coefficient between predicted and target outputs
def comp_corrCoeff(output_tensor, target_tensor):
    target = target_tensor.numpy()
    target = np.ndarray.flatten(target)
    output = output_tensor.detach().numpy()
    output = np.ndarray.flatten(output)
    corrCoeff = np.corrcoef([output,target])
    return corrCoeff


### Count the number of trainable parameters in a model
def count_weights(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params



