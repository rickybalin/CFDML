##### 
##### This script contains general utilities that can be useful
##### to many training applications and driver scripts 
#####

from os.path import exists
import sys
import torch
import numpy as np

### MPI Communicator class
class MPI_COMM:
    def __init__(self):
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
        self.rankl = self.rank % cfg.run_args.mlprocs_pn
        self.sum = MPI.SUM
        self.minloc = MPI.MINLOC
        self.maxloc = MPI.MAXLOC
        if print_hello:
            print(f"Hello from MPI rank {self.rank}/{self.size}")
            sys.stdout.flush()

### Horovod Communicator class
class HVD_COMM:
    def __init__(self):
        self.rank = None
        self.size = None
        self.rankl = None

    def init(self, print_hello=False):
        import horovod.torch as hvd
        hvd.init()
        self.rank = hvd.rank()
        self.size = hvd.size()
        self.rankl = hvd.local_rank()
        if print_hello:
            print(f"Hello from HVD rank {self.rank}/{self.size}")
            sys.stdout.flush()

### Compute the average of a quantity across all ranks
def metric_average(comm, val):
    avg_val = comm.comm.allreduce(val, op=comm.sum)
    avg_val = avg_val / comm.size
    return avg_val


### Compute the correlation coefficient between predicted and target outputs
def comp_corrCoeff(output_tensor, target_tensor):
    target = target_tensor.numpy()
    target = np.ndarray.flatten(target)
    output = output_tensor.detach().numpy()
    output = np.ndarray.flatten(output)
    corrCoeff = np.corrcoef([output,target])
    return corrCoeff


### Set the model weights and biases to fixed value for reproducibility
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1: 
        # apply a fixed value to the weights and a set bias=0
        m.weight.data.fill_(0.5)
        m.bias.data.fill_(0)

### Load training data from file
def load_data(cfg, rng):
    if (cfg.train.data_path == "synthetic"):
        samples = 10 * cfg.train.mini_batch
        if (cfg.train.model == 'sgs'):
            data = np.float32(rng.normal(size=(samples,12)))
            mesh = None
        elif ("qcnn" in cfg.train.model):
            N = 32
            data = np.float32(rng.normal(size=(samples,cfg.train.channels,N**3)))
            mesh = np.zeros((N**3,3), dtype=np.float32)
            for i in range(N):
                x = 0. + 1. * (i - 1) / (N - 1)
                for j in range(N):
                    y = 0. + 1. * (j - 1) / (N - 1)
                    for k in range(N):
                        z = 0. + 1. * (k - 1) / (N - 1)
                        ind = i*N**2 + j*N +k
                        mesh[ind,0] = x
                        mesh[ind,1] = y
                        mesh[ind,2] = z
    else: 
        extension = cfg.train.data_path.split(".")[-1]
        if "npy" in extension:
            data = np.float32(np.load(cfg.train.data_path))
        if ("qcnn" in cfg.train.model):
            extension = mesh_file.split(".")[-1]
            if "npy" in extension:
                mesh = np.float32(np.load(mesh_file))
        else:
            mesh = None
    return data, mesh


