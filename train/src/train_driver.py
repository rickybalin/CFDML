# Import general libraries
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from time import perf_counter
import random

# Import ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import required functions
from online_train import onlineTrainLoop
from offline_train import offlineTrainLoop
import models
from time_prof import timeStats
import utils
import ssim_utils

## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    
    # Import and init MPI
    print_hello = True if cfg.logging=='debug' else False
    comm = utils.MPI_COMM()
    comm.init(cfg, print_hello=print_hello)

    # Intel imports
    try:
        import intel_extension_for_pytorch
    except ModuleNotFoundError as err:
        if comm.rank==0: print(err)
    try:
        import oneccl_bindings_for_pytorch
    except ModuleNotFoundError as err:
        if comm.rank==0: print(err)

    # Import Horovod and initialize
    hvd_comm = None
    if (cfg.train.distributed=='horovod'):
        #import horovod.torch as hvd
        hvd_comm = utils.HVD_COMM()
        hvd_comm.init(print_hello=print_hello)
    elif (cfg.train.distributed=='ddp'):
        import socket
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        os.environ['RANK'] = str(comm.rank)
        os.environ['WORLD_SIZE'] = str(comm.size)
        master_addr = socket.gethostname() if comm.rank == 0 else None
        master_addr = comm.comm.bcast(master_addr, root=0)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(2345)
        if (cfg.train.device=='cpu'):
            backend = 'gloo'
        elif (cfg.train.device=='cuda'):
            backend = 'nccl'
        elif (cfg.train.device=='xpu'):
            backend = 'ccl'
        dist.init_process_group(backend,
                                rank=int(comm.rank),
                                world_size=int(comm.size),
                                init_method='env://')

    # Set all seeds if need reproducibility
    if cfg.train.repeatability:
        random_seed = 123456789
        random.seed(a=random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.manual_seed(random_seed)
        if (cfg.train.device=='cuda' and torch.cuda.is_available()):
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True # only pick deterministic algorithms
            torch.backends.cudnn.benchmark = False # do not select fastest algorithms
    else:
        rng = np.random.default_rng()

    # Instantiate performance data class
    t_data = timeStats()

    # Initialize SmartRedis client and gather metadata
    if cfg.database.launch:
        # Import SmartSim libraries
        global Client
        from smartredis import Client
        client = ssim_utils.SmartRedisClient()
        client.init(cfg, comm, t_data)
        client.read_sizeInfo(cfg, comm, t_data)
        client.read_overwrite(comm, t_data)
        client.read_filters(cfg, t_data)
        mesh_nodes = client.read_mesh(cfg, comm, t_data)

    # Load data from file if not launching database
    if not cfg.database.launch:
         data, mesh_nodes = utils.load_data(cfg, rng)

    # Instantiate the NN model and optimizer 
    if (cfg.train.model=="sgs"):
        model = models.anisoSGS(inputDim=6, outputDim=6, numNeurons=20)
    elif ("qcnn" in cfg.train.model):
        mesh_nodes = torch.from_numpy(mesh_nodes)
        model = models.qcnn(comm.rank, mesh_nodes, cfg.train.qcnn_config, cfg.train.channels)

    # Set device to run on and offload model
    if (comm.rank == 0):
        print(f"\nRunning on device: {cfg.train.device} \n")
        sys.stdout.flush()
    device = torch.device(cfg.train.device)
    torch.set_num_threads(1)
    if (cfg.train.device == 'cuda'):
        if torch.cuda.is_available():
            torch.cuda.set_device(comm.rankl)
    elif (cfg.train.device=='xpu'):
        if torch.xpu.is_available():
            torch.xpu.set_device(comm.rankl)
    if (cfg.train.device != 'cpu'):
        model.to(device)
        if ("qcnn" in cfg.train.model):
            mesh_nodes = mesh_nodes.to(device)

    # Initializa DDP model
    if (cfg.train.distributed=='ddp'):
        model = DDP(model)

    # Train model
    if cfg.database.launch:
        model, testData = onlineTrainLoop(cfg, comm, hvd_comm, client, t_data, model)
    else:
        model, testData = offlineTrainLoop(cfg, comm, hvd_comm, t_data, model, data)

    # Save model to file before exiting
    if (cfg.train.distributed=='ddp'):
        model = model.module
        dist.destroy_process_group()
    if (comm.rank == 0):
        torch.save(model.state_dict(), f"{cfg.train.name}.pt", _use_new_zipfile_serialization=False)
        model.eval()
        if (cfg.train.model=="sgs"):
            module = torch.jit.trace(model, testData)
            torch.jit.save(module, f"{cfg.train.name}_jit.pt")
        elif ("qcnn" in cfg.train.model):
            encoder = models.qcnnEncoder(model.encoder, model.mesh)
            decoder = models.qcnnDecoder(model.decoder, model.mesh)
            dummy_latent = encoder(testData).detach()
            predicted = decoder(dummy_latent).detach()
            module_encode = torch.jit.trace(encoder, testData)
            torch.jit.save(module_encode, f"{cfg.train.name}_encoder_jit.pt")
            dummy_latent = module_encode(testData).detach()
            module_decode = torch.jit.trace(decoder, dummy_latent)
            torch.jit.save(module_decode, f"{cfg.train.name}_decoder_jit.pt")
            predicted = module_decode(dummy_latent).detach()

        print("")
        print("Saved model to disk\n")
        sys.stdout.flush()

    
    # Collect timing statistics
    if (comm.rank==0):
        print("\nTiming data:")
        sys.stdout.flush()
    t_data.printTimeData(cfg, comm)
 

    # Exit
    if (comm.rank == 0):
        print("Exiting ...")
        sys.stdout.flush()


## Run main
if __name__ == "__main__":
    main()
