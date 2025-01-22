# Import general libraries
import os
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from time import perf_counter
import random
import datetime
import socket

# Import ML libraries
import torch
import torch.distributed as dist

# Import required functions
from online_train import onlineTrainLoop
from offline_train import offlineTrainLoop
import models
from time_prof import timeStats
import utils
import ssim_utils

## Main function
@hydra.main(version_base=None, config_path="./conf", config_name="train_config")
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
        ONECCL = True
    except ModuleNotFoundError as err:
        ONECCL = False
        if comm.rank==0: print(err)

    # Set device to run on and offload model
    torch.set_num_threads(cfg.threads)
    if (cfg.device == 'cuda'):
        if torch.cuda.is_available():
            cuda_id = comm.rankl//cfg.ppd if torch.cuda.device_count()>1 else 0
            assert cuda_id>=0 and cuda_id<torch.cuda.device_count(), \
                   f"Assert failed: cuda_id={cuda_id} and {torch.cuda.device_count()} available devices"
            sys.stdout.flush()
            device = torch.device(cfg.device)
            torch.cuda.set_device(cuda_id)
        else:
            device = torch.device('cpu')
            print(f"[{comm.rank}]: no cuda devices available, cuda.device_count={torch.cuda.device_count()}",flush=True)
    elif (cfg.device=='xpu'):
        if torch.xpu.is_available():
            xpu_id = comm.rankl//cfg.ppd if torch.xpu.device_count()>1 else 0
            assert xpu_id>=0 and xpu_id<torch.xpu.device_count(), \
                   f"Assert failed: xpu_id={xpu_id} and {torch.xpu.device_count()} available devices"
            device = torch.device(cfg.device)
            torch.xpu.set_device(xpu_id)
        else:
            device = torch.device('cpu')
            print(f"[{comm.rank}]: no XPU devices available, xpu.device_count={torch.xpu.device_count()}", flush=True)
    if (comm.rank == 0):
        print(f"\nRunning on device: {cfg.device} \n", flush=True)

    # Initialize torch distributed
    os.environ['RANK'] = str(comm.rank)
    os.environ['WORLD_SIZE'] = str(comm.size)
    master_addr = socket.gethostname() if comm.rank == 0 else None
    master_addr = comm.comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (cfg.device=='cuda' and cfg.ppd==1): backend = 'nccl'
    elif (cfg.device=='cuda' and cfg.ppd>1): backend = 'gloo'
    elif (cfg.device=='xpu' or ONECCL): backend = 'ccl'
    elif (cfg.device=='cpu'): backend = 'gloo'
    dist.init_process_group(backend,
                            rank=int(comm.rank),
                            world_size=int(comm.size),
                            init_method='env://',
                            timeout=datetime.timedelta(seconds=600))

    # Set all seeds if need reproducibility
    if cfg.reproducibility:
        random_seed = 123456789
        random.seed(a=random_seed)
        np.random.seed(random_seed)
        rng = np.random.default_rng(seed=random_seed)
        torch.manual_seed(random_seed)
        if (cfg.device=='cuda' and torch.cuda.is_available()):
            torch.cuda.manual_seed_all(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True) # only pick deterministic algorithms
        elif (cfg.device=='xpu' and torch.xpu.is_available()):
            torch.xpu.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True) # only pick deterministic algorithms
    else:
        rng = np.random.default_rng()

    # Instantiate performance data class
    t_data = timeStats()

    # Initialize SmartRedis client and gather metadata
    client = None
    if cfg.online.db_launch:
        client = ssim_utils.SmartRedisClient()
        client.init(cfg, comm, t_data)
        client.read_sizeInfo(cfg, t_data)
        client.read_overwrite(t_data)
        client.read_num_filters(cfg.model, t_data)
        client.get_batch(cfg)

    # Instantiate the model and get the training data
    model, data = models.load_model(cfg, comm, client, rng, t_data)
    if (cfg.device != 'cpu'):
        model.to(device)
    
    # Train model
    if cfg.online.db_launch:
        model = onlineTrainLoop(cfg, comm, client, t_data, model)
    else:
        model, testData = offlineTrainLoop(cfg, comm, t_data, model, data)

    # Save model to file before exiting
    model = model.module
    dist.destroy_process_group()
    if (comm.rank == 0):
        model.eval()
        model.save_checkpoint(cfg.name, num_samples=cfg.mini_batch)
        print("")
        print("Saved model to disk\n")
        sys.stdout.flush()

    # Collect timing statistics
    if (t_data.i_train>0):
        if (comm.rank==0):
            print("\nTiming data (excluding first epoch):")
            sys.stdout.flush()
        t_data.printTimeData(cfg, comm)

    # Exit
    if (comm.rank == 0):
        print("Exiting ...")
        sys.stdout.flush()


## Run main
if __name__ == "__main__":
    main()
