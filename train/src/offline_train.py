#####
##### This script contains training loops, validation loops, and testing loops
##### used during online training from simulation data to be called from the 
##### training driver to assist in learning and evaluation model performance.
#####
import sys
from time import perf_counter
import numpy as np
import math as m

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    import horovod as hvd
except:
    pass

from utils import metric_average
from datasets import OfflineDataset

### Train the model
def offline_train(comm, model, train_loader, optimizer, epoch, t_data, cfg):
    model.train()
    num_batches = len(train_loader)
    running_loss = 0

    # Loop over mini-batches
    for batch_idx, data in enumerate(train_loader):
            # Offload batch data
            if (cfg.train.device != 'cpu'):
               data = data.to(cfg.train.device)

            # Perform forward and backward passes
            rtime = perf_counter()
            optimizer.zero_grad()
            if (cfg.train.distributed=='horovod'):
                loss = model.training_step(data)
            elif (cfg.train.distributed=='ddp'):
                loss = model.module.training_step(data)
            loss.backward()
            optimizer.step()
            rtime = perf_counter() - rtime
            if (epoch>1):
                t_data.t_compMiniBatch = t_data.t_compMiniBatch + rtime
                t_data.i_compMiniBatch = t_data.i_compMiniBatch + 1 
                fact = float(1.0/t_data.i_compMiniBatch)
                t_data.t_AveCompMiniBatch = fact*rtime + (1.0-fact)*t_data.t_AveCompMiniBatch            

            # Update running loss
            running_loss += loss.item()

            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank%20==0 and (batch_idx)%50==0):
                print(f'{comm.rank}: Train Epoch: {epoch+1} | ' + \
                      f'[{batch_idx+1}/{num_batches}] | ' + \
                      f'Loss: {loss.item():>8e}')
                sys.stdout.flush()

    # Accumulate loss
    running_loss = running_loss / num_batches
    loss_avg = metric_average(comm, running_loss)
    if comm.rank == 0: 
        print(f"Training set: | Epoch: {epoch+1} | Average loss: {loss_avg:>8e}")
        sys.stdout.flush()

    return loss_avg, t_data

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Validate the model
def offline_validate(comm, model, val_loader, epoch, cfg):

    model.eval()
    num_batches = len(val_loader)
    running_acc = 0.0
    running_loss = 0.0

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Offload batch data
            if (cfg.train.device != 'cpu'):
                data = data.to(cfg.train.device)

            # Perform forward pass
            if (cfg.train.distributed=='horovod'):
                acc, loss = model.validation_step(data, return_loss=True)
            elif (cfg.train.distributed=='ddp'):
                acc, loss = model.module.validation_step(data, return_loss=True)
            running_acc += acc.item()
            running_loss += loss.item()
                
            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank%20==0 and (batch_idx)%50==0):
                print(f'{comm.rank}: Validation Epoch: {epoch+1} | ' + \
                        f'[{batch_idx+1}/{num_batches}] | ' + \
                        f'Accuracy: {acc.item():>8e} | Loss {loss.item():>8e}')
                sys.stdout.flush()

    # Accumulate accuracy measures
    running_acc = running_acc / num_batches
    acc_avg = metric_average(comm, running_acc)
    running_loss = running_loss / num_batches
    loss_avg = metric_average(comm, running_loss)
    if comm.rank == 0:
        print(f"Validation set: | Epoch: {epoch+1} | Average accuracy: {acc_avg:>8e} | Average Loss: {loss_avg:>8e}")
        sys.stdout.flush()

    if (cfg.train.model=='sgs'):
        valData = data[:,6:]
    else:
        valData = data

    return acc_avg, loss_avg, valData

        

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Main online training loop driver
def offlineTrainLoop(cfg, comm, hvd_comm, t_data, model, data):
    # Import Horovod if needed here too
    if (cfg.train.distributed=='horovod'):
        import horovod.torch as hvd

    # Set precision of model and data
    if (cfg.train.precision == "fp32"):
        data = torch.tensor(data, dtype=torch.float32)
    elif (cfg.train.precision == "fp64"):
        model.double()
        data = torch.tensor(data, dtype=torch.float64)
    elif (cfg.train.precision == "bf16"):
        model.bfloat16()
        data = torch.tensor(data, dtype=torch.bfloat16)
    
    # Offload entire data (this was actually slower...)
    #if (cfg.train.device != 'cpu'):
    #    data = data.to(cfg.train.device)

    # Split data and create datasets
    samples = data.shape[0]
    nVal = m.floor(samples*cfg.train.validation_split)
    nTrain = samples-nVal
    dataset = OfflineDataset(data)
    trainDataset, valDataset = random_split(dataset, [nTrain, nVal])

    # Data parallel loader
    # Try:
    # - pin_memory=True - should be faster for GPU training
    # - num_workers > 1 - enables multi-process data loading 
    # - prefetch_factor >1 - enables pre-fetching of data
    if (cfg.train.data_path == "synthetic"):
        train_sampler = None
        val_sampler = None
        train_dataloader = DataLoader(trainDataset, batch_size=cfg.train.mini_batch, 
                                      shuffle=True, drop_last=True) #pin_memory=False, num_workers=0, prefetch_factor=None)
        val_dataloader = DataLoader(valDataset, batch_size=cfg.train.mini_batch, 
                                    drop_last=True)
    else:
        # Each rank has loaded all the training data, so restrict data loader to a subset of dataset
        train_sampler = DistributedSampler(trainDataset, num_replicas=comm.size, rank=comm.rank,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(valDataset, num_replicas=comm.size, rank=comm.rank,
                                         drop_last=True)  
        train_dataloader = DataLoader(trainDataset, batch_size=cfg.train.mini_batch, 
                                  sampler=train_sampler)
        val_dataloader = DataLoader(valDataset, batch_size=cfg.train.mini_batch, 
                                sampler=val_sampler)

    # Initialize optimizer
    if (cfg.train.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate*comm.size)
    else:
        print("ERROR: only Adam optimizer implemented at the moment")
    if (cfg.train.distributed=='horovod'):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                              named_parameters=model.named_parameters(),op=hvd.mpi_ops.Sum)

    # Loop over epochs
    for ep in range(cfg.train.epochs):
        tic_l = perf_counter()
        if (comm.rank == 0):
            print(f"\n Epoch {ep+1} of {cfg.train.epochs}")
            print("-------------------------------")
            sys.stdout.flush()

        # Train
        if train_sampler:
            train_sampler.set_epoch(ep)
        tic_t = perf_counter()
        global_loss, t_data = offline_train(comm, model, train_dataloader, 
                                            optimizer, ep, t_data, cfg)
        toc_t = perf_counter()
        if (ep>1):
            t_data.t_train = t_data.t_train + (toc_t - tic_t)
            t_data.tp_train = t_data.tp_train + nTrain/(toc_t - tic_t)
            t_data.i_train = t_data.i_train + 1

        # Validate
        tic_v = perf_counter()
        global_acc, global_val_loss, valData = offline_validate(comm, model, 
                                                                val_dataloader, ep, cfg)
        toc_v = perf_counter()
        if (ep>1):
            t_data.t_val = t_data.t_val + (toc_v - tic_v)
            t_data.tp_val = t_data.tp_val + nVal/(toc_v - tic_v)
            t_data.i_val = t_data.i_val + 1

        # Check if tolerance on loss is satisfied
        if (global_val_loss <= cfg.train.tolerance):
            if (comm.rank == 0):
                print("\nConvergence tolerance met. Stopping training loop. \n")
            break
        
        # Check if max number of epochs is reached
        if (ep >= cfg.train.epochs):
            if (comm.rank == 0):
                print("\nMax number of epochs reached. Stopping training loop. \n")
            break

        toc_l = perf_counter()
        if (ep>1):
            t_data.t_tot = t_data.t_tot + (toc_l - tic_l)

    return model, valData
