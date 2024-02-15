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
    from torch.cuda.amp import autocast, GradScaler
    from torch.xpu.amp import autocast, GradScaler
except:
    pass
try:
    import horovod.torch as hvd
except:
    pass

from utils import metric_average
from datasets import OfflineDataset

### Train the model
def offline_train(comm, model, train_loader, optimizer, scaler, mixed_dtype, 
                  epoch, t_data, cfg):
    model.train()
    num_batches = len(train_loader)
    running_loss = torch.tensor([0.0],device=torch.device(cfg.device))

    # Loop over mini-batches
    for batch_idx, data in enumerate(train_loader):
            # Offload batch data
            if (cfg.device != 'cpu'):
               data = data.to(cfg.device)

            # Perform forward and backward passes
            rtime = perf_counter()
            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision, dtype=mixed_dtype):
                if (cfg.distributed=='horovod'):
                    loss = model.training_step(data)
                elif (cfg.distributed=='ddp'):
                    loss = model.module.training_step(data)
            if (cfg.mixed_precision):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            rtime = perf_counter() - rtime
            if (epoch>0):
                t_data.t_compMiniBatch = t_data.t_compMiniBatch + rtime
                t_data.i_compMiniBatch = t_data.i_compMiniBatch + 1 
                fact = float(1.0/t_data.i_compMiniBatch)
                t_data.t_AveCompMiniBatch = fact*rtime + (1.0-fact)*t_data.t_AveCompMiniBatch            

            # Update running loss
            running_loss += loss

            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank==0 and (batch_idx)%10==0):
                print(f'{comm.rank}: Train Epoch: {epoch+1} | ' + \
                      f'[{batch_idx+1}/{num_batches}] | ' + \
                      f'Loss: {loss.item():>8e}')
                sys.stdout.flush()

    # Accumulate loss
    running_loss = running_loss.item() / num_batches
    print(f"[{comm.rank}]: running loss = {running_loss:>8e}")
    sys.stdout.flush()
    comm.comm.Barrier()
    loss_avg = metric_average(comm, running_loss)
    if comm.rank == 0: 
        print(f"Training set: | Epoch: {epoch+1} | Average loss: {loss_avg:>8e}")
        sys.stdout.flush()

    return loss_avg, t_data

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Validate the model
def offline_validate(comm, model, val_loader, mixed_dtype, epoch, cfg):

    model.eval()
    num_batches = len(val_loader)
    running_acc = torch.tensor([0.0],device=torch.device(cfg.device))
    running_loss = torch.tensor([0.0],device=torch.device(cfg.device))

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Offload batch data
            if (cfg.device != 'cpu'):
                data = data.to(cfg.device)

            # Perform forward pass
            with autocast(enabled=cfg.mixed_precision, dtype=mixed_dtype):
                if (cfg.distributed=='horovod'):
                    acc, loss = model.validation_step(data)
                elif (cfg.distributed=='ddp'):
                    acc, loss = model.module.validation_step(data)
            running_acc += acc
            running_loss += loss
                
            # Print data for some ranks only
            if (cfg.logging=='debug' and comm.rank==0 and (batch_idx)%50==0):
                print(f'{comm.rank}: Validation Epoch: {epoch+1} | ' + \
                        f'[{batch_idx+1}/{num_batches}] | ' + \
                        f'Accuracy: {acc.item():>8e} | Loss {loss.item():>8e}')
                sys.stdout.flush()

    # Accumulate accuracy measures
    running_acc = running_acc.item() / num_batches
    acc_avg = metric_average(comm, running_acc)
    running_loss = running_loss.item() / num_batches
    loss_avg = metric_average(comm, running_loss)
    if comm.rank == 0:
        print(f"Validation set: | Epoch: {epoch+1} | Average accuracy: {acc_avg:>8e} | Average Loss: {loss_avg:>8e}")
        sys.stdout.flush()

    if (cfg.model=='sgs'):
        valData = data[:,6:]
    else:
        valData = data

    return acc_avg, loss_avg, valData

        

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Main online training loop driver
def offlineTrainLoop(cfg, comm, t_data, model, data):
    # Set precision of model and data
    if (cfg.precision == "fp32" or cfg.precision == "tf32"):
        model.float()
        data = torch.tensor(data, dtype=torch.float32)
    elif (cfg.precision == "fp64"):
        model.double()
        data = torch.tensor(data, dtype=torch.float64)
    elif (cfg.precision == "fp16"):
        model.half()
        data = torch.tensor(data, dtype=torch.float16)
    elif (cfg.precision == "bf16"):
        model.bfloat16()
        data = torch.tensor(data, dtype=torch.bfloat16)
    if (cfg.mixed_precision):
        scaler = GradScaler(enabled=True)
        if (cfg.device == "cuda"):
            mixed_dtype = torch.float16
        elif (cfg.device == "xpu"):
            mixed_dtype = torch.bfloat16
    else:
        scaler = None
        mixed_dtype = None
    
    # Offload entire data (this was actually slower...)
    #if (cfg.device != 'cpu'):
    #    data = data.to(cfg.device)

    # Split data and create datasets
    samples = data.shape[0]
    nVal = m.floor(samples*cfg.validation_split)
    nTrain = samples-nVal
    if (nVal==0 and cfg.validation_split>0):
        if (comm.rank==0): print("Insufficient number of samples for validation -- skipping it")
    dataset = OfflineDataset(data)
    trainDataset, valDataset = random_split(dataset, [nTrain, nVal])

    # Data loader
    # Try:
    # - pin_memory=True - should be faster for GPU training
    # - num_workers > 1 - enables multi-process data loading 
    # - prefetch_factor >1 - enables pre-fetching of data
    if (cfg.data_path == "synthetic"):
        train_sampler = None
        val_sampler = None
        train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                      shuffle=True, drop_last=True) 
        val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                    drop_last=True)
    else:
        # Each rank has loaded all the training data, so restrict data loader to a subset of dataset
        train_sampler = DistributedSampler(trainDataset, num_replicas=comm.size, rank=comm.rank,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(valDataset, num_replicas=comm.size, rank=comm.rank,
                                         drop_last=True)  
        train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                  sampler=train_sampler)
        val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                sampler=val_sampler)

    # Initialize optimizer
    if (cfg.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate*comm.size)
    elif (cfg.optimizer == "RAdam"):
        optimizer = optim.RAdam(model.parameters(), lr=cfg.learning_rate*comm.size)
    else:
        print("ERROR: optimizer not implemented at the moment")
    if (cfg.scheduler == "Plateau"):
        if (comm.rank==0): print("Applying plateau scheduler\n")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    if (cfg.distributed=='horovod'):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=model.named_parameters(),
                                             op=hvd.mpi_ops.Sum,
                                             num_groups=1)

    # Loop over epochs
    for ep in range(cfg.epochs):
        tic_l = perf_counter()
        if (comm.rank == 0):
            print(f"\n Epoch {ep+1} of {cfg.epochs}")
            print("-------------------------------")
            sys.stdout.flush()

        # Train
        if train_sampler:
            train_sampler.set_epoch(ep)
        tic_t = perf_counter()
        global_loss, t_data = offline_train(comm, model, train_dataloader, 
                                            optimizer, scaler, mixed_dtype,
                                            ep, t_data, cfg)
        toc_t = perf_counter()

        # Validate
        if (nVal==0):
            global_val_loss = global_loss
            if (cfg.model=='sgs'):
                valData = data[cfg.mini_batch,6:].to(cfg.device)
            else:
                valData = data.to(cfg.device)
        else:
            tic_v = perf_counter()
            global_acc, global_val_loss, valData = offline_validate(comm, model, 
                                                                val_dataloader, 
                                                                mixed_dtype, ep, cfg)
            toc_v = perf_counter()
            if (ep>0):
                t_data.t_val = t_data.t_val + (toc_v - tic_v)
                t_data.tp_val = t_data.tp_val + nVal/(toc_v - tic_v)
                t_data.i_val = t_data.i_val + 1

        # Apply scheduler
        if (cfg.scheduler == "Plateau"):
            scheduler.step(global_val_loss)

        toc_l = perf_counter()
        if (ep>0):
            t_data.t_tot = t_data.t_tot + (toc_l - tic_l)
            t_data.t_train = t_data.t_train + (toc_t - tic_t)
            t_data.tp_train = t_data.tp_train + nTrain/(toc_t - tic_t)
            t_data.i_train = t_data.i_train + 1
        
        # Check if tolerance on loss is satisfied
        if (global_val_loss <= cfg.tolerance):
            if (comm.rank == 0):
                print("\nConvergence tolerance met. Stopping training loop. \n")
            break
        
        # Check if max number of epochs is reached
        if (ep >= cfg.epochs):
            if (comm.rank == 0):
                print("\nMax number of epochs reached. Stopping training loop. \n")
            break

    return model, valData
