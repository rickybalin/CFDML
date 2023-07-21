#####
##### This script contains training loops, validation loops, and testing loops
##### used during online training from simulation data to be called from the 
##### training driver to assist in learning and evaluation model performance.
#####
import sys
from datetime import datetime
from time import sleep,perf_counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    import horovod as hvd
except:
    pass

from utils import metric_average
from datasets import MiniBatchDataset, KeyDataset


### Train the model
def online_train(comm, model, train_sampler, train_tensor_loader, optimizer, epoch,
                 t_data, client, cfg):

    model.train()
    running_loss = torch.tensor([0.0], device=torch.device(cfg.device))
    train_sampler.set_epoch(epoch)

    # Loop over batches, which in this case are the tensors to grab from database
    for tensor_idx, tensor_keys in enumerate(train_tensor_loader):
        # Grab data from database
        if (cfg.logging=='debug'):
            print(f'[{comm.Get_rank()}]: Grabbing tensors with key {tensor_keys}')
            sys.stdout.flush()
        rtime = perf_counter()
        if (cfg.model=='sgs'):
            concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)
        elif ('qcnn' in cfg.model):
            concat_tensor = torch.stack([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)
        rtime = perf_counter() - rtime
        t_data.t_getBatch = t_data.t_getBatch + rtime
        t_data.i_getBatch = t_data.i_getBatch + 1
        fact = float(1.0/t_data.i_getBatch)
        t_data.t_AveGetBatch = fact*rtime + (1.0-fact)*t_data.t_AveGetBatch

        # Create mini-batch dataset and loader
        mbdata = MiniBatchDataset(concat_tensor)
        train_loader = torch.utils.data.DataLoader(mbdata, shuffle=True,
                                                   batch_size=cfg.mini_batch)
        # Loop over mini-batches
        for batch_idx, dbdata in enumerate(train_loader):
            # Offload data
            if (cfg.device != 'cpu'):
               dbdata = dbdata.to(cfg.device)

            # Perform forward and backward passes
            rtime = perf_counter()
            optimizer.zero_grad()
            loss = model.module.training_step(dbdata)
            loss.backward()
            optimizer.step()
            rtime = perf_counter() - rtime
            t_data.t_compMiniBatch = t_data.t_compMiniBatch + rtime
            t_data.i_compMiniBatch = t_data.i_compMiniBatch + 1 
            fact = float(1.0/t_data.i_compMiniBatch)
            t_data.t_AveCompMiniBatch = fact*rtime + (1.0-fact)*t_data.t_AveCompMiniBatch            

            # Update running loss
            running_loss += loss

            # Print data for some ranks only
            if (comm.Get_rank()%20==0 and (batch_idx+1)%50==0):
                print(f'{comm.Get_rank()}: Train Epoch: {epoch} | ' + \
                      f'[{tensor_idx+1}/{len(train_tensor_loader)}] | ' + \
                      f'[{batch_idx+1}/{len(train_loader)}] | ' + \
                      f'Loss: {loss.item():>8e}')
                      #f'Loss: {loss.item():>8e} | corrCoeff: {corrCoeff[0,1]:>8e}')
                sys.stdout.flush()

    # Accumulate loss
    running_loss = running_loss.item() / len(train_tensor_loader) / len(train_loader)
    loss_avg = metric_average(comm, running_loss, comm.Get_size())
    if comm.Get_rank() == 0: 
        print(f"Training set: | Epoch: {epoch} | Average loss: {loss_avg:>8e} \n")
        sys.stdout.flush()

    return loss_avg, t_data

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Validate the model
def online_validate(comm, model, val_sampler, val_tensor_loader, epoch, mini_batch, 
                    client, cfg):

    model.eval()
    running_acc = torch.tensor([0.0], device=torch.device(cfg.device))
    running_loss = torch.tensor([0.0], device=torch.device(cfg.device))

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for tensor_idx, tensor_keys in enumerate(val_tensor_loader):
            # Grab data from database
            if (cfg.logging=='debug'):
                print(f'[{comm.Get_rank()}]: Grabbing tensors with key {tensor_keys}')
                sys.stdout.flush()
            if (cfg.model=='sgs'):
                concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)
            elif ('qcnn' in cfg.model):
                concat_tensor = torch.stack([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)

            # Create mini-batch dataset and loader
            mbdata = MiniBatchDataset(concat_tensor)
            val_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=mini_batch)
    
            # Loop over mini-batches
            for batch_idx, dbdata in enumerate(val_loader):
                # Offload data
                if (cfg.device != 'cpu'):
                    dbdata = dbdata.to(cfg.device)

                # Perform forward pass
                acc, loss = model.module.validation_step(dbdata, return_loss=True)
                running_acc += acc
                running_loss += loss
                
                # Print data for some ranks only
                if (comm.Get_rank()%20==0 and (batch_idx+1)%50==0):
                    print(f'{comm.Get_rank()}: Validation Epoch: {epoch} | ' + \
                          f'[{tensor_idx+1}/{len(val_tensor_loader)}] | ' + \
                          f'[{batch_idx+1}/{len(val_loader)}] | ' + \
                          f'Accuracy: {acc.item():>8e} | Loss {loss.item():>8e}')
                    sys.stdout.flush()

    # Accumulate accuracy measures
    running_acc = running_acc.item() / len(val_tensor_loader) / len(val_loader)
    acc_avg = metric_average(comm, running_acc, comm.Get_size())
    running_loss = running_loss.item() / len(val_tensor_loader) / len(val_loader)
    loss_avg = metric_average(comm, running_loss, comm.Get_size())
    if comm.Get_rank() == 0:
        print(f"Validation set: | Epoch: {epoch} | Average accuracy: {acc_avg:>8e} | Average Loss: {loss_avg:>8e}")
        sys.stdout.flush()

    return acc_avg, loss_avg

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Test the model
def oneline_test(comm, model, test_sampler, test_tensor_loader, mini_batch, 
                 client, cfg):

    model.eval()
    running_acc = torch.tensor([0.0], device=torch.device(cfg.device))
    running_loss = torch.tensor([0.0], device=torch.device(cfg.device))

    # Loop over batches, which in this case are the tensors to grab from database
    with torch.no_grad():
        for tensor_idx, tensor_keys in enumerate(test_tensor_loader):
            # Grab data from database
            if (cfg.logging=='debug'):
                print(f'[{comm.Get_rank()}]: Grabbing tensors with key {tensor_keys}')
                sys.stdout.flush()
            if (cfg.model=='sgs'):
                concat_tensor = torch.cat([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)
            elif ('qcnn' in cfg.model):
                concat_tensor = torch.stack([torch.from_numpy(client.get_tensor(key).astype('float32')) \
                             for key in tensor_keys], dim=0)

            # Create mini-batch dataset and loader
            mbdata = MiniBatchDataset(concat_tensor)
            test_loader = torch.utils.data.DataLoader(mbdata, shuffle=True, batch_size=mini_batch)
    
            # Loop over mini-batches
            for batch_idx, dbdata in enumerate(test_loader):
                # Offload data
                if (cfg.device != 'cpu'):
                    dbdata = dbdata.to(cfg.device)

                # Perform forward pass
                acc, loss = model.module.test_step(dbdata, return_loss=True)
                running_acc += acc
                running_loss += loss

                # Print data for some ranks only
                if (comm.Get_rank()%20==0 and (batch_idx+1)%50==0):
                    print(f'{comm.Get_rank()}: Testing | ' + \
                          f'[{tensor_idx+1}/{len(test_tensor_loader)}] | ' + \
                          f'[{batch_idx+1}/{len(test_loader)}] | ' + \
                          f'Accuracy: {acc.item():>8e} | Loss {loss.item():>8e}')

    # Accumulate accuracy measures
    running_acc = running_acc.item() / len(test_tensor_loader) / len(test_loader)
    acc_avg = metric_average(comm, running_acc, comm.Get_size())
    running_loss = running_loss.item() / len(test_tensor_loader) / len(test_loader)
    loss_avg = metric_average(comm, running_loss, comm.Get_size())
    if comm.Get_rank() == 0:
        print(f"Testing set: Average accuracy: {acc_avg:>8e} | Average Loss: {loss_avg:>8e}")

    if (cfg.model=='sgs'):
        testData = dbdata[:, model.module.ndOut:]
    else:
        testData = dbdata

    return testData, acc_avg, loss_avg

### ================================================ ###
### ================================================ ###
### ================================================ ###

### Main online training loop driver
def onlineTrainLoop(cfg, comm, client, t_data, model):
    # Setup and variable initialization
    istep = -1 # initialize the simulation step number to 0
    iepoch = 1 # epoch number
    rerun_check = 1 # 0 means quit training
    rank_list = np.arange(0,client.num_db_tensors*client.nfilters,dtype=int)
    num_val_tensors = int(client.num_db_tensors*client.nfilters*cfg.validation_split)
    num_train_tensors = client.num_db_tensors*client.nfilters - num_val_tensors
    if (num_val_tensors==0 and cfg.validation_split>0):
        num_val_tensors += 1
        num_train_tensors -= 1

    # Initialize optimizer
    if (cfg.optimizer == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate*comm.size)
    else:
        print("ERROR: only Adam optimizer implemented at the moment")
    if (cfg.distributed=='horovod'):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer,
                              named_parameters=model.named_parameters(),op=hvd.mpi_ops.Sum)

    # While loop that checks when training data is available on database
    if (comm.rank == 0):
        print("\nStarting training loop ... \n")
        sys.stdout.flush()
    while True:
        # Check to see if simulation says time to quit, if so break loop
        if (client.client.poll_tensor("check-run",0,1)):
            rtime = perf_counter()
            tmp = client.client.get_tensor('check-run')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1
            if (tmp[0] < 0.5):
                if (rank == 0):
                    print("Simulation says time to quit ... \n")
                    sys.stdout.flush()
                iTest = False
                rerun_check = 0
                break

        # check to see if the time step number has been sent to database, if not cycle
        if (client.client.poll_tensor("step",0,1)):
            rtime = perf_counter()
            tmp = client.client.get_tensor('step')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1
        else:
            continue

        # new data is available in database
        if (istep != tmp[0]): 
            istep = tmp[0]
            if (comm.rank == 0):
                print("\nNew training data was sent to the DB ...")
                print(f"Working with time step {istep} \n")
                sys.stdout.flush()
       
        # Create training and validation Datasets based on list of simulation ranks
        rng.shuffle(rank_list)
        train_tensors = rank_list[:num_train_tensors]
        val_tensors = rank_list[num_train_tensors:]
        if (cfg.model=="sgs" and client.nfilters>1):
            train_dataset = PhastaKeyMFDataset(train_tensors,client.num_db_tensors,
                                               client.head_rank,client.filters)
            val_dataset = PhastaKeyMFDataset(val_tensors,client.num_db_tensors,
                                             client.head_rank,client.filters)
        else:
            train_dataset = KeyDataset(train_tensors,client.head_rank,istep,client.dataOverWr)
            val_dataset = KeyDataset(val_tensors,client.head_rank,istep,client.dataOverWr)
        
        # Use DistributedSampler to partition the training and validation data across ranks
        if (cfg.online.db_launch=="colocated"):
            replicas = cfg.ppn
            rank_arg = comm.rankl
        else:
            replicas = comm.size
            rank_arg = comm.rank
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=replicas, rank=rank_arg, drop_last=False)
        train_tensor_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=client.tensor_batch, sampler=train_sampler, shuffle=False)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset, num_replicas=replicas, rank=rank_arg, drop_last=False)
        val_tensor_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=client.tensor_batch, sampler=val_sampler, shuffle=False)
        
        # Print epoch number
        if (comm.rank == 0):
            print(f"\n Epoch {iepoch} of {cfg.epochs}")
            print("-------------------------------")
            print(datetime.now())
            sys.stdout.flush()
        
        # Call training function
        rtime = perf_counter() 
        global_loss, t_data = online_train(comm, model, train_sampler, 
                                           train_tensor_loader, optimizer, 
                                           iepoch, 
                                           t_data, client, cfg)
        rtime = perf_counter() - rtime
        if (iepoch>1):
            t_data.t_train = t_data.t_train + rtime
            t_data.i_train = t_data.i_train + 1

        # Call validation function
        acc_avg, corr_avg = online_validate(comm, model, val_sampler, val_tensor_loader, 
                                            iepoch, cfg.mini_batch, client, cfg)
        
        # Check if tolerance on loss is satisfied
        if (global_loss <= cfg.tolerance):
            if (comm.rank == 0):
                print("\nConvergence tolerance met. Stopping training loop. \n")
            iTest = True
            break
        
        # Check if max number of epochs is reached
        if (iepoch >= cfg.epochs):
            if (comm.rank == 0):
                print("\nMax number of epochs reached. Stopping training loop. \n")
            iTest = True
            break

        iepoch = iepoch + 1 


    # Perform testing on a new snapshot
    if (iTest):
        if (comm.rank==0):
            print("\nTesting model\n-------------------------------")
 
        # Wait for new data to be sent to DB
        while True:
            if (client.poll_tensor("step",0,1)):
                tmp = client.client.get_tensor('step')
            
            if (istep != tmp[0]):
                istep = tmp[0]
                if (comm.rank == 0):
                    print(f"Working with time step {istep} \n")
                break

        # Create dataset, samples and loader for the test data
        test_dataset = KeyDataset(rank_list,client.head_rank,istep,client.dataOverWr)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset, num_replicas=replicas, rank=rank_arg, drop_last=False)
        test_tensor_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=client.tensor_batch, sampler=test_sampler, shuffle=False)

        # Call testing function
        testData, acc_avg, corr_avg = online_test(comm, model, test_sampler, test_tensor_loader,
                                 cfg.mini_batch, client, cfg)

    # Tell simulation to quit
    if (comm.rank==0 and rerun_check!=0):
        print("Telling simulation to quit ... \n")
        arrMLrun = np.zeros(2)
        client.client.put_tensor("check-run",arrMLrun)
 
    return model, testData



