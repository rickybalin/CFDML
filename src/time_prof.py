##### 
##### This script defines the class that is used to measure the performance
##### of the training and data transfer parts of the algorithm 
######

import sys
import numpy as np
import math
import torch

class timeStats:
    t_getBatch = 0.0 # local accumulated time spent grabbing training data for each batch
    i_getBatch = 0 # local number of times data is grabbed
    t_AveGetBatch = 0.0 # local average time spent grabbing training data for each batch
    t_train = 0.0 # local accumulated time spent in training function
    i_train = 0 # local number of times training function called
    t_init = 0.0 # local accumulated time spent initializing Redis clients
    i_init = 0 # local number of times initializing Redis clients
    t_meta = 0.0 # local accumulated time spent transfering metadata
    i_meta = 0 # local number of times metadata is transferred
    t_compMiniBatch = 0.0 # local accumulated time spent computing mini-batch
    i_compMiniBatch = 0 # local number of times computing mini-batch
    t_AveCompMiniBatch = 0.0 # local average time spent computing mini-batch

    # Compute the average across all processes of the local time data
    #def computeAvg(self, comm):
    #    t_getBatch_avg = comm.allreduce(self.t_getBatch,op=MPI.SUM)
    #    t_getBatch_avg = t_getBatch_avg / comm.Get_size()
    #    t_train_avg = comm.allreduce(self.t_train,op=MPI.SUM)
    #    t_train_avg = t_train_avg / comm.Get_size()
    #    return t_getBatch_avg, t_train_avg

    # Compute the standard deviation across all processes of the local time data
    #def computeStd(self, comm, t_getBatch_avg, t_train_avg):
    #    tmp = np.array((self.t_getBatch - t_getBatch_avg)**2)
    #    t_getBatch_std = comm.allreduce(tmp,op=MPI.SUM)
    #    t_getBatch_std = t_getBatch_std / comm.Get_size()
    #    t_getBatch_std = math.sqrt(t_getBatch_std)
    #    tmp = np.array((self.t_train - t_train_avg)**2)
    #    t_train_std = comm.allreduce(tmp,op=MPI.SUM)
    #    t_train_std = t_train_std / comm.Get_size()
    #    t_train_std = math.sqrt(t_train_std)
    #    return t_getBatch_std, t_train_std

    # Compute min, max, mean and standard deviation across all processes for a time measure
    def computeStats_f(self, comm, var):
        avg = comm.comm.allreduce(np.array(var),op=comm.sum)
        avg = avg / comm.size
        tmp = np.array((var - avg)**2) 
        std = comm.comm.allreduce(tmp,op=comm.sum)
        std = std / comm.size
        std = math.sqrt(std)
        min_loc = comm.comm.allreduce((var,comm.rank),op=comm.minloc)
        max_loc = comm.comm.allreduce((var,comm.rank),op=comm.maxloc)
        return avg, std, [min_loc[0],min_loc[1]], [max_loc[0],max_loc[1]]

    # Compute min, max, mean and standard deviation across all processes for a counter
    def computeStats_i(self, comm, var):
        avg = comm.comm.allreduce(np.array(var),op=comm.sum)
        avg = avg / comm.size
        tmp = np.array((var - avg)**2) 
        std = comm.comm.allreduce(tmp)
        std = std / comm.size
        std = math.sqrt(std)
        return avg, std

    # Print the timing data
    def printTimeData(self, comm):
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_init)
        if comm.rank==0:
            print(f"tSSIMInit : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_meta)
        if comm.rank==0:
            print(f"tSSIMMeta : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_train)
        if comm.rank==0:
            print(f"tTrain : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_compMiniBatch)
        if comm.rank==0:
            print(f"tcompMiniBatch : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_AveCompMiniBatch)
        if comm.rank==0:
            print(f"taveCompMiniBatch : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_getBatch)
        if comm.rank==0:
            print(f"tSSIMgetBatch : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        avg, std, min_arr, max_arr = self.computeStats_f(comm, self.t_AveGetBatch)
        if comm.rank==0:
            print(f"tSSIMaveGetBatch : min [{min_arr[0]:>8e},{min_arr[1]:>d}], max [{max_arr[0]:>8e},{max_arr[1]:>d}], avg [{avg:>8e},.], std [{std:>8e},.]")
        if comm.rank==0:
            print("")
        avg, std = self.computeStats_i(comm, self.i_init)
        if comm.rank==0:
            print(f"iSSIMInit : min [.,.], max [.,.], avg [{avg},.], std [{std},.]")
        avg, std = self.computeStats_i(comm, self.i_meta)
        if comm.rank==0:
            print(f"iSSIMMeta : min [.,.], max [.,.], avg [{avg},.], std [{std},.]")
        avg, std = self.computeStats_i(comm, self.i_train)
        if comm.rank==0:
            print(f"iTrain : min [.,.], max [.,.], avg [{avg},.], std [{std},.]")
        avg, std = self.computeStats_i(comm, self.i_compMiniBatch)
        if comm.rank==0:
            print(f"iCompMiniBatch : min [.,.], max [.,.], avg [{avg},.], std [{std},.]")
        avg, std = self.computeStats_i(comm, self.i_getBatch)
        if comm.rank==0:
            print(f"iSSIMgetBatch : min [.,.], max [.,.], avg [{avg},.], std [{std},.]")

