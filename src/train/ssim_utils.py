##### 
##### This script contains SmartSim utilities that can be called from
##### other parts of the code 
#####

import sys
from os import environ
from os.path import exists
from time import perf_counter

class SmartRedisClient:
    def __init__(self):
        self.client = None
        self.npts = None
        self.ndTot = None
        self.ndIn = None
        self.ndOut = None
        self.num_tot_tensors = None
        self.num_db_tensors = None
        self.head_rank = None
        self.tensor_batch = None
        self.nfilters = 1
        self.dataOverWr = None
        self.rank = None
        self.size = None
        self.max_wait_time = 20.

    # Initializa client
    def init(self, cfg, comm, t_data):
        """Initialize the SmartRedis client
        """
        self.rank = comm.rank
        self.size = comm.size
        try:
            from smartredis import Client
        except ModuleNotFoundError as err:
            if self.rank==0: print(err)

        # Read the address of the co-located database first
        if (cfg.online.db_launch=='colocated'):
            #prefix = f'{cfg.online.simprocs}-procs_case/'
            #address = self.read_SSDB(prefix, comm)
            address = environ['SSDB']
        else:
            address = None

        # Initialize Redis clients on each rank #####
        if (self.rank == 0):
            print("\nInitializing Python clients ...", flush=True)
        if (cfg.online.db_nodes==1):
            rtime = perf_counter()
            sys.stdout.flush()
            self.client = Client(address=address, cluster=False)
            rtime = perf_counter() - rtime
            t_data.t_init = t_data.t_init + rtime
            t_data.i_init = t_data.i_init + 1
        else:
            rtime = perf_counter()
            client = Client(address=address, cluster=True)
            rtime = perf_counter() - rtime
            t_data.t_init = t_data.t_init + rtime
            t_data.i_init = t_data.i_init + 1
        comm.comm.Barrier()
        if (self.rank == 0):
            print("All done\n", flush=True)

    # Read the address of the co-located database
    def read_SSDB(self, prefix, comm):
        SSDB_file = prefix + f'SSDB_{comm.name}.dat'
        c = 0 
        while True:
            if (exists(SSDB_file)):
                f = open(SSDB_file, "r")
                SSDB = f.read()
                f.close()
                if (SSDB == ''):
                    continue
                else:
                    print(f'[{comm.rank}]: read SSDB={SSDB}')
                    sys.stdout.flush()
                    break
            else:
                if (c==0):
                    print(f'[{comm.rank}]: WARNING, looked for {SSDB_file} but did not find it')
                    sys.stdout.flush()
                c+=1
                continue
        comm.comm.Barrier()
        if ('\n' in SSDB):
            SSDB = SSDB.replace('\n', '') 
        return SSDB

    # Read the size information from DB
    def read_sizeInfo(self, cfg, t_data):
        if (self.rank == 0):
            print("\nGetting setup info from DB ...", flush=True)
        while True:
            if (self.client.poll_tensor("sizeInfo",0,1)):
                rtime = perf_counter()
                dataSizeInfo = self.client.get_tensor('sizeInfo')
                rtime = perf_counter() - rtime
                t_data.t_meta = t_data.t_meta + rtime
                t_data.i_meta = t_data.i_meta + 1
                break
        self.npts = dataSizeInfo[0]
        self.ndTot = dataSizeInfo[1]
        self.ndIn = dataSizeInfo[2]
        self.ndOut = self.ndTot - self.ndIn
        self.num_tot_tensors = dataSizeInfo[3]
        self.num_db_tensors = dataSizeInfo[4]
        self.head_rank = dataSizeInfo[5]
        
        if (self.rank == 0):
            print(f"Samples per simulation tensor: {self.npts}")
            print(f"Model input features: {self.ndIn}")
            print(f"Model output targets: {self.ndOut}")
            print(f"Total tensors in all DB: {self.num_tot_tensors}")
            print(f"Tensors in local DB: {self.num_db_tensors}")
            sys.stdout.flush()

    # Read the flag determining if data is overwritten in DB
    def read_overwrite(self, t_data):
        while True:
            if (self.client.poll_tensor("tensor-ow",0,1)):
                rtime = perf_counter()
                tmp = self.client.get_tensor('tensor-ow')
                rtime = perf_counter() - rtime
                t_data.t_meta = t_data.t_meta + rtime
                t_data.i_meta = t_data.i_meta + 1 
                break
        self.dataOverWr = tmp[0]
        if (self.rank==0):
            if (self.dataOverWr>0.5): 
                print("Training data is overwritten in DB")
            else:
                print("Training data is accumulated in DB")
            sys.stdout.flush()

    # Read the flag determining how many filterwidths to train on
    def read_num_filters(self, model_name, t_data):
        if (model_name == "sgs"):
            elapsed_time = 0.
            tic = perf_counter()
            while elapsed_time<self.max_wait_time:
                if (self.client.poll_tensor("num_filter_widths",0,1)):
                    rtime = perf_counter()
                    self.nfilters = self.client.get_tensor('num_filter_widths')[0]
                    rtime = perf_counter() - rtime
                    t_data.t_meta = t_data.t_meta + rtime
                    t_data.i_meta = t_data.i_meta + 1 
                    break
                elapsed_time = perf_counter() - tic
            if (self.rank==0):
                print(f"Using {self.nfilters} filters widths for training data", flush=True)
   
    # Calculate the filter batch size
    def get_batch(self, cfg):     
        max_batch_size = int(self.num_db_tensors*self.nfilters/cfg.ppn)
        if (not cfg.online.global_shuffling):
            self.tensor_batch = max_batch_size
        else:
            if (cfg.online.batch==0 or cfg.online.batch>max_batch_size):
                self.tensor_batch = max_batch_size
            else:
                self.tensor_batch = cfg.online.batch
        if (self.rank==0):
            print(f"Grabbing {self.tensor_batch} simulation tensors per batch\n")
    
