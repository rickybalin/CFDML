##### 
##### This script contains SmartSim utilities that can be called from
##### other parts of the code 
#####

import sys
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

    # Initializa client
    def init(self, cfg, comm, t_data):
        # Read the address of the co-located database first
        if (cfg.database.launch=='colocated'):
            prefix = f'{cfg.run_args.simprocs}-procs_case/'
            address = self.read_SSDB(prefix, comm)
        else:
            address = None
        #address = os.environ['SSDB']

        # Initialize Redis clients on each rank #####
        if (comm.rank == 0):
            print("\nInitializing Python clients ...")
            sys.stdout.flush()
        if (cfg.run_args.db_nodes==1):
            rtime = perf_counter()
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
        if (comm.rank == 0):
            print("All done\n")
            sys.stdout.flush()

    # Read the address of the co-located database
    def read_SSDB(prefix, comm):
        SSDB_file = prefix + f'SSDB_{comm.rank_name}.dat'
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
    def read_sizeInfo(self, cfg, comm, t_data):
        if (comm.rank == 0):
            print("\nGetting size info from DB ...")
            sys.stdout.flush()
        while True:
            if (self.client.client.poll_tensor("sizeInfo",0,1)):
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

        if (comm.rank == 0):
            print(f"Number of samples per Simulation tensor: {self.npts}")
            print(f"Number of tensors in local DB: {self.num_db_tensors}")
            print(f"Number of total tensors in all DB: {self.num_tot_tensors}")
            sys.stdout.flush()

        max_batch_size = int(self.num_db_tensors/cfg.run_args.mlprocs_pn)
        self.tensor_batch = cfg.train.batch
        if (cfg.train.batch==0 or self.tensor_batch>max_batch_size): 
            self.tensor_batch =  max_batch_size
        if (comm.rank == 0):
            print(f"Number of Simulation tensors per batch: {self.tensor_batch}")
            sys.stdout.flush()
    
    # Read the flag determining if data is overwritten in DB
    def read_overwrite(self, comm, t_data):
        while True:
            if (self.client.poll_tensor("tensor-ow",0,1)):
                rtime = perf_counter()
                tmp = self.client.get_tensor('tensor-ow')
                rtime = perf_counter() - rtime
                t_data.t_meta = t_data.t_meta + rtime
                t_data.i_meta = t_data.i_meta + 1 
                break
        dataOverWr = tmp[0]
        if (comm.rank==0):
            if (dataOverWr>0.5): 
                print("Training data is overwritten in DB \n")
            else:
                print("Training data is NOT overwritten in DB \n")
            sys.stdout.flush()

    # Read the flag determining how many filterwidths to train on
    def read_filters(self, cfg, t_data):
        if (cfg.train.model == "sgs"):
            while True:
                if (self.client.poll_tensor("filters",0,1)):
                    rtime = perf_counter()
                    filters = self.client.get_tensor('filters')
                    rtime = perf_counter() - rtime
                    t_data.t_meta = t_data.t_meta + rtime
                    t_data.i_meta = t_data.i_meta + 1 
                    break
            self.nfilters = filters.size
    
    # Read the mesh nodes
    def read_mesh(self, cfg, comm, t_data):
        mesh_nodes = None
        if ("qcnn" in cfg.train.model):
            rtime = perf_counter()
            mesh_nodes = self.client.get_tensor('mesh').astype('float32')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1 
            comm.comm.Barrier()
            if (comm.comm.rank==0): 
                print(f"Loaded mesh for QCNN model with size {mesh_nodes.shape}")
        return mesh_nodes