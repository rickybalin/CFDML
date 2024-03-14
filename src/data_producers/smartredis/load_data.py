from argparse import ArgumentParser, Namespace
from time import perf_counter, sleep
from typing import Tuple
import numpy as np

from smartredis import Client

# SmartRedis Client Class
class SmartRedisClient:
    def __init__(self, args, rank: int, size: int):
        self.client = None
        self.db_launch = args.db_launch
        self.db_nodes = args.db_nodes
        self.rank = rank
        self.ppn = args.ppn
        self.t_init = 0.
        self.t_meta = 0.
        self.t_train = 0.

        if (self.db_launch == "colocated"):
            self.db_nodes = 1
            self.head_rank = self.ppn * self.rank/self.ppn
        elif (self.db_launch == "clustered"):
            self.ppn = size
            self.head_rank = 0

    # Initialize client
    def init_client(self, comm):
        if (self.db_nodes==1):
            tic = perf_counter()
            self.client = Client(cluster=False)
            toc = perf_counter()
        else:
            tic = perf_counter()
            self.client = Client(cluster=True)
            toc = perf_counter()
        self.t_init = toc-tic
        comm.Barrier()
        if (self.rank==0):
            print('All SmartRedis clients initialized \n', flush=True)
    
    # Set up training case and write metadata
    def setup(self, comm, n_samples: int, ndTot: int, ndIn: int):
        if (self.rank%self.ppn == 0):
            # Run-check
            arr = np.array([1, 1], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('check-run', arr)
            toc = perf_counter()
            self.t_meta += toc - tic

            # Training data setup
            dataSizeInfo = np.empty((6,), dtype=np.int64)
            dataSizeInfo[0] = n_samples
            dataSizeInfo[1] = ndTot
            dataSizeInfo[2] = ndIn
            dataSizeInfo[3] = comm.Get_size()
            dataSizeInfo[4] = self.ppn
            dataSizeInfo[5] = self.head_rank
            tic = perf_counter()
            self.client.put_tensor('sizeInfo', dataSizeInfo)
            toc = perf_counter()
            self.t_meta += toc - tic

            # Write check-run
            tic = perf_counter()
            self.client.put_tensor('tensor-ow', arr)
            toc = perf_counter()
            self.t_meta += toc - tic

            # Time step number
            step = np.array([0, 0], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('step', step)
            toc = perf_counter()
            self.t_meta += toc - tic

        comm.Barrier()
        if (self.rank==0):
            print('Metadata sent to DB \n', flush=True)

    # Check if should keep running
    def check_run(self) -> bool:
        arr = self.client.get_tensor('check-run')
        if (arr[0]==1):
            return True
        else:
            return False
        
    # Send training snapshot
    def send_snapshot(self, array: np.ndarray):
        key = 'y.'+str(self.rank)+'.'+str(step)
        if (self.rank==0):
            print(f'Sending training data with key {key} and shape {array.shape}')
        tic = perf_counter()
        self.client.put_tensor(key, array)
        toc = perf_counter()
        self.t_train += toc - tic

    # Send time step
    def send_snapshot(self, step: int):
        if (self.rank%self.ppn == 0):
            step_arr = np.array([step, step], dtype=np.int64)
            tic = perf_counter()
            self.client.put_tensor('step', step_arr)
            toc = perf_counter()
            self.t_meta += toc - tic

    # Collect time data
    def collect_time_data(self, comm):
        stats = np.array([self.t_init, self.t_meta, self.t_train])
        summ = comm.comm.allreduce(np.array(var),op=comm.sum)
        avg = summ / comm.size
        tmp = np.array((var - avg)**2) 
        std = comm.comm.allreduce(tmp,op=comm.sum)
        std = std / comm.size
        std = math.sqrt(std)
        min_loc = comm.comm.allreduce((var,comm.rank),op=comm.minloc)
        max_loc = comm.comm.allreduce((var,comm.rank),op=comm.maxloc)

def generate_training_data(args) -> Tuple[np.ndarray, dict]:
    """Generate training data for each model
    """

def main():
    """Emulate a data producing simulation for online training with SmartSim/SmartRedis
    """
    # MPI Init
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    comm.Barrier()

    # Parse arguments
    parser = ArgumentParser(description='SmartRedis Data Producer')
    parser.add_argument('--model', default="sgs", type=str, help='ML model identifier (sgs, quadconv, gnn)')
    parser.add_argument('--problem_size', default="small", type=str, help='Size of problem to emulate (small, medium, large)')
    parser.add_argument('--db_launch', default="colocated", type=str, help='Database deployment (colocated, clustered)')
    parser.add_argument('--db_nodes', default=1, type=int, help='Number of database nodes')
    parser.add_argument('--ppn', default=4, type=int, help='Number of processes per node')
    parser.add_argument('--logging', default='no', help='Level of performance logging (no, verbose)')
    args = parser.parse_args()

    rankl = rank % args.ppn
    if (rank==0 and args.logging=="verbose"):
        print(f"Hello from MPI rank {rank}/{size}, local rank {rankl} and node {name}")
    if (rank==0):
        print(f'\nRunning with {args.dbnodes} DB nodes', flush=True)
        print(f'and with {args.ppn} processes per node \n', flush=True)

    # Initialize SmartRedis clients
    client = SmartRedisClient()
    client.init_client(comm)

    # Generate synthetic data for the specific model
    train_array, stats = generate_training_data(args)

    # Send training metadata
    client.setup(comm, stats["n_samples"], 
                 stats["n_dim_tot"], stats["n_dim_in"])

    # Emulate integration of PDEs with a do loop
    numts = 1000
    for step in range(numts):
        # Sleep for a few seconds to emulate the time required by PDE integration
        sleep(2)

        # First off check if ML is done training, if so exit from loop
        if (client.check_run()): break

        # Send training data to database
        client.send_snapshot(train_array)
        comm.Barrier()
        if (rank==0):
            print(f'All ranks finished sending training data', flush=True)
        client.send_step(step)

    comm.Barrier()
    if (rank==0):
        print("\nExited time step loop\n", flush=True)

    # Accumulate timing data and print summary
    if (rank==0):
        print("Summary of timing data:", flush=True)
    client.collect_time_data(comm)


if __name__ == "__main__":
    main()