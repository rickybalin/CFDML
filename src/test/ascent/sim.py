import os
import numpy as np
from time import sleep
from smartredis import Client

from mpi4py import MPI

def gen_hex_mesh(num_elms, rank):
    num_vtx = num_elms+1
    x = np.linspace(rank,rank+1,num_vtx)
    y = np.linspace(0,1,num_vtx)
    z = np.linspace(0,1,num_vtx)
    crd = np.zeros((num_vtx**3,3), dtype=np.float64)
    iv = 0
    for k in range(num_vtx):
        for j in range(num_vtx):
            for i in range(num_vtx):
                crd[iv,0] = x[i]
                crd[iv,1] = y[j]
                crd[iv,2] = z[k]
                iv+=1

    cnn = np.zeros((num_elms**3,8), dtype=np.int64)
    n = np.empty([8])
    #n[0] = 0
    #n[1] = num_elms+1
    #n[2] = num_elms+2
    #n[3] = 1
    #n[4] = num_elms*num_elms+1
    #n[7] = num_elms*num_elms+2
    #n[6] = num_elms*(num_elms+1)+2
    #n[5] = num_elms*(num_elms+1)+1
    n[0] = 0
    n[1] = 1
    n[2] = num_vtx+1
    n[3] = num_vtx
    n[4] = num_vtx*num_vtx
    n[5] = num_vtx*num_vtx+1
    n[6] = num_vtx*(num_vtx+1)+1
    n[7] = num_vtx*(num_vtx+1)
    ie = 0
    for k in range(num_elms):
        for j in range(num_elms):
            for i in range(num_elms):
                #tmp1 = k+j*num_elms+i*num_elms*num_elms
                tmp1 = i+j*num_vtx+k*num_vtx*num_vtx
                tmp = n+tmp1
                cnn[ie] = tmp
                ie+=1

    """
    truth = np.array([[0, 1, 4, 3, 9, 10, 13, 12], 
                        [1, 2, 5, 4, 10, 11, 14, 13], 
                        [3, 4, 7, 6, 12, 13, 16, 15], 
                        [4, 5, 8, 7, 13, 14, 17, 16], 
                        [9, 10, 13, 12, 18, 19, 22, 21], 
                        [10, 11, 14, 13, 19, 20, 23, 22], 
                        [12, 13, 16, 15, 21, 22, 25, 24], 
                        [13, 14, 17, 16, 22, 23, 26, 25]])
    assert np.allclose(truth,cnn)
    """
    return crd, cnn

if __name__ == '__main__':
    # MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # SmartRedis client init
    address = os.getenv('SSDB')
    client = Client(address=address, cluster=False)
    if rank == 0: client.put_tensor('stop',np.array([0]))
    comm.Barrier()
    if rank == 0: print('All clients initialized\n')

    # Generate and send a distributed mesh
    num_elms = 2
    crd, cnn = gen_hex_mesh(num_elms, rank)
    client.put_tensor(f'crd_{rank}',crd)
    client.put_tensor(f'cnn_{rank}',cnn)
    comm.Barrier()
    if rank == 0: print('All ranks sent crd and cnn data\n')

    # Wait until ready to visualize
    if rank == 0: print('Waiting for vis script to be ready ...')
    while True:
        if (client.poll_tensor("ready",0,1)):
            break
    comm.Barrier()
    if rank == 0: print('Ready!\n')

    # Iterate and produce some fields to visualize
    var = np.ones((crd.shape[0],)) * rank
    increment = np.ones((crd.shape[0],)) * size
    if rank==0: print(increment)
    num_iters = 2
    for i in range(num_iters):
        var = var + i*increment
        if rank==0: print(var)
        print(np.amax(var))
        client.put_tensor(f'var_{rank}_{i}',var)
        if rank == 0: client.put_tensor('step',np.array([i]))
        comm.Barrier()
        if rank == 0: print('Sent new field to visualize')
        sleep(10)

    # Done
    if rank == 0: client.put_tensor('stop',np.array([1]))
    sleep(10)
    comm.Barrier()
    if rank == 0: print('Exiting ...')

    
