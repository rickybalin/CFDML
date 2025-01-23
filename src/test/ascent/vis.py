import os
import numpy as np
from time import sleep
from smartredis import Client

import conduit
import conduit.blueprint
import ascent.mpi

from mpi4py import MPI



if __name__ == '__main__':
    # MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # SmartRedis client init
    address = os.getenv('SSDB')
    client = Client(address=address, cluster=False)
    comm.Barrier()
    if rank == 0: print('All clients initialized\n')

    # Open ascent
    a = ascent.mpi.Ascent()
    ascent_opts = conduit.Node()
    ascent_opts["mpi_comm"].set(MPI.COMM_WORLD.py2f())
    a.open(ascent_opts)

    # Read mesh and cnn from DB
    while True:
        if (client.poll_tensor(f'cnn_{rank}',0,1)):
            crd = client.get_tensor(f'crd_{rank}')
            cnn = client.get_tensor(f'cnn_{rank}')
            break
    comm.Barrier()
    if rank == 0: print('All ranks read mesh\n')

    # Create a conduit mesh
    mesh = conduit.Node()
    mesh["coordsets/coords/type"] = "explicit"
    mesh["coordsets/coords/values/x"].set_external(crd[:,0])
    mesh["coordsets/coords/values/y"].set_external(crd[:,1])
    mesh["coordsets/coords/values/z"].set_external(crd[:,2])
    mesh["coordsets/coords/origin/x"] = np.amin(crd[:,0])
    mesh["state/domain_id"] = rank
    mesh["topologies/topo/coordset"] = "coords"
    mesh["topologies/topo/type"] = "unstructured"
    mesh["topologies/topo/elements/shape"] = "hex"
    mesh["topologies/topo/elements/connectivity"].set_external(cnn.flatten())
    #print(mesh.to_yaml())

    # make sure the mesh we created conforms to the blueprint
    verify_info = conduit.Node()
    assert conduit.blueprint.mesh.verify(mesh,verify_info), "Mesh verified failed"
    comm.Barrier()

    # Setup vis
    #scenes  = conduit.Node()
    #scenes["s1/plots/p1/type"] = "pseudocolor"
    #scenes["s1/plots/p1/field"] = "var"
    #scenes["s1/image_prefix"] = "out_ascent_render_mpi_3d"
    #actions = conduit.Node()
    #add_act =actions.append()
    #add_act["action"] = "add_scenes"
    #add_act["scenes"] = scenes

    # Ready to visialize
    if rank == 0: client.put_tensor('ready',np.array([1]))

    # Create visualizations
    step = -100
    while True:
        sleep(1)
        if not client.get_tensor('stop').item():
            if client.poll_tensor("step",0,1):
                step_read = client.get_tensor('step').item()
                if step_read != step:
                    step = step_read
                    if rank == 0: print(f'New data available to read for step {step}!')
                    var = client.get_tensor(f'var_{rank}_{step}')
                    mesh[f"fields/var_{step}/association"] = "vertex"
                    mesh[f"fields/var_{step}/topology"] = "topo"
                    mesh[f"fields/var_{step}/values"].set_external(var)
                    a.publish(mesh)

                    scenes  = conduit.Node()
                    scenes["s1/plots/p1/type"] = "pseudocolor"
                    scenes["s1/plots/p1/field"] = f"var_{step}"
                    scenes["s1/image_prefix"] = f"step_{step}"

                    actions = conduit.Node()
                    add_act = actions.append()
                    add_act["action"] = "add_scenes"
                    add_act["scenes"] = scenes
                    a.execute(actions)
            else:
                if rank == 0: print('No data available yet\n')
                continue
        else:
            if rank == 0: print('Sim said to stop\n')
            break

    # Done
    a.close()
    if rank == 0: print('Exiting ...')


    