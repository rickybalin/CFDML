# This script loads in an ASCII mesh (Matlab, Python or MGEN generated)
# and partitions it in equal parts such that it can be passed through
# mner (matchedNodeElmReader) in parallel and create a parallel mds mesh.

import numpy as np
from itertools import chain
import auxFunctions as af

## Define split factor
x_split_factor = 8 # split the mesh in this many parts along x direction
z_split_factor = 2 # split the mesh in this many parts along z direction
split_factor = x_split_factor*z_split_factor

## Load serial files
fprefix = './'
fname = 'FlatPlate_CRS_6-15_4d'
m = af.load_serial_data(fprefix+fname)

## Determine how to split the mesh
if (x_split_factor>1 and z_split_factor==1):
    print("\nSplitting in x only\n")
    split_z = False
    split_x = True
    error_msg = "Number of elements in x not divisible by requested split factor"
    assert m.stats[1,0]%x_split_factor==0, error_msg
elif (z_split_factor>1 and x_split_factor==1):
    print("\nSplitting in z only\n")
    split_z = True
    split_x = False
    error_msg = "Number of elements in z not divisible by requested split factor"
    assert m.stats[1,2]%z_split_factor==0, error_msg
else:
    print("\nSplitting in x and z\n")
    split_z = True
    split_x = True
    error_msg = "Number of elements in x and z not divisible by requested split factor"
    assert m.stats[1,0]%x_split_factor==0 and m.stats[1,2]%z_split_factor==0, error_msg

numel_loc = int(m.stats[1,3] / split_factor)
numelx_loc = max(1, int(m.stats[1,0] / x_split_factor))
numelz_loc = max(1, int(m.stats[1,2] / z_split_factor))
if (split_x and not split_z):
    nx_loc = numelx_loc
    nz_loc = m.stats[0,2]
elif (split_z and not split_x):
    nx_loc = m.stats[0,0]
    nz_loc = numelz_loc
elif (split_z and split_x):
    nx_loc = numelx_loc
    nz_loc = numelz_loc
numnp_loc = nx_loc*m.stats[0,1]*numelz_loc
print("Creating partitions with:")
print(f"{numel_loc} total elements")
print(f"{numelx_loc} elements in x")
print(f"{numelz_loc} elements in z")
print(f"{numnp_loc} total nodes")

## Split mesh and write new mesh files
print("Splitting mesh ...")
if (split_x and not split_z):
    af.split_mesh_x(fname, x_split_factor, numnp_loc, numel_loc, m)
elif (split_z and not split_x):
    af.split_mesh_z(fname, z_split_factor, numnp_loc, nz_loc,
                    numel_loc, numelz_loc, m)
elif (split_x and split_z):
    af.split_mesh_z_x(fname, x_split_factor, z_split_factor, numnp_loc, nx_loc, 
                      nz_loc, numel_loc, numelx_loc, numelz_loc, m)
print("Done\n")



