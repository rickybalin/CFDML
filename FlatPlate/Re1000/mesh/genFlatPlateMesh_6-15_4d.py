# This tool generates the mesh data files that will be then converted by SCOREC-core
# and partitioned to be run with PHASTA

import numpy as np
import matplotlib.pyplot as plt
import auxFunctions as af

#%% Define some flow and fluid properties
nu = 1.250224688859004e-05
rho = 1.0
mu = rho*nu
uTauIn = 0.046987317422423
lvisc = nu/uTauIn

#%% Define the starting corner of the domain and top surface
xmin = -3.1; xmax = 0
Lx = xmax-xmin
# create a sloped top boundary so it can be set as an outflow (better for ZPG)
def top(t):
    return -0.14/3.1*t+2.26
ymin = 0.; ymax = top(xmin)
Ly = ymax-ymin
zmin = 0; zmax = 0.452
Lz = zmax-zmin

#%% Define grid spacing
nz = 283 # this creates 282 elements
dz = 6*lvisc
z = np.linspace(zmin,zmax-dz,nz)

dx = 15*lvisc
nx = round(Lx/dx) # this creates 776 elements
x = np.linspace(xmin,xmax,nx)

dyMin = 0.1*lvisc # first point off wall thickness
dyMax = 10*lvisc # max spacing inside BL
alphaY = 1.05 # growth inside BL
BLMeshHeight = 0.15 # thickness of BL mesh (mesh contaiing pysical BL)
dyBL,yBL = af.getDySpacingBL(ymin,1.4*BLMeshHeight,dyMin,dyMax,alphaY)
nyBL = yBL.size
alphaY = 1.2 # growth outside BL in the free stream
dyMax = 0.1 # max spacing int he free stream
distFSInflow = top(xmin)-yBL[-1] # distance from edge of BL mesh to top
dyFS,yFS = af.getDySpacingFS(yBL,dyBL,distFSInflow,dyMax,alphaY)
nyFS = yFS.size
y = np.concatenate((yBL,yFS))
ny = y.size
numnp = nx*ny*nz # total number of points in the mesh
print(f"Generating a mesh with (nx,ny,nz) = ({nx},{ny},{nz})\n")

## Load in the initial condition to attach to the mesh
print("Loading the IC ...")
nx_ic = min(nx+1,777)  # total is 777
ny_ic = 191
nz_ic = 284
numnp_ic = nx_ic*ny_ic*nz_ic
IC = np.load('IC.npy') # NB: this is a solution from 4xdelta domain
print("Done\n")

#%% Write node coordinates and other mesh data structures to file
print("Looping over mesh vertices ...")
fname = 'FlatPlate_CRS_6-15_4d'
f0 = open(fname+'.stats', 'w')
f0.write('%d %d %d %d\n' % (nx, ny, nz, numnp))
f1 = open(fname+'.crd.0', 'w') # contains coordinates of mesh points
f1.write('%d\n' % numnp)
f2 = open(fname+'.match.0', 'w') # matching information for periodic BCs
f3 = open(fname+'.class.0', 'w') # classification of mesh points on geometry
f4 = open(fname+'.fathers2D.0', 'w') # used for spanwise averaging (nice for flow analysis)
f6 = open(fname+'.soln.0', 'w') # stores initial condition
inode = 0; inode2D = 0
for i in range(nx):
    for j in range(ny):
        inode2D = inode2D+1
        if j<=nyBL-1:
            ytmp = yBL[j]
        else:
            currFSh = top(x[i])-yBL[-1]
            vec = dyFS/distFSInflow*currFSh
            ytmp = yBL[-1]+np.sum(vec[0:j-nyBL+1])
        for k in range(nz):
            inode = inode+1
            pc_done = np.round(inode/numnp,7)
            if (pc_done*100%10==0):
                print(f"{pc_done*100}%")
            # write coordinates to file
            f1.write('%.12e %.12e %.12e\n' % (x[i],ytmp,z[k]))
            # check if current node is on the dependent periodic face
            # the dependent face is the one at zmin
            if k==0:
                match = inode+(nz-1)
            elif k==nz-1:
                match = inode-(nz-1)
            else:
                match = -1
            # write to the matching data file
            f2.write('%d\n' % (match))
            # obtain the classification of the current node and write
            iclass = af.getClassification_dmgModel(i,j,k,nx,ny,nz)
            f3.write('%d\n' % (iclass))
            # write the 2D node number of each node
            f4.write('%d\n' % (inode2D))
            # write the initial condition
            z_index = k%nz_ic
            inode_ic = min(i,nx_ic-1)*ny_ic*nz_ic + j*nz_ic + z_index
            f6.write('%.12e %.12e %.12e %.12e\n' % (IC[inode_ic,0],
                      IC[inode_ic,1],IC[inode_ic,2],IC[inode_ic,3]))
            
f1.close()
f2.close()
f3.close()
f4.close()
f6.close()
print("Done\n")


#%% Write mesh connectivity to file
print("Looping over mesh elements ...")
f5 = open(fname+'.cnn.0', 'w') # contains connectivity information (defines mesh elements)
f5b = open(fname+'Head.cnn', 'w')
# Code below is for hexahedral elemets (can make a tet mesh too if desired)
numelx = (nx-1) 
numely = (ny-1) 
numelz = (nz-1)
numel = numelx*numely*numelz
f0.write('%d %d %d %d\n' % (numelx, numely, numelz, numel))
#f5b.write('%d %d\n' % (1,1))
#f5b.write('%d\n' % (3))
#f5b.write('%d %d\n' % (numel,8))
f5b.write(' 0\n')
f5b.write('%d %d\n' % (numel,8))
n = np.empty([8])
n[0] = 1
n[1] = nz+1
n[2] = nz+2
n[3] = 2
n[4] = nz*ny+1
n[7] = nz*ny+2
n[6] = nz*(ny+1)+2
n[5] = nz*(ny+1)+1

iel = 0
for i in range(numelx):
    for j in range(numely):
        for k in range(numelz):
            iel = iel+1
            pc_done = np.round(iel/numel,7)
            if (pc_done*100%10==0):
                print(f"{pc_done*100}%")
            tmp1 = k+j*nz+i*nz*ny
            tmp = n+tmp1
            f5.write('%d %d %d %d %d %d %d %d\n' % (tmp[0], tmp[1],
                                                    tmp[2], tmp[3], tmp[4],
                                                    tmp[5], tmp[6], tmp[7]))
            
f0.close()
f5.close()
f5b.close()
print("Done\n")

