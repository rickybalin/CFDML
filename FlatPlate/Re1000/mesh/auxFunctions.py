import numpy as np
from itertools import chain

#%% Compute dx spacing and x coordinate of nodes on surface
def getDxSpacing(xmin,xmax,dxMin,dxMax,alpha):
    dx = np.array([dxMin])
    x = np.array([xmin, xmin+dxMin])
    
    i = 1
    while x[i] <= xmax:
        tmp = dx[-1]*alpha
        if tmp>dxMax:
            tmp = dxMax
        dx = np.append(dx,[tmp])
        x = np.append(x,[x[i]+tmp])
        i = i+1
        
    return dx,x
    

#%% Compute dy spacing inside the BL mesh
def getDySpacingBL(ymin,BLHeight,dyMin,dyMax,alpha):
    dy = np.array([dyMin])
    y = np.array([ymin, ymin+dyMin])
    
    j = 1
    while y[j] <= BLHeight:
        tmp = dy[-1]*alpha
        if tmp>dyMax:
            tmp = dyMax
        dy = np.append(dy,[tmp])
        y = np.append(y,[y[j]+tmp])
        j = j+1

    y[-1] = BLHeight
        
    return dy,y


#%% Compute dy spacing outside the BL mesh to fill the rest of the domain
# This is the free stream region of the flow
def getDySpacingFS(yBL,dyBL,distFSInflow,dyMax,alpha):
    #dy = np.array([dyBL[-1]])
    dy = []
    y = []
    height = 0.
    
    j = 0
    while height < distFSInflow:
        tmp = dyBL[-1]*alpha**(j+1)
        if tmp>dyMax:
            tmp = dyMax      
        dy = np.append(dy,[tmp])
        height += tmp
        y = np.append(y,[height])
        j = j+1
        
    y[-1] = distFSInflow
    y = y+yBL[-1]
    dy[-1] = y[-1]-y[-2]
          
    return dy,y


#%% Get the classification on the model geometry
def getClassification(i,j,k,nt,nn,nz):
    # This function takes in the counter of the mesh coordinate generation loop
    # and returns the classification of the node described by (i,j,k).
    # The classification is:
    # 0 = fully interior of the volume
    # 1-6 = classified on face (not edge or vertex)
    # 11-22 = classified on model edge (not end points which are model vertices)
    # 31-38 = classified on a model vertex.

    # Given the way the for loops in BumpStructMesh, the idea is that if one of
    # the indices is either 1 or it's max then then node is on a face, if 2
    # indices meet this criteria then it's on an edge, if all 3 then it's on a
    # vertex.

    iclass = -1
    
    if i==0: # node is on the inflow face (xmin face)
        if j==0: # then node is on ymin edge of inflow
            if k==0: # then on xmin,ymin,zmin vertex
                iclass = 31
            elif k==nz-1: # then on xmin,ymin,zmax vertex
                iclass = 35
            else: # then on xmin,ymin edge
                iclass = 15
        elif j==nn-1: # then node is on ymax edge 
            if k==0: # then on xmin,ymax,zmin vertex
                iclass = 34
            elif k==nz-1: # then on xmin,ymax,zmax vertex
                iclass = 38
            else: # then on xmin,ymax edge
                iclass = 18
        elif k==0: # then on zmin edge of inflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 31 or 34
                iclass = 14
        elif k==nz-1: # then on zmax edge of inflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 35 or 38
                iclass = 22
        else: # then only on inflow face
            iclass = 5
    elif i==nt-1: # then on the outflow face
        if j==0: # then node is on ymin edge of outflow
            if k==0: # then on xmax,ymin,zmin vertex
                iclass = 32
            elif k==nz-1: # then on xmax,ymin,zmax vertex
                iclass = 36
            else: # then on xmax,ymin edge
                iclass = 16
        elif j==nn-1: # then node is on ymax edge of outflow
            if k==0: # then on xmax,ymax,zmin vertex
                iclass = 33
            elif k==nz-1: # then on xmax,ymax,zmax vertex
                iclass = 37
            else: # then on xmax,ymax edge
                iclass = 17
        elif k==0: # then on zmin edge of outflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 32 or 33
                iclass = 12
        elif k==nz-1: # then on zmax edge of outlow
            if j!=0 and j!=nn-1: # make sure it's not vertex 36 or 37
                iclass = 20
        else: # then only on outlow face
            iclass = 3
    elif j==0: # then on bottom surface
        # at this point already assigned all vertices, and edges shared with
        # the inflow and outflow faces
        if i!=0 and i!=nt-1: # then must be on sides
            if k==0: # then ymin,zmin edge
                iclass = 11
            elif k==nz-1: # then ymin,zmax edge
                iclass = 19
            else: # then just on ymin face
                iclass = 2
    elif j==nn-1: # then on top surface
        # at this point already assigned all vertices, and edges shared with
        # the inflow and outflow faces
        if i!=0 and i!=nt-1: # then must be on sides
            if k==0: # then ymax,zmin edge
                iclass = 13
            elif k==nz-1: # then ymax,zmax edge
                iclass = 21
            else: # then just on ymax face
                iclass = 4
    elif k==0: # then on zmin side surface
        # at this point all vertices are being assigned, and also all edges
        # so these are just nodes on the zmin face only
        if i!=0 and i!=nt-1: # then must be on sides
            if j!=0 and j!=nn-1: # then must be on a face only
                iclass = 1
    elif k==nz-1: # then on zmax side surface
        # at this point all vertices are being assigned, and also all edges
        # so these are just nodes on the zmax face only
        if i!=0 and i!=nt-1: # then must be on sides
            if j!=0 and j!=nn-1: # then must be on a face only
                iclass = 6
    else: # this is an interior node
        iclass = 0


    return iclass


#%% Get the classification on the model geometry based on SCOREC-core master branch
def getClassification_dmgModel(i,j,k,nt,nn,nz):
    # This function takes in the counter of the mesh coordinate generation loop
    # and returns the classification of the node described by (i,j,k).
    # Classification is based on a box dmg geometric model from SCOREC-core,
    # and follows these rules: 
    #  - add 3 million to model regions
    #  - add 2 million to model faces
    #  - add 1 million to model edges
    #  - don’t add anything to model vertices

    # Given the way the for loops in BumpStructMesh, the idea is that if one of
    # the indices is either 1 or it's max then then node is on a face, if 2
    # indices meet this criteria then it's on an edge, if all 3 then it's on a
    # vertex.

    iclass = -1
    addVtx = 0
    addEdge = 1000000
    addFace = 2000000
    addRegion = 3000000
    
    if i==0: # node is on the inflow face (xmin face)
        if j==0: # then node is on ymin edge of inflow
            if k==0: # then on xmin,ymin,zmin vertex
                iclass = 0+addVtx
            elif k==nz-1: # then on xmin,ymin,zmax vertex
                iclass = 4+addVtx
            else: # then on xmin,ymin edge
                iclass = 4+addEdge
        elif j==nn-1: # then node is on ymax edge 
            if k==0: # then on xmin,ymax,zmin vertex
                iclass = 2+addVtx
            elif k==nz-1: # then on xmin,ymax,zmax vertex
                iclass = 6+addVtx
            else: # then on xmin,ymax edge
                iclass = 6+addEdge
        elif k==0: # then on zmin edge of inflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 0 or 2
                iclass = 1+addEdge
        elif k==nz-1: # then on zmax edge of inflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 4 or 6
                iclass = 9+addEdge
        else: # then only on inflow face
            iclass = 2+addFace
    elif i==nt-1: # then on the outflow face
        if j==0: # then node is on ymin edge of outflow
            if k==0: # then on xmax,ymin,zmin vertex
                iclass = 1+addVtx
            elif k==nz-1: # then on xmax,ymin,zmax vertex
                iclass = 5+addVtx
            else: # then on xmax,ymin edge
                iclass = 5+addEdge
        elif j==nn-1: # then node is on ymax edge of outflow
            if k==0: # then on xmax,ymax,zmin vertex
                iclass = 3+addVtx
            elif k==nz-1: # then on xmax,ymax,zmax vertex
                iclass = 7+addVtx
            else: # then on xmax,ymax edge
                iclass = 7+addEdge
        elif k==0: # then on zmin edge of outflow
            if j!=0 and j!=nn-1: # make sure it's not vertex 1 or 3
                iclass = 2+addEdge
        elif k==nz-1: # then on zmax edge of outlow
            if j!=0 and j!=nn-1: # make sure it's not vertex 5 or 7
                iclass = 10+addEdge
        else: # then only on outlow face
            iclass = 3+addFace
    elif j==0: # then on bottom surface
        # at this point already assigned all vertices, and edges shared with
        # the inflow and outflow faces
        if i!=0 and i!=nt-1: # then must be on sides
            if k==0: # then ymin,zmin edge
                iclass = 0+addEdge
            elif k==nz-1: # then ymin,zmax edge
                iclass = 8+addEdge
            else: # then just on ymin face
                iclass = 1+addFace
    elif j==nn-1: # then on top surface
        # at this point already assigned all vertices, and edges shared with
        # the inflow and outflow faces
        if i!=0 and i!=nt-1: # then must be on sides
            if k==0: # then ymax,zmin edge
                iclass = 3+addEdge
            elif k==nz-1: # then ymax,zmax edge
                iclass = 11+addEdge
            else: # then just on ymax face
                iclass = 4+addFace
    elif k==0: # then on zmin side surface
        # at this point all vertices are being assigned, and also all edges
        # so these are just nodes on the zmin face only
        if i!=0 and i!=nt-1: # then must be on sides
            if j!=0 and j!=nn-1: # then must be on a face only
                iclass = 0+addFace
    elif k==nz-1: # then on zmax side surface
        # at this point all vertices are being assigned, and also all edges
        # so these are just nodes on the zmax face only
        if i!=0 and i!=nt-1: # then must be on sides
            if j!=0 and j!=nn-1: # then must be on a face only
                iclass = 5+addFace
    else: # this is an interior node
        iclass = 0+addRegion


    return iclass


#%% Load the serial mesh data
class Mesh:
    def __init__(self, stats, coords, match, classif, fathers, cnn, sol):
        self.stats = stats
        self.coords = coords
        self.match = match
        self.classif = classif
        self.fathers = fathers
        self.cnn = cnn
        self.sol = sol

def load_serial_data(fname):
    print("Loading serial mesh ...")
    f0 = open(fname+'.stats', 'r')
    f1 = open(fname+'.crd.0', 'r') # contains coordinates of mesh points
    f2 = open(fname+'.match.0', 'r') # matching information for periodic BCs
    f3 = open(fname+'.class.0', 'r') # classification of mesh points on geometry
    f4 = open(fname+'.fathers2D.0', 'r') # used for spanwise averaging (nice for flow analysis)
    f5 = open(fname+'.cnn.0', 'r') # contains connectivity information (defines mesh elements)
    f5b = open(fname+'Head.cnn', 'r')
    f6 = open(fname+'.soln.0', 'r') # stores initial condition

    # Read useful metadata first
    stats = np.genfromtxt(f0, delimiter=' ', dtype=np.int32)
    print("Mesh stats:")
    print(f"Number of vertices: {stats[0,0]}, {stats[0,1]}, {stats[0,2]}, {stats[0,3]}")
    print(f"Number of elements: {stats[1,0]}, {stats[1,1]}, {stats[1,2]}, {stats[1,3]}")

    # Read rest of the data
    coords = np.genfromtxt(f1, delimiter=' ',skip_header=1)
    match = np.genfromtxt(f2)
    classif = np.genfromtxt(f3)
    fathers = np.genfromtxt(f4)
    cnn = np.genfromtxt(f5, delimiter=' ')
    sol = np.genfromtxt(f6, delimiter=' ')
    
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f5b.close()
    f6.close()
    print("Done\n")

    return Mesh(stats, coords, match, classif, fathers, cnn, sol)


#%% Split the mesh in the x direction only
def split_mesh_x(fname, split_factor, numnp_loc, numel_loc, m):
    """This is the easy case where there is no need to change the
    global numbering of the mesh because the mesh is already ordered
    as z,y,x
    """
    f5b = open(fname+f'_{split_factor}partHead.cnn', 'w')

    ive = 0
    for part_id in range(split_factor):
        f1 = open(fname+f'_{split_factor}part.crd.{part_id}', 'w') 
        f2 = open(fname+f'_{split_factor}part.match.{part_id}', 'w')
        f3 = open(fname+f'_{split_factor}part.class.{part_id}', 'w') 
        f4 = open(fname+f'_{split_factor}part.fathers2D.{part_id}', 'w') 
        f5 = open(fname+f'_{split_factor}part.cnn.{part_id}', 'w')
        f6 = open(fname+f'_{split_factor}part.soln.{part_id}', 'w')

        ivs = numnp_loc*part_id
        ive = numnp_loc*(part_id+1)
        if (part_id == (split_factor-1)): 
            ive = ive+m.stats[0,1]*m.stats[0,2]
        node_list = range(int(ivs),int(ive))
        elm_list = range(int(numel_loc*part_id),int(numel_loc*(part_id+1)))

        f1.write('%d\n' % len(node_list))
        for iv in node_list:
            f1.write('%.12e %.12e %.12e\n' % \
                (m.coords[iv,0],m.coords[iv,1],m.coords[iv,2]))
            f2.write('%d\n' % (m.match[iv]))
            f3.write('%d\n' % (m.classif[iv]))
            f4.write('%d\n' % (m.fathers[iv]))
            f6.write('%.12e %.12e %.12e %.12e\n' % \
                (m.sol[iv,0],m.sol[iv,1],m.sol[iv,2],m.sol[iv,3]))

        assert len(elm_list)==numel_loc, "Wrong number of elements in element list"
        f5b.write(' %d\n' % (part_id))
        f5b.write('%d %d\n' % (numel_loc,8))
        for ie in elm_list:
            f5.write('%d %d %d %d %d %d %d %d\n' % (m.cnn[ie,0], m.cnn[ie,1],
                                                    m.cnn[ie,2], m.cnn[ie,3], m.cnn[ie,4],
                                                    m.cnn[ie,5], m.cnn[ie,6], m.cnn[ie,7]))
        
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()

    f5b.close()


#%% Split the mesh in the z direction only
def split_mesh_z(fname, split_factor, numnp_loc, nz_loc, 
                 numel_loc, numelz_loc, m):
    """ This case is more complex because the global numbering of the mesh
    nodes needs to change. The SCOREC-core executable called mner assumes that
    the global numbering of the nodes follows linearly with the partitions.
    So, part 1 will have global nodes 1 - numnp_loc
        part 2 will have global nodes numnp_loc+1 - 2xnumnp_loc
        ...
    and this is no longer true for the z,y,x ordered mesh when we split in z.
    So we need to update global numbering, and this really means updating
    the matching information, father-son information and connectivity information.
    """

    f5b = open(fname+f'_{split_factor}partHead.cnn', 'w')
    ive = 0
    for part_id in range(split_factor):
        # Open files for this part
        f1 = open(fname+f'_{split_factor}part.crd.{part_id}', 'w') 
        f2 = open(fname+f'_{split_factor}part.match.{part_id}', 'w')
        f3 = open(fname+f'_{split_factor}part.class.{part_id}', 'w') 
        f4 = open(fname+f'_{split_factor}part.fathers2D.{part_id}', 'w') 
        f5 = open(fname+f'_{split_factor}part.cnn.{part_id}', 'w')
        f6 = open(fname+f'_{split_factor}part.soln.{part_id}', 'w')

        # Obtain list of nodes and elements contained in this part based on 
        # serial mesh ordering
        node_list = []
        elm_list = []
        z_parts = split_factor
        z_part = part_id
        nz_part = nz_loc+1 if (z_part==z_parts-1) else nz_loc
        nz_part_next = nz_loc if (z_part+1<z_parts-1) else nz_loc+1
        ivs_part = nz_loc*z_part
        ies_part = numelz_loc*z_part
        for i in range(m.stats[0,0]):
            for j in range(m.stats[0,1]):
                ivs = i*m.stats[0,1]*m.stats[0,2] + ivs_part+j*m.stats[0,2]
                ive = ivs+nz_part
                node_list = chain(node_list, range(ivs,ive))
                if (j<m.stats[1,1] and i<m.stats[1,0]):
                    ies = i*m.stats[1,1]*m.stats[1,2] + ies_part+j*m.stats[1,2]
                    iee = ies+numelz_loc
                    elm_list = chain(elm_list, range(ies,iee))
        node_list = list(node_list)
        elm_list = list(elm_list)
        if (z_part==z_parts-1):
            assert len(node_list)==numnp_loc+m.stats[0,1]*m.stats[0,0], "Wrong number of nodes in node list"
        else:
            assert len(node_list)==numnp_loc, "Wrong number of nodes in node list"

        # Write the nodal mesh fields to file for this part
        f1.write('%d\n' % len(node_list))
        inode = 0
        offset = offset+node_list_prev if (part_id>0) else 0
        glob_inode = inode+offset
        for i in range(m.stats[0,0]):
            for j in range(m.stats[0,1]):
                for k in range(nz_part):
                    ind = node_list[inode]
                    f1.write('%.12e %.12e %.12e\n' % \
                        (m.coords[ind,0],m.coords[ind,1],m.coords[ind,2]))
                    f3.write('%d\n' % (m.classif[ind]))
                    f4.write('%d\n' % (m.fathers[ind]))
                    f6.write('%.12e %.12e %.12e %.12e\n' % \
                        (m.sol[ind,0],m.sol[ind,1],m.sol[ind,2],m.sol[ind,3]))
                    if (m.match[ind]==-1):
                        f2.write('%d\n' % (m.match[ind]))
                    else:
                        if (m.coords[ind,2]==m.coords[0,2]):
                            new_match = numnp_loc*(z_parts-1)+(nz_loc+1) \
                                        + i*(nz_loc+1)*m.stats[0,1] \
                                        + j*(nz_loc+1) + k
                        else:
                            new_match = i*nz_loc*m.stats[0,1] \
                                        + j*nz_loc + k-nz_loc +1
                        f2.write('%d\n' % (new_match))
                    inode += 1
                    glob_inode +=1
        
        assert len(elm_list)==numel_loc, "Wrong number of elements in element list"
        f5b.write(' %d\n' % (part_id))
        f5b.write('%d %d\n' % (numel_loc,8))
        ny = m.stats[0,1]
        n = np.empty([8])
        n[0] = 1
        n[1] = nz_part+1
        n[2] = nz_part+2
        n[3] = 2
        n[4] = nz_part*ny+1
        n[7] = nz_part*ny+2
        n[6] = nz_part*(ny+1)+2
        n[5] = nz_part*(ny+1)+1
        for i in range(m.stats[1,0]):
            for j in range(m.stats[1,1]):
                for k in range(numelz_loc):
                    tmp1 = k+j*nz_part+i*nz_part*ny + offset
                    tmp = n+tmp1
                    if (part_id<split_factor-1 and k==numelz_loc-1):
                        tmp[3] = numnp_loc*(part_id+1)+i*ny*nz_part_next+j*nz_part_next+1
                        tmp[2] = tmp[3]+nz_part_next
                        tmp[7] = tmp[3]+nz_part_next*ny
                        tmp[6] = tmp[7]+nz_part_next
                    f5.write('%d %d %d %d %d %d %d %d\n' % (tmp[0], tmp[1],
                                                    tmp[2], tmp[3], tmp[4],
                                                    tmp[5], tmp[6], tmp[7]))
        node_list_prev = len(node_list)

        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()

    f5b.close()


#%% Split the mesh in the z direction first and then in x
def split_mesh_z_x(fname, x_split_factor, z_split_factor, numnp_loc, nx_loc, 
                   nz_loc, numel_loc, numelx_loc, numelz_loc, m):
    """ This case is more complex because the global numbering of the mesh
    nodes needs to change. The SCOREC-core executable called mner assumes that
    the global numbering of the nodes follows linearly with the partitions.
    So, part 1 will have global nodes 1 - numnp_loc
        part 2 will have global nodes numnp_loc+1 - 2xnumnp_loc
        ...
    and this is no longer true for the z,y,x ordered mesh when we split in z.
    So we need to update global numbering, and this really means updating
    the matching information, father-son information and connectivity information.
    """
    split_factor = x_split_factor*z_split_factor
    ny = m.stats[0,1]
    nz = m.stats[0,2]

    f5b = open(fname+f'_{split_factor}partHead.cnn', 'w')
    ive = 0
    for part_id in range(split_factor):
        print(f"Working on part {part_id+1}/{split_factor}")
        # Open files for this part
        f1 = open(fname+f'_{split_factor}part.crd.{part_id}', 'w') 
        f2 = open(fname+f'_{split_factor}part.match.{part_id}', 'w')
        f3 = open(fname+f'_{split_factor}part.class.{part_id}', 'w') 
        f4 = open(fname+f'_{split_factor}part.fathers2D.{part_id}', 'w') 
        f5 = open(fname+f'_{split_factor}part.cnn.{part_id}', 'w')
        f6 = open(fname+f'_{split_factor}part.soln.{part_id}', 'w')

        # Obtain list of nodes and elements contained in this part based on 
        # serial mesh ordering
        node_list = []
        elm_list = []
        z_parts = z_split_factor
        x_parts = x_split_factor
        z_part = part_id%z_parts
        x_part = part_id//z_parts
        nz_part = nz_loc if (z_part<z_parts-1) else nz_loc+1
        nz_part_next = nz_loc+1 if (z_part+1==z_parts-1) else nz_loc
        nx_part = nx_loc if (x_part<x_parts-1) else nx_loc+1
        nx_part_next = nx_loc+1 if (x_part+1==x_parts-1) else nx_loc
        ivs_part = x_part*ny*nz*nx_loc + nz_loc*z_part
        for i in range(nx_part):
            for j in range(ny):
                ivs = i*ny*nz + ivs_part+j*nz
                ive = ivs+nz_part
                node_list = chain(node_list, range(ivs,ive))
        node_list = list(node_list)
        if (z_part==z_parts-1 and x_part==x_parts-1):
            numnp_part = numnp_loc + ny*nx_part + ny*nz_part - ny
            numnp_part_next = None
            assert len(node_list)==numnp_part, "Wrong number of nodes in node list"
        elif (x_part==x_parts-1):
            numnp_part = numnp_loc+nz_part*ny
            numnp_part_next = numnp_loc+nz_part_next*ny
            assert len(node_list)==numnp_part, "Wrong number of nodes in node list"
        elif (z_part==z_parts-1):
            numnp_part = numnp_loc+nx_part*ny
            if (x_part+1==x_parts-1):
                numnp_part_next = numnp_loc+nz_part_next*ny
            else:
                numnp_part_next = numnp_loc
            assert len(node_list)==numnp_part, "Wrong number of nodes in node list"
        else:
            numnp_part = numnp_loc
            if (z_part+1==z_parts-1):
                numnp_part_next = numnp_loc+nz_part_next*ny
            else:
                numnp_part_next = numnp_loc
            assert len(node_list)==numnp_loc, "Wrong number of nodes in node list"

        # Write the nodal mesh fields to file for this part
        f1.write('%d\n' % len(node_list))
        inode = 0
        offset = offset+node_list_prev if (part_id>0) else 0
        glob_inode = inode+offset
        for i in range(nx_part):
            for j in range(m.stats[0,1]):
                for k in range(nz_part):
                    ind = node_list[inode]
                    f1.write('%.12e %.12e %.12e\n' % \
                        (m.coords[ind,0],m.coords[ind,1],m.coords[ind,2]))
                    f3.write('%d\n' % (m.classif[ind]))
                    f4.write('%d\n' % (m.fathers[ind]))
                    f6.write('%.12e %.12e %.12e %.12e\n' % \
                        (m.sol[ind,0],m.sol[ind,1],m.sol[ind,2],m.sol[ind,3]))
                    if (m.match[ind]==-1):
                        f2.write('%d\n' % (m.match[ind]))
                    else:
                        skip = x_part*nx_loc*ny*nz
                        if (m.coords[ind,2]==m.coords[0,2]):
                            new_match = skip + numnp_part*(z_parts-1) \
                                        + (nz_loc+1) \
                                        + i*(nz_loc+1)*ny \
                                        + j*(nz_loc+1)
                        else:
                            new_match = skip + i*nz_loc*ny \
                                        + j*nz_loc + k-nz_loc +1
                        f2.write('%d\n' % (new_match))
                    inode += 1
                    glob_inode +=1
        
        # Write the element connectivity to file for this part
        f5b.write(' %d\n' % (part_id))
        f5b.write('%d %d\n' % (numel_loc,8))
        n = np.empty([8])
        n[0] = 1
        n[1] = nz_part+1
        n[2] = nz_part+2
        n[3] = 2
        n[4] = nz_part*ny+1
        n[7] = nz_part*ny+2
        n[6] = nz_part*(ny+1)+2
        n[5] = nz_part*(ny+1)+1
        for i in range(numelx_loc):
            for j in range(m.stats[1,1]):
                for k in range(numelz_loc):
                    tmp1 = k+j*nz_part+i*nz_part*ny + offset
                    tmp = n+tmp1
                    # If NOT in last z part AND last z elm in part
                    # 1 fact of elm is on next z part
                    if (z_part<z_parts-1 and k==numelz_loc-1):
                        tmp[3] = offset + numnp_part \
                                 + i*ny*nz_part_next \
                                 + j*nz_part_next+1
                        tmp[2] = tmp[3]+nz_part_next
                        tmp[7] = tmp[3]+nz_part_next*ny
                        tmp[6] = tmp[7]+nz_part_next
                    # If NOT in last x part AND last x elm in part
                    # 1 face of element is on adjacent x part
                    if (x_part<x_parts-1 and i==numelx_loc-1):
                        skip = (x_part+1)*nx_loc*ny*nz
                        skip = skip + z_part*nx_part_next*ny*nz_loc
                        tmp[4] = skip + j*nz_part + (k+1)
                        tmp[5] = tmp[4]+nz_part
                        tmp[6] = tmp[5]+1
                        tmp[7] = tmp[4]+1
                        # If NOT in last z part AND last z elm in part
                        # 2 vertices are on next z part of adjacent x part
                        if (k==numelz_loc-1 and z_part<z_parts-1):
                            skip_again = nx_part_next*nz_part*ny
                            tmp[7] = tmp[4]+skip_again-(nz_part-1)
                            if (z_part+1==z_parts-1):
                                tmp[7] = tmp[7] + j
                            tmp[6] = tmp[7]+nz_part_next
                    f5.write('%d %d %d %d %d %d %d %d\n' % (tmp[0], tmp[1],
                                                    tmp[2], tmp[3], tmp[4],
                                                    tmp[5], tmp[6], tmp[7]))
        node_list_prev = len(node_list)

        # Close all files for this part
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()

    f5b.close()

