##### 
##### This script contains general utilities that can be useful
##### to many training applications and driver scripts 
#####

from os.path import exists
import sys
import numpy as np
from numpy import linalg as la
import math as m
import vtk
from vtk.util import numpy_support as VN

### MPI Communicator class
class MPI_COMM:
    def __init__(self):
        self.comm = None
        self.size = None
        self.rank = None
        self.name = None
        self.rankl = None
        self.sum = None
        self.minloc = None
        self.maxloc = None

    def init(self, cfg, print_hello=False):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()
        self.rankl = self.rank % cfg.run_args.mlprocs_pn
        self.sum = MPI.SUM
        self.minloc = MPI.MINLOC
        self.maxloc = MPI.MAXLOC
        if print_hello:
            print(f"Hello from MPI rank {self.rank}/{self.size}")
            sys.stdout.flush()

### Horovod Communicator class
class HVD_COMM:
    def __init__(self):
        self.rank = None
        self.size = None
        self.rankl = None

    def init(self, print_hello=False):
        import horovod.torch as hvd
        hvd.init()
        self.rank = hvd.rank()
        self.size = hvd.size()
        self.rankl = hvd.local_rank()
        if print_hello:
            print(f"Hello from HVD rank {self.rank}/{self.size}")
            sys.stdout.flush()

### Compute the average of a quantity across all ranks
def metric_average(comm, val):
    avg_val = comm.comm.allreduce(val, op=comm.sum)
    avg_val = avg_val / comm.size
    return avg_val


### Compute the correlation coefficient between predicted and target outputs
def comp_corrCoeff(output_tensor, target_tensor):
    target = target_tensor.numpy()
    target = np.ndarray.flatten(target)
    output = output_tensor.detach().numpy()
    output = np.ndarray.flatten(output)
    corrCoeff = np.corrcoef([output,target])
    return corrCoeff


### Set the model weights and biases to fixed value for reproducibility
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1: 
        # apply a fixed value to the weights and a set bias=0
        m.weight.data.fill_(0.5)
        m.bias.data.fill_(0)


### Load training data from file or create synthetic data
def load_data(cfg, rng):
    if (cfg.train.data_path == "synthetic"):
        samples = 10 * cfg.train.mini_batch
        if (cfg.train.model == 'sgs'):
            data = np.float32(rng.normal(size=(samples,12)))
            mesh = None
        elif ("qcnn" in cfg.train.model):
            N = 32
            data = np.float32(rng.normal(size=(samples,cfg.train.channels,N**3)))
            mesh = np.zeros((N**3,3), dtype=np.float32)
            for i in range(N):
                x = 0. + 1. * (i - 1) / (N - 1)
                for j in range(N):
                    y = 0. + 1. * (j - 1) / (N - 1)
                    for k in range(N):
                        z = 0. + 1. * (k - 1) / (N - 1)
                        ind = i*N**2 + j*N +k
                        mesh[ind,0] = x
                        mesh[ind,1] = y
                        mesh[ind,2] = z
    else:
        extension = cfg.train.data_path.split(".")[-1]
        if "npy" in extension:
            data = np.float32(np.load(cfg.train.data_path))
        elif "vtu" in extension or "vtk" in extension:
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(cfg.train.data_path)
            reader.Update()
            polydata = reader.GetOutput()
            if (cfg.train.model == 'sgs'):
                if not cfg.train.comp_model_ins_outs:
                    features = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("input123_py")),
                                  VN.vtk_to_numpy(polydata.GetPointData().GetArray("input456_py"))))
                    targets = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("output123_py")),
                                  VN.vtk_to_numpy(polydata.GetPointData().GetArray("output456_py"))))
                else:
                    features, targets = comp_ins_outs_SGS(polydata)
                data = np.hstack((targets,features))
        
        # Model specific data loading and manipulation
        mesh = None
        if (cfg.train.model=='sgs'):
            if (np.amin(data[:,0]) < 0 or np.amax(data[:,0]) > 1):
                with open(cfg.train.name+"_scaling.dat", "w") as fh:
                    for i in range(6):
                        min_val = np.amin(data[:,i])
                        max_val = np.amax(data[:,i])
                        fh.write(f"{min_val:>8e} {max_val:>8e}\n")
                        data[:,i] = (data[:,i] - min_val)/(max_val - min_val)
        elif ("qcnn" in cfg.train.model):
            extension = cfg.train.mesh_file.split(".")[-1]
            if "npy" in extension:
                mesh = np.float32(np.load(cfg.train.mesh_file))
            with open(cfg.train.name+"_scaling.dat", "w") as fh:
                for i in range(4):
                    min_val = np.amin(data[:,i,:])
                    max_val = np.amax(data[:,i,:])
                    fh.write(f"{min_val:>8e} {max_val:>8e}\n")
                    data[:,i,:] = 1.0*(data[:,i,:] - min_val)/(max_val - min_val) - 0.0
                
    return data, mesh


### Compute the inputs and outputs for the anisotropic SGS model from raw data
def comp_ins_outs_SGS(polydata, alignment="vorticity"):
    # Read raw data from file
    GradU = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradUFilt")),
                        VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradVFilt")),
                        VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradZFilt"))))
    GradU = np.reshape(GradU, (-1,3,3))
    Delta = VN.vtk_to_numpy(polydata.GetPointData().GetArray("gij"))
    SGS = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("SGS_diag")),
                     VN.vtk_to_numpy(polydata.GetPointData().GetArray("SGS_offdiag"))))
    
    # Initialize new arrays
    nsamples = GradU.shape[0]
    Deltaij = np.zeros((3,3))
    Gij = np.zeros((3,3))
    Sij = np.zeros((3,3))
    Oij = np.zeros((3,3))
    vort = np.zeros((3))
    lda = np.zeros((3))
    eigvecs = np.zeros((3,3))
    eigvecs_aligned = np.zeros((3,3))
    vort_Sframe = np.zeros((3))
    inputs = np.zeros((nsamples,6))
    tmp = np.zeros((3,3))
    outputs = np.zeros((nsamples,6))

    # Loop over number of grid points and compute model inputs and outputs
    scaling = [3, 3, 3]
    eps = 1.0e-14
    nu = 1.25e-5
    for i in range(nsamples):
        Deltaij[0,0] = Delta[i,0]*scaling[0]
        Deltaij[1,1] = Delta[i,1]*scaling[1]
        Deltaij[2,2] = Delta[i,2]*scaling[2]
        Deltaij_norm = m.sqrt(Deltaij[0,0]**2 + Deltaij[1,1]**2 + Deltaij[2,2]**2)
        Deltaij = Deltaij / (Deltaij_norm+eps)

        Gij = np.matmul(GradU[i],Deltaij)
        Sij[0,0] = Gij[0,0]
        Sij[1,1] = Gij[1,1]
        Sij[2,2] = Gij[2,2]
        Sij[0,1] = 0.5*(Gij[0,1]+Gij[1,0])
        Sij[0,2] = 0.5*(Gij[0,2]+Gij[2,0])
        Sij[1,2] = 0.5*(Gij[1,2]+Gij[2,1])
        Sij[1,0] = Sij[0,1]
        Sij[2,0] = Sij[0,2]
        Sij[2,1] = Sij[1,2]
        Oij[0,1] = 0.5*(Gij[0,1]-Gij[1,0])
        Oij[0,2] = 0.5*(Gij[0,2]-Gij[2,0])
        Oij[1,2] = 0.5*(Gij[1,2]-Gij[2,1])
        Oij[1,0] = -Oij[0,1]
        Oij[2,0] = -Oij[0,2]
        Oij[2,1] = -Oij[1,2]
        vort[0] = -2*Oij[1,2]
        vort[1] = -2*Oij[0,2]
        vort[2] = -2*Oij[0,1]

        evals, evecs = la.eig(Sij)
        if (alignment=="vorticity"):
            vec = vort.copy()
        elif (alignment=="wall-normal"):
            vec = np.array([0,1,0])
        else:
            print("Alignment option not known, used default vorticity alignment")
            vec = vort.copy()
        lda, eigvecs, eigvecs_aligned = align_tensors(evals,evecs,vec)

        Sij_norm = m.sqrt(Sij[0,0]**2+Sij[1,1]**2+Sij[2,2]**2 \
                          + 2*(Sij[0,1]**2+Sij[0,2]**2+Sij[1,2]**2))
        vort_norm = m.sqrt(vort[0]**2 + vort[1]**2 + vort[2]**2)
        SpO = Sij_norm**2 + 0.5*vort_norm**2

        vort_Sframe[0] = np.dot(vort,eigvecs_aligned[:,0])
        vort_Sframe[1] = np.dot(vort,eigvecs_aligned[:,1])
        vort_Sframe[2] = np.dot(vort,eigvecs_aligned[:,2])
        inputs[i,0] = lda[0] / (m.sqrt(SpO)+eps)
        inputs[i,1] = lda[1] / (m.sqrt(SpO)+eps)
        inputs[i,2] = lda[2] / (m.sqrt(SpO)+eps)
        inputs[i,3] = vort_Sframe[0] / (m.sqrt(SpO)+eps)
        inputs[i,4] = vort_Sframe[1] / (m.sqrt(SpO)+eps)
        inputs[i,5] = nu / (Deltaij_norm**2 * m.sqrt(SpO) + eps)

        tmp[0,0] = SGS[i,0] / (Deltaij_norm**2 * SpO + eps)
        tmp[1,1] = SGS[i,1] / (Deltaij_norm**2 * SpO + eps)
        tmp[2,2] = SGS[i,2] / (Deltaij_norm**2 * SpO + eps)
        tmp[0,1] = SGS[i,3] / (Deltaij_norm**2 * SpO + eps)
        tmp[0,2] = SGS[i,4] / (Deltaij_norm**2 * SpO + eps)
        tmp[1,2] = SGS[i,5] / (Deltaij_norm**2 * SpO + eps)
        tmp[1,0] = tmp[0,1]
        tmp[2,0] = tmp[0,2]
        tmp[2,1] = tmp[1,2]
        tmp = np.matmul(np.transpose(eigvecs_aligned),
                           np.matmul(tmp,eigvecs_aligned))
        outputs[i,0] = tmp[0,0]
        outputs[i,1] = tmp[1,1]
        outputs[i,2] = tmp[2,2]
        outputs[i,3] = tmp[0,1]
        outputs[i,4] = tmp[0,2]
        outputs[i,5] = tmp[1,2]
    
    return inputs, outputs


### Align the eigenvalues and eignevectors according to the local vector (used by comp_ins_outs_SGS)
def align_tensors(evals,evecs,vec):
    if (evals[0]<1.0e-8 and evals[1]<1.0e-8 and evals[2]<1.0e-8):
        index = [0,1,2]
        print("here")
    else:
        index = np.flip(np.argsort(evals))
    lda = evals[index]

    vec_norm = m.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    vec = vec/vec_norm

    eigvec = np.zeros((3,3))
    eigvec[:,0] = evecs[:,index[0]]
    eigvec[:,1] = evecs[:,index[1]]
    eigvec[:,2] = evecs[:,index[2]]

    eigvec_vort_aligned = eigvec.copy()
    if (np.dot(vec,eigvec_vort_aligned[:,0]) < np.dot(vec,-eigvec_vort_aligned[:,0])):
        eigvec_vort_aligned[:,0] = -eigvec_vort_aligned[:,0]
    if (np.dot(vec,eigvec_vort_aligned[:,2]) < np.dot(vec,-eigvec_vort_aligned[:,2])):
        eigvec_vort_aligned[:,2] = -eigvec_vort_aligned[:,2]
    eigvec_vort_aligned[0,1] = (eigvec_vort_aligned[1,2]*eigvec_vort_aligned[2,0]) \
                               - (eigvec_vort_aligned[2,2]*eigvec_vort_aligned[1,0])
    eigvec_vort_aligned[1,1] = (eigvec_vort_aligned[2,2]*eigvec_vort_aligned[0,0]) \
                               - (eigvec_vort_aligned[0,2]*eigvec_vort_aligned[2,0])
    eigvec_vort_aligned[2,1] = (eigvec_vort_aligned[0,2]*eigvec_vort_aligned[1,0]) \
                               - (eigvec_vort_aligned[1,2]*eigvec_vort_aligned[0,0])

    return lda, eigvec, eigvec_vort_aligned
