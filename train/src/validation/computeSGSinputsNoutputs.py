# Compute the inputs and outputs for the anisotrpic SGS model
# Requires input file with the filtered velocity gradients and the filter width

import numpy as np
from numpy import linalg as la
import math as m
import vtk

## Load VTK data
def load_VTKdata(fname):
    extension = fname.split(".")[-1]
    #if "npy" in extension:
    #    test_data = np.load(self.data_path)
    #    self.y = test_data[:,:6]
    #    self.X = test_data[:,6:]
    #    self.crd = np.load(self.crd_path)
    if "vtu" in extension or "vtk" in extension:
        from vtk.util import numpy_support as VN
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        output = reader.GetOutput()
        GradU = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("GradUFilt")),
                            VN.vtk_to_numpy(output.GetPointData().GetArray("GradVFilt")),
                            VN.vtk_to_numpy(output.GetPointData().GetArray("GradZFilt"))))
        GradU = np.reshape(GradU, (-1,3,3))
        Delta = VN.vtk_to_numpy(output.GetPointData().GetArray("gij"))
        SGS = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("SGS_diag")),
                            VN.vtk_to_numpy(output.GetPointData().GetArray("SGS_offdiag"))))
        return output, GradU, Delta, SGS

## Write VTK data
def write_VTKdata(fname, polydata, Gij, Sij, Oij, vort, lda, eigvecs,
                  eigvecs_aligned, vort_Sframe, inputs, outputs):
    extension = fname.split(".")[-1]
    if "vtu" in extension or "vtk" in extension:
        from vtk.numpy_interface import dataset_adapter as dsa
        new = dsa.WrapDataObject(polydata)
        new.PointData.append(Gij[:,0,:], "Gij_u")
        new.PointData.append(Gij[:,1,:], "Gij_v")
        new.PointData.append(Gij[:,2,:], "Gij_w")
        new.PointData.append(Sij[:,0,:], "Sij_1")
        new.PointData.append(Sij[:,1,:], "Sij_2")
        new.PointData.append(Sij[:,2,:], "Sij_3")
        new.PointData.append(Oij[:,0,:], "Oij_1")
        new.PointData.append(Oij[:,1,:], "Oij_2")
        new.PointData.append(Oij[:,2,:], "Oij_3")
        new.PointData.append(vort, "vorticity")
        new.PointData.append(lda, "eigval")
        new.PointData.append(eigvecs[:,:,0], "eigvec_1")
        new.PointData.append(eigvecs[:,:,1], "eigvec_2")
        new.PointData.append(eigvecs[:,:,2], "eigvec_3")
        new.PointData.append(eigvecs_aligned[:,:,0], "eigvec_a_1")
        new.PointData.append(eigvecs_aligned[:,:,1], "eigvec_a_2")
        new.PointData.append(eigvecs_aligned[:,:,2], "eigvec_a_3")
        new.PointData.append(vort_Sframe, "vorticity_Sframe")
        new.PointData.append(inputs[:,:3], "input123_py")
        new.PointData.append(inputs[:,3:], "input456_py")
        new.PointData.append(outputs[:,:3], "output123_py")
        new.PointData.append(outputs[:,3:], "output456_py")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(new.VTKObject)
        writer.Write()

## Compute eigenvalues and eigenvectors with the Jacobi iterative method from PHASTA
def jacobi(A):
    n = 3
    ic = 0
    icMax = 100
    abserr = 1.0e-10

    V = np.zeros((n,n))
    E = A.copy()
    for i in range(n):
        V[i,i] = 1
    b2 = 0.0
    for i in range(3):
        for j in range(3):
            if (i!=j):
                b2 = b2 + E[i,j]**2
    if (b2 <= abserr): 
        return np.array([E[0,0],E[1,1],E[2,2]]), V
      
    bar = 0.5*b2/n**2
    while ((b2>abserr) and (ic<=icMax)):
        for i in range(n-1):
            for j in range(i+1,n):
                if (E[j,i]**2 <= bar): continue
                b2 = b2 - 2.0*E[j,i]**2
                bar = 0.5*b2/n**2
                beta = (E[j,j]-E[i,i])/(2.0*E[j,i])
                coeff = 0.5*beta/m.sqrt(1.0+beta**2)
                s = m.sqrt(np.maximum(0.5+coeff,0.0))
                c = m.sqrt(np.maximum(0.5-coeff,0.0))
                for k in range(3):
                    cs =  c*E[i,k]+s*E[j,k]
                    sc = -s*E[i,k]+c*E[j,k]
                    E[i,k] = cs
                    E[j,k] = sc 
                for k in range(3):
                    cs =  c*E[k,i]+s*E[k,j]
                    sc = -s*E[k,i]+c*E[k,j]
                    E[k,i] = cs
                    E[k,j] = sc
                    cs =  c*V[k,i]+s*V[k,j]
                    sc = -s*V[k,i]+c*V[k,j]
                    V[k,i] = cs
                    V[k,j] = sc
        ic = ic+1
    if (ic>icMax): print("WARNING: max iterations in Jocobi")
    return np.array([E[0,0],E[1,1],E[2,2]]), V

## Align the eigenvalues and eignevectors according to the local vorticity
def align_tensors(evals,evecs,vec):
    if (evals[0]<1.0e-8 and evals[1]<1.0e-8 and evals[2]<1.0e-8):
        index = [0,1,2]
        print("here")
    else:
        index = np.flip(np.argsort(evals))
    lda = evals[index]

    vec = np.array([0, 1, 0])
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

## Main
def main():
    # Load data from .vtu file
    dir = "/Users/rbalin/Documents/Research/ALCF_PostDoc/Conferences/PASC23/FlatPlate/Train/CRS_6-15_4d/"
    fname = dir + "train_data/FlatPlate_ReTheta1000_6-15_ts30005_3x_noDamp_jacobi_test.vtu"
    polydata, GradU, Delta, SGS = load_VTKdata(fname)
    nsamples = GradU.shape[0]

    # Initialize new arrays
    Deltaij = np.zeros((nsamples,3,3))
    Gij = np.zeros((nsamples,3,3))
    Sij = np.zeros((nsamples,3,3))
    Oij = np.zeros((nsamples,3,3))
    vort = np.zeros((nsamples,3))
    lda = np.zeros((nsamples,3))
    eigvecs = np.zeros((nsamples,3,3))
    eigvecs_aligned = np.zeros((nsamples,3,3))
    vort_Sframe = np.zeros((nsamples,3))
    inputs = np.zeros((nsamples,6))
    tmp = np.zeros((nsamples,3,3))
    outputs = np.zeros((nsamples,6))

    # Loop over number of grid points and compute model inputs and outputs
    scaling = [3, 3, 3]
    eps = 1.0e-14
    nu = 1.25e-5
    for i in range(nsamples):
        Deltaij[i,0,0] = Delta[i,0]*scaling[0]
        Deltaij[i,1,1] = Delta[i,1]*scaling[1]
        Deltaij[i,2,2] = Delta[i,2]*scaling[2]
        Deltaij_norm = m.sqrt(Deltaij[i,0,0]**2 + Deltaij[i,1,1]**2 + Deltaij[i,2,2]**2)
        Deltaij[i] = Deltaij[i] / (Deltaij_norm+eps)

        Gij[i] = np.matmul(GradU[i],Deltaij[i])
        Sij[i,0,0] = Gij[i,0,0]
        Sij[i,1,1] = Gij[i,1,1]
        Sij[i,2,2] = Gij[i,2,2]
        Sij[i,0,1] = 0.5*(Gij[i,0,1]+Gij[i,1,0])
        Sij[i,0,2] = 0.5*(Gij[i,0,2]+Gij[i,2,0])
        Sij[i,1,2] = 0.5*(Gij[i,1,2]+Gij[i,2,1])
        Sij[i,1,0] = Sij[i,0,1]
        Sij[i,2,0] = Sij[i,0,2]
        Sij[i,2,1] = Sij[i,1,2]
        Oij[i,0,1] = 0.5*(Gij[i,0,1]-Gij[i,1,0])
        Oij[i,0,2] = 0.5*(Gij[i,0,2]-Gij[i,2,0])
        Oij[i,1,2] = 0.5*(Gij[i,1,2]-Gij[i,2,1])
        Oij[i,1,0] = -Oij[i,0,1]
        Oij[i,2,0] = -Oij[i,0,2]
        Oij[i,2,1] = -Oij[i,1,2]
        vort[i,0] = -2*Oij[i,1,2]
        vort[i,1] = -2*Oij[i,0,2]
        vort[i,2] = -2*Oij[i,0,1]

        #evals, evecs = la.eig(Sij[i])
        evals, evecs = jacobi(Sij[i])
        lda[i], eigvecs[i], eigvecs_aligned[i] = align_tensors(evals,evecs,vort[i])

        Sij_norm = m.sqrt(Sij[i,0,0]**2+Sij[i,1,1]**2+Sij[i,2,2]**2 \
                          + 2*(Sij[i,0,1]**2+Sij[i,0,2]**2+Sij[i,1,2]**2))
        vort_norm = m.sqrt(vort[i,0]**2 + vort[i,1]**2 + vort[i,2]**2)
        SpO = Sij_norm**2 + 0.5*vort_norm**2

        vort_Sframe[i,0] = np.dot(vort[i],eigvecs_aligned[i,:,0])
        vort_Sframe[i,1] = np.dot(vort[i],eigvecs_aligned[i,:,1])
        vort_Sframe[i,2] = np.dot(vort[i],eigvecs_aligned[i,:,2])
        inputs[i,0] = lda[i,0] / (m.sqrt(SpO)+eps)
        inputs[i,1] = lda[i,1] / (m.sqrt(SpO)+eps)
        inputs[i,2] = lda[i,2] / (m.sqrt(SpO)+eps)
        inputs[i,3] = vort_Sframe[i,0] / (m.sqrt(SpO)+eps)
        inputs[i,4] = vort_Sframe[i,1] / (m.sqrt(SpO)+eps)
        inputs[i,5] = nu / (Deltaij_norm**2 * m.sqrt(SpO) + eps)

        tmp[i,0,0] = SGS[i,0] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,1,1] = SGS[i,1] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,2,2] = SGS[i,2] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,0,1] = SGS[i,3] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,0,2] = SGS[i,4] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,1,2] = SGS[i,5] / (Deltaij_norm**2 * SpO + eps)
        tmp[i,1,0] = tmp[i,0,1]
        tmp[i,2,0] = tmp[i,0,2]
        tmp[i,2,1] = tmp[i,1,2]
        #tmp[i] = np.matmul(tmp[i],eigvecs_aligned[i])
        tmp[i] = np.matmul(np.transpose(eigvecs_aligned[i]),
                           np.matmul(tmp[i],eigvecs_aligned[i]))
        outputs[i,0] = tmp[i,0,0]
        outputs[i,1] = tmp[i,1,1]
        outputs[i,2] = tmp[i,2,2]
        outputs[i,3] = tmp[i,0,1]
        outputs[i,4] = tmp[i,0,2]
        outputs[i,5] = tmp[i,1,2]

    # Write the computed fields to file
    fname = dir + "predictions.vtu"
    write_VTKdata(fname, polydata, Gij, Sij, Oij, vort, lda, 
                  eigvecs, eigvecs_aligned, vort_Sframe, inputs, outputs)


## Run Main
if __name__ == "__main__":
    main()