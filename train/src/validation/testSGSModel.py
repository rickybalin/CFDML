import numpy as np
from numpy import linalg as la
import math as m
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import evtk
import vtk
from vtk.util import numpy_support as VN

from computeSGSinputsNoutputs import jacobi

# Define class for the model
class SGS:
    def __init__(self, data_path, model_path, crd_path=None, device="cpu"):
        self.data_path = data_path
        self.crd_path = crd_path
        self.model_path = model_path
        self.device = device
        self.test_data = None
        self.X = None
        self.y = None
        self.SGS = None
        self.crd = None
        self.model = None
        self.y_pred_glob = None
        self.min_val = np.zeros(6)
        self.max_val = np.zeros(6)
        self.eigvecs_aligned = None
        self.SpO = None
        self.Deltaij_norm = None
        self.SGS_GM = None
        self.new_outputs = None

    # Load data and model
    def load_VTKdata_model(self):
        self.model = torch.jit.load(self.model_path+"_jit.pt", 
                                    map_location=torch.device(self.device))
        tmp = np.loadtxt(self.model_path+"_scaling.txt")
        self.min_val = tmp[:,0]
        self.max_val = tmp[:,1]

        extension = self.data_path.split(".")[-1]
        #if "npy" in extension:
        #    test_data = np.load(self.data_path)
        #    self.y = test_data[:,:6]
        #    self.X = test_data[:,6:]
        #    self.crd = np.load(self.crd_path)
        if "vtu" in extension or "vtk" in extension:
            from vtk.util import numpy_support as VN
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(self.data_path)
            reader.PointArrayStatus = ['input123', 'input456', 'output123', 'output456']
            reader.Update()
            output = reader.GetOutput()
            self.X = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("input123")),
                                VN.vtk_to_numpy(output.GetPointData().GetArray("input456"))))
            self.y = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("output123")),
                                VN.vtk_to_numpy(output.GetPointData().GetArray("output456"))))
            self.SGS = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("SGS_diag")),
                                  VN.vtk_to_numpy(output.GetPointData().GetArray("SGS_offdiag"))))
            self.crd = VN.vtk_to_numpy(output.GetPoints().GetData())
        return output

    # Run inference on all data for global metrics
    def test_global(self, X_test=None, y_test=None):
        if X_test and y_test:
            X = X_test; y = y_test
        else:
            X = self.X; y = self.y; SGS = self.SGS
        X_torch = torch.from_numpy(np.float32(X))
        self.y_pred_glob = self.model(X_torch).detach().numpy()
        self.y_pred_glob = self.undo_min_max(self.y_pred_glob)
        mse_glob = self.MSE(y,self.y_pred_glob)
        cc_glob = self.CC(y,self.y_pred_glob)
        print("Inference on global data:")
        print(f"Output MSE = {mse_glob:>8e}")
        print(f"Output Corr. Coefficient = {cc_glob:>8e}")
        self.SGS_pred_glob = self.compute_SGS(self.y_pred_glob)
        mse_glob = self.MSE(SGS,self.SGS_pred_glob)
        cc_glob = self.CC(SGS,self.SGS_pred_glob)
        print(f"SGS MSE = {mse_glob:>8e}")
        print(f"SGS Corr. Coefficient = {cc_glob:>8e}")
        print("")
    
    # Run inference one 1 wall-parallel layer at a time
    def test_y_layers(self, X_test=None, y_test=None, crd_test=None):
        if X_test and y_test and crd_test:
            X = X_test; y = y_test; crd = crd_test
        else:
            X = self.X; y = self.y; crd = self.crd
        crd_y = np.unique(crd[:,1])
        ny = crd_y.size
        mse_y = np.zeros_like(crd_y)
        cc_y = np.zeros_like(crd_y)
        for j in range(ny):
            index = np.where(crd[:,1]==crd_y[j])
            X_tmp = torch.from_numpy(np.float32(X[index]))
            y_pred_tmp = self.model(X_tmp).detach().numpy()
            y_pred_tmp = self.undo_min_max(y_pred_tmp)
            mse_y[j] = self.MSE(y[index],y_pred_tmp)
            cc_y[j] = self.CC(y[index],y_pred_tmp)
        return crd_y, mse_y, cc_y

    # Undo the model's min-max scaling
    def undo_min_max(self,y):
        for i in range(6):
            y[:,i] = y[:,i] * (self.max_val[i] - self.min_val[i]) + self.min_val[i]
        return y

    # Deine MSE function
    def MSE(self, y, y_pred):
        return mse_loss(torch.from_numpy(y), torch.from_numpy(y_pred)).numpy()

    # Define Correlation Coefficient function
    def CC(self, y, y_pred):
        return np.corrcoef([np.ndarray.flatten(y),np.ndarray.flatten(y_pred)])[0][1]

    # Align the eigenvalues and eignevectors according to the local vorticity
    def align_tensors(self,evals,evecs,vec):
        if (evals[0]<1.0e-8 and evals[1]<1.0e-8 and evals[2]<1.0e-8):
            index = [0,1,2]
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

    # Compute the transformation needed to obtain the physical stresses
    def compute_transformation(self, polydata, scaling):
        GradU = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradUFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradVFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradZFilt"))))
        GradU = np.reshape(GradU, (-1,3,3))
        Delta = VN.vtk_to_numpy(polydata.GetPointData().GetArray("gij"))
        nsamples = GradU.shape[0]
        self.eigvecs_aligned = np.zeros((nsamples,3,3))
        self.SpO = np.zeros((nsamples,))
        self.Deltaij_norm = np.zeros((nsamples,))
        self.new_outputs = np.zeros((nsamples,6))
        Deltaij = np.zeros((3,3))
        Sij = np.zeros((3,3))
        Oij = np.zeros((3,3))
        vort = np.zeros((3,))
        tmp = np.zeros((3,3))
        for i in range(nsamples):
            Deltaij[0,0] = Delta[i,0]*scaling[0]
            Deltaij[1,1] = Delta[i,1]*scaling[1]
            Deltaij[2,2] = Delta[i,2]*scaling[2]
            self.Deltaij_norm[i] = m.sqrt(Deltaij[0,0]**2 + Deltaij[1,1]**2 + Deltaij[2,2]**2)
            Deltaij = Deltaij / (self.Deltaij_norm[i]+1.0e-14)
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
            Sij_norm = m.sqrt(Sij[0,0]**2+Sij[1,1]**2+Sij[2,2]**2 \
                          + 2*(Sij[0,1]**2+Sij[0,2]**2+Sij[1,2]**2))
            vort_norm = m.sqrt(vort[0]**2 + vort[1]**2 + vort[2]**2)
            self.SpO[i] = Sij_norm**2 + 0.5*vort_norm**2
            #evals, evecs = jacobi(Sij) 
            evals, evecs = la.eig(Sij)
            vec = vort.copy()
            #vec = np.array([0,1,0])
            lda, eigvecs, self.eigvecs_aligned[i] = self.align_tensors(evals,evecs,vec)
            tmp[0,0] = SGS[i,0] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[1,1] = SGS[i,1] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[2,2] = SGS[i,2] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[0,1] = SGS[i,3] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[0,2] = SGS[i,4] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[1,2] = SGS[i,5] / (self.Deltaij_norm[i]**2 * self.SpO[i] + 1.0e-14)
            tmp[1,0] = tmp[0,1]
            tmp[2,0] = tmp[0,2]
            tmp[2,1] = tmp[1,2]
            tmp = np.matmul(np.transpose(self.eigvecs_aligned[i]),
                           np.matmul(tmp,self.eigvecs_aligned[i]))
            self.new_outputs[i,0] = tmp[0,0]
            self.new_outputs[i,1] = tmp[1,1]
            self.new_outputs[i,2] = tmp[2,2]
            self.new_outputs[i,3] = tmp[0,1]
            self.new_outputs[i,4] = tmp[0,2]
            self.new_outputs[i,5] = tmp[1,2]


    # Compute physical stresses
    def compute_SGS(self,y,index=None):
        tmp = np.zeros((3,3))
        nsamples = y.shape[0]
        SGS_pred = np.zeros((nsamples,6))
        for i in range(nsamples):
            tmp[0,0] = y[i,0]
            tmp[1,1] = y[i,1]
            tmp[2,2] = y[i,2]
            tmp[0,1] = y[i,3]
            tmp[0,2] = y[i,4]
            tmp[1,2] = y[i,5]
            tmp[1,0] = tmp[0,1]
            tmp[2,0] = tmp[0,2]
            tmp[2,1] = tmp[1,2]
            tmp = np.matmul(self.eigvecs_aligned[i],tmp)
            tmp = np.matmul(tmp,np.transpose(self.eigvecs_aligned[i]))
            SGS_pred[i,0] = tmp[0,0] * (self.Deltaij_norm[i]**2 * self.SpO[i])
            SGS_pred[i,1] = tmp[1,1] * (self.Deltaij_norm[i]**2 * self.SpO[i])
            SGS_pred[i,2] = tmp[2,2] * (self.Deltaij_norm[i]**2 * self.SpO[i])
            SGS_pred[i,3] = tmp[0,1] * (self.Deltaij_norm[i]**2 * self.SpO[i])
            SGS_pred[i,4] = tmp[0,2] * (self.Deltaij_norm[i]**2 * self.SpO[i])
            SGS_pred[i,5] = tmp[1,2] * (self.Deltaij_norm[i]**2 * self.SpO[i])
        return SGS_pred

    # Compute the stress with the Gradient Model
    def gradient_SGS_model(self,polydata):
        GradU = np.hstack((VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradUFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradVFilt")),
                            VN.vtk_to_numpy(polydata.GetPointData().GetArray("GradZFilt"))))
        GradU = np.reshape(GradU, (-1,3,3))
        Delta = VN.vtk_to_numpy(polydata.GetPointData().GetArray("gij"))
        nsamples = GradU.shape[0]
        self.SGS_GM = np.zeros((nsamples,6))
        for i in range(nsamples):
            self.SGS_GM[i,0] = (Delta[i,0]**2 * GradU[i,0,0]**2 + \
                                Delta[i,1]**2 * GradU[i,0,1]**2 + \
                                Delta[i,2]**2 * GradU[i,0,2]**2) / 12 # 11
            self.SGS_GM[i,1] = (Delta[i,0]**2 * GradU[i,1,0]**2 + \
                                Delta[i,1]**2 * GradU[i,1,1]**2 + \
                                Delta[i,2]**2 * GradU[i,1,2]**2) / 12 # 22
            self.SGS_GM[i,2] = (Delta[i,0]**2 * GradU[i,2,0]**2 + \
                                Delta[i,1]**2 * GradU[i,2,1]**2 + \
                                Delta[i,2]**2 * GradU[i,2,2]**2) / 12 # 33
            self.SGS_GM[i,3] = (Delta[i,0]**2 * GradU[i,0,0]*GradU[i,1,0] + \
                                Delta[i,1]**2 * GradU[i,0,1]*GradU[i,1,1] + \
                                Delta[i,2]**2 * GradU[i,0,2]*GradU[i,1,2]) / 12 # 12
            self.SGS_GM[i,4] = (Delta[i,0]**2 * GradU[i,0,0]*GradU[i,2,0] + \
                                Delta[i,1]**2 * GradU[i,0,1]*GradU[i,2,1] + \
                                Delta[i,2]**2 * GradU[i,0,2]*GradU[i,2,2]) / 12 # 13
            self.SGS_GM[i,5] = (Delta[i,0]**2 * GradU[i,1,0]*GradU[i,2,0] + \
                                Delta[i,1]**2 * GradU[i,1,1]*GradU[i,2,1] + \
                                Delta[i,2]**2 * GradU[i,1,2]*GradU[i,2,2]) / 12 # 23
        
    # Save to vtk files for import into Paraview
    def save_vtk(self, polydata, fname):
        from vtk.numpy_interface import dataset_adapter as dsa
        new = dsa.WrapDataObject(polydata)
        new.PointData.append(self.y_pred_glob[:,:3], "pred_output123")
        new.PointData.append(self.y_pred_glob[:,3:], "pred_output456")
        new.PointData.append(self.SGS_pred_glob[:,:3], "pred_SGS_diag")
        new.PointData.append(self.SGS_pred_glob[:,3:], "pred_SGS_offdiag")
        if (self.new_outputs!=None):
            new.PointData.append(self.new_outputs[:,:3], "new_output123")
            new.PointData.append(self.new_outputs[:,3:], "new_output456")
        if (self.SGS_GM!=None):
            new.PointData.append(self.SGS_pred_glob[:,:3], "SGSGM_diag")
            new.PointData.append(self.SGS_pred_glob[:,3:], "SGSGM_offdiag")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(new.VTKObject)
        writer.Write()




# Baseline model
data_path = "train_data/FlatPlate_ReTheta1000_6-15_ts30005_3x_noDamp_jacobi_test.vtu"
model_path = "./models/3x/NNmodel"
base = SGS(data_path, model_path)
polydata = base.load_VTKdata_model()
base.compute_transformation(polydata, [3, 3, 3])
base.test_global()
base.gradient_SGS_model(polydata)
#crd_y, mse_y_off_bl_3x_3x, cc_y_off_bl_3x_3x =  off_bl_3x.test_y_layers()
base.save_vtk(polydata,model_path+"_predictions.vtu")

# Wall aligned model
data_path = "train_data/FlatPlate_ReTheta1000_6-15_ts30005_3x_noDamp_jacobi_test.vtu"
model_path = "./models/3x/py_inputs/NNmodel_wall"
wall = SGS(data_path, model_path)
polydata = wall.load_VTKdata_model()
wall.compute_transformation(polydata, [3, 3, 3])
wall.test_global()
wall.gradient_SGS_model(polydata)
#crd_y, mse_y_off_bl_3x_3x, cc_y_off_bl_3x_3x =  off_bl_3x.test_y_layers()
wall.save_vtk(polydata,model_path+"_predictions.vtu")


"""
# Offline trained model on 1k flat plate BL data with 3x mesh filter width
data_path = "./3x/data/train_data_3x.npy"
crd_path = "./3x/data/crd_data_3x.npy"
model_path = "./NNmodel_HIT_Aviral_jit.pt"
device = "cpu"
off_hit = SGS(data_path, crd_path, model_path, device, vtk_export=True)
X_off_hit, y_off_hit = off_hit.load_data_model()
y_pred_off_hit, mse_glob, cc_glob = off_hit.test_global(X_off_hit, y_off_hit)
crd_y, mse_y_off_hit, cc_y_off_hit =  off_hit.test_y_layers(X_off_hit,y_off_hit)
"""
"""
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
axs[0].plot(crd_y, mse_y_off_bl_3x_3x, 's', label="off_bl_3x_3x")
axs[0].plot(crd_y, mse_y_off_bl_6x_6x, 'o', label="off_bl_6x_6x")
axs[0].plot(crd_y, mse_y_off_bl_3x_6x, 'o', label="off_bl_3x_6x")
axs[1].plot(crd_y, cc_y_off_bl_3x_3x, 's', label="off_bl_3x")
axs[1].plot(crd_y, cc_y_off_bl_6x_6x, 'o', label="off_bl_6x")
axs[0].set_yscale("log")
axs[0].set_xscale("log")
#axs[1].set_yscale("log")
axs[1].set_xscale("log")
axs[0].grid()
axs[1].grid()
fig.tight_layout(pad=3.0)
axs[0].set_ylabel('MSE')
axs[0].set_xlabel('y')
axs[0].set_title('Mean Squared Error')
axs[0].legend()
axs[1].set_ylabel('Correlation Coefficient')
axs[1].set_xlabel('y')
axs[1].set_title('Correlation Coefficient')
axs[1].legend()
#plt.savefig(fig_name+"_yerrors.png", dpi='figure', format="png")
plt.show()

"""
