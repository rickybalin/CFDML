import numpy as np
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import evtk
import vtk

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
        self.crd = None
        self.model = None
        self.y_pred_glob = None
        self.min_val = np.zeros(6)
        self.max_val = np.zeros(6)

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
            output.GetPointData().RemoveArray("gij")
            output.GetPointData().RemoveArray("GradUFilt")
            output.GetPointData().RemoveArray("GradVFilt")
            output.GetPointData().RemoveArray("GradZFilt")
            self.X = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("input123")),
                                  VN.vtk_to_numpy(output.GetPointData().GetArray("input456"))))
            self.y = np.hstack((VN.vtk_to_numpy(output.GetPointData().GetArray("output123")),
                                  VN.vtk_to_numpy(output.GetPointData().GetArray("output456"))))
            self.crd = VN.vtk_to_numpy(output.GetPoints().GetData())

        if (np.amin(self.y[:,0]) < 0 or np.amax(self.y[:,0]) > 1):
            for i in range(6):
                self.y[:,i] = (self.y[:,i] - self.min_val[i])/(self.max_val[i] - self.min_val[i])
            
        return output


    # Run inference on all data for global metrics
    def test_global(self, X_test=None, y_test=None):
        if X_test and y_test:
            X = X_test; y = y_test
        else:
            X = self.X; y = self.y
        X_torch = torch.from_numpy(np.float32(X))
        self.y_pred_glob = self.model(X_torch).detach().numpy()
        mse_glob = self.MSE(y,self.y_pred_glob)
        cc_glob = self.CC(y,self.y_pred_glob)
        print("Inference on global data:")
        print(f"MSE = {mse_glob:>8e}")
        print(f"Corr. Coefficient = {cc_glob:>8e}")
        print("")
        return mse_glob, cc_glob
    
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
            mse_y[j] = self.MSE(y[index],y_pred_tmp)
            cc_y[j] = self.CC(y[index],y_pred_tmp)
        return crd_y, mse_y, cc_y

    # Deine MSE function
    def MSE(self, y, y_pred):
        return mse_loss(torch.from_numpy(y), torch.from_numpy(y_pred)).numpy()

    # Define Correlation Coefficient function
    def CC(self, y, y_pred):
        return np.corrcoef([np.ndarray.flatten(y),np.ndarray.flatten(y_pred)])[0][1]

    # Save to vtk files for import into Paraview
    def save_vtk(self, polydata):
        from vtk.numpy_interface import dataset_adapter as dsa
        new = dsa.WrapDataObject(polydata)
        output = self.y_pred_glob
        for i in range(6):
            output[:,i] = output[:,i] * (self.max_val[i] - self.min_val[i]) + self.min_val[i]
        new.PointData.append(output[:,:3], "pred_output123")
        new.PointData.append(output[:,3:], "pred_output456")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName("predictions.vtu")
        writer.SetInputData(new.VTKObject)
        writer.Write()


# Offline trained model on 1k flat plate BL data with 3x mesh filter width
data_path = "train_data/FlatPlate_ReTheta1000_6-15_ts29085_1x_clip_noWall_noFS.vtu"
model_path = "./models/rb/NNmodel_1x"
off_bl_1x = SGS(data_path, model_path)
polydata = off_bl_1x.load_VTKdata_model()
mse_glob, cc_glob = off_bl_1x.test_global()
#crd_y, mse_y_off_bl_3x_3x, cc_y_off_bl_3x_3x =  off_bl_3x.test_y_layers()
#off_bl_1x.save_vtk(polydata)


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
