import numpy as np
import torch
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
import evtk

# Define class for the model
class SGS:
    def __init__(self, data_path, crd_path, model_path, device, vtk_export=False):
        self.data_path = data_path
        self.crd_path = crd_path
        self.model_path = model_path
        self.device = device
        self.vtk_export = vtk_export
        self.test_data = None
        self.X = None
        self.y = None
        self.crd = None
        self.model = None
        self.y_pred_glob = None

    # Load data and model
    def load_data_model(self):
        test_data = np.load(self.data_path)
        self.y = test_data[:,:6]
        self.X = test_data[:,6:]
        self.crd = np.load(self.crd_path)
        self.model = torch.jit.load(self.model_path, 
                                    map_location=torch.device(self.device))

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
    def save_vtk(self, y, y_pred):
        if self.vtk_export:
            points_x = np.ascontiguousarray(self.crd[:,0])
            points_y = np.ascontiguousarray(self.crd[:,1])
            points_z = np.ascontiguousarray(self.crd[:,2])

            true_11 = np.ascontiguousarray(y[:,0])
            pred_11 = np.ascontiguousarray(y_pred[:,0])

            evtk.hl.pointsToVTK("./predictions", points_x, points_y, points_z,
                                data={"true_11": true_11, "pred_11": pred_11}
                                )

# Offline trained model on 1k flat plate BL data with 3x mesh filter width
data_path = "./3x/data/train_data_3x.npy"
crd_path = "./3x/data/crd_data_3x.npy"
model_path = "./3x/NNmodel_jit.pt"
device = "cpu"
off_bl_3x = SGS(data_path, crd_path, model_path, device, vtk_export=True)
off_bl_3x.load_data_model()
mse_glob, cc_glob = off_bl_3x.test_global()
crd_y, mse_y_off_bl_3x_3x, cc_y_off_bl_3x_3x =  off_bl_3x.test_y_layers()

# Offline trained model on 1k flat plate BL data with 3x mesh filter width
data_path = "./6x/data/train_data_6x.npy"
crd_path = "./6x/data/crd_data_6x.npy"
model_path = "./6x/NNmodel_jit.pt"
device = "cpu"
off_bl_6x = SGS(data_path, crd_path, model_path, device, vtk_export=True)
off_bl_6x.load_data_model()
mse_glob, cc_glob = off_bl_6x.test_global()
crd_y, mse_y_off_bl_6x_6x, cc_y_off_bl_6x_6x =  off_bl_6x.test_y_layers()

# Offline trained model on 1k flat plate BL data with 3x mesh filter width
X_off_bl_6x = off_bl_6x.X
y_off_bl_6x = off_bl_6x.y
crd_off_bl_6x = off_bl_6x.crd
crd_y, mse_y_off_bl_3x_6x, cc_y_off_bl_3x_6x = off_bl_3x.test_y_layers(X_off_bl_6x,y_off_bl_6x,crd_off_bl_6x)

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


