##### 
##### This script contains all the Pytorch Datasets that might be needed for
##### different training approaches and algorithms 
#####

import torch
from torch.utils.data import Dataset,TensorDataset

class BaseDataset(TensorDataset):
    def __init__(self,tensor):
        super().__init__(tensor)

class MiniBatchDataset(Dataset):
    def __init__(self,tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]




