'''
'''

import torch
import torch.nn as nn

def relative_mse(input, target, reduction="mean"):
    return (input-target)**2/target**2

class RMSELoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, input, target):
        return torch.mean(relative_mse(input, target, self.reduction))
    
class RRMSELoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, input, target):
        return torch.sqrt(torch.mean(relative_mse(input, target, self.reduction)))
    
def relative_re(input, target, reduction="mean"):

    n = torch.sum((input-target)**2, dim=(2))
    d = torch.sum((target)**2, dim=(2))

    return n/d

class RRELoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, input, target):
        return torch.mean(relative_re(input, target, self.reduction))