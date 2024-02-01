import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
from tg.data import Data
import tg.utils as utils
import tg.nn as tgnn

import gnn as gnn
import graph_connectivity as gcon
import graph_plotting as gplot
from train.quadconv_core import loss

class GNNModel(nn.Module):
    def __init__(self, cfg, ):
        self.cfg = cfg

        self.model = self.build_model()

        # Define the loss function
        self.loss_fn = nn.MSE()
        # Define the loss function to measure accuracy
        self.acc_fn = nn.MSE()

    def build_model(self):
        sample = self.data['train']['example']
        input_channels = sample.x.shape[1] 
        output_channels = sample.y.shape[1]
        activation = getattr(F, self.cfg.activation)

        model = gnn.mp_gnn(input_channels, 
                           self.cfg.hidden_channels, 
                           output_channels, 
                           self.cfg.n_mlp_layers, 
                           activation)
        return model

    def training_step(self, batch):
        output = self.model(batch.x, self.edge_index, self.edge_attr, self.pos, self.batch)
        loss = self.loss_fn(output, batch.y)
        return loss
    
    def validation_step(self, batch, return_loss=False):
        output = self.model(batch.x, self.edge_index, self.edge_attr, self.pos, self.batch)
        error = self.loss_fn(output, batch.y)

        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(output, batch.y)
            return error, loss
        else:
            return error
        
    def test_step(self, batch, return_loss=False):
        output = self.model(batch.x, self.edge_index, self.edge_attr, self.pos, self.batch)
        error = self.loss_fn(output, batch.y)

        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(output, batch)
            return error, loss
        else:
            return error