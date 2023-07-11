##### 
##### This script contains all the NN models used for training 
#####

import torch
import torch.nn as nn
import numpy as np
from utils import comp_corrCoeff
from quadconv_core.utils import load_model_config
from quadconv_core.model import Model
from torch_quadconv import MeshHandler


### Anisotropic SGS model for LES developed by Aviral Prakash and John A. Evans at UCB
class anisoSGS(nn.Module): 
    # The class takes as inputs the input and output dimensions and the number of layers   
    def __init__(self, inputDim=6, outputDim=6, numNeurons=20, numLayers=1):
        super().__init__()
        self.ndIn = inputDim
        self.ndOut = outputDim
        self.nNeurons = numNeurons
        self.nLayers = numLayers
        #self.net = nn.Sequential(
        #    nn.Linear(self.ndIn, self.nNeurons),
        #    nn.LeakyReLU(0.3),
        #    nn.Linear(self.nNeurons, self.ndOut))
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(self.ndIn, self.nNeurons)) # input layer
        self.net.append(nn.LeakyReLU(0.3))
        for l in range(self.nLayers-1): # hidden layers
            self.net.append(nn.Linear(self.nNeurons, self.nNeurons))
            self.net.append(nn.LeakyReLU(0.3))
        self.net.append(nn.Linear(self.nNeurons, self.ndOut)) # output layer
        
        # Define the loss function
        self.loss_fn = nn.functional.mse_loss
        # Define the loss function to measure accuracy
        self.acc_fn = nn.functional.mse_loss #comp_corrCoeff

    # Define the method to do a forward pass
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        #return self.net(x)
        return x

    # Define the methods to do a training, validation and test step
    def training_step(self, batch):
        """returns loss"""
        target = batch[:, :self.ndOut]
        features = batch[:, self.ndOut:]
        output = self.forward(features)
        loss = self.loss_fn(output, target)
        return loss

    def validation_step(self, batch, return_loss=False):
        """returns performance metric"""
        target = batch[:, :self.ndOut]
        features = batch[:, self.ndOut:]
        prediction = self.forward(features)
        acc = self.acc_fn(prediction, target)
        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(prediction, target)
        else:
            loss = 0.
        return acc, loss

    def test_step(self, batch, return_loss=False):
        """returns performance metric"""
        target = batch[:, :self.ndOut]
        features = batch[:, self.ndOut:]
        prediction = self.forward(features)
        acc = self.acc_fn(prediction, target)
        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(prediction, target)
        else:
            loss = 0.
        return acc, loss




### Quad-Conv (QCNN) AE model developed by Cooper Simpson, Alireza Doostan, Stephen Becker at UCB
def qcnn(rank, mesh_nodes, config_file, num_channels):
    mesh = MeshHandler(mesh_nodes)

    #load and update model config
    model_config = load_model_config(config_file)
    model_config["_num_points"] = mesh_nodes.shape[0]
    point_seq = model_config["point_seq"]
    point_seq[0] = mesh_nodes.shape[0]
    model_config["point_seq"] = point_seq
    in_points = model_config["conv_params"]["in_points"]
    in_points[0] = mesh_nodes.shape[0]
    model_config["conv_params"]["in_points"] = in_points
    if (rank==0):
        print('Quad-Conv model with configuration:\n')
        print(model_config)
        print("")    

    return Model(**model_config, mesh=mesh)
    

# Classes used for tracing the encoder and decoder separately
class qcnnEncoder(torch.nn.Module):
    def __init__(self, encoder, mesh):
        """Usage: trace = torch.jit.trace(Encoder(model.encoder, model.mesh), input_data)
        """
        super().__init__()
        self.encoder = encoder
        self.mesh = mesh

    def forward(self, X):
        self.mesh.reset()

        return self.encoder(self.mesh, X)

class qcnnDecoder(torch.nn.Module):
    def __init__(self, decoder, mesh):
        super().__init__()
        self.decoder = decoder
        self.mesh = mesh

    def forward(self, X):
        self.mesh.reset(mirror=True)

        return self.decoder(self.mesh, X)


###



