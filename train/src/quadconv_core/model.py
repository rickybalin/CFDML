'''
'''

import torch
from torch import nn

from .modules import Encoder, Decoder
from .loss import relative_re, root_relative_re, RRELoss

'''
Quadrature convolution autoencoder.

Input:
    spatial_dim: spatial dimension of data
    point_seq:
    data_info:
    loss_fn: loss function specification
    optimizer: optimizer specification
    learning_rate: learning rate
    noise_scale: scale of noise to be added to latent representation in training
    internal_activation: activation of internal layers
    output_activation: final activation
    kwargs: keyword arguments to be passed to encoder and decoder
'''
class Model(nn.Module):

    def __init__(self,*,
            spatial_dim,
            mesh,
            point_seq,
            quad_map = "newton_cotes_quad",
            quad_args = {},
            loss_fn = "MSELoss",
            optimizer = "Adam",
            learning_rate = 1e-2,
            noise_scale = 0.0,
            internal_activation = "CELU",
            output_activation = "Tanh",
            load_mesh_weights = [True],
            **kwargs
        ):
        super().__init__()

        #training hyperparameters
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.noise_scale = noise_scale

        #construct mesh sub-levels
        self.mesh = mesh.construct(point_seq, mirror=True, quad_map=quad_map, quad_args=quad_args)

        #loss function
        if (loss_fn=="MSELoss"):
            self.loss_fn = getattr(nn, loss_fn)()
        elif (loss_fn=="RRELoss"):
            self.loss_fn = RRELoss()

        #activations
        self.internal_activation = getattr(nn, internal_activation)
        self.output_activation = getattr(nn, output_activation)()

        self.encoder = Encoder(spatial_dim=spatial_dim, 
                               forward_activation=self.internal_activation,
                               latent_activation=self.internal_activation,
                               **kwargs)
        self.decoder = Decoder(spatial_dim=spatial_dim,
                               forward_activation=self.internal_activation,
                               latent_activation=self.internal_activation,
                               **kwargs)

        #
        if len(load_mesh_weights) == 1:
            load_mesh_weights = load_mesh_weights*len(point_seq)

        self.load_mesh_weights = load_mesh_weights

        return

    '''
    Forward pass of encoder.

    Input:
        x: input data

    Output: compressed data
    '''
    def encode(self, x):
        return self.encoder(self.mesh, x)

    '''
    Forward pass of decoder.

    Input:
        z: compressed data

    Output: compressed data reconstruction
    '''
    def decode(self, z):
        return self.output_activation(self.decoder(self.mesh, z))

    '''
    Forward pass of model.

    Input:
        x: input data

    Output: compressed data reconstruction
    '''
    def forward(self, x):
        return self.decode(self.encode(x))

    '''
    Single training step.

    Input:
        batch: batch of data

    Output: pytorch loss object
    '''
    def training_step(self, batch):
        #encode and add noise to latent rep.
        latent = self.encode(batch)
        if self.noise_scale != 0.0:
            latent = latent + self.noise_scale*torch.randn(latent.shape, device=batch.device)

        #decode
        pred = self.decode(latent)

        #compute loss
        loss = self.loss_fn(pred, batch)

        return loss

    '''
    Single validation_step; logs validation error.

    Input:
        batch: batch of data
    '''
    def validation_step(self, batch, return_loss=False):
        #predictions
        pred = self(batch)

        #compute relative reconstruction error
        error = root_relative_re(pred, batch)
        error =  torch.mean(error)

        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(pred, batch)
            return error, loss
        else:
            return error

    '''
    Single test step; logs average and max test error

    Input:
        batch: batch of data
    '''
    def test_step(self, batch, return_loss=False):
        #predictions
        pred = self(batch)

        #compute relative reconstruction error
        error = root_relative_re(pred, batch)
        error =  torch.mean(error, dim=0)
 
        if return_loss:
            # compute loss to compare agains training
            loss = self.loss_fn(pred, batch)
            return error, loss
        else:
            return error

    '''
    Single prediction step.

    Input:
        batch: batch of data
        idx: batch index

    Output: compressed data reconstruction
    '''
    def predict_step(self, batch, idx):
        return self(batch)

    '''
    Instantiates optimizer

    Output: pytorch optimizer
    '''
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
        scheduler_config = {"scheduler": scheduler, "monitor": "val_err"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    '''
    Edit the checkpoint state_dict
    '''
    def on_load_checkpoint(self, checkpoint):

        state_dict = checkpoint["state_dict"]

        for param_name in list(state_dict.keys()):
            if param_name[:4] == "mesh":
                if self.load_mesh_weights.pop(0) == False:
                    state_dict.pop(param_name)

        return

    '''
    Edit the checkpoint state_dict
    '''
    def on_save_checkpoint(self, checkpoint):

        state_dict = checkpoint["state_dict"]

        for param_name in list(state_dict.keys()):
            if param_name[:12] == "mesh._points" or param_name[-12:] == "eval_indices":
                state_dict.pop(param_name)

        return
