################################################
######## QuadConv ##############################
################################################
# Quad-Conv (QCNN) AE model developed by Cooper Simpson, Alireza Doostan, Stephen Becker at Univ. Colorado Boulder
# K. Doherty, C. Simpson, et al. 2023. QuadConv: Quadrature-Based Convolutions with Applications to Non-Uniform PDE Data Compression. arXiv:2211.05151 [cs.LG]

from typing import Optional, Union
from omegaconf import DictConfig
from time import perf_counter
import math as m
import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import OfflineDataset
from .modules import Encoder, Decoder
from .loss import relative_re, root_relative_re, RRELoss
from .utils import load_model_config
try:
    from torch_quadconv import MeshHandler
except:
    pass

class QuadConv(nn.Module):
    def __init__(self,
            train_config,
            client,
            t_data):
        '''
        Quadrature convolution autoencoder
        '''

        super().__init__()

        # Load model config
        self.cfg = load_model_config(train_config.quadconv.quadconv_config)

        # Load/generate the mesh nodes
        mesh_nodes = self.generate_mesh(train_config, client, t_data)
        self.cfg["_num_points"] = mesh_nodes.shape[0]
        point_seq = self.cfg["point_seq"]
        point_seq[0] = mesh_nodes.shape[0]
        self.cfg["point_seq"] = point_seq
        in_points = self.cfg["conv_params"]["in_points"]
        in_points[0] = mesh_nodes.shape[0]
        self.cfg["conv_params"]["in_points"] = in_points

        # consturct the mesh
        mesh_nodes = torch.from_numpy(mesh_nodes)
        mesh = MeshHandler(mesh_nodes)
        self.mesh = mesh.construct(self.cfg["point_seq"], mirror=True, 
                                   quad_map=self.cfg["quad_map"], 
                                   quad_args=self.cfg["quad_args"])

        #loss function
        if self.cfg["loss_fn"] == "RRELoss":
            self.loss_fn = RRELoss()
        else:
            self.loss_fn = getattr(nn, self.cfg["loss_fn"])()

        #activations
        self.internal_activation = getattr(nn, self.cfg["internal_activation"])
        self.output_activation = getattr(nn, self.cfg["output_activation"])()

        # Build encoder and decoder
        self.encoder = Encoder(spatial_dim=self.cfg["spatial_dim"],
                               stages=self.cfg["stages"],
                               conv_params=self.cfg["conv_params"],
                               latent_dim=self.cfg["latent_dim"],
                               forward_activation=self.internal_activation,
                               latent_activation=self.internal_activation,
                               )
        self.decoder = Decoder(spatial_dim=self.cfg["spatial_dim"],
                               stages=self.cfg["stages"],
                               conv_params=self.cfg["conv_params"],
                               latent_dim=self.cfg["latent_dim"],
                               forward_activation=self.internal_activation,
                               latent_activation=self.internal_activation,
                               )

        # Other initializations
        load_mesh_weights = self.cfg["load_mesh_weights"]
        if len(load_mesh_weights) == 1:
            load_mesh_weights = load_mesh_weights*len(self.cfg["point_seq"])
        self.load_mesh_weights = load_mesh_weights
        self.noise_scale = self.cfg["noise_scale"]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of encoder

        :param x: input data
        :return: compressed data
        '''
        return self.encoder(self.mesh, x)

    def decode(self, z):
        '''
        Forward pass of decoder

        :param z: compressed data
        :return: reconstructed data
        '''
        return self.output_activation(self.decoder(self.mesh, z))

    def forward(self, x):
        '''
        Forward pass of model

        :param x: input data
        :return: reconstructed data
        '''
        return self.decode(self.encode(x))

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Perform a training step

        :param batch: a torch.Tensor containing the batched inputs
        :return: loss for the batch
        """
        latent = self.encode(batch)
        if self.noise_scale != 0.0:
            latent = latent + self.noise_scale*torch.randn(latent.shape, device=batch.device)

        output = self.decode(latent)
        loss = self.loss_fn(output, batch)
        return loss

    def validation_step(self, batch: torch.Tensor) -> tuple((torch.Tensor, torch.Tensor)):
        """
        Perform a validation step

        :param batch: a torch.Tensor containing the batched inputs and outputs
        :return: tuple with the accuracy and loss for the batch
        """
        output = self(batch)

        error = root_relative_re(output, batch)
        error = torch.mean(error)
        loss = self.loss_fn(output, batch)
        return error, loss

    def test_step(self, batch: torch.Tensor, return_loss: Optional[bool] = False) -> tuple((torch.Tensor, torch.Tensor)):
        """
        Perform a test step

        :param batch: a tensor containing the batched inputs and outputs
        :param return_loss: whether to compute the loss on the testing data
        :return: tuple with the accuracy and loss for the batch
        """
        output = self(batch)

        error = root_relative_re(output, batch)
        error = torch.mean(error, dim=0)
 
        if return_loss:
            loss = self.loss_fn(output, batch)
        else:
            loss = torch.Tensor([0.])
        return error, loss
    
    def generate_mesh(self, cfg: DictConfig, client, t_data):
        if cfg.online.db_launch:
            rtime = perf_counter()
            mesh = client.client.get_tensor('mesh').astype('float32')
            rtime = perf_counter() - rtime
            t_data.t_meta = t_data.t_meta + rtime
            t_data.i_meta = t_data.i_meta + 1 
        else:
            if (cfg.data_path == "synthetic"):
                N = 32
                spatial_dim = self.cfg["spatial_dim"]
                mesh = np.zeros((N**spatial_dim,spatial_dim), dtype=np.float32)
                for i in range(N):
                    x = 0. + 1. * (i - 1) / (N - 1)
                    for j in range(N):
                        y = 0. + 1. * (j - 1) / (N - 1)
                        if (spatial_dim==2):
                            ind = i*N + j
                            mesh[ind,0] = x
                            mesh[ind,1] = y
                        elif (spatial_dim==3):
                            for k in range(N):
                                z = 0. + 1. * (k - 1) / (N - 1)
                                ind = i*N**2 + j*N +k
                                mesh[ind,0] = x
                                mesh[ind,1] = y
                                mesh[ind,2] = z
            else:
                extension = cfg.quadconv.mesh_file.split(".")[-1]
                if "npy" in extension:
                    mesh = np.float32(np.load(cfg.quadconv.mesh_file))

        return mesh

    
    def create_data(self, cfg: DictConfig, rng) -> np.ndarray:
        """"
        Create synthetic training data for the model

        :param cfg: DictConfig with training configuration parameters
        :param rng: numpy random number generator
        :return: numpy array with the rank-local training data 
        """
        if (cfg.num_samples_per_rank==111):
            samples = 20 * cfg.mini_batch
        else:
            samples = cfg.num_samples_per_rank
        N = 32
        spatial_dim = self.cfg["spatial_dim"]
        data = np.float32(rng.normal(size=(samples,cfg.quadconv.channels,N**spatial_dim)))
        return data
    
    def load_data(self, cfg: DictConfig, comm) -> np.ndarray:
        """"
        Load training data for the model

        :param cfg: DictConfig with training configuration parameters
        :return: numpy array with the rank-local training data 
        """
        
        extension = cfg.data_path.split(".")[-1]
        if "npy" in extension:
            data = np.float32(np.load(cfg.data_path))

        # Scale input data from [-1,1]
        with open(cfg.name+"_scaling.dat", "w") as fh:
            for i in range(cfg.quadconv.channels):
                min_val = np.amin(data[:,i,:])
                max_val = np.amax(data[:,i,:])
                fh.write(f"{min_val:>8e} {max_val:>8e}\n")
                data[:,i,:] = 2.0*(data[:,i,:] - min_val)/(max_val - min_val) - 1.0
        return data
    
    def setup_dataloaders(self, data: np.ndarray, cfg, comm) -> dict:
        """
        Prepare the training and validation data loaders 

        :param data: training data
        :param cfg: DictConfig with training configuration parameters
        :param comm: MPI communication class
        :return: tuple of DataLoaders 
        """
        # DataSet
        samples = data.shape[0]
        nVal = m.floor(samples*cfg.validation_split)
        nTrain = samples-nVal
        if (nVal==0 and cfg.validation_split>0):
            if (comm.rank==0): print("Insufficient number of samples for validation -- skipping it")
        dataset = OfflineDataset(data)
        trainDataset, valDataset = random_split(dataset, [nTrain, nVal])

        # DataLoader
        # Try:
        # - pin_memory=True - should be faster for GPU training
        # - num_workers > 1 - enables multi-process data loading 
        # - prefetch_factor >1 - enables pre-fetching of data
        if (cfg.data_path == "synthetic"):
            # Each rank has loaded only their part of training data
            train_sampler = None
            val_sampler = None
            val_dataloader = None
            train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                          shuffle=True, drop_last=True) 
            if (nVal>0):
                val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                            drop_last=True)
        else:
            # Each rank has loaded all the training data, so restrict data loader to a subset of dataset
            val_sampler = None
            val_dataloader = None
            train_sampler = DistributedSampler(trainDataset, num_replicas=comm.size, rank=comm.rank,
                                           shuffle=True, drop_last=True) 
            train_dataloader = DataLoader(trainDataset, batch_size=cfg.mini_batch, 
                                  sampler=train_sampler)
            if (nVal>0):
                val_sampler = DistributedSampler(valDataset, num_replicas=comm.size, rank=comm.rank,
                                                 drop_last=True) 
                val_dataloader = DataLoader(valDataset, batch_size=cfg.mini_batch, 
                                            sampler=val_sampler)
                
        return {
            'train': {
                'loader': train_dataloader,
                'sampler': train_sampler,
                'samples': nTrain
            },
            'validation': {
                'loader': val_dataloader,
                'sampler': val_sampler,
                'samples': nVal
            }
        }

    def save_checkpoint(self, fname: str, data: torch.Tensor):
        torch.save(self.state_dict(), f"{fname}.pt", _use_new_zipfile_serialization=False)
        encoder = quadconvEncoder(self.encoder, self.mesh)
        decoder = quadconvDecoder(self.decoder, self.mesh, self.output_activation)
        dummy_latent = encoder(data).detach()
        predicted = decoder(dummy_latent).detach()
        module_encode = torch.jit.trace(encoder, data)
        torch.jit.save(module_encode, f"{fname}_encoder_jit.pt")
        dummy_latent = module_encode(data).detach()
        module_decode = torch.jit.trace(decoder, dummy_latent)
        torch.jit.save(module_decode, f"{fname}_decoder_jit.pt")
        predicted = module_decode(dummy_latent).detach()

    

# Classes used for tracing the encoder and decoder separately
class quadconvEncoder(torch.nn.Module):
    def __init__(self, encoder, mesh):
        """
        Usage: trace = torch.jit.trace(Encoder(model.encoder, model.mesh), input_data)
        """
        super().__init__()
        self.encoder = encoder
        self.mesh = mesh

    def forward(self, X):
        self.mesh.reset()

        return self.encoder(self.mesh, X)

class quadconvDecoder(torch.nn.Module):
    def __init__(self, decoder, mesh, output_activation):
        super().__init__()
        self.decoder = decoder
        self.mesh = mesh
        self.output_activation = output_activation

    def forward(self, X):
        self.mesh.reset(mirror=True)

        return self.output_activation(self.decoder(self.mesh, X))
