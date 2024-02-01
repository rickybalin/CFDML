##### 
##### Wrapper functions for the avaliable models to initialize them
##### and load/create their required data structures 
#####
import typing
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from utils import count_weights

from sgs.model import anisoSGS

from quadconv_core.model import QuadConv


from gnn.model import GNNModel


def load_model(cfg: DictConfig, comm, rng) -> tuple[nn.Module, np.ndarray]: 
    """ 
    Return the selected model and its training data

    :param cfg: DictConfig with training configuration parameters
    :param comm: Class containing the MPI communicator information
    :param rng: numpy random number generator
    :return: touple with the model and the data
    """

    if (cfg.model=="sgs"):
        model  = anisoSGS(numNeurons=cfg.sgs.neurons, numLayers=cfg.sgs.layers)
    elif (cfg.model=="quadconv"):
        model = QuadConv(**model_config, train_config=cfg)
        if (comm.rank==0):
            print('Quad-Conv model with configuration:\n')
            print(model.cfg)
            print("")
    elif (cfg.model=="gnn"):
        model = models.GNN(cfg.gnn.gnn_config)
    
    n_params = count_weights(model)
    if (comm.rank == 0):
        print(f"\nLoaded {cfg.model} model with {n_params} trainable parameters \n")

    if not cfg.online.db_launch:
        if (cfg.data_path == "synthetic"):
            data = model.create_data(cfg, rng)
        else:
            data = model.load_data(cfg)
    else:
        data = np.array[[0]]

    return model, data    



################################################
######## GNN ###################################
################################################
### Distributed GNN developed by Shivam Barwey at Argonne National Laboratory
def GNN(config_file):
    return GNNModel(config_file)


