'''
Encoder and decoder modules based on the convolution block with skips and pooling.

Input:
    spatial_dim: spatial dimension of input data
    stages: number of convolution block stages
    conv_params: convolution parameters
    latent_dim: dimension of latent representation
    forward_activation: block activations
    latent_activation: mlp activations
    kwargs: keyword arguments for conv block
'''

from torch import nn

from torch_quadconv import QuadConv

from .utils import package_args, swap
from .quadconv_blocks import PoolBlock

'''
Traceable Unflatten layer.
'''
class Unflatten(nn.Module):
    def __init__(self, dim,  shape):
        super().__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)

################################################################################

class Encoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            block_args={},
            forward_activation = nn.GELU,
            latent_activation = nn.GELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, conv_params)

        #build network
        self.init_layer = QuadConv(spatial_dim = spatial_dim, **arg_stack[0])

        self.qcnn = nn.Sequential()

        for i in range(1, stages+1):
            self.qcnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        step = False if i == stages else True,
                                        **block_args
                                        ))

        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

        self.linear = nn.Sequential()

        linear_input_size = conv_params['out_channels'][-1] * int(conv_params['out_points'][-1])

        self.linear.append(nn.Linear(linear_input_size, latent_dim))
        self.linear.append(latent_activation())
        self.linear.append(nn.Linear(latent_dim, latent_dim))
        self.linear.append(latent_activation())

    '''
    Forward
    '''
    def forward(self, mesh, x):
        x = self.init_layer(mesh, x)
        _, x = self.qcnn((mesh, x))
        x = self.flat(x)
        output = self.linear(x)

        return output

################################################################################

class Decoder(nn.Module):

    def __init__(self,*,
            spatial_dim,
            stages,
            conv_params,
            latent_dim,
            block_args={},
            forward_activation = nn.GELU,
            latent_activation = nn.GELU,
            **kwargs
        ):
        super().__init__()

        #block arguments
        arg_stack = package_args(stages+1, swap(conv_params), mirror=True)

        #build network
        self.unflat = Unflatten(1, (conv_params['out_channels'][0], int(conv_params['out_points'][0])))

        self.linear = nn.Sequential()

        linear_output_size = conv_params['out_channels'][0] * int(conv_params['out_points'][0])

        self.linear.append(nn.Linear(latent_dim, latent_dim))
        self.linear.append(latent_activation())
        self.linear.append(nn.Linear(latent_dim, linear_output_size))
        self.linear.append(latent_activation())

        self.qcnn = nn.Sequential()

        for i in range(stages):
            self.qcnn.append(PoolBlock(spatial_dim = spatial_dim,
                                        **arg_stack[i],
                                        activation1 = forward_activation,
                                        activation2 = forward_activation,
                                        adjoint = True,
                                        step = False if i == 0 else True,
                                        **block_args
                                        ))

        self.init_layer = QuadConv(spatial_dim = spatial_dim, **arg_stack[-1])

    '''
    Forward
    '''
    def forward(self, mesh, x):
        x = self.linear(x)
        x = self.unflat(x)
        _, x = self.qcnn((mesh, x))
        output = self.init_layer(mesh, x)

        return output
