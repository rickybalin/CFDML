# In Situ Machine Learning for Computational Fluid Dynamics

CFDML is a Python based tool designed to facilitate the application of machine learning (ML) to the fields of computational fluid dynamics (CFD) and turbulence modeling.
It provides a number of useful features, including:
* PyTorch implementation of multiple [models](#models) for CFD applications
* Scalable distributed data-parallel training with Horovod and DDP
* Both offline and online (aka in situ) training approaches 
* Integration with multiple CFD codes for training and inference of ML models
* Integration with vtk for visualization of data with Paraview
* Portable across HPC hardware (CPU and GPU from Nvidia, AMD and Intel)

Integration with different CFD codes and the infrastructure to perform online training and inference are enabled through the use of the [SmartSim](https://github.com/CrayLabs/SmartSim) and [SmartRedis](https://github.com/CrayLabs/SmartRedis) libraries. 
Central to these online workflows is the deployment of a database which allows training and inference data, useful metadata and ML models to be stored in memory for the duration of the job, thus avoiding the file system. 


## Models
### Data Driven Sub-Grid Stress (SGS) Modeling of Turbulent Flows
This is a model for the the sug-grid stress tensor, which is the unclosed term in the governing equations for large eddy simulation (LES). By careful selection of the input and output variables, the model satisfies Galilean, rotational, reflectional, and unit invariance. Additionally, the anisotropy of the filter width is embedded in the model through a mapping that transforms the filtered velocity gradient tensor in physical space to a mapped space with isotropic filter kernel. 
More details on the model formulation and the experiments performed can be found in [Prakash et al., 2023](https://arxiv.org/abs/2212.00332).

The model is composed of an artificial neural network (ANN) with 6 input features, a single fully connected layer with 20 neurons, and a final output layer with 6 targets (i.e., the six unique components of the SGS tensor). It uses Leaky ReLU activation functions with a slope for x<0 of 0.3. The model used in the Prakash et al., 2023 work can be found [here](Models/HIT) in the form of a `.pt` PyTorch file and a jit-traced file, along with the min-max scaling needed to dimentionalize the outputs before converting them back to physical stresses. 
Note that the size of the model, in terms of the number of hidden layers of the ANN and the number of neurons per layer, can be modified from the [configuration file](srt/train/conf/train_config.yaml) in order to test training with a larger network. 

To train this model offline or online:
* Set the model string to `sgs` in the [training config file](src/train/conf/train_config.yaml)
* Set the number of layers and neurons per layers under the `sgs` heading in the config file
* Under the same heading, select whether to compute the model inputs and outputs from the raw filtered velocity gradients before training. The default for this option is False which assumes the already processed inputs and outputs are being fed to CFDML.

### QuadConv Autoencoder for Compression of Flow Solution States
This is an autoencoder which can be used for the compression of solution states (three velocity components and pressure) discretized with uniform, non-uniform and unstructured grids. This is achieved thanks to the quadrature-based convolution (QuadConv) layers, which approximate continuous convolution via quadrature (i.e., a single weighted sum), where both the weights and the kernels are learned during training. The kernels are parametrized by an MLP, therefore no convolutional layers are present. 
More details on the QuadConv operator and the model can be found in the [original publication](https://arxiv.org/abs/2211.05151), and a [PyTorch implementation](https://github.com/AlgorithmicDataReduction/PyTorch-QuadConv) of the operator is available to install. 

To train this model offline or online:
* Set the model string to `quadconv` in the [training config file](src/train/conf/train_config.yaml)
* Under the `quadconv` heading:
  * Set the path to the mesh file (leave this string empty for online training)
  * Set the number of channels. The default is 4, which assumes 3 velocity components and pressure
  * Set the path to the config file for the QuadConv model and the QCNN layers. An example of this config script can be found [here](src/train/conf/quadconv_config.yaml)


## Build and Run Instructions
### ALCF Polaris
On the Polaris system at ALCF, SmartSim and SmartRedis can be installed running the [build script](Polaris/build_SSIM_Polaris.sh) from an interactive node with
```sh
source build_SSIM_Polaris.sh
```
Note that this script creates a new conda enviroment which does not contain all the modules available with the base env from the conda/2022-09-08 module on Polaris, however it contains many of the essential packages, such as PyTorch, TensorFlow, Horovod, and MPI4PY.
Integration of SmartSim and SmartRedis within the provided data science modules available on the system is coming soon.


