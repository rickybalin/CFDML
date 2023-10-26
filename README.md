# In Situ Machine Learning for Computational Fluid Dynamics

CFDML is a Python based tool designed to facilitate the application of machine learning (ML) to the fields of computational fluid dynamics (CFD) and turbulence modeling. 
It provides a suite of pre-trained ML models capable of performing various taks together with a framework to facilitate the integration of these models within CFD codes to enable live inferencing and deployment in production simulations.
Additionally, this tool provides the framework to train these models on new data, whether this has been saved to disk or whether it is being streamed live from an ongoing simulation or experiment, and therefore extend their capability beyond the pre-trained versions.
Overall, the main objectives of CFDML are to facilitate the coupling of ML with CFD codes and to promote collaboration and sharing between different groups working in this field.

CFDML provides a number of useful features, including:
* PyTorch implementation of multiple [models](#models) for CFD applications
* Scalable distributed data-parallel training with Horovod and DDP
* Both offline and online (aka in situ) training approaches 
* Integration with multiple CFD codes for training and inference of ML models at scale
* Integration with vtk for visualization of data with Paraview
* Portable across various HPC hardware (CPU and GPU from Nvidia, AMD and Intel)

In order to perform online training and inference from live simulation (or experiment) data, CFDML leverages the [SmartSim](https://github.com/CrayLabs/SmartSim) and [SmartRedis](https://github.com/CrayLabs/SmartRedis) libraries. 
Central to these online workflows is the deployment of a database which allows training and inference data, useful metadata, and ML models to be stored in memory for the duration of the job, thus avoiding I/O and storage bottlenecks that come with using the file system. 
During online training, the database allows the simulation and ML training to run concurrently and in a fully decoupled manner on separate resources (CPU and/or GPU), thus minimizing the idle time and overhead on both components. 
This strategy is different from the traditional offline training approach, wherein the training data is produced separately from training and then stored to the disk.
During online inference, the database allows the simulation to load one or more ML models and its input data onto the database, therefore and with the SmartRedis API the model can be evaluated on the data on the CPU or GPU to obtain the desired predictions.

CFDML provides the workflow driver which launches the components involved for online training and inference (i.e., the simulation, database, and/or distributed training), as well as the standard offline training. It also controls the deployment approach used for the workflow, which can be clustered or colocated. 
In the former, all components are deployed on distinct set of nodes and the data being transferred across components is communicated through the machine?s interconnect.
In the latter, the components share resources on the same nodes, thus keeping all data transfer within each node. This approach was demonstrated to scale very efficiently to hundreds of nodes on the Polaris machine at ALCF in [Balin et al., 2023](https://arxiv.org/abs/2306.12900).

Note that CFDML takes the path to the simulation executable as an input to the online workflow driver script, but it does not offer a CFD code as default.
In fact, CFDML is independent of the CFD package used.
However, for CFDML to be used for online training and inference, a particular code must be integrated with the SmartRedis library such that a connection to the database can be established. 
Moreover, currently the CFD solver is responsible for computing the model?s input and outputs from the raw solution variables.
The CFD codes [PHASTA](https://github.com/PHASTA/phasta) and [libCEED](https://github.com/CEED/libCEED) have already been augmented with the SmartRedis API and linked to CFDML. 
 

## Offline Training
To perform offline training with CFDML, run the training driver script directly with
````
python src/train/train_driver.py
```
For data-parallel distributed training with multiple processes, use a parallel job launcher, for example
```
mpiexec -n <num_procs> ?ppn <num_procs_per_node> python src/train/train_driver.py
```

The training run will be set up according to the default configuration parameters specified in
```
src/train/conf/train_config.yaml
```
Take a look at this file for the full list of features and their options. 
To change the run parameters:
* Change the entries to src/train/conf/train_config.yaml directly
* Pass the parameter names and options at the command line (for example `python src/train/train_driver.py epochs=100 mini_batch=16`)
* Copy the config file and pass its directory path at the command line (for example `python src/train/train_driver.py --config-path </path/to/config/directory>`)


## Online Training and Inference
To perform online training or inference, run the SmartSim driver script directly with
````
python src/train/ssim_driver.py
```
The workflow will be set up according to the default configuration parameters specified in
```
src/train/conf/ssim_config.yaml
```
Take a look at this file for the full list of features and their options. 

For online training, set the following parameters:
* In `src/train/conf/ssim_config.yaml`
	* `database.launch=True`
	* Make sure the scheduler in `database.launcher` matches the one on your machine
	* Set the number of nodes, processes, processes per node and CPU binding for each component under `run_args`
	* Specify the path to the simulation executable `sim.executable`. This is the binary to the CFD code
	* Pass any command line arguments to the simulation code with `sim.arguments`
	* Set GPU affinity for the simulation with `sim.affinity`. One for Polaris at ALCF is provided [here](Polaris/afinity_sim.sh).
	* Specify the path to the training executable with `train.executable`. This is the full path to the training driver script `src/train/train_driver.py`
	* Set GPU affinity for the simulation with `train.affinity`. One for Polaris at ALCF is provided [here](Polaris/afinity_ml.sh).
	* If different than the default path `src/train/conf/train_config.yaml`, specify the path to the config file with `train.config`
* In `src/train/conf/train_config.yaml`
	* Set the number of processes per node used in distributed training with `ppn`
	* Set the online training parameters under the `online` heading
	* Set the other training parameters as desired

For online inference, set the following parameters:
* In `src/train/conf/ssim_config.yaml`
	* `database.launch=True`
	* Make sure the scheduler in `database.launcher` matches the one on your machine
	* Set the number of nodes, processes, processes per node and CPU binding for each component under `run_args`. The parameters for the ML component are not used in this case, but the ones for the database are.
	* Specify the path to the simulation executable `sim.executable`. This is the binary to the CFD code
	* Pass any command line arguments to the simulation code with `sim.arguments`
	* Set GPU affinity for the simulation with `sim.affinity`. One for Polaris at ALCF is provided [here](Polaris/afinity_sim.sh).
	* Specify the path to the model to load with `inference.model_path`
	* Specify the ML framework backend to use to run the model (e.g., TORCH, TF or ONNX)
	* Specify the hardware on which to perform inference (e.g., CPU or GPU)
	* Set other parameters under the `inference` header for further control


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

We are currently working on applying this model to wall-bounded turbulent flows, so a pre-trained version is not available yet. To reproduce the model from the original publication, follow the instructions in [this repo](https://github.com/kvndhrty/QuadConv).

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

### ALCF Aurora
Information is upcoming, check back soon.

### Other systems
General information on how to run SmartSim and SmartRedis on other systems is found [here](https://www.craylabs.org/docs/installation_instructions/platform.html).


## Publications
[Balin et al., ?In Situ Framework for Coupling Simulation and Machine Learning with Application to CFD?, arXiv:2306.12900, 2023](https://arxiv.org/abs/2306.12900)



