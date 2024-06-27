# CFDML with libCEED on ALCF Aurora

## Build Instructions

### Build and Run Environment (including SmartSim/SmartRedis)

Execute the following code on a compute node of Aurora to create a Python virtual environment with SmartSim and SmartRedis
```
#!/bin/bash

BASE=</path/to/env/installation>

# Load the frameworks module
module load frameworks/2024.1

# Create a Python venv for SmartSim
python -m venv --clear $BASE/_ssim_env --system-site-packages
source $BASE/_ssim_env/bin/activate
pip install --upgrade pip

# Set SmartSim build variables
export SMARTSIM_REDISAI=1.2.7

# Build SmartSim
git clone https://github.com/rickybalin/SmartSim.git
cd SmartSim
pip install -e .
# Note: disregard errors
# - intel-extension-for-tensorflow 2.15.0.0 requires absl-py==1.4.0, but you have absl-py 2.1.0 which is incompatible.
# - intel-extension-for-tensorflow 2.15.0.0 requires numpy>=1.24.0, but you have numpy 1.23.5 which is incompatible.
cd ..

# Install the CPU backend
# NB: GPU backend for RedisAI not supported on Intel PVC
cd SmartSim
export TORCH_CMAKE_PATH=$( python -c 'import torch;print(torch.utils.cmake_prefix_path)' )
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
smart build -v --device cpu --torch_dir $TORCH_CMAKE_PATH --no_tf | tee build.log
smart validate
cd ..

# Install SmartRedis
git clone https://github.com/rickybalin/SmartRedis.git
cd SmartRedis
pip install -e .
make lib
cd ..
```

Once the installation is complete, source the following before running libCEED+CFDML 
```
module load frameworks/2024.1
source </path/to/env/installation>/_ssim_env/bin/activate

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
```

### Clone CFDML

Clone CFDML from Github and switch to the `aesp` branch
```
git clone https://github.com/rickybalin/CFDML.git
cd CFDML
git checkout aesp
cd ..
```

### Build PETSc

From a compute node of Aurora, load the frameworks module
```
module load frameworks/2024.1
```

and clone Kenneth E. Jansen's fork of PETSc which enables CGNS I/O
```
git clone https://gitlab.com/KennethEJansen1/petsc-kjansen-fork.git petsc_kej
cd petsc_kej
git checkout ReadCGNS_Squashed_RB240521
```

then build PETSc with the following configuration file
```
#!/opt/cray/pe/python/3.9.13.1/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    '--with-mpiexec-tail=gpu_tile_compact.sh', # a workaround, e.g., use 'mpiexec -n <np> gpu_tile_compact.sh ./ex1' to run petsc tests
    '--with-64-bit-indices',
    '--with-cc=mpicc',
    '--with-cxx=mpicxx',
    '--with-fc=0',
    '--COPTFLAGS=-O2',
    '--CXXOPTFLAGS=-O2',
    '--FOPTFLAGS=-O2',
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--SYCLOPTFLAGS=-O2',
    '--download-kokkos',
    '--download-kokkos-kernels', # default is Kokkos-Kernels-4.0.1
    '--download-hdf5',          # optional packages
    '--download-cgns',          # optional packages
    '--download-metis',
    '--download-parmetis',
    '--with-sycl',
    '--with-syclc=icpx',
    '--with-sycl-arch=pvc',      # enable AOT for Ponte Vecchio (Sunspot); Use xehp for Arcticus
    '--PETSC_ARCH=frameworks24.1',
  ]
  configure.petsc_configure(configure_options)
```

### Build libCEED (and the fluids code PHASTA-CEED)

Clone the libCEED code from Github and switch to the CGNS enabled branch
```
git clone https://github.com/CEED/libCEED.git
cd libCEED
git checkout ken/ReadCGNS_RB240427
```

then build the libCEED library
```
make configure SYCL_DIR=$CMPROOT
make -j
```

followed by the fluids solver (PHASTA-CEED)
```
cd examples/fluids
make SMARTREDIS_DIR=</path/to/env/installation>/SmartRedis/install
```

Finally, execute the libCEED and PHASTA-CEED tests to verify correctness
```
cd ../..

# For the online training test only, execute
make SMARTREDIS_DIR=</path/to/env/installation>/SmartRedis/install test search='fluids-p' BACKENDS='/gpu/sycl/ref'

# For the general PHASTA-CEED tests execute
make test search='fluids' BACKENDS='/gpu/sycl/ref'
```



