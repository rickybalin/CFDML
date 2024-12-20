# CFDML with libCEED on ALCF Aurora

## Build Instructions

### Build and Run Environment (including SmartSim/SmartRedis)

Execute the following code on a compute node of Aurora to create a Python virtual environment with SmartSim and SmartRedis
```
#!/bin/bash

BASE=</path/to/env/installation>

# Load the frameworks module
module load frameworks/2024.2.1_u1

# Create a Python venv for SmartSim
python -m venv --clear $BASE/_ssim_env --system-site-packages
source $BASE/_ssim_env/bin/activate
pip install --upgrade pip

# Set SmartSim build variables
export SMARTSIM_REDISAI=1.2.7

# Build SmartSim and CPU backend
# NB: GPU backend for RedisAI not supported on Intel PVC, so only building CPU backend
git clone https://github.com/rickybalin/SmartSim.git
cd SmartSim
git checkout rollback_aurora
pip install -e .
# NB: disregard pip dependency errors

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
module load frameworks/2024.2.1_u1
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
module load frameworks/2024.2.1_u1
```

and clone Kenneth E. Jansen's fork of PETSc
```
git clone https://gitlab.com/KennethEJansen1/petsc-kjansen-fork.git
cd petsc-kjansen-fork
git checkout ken/ReadCGNS_RB240830 ### Q3cherryto240108_B ReadCGNS_Squashed_RB240521
```

then build PETSc with the following configuration file
```
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
    '--COPTFLAGS=-O2 -fp-model=precise',
    '--CXXOPTFLAGS=-O2 -fp-model=precise',
    '--FOPTFLAGS=-O2 -fp-model=precise',
    '--SYCLPPFLAGS=-Wno-tautological-constant-compare',
    '--SYCLOPTFLAGS=-O2 -fp-model=precise',
    '--download-kokkos',
    '--download-kokkos-kernels',
#  next two were pulled from August 30 PETSc rebased pkg.gitcommit
    '--download-kokkos-commit=08ceff92bcf3a828844480bc1e6137eb74028517',
    '--download-kokkos-kernels-commit=cfa77aef9c6fc1531b839dd1758cafc1d8592cfc',
    '--download-hdf5',          # optional packages
    '--download-cgns-commit=remove_access_calls',
    '--download-cgns=git://https://github.com/jrwrigh/CGNS',        # optional packages
    '--download-metis',
    '--download-parmetis',
#    '--download-ptscotch=../scotch_7.0.4beta3.tar.gz',
#    '--download-ptscotch',
    '--with-sycl',
    '--with-syclc=icpx',
    '--with-sycl-arch=pvc',      # enable AOT for Ponte Vecchio (Sunspot); Use xehp for Arcticus
    '--PETSC_ARCH=frameworks24.2.1_RB240830_Kokkos4.4',
  ]
  configure.petsc_configure(configure_options)
```

After building PETSc, make sure to export the `PETSC_DIR` and `PETSC_ARCH` environment variables.

### Build libCEED (and the fluids code PHASTA-CEED)

Clone the libCEED code from Github and switch to the CGNS enabled branch
```
git clone https://github.com/CEED/libCEED.git
cd libCEED
git checkout jrwrigh/ReadCGNS_RB240427
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

For the online training test only, execute (make sure the Python virtual env is active, or will get error about `smartsim` module not being found)
make SMARTREDIS_DIR=</path/to/env/installation>/SmartRedis/install test search='fluids-p' BACKENDS='/gpu/sycl/ref'

For the general PHASTA-CEED tests execute
make test search='fluids' BACKENDS='/gpu/sycl/ref'
```

### Run 1 node tests

Run a single-node simulation only test [here](./1_node_tests/sim) to make sure the execution of libCEED is correct.

For the workflow, run single node tests with ML training on the GPU [here](./1_node_tests/workflow_ml_on_gpu) and ML training on the CPU here [here](./1_node_tests/workflow_ml_on_cpu).
The output of the workflow is contained within the `cfdml/sim` and `cfdml/train` directories that are created at runtime. 




