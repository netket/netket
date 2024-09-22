# HPC Cluster setup examples

This guide includes some installation/setup instructions for NetKet using the two distributed computing modes (MPI and/or sharding).
The guide assumes that you want to run on GPUs.

The guide is based on our own experience getting NetKet running on France's Jean-Zay HPC cluster. If you struggle installing it on your own machine, do get in touch on Github.

## Setting up MPI with CUDA-aware MPI

We here assume that you want to use CUDA-aware MPI because that allows for maximal performance. 
If your cluster does not provide CUDA-aware MPI, the installation is usually simpler as you do not need to load all cuda modules, and can just install `mpi4jax` without build time isolation.

```bash
# Pick your environment name
ENV_NAME=jax_gpu_mpi_amd

module load anaconda-py3 # To create the environment. You might just load python or some equivalent
module load gcc/12.2.0   # Compilers necessary to compile mpi4py/jax. You might have to load something equivalent
                         # prefer recent compilers
# You must load the cuda-aware mpi versions, and all related cuda and cudnn libraries
module load cuda/12.2.0 cudnn/8.9.7.29-cuda openmpi/4.1.5-cuda 

# Create the environment
conda create -y --name $ENV_NAME python=3.11 
conda activate $ENV_NAME

# Always update pip
pip install --upgrade pip

# Remove mpi4py and mpi4jax from build cache, to make sure you are building them anew
pip cache remove mpi4py
pip cache remove mpi4jax

# Install jax version that makes use of the local cuda version
pip install --upgrade "jax[cuda12_local]"
pip install --upgrade mpi4py cython
pip install --upgrade --no-build-isolation mpi4jax
pip install --upgrade netket 
```

Note that the `pip cache remove` is useful to avoid issues because pip caches previously built versions of mpi4py and mpi4jax.
Also, if you have errors always create a brand new environment.

## Running simulations with MPI on GPUS

```bash
#SBATCH --job-name=test_mpi
#SBATCH --output=test_mpi_%j.txt
#SBATCH --hint=nomultithread  # Disable Hyperthreading

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4          # here you should insert the total number of gpus per node
#SBATCH --ntasks-per-node=4   # maximum number of tasks per node, should match the maximum 
                              # of GPUS per node.
#SBATCH --time=01:00:00

ENV_NAME=jax_gpu_mpi_amd

# Load the same packages you used during installation. In our case that is
module purge
module load gcc/12.2.0 anaconda-py3 
module load cuda/12.2.0 cudnn/9.2-v7.5.1.10 openmpi/4.1.5-cuda

# Load the conda environment or equivalent
conda activate $ENV_NAME

# This is to use fast direct gpu-to-gpu communication
# If you do not have CUDA-aware MPI, set this to 0 instead
export MPI4JAX_USE_CUDA_MPI=1
# This is not strictly needed, simply tells netket to forcefully use MPI
export NETKET_MPI=1
# This automatically assigns only 1 GPU per rank (MPI cannoot use more than 1)
# If you do not use this, you should make sure that every rank sees only 1 GPU.
export NETKET_MPI_AUTODETECT_LOCAL_GPU=1
# Tell Jax that we want to use GPUs. THis is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

srun python yourscript.py
```

(cluster-sharding-setup)=
## Running simulations with Sharding on GPUS

To run simulations with sharding on GPUs, just install the following packages in the environemnt:
```bash
netket
jax[cuda]
```
nothing else is needed: no MPI.

```bash
#SBATCH --job-name=test_mpi
#SBATCH --output=test_mpi_%j.txt
#SBATCH --hint=nomultithread  # Disable Hyperthreading

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:4          # here you should insert the total number of gpus per node
#SBATCH --time=01:00:00

module purge

# Load the same packages you used during installation. In our case that is
module load gcc/12.2.0 anaconda-py3 

# Load the conda environment or equivalent
conda activate ENV_NAME

# Tell NetKet to use experimental sharding mode.
export NETKET_EXPERIMENTAL_SHARDING=1
# Tell Jax that we want to use GPUs. THis is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

srun python yourscript.py
```

And the script is structured as

```python
import jax

jax.distributed.initialize()

print(jax.devices())
print(jax.local_devices())

import netket as nk
```


## Cluster-specific informations

This is a sparse collection of instructions written for some specific HPC clusters, which might help
users getting started elsewhere. 

 - France: [Jean Zay](https://quantum-ai-lab.getoutline.com/s/cb890bcf-0cfd-4a20-b98b-18b66e80138f)
 - France: [Cholesky](https://quantum-ai-lab.getoutline.com/s/45a8aa4f-86be-4159-b68f-e354248f64c5)