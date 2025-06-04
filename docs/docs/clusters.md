# HPC Cluster setup examples

This guide includes some installation/setup instructions for NetKet using the distributed computing modes based on sharding.
The guide assumes that you want to run on GPUs.
In the past, NetKet also supported MPI, largely on CPUs, but support has been removed sometime in summer 2025 because Jax already supports running on multiple nodes.

The guide is based on our own experience getting NetKet running on France's Jean-Zay HPC cluster. If you struggle installing it on your own machine, do get in touch on Github.


(cluster-sharding-setup)=
## Running simulations with Sharding on GPUS

To run simulations with sharding on GPUs, just install the following packages in the environemnt:
```bash
netket
jax[cuda]
```
nothing else is needed.

```bash
#SBATCH --job-name=test
#SBATCH --output=test_%j.txt
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