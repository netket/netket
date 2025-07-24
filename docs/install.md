# Installation

NetKet is a standard Python package that can be installed using pip or uv. This page covers the basic installation process and GPU setup.

```{admonition} Conda Installation Not Supported
:class: warning

**⚠️ Do not install NetKet through conda.** JAX has known issues when installed through conda, and you will likely get old, outdated versions of NetKet. Use pip or uv instead as described below.
```

## Basic Installation

NetKet requires Python 3.11 or later. To install NetKet, simply run:

**Using pip:**
```bash
pip install --upgrade pip
pip install netket
```

**Using uv:**
```bash
uv add netket
```

**With GPU support (Linux only):**
```bash
# Using pip
pip install 'netket[cuda]'

# Using uv  
uv add 'netket[cuda]'
```

## Verifying Installation

To verify your installation, check the NetKet version:

```bash
python -c "import netket; print(netket.__version__)"
```

If all went well, you should have at least version 3.18 installed. 
We recommend always starting new projects with the latest available version.

## GPU Support

Whether you can use GPUs with NetKet depends entirely on how you install JAX. **By default, NetKet installs a CPU-only version of JAX everywhere.**

**Important:** JAX will automatically install its own CUDA version and only requires up-to-date CUDA drivers on your system. **Do not try to use your own CUDA installation** as this generally creates problems and conflicts. This applies to both local machines and HPC clusters.

The most robust approach is to explicitly install **JAX with GPU support** in addition to NetKet. For detailed instructions, refer to the [JAX documentation](https://docs.jax.dev/en/latest/installation.html), as these instructions change frequently.

The `netket[cuda]` installation approach shown above in the Basic Installation section is instead the simplest, and will work most of the time, but might fail at times.


### Verifying GPU Detection

**To verify GPU support:**
```bash
python -c 'import jax; print(jax.devices())'
```

If JAX detects your GPUs correctly, you should see CUDA or GPU devices listed. If you have multiple GPUs, all of them should appear in the output.


### Manual JAX GPU Installation


```{admonition} Manual GPU Installation (Current as of 2025)
:class: warning

**⚠️ This information may be outdated - always check the [JAX documentation](https://docs.jax.dev/en/latest/installation.html) for the latest instructions.**

As of today, to manually install a CUDA-enabled version of JAX:

```bash
pip install 'jax[cuda]'
uv add 'jax[cuda]'
```


## Development Version (Git)

If you want to install the latest development version of NetKet directly from the Git repository, you can use:

**Using pip:**
```bash
pip install --upgrade pip
pip install git+https://github.com/netket/netket.git
```

**Using uv:**
```bash
uv add git+https://github.com/netket/netket.git
```

```{admonition} Development Version Warning
:class: warning

The development version may contain experimental features and bugs. Use at your own risk for production work. For stable releases, use the regular installation methods above.
```

## Distributed Computing (HPC Clusters)

The standard NetKet installation with JAX is sufficient to run distributed calculations across multiple nodes and GPUs. No additional packages are required for multi-node distributed computing.

### HPC Cluster FAQ

**CUDA on clusters:**
- **Do not activate** the cluster-provided CUDA library (e.g., don't run `module load cuda`). JAX manages its own CUDA installation as mentioned above.

**Operating system compatibility:**
- JAX requires a relatively recent operating system (≥7 years old). On older clusters, you might only be able to install older versions of JAX and consequently NetKet. If this happens, the only solution is to request that cluster administrators update the OS version.

**Package manager recommendation:**
- We strongly recommend using **uv instead of pip** on clusters, as uv guarantees dependencies that work well together and provides more reliable package resolution in complex environments.

## Installation Issues

```{admonition} Installation Issues?
:class: seealso

If you experience installation errors:
1. Make sure you have upgraded pip: `pip install --upgrade pip`
2. Ensure you're not inside a conda environment (unless intended)
3. If NetKet installs but won't load, you may need to update dependencies

For help, please open an issue and include the output of:
```bash
python -m netket.tools.info
```
